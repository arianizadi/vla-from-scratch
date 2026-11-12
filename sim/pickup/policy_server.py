"""Policy server: serves action chunks from a trained VLA Foundry checkpoint over a local socket.

Runs inside the Foundry environment (torch + vla_foundry); the simulator runs in a separate
process/environment and talks to it through `multiprocessing.connection`.

Inputs are pushed through the *same* RoboticsProcessor code path as training
(`process_inputs` -> `add_action_and_proprioception_fields`), so tokenization, image
preprocessing and normalization cannot drift from what the model saw.

  python -m sim.pickup.policy_server --checkpoint_dir experiments/pickup/<run> --port 6200
"""

import argparse
import glob
import os
import re
import time
from multiprocessing.connection import Listener

import numpy as np
import torch

from vla_foundry.data.augmentations.decode_and_augment import fast_image_decoder
from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.file_utils import load_ema_checkpoint, load_model_checkpoint
from vla_foundry.models import create_model
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml
from vla_foundry.precision import get_autocast

AUTHKEY = b"pickup-sim"


def latest_checkpoint(checkpoint_dir: str, prefix: str) -> str:
    paths = glob.glob(os.path.join(checkpoint_dir, "checkpoints", f"{prefix}_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no {prefix}_*.pt under {checkpoint_dir}/checkpoints")
    return max(paths, key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1)))


class Policy:
    def __init__(self, checkpoint_dir: str, checkpoint: str | None, device: str, weights: str = "auto"):
        self.cfg = load_experiment_params_from_yaml(os.path.join(checkpoint_dir, "config.yaml"), localize_params=True)
        self.device = torch.device(device)
        self.model = create_model(self.cfg.model, load_pretrained=False)
        # weights: "ema" / "raw" pick the averaged or the directly trained weights of a run trained with
        # --ema.enabled True (both are saved, as ema_N.pt and checkpoint_N.pt); "auto" = EMA when available.
        use_ema = bool(self.cfg.ema.enabled) if weights == "auto" else weights == "ema"
        if checkpoint is not None:  # accept either file of a pair and serve the requested kind
            name = re.sub(r"^(ema|checkpoint)_", "ema_" if use_ema else "checkpoint_", os.path.basename(checkpoint))
            checkpoint = os.path.join(os.path.dirname(checkpoint), name)
        path = checkpoint or latest_checkpoint(checkpoint_dir, "ema" if use_ema else "checkpoint")
        (load_ema_checkpoint if use_ema else load_model_checkpoint)(self.model, path)
        self.checkpoint_path = path
        self.model.to(self.device).eval()
        self.processor = RoboticsProcessor.from_pretrained(checkpoint_dir)
        data = self.cfg.data
        self.past = data.lowdim_past_timesteps
        self.future = data.lowdim_future_timesteps
        self.total = self.past + 1 + self.future
        self.image_names = list(data.image_names)
        assert len(data.action_fields) == 1 and len(data.proprioception_fields) == 1, "expects one field each"
        self.action_field = data.action_fields[0]
        self.state_field = data.proprioception_fields[0]
        self.autocast = get_autocast(self.cfg.hparams.precision)

    @torch.no_grad()
    def infer(self, images: dict, state, prev_action, instruction: str, num_flow_steps: int = 10):
        """images: {"<camera>_t<idx>": jpeg bytes}; state: [past+1, D]; prev_action: [D] last executed command."""
        state = np.asarray(state, dtype=np.float32)[-(self.past + 1) :]  # no-history policies take only the last
        prev_action = np.asarray(prev_action, dtype=np.float32)
        action_seq = np.zeros((self.total, prev_action.shape[-1]), dtype=np.float32)
        action_seq[: self.past] = prev_action
        state_seq = np.concatenate([state, np.repeat(state[-1:], self.total - len(state), axis=0)])
        batch = {
            "images": [{name: fast_image_decoder(f"{name}.jpg", images[name]) for name in self.image_names}],
            "language_instruction": [instruction],
            "lowdim": [{self.action_field: action_seq, self.state_field: state_seq}],
        }
        batch = self.processor.process_inputs(
            batch, image_names=self.image_names, max_text_seq_len=self.cfg.data.max_text_seq_len
        )
        batch = self.processor.add_action_and_proprioception_fields(
            batch, action_fields=[self.action_field], proprioception_fields=[self.state_field]
        )
        inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        past_mask = torch.zeros(1, self.total, dtype=torch.bool, device=self.device)
        past_mask[:, : self.past] = True
        with self.autocast():
            out = self.model.generate_actions(**inputs, past_mask=past_mask, num_inference_steps=num_flow_steps)
        out = out.float().cpu()
        if self.processor.normalizer is not None:
            out = self.processor.normalizer.denormalize_tensor(out, self.action_field, anchor_timestep=self.past)
        return out[0, self.past :].numpy()  # [1 + future, D]: actions for t, t+1, ...


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--checkpoint", default=None, help="explicit .pt path; default = latest (EMA if enabled)")
    ap.add_argument("--port", type=int, default=6200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default="auto", choices=["auto", "ema", "raw"])
    args = ap.parse_args()

    # Bind before loading the model: if the port is taken this fails at once, instead of after the
    # client has already connected to whoever holds the port.
    with Listener(("127.0.0.1", args.port), authkey=AUTHKEY) as listener:
        policy = Policy(args.checkpoint_dir, args.checkpoint, args.device, args.weights)
        n_params = sum(p.numel() for p in policy.model.parameters())
        print(f"[policy_server] loaded {policy.checkpoint_path} ({n_params / 1e6:.1f}M params)", flush=True)
        print(f"[policy_server] listening on 127.0.0.1:{args.port}", flush=True)
        while True:
            with listener.accept() as conn:
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    if msg["cmd"] == "shutdown":
                        conn.send({"ok": True})
                        return
                    if msg["cmd"] == "info":
                        conn.send(
                            dict(
                                checkpoint=policy.checkpoint_path,
                                params_m=n_params / 1e6,
                                image_names=policy.image_names,
                                past=policy.past,
                                future=policy.future,
                            )
                        )
                        continue
                    t0 = time.perf_counter()
                    actions = policy.infer(
                        msg["images"],
                        msg["state"],
                        msg["prev_action"],
                        msg["instruction"],
                        msg.get("num_flow_steps", 10),
                    )
                    if policy.device.type == "cuda":
                        torch.cuda.synchronize()
                    conn.send({"actions": actions.tolist(), "latency_s": time.perf_counter() - t0})


if __name__ == "__main__":
    main()
