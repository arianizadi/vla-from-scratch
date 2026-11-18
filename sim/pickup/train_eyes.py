"""Train the eyes (eyes.py) on perception datasets (collect_scene.py, read by eyes_data.py):

    python -m sim.pickup.train_eyes --data data/scene/reason_mix_2k --val data/scene/reason_mix_val \\
        --out experiments/pickup/eyes_vit --samples 2000000

AdamW with a short linear warm-up and then a cosine down to 1e-5, gradients clipped at 1.0, bfloat16 autocast on a GPU,
and a running average of the weights kept beside them (the repository's adaptive EMA, vla_foundry/models/ema.py): every
checkpoint holds both, as "model" and "ema", and the averaged weights are what eyes.load_eyes gives by default, as the
policies here are evaluated with theirs.

The run directory gets eyes_config.json (eyes.EyesConfig), args.json (this command's arguments),
checkpoints/checkpoint_<k>.pt at --checkpoints evenly spaced points (k from 1), and metrics.jsonl: after each
checkpoint, one line with eyes.evaluate on the validation set for the trained and for the averaged weights. A progress
line comes every few hundred steps (samples seen, the loss and its terms, samples per second), in the form the run
monitors read (runstate.py), and the last line of a finished run is "Model has seen the desired number of samples",
which the scripts on the training machine wait for. --resume goes on from the newest checkpoint of --out.
"""

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .eyes import CONFIG_FILE, TARGETS, TOWER_TOKENS, Eyes, EyesConfig, checkpoints, evaluate, save
from .eyes_data import SceneFrames, collate
from .scene import TRUTH_ATTRIBUTES

FINISHED = "Model has seen the desired number of samples"
MIN_LR = 1e-5  # where the cosine ends
LOG_EVERY = 200  # steps between progress lines
# Trained without weight decay, besides biases and norms: what says where a token is or what it is for
NO_DECAY = ("position_embedding", "view_embedding", "queries", "cls_token", "class_embedding")


def learning_rate(step: int, steps: int, peak: float, warmup: int) -> float:
    """A linear warm-up over `warmup` steps, then a cosine from `peak` down to MIN_LR at the last step."""
    if step < warmup:
        return peak * (step + 1) / warmup
    end = min(MIN_LR, peak)
    progress = min((step - warmup) / max(steps - warmup, 1), 1.0)
    return end + 0.5 * (1 + math.cos(math.pi * progress)) * (peak - end)


def parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """The weights to train, in two groups: with weight decay the weight matrices, without it everything else."""
    decay, rest = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (rest if p.ndim < 2 or any(s in name for s in NO_DECAY) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}]


def _keep_metrics_up_to(path: Path, k: int):
    """Drop the lines of metrics.jsonl measured after checkpoint k: a run resumed from k writes them again, and the
    weights they describe are gone. A line cut short (a run stopped while writing it) goes too."""
    if not path.exists():
        return
    kept = []
    for line in path.read_text().splitlines():
        try:
            if json.loads(line)["checkpoint"] <= k:
                kept.append(line)
        except (ValueError, KeyError, TypeError):
            continue
    path.write_text("".join(line + "\n" for line in kept))


def _percent(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _summary(name: str, m: dict) -> str:
    """One line of eyes.evaluate's metrics."""
    height = "-" if m["height_err_mm"] is None else f"{m['height_err_mm']:.1f} mm"
    bearing = "-" if m["bearing_err_deg"] is None else f"{m['bearing_err_deg']:.1f} deg"
    attributes = "".join(
        f", {a} {_percent(m[f'{a}_acc'])} (unreadable {_percent(m[f'{a}_unreadable_acc'])})"
        for a in TRUTH_ATTRIBUTES
        if f"{a}_acc" in m
    )
    return (
        f"{name}: count {_percent(m['count_acc'])} (3 parts {_percent(m.get('count_acc_3'))}), "
        f"attached {_percent(m['attached_acc'])}, cut {_percent(m['cut_acc'])}, visible {_percent(m['visible_acc'])}, "
        f"height {height}, bearing {bearing}, flower {_percent(m['flower_acc'])}, label {_percent(m['label_acc'])} "
        f"(card {_percent(m['label_acc_card'])}){attributes}; {m['frames']} frames"
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="train the eyes (sim/pickup/eyes.py) on perception datasets")
    ap.add_argument("--data", nargs="+", required=True, help="perception datasets to train on (collect_scene.py)")
    ap.add_argument("--val", nargs="+", required=True, help="perception datasets to evaluate on at each checkpoint")
    ap.add_argument("--out", required=True, help="the run directory, e.g. experiments/pickup/<run>")
    ap.add_argument("--samples", type=int, required=True, help="frames to train on, in all")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4, help="the peak learning rate")
    ap.add_argument("--wd", type=float, default=0.05, help="AdamW's weight decay")
    ap.add_argument("--warmup", type=int, default=500, help="steps of linear warm-up (at most a tenth of the run)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cameras", nargs="+", default=["front", "wrist"], help="front, wrist, plant (the plant crop)")
    ap.add_argument("--frames", nargs="+", type=int, default=[0], help="frame offsets: -20 0 adds 1 s earlier")
    ap.add_argument(
        "--attributes",
        nargs="*",
        choices=TRUTH_ATTRIBUTES,
        default=[],
        help="architecture v3: the attributes each part slot reads as well (tie sick bud); none: v2's eyes",
    )
    ap.add_argument("--backbone", choices=("vit", "hf", "tower"), default="vit")
    ap.add_argument("--hf_name", default=None, help="hf: a model name or a local directory")
    ap.add_argument("--hf_trainable_layers", type=int, default=0, help="hf: train its last n layers (0: frozen)")
    ap.add_argument("--tower_name", default=None, help="tower: a vision-language model's name or local directory")
    ap.add_argument(
        "--tower_tokens", choices=TOWER_TOKENS, default="patches", help="tower: before its merger, or after it"
    )
    ap.add_argument("--tower_trainable_layers", type=int, default=0, help="tower: train its last n blocks (0: frozen)")
    ap.add_argument("--workers", type=int, default=8, help="data-loader workers")
    ap.add_argument("--checkpoints", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true", help="go on from the newest checkpoint of --out")
    ap.add_argument("--augment", action="store_true", help="mild brightness, contrast and shift changes")
    ap.add_argument("--log_every", type=int, default=LOG_EVERY, help="steps between progress lines")
    args = ap.parse_args(argv)
    if min(args.samples, args.batch, args.checkpoints, args.log_every) < 1 or args.workers < 0:
        ap.error("--samples, --batch, --checkpoints and --log_every must be positive, --workers 0 or more")
    return args


def main(argv=None):
    args = parse_args(argv)
    out, device = Path(args.out), torch.device(args.device)
    cuda = device.type == "cuda"
    config = EyesConfig(
        cameras=args.cameras,
        frames=args.frames,
        backbone=args.backbone,
        hf_name=args.hf_name,
        hf_trainable_layers=args.hf_trainable_layers,
        tower_name=args.tower_name,
        tower_tokens=args.tower_tokens,
        tower_trainable_layers=args.tower_trainable_layers,
        attributes=args.attributes,
    )
    found = checkpoints(out)
    if found and not args.resume:
        raise SystemExit(f"{out} has checkpoints already: --resume to go on with that run, or another --out")
    if found and EyesConfig.from_json((out / CONFIG_FILE).read_text()) != config:
        raise SystemExit(f"{out} was started with another configuration: {out / CONFIG_FILE}")
    if not found:
        out.mkdir(parents=True, exist_ok=True)
        (out / CONFIG_FILE).write_text(config.to_json())
        (out / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    torch.manual_seed(args.seed)
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True  # for the evaluations, which run in float32
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    from vla_foundry.models.ema import EMAModel

    model = Eyes(config).to(device)
    ema = EMAModel(model)  # a copy of the model, on its device, averaged after every step
    trained = [p for p in model.parameters() if p.requires_grad]
    groups = parameter_groups(model, args.wd)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), fused=True if cuda else None)

    steps = math.ceil(args.samples / args.batch)
    count = min(args.checkpoints, steps)
    at = {k * steps // count: k for k in range(1, count + 1)}  # step -> the checkpoint written after it
    warmup = max(1, min(args.warmup, steps // 10))
    step = 0
    if found:
        state = torch.load(found[max(found)], map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        ema.model.load_state_dict(state["ema"])
        ema.optimization_step.fill_(int(state["ema_step"]))
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        _keep_metrics_up_to(out / "metrics.jsonl", int(state["checkpoint"]))
        print(f"resumed from checkpoint {state['checkpoint']}: step {step}, {state['samples']} samples", flush=True)

    # The images travel as bytes (as_float False) and become floats on the device: a quarter of the copying
    batches_of = partial(collate, config=config, as_float=False)
    # a resumed run reads its data in another order than the start did
    train_set = SceneFrames(args.data, config, train=True, seed=args.seed * 1000 + len(found), augment=args.augment)
    val_set = SceneFrames(args.val, config, train=False)
    workers = dict(num_workers=args.workers, pin_memory=cuda, collate_fn=batches_of)
    if args.workers > 0:
        workers.update(prefetch_factor=4, persistent_workers=True)
    train_loader = DataLoader(train_set, batch_size=args.batch, **workers)
    val_workers = min(args.workers, len(val_set.episodes))
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        num_workers=val_workers,
        pin_memory=cuda,
        collate_fn=batches_of,
        persistent_workers=val_workers > 0,
    )

    total = sum(p.numel() for p in model.parameters())
    print(
        f"eyes: {total} parameters, {sum(p.numel() for p in trained)} trained; backbone {config.backbone}; "
        f"views {config.keys}; device {device}",
        flush=True,
    )
    print(
        f"training on {len(train_set.episodes)} episodes ({train_set.frames} frames), validating on "
        f"{len(val_set.episodes)} ({val_set.frames} frames); {steps} steps of {args.batch}, "
        f"checkpoints after steps {sorted(at)}",
        flush=True,
    )

    ends = sorted(at)

    def chunk(done: int) -> tuple[int, int, int]:
        """(the next checkpoint, the step it starts from, the step it is written at), for `done` steps done."""
        end = min((e for e in ends if e > done), default=ends[-1])
        return at[end], max((e for e in ends if e < end), default=0), end

    started, batches = time.time(), iter(train_loader)
    if step < steps:
        print(f"Samples: {chunk(step)[1] * args.batch} / {steps * args.batch}", flush=True)
    log_sum, log_steps, log_time = None, 0, time.time()
    model.train()
    while step < steps:
        batch = next(batches)
        images = batch["images"].to(device, non_blocking=True)
        targets = {key: batch[key].to(device, non_blocking=True) for key in TARGETS}
        lr = learning_rate(step, steps, args.lr, warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
            outputs = model(images)
        loss, terms = model.loss_terms(outputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trained, 1.0)
        optimizer.step()
        ema.step(model)
        step += 1
        # summed on the device and read once per progress line: no wait for the GPU at every step
        values = torch.stack([loss.detach(), *(term.detach() for term in terms.values())])
        log_sum, log_steps = values if log_sum is None else log_sum + values, log_steps + 1

        if step % args.log_every == 0 or step in at:
            mean = (log_sum / log_steps).tolist()
            rate = log_steps * args.batch / max(time.time() - log_time, 1e-9)
            k, start, end = chunk(step - 1)
            done, size = (step - start) * args.batch, (end - start) * args.batch
            parts = " ".join(f"{name} {value:.4f}" for name, value in zip(terms, mean[1:], strict=True))
            print(
                f"Train Checkpoint: {k} [{done:>{len(str(size))}}/{size} ({100 * done / size:.0f}%)] "
                f"Loss: {mean[0]:.4f} ({parts}) samples {step * args.batch}/{steps * args.batch}, "
                f"{rate:.1f}/s, LR: {lr:.3e}",
                flush=True,
            )
            log_sum, log_steps, log_time = None, 0, time.time()

        if step in at:
            k = at[step]
            ema_step = int(ema.optimization_step.item())
            extra = dict(optimizer=optimizer.state_dict(), ema_step=ema_step, step=step, samples=step * args.batch)
            path = save(out, k, model, ema.model, **extra)
            record = dict(checkpoint=k, step=step, samples=step * args.batch, seconds=round(time.time() - started, 1))
            record["model"] = evaluate(model, val_loader)
            record["ema"] = evaluate(ema.model, val_loader)
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"checkpoint {k}: {path}", flush=True)
            for name in ("model", "ema"):
                print(f"  {_summary(name, record[name])}", flush=True)
            if step < steps:
                print(f"Samples: {step * args.batch} / {steps * args.batch}", flush=True)
            log_time = time.time()  # the evaluation is not training time
    print(FINISHED, flush=True)


if __name__ == "__main__":
    main()
