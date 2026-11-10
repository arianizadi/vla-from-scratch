"""Open-loop check: run the policy-server inference path on samples from preprocessed shards and
compare predicted action chunks against the recorded expert actions.

  python -m sim.pickup.open_loop_check --checkpoint_dir experiments/pickup/<run> \
      --shards data/preprocessed/pick_cube/shards --samples 64
"""

import argparse
import glob
import io
import json
import tarfile

import numpy as np

from .policy_server import Policy


def iter_samples(shard_paths, limit):
    n = 0
    for path in shard_paths:
        with tarfile.open(path) as tar:
            current, key = {}, None
            for member in tar:
                k, ext = member.name.split(".", 1)
                if key is not None and k != key:
                    yield current
                    n += 1
                    if n >= limit:
                        return
                    current = {}
                key = k
                current[ext] = tar.extractfile(member).read()
            if current:
                yield current
                n += 1
                if n >= limit:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--flow_steps", type=int, default=10)
    args = ap.parse_args()

    policy = Policy(args.checkpoint_dir, None, "cuda")
    paths = sorted(glob.glob(f"{args.shards}/shard_*.tar"))
    paths = paths[:: max(1, len(paths) // 8)]
    err_pred, err_copy, err_first = [], [], []
    for sample in iter_samples(paths, args.samples):
        lowdim = dict(np.load(io.BytesIO(sample["lowdim.npz"])))
        if not lowdim["future_mask"][policy.past :].all() or not lowdim["past_mask"][: policy.past].all():
            continue  # skip padded windows
        images = {name: sample[f"{name}.jpg"] for name in policy.image_names}
        instruction = json.loads(sample["language_instructions.json"])["original"]
        if isinstance(instruction, list):
            instruction = instruction[0]
        action, state = lowdim[policy.action_field], lowdim[policy.state_field]
        pred = policy.infer(images, state[: policy.past + 1], action[policy.past - 1], instruction, args.flow_steps)
        gt = action[policy.past :]
        err_pred.append(np.abs(pred - gt).mean(axis=0))
        err_first.append(np.abs(pred[0] - gt[0]))
        err_copy.append(np.abs(action[policy.past - 1][None] - gt).mean(axis=0))
    names = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "grip"]
    fmt = lambda v: " ".join(f"{n}={x:.4f}" for n, x in zip(names, v, strict=True))  # noqa: E731
    print(f"samples: {len(err_pred)}  checkpoint: {policy.checkpoint_path}")
    print("mean |pred - expert| over the chunk :", fmt(np.mean(err_pred, axis=0)))
    print("mean |pred - expert| first action   :", fmt(np.mean(err_first, axis=0)))
    print("baseline: repeat previous action    :", fmt(np.mean(err_copy, axis=0)))


if __name__ == "__main__":
    main()
