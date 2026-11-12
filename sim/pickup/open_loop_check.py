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


def window_rows(past_mask, future_mask, past, future):
    """Which rows of a stored window a policy with `past` past steps and `future` future steps uses.

    The shards store a fixed number of steps before the current one (one, for the pickup datasets),
    whatever the policy takes: a policy without history uses none of them. Returns
    (first, now, end, previous): rows [first, now) are the policy's past steps, `now` is the current
    step, rows [now, end) are the actions it predicts, and `previous` is the row of the action
    executed just before the current step (None if the window has none). Returns None for a window
    that is padded where the policy would read, so that it can be skipped.
    """
    past_mask, future_mask = np.asarray(past_mask, dtype=bool), np.asarray(future_mask, dtype=bool)
    # future_mask is never set on the rows before the current step, and always on the current one
    # (create_past_and_future_masks), so the first row on which it is set is the current step.
    now = int(np.argmax(future_mask))
    first, end = now - past, now + 1 + future
    if first < 0 or end > len(future_mask):
        raise ValueError(
            f"the shards store {now} past and {len(future_mask) - now - 1} future steps; "
            f"the policy takes {past} and {future}"
        )
    if not future_mask[now:end].all() or not past_mask[first:now].all():
        return None
    previous = now - 1 if now >= 1 and past_mask[now - 1] else None
    return first, now, end, previous


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

    from .policy_server import Policy  # here, so that window_rows can be imported without torch

    policy = Policy(args.checkpoint_dir, None, "cuda")
    paths = sorted(glob.glob(f"{args.shards}/shard_*.tar"))
    paths = paths[:: max(1, len(paths) // 8)]
    err_pred, err_copy, err_first = [], [], []
    for sample in iter_samples(paths, args.samples):
        lowdim = dict(np.load(io.BytesIO(sample["lowdim.npz"])))
        rows = window_rows(lowdim["past_mask"], lowdim["future_mask"], policy.past, policy.future)
        if rows is None or rows[3] is None:
            continue  # padded where the policy reads, or no action before the current step to compare with
        first, now, end, previous = rows
        images = {name: sample[f"{name}.jpg"] for name in policy.image_names}
        instruction = json.loads(sample["language_instructions.json"])["original"]
        if isinstance(instruction, list):
            instruction = instruction[0]
        action, state = lowdim[policy.action_field], lowdim[policy.state_field]
        pred = policy.infer(images, state[first : now + 1], action[previous], instruction, args.flow_steps)
        gt = action[now:end]
        err_pred.append(np.abs(pred - gt).mean(axis=0))
        err_first.append(np.abs(pred[0] - gt[0]))
        err_copy.append(np.abs(action[previous][None] - gt).mean(axis=0))
    print(f"samples: {len(err_pred)}  checkpoint: {policy.checkpoint_path}")
    if not err_pred:
        raise SystemExit("no window without padding among the samples read; raise --samples")
    names = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "grip"]
    fmt = lambda v: " ".join(f"{n}={x:.4f}" for n, x in zip(names, v, strict=True))  # noqa: E731
    print("mean |pred - expert| over the chunk :", fmt(np.mean(err_pred, axis=0)))
    print("mean |pred - expert| first action   :", fmt(np.mean(err_first, axis=0)))
    print("baseline: repeat previous action    :", fmt(np.mean(err_copy, axis=0)))


if __name__ == "__main__":
    main()
