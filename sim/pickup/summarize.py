"""Collect closed-loop results into one table.

Walks results/pickup/<run>/<task>/**/results.json and prints one row per evaluation: success with
its Wilson interval, how often the gripper closed at the named object, the median miss, and success
per object. Evaluations of individual checkpoints (written by scripts/eval_checkpoints.sh) are
matched to the number of training samples seen, read from the run's training log, so the rows give
success as a function of training.

    python -m sim.pickup.summarize                                  # everything
    python -m sim.pickup.summarize --task pick_target --runs dp_vit dp_vit_lang
    python -m sim.pickup.summarize --best --json results/pickup/summary.json
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path


def served_checkpoint_matches(results, run, checkpoint, weights):
    """Was this evaluation served by the checkpoint its directory says (run, number, raw or ema)?

    False means the evaluation reached another run's policy server (two evaluations on one port).
    Results written before the server recorded its checkpoint count as matching.
    """
    served = str((results.get("server") or {}).get("checkpoint", "")).replace("\\", "/")
    if not served:
        return True
    parts = served.split("/")
    want = f"{'ema' if weights == 'ema' else 'checkpoint'}_{checkpoint}.pt"
    return len(parts) >= 3 and parts[-3] == run and parts[-1] == want


def checkpoint_samples(experiments: Path, run: str) -> dict[int, int]:
    """Training samples seen when checkpoint N was written, from the "Samples: X / Y" log lines."""
    seen, total = set(), None
    for log in glob.glob(str(experiments / f"{run}.train.log*")):
        with open(log, errors="ignore") as f:
            for line in f:
                m = re.search(r"Samples: (\d+) / (\d+)", line)
                if m:
                    seen.add(int(m.group(1)))
                    total = int(m.group(2))
    ordered = sorted(seen)  # ordered[n] = samples at the start of chunk n = when checkpoint n was written
    out = {n: s for n, s in enumerate(ordered) if n > 0}
    if total is not None:
        out[len(ordered)] = total  # the final checkpoint has no following chunk to log its count
    return out


def load_rows(results: Path, experiments: Path, task: str | None, runs: list[str] | None):
    rows, mismatched = [], []
    for path in sorted(results.glob("*/*/**/results.json")):
        run, run_task = path.relative_to(results).parts[:2]
        variant = "/".join(path.relative_to(results).parts[2:-1])
        if (task and run_task != task) or (runs and run not in runs):
            continue
        with open(path) as f:
            r = json.load(f)
        m = re.fullmatch(r"ckpt_(\d+)_(raw|ema)", variant)
        checkpoint, weights = (int(m.group(1)), m.group(2)) if m else (None, None)
        if m and not served_checkpoint_matches(r, run, checkpoint, weights):
            mismatched.append(str(path))
            continue
        grasp = r.get("grasp") or {}
        rows.append(
            dict(
                run=run,
                task=run_task,
                variant=variant,
                checkpoint=checkpoint,
                weights=weights,
                samples=checkpoint_samples(experiments, run).get(checkpoint) if checkpoint else None,
                successes=r["successes"],
                episodes=r["episodes"],
                success_rate=r["success_rate"],
                ci95=r["success_ci95"],
                at_named_object=grasp.get("closed_nearest_is_target_rate"),
                median_miss_named_mm=grasp.get("median_xy_err_target_mm"),
                median_miss_nearest_mm=grasp.get("median_xy_err_nearest_mm"),
                within_20mm=grasp.get("within_20mm_of_target_rate"),
                wrong_object_rate=r.get("wrong_object_rate"),
                per_target={k: f"{v['successes']}/{v['episodes']}" for k, v in r.get("per_target", {}).items()},
                exec_horizon=r.get("exec_horizon"),
                flow_steps=r.get("flow_steps"),
            )
        )
    rows.sort(key=lambda x: (x["task"], x["run"], x["checkpoint"] or 0, x["weights"] or "", x["variant"]))
    if mismatched:
        print(
            f"left out {len(mismatched)} evaluations that were served by another checkpoint than their directory "
            f"names (two evaluations on one port); delete and redo them: {' '.join(mismatched)}",
            file=sys.stderr,
        )
    return rows


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def mm(x):
    return "-" if x is None else f"{x:.0f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/pickup")
    p.add_argument("--experiments", default="experiments/pickup")
    p.add_argument("--task", default=None)
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--best", action="store_true", help="only the best evaluation of each run")
    p.add_argument("--json", default=None, help="also write the rows to this file")
    args = p.parse_args()

    rows = load_rows(Path(args.results), Path(args.experiments), args.task, args.runs)
    if args.best:
        best = {}
        for row in rows:
            key = (row["task"], row["run"])
            if key not in best or row["success_rate"] > best[key]["success_rate"]:
                best[key] = row
        rows = list(best.values())

    header = ["Run", "Task", "Checkpoint", "Samples", "Success", "95% CI", "At named object"]
    header += ["Miss, named (mm)", "Miss, nearest (mm)", "Per object"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        ckpt = f"{r['checkpoint']} {r['weights']}" if r["checkpoint"] else (r["variant"] or "latest")
        samples = "-" if r["samples"] is None else f"{r['samples'] / 1e6:.2f}M"
        per = " ".join(f"{k} {v}" for k, v in r["per_target"].items())
        print(
            f"| {r['run']} | {r['task']} | {ckpt} | {samples} | {r['successes']}/{r['episodes']} = "
            f"{pct(r['success_rate'])} | {pct(r['ci95'][0])}-{pct(r['ci95'][1])} | {pct(r['at_named_object'])} | "
            f"{mm(r['median_miss_named_mm'])} | {mm(r['median_miss_nearest_mm'])} | {per} |"
        )
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
