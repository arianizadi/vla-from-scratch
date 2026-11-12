"""Export runs and evaluations as plain JSON, for plotting or for a results page.

Reads the same files as the dashboards (training logs and evaluation results) and writes:

    <out>/runs.json                      every run: progress, loss curve, one row per evaluation
    <out>/grasp/<run>__<task>__<variant>.json
                                         per-episode gripper and object positions of one evaluation,
                                         enough to draw a top-down map (only for evaluations that have them)
    <out>/tolerance.json                 grasp success against placement error, if it was measured

    python -m sim.pickup.export --out export/
    python -m sim.pickup.export --out export/ --runs dp_vit_lang_reactive --grasp best
"""

import argparse
import json
import os
import shutil

from .runstate import Evals, discover, snapshot


def compact_episode(e):
    """The fields of one evaluated episode that a map of the grasp attempt needs."""
    return dict(
        seed=e["seed"],
        target=e["target"],
        success=bool(e["success"]),
        steps=e["steps"],
        wrong_object=bool(e.get("wrong_object", False)),
        close_step=e.get("close_step"),
        nearest=e.get("close_nearest"),
        tcp=[round(v, 4) for v in e["close_tcp_xy"]] if e.get("close_tcp_xy") else None,
        objects={k: [round(v, 4) for v in xy] for k, xy in (e.get("close_object_xy") or {}).items()},
        yaw_err=None if e.get("close_yaw_err_target") is None else round(e["close_yaw_err_target"], 4),
        z_above=None if e.get("close_z_above_grasp") is None else round(e["close_z_above_grasp"], 4),
    )


def main():
    p = argparse.ArgumentParser(description="Export runs and evaluations as JSON.")
    p.add_argument("--out", required=True)
    p.add_argument("--experiments", default="experiments/pickup")
    p.add_argument("--results", default="results/pickup")
    p.add_argument("--runs", nargs="*", default=None, help="only runs whose name contains one of these")
    p.add_argument("--curve_points", type=int, default=300, help="points kept per loss curve")
    p.add_argument(
        "--grasp",
        choices=["all", "best", "none"],
        default="all",
        help="which evaluations get a per-episode file: all, the best and the last of each run, or none",
    )
    args = p.parse_args()

    runs = discover(args.experiments, {})
    if args.runs:
        runs = [r for r in runs if any(s in r.name for s in args.runs)]
    evals = Evals(args.results)
    state = snapshot(runs, evals, args.curve_points)
    state.pop("gpus", None)  # a property of the machine, not of the runs
    os.makedirs(os.path.join(args.out, "grasp"), exist_ok=True)

    written = 0
    for run in state["runs"]:
        rows, _ = evals.of(run["name"])
        keep = set()
        if args.grasp == "best" and rows:
            best = max(rows, key=lambda r: (r[3]["success_rate"], r[0]))
            last = max(rows, key=lambda r: (r[0], r[1]))
            keep = {(best[0], best[1], best[2]), (last[0], last[1], last[2])}
        for checkpoint, weights, task, results in rows:
            if args.grasp == "none" or (args.grasp == "best" and (checkpoint, weights, task) not in keep):
                continue
            episodes = [compact_episode(e) for e in results.get("per_episode", [])]
            if not any(e["tcp"] for e in episodes):
                continue
            name = f"{run['name']}__{task}__ckpt_{checkpoint}_{weights}.json"
            with open(os.path.join(args.out, "grasp", name), "w") as f:
                json.dump(dict(run=run["name"], task=task, checkpoint=checkpoint, weights=weights, episodes=episodes), f)
            written += 1
            for row in run["evaluations"]:
                if (row["checkpoint"], row["weights"], row["task"]) == (checkpoint, weights, task):
                    row["grasp_file"] = f"grasp/{name}"
    # The scripted expert's evaluation, if present, is the reference for every map.
    expert = os.path.join(args.results, "expert")
    if os.path.isdir(expert):
        for task in sorted(os.listdir(expert)):
            path = os.path.join(expert, task, "results.json")
            if os.path.exists(path):
                with open(path) as f:
                    results = json.load(f)
                episodes = [compact_episode(e) for e in results.get("per_episode", [])]
                with open(os.path.join(args.out, "grasp", f"expert__{task}.json"), "w") as f:
                    json.dump(dict(run="expert", task=task, checkpoint=None, weights=None, episodes=episodes), f)
                written += 1

    with open(os.path.join(args.out, "runs.json"), "w") as f:
        json.dump(state, f)
    tolerance = os.path.join(args.results, "grasp_tolerance.json")
    if os.path.exists(tolerance):
        shutil.copy(tolerance, os.path.join(args.out, "tolerance.json"))
    print(f"{len(state['runs'])} runs, {written} per-episode files -> {args.out}")


if __name__ == "__main__":
    main()
