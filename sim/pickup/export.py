"""Export runs and evaluations as plain JSON, for plotting or for a results page.

Reads the same files as the dashboards (training logs and evaluation results) and writes:

    <out>/runs.json                      every run: progress, loss curve, one row per evaluation
    <out>/grasp/<run>__<task>__<variant>.json
                                         per-episode gripper and object positions of one evaluation,
                                         enough to draw a top-down map (only for evaluations that have them)
    <out>/tolerance.json                 grasp success against placement error, if it was measured
    <out>/traces/<run>__<task>__<variant>__trace_NNN_*.json
                                         rollouts saved with eval.py --traces (joint angles, gripper and
                                         object poses per control step), for replays

Each run in runs.json also lists its "probes": evaluations of one checkpoint under other conditions
(sampler settings, interventions such as frozen cameras or an expert prefix, disturbances), each with
the settings it was run with.

    python -m sim.pickup.export --out export/
    python -m sim.pickup.export --out export/ --runs dp_vit_lang_reactive --grasp best
"""

import argparse
import glob
import json
import math
import os
import re
import shutil
import statistics

from .runstate import Evals, discover, evaluation_row, snapshot


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
        closes=e.get("closes"),
        moved=(e.get("disturbed") or {}).get("to_xy")
        and dict(
            step=e["disturbed"]["step"],
            from_xy=[round(v, 4) for v in e["disturbed"]["from_xy"]],
            to_xy=[round(v, 4) for v in e["disturbed"]["to_xy"]],
        ),
    )


def copy_traces(results_dir, run, task, variant, out):
    """Copy the rollout traces of one evaluation next to the export; returns their relative paths."""
    files = []
    for path in sorted(glob.glob(os.path.join(results_dir, run, task, variant, "trace_*.json"))):
        name = f"{run}__{task}__{variant or 'plain'}__{os.path.basename(path)}"
        shutil.copy(path, os.path.join(out, "traces", name))
        files.append(f"traces/{name}")
    return files


def probe_row(variant, task, results):
    """One evaluation under non-default conditions: what was changed, and what happened."""
    served = re.search(r"(ema|checkpoint)_(\d+)\.pt$", str((results.get("server") or {}).get("checkpoint", "")))
    row = evaluation_row(int(served.group(2)) if served else None, None, task, results)
    row["weights"] = None if not served else ("ema" if served.group(1) == "ema" else "raw")
    row.update(
        variant=variant,
        exec_horizon=results.get("exec_horizon"),
        flow_steps=results.get("flow_steps"),
        noise=results.get("noise", "fresh"),
        random_start=results.get("random_start", False),
        seed_start=results.get("seed_start"),
        latency_ms=(results.get("inference_latency_ms") or {}).get("p50"),
        interventions=results.get("interventions") or {},
    )
    episodes = results.get("per_episode", [])
    moved = [e for e in episodes if (e.get("disturbed") or {}).get("to_xy")]
    if moved:  # how the episodes in which the object was actually moved went

        def distance(e, key):
            return math.dist(e["close_tcp_xy"], e["disturbed"][key])

        closed = [e for e in moved if e.get("close_tcp_xy")]
        row["moved"] = dict(
            episodes=len(moved),
            successes=sum(bool(e["success"]) for e in moved),
            retried=sum((e.get("closes") or 1) >= 2 for e in moved),
            median_mm_from_old_place=1000 * statistics.median(distance(e, "from_xy") for e in closed)
            if closed
            else None,
            median_mm_from_new_place=1000 * statistics.median(distance(e, "to_xy") for e in closed) if closed else None,
        )
    goal = [e for e in episodes if e.get("prefix_goal") and e.get("close_nearest")]
    if goal:  # after an expert prefix: did the policy close at the object the expert was heading for?
        row["closed_at_prefix_goal"] = dict(
            episodes=len(goal), count=sum(e["close_nearest"] == e["prefix_goal"] for e in goal)
        )
    return row


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
    os.makedirs(os.path.join(args.out, "traces"), exist_ok=True)

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
                json.dump(
                    dict(run=run["name"], task=task, checkpoint=checkpoint, weights=weights, episodes=episodes), f
                )
            written += 1
            for row in run["evaluations"]:
                if (row["checkpoint"], row["weights"], row["task"]) == (checkpoint, weights, task):
                    row["grasp_file"] = f"grasp/{name}"
        for row in run["evaluations"]:
            variant = f"ckpt_{row['checkpoint']}_{row['weights']}"
            row["trace_files"] = copy_traces(args.results, run["name"], row["task"], variant, args.out)
        run["probes"] = []
        for variant, task, results in evals.probes(run["name"]):
            row = probe_row(variant, task, results)
            episodes = [compact_episode(e) for e in results.get("per_episode", [])]
            if args.grasp == "all" and any(e["tcp"] for e in episodes):
                name = f"{run['name']}__{task}__{variant}.json"
                with open(os.path.join(args.out, "grasp", name), "w") as f:
                    json.dump(dict(run=run["name"], task=task, variant=variant, episodes=episodes), f)
                written += 1
                row["grasp_file"] = f"grasp/{name}"
            row["trace_files"] = copy_traces(args.results, run["name"], task, variant, args.out)
            run["probes"].append(row)
    # The scripted expert through the same evaluation loop: the reference for every map and table.
    # results/<expert>/<task>/results.json is the plain evaluation; subdirectories hold the expert
    # under other conditions (disturbances, planning again, random starts).
    state["expert"] = []
    expert = os.path.join(args.results, "expert")
    if os.path.isdir(expert):
        plain = [("", task, os.path.join(expert, task, "results.json")) for task in sorted(os.listdir(expert))]
        rows = [(v, t, evals.load(path)) for v, t, path in plain if os.path.exists(path)] + evals.probes("expert")
        for variant, task, results in rows:
            row = probe_row(variant or "plain", task, results)
            row["plans_again"] = bool((results.get("server") or {}).get("reactive", False))
            episodes = [compact_episode(e) for e in results.get("per_episode", [])]
            name = f"expert__{task}.json" if not variant else f"expert__{task}__{variant}.json"
            with open(os.path.join(args.out, "grasp", name), "w") as f:
                json.dump(dict(run="expert", task=task, variant=variant or None, episodes=episodes), f)
            written += 1
            row["grasp_file"] = f"grasp/{name}"
            row["trace_files"] = copy_traces(args.results, "expert", task, variant, args.out)
            state["expert"].append(row)

    with open(os.path.join(args.out, "runs.json"), "w") as f:
        json.dump(state, f)
    tolerance = os.path.join(args.results, "grasp_tolerance.json")
    if os.path.exists(tolerance):
        shutil.copy(tolerance, os.path.join(args.out, "tolerance.json"))
    print(f"{len(state['runs'])} runs, {written} per-episode files -> {args.out}")


if __name__ == "__main__":
    main()
