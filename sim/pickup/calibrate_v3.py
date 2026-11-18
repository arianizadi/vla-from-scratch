"""Two settings of architecture v3's runner (tutorial/architecture-v3.md, step 4), measured with the scripted expert on
CPU, without images:

  --starts   how many layouts of each situation the expert solves from starts kept clear of the plant
             (expert.clear_start), for each clearance given: CLEARANCE is the smallest at which it solves at least 97
             of 100 of each
  --gate     where the jaws are when they shut on a shoot: their height above the main stem's base against the height
             at which the shoot leaves the stem (the memory's height of a part), and how far they are from the stem's
             axis; and where they are when they shut for anything else (pressing a button). The cut gate
             (slow_level.cut_gate) maps the jaws' height onto the stem with the offset measured here, and its band of
             heights among the shoots is set from these numbers

    python -m sim.pickup.calibrate_v3 --starts 0.12,0.17,0.22 --gate --episodes 100 --workers 8 --out calib.json

Layouts are validation layouts (seeds from 900,000).
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

SITUATIONS = ("prune", "order", "nothing_to_do", "ties", "sick", "buds")
SEED = 900_000
SHUT = 0.45  # a gripper command below this shuts the jaws (slow_level.JAWS_CUT[0])
JAWS_OPEN_NOW = 0.3  # jaws open more than this are shut by such a command (slow_level.GATE_OPEN)


def _env(situation):
    from .env import PickupEnv

    env = PickupEnv("prune", scenario=None if situation == "prune" else situation)
    env.get_obs = lambda: {"state": env.state()}  # no images
    return env


def episode(env, expert, seed, clearance, gate=False) -> dict:
    """One episode of the expert from a clear start. With `gate`, every run of control steps whose command shuts
    the jaws: where the gripper was (height above the stem's base, distance from its axis), the shoot it cut in that
    run (None: none), and the heights of the plant's shoots."""
    from .env import STEM_BASE
    from .expert import clear_start

    env.reset(seed=seed)
    _, yaw = clear_start(env, expert, seed, clearance)
    expert.reset(yaw=yaw)
    base, up = env.stem_axis()
    heights = {name: float(env.shoots[name]["origin"][2] - STEM_BASE) for name in env.active}  # as scene_truth
    closings, run = [], None
    done, info = False, {}
    while not done:
        action = expert.act()
        shut = float(action[7]) < SHUT
        if gate and shut and run is None:
            rel = env.tcp_pos() - base
            run = dict(
                step=int(env.steps),
                height=float(rel @ up),
                axis=float(np.linalg.norm(rel - up * (rel @ up))),
                cut=None,
                target=env.target,
                jaws=float(env.state()[7]),  # how far open the jaws were when the command shut them
            )
        before = set(env.cut)
        _, done, info = env.step(action)
        if run is not None:
            new = sorted(set(env.cut) - before)
            if new:
                run["cut"] = new[0]
            if not shut or done:
                closings.append(run)
                run = None
    return dict(
        seed=int(seed),
        success=bool(info.get("success")),
        steps=int(env.steps),
        heights=heights,
        closings=closings,
    )


def _worker(args):
    from .expert import ScriptedExpert

    situation, seeds, clearance, gate = args
    env = _env(situation)
    expert = ScriptedExpert(env, seed=seeds[0])
    try:
        return [episode(env, expert, seed, clearance, gate) for seed in seeds]
    finally:
        env.close()


def run(situation, clearance, episodes, seed=SEED, workers=1, gate=False) -> list[dict]:
    seeds = list(range(seed, seed + episodes))
    chunks = [(situation, seeds[i::workers], clearance, gate) for i in range(workers) if seeds[i::workers]]
    if workers > 1:
        with mp.get_context("spawn").Pool(len(chunks)) as pool:
            results = pool.map(_worker, chunks)
    else:
        results = [_worker(chunk) for chunk in chunks]
    return sorted((r for chunk in results for r in chunk), key=lambda r: r["seed"])


def gate_numbers(episodes: list[dict]) -> dict:
    """The jaws on a shoot they cut: height above the stem's base minus the shoot's height (offset), and their distance
    from the axis; open jaws shut on no shoot: their height and distance; and, with the offset's median, how
    often the nearest shoot by the mapped height is the one cut, and the largest error of the mapped height."""
    cuts, other = [], []
    for e in episodes:
        for c in e["closings"]:
            if c["cut"] is not None:
                cuts.append((c["height"], c["axis"], e["heights"][c["cut"]], e["heights"]))
            elif c["jaws"] > JAWS_OPEN_NOW:  # jaws open shut on no shoot (one already shut, as to press, is not)
                other.append((c["height"], c["axis"]))
    if not cuts:
        return {}
    offsets = np.array([h - own for h, _, own, _ in cuts])
    offset = float(np.median(offsets))
    nearest_right, errors = 0, []
    for h, _, own, heights in cuts:
        mapped = h - offset
        nearest = min(heights.values(), key=lambda x: abs(x - mapped))
        nearest_right += nearest == own
        errors.append(abs(mapped - own))

    def spread(values):
        values = np.asarray(values)
        return dict(
            min=float(values.min()),
            p5=float(np.percentile(values, 5)),
            median=float(np.median(values)),
            p95=float(np.percentile(values, 95)),
            max=float(values.max()),
        )

    return dict(
        cuts=len(cuts),
        offset=spread(offsets),
        cut_height=spread([h for h, _, _, _ in cuts]),
        cut_axis=spread([a for _, a, _, _ in cuts]),
        mapped_error=spread(errors),
        nearest_is_the_one_cut=int(nearest_right),
        other_closings=len(other),
        other_height=spread([h for h, _ in other]) if other else None,
        other_axis=spread([a for _, a in other]) if other else None,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="the clear start and the cut gate of architecture v3's runner, measured")
    ap.add_argument("--situations", default=",".join(SITUATIONS))
    ap.add_argument("--starts", default="", help="comma-separated clearances (metres) to try, e.g. 0.12,0.17,0.22")
    ap.add_argument("--gate", action="store_true", help="measure where the jaws shut (at the clearance CLEARANCE)")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from .expert import CLEARANCE

    out = dict(seed=args.seed, episodes=args.episodes, starts={}, gate={})
    for clearance in [float(c) for c in args.starts.split(",") if c]:
        out["starts"][clearance] = {}
        for situation in args.situations.split(","):
            t = time.time()
            records = run(situation, clearance, args.episodes, args.seed, args.workers)
            solved = sum(r["success"] for r in records)
            out["starts"][clearance][situation] = solved
            print(f"clearance {clearance}: {situation} {solved}/{len(records)} ({time.time() - t:.0f} s)", flush=True)
    if args.gate:
        everything = []
        for situation in args.situations.split(","):
            records = run(situation, CLEARANCE, args.episodes, args.seed, args.workers, gate=True)
            out["gate"][situation] = gate_numbers(records)
            everything += records
        out["gate"]["all"] = gate_numbers(everything)
        print(json.dumps(out["gate"]["all"], indent=1), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
