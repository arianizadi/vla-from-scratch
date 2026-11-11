"""How much positioning error does a top-down grasp tolerate?

Runs the scripted expert with a fixed horizontal offset added to its grasp point (random direction,
on top of the expert's own ~5 mm placement scatter) and reports success per object against the
offset. Nothing is rendered, so this takes about a minute.

  python -m sim.pickup.grasp_tolerance --episodes 320 --workers 8 --out results/pickup/grasp_tolerance.json
"""

import argparse
import json
import multiprocessing as mp
from collections import defaultdict

import numpy as np

from .env import PickupEnv
from .expert import ScriptedExpert

SEED_START = 2_500_000  # disjoint from the demonstration (< 1_000_000) and evaluation (1_000_000+) seeds


def _worker(args):
    task, offset, seeds, worker_id = args
    env = PickupEnv(task)
    env.get_obs = lambda: {"state": env.state()}  # no rendering
    expert = ScriptedExpert(env, seed=worker_id)
    rng = np.random.default_rng(1234 + worker_id)
    out = []
    for seed in seeds:
        env.reset(seed=seed)
        expert.reset()
        angle = rng.uniform(-np.pi, np.pi)
        shift = offset * np.array([np.cos(angle), np.sin(angle), 0.0])
        expert.waypoints = [(p + shift, grip, speed, dwell) for p, grip, speed, dwell in expert.waypoints]
        done, info = False, {}
        while not done:
            _, done, info = env.step(expert.act())
        out.append((env.target, bool(info["success"])))
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_target")
    ap.add_argument("--episodes", type=int, default=320, help="episodes per offset")
    ap.add_argument("--offsets_mm", type=float, nargs="+", default=[0, 5, 10, 15, 20, 25, 30])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = list(range(SEED_START, SEED_START + args.episodes))
    table = {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for off in args.offsets_mm:
            jobs = [(args.task, off / 1000.0, seeds[i :: args.workers], i) for i in range(args.workers)]
            counts = defaultdict(lambda: [0, 0])
            for res in pool.map(_worker, jobs):
                for target, ok in res:
                    counts[target][0] += ok
                    counts[target][1] += 1
                    counts["all"][0] += ok
                    counts["all"][1] += 1
            table[off] = {k: dict(successes=s, episodes=n, rate=s / n) for k, (s, n) in sorted(counts.items())}
            row = "  ".join(f"{k}={v['rate']:.0%}" for k, v in table[off].items())
            print(f"offset {off:4.0f} mm: {row}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(task=args.task, episodes_per_offset=args.episodes, by_offset_mm=table), f, indent=2)


if __name__ == "__main__":
    main()
