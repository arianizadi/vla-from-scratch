"""The scripted expert on the full-3D-reach situations (scenarios_reach.py), and how much of the plant the arm can
reach.

  python -m sim.pickup.reach_report expert --scenarios plain locked heights pitched blocked --episodes 100 \
      --start random --workers 50 --out results/pickup/reach/expert.json [--settle]
  python -m sim.pickup.reach_report map --out results/pickup/reach/map.json --workers 50

`expert`: the expert on `prune` (`plain`) or prune with each situation, evaluation layouts (seeds from 1,000,000) from
random starts (`--start random`, expert.random_start), clear starts (`clear`) or the home pose, the leaves drawn where
they are and unused shoots parked as built (as new datasets are collected). Per situation: episodes solved, damaged,
wrong (a cut outside the height band), out of time, control steps of the solved ones, how each cut was approached, and
the cut's error (scenarios_reach.cut_error): across the shoot, in height (z) and laterally, and in the hand's frame
(along the closing jaws, across them, the way the hand points).

`map`: the share of directions in which one shoot can be cut with the plate locked (scenarios_reach.plan_cut from the
arm's canonical start): one shoot on the plant of `prune` for a few seeds, every 15 degrees of azimuth (seen from the
robot, 180 = pointing at it), at heights over the `heights` range and elevations of prune (15 to 35 degrees) and of
`pitched` (45 to 70 up and down).
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

EVAL_SEED = 1_000_000


def _episode(job):
    try:
        return _run_episode(job)
    except Exception as error:  # an episode that could not be drawn or run is recorded, not fatal to the report
        scenario, seed, *_ = job
        return dict(
            scenario=scenario, seed=seed, success=False, steps=0, damaged=None, off_band=[], shoots=0, cuts=[],
            approach=[], draws=None, seconds=0.0, error=f"{type(error).__name__}: {error}",
        )


def _run_episode(job):
    scenario, seed, start, settle = job
    import mujoco  # noqa: F401  (each worker builds its own environment)

    from .env import PickupEnv
    from .expert import ScriptedExpert, clear_start, random_start
    from .scenarios_reach import cut_error

    key = (scenario, settle)
    env, expert = _cache.get(key, (None, None))
    if env is None:
        env = PickupEnv("prune", scenario=None if scenario == "plain" else scenario, leaf_fix=True, park_reset=True)
        expert = ScriptedExpert(env, seed=0, settle=settle)
        _cache[key] = (env, expert)
    t = time.time()
    env.reset(seed)
    yaw = 0.0
    if start == "random":
        _, yaw = random_start(env, expert, seed)
    elif start == "clear":
        _, yaw = clear_start(env, expert, seed)
    expert.reset(yaw=yaw)
    kinds, cuts, done, info = [], [], False, {}
    last = None
    while not done:
        action = expert.act()
        plan = getattr(expert, "reach_plan", None)
        if plan is not None and plan is not last:
            last = plan
            kinds.append(plan.kind)
        before = set(env.cut)
        _, done, info = env.step(action)
        cuts += [dict(shoot=n, **cut_error(env, n)) for n in sorted(set(env.cut) - before)]
    variation = env.scenario_info.get(scenario, {})
    return dict(
        scenario=scenario,
        seed=seed,
        success=bool(info["success"]),
        steps=env.steps,
        damaged=info.get("damaged"),
        off_band=info.get("cut_off_band", []),
        shoots=len(env.active),
        cuts=cuts,
        approach=kinds,
        draws=variation.get("draws"),
        seconds=round(time.time() - t, 1),
    )


_cache: dict = {}


def _stats(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return None
    a = np.abs(v)
    return dict(
        median_abs=round(float(np.median(a)), 2),
        p90_abs=round(float(np.percentile(a, 90)), 2),
        mean=round(float(v.mean()), 2),
        sd=round(float(v.std()), 2),
    )


def summarise(episodes):
    n = len(episodes)
    solved = [e for e in episodes if e["success"]]
    cuts = [c for e in episodes for c in e["cuts"]]
    across = [float(np.hypot(c["z_mm"], c["lat_mm"])) for c in cuts]
    kinds = {}
    for e in episodes:
        for k in e["approach"]:
            kinds[k] = kinds.get(k, 0) + 1
    return dict(
        episodes=n,
        successes=len(solved),
        damaged=sum(e["damaged"] is not None for e in episodes),
        off_band=sum(bool(e["off_band"]) for e in episodes),
        errors=sum(bool(e.get("error")) for e in episodes),
        timeout=sum(
            not e["success"] and e["damaged"] is None and not e["off_band"] and not e.get("error") for e in episodes
        ),
        steps_median=float(np.median([e["steps"] for e in solved])) if solved else None,
        seconds_median=float(np.median([e["seconds"] for e in episodes])) if episodes else None,
        draws_mean=round(float(np.mean([e["draws"] for e in episodes if e["draws"]])), 2)
        if any(e["draws"] for e in episodes)
        else None,
        approach=kinds,
        cuts=len(cuts),
        across_mm=dict(
            median=round(float(np.median(across)), 2) if cuts else None,
            p90=round(float(np.percentile(across, 90)), 2) if cuts else None,
        ),
        **{f"{k}_mm": _stats([c[f"{k}_mm"] for c in cuts]) for k in ("z", "lat", "along", "jaws", "side", "depth")},
        seconds_per_episode=round(float(np.mean([e["seconds"] for e in episodes])), 1),
    )


def expert_report(args):
    jobs = [(s, EVAL_SEED + k, args.start, args.settle) for s in args.scenarios for k in range(args.episodes)]
    episodes = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for e in pool.imap_unordered(_episode, jobs, chunksize=1):
            episodes.append(e)
            if len(episodes) % 25 == 0:
                print(f"{len(episodes)}/{len(jobs)} episodes, {sum(x['success'] for x in episodes)} solved", flush=True)
    episodes.sort(key=lambda e: (args.scenarios.index(e["scenario"]), e["seed"]))
    out = dict(
        start=args.start,
        settle=args.settle,
        seed_start=EVAL_SEED,
        by_scenario={s: summarise([e for e in episodes if e["scenario"] == s]) for s in args.scenarios},
        per_episode=episodes,
    )
    return out


# ------------------------------------------------------------------------------------------------ the reach map
MAP_AZIMUTHS = np.radians(np.arange(0, 360, 15))  # world azimuth seen from above: 180 points at the robot


def _map_cell(job):
    seed, height, elevation, rel_az = job
    import mujoco

    from .env import PickupEnv
    from .scenarios_reach import plan_cut, regrow, shoot_tip_clear

    env = _cache.get("map")
    if env is None:
        env = _cache["map"] = PickupEnv("prune", scenario="locked", leaf_fix=True, park_reset=True)
    env.reset(seed)
    keep = env.active[0]
    for n in env.active[1:]:
        env._park_shoot(n)
        del env.shoots[n]
    env.active = [keep]
    env._set_exclusions()
    regrow(env, keep, height, rel_az, elevation)  # (the robot is at azimuth 180 degrees seen from the plant)
    if not shoot_tip_clear(env, keep):
        return None  # a shoot into the pot or the flower: not a plant this scenario draws
    env._place_shoots()
    mujoco.mj_forward(env.model, env.data)
    plan = plan_cut(env, keep)
    return dict(
        seed=seed,
        height=height,
        elevation_deg=round(float(np.degrees(elevation)), 1),
        azimuth_deg=round(float(np.degrees(rel_az)), 1),
        reachable=plan is not None,
        kind=None if plan is None else plan.kind,
        opening=None if plan is None else plan.opening,
    )


def reach_map(args):
    heights = (0.03, 0.07, 0.11, 0.155, 0.175)
    elevations = np.radians([15, 25, 35, 45, 60, -45, -60])
    jobs = [
        (EVAL_SEED + s, h, e, a)
        for s in range(args.map_seeds)
        for h in heights
        for e in elevations
        for a in MAP_AZIMUTHS
    ]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        cells = [c for c in pool.map(_map_cell, jobs, chunksize=4) if c is not None]

    def share(cs):
        return round(sum(c["reachable"] for c in cs) / max(1, len(cs)), 3)

    by_az = {
        int(np.degrees(a)): share([c for c in cells if c["azimuth_deg"] == round(float(np.degrees(a)), 1)])
        for a in MAP_AZIMUTHS
    }
    by_el = {
        int(round(np.degrees(e))): share([c for c in cells if c["elevation_deg"] == round(float(np.degrees(e)), 1)])
        for e in elevations
    }
    by_h = {h: share([c for c in cells if c["height"] == h]) for h in heights}
    prune_like = [c for c in cells if 15 <= c["elevation_deg"] <= 35 and 0.07 <= c["height"] <= 0.155]
    return dict(
        overall=share(cells),
        prune_heights_and_elevations=share(prune_like),
        by_azimuth_deg=by_az,
        by_elevation_deg=by_el,
        by_height_m=by_h,
        cells=cells,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["expert", "map"])
    ap.add_argument("--scenarios", nargs="+", default=["plain", "locked", "heights", "pitched", "blocked"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--start", choices=["random", "clear", "home"], default="random")
    ap.add_argument("--settle", action="store_true", help="the expert settles before it closes (ScriptedExpert settle)")
    ap.add_argument("--map_seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t = time.time()
    out = expert_report(args) if args.what == "expert" else reach_map(args)
    out["wall_s"] = round(time.time() - t, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    if args.what == "expert":
        for s, r in out["by_scenario"].items():
            print(
                f"{s:8s} {r['successes']}/{r['episodes']} damaged {r['damaged']} off-band {r['off_band']} "
                f"timeout {r['timeout']} steps {r['steps_median']} across {r['across_mm']} z {r['z_mm']} "
                f"lat {r['lat_mm']} jaws sd {r['jaws_mm'] and r['jaws_mm']['sd']} side sd "
                f"{r['side_mm'] and r['side_mm']['sd']} depth sd {r['depth_mm'] and r['depth_mm']['sd']} "
                f"approach {r['approach']} {r['seconds_per_episode']} s/episode"
            )
    else:
        print({k: v for k, v in out.items() if k != "cells"})


if __name__ == "__main__":
    main()
