"""The census of architecture v3 (tutorial/architecture-v3.md, E14): how often each attribute of a shoot can be read.

The scripted expert drives layouts of a situation (ties, sick, buds; or plain pruning) without rendering, as it does
for the demonstrations, and at every look (every LOOK_EVERY control steps, as the slow level looks: 0.4 s) the
simulator's rays say, for every shoot and every attribute (tie, sick, bud), whether a camera can read it
(scene.attribute_views: the tie's place, a leaf, the tip seen, and nothing of a cut shoot). Each shoot of each episode
counts once per attribute, in each of:

  first_look         readable at the first look
  before_first_cut   readable at some look before the episode's first cut (the first look included)
  before_own_cut     readable at some look while the shoot was still on the plant
  never              not readable at any look while it was on the plant

by any camera (front or wrist) and by the front camera alone, and per value of the attribute (a bud that is there, and
one that is not). How much the plan will have to look for on rungs 3 and 4 is what the census measures.

    python -m sim.pickup.census --situations ties,sick,buds --episodes 1000 --workers 8 --out census.json

Episodes start at the home pose, as an evaluation of v3 does (clear of the plant); --random_start starts them where the
demonstrations start. Layouts are validation layouts (seeds from 900,000) unless --seed says otherwise.
"""

import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

LOOK_EVERY = 8  # control steps between two looks (the fast level's chunk: 0.4 s)
COUNTS = ("first_look", "before_first_cut", "before_own_cut", "never")
VIEWS = ("any", "front")
SEED = 900_000


def run_episode(env, expert, seed: int, random_start: bool = False) -> dict:
    """One episode of the expert on `seed`; returns, per shoot (lowest first), its drawn attributes and, per attribute
    and view (any camera, front alone), the control steps of the looks that could read it, with the steps of the
    looks, of the episode's first cut and of each shoot's cut."""
    from .expert import random_start as start_anywhere
    from .scene import TRUTH_ATTRIBUTES, _Camera, attribute_views, drawn_attributes

    env.reset(seed=seed)
    yaw = 0.0
    if random_start:
        _, yaw = start_anywhere(env, expert, seed)
    expert.reset(yaw=yaw)
    drawn = drawn_attributes(env)
    shoots = list(env.active)
    readable = {name: {attr: {view: [] for view in VIEWS} for attr in TRUTH_ATTRIBUTES} for name in shoots}
    looks, cut_at = [], {}
    done, info = False, {}
    while True:
        if env.steps % LOOK_EVERY == 0:
            looks.append(env.steps)
            cameras = (_Camera(env, "front"), _Camera(env, "wrist"))
            for name in shoots:
                for attr, (front, wrist) in attribute_views(env, name, cameras).items():
                    if front or wrist:
                        readable[name][attr]["any"].append(env.steps)
                    if front:
                        readable[name][attr]["front"].append(env.steps)
        if done:
            break
        _, done, info = env.step(expert.act())
        for name in env.cut:
            cut_at.setdefault(name, env.steps)
    return dict(
        seed=int(seed),
        success=bool(info.get("success")),
        steps=int(env.steps),
        looks=looks,
        first_cut=min(cut_at.values(), default=None),
        shoots=[dict(name=name, drawn=drawn[name], cut=cut_at.get(name), readable=readable[name]) for name in shoots],
    )


def tally(episodes: list[dict]) -> dict:
    """Per attribute, view and COUNTS, the shoots counted, in all and per drawn value: {"shoots", COUNTS...}."""
    from .scene import TRUTH_ATTRIBUTES

    out = {}
    for attr in TRUTH_ATTRIBUTES:
        out[attr] = {}
        for view in VIEWS:
            total, by_value = Counter(), {}
            for episode in episodes:
                first_look = episode["looks"][0]
                first_cut = episode["first_cut"]
                for shoot in episode["shoots"]:
                    steps = shoot["readable"][attr][view]
                    own_cut = shoot["cut"]
                    on_plant = [s for s in steps if own_cut is None or s < own_cut]
                    counts = dict(
                        shoots=1,
                        first_look=int(first_look in steps),
                        before_first_cut=int(any(first_cut is None or s < first_cut for s in steps)),
                        before_own_cut=int(bool(on_plant)),
                        never=int(not on_plant),
                    )
                    value = shoot["drawn"][attr]
                    total.update(counts)
                    by_value.setdefault(value, Counter()).update(counts)
            out[attr][view] = dict(
                **{key: int(total[key]) for key in ("shoots", *COUNTS)},
                by_value={v: {key: int(c[key]) for key in ("shoots", *COUNTS)} for v, c in sorted(by_value.items())},
            )
    return out


def _worker(args) -> list[dict]:
    situation, seeds, random_start = args
    from .env import PickupEnv
    from .expert import ScriptedExpert

    env = PickupEnv("prune", scenario=situation)
    env.get_obs = lambda: {"state": env.state()}  # no rendering: the census needs the rays only
    expert = ScriptedExpert(env, seed=seeds[0] if seeds else 0)
    try:
        return [run_episode(env, expert, seed, random_start) for seed in seeds]
    finally:
        env.close()


def census(situation: str | None, episodes: int, seed: int = SEED, workers: int = 1, random_start=False) -> dict:
    """The census of one situation (None: plain pruning) over `episodes` layouts from `seed`."""
    seeds = list(range(seed, seed + episodes))
    chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
    started = time.time()
    if workers > 1:
        with mp.get_context("spawn").Pool(len(chunks)) as pool:
            results = pool.map(_worker, [(situation, chunk, random_start) for chunk in chunks])
    else:
        results = [_worker((situation, chunk, random_start)) for chunk in chunks]
    records = sorted((r for chunk in results for r in chunk), key=lambda r: r["seed"])
    return dict(
        situation=situation or "prune",
        episodes=len(records),
        seeds=[seed, seed + episodes - 1],
        random_start=bool(random_start),
        look_every=LOOK_EVERY,
        successes=sum(r["success"] for r in records),
        wall_time_s=round(time.time() - started, 1),
        attributes=tally(records),
    )


def table(results: list[dict], view: str = "any") -> str:
    """A Markdown table: per situation and attribute, the share of shoots readable at the first look, before the first
    cut, before their own cut, and never."""
    rows = [
        "| Situation | Attribute | Shoots | First look | Before the first cut | Before its own cut | Never |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        for attr, views in result["attributes"].items():
            counts = views[view]
            n = counts["shoots"]
            cells = [f"{100 * counts[key] / n:.1f}%" if n else "-" for key in COUNTS]
            rows.append(f"| {result['situation']} | {attr} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="how often each attribute of a shoot can be read (architecture v3, E14)")
    ap.add_argument("--situations", default="ties,sick,buds", help="comma-separated; 'prune' for no situation")
    ap.add_argument("--episodes", type=int, default=1000, help="layouts per situation")
    ap.add_argument("--seed", type=int, default=SEED, help="the first layout's seed (validation layouts by default)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--random_start", action="store_true", help="start where the demonstrations start, not at home")
    ap.add_argument("--out", required=True, help="the results, as JSON")
    args = ap.parse_args(argv)
    results = []
    for situation in args.situations.split(","):
        name = None if situation == "prune" else situation
        result = census(name, args.episodes, args.seed, args.workers, args.random_start)
        results.append(result)
        print(f"{result['situation']}: {result['episodes']} episodes in {result['wall_time_s']} s", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    for view in VIEWS:
        print(f"\nreadable by {'any camera' if view == 'any' else 'the front camera'}:\n{table(results, view)}")


if __name__ == "__main__":
    main()
