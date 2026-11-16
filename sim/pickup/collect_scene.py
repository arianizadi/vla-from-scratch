"""Collect a perception dataset: camera images and the scene's truth (scene.py) frame by frame, to train the eyes of
architecture v2 (tutorial/architecture-v2.md) on.

The scripted expert drives, as for the demonstrations (collect.py), so the frames show the states a rollout visits,
with the arm in the way. Nothing here is a demonstration: no actions or notes are stored, and every episode is kept,
failed ones too, because the scene's truth is right in any state. Every `stride`-th control step is recorded, from an
offset drawn per episode so that the episodes do not all sample the same phase of a stride; after a success,
`tail_steps` more frames are recorded, every one, with the robot holding its last command: a cut is the last thing that
happens in an episode, and without the tail the dataset would hold few frames of cut shoots.

    python -m sim.pickup.collect_scene --task reason_mix --episodes 2000 --out data/scene/reason_mix_2k --crop

Layout:

  <out>/meta.json              the task, the counts and the settings
  <out>/episodes.jsonl         one line per episode: file, seed, task, frames, steps, success, shoots (and how it ended)
  <out>/episodes/ep_<seed:07d>.parquet
                               one row per recorded frame: step (int32, the control step), front and wrist (JPEG bytes,
                               224 x 224), plant (the plant crop, with --crop) and scene (the truth, Scene.to_json)

Only plant tasks have a scene. Held-out tasks are allowed: the labels do not depend on the instruction.
"""

import argparse
import io
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .env import TASKS, PickupEnv
from .expert import ScriptedExpert, random_start
from .scene import PlantCrop, scene_truth

IMAGE_SIZE = 224
CROP_ZOOM = 2  # the plant crop: the front camera rendered at twice the size, the plant's window cut out
CAMERAS = ("front", "wrist")


def jpeg(img, quality=95) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def episode_file(seed: int) -> str:
    return f"episodes/ep_{seed:07d}.parquet"


def _ending(info) -> str:
    """How an episode ended: 'success', what was damaged, 'wrong shoot' or 'time'."""
    if info.get("success"):
        return "success"
    return info.get("damaged") or ("wrong shoot" if info.get("wrong_object") else "time")


def run_episode(env, expert, seed, stride=4, tail_steps=10, start_anywhere=True, crop=None) -> tuple[dict, dict]:
    """One episode of `seed` with `expert` driving. Returns (columns, record): the recorded frames as lists, by column,
    and the episode's line for episodes.jsonl."""
    obs, yaw = env.reset(seed=seed), 0.0
    if start_anywhere:
        obs, yaw = random_start(env, expert, seed)
    expert.reset(yaw=yaw)
    # the first recorded step, from a stream of its own: the layout of `seed` is unchanged
    offset = int(np.random.default_rng([seed, 59]).integers(stride))
    columns = {"step": [], **{c: [] for c in CAMERAS}, **({"plant": []} if crop is not None else {}), "scene": []}

    def record(obs):
        columns["step"].append(env.steps)
        for c in CAMERAS:
            columns[c].append(jpeg(obs[c]))
        if crop is not None:
            columns["plant"].append(jpeg(crop.render()))
        columns["scene"].append(scene_truth(env).to_json())

    done, info = False, {}
    while not done:
        if env.steps % stride == offset:
            record(obs)
        obs, done, info = env.step(expert.act())
    steps = env.steps
    if info["success"]:
        for _ in range(tail_steps):  # the robot holds its last command (for a cut, the jaws stay shut)
            record(obs)
            obs, _, _ = env.step(env.last_action.copy())
    entry = dict(
        file=episode_file(seed),
        seed=seed,
        task=env.task,
        frames=len(columns["step"]),
        steps=steps,
        success=bool(info["success"]),
        shoots=len(env.active),
        ending=_ending(info),
        **({"kind": env.reason_kind} if env.reason else {}),
        **({"scenario": env.scenario_info} if env.scenarios else {}),
    )
    return columns, entry


def _table(columns: dict) -> pa.Table:
    types = {"step": pa.int32(), "front": pa.binary(), "wrist": pa.binary(), "plant": pa.binary(), "scene": pa.string()}
    return pa.table({k: pa.array(v, type=types[k]) for k, v in columns.items()})


def _worker(args):
    task, seeds, out_dir, worker_id, options = args
    env = PickupEnv(
        task,
        image_size=IMAGE_SIZE,
        max_steps=options["max_steps"],
        scenario=options["scenario"],
        leaf_fix=options["leaf_fix"],
    )
    expert = ScriptedExpert(env, seed=worker_id)
    crop = PlantCrop(env, size=IMAGE_SIZE, zoom=CROP_ZOOM) if options["crop"] else None
    records = []
    try:
        for seed in seeds:
            columns, entry = run_episode(
                env,
                expert,
                seed,
                stride=options["stride"],
                tail_steps=options["tail_steps"],
                start_anywhere=options["random_start"],
                crop=crop,
            )
            pq.write_table(_table(columns), Path(out_dir) / entry["file"])
            records.append(entry)
    finally:
        if crop is not None:
            crop.close()
        env.close()
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description="collect camera images and the scene's truth for training the eyes")
    ap.add_argument("--task", default="reason_mix", help="a plant task (prune*, prune_named*, reason_*)")
    ap.add_argument("--episodes", type=int, default=2000, help="episodes to run; every one is kept, failed ones too")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="the first episode's seed; one seed per episode, consecutive")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4, help="record every stride-th control step")
    ap.add_argument(
        "--crop",
        action="store_true",
        help="also record the plant crop (scene.PlantCrop): the front camera's view narrowed onto the plant, from a "
        f"render at {CROP_ZOOM} times the size",
    )
    ap.add_argument(
        "--no_random_start",
        action="store_true",
        help="start every episode at the home pose (by default from a random pose above the table, as the "
        "demonstrations are collected with --random_start)",
    )
    ap.add_argument(
        "--tail_steps",
        type=int,
        default=10,
        help="after a success, record this many more frames, every one, of the robot holding its last command",
    )
    ap.add_argument("--scenario", default=None, help="test situations on top of the task (scenarios.py)")
    ap.add_argument("--leaf_fix", action="store_true", help="draw the leaves where their collision boxes are")
    ap.add_argument("--max_steps", type=int, default=None, help="episode limit (default: the task's)")
    args = ap.parse_args(argv)
    assert args.task in TASKS and TASKS[args.task].get("plant"), f"{args.task}: only plant tasks have a scene"
    assert args.episodes >= 1 and args.workers >= 1 and args.stride >= 1 and args.tail_steps >= 0

    out = Path(args.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    options = dict(
        stride=args.stride,
        tail_steps=args.tail_steps,
        random_start=not args.no_random_start,
        crop=args.crop,
        scenario=args.scenario,
        leaf_fix=args.leaf_fix,
        max_steps=args.max_steps,
    )
    seeds = list(range(args.seed, args.seed + args.episodes))
    workers = min(args.workers, len(seeds))
    chunks = [(args.task, seeds[i::workers], str(out), i, options) for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        records = sorted((r for rs in pool.map(_worker, chunks) for r in rs), key=lambda r: r["seed"])

    with open(out / "episodes.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    frames = sum(r["frames"] for r in records)
    successes = sum(r["success"] for r in records)
    meta = dict(
        task=args.task,
        episodes=len(records),
        frames=frames,
        stride=args.stride,
        cameras=[*CAMERAS, *(["plant"] if args.crop else [])],
        image_size=IMAGE_SIZE,
        crop_zoom=CROP_ZOOM if args.crop else None,
        driver="expert",
        random_start=not args.no_random_start,
        scenario=args.scenario,
        seed=args.seed,
        tail_steps=args.tail_steps,
        leaf_fix=args.leaf_fix,
        max_steps=args.max_steps,
        successes=successes,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"wrote {len(records)} episodes ({successes} successes) / {frames} frames to {out} in {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
