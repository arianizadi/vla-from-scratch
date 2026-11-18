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

With --attributes every frame's truth has each part's tie, sick and bud (architecture v3: scene.scene_truth with
attributes, "unreadable" where no camera can read one), also on a plant that no situation dresses, which then has no
tie, no sick shoot and no bud: a label the eyes learn as well. Without it a frame has them only where a situation that
draws them applies (ties, sick, buds), as before.

SceneRecorder writes this layout whoever drives. The expert never shows the eyes some states a policy reaches (jaws
shut beside a shoot they did not cut), so eval.py --record_scene records the rollouts of a closed-loop evaluation with
it, each frame labelled by the simulator; that dataset's meta.json says "source": "eval" and how they were driven.
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


def stride_offset(seed: int, stride: int) -> int:
    """The step of each stride that episode `seed` records (0 to stride - 1), from a stream of its own: the layout of
    `seed` is unchanged."""
    return int(np.random.default_rng([seed, 59]).integers(stride))


def _ending(info, stopped=False) -> str:
    """How an episode ended: 'success', what was damaged, 'wrong shoot', 'time', or 'stopped' (whoever drove ended it
    before its time was up: eval.py's slow level, once its verdict has stood)."""
    if info.get("success"):
        return "success"
    return info.get("damaged") or ("wrong shoot" if info.get("wrong_object") else "stopped" if stopped else "time")


def _table(columns: dict) -> pa.Table:
    types = {"step": pa.int32(), "front": pa.binary(), "wrist": pa.binary(), "plant": pa.binary(), "scene": pa.string()}
    return pa.table({k: pa.array(v, type=types[k]) for k, v in columns.items()})


class SceneRecorder:
    """Records the episodes of `env` (a PickupEnv of a plant task) into the dataset at `out`, whoever drives.

    start(seed) after env.reset and any random start; due() says whether env's control step now is one to record, every
    `stride`-th from the episode's offset (stride_offset); frame(obs) records the scene now: `obs` (env's newest
    observation), the plant crop if `crop`, and the simulator's truth; end(info, steps) writes the episode's file and
    returns its line for episodes.jsonl, also kept in `records` (`steps`: the episode's control steps); close() releases
    the crop's renderer. episodes.jsonl and meta.json are written apart, once every episode is (write_listing).

    keep_streams: rendering the crop puts the scenarios' random streams back as it found them. A knocked camera draws
    its noise from one (scenarios.Camera), and the crop is that camera's view: every crop would otherwise change the
    noise of the images that follow. eval.py's policy looks at those; the expert here does not, and the datasets
    collected here were made without keep_streams."""

    def __init__(
        self,
        out,
        env,
        crop: bool = True,
        stride: int = 4,
        keep_streams: bool = False,
        attributes: bool | None = None,
        attribute_views: bool = False,
    ):
        assert env.image_size == IMAGE_SIZE, f"the dataset's images are {IMAGE_SIZE} pixels, not {env.image_size}"
        self.out, self.env, self.stride, self.keep_streams = Path(out), env, stride, keep_streams
        self.attributes = attributes  # scene_truth's: None, where a situation draws them; True, on every frame
        self.attribute_views = attribute_views  # scene_truth's views: which camera can read each attribute
        self.crop = PlantCrop(env, size=IMAGE_SIZE, zoom=CROP_ZOOM) if crop else None
        self.records: list[dict] = []
        (self.out / "episodes").mkdir(parents=True, exist_ok=True)

    def start(self, seed: int):
        self.seed, self.offset = seed, stride_offset(seed, self.stride)
        plant = {"plant": []} if self.crop is not None else {}
        self.columns = {"step": [], **{c: [] for c in CAMERAS}, **plant, "scene": []}

    def due(self) -> bool:
        return self.env.steps % self.stride == self.offset

    def frame(self, obs):
        env, columns = self.env, self.columns
        columns["step"].append(env.steps)
        for c in CAMERAS:
            columns[c].append(jpeg(obs[c]))
        if self.crop is not None:
            columns["plant"].append(jpeg(self._crop()))
        columns["scene"].append(scene_truth(env, attributes=self.attributes, views=self.attribute_views).to_json())

    def _crop(self):
        if not self.keep_streams:
            return self.crop.render()
        streams = [s.rng for s in self.env.scenarios if getattr(s, "rng", None) is not None]
        states = [rng.bit_generator.state for rng in streams]
        image = self.crop.render()
        for rng, state in zip(streams, states, strict=True):
            rng.bit_generator.state = state
        return image

    def end(self, info: dict, steps: int) -> dict:
        env = self.env
        entry = dict(
            file=episode_file(self.seed),
            seed=self.seed,
            task=env.task,
            frames=len(self.columns["step"]),
            steps=steps,
            success=bool(info["success"]),
            shoots=len(env.active),
            ending=_ending(info, stopped=steps < env.max_steps),
            **({"kind": env.reason_kind} if env.reason else {}),
            **({"scenario": env.scenario_info} if env.scenarios else {}),
        )
        pq.write_table(_table(self.columns), self.out / entry["file"])
        self.records.append(entry)
        return entry

    def close(self):
        if self.crop is not None:
            self.crop.close()
            self.crop = None


def write_listing(out, records: list[dict], task: str, stride: int, crop: bool, **settings) -> dict:
    """Write episodes.jsonl (`records`, the episodes' lines, by seed) and meta.json: the task, the counts, the stride,
    the cameras, then `settings` in their order. Returns the meta."""
    out, records = Path(out), sorted(records, key=lambda r: r["seed"])
    with open(out / "episodes.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    meta = dict(
        task=task,
        episodes=len(records),
        frames=sum(r["frames"] for r in records),
        stride=stride,
        cameras=[*CAMERAS, *(["plant"] if crop else [])],
        image_size=IMAGE_SIZE,
        crop_zoom=CROP_ZOOM if crop else None,
        **settings,
        successes=sum(r["success"] for r in records),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def run_episode(env, expert, recorder: SceneRecorder, seed, tail_steps=10, start_anywhere=True) -> dict:
    """One episode of `seed` with `expert` driving, recorded by `recorder`: every stride-th step, and after a success
    `tail_steps` more, every one. Returns the episode's line for episodes.jsonl (its file is written)."""
    obs, yaw = env.reset(seed=seed), 0.0
    if start_anywhere:
        obs, yaw = random_start(env, expert, seed)
    expert.reset(yaw=yaw)
    recorder.start(seed)
    done, info = False, {}
    while not done:
        if recorder.due():
            recorder.frame(obs)
        obs, done, info = env.step(expert.act())
    steps = env.steps
    if info["success"]:
        for _ in range(tail_steps):  # the robot holds its last command (for a cut, the jaws stay shut)
            recorder.frame(obs)
            obs, _, _ = env.step(env.last_action.copy())
    return recorder.end(info, steps)


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
    recorder = SceneRecorder(
        out_dir,
        env,
        crop=options["crop"],
        stride=options["stride"],
        attributes=options["attributes"] or None,
        attribute_views=options["attribute_views"],
    )
    try:
        for seed in seeds:
            run_episode(
                env, expert, recorder, seed, tail_steps=options["tail_steps"], start_anywhere=options["random_start"]
            )
    finally:
        recorder.close()
        env.close()
    return recorder.records


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
    ap.add_argument(
        "--attributes",
        action="store_true",
        help="every frame's truth with each part's tie, sick and bud (architecture v3), also on plants no situation "
        "dresses",
    )
    ap.add_argument(
        "--attribute_views",
        action="store_true",
        help="with --attributes: each part also says which camera (front, wrist) can read each attribute, for "
        "measuring eyes that look through one camera",
    )
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
        attributes=args.attributes,
        attribute_views=args.attribute_views,
    )
    seeds = list(range(args.seed, args.seed + args.episodes))
    workers = min(args.workers, len(seeds))
    chunks = [(args.task, seeds[i::workers], str(out), i, options) for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        records = [r for rs in pool.map(_worker, chunks) for r in rs]

    meta = write_listing(
        out,
        records,
        args.task,
        args.stride,
        args.crop,
        driver="expert",
        random_start=not args.no_random_start,
        scenario=args.scenario,
        seed=args.seed,
        tail_steps=args.tail_steps,
        leaf_fix=args.leaf_fix,
        max_steps=args.max_steps,
        **({"attributes": True} if args.attributes else {}),
        **({"attribute_views": True} if args.attribute_views else {}),
    )
    episodes, successes, frames = meta["episodes"], meta["successes"], meta["frames"]
    print(f"wrote {episodes} episodes ({successes} successes) / {frames} frames to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
