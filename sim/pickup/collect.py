"""Collect scripted-expert demonstrations and write them as a LeRobot v2.0 dataset.

The layout (meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl, data/chunk-000/episode_*.parquet
with inline JPEG image columns) is what vla_foundry's `--type lerobot` preprocessing converter reads.

Per-frame columns: front, wrist (JPEG bytes), state [8], action [8], plus LeRobot bookkeeping.
action[t] is the command applied after observing frame t.

  python -m sim.pickup.collect --task pick_cube --episodes 200 --out data/lerobot/pick_cube
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

from .disturb import Disturbance
from .env import CONTROL_HZ, TASKS, PickupEnv
from .expert import ScriptedExpert, random_start

CAMERAS = ("front", "wrist")


def _jpeg(img, quality=95):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _worker(args):
    task, jobs, out_dir, worker_id, disturb_prob, anywhere, hold_steps, drift_prob, drift_cm_s = args
    env = PickupEnv(task)
    expert = ScriptedExpert(env, seed=worker_id, reactive=disturb_prob > 0 or drift_prob > 0)
    results = []
    for seed, target_index in jobs:
        obs, start_yaw = env.reset(seed=seed, target_index=target_index), 0.0
        if anywhere:
            obs, start_yaw = random_start(env, expert, seed)
        expert.reset(yaw=start_yaw)
        # Per-episode draw (from the seed, so a dataset can be rebuilt): is this episode disturbed, when
        # and by how much. Half of the moves come during the reach, half as the gripper closes.
        rng = np.random.default_rng([seed, 7])
        kind = str(rng.choice(["reach", "grasp"])) if rng.random() < disturb_prob else "none"
        step, distance = int(rng.integers(5, 30)), float(rng.uniform(0.05, 0.10))
        speed = 0.0
        if drift_prob > 0:  # drawn after the others, so that datasets made without drift are unchanged
            drifts, speed = rng.random() < drift_prob, float(rng.uniform(0.5, 1.5)) * drift_cm_s / 100
            if kind == "none" and drifts:
                kind = "drift"
        disturbance = Disturbance(env, kind, step=step, distance=distance, seed=seed, speed=speed)
        frames = {c: [] for c in CAMERAS}
        states, actions = [], []
        done, info = False, {}
        while not done:
            # The first hold_steps frames record the robot at rest under its hold command.
            action = env.last_action.copy() if len(actions) < hold_steps else expert.act()
            for c in CAMERAS:
                frames[c].append(_jpeg(obs[c]))
            states.append(obs["state"])
            actions.append(action)
            disturbance.update(action)
            obs, done, info = env.step(action)
        if not info["success"]:
            results.append(dict(seed=seed, success=False))
            continue
        tmp = Path(out_dir) / "tmp" / f"seed_{seed:07d}_{target_index or 0}.parquet"
        table = pa.table(
            {
                **{c: pa.array([{"bytes": b, "path": None} for b in frames[c]]) for c in CAMERAS},
                "state": pa.array([s.tolist() for s in states], type=pa.list_(pa.float32())),
                "action": pa.array([a.tolist() for a in actions], type=pa.list_(pa.float32())),
            }
        )
        pq.write_table(table, tmp)
        results.append(
            dict(
                seed=seed,
                target_index=target_index or 0,
                success=True,
                length=len(actions),
                instruction=env.instruction,
                path=str(tmp),
                disturbed=kind if disturbance.record and disturbance.record["to_xy"] else "none",
                replans=expert.replans,
                tracks=expert.tracks,
            )
        )
    env.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_cube")
    ap.add_argument("--episodes", type=int, default=200, help="number of successful episodes to keep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="first environment seed (training seeds are < 1_000_000)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--targets_per_layout",
        type=int,
        default=1,
        help="demonstrate this many different targets in every layout (pick_target: up to 3). With 1, each "
        "layout appears with a single instruction, and nothing in the data shows that the instruction matters.",
    )
    ap.add_argument(
        "--disturb_prob",
        type=float,
        default=0.0,
        help="share of episodes in which the named object is moved once, during the reach or as the gripper "
        "closes, and the expert recovers (plans again, retries the grasp). 0: every demonstration goes to plan, "
        "so the data holds no example of a correction.",
    )
    ap.add_argument(
        "--drift_prob",
        type=float,
        default=0.0,
        help="share of the episodes not otherwise disturbed in which the named object keeps sliding on a smooth "
        "random path until it is held, and the expert follows it",
    )
    ap.add_argument(
        "--drift_cm_s",
        type=float,
        default=2.0,
        help="typical drift speed in cm/s; each drifting episode draws its speed between 0.5 and 1.5 times this",
    )
    ap.add_argument(
        "--random_start",
        action="store_true",
        help="start every episode from a random gripper pose above the table instead of the home pose. From "
        "one fixed start, the arm's own position gives the goal away after a few steps of any reach.",
    )
    ap.add_argument(
        "--hold_steps",
        type=int,
        default=0,
        help="begin every episode with this many frames of the robot holding still. A policy that is given its "
        "previous action is served the hold command before its first step; without such frames, the first "
        "training window fills that slot with a copy of the expert's first command, which already points at the "
        "target.",
    )
    args = ap.parse_args()
    num_objects = TASKS[args.task]["num_objects"]
    assert 1 <= args.targets_per_layout <= num_objects, f"{args.task} has {num_objects} objects per layout"
    per_layout = args.targets_per_layout

    out = Path(args.out)
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    kept, attempted, successes, next_seed = [], 0, 0, args.seed
    with mp.get_context("spawn").Pool(args.workers) as pool:
        while len(kept) < args.episodes:
            need = args.episodes - len(kept)
            seeds = list(range(next_seed, next_seed + max(-(-(need + need // 10) // per_layout), args.workers)))
            next_seed = seeds[-1] + 1
            # With one target per layout the seed's own choice is kept (target_index None).
            jobs = [(seed, i if per_layout > 1 else None) for seed in seeds for i in range(per_layout)]
            extra = (args.disturb_prob, args.random_start, args.hold_steps, args.drift_prob, args.drift_cm_s)
            chunks = [(args.task, jobs[i :: args.workers], str(out), i, *extra) for i in range(args.workers)]
            for res in pool.map(_worker, chunks):
                attempted += len(res)
                successes += sum(r["success"] for r in res)
                kept.extend(r for r in res if r["success"])
    kept = sorted(kept, key=lambda r: (r["seed"], r["target_index"]))[: args.episodes]

    tasks, episodes, index = {}, [], 0
    for ep_idx, r in enumerate(kept):
        task_idx = tasks.setdefault(r["instruction"], len(tasks))
        table = pq.read_table(r["path"])
        n = table.num_rows
        table = table.append_column("timestamp", pa.array(np.arange(n, dtype=np.float32) / CONTROL_HZ))
        table = table.append_column("frame_index", pa.array(np.arange(n, dtype=np.int64)))
        table = table.append_column("episode_index", pa.array(np.full(n, ep_idx, dtype=np.int64)))
        table = table.append_column("index", pa.array(np.arange(index, index + n, dtype=np.int64)))
        table = table.append_column("task_index", pa.array(np.full(n, task_idx, dtype=np.int64)))
        pq.write_table(table, out / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")
        episodes.append(
            dict(
                episode_index=ep_idx,
                tasks=[r["instruction"]],
                length=n,
                seed=r["seed"],
                disturbed=r["disturbed"],
                replans=r["replans"],
                tracks=r["tracks"],
            )
        )
        index += n
    for f in (out / "tmp").glob("*.parquet"):
        f.unlink()
    (out / "tmp").rmdir()

    with open(out / "meta" / "episodes.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in episodes)
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        f.writelines(json.dumps(dict(task_index=i, task=t)) + "\n" for t, i in tasks.items())
    info = dict(
        codebase_version="v2.0",
        robot_type="franka_panda_mujoco",
        sim_task=args.task,
        fps=CONTROL_HZ,
        total_episodes=len(episodes),
        total_frames=index,
        total_tasks=len(tasks),
        expert_success_rate=successes / attempted if attempted else None,
        expert_attempted=attempted,
        targets_per_layout=per_layout,
        disturb_prob=args.disturb_prob,
        random_start=args.random_start,
        hold_steps=args.hold_steps,
        disturbed_episodes=sum(r["disturbed"] != "none" for r in kept),
        drift_prob=args.drift_prob,
        drift_cm_s=args.drift_cm_s if args.drift_prob > 0 else None,
        drifting_episodes=sum(r["disturbed"] == "drift" for r in kept),
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        features={
            "front": dict(dtype="image", shape=[224, 224, 3]),
            "wrist": dict(dtype="image", shape=[224, 224, 3]),
            "state": dict(dtype="float32", shape=[8], names=["q1", "q2", "q3", "q4", "q5", "q6", "q7", "gripper_open"]),
            "action": dict(
                dtype="float32", shape=[8], names=["q1", "q2", "q3", "q4", "q5", "q6", "q7", "gripper_open"]
            ),
        },
    )
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(
        f"wrote {len(episodes)} episodes / {index} frames to {out} "
        f"(expert attempted {attempted}, {time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
