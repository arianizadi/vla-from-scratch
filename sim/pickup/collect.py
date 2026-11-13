"""Collect scripted-expert demonstrations and write them as a LeRobot v2.0 dataset.

The layout (meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl, data/chunk-000/episode_*.parquet
with inline JPEG image columns) is what vla_foundry's `--type lerobot` preprocessing converter reads.

Per-frame columns: front, wrist (JPEG bytes), state [8], action [8], plus LeRobot bookkeeping.
action[t] is the command applied after observing frame t.

  python -m sim.pickup.collect --task pick_cube --episodes 200 --out data/lerobot/pick_cube

With --policy_ports, a trained policy (served by policy_server.py, one server per worker) drives the first K steps
of every episode and the scripted expert takes over from wherever the arm is and finishes the task; only the
expert's part is recorded (scripts/make_takeover_dataset.sh starts the servers).
"""

import argparse
import io
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .disturb import Disturbance
from .env import CONTROL_HZ, TASKS, PickupEnv
from .eval import connect, infer_request
from .expert import ScriptedExpert, random_start

CAMERAS = ("front", "wrist")


def _jpeg(img, quality=95):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def hand_over(env, expert, obs, policy, steps, budget):
    """Let `policy` drive the first `steps` control steps of an episode, then hand the arm to `expert` where it is
    (ScriptedExpert.take_over), with `budget` more steps to finish in. `policy(prev_obs, obs)` returns the next
    actions to execute. Returns (observation at the handover, None), or (None, info) if the episode ended first."""
    prev_obs, queue = obs, []
    for _ in range(steps):
        if not queue:
            queue = list(policy(prev_obs, obs))
        prev_obs = obs
        obs, done, info = env.step(queue.pop(0))
        if done:
            return None, info
    env.max_steps = env.steps + budget
    expert.take_over()
    return obs, None


def served_policy(conn, env, exec_horizon=8, flow_steps=10):
    """The policy behind a policy_server.py connection, queried as eval.py does (fresh noise for every chunk)."""

    def policy(prev_obs, obs):
        conn.send(infer_request(prev_obs, obs, env.last_action, env.instruction, flow_steps))
        return conn.recv()["actions"][:exec_horizon]

    return policy


def _worker(args):
    (
        task,
        jobs,
        out_dir,
        worker_id,
        disturb_prob,
        anywhere,
        hold_steps,
        drift_prob,
        drift_cm_s,
        side,
        tail_steps,
        takeover,
    ) = args  # takeover: None, or this worker's policy server and the range the handover step K is drawn from
    env = PickupEnv(task)
    expert = ScriptedExpert(env, seed=worker_id, reactive=disturb_prob > 0 or drift_prob > 0, side=side)
    budget = env.max_steps
    conn = policy = None
    if takeover is not None:
        conn = connect(takeover["port"])
        policy = served_policy(conn, env, takeover["exec_horizon"], takeover["flow_steps"])
    results = []
    for seed, target_index in jobs:
        env.max_steps = budget
        obs, start_yaw = env.reset(seed=seed, target_index=target_index), 0.0
        if anywhere:
            obs, start_yaw = random_start(env, expert, seed)
        handover = {}
        if takeover is None:
            expert.reset(yaw=start_yaw)
        else:
            lo, hi = takeover["steps"]
            k = int(np.random.default_rng([seed, 17]).integers(lo, hi + 1))  # its own stream, as random_start's
            obs, ended = hand_over(env, expert, obs, policy, k, budget)
            if obs is None:  # the policy finished the episode itself (pruned everything, or damaged the plant)
                results.append(dict(seed=seed, success=False, takeover_step=k, before_handover=_ending(ended)))
                continue
            handover = dict(takeover_step=k, cut_at_handover=len(env.cut))
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
            results.append(dict(seed=seed, success=False, **handover, after_handover=_ending(info)))
            continue
        for _ in range(tail_steps):  # after the success: the robot holds its last command (for a cut, jaws shut)
            action = env.last_action.copy()
            for c in CAMERAS:
                frames[c].append(_jpeg(obs[c]))
            states.append(obs["state"])
            actions.append(action)
            obs, _, _ = env.step(action)
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
                **handover,
            )
        )
    if conn is not None:
        conn.close()
    env.close()
    return results


def _ending(info):
    """How an episode ended: 'success', what was damaged, or 'time' (or 'wrong object' for the pick tasks)."""
    if info["success"]:
        return "success"
    return info.get("damaged") or ("wrong object" if info.get("wrong_object") else "time")


def _included(path, task):
    """The episodes of an earlier dataset written by this script, as entries to copy into the new one."""
    src = Path(path)
    info = json.loads((src / "meta" / "info.json").read_text())
    assert info["sim_task"] == task, f"{src} holds {info['sim_task']} demonstrations, not {task}"
    entries = [json.loads(line) for line in (src / "meta" / "episodes.jsonl").read_text().splitlines()]
    return [
        dict(
            e,
            instruction=e["tasks"][0],
            path=str(src / "data" / "chunk-000" / f"episode_{e['episode_index']:06d}.parquet"),
            source=src.name,
        )
        for e in entries
    ]


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
        "--expert_side",
        choices=["clearance", "preferred"],
        default="clearance",
        help="stems: which side the expert approaches from (see ScriptedExpert)",
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
    ap.add_argument(
        "--tail_steps",
        type=int,
        default=0,
        help="after each success, record this many more frames of the robot holding its last command. A cut "
        "succeeds 3 steps after the jaws reach the stem, so without a tail the shut gripper appears in only the "
        "last few frames of each demonstration, and a policy trained on them opens again before the jaws arrive.",
    )
    ap.add_argument(
        "--policy_ports",
        type=int,
        nargs="+",
        default=None,
        help="takeover demonstrations: one policy_server.py port per worker. A trained policy drives the first K "
        "steps of each episode, then the expert takes over from wherever the arm is and finishes the task. Only "
        "the expert's part is recorded, and only when it succeeds: demonstrations that start from states the policy "
        "reaches, which demonstrations from the expert's own starts rarely visit (DAgger-style data).",
    )
    ap.add_argument(
        "--takeover_steps",
        type=int,
        nargs=2,
        default=(0, 700),
        metavar=("LO", "HI"),
        help="with --policy_ports: K is drawn uniformly from LO..HI (inclusive) for each episode. The expert then "
        "has the task's whole step limit from the handover on.",
    )
    ap.add_argument("--exec_horizon", type=int, default=8, help="with --policy_ports: actions executed per chunk")
    ap.add_argument("--flow_steps", type=int, default=10, help="with --policy_ports: sampler steps per chunk")
    ap.add_argument(
        "--include",
        nargs="+",
        default=[],
        help="datasets written by this script whose episodes come first in the new one, unchanged: new "
        "demonstrations and earlier ones preprocessed together, so that one set of normalisation statistics "
        "covers both",
    )
    args = ap.parse_args()
    num_objects = TASKS[args.task]["num_objects"]
    assert 1 <= args.targets_per_layout <= num_objects, f"{args.task} has {num_objects} objects per layout"
    per_layout = args.targets_per_layout
    if args.policy_ports is not None:
        assert len(args.policy_ports) == args.workers, "--policy_ports needs one port per worker"
    included = [e for path in args.include for e in _included(path, args.task)]

    out = Path(args.out)
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    servers = None
    if args.policy_ports is not None:  # what is being served, for the record (each server takes one client at a time)
        servers = set()
        for port in args.policy_ports:
            conn = connect(port)
            conn.send({"cmd": "info"})
            servers.add(conn.recv()["checkpoint"])
            conn.close()
        assert len(servers) == 1, f"the policy servers serve different checkpoints: {servers}"
    kept, attempted, successes, next_seed, endings = [], 0, 0, args.seed, []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        while len(kept) < args.episodes:
            need = args.episodes - len(kept)
            seeds = list(range(next_seed, next_seed + max(-(-(need + need // 10) // per_layout), args.workers)))
            next_seed = seeds[-1] + 1
            # With one target per layout the seed's own choice is kept (target_index None).
            jobs = [(seed, i if per_layout > 1 else None) for seed in seeds for i in range(per_layout)]
            extra = (
                args.disturb_prob,
                args.random_start,
                args.hold_steps,
                args.drift_prob,
                args.drift_cm_s,
                args.expert_side,
                args.tail_steps,
            )
            chunks = [
                (args.task, jobs[i :: args.workers], str(out), i, *extra, _takeover(args, i))
                for i in range(args.workers)
            ]
            for res in pool.map(_worker, chunks):
                attempted += len(res)
                successes += sum(r["success"] for r in res)
                kept.extend(r for r in res if r["success"])
                endings.extend(r for r in res if not r["success"])
    kept = sorted(kept, key=lambda r: (r["seed"], r["target_index"]))[: args.episodes]
    new = len(kept)
    kept = included + kept

    tasks, episodes, index = {}, [], 0
    for ep_idx, r in enumerate(kept):
        task_idx = tasks.setdefault(r["instruction"], len(tasks))
        table = pq.read_table(r["path"], columns=[*CAMERAS, "state", "action"])  # without an included one's bookkeeping
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
                **{k: r[k] for k in ("takeover_step", "cut_at_handover") if k in r},
                **({"source": r.get("source", out.name)} if args.include else {}),
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
        disturbed_episodes=sum(r["disturbed"] != "none" for r in kept[len(included) :]),
        drift_prob=args.drift_prob,
        drift_cm_s=args.drift_cm_s if args.drift_prob > 0 else None,
        drifting_episodes=sum(r["disturbed"] == "drift" for r in kept[len(included) :]),
        expert_side=args.expert_side,
        tail_steps=args.tail_steps,
        takeover=_takeover_summary(args, servers, attempted, kept[len(included) :], endings),
        included=[
            dict(dataset=Path(p).name, episodes=sum(e["source"] == Path(p).name for e in included))
            for p in args.include
        ],
        new_episodes=new,
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
        f"({len(included)} included; expert attempted {attempted}, {time.time() - t0:.0f}s)"
    )


def _takeover(args, worker):
    if args.policy_ports is None:
        return None
    return dict(
        port=args.policy_ports[worker],
        steps=tuple(args.takeover_steps),
        exec_horizon=args.exec_horizon,
        flow_steps=args.flow_steps,
    )


def _takeover_summary(args, servers, attempted, kept, failed):
    """Takeover collection: what was served, how the attempts ended, and at which step the kept ones were handed
    over (counts per 100 steps) with how many shoots or objects were already cut."""
    if servers is None:
        return None
    lo, hi = args.takeover_steps
    steps = np.array([r["takeover_step"] for r in kept])
    return dict(
        policy=sorted(servers)[0],
        takeover_steps=[lo, hi],
        exec_horizon=args.exec_horizon,
        flow_steps=args.flow_steps,
        attempts=attempted,
        ended_before_handover=dict(
            sorted(Counter(r["before_handover"] for r in failed if "before_handover" in r).items())
        ),
        expert_failed_after_handover=dict(
            sorted(Counter(r["after_handover"] for r in failed if "after_handover" in r).items())
        ),
        kept_by_step={f"{a}-{a + 99}": int(((steps >= a) & (steps < a + 100)).sum()) for a in range(lo, hi + 1, 100)},
        kept_cut_at_handover=dict(sorted(Counter(r["cut_at_handover"] for r in kept).items())),
    )


if __name__ == "__main__":
    main()
