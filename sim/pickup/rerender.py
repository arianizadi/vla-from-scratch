"""Paired re-rendering: the demonstrations of a dataset written by collect.py, shown again in other looks.

Every episode is played again in the simulator, which reproduces its states and actions bit for bit (checked at every
frame), and each frame is rendered a second time in a copy of the scene that a visual scenario changes (scenarios.py:
lighting, camera, bench, distractors).

Playing an episode again needs the simulator in the state collect.py had it in: an environment that has run other
episodes is not quite a fresh one (a shoot that a plant does not use keeps the shape and the leaves of the last
episode that used it, parked on the floor, and the floor contacts and the solver differ a little), and from there the
physics drifts apart within a few hundred steps. So the collection is re-run as collect.py ran it: the same rounds of
seeds, split among the same number of workers (--collect_workers), each worker a fresh environment and expert running
its seeds in order, the failed ones included; the episodes the dataset kept are checked against it frame by frame.

The new episodes keep the recorded states and actions unchanged; only the camera images differ. A trajectory then
appears under several looks with exactly the same actions, which teaches what does not matter without changing what to
do (with live collection under a scenario, the expert's trajectories drift apart from the plain ones and the data cannot
say whether the look or the motion changed).

One look per copy of an episode, drawn once and kept for the whole episode (both cameras, every frame): copy k of
source episode i gets look LOOKS[(i + k) mod len(LOOKS)]. A look's distractors are solid objects in the arm's world;
since a replayed trajectory cannot react to them, a placement that any part of the arm or the plant touches during the
episode is drawn again (up to RETRIES times, else the copy takes the next look).

  python -m sim.pickup.rerender --src data/lerobot/prune_2k_rs_tail10_leaffix --out data/lerobot/<name> \\
      --copies 2 --include_source --collect_workers 32 --workers 32

The look "plain" (no scenario) with --image_size draws the demonstrations again at another resolution, for an encoder
that reads finer detail (--copies 1 --looks plain --image_size 448).
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .collect import CAMERAS, _included, _jpeg
from .env import CONTROL_HZ, PickupEnv
from .expert import ScriptedExpert, random_start

# The training halves of the visual scenarios, one setting per episode: flickering light (which changes from one
# frame to the next) is left out.
LOOKS = ("lighting:dusk|noon/train", "camera/train", "bench/train", "distractors/train")
RETRIES = 8
PLAIN = "plain"  # a look of its own: no scenario (with --image_size, the plain scene at another resolution)
UNSYNCED = {"table_visual"}  # geoms a look itself sets (the bench hides the table's own top)


class Replayer:
    """Replays the episodes of one source dataset (its task, leaf drawing and start) and renders them in looks."""

    def __init__(self, info, image_size=224):
        self.task, self.leaf_fix = info["sim_task"], bool(info.get("leaf_fix"))
        self.image_size = image_size
        self.park_reset = bool(info.get("park_reset"))
        self.random_start = bool(info.get("random_start"))
        self.env = PickupEnv(self.task, leaf_fix=self.leaf_fix, park_reset=self.park_reset)
        self.expert = ScriptedExpert(self.env, seed=0)  # for random_start's inverse kinematics only
        self.looks: dict[str, PickupEnv] = {}

    def trajectory(self, seed, states, actions):
        """Replay; returns the simulator's state at every frame (what a look needs to draw it). Raises if a recorded
        state is not reproduced exactly."""
        env = self.env
        env.reset(seed=seed)
        if self.random_start:
            random_start(env, self.expert, seed)
        frames = []
        for t, action in enumerate(actions):
            if not np.array_equal(env.state(), states[t]):
                raise RuntimeError(f"seed {seed}: the replay left the recorded states at frame {t}")
            frames.append(self.frame())
            env.step(action)
        return frames

    def frame(self):
        """What a look needs to draw the simulator's present state."""
        env, m = self.env, self.env.model
        return dict(
            qpos=env.data.qpos.copy(),
            mocap_pos=env.data.mocap_pos.copy(),
            mocap_quat=env.data.mocap_quat.copy(),
            body_pos=m.body_pos.copy(),
            body_quat=m.body_quat.copy(),
            geom=[a.copy() for a in (m.geom_pos, m.geom_quat, m.geom_size, m.geom_rgba, m.geom_group)],
            geom_dataid=m.geom_dataid.copy(),
        )

    def collect_chunk(self, worker_id, seeds, tail_steps):
        """Run the expert on `seeds` in order as one collect.py worker did (a fresh environment and expert per chunk);
        yields (seed, states, actions, frames) for every successful episode."""
        env = self.env = PickupEnv(self.task, leaf_fix=self.leaf_fix, park_reset=self.park_reset)
        expert = ScriptedExpert(env, seed=worker_id)
        self.expert = expert
        budget = env.max_steps
        for seed in seeds:
            env.max_steps = budget
            obs, yaw = env.reset(seed=seed), 0.0
            if self.random_start:
                obs, yaw = random_start(env, expert, seed)
            expert.reset(yaw=yaw)
            states, actions, frames, done, info = [], [], [], False, {}
            while not done:
                action = expert.act()
                states.append(obs["state"])
                actions.append(action)
                frames.append(self.frame())
                obs, done, info = env.step(action)
            if not info["success"]:
                continue
            for _ in range(tail_steps):
                action = env.last_action.copy()
                states.append(obs["state"])
                actions.append(action)
                frames.append(self.frame())
                obs, _, _ = env.step(action)
            yield seed, np.array(states), np.array(actions), frames

    def look_env(self, spec):
        if spec not in self.looks:
            scenario = None if spec == PLAIN else spec
            look = PickupEnv(
                self.task,
                scenario=scenario,
                leaf_fix=self.leaf_fix,
                park_reset=self.park_reset,
                image_size=self.image_size,
            )
            src, dst = self.env.model, look.model  # (every PickupEnv of the task has the same model: same ids)
            # the same bodies, joints and geoms by name (a look's model has more of them, so indices differ)
            bodies = [(b, dst.body(src.body(b).name).id) for b in range(src.nbody)]
            look._sync_qpos = [
                (src.jnt_qposadr[src.body_jntadr[b]], dst.jnt_qposadr[dst.body_jntadr[d]], self._qpos_size(src, b))
                for b, d in bodies
                if src.body_jntnum[b]
            ]
            look._sync_bodies = np.array(bodies)
            named = [g for g in range(src.ngeom) if src.geom(g).name and src.geom(g).name not in UNSYNCED]
            look._sync_geoms = np.array([(g, dst.geom(src.geom(g).name).id) for g in named])
            look._sync_mocap = np.array(
                [(src.body_mocapid[b], dst.body_mocapid[d]) for b, d in bodies if src.body_mocapid[b] >= 0]
            )
            look._own = self._own_bodies(look)
            self.looks[spec] = look
        return self.looks[spec]

    @staticmethod
    def _qpos_size(m, b):
        first, n = m.body_jntadr[b], m.body_jntnum[b]
        end = m.jnt_qposadr[first + n] if first + n < m.njnt else m.nq
        return int(end - m.jnt_qposadr[first])

    @staticmethod
    def _own_bodies(look):
        """Bodies a look adds that can touch something (its distractors)."""
        m = look.model
        return {b for b in range(m.nbody) if m.body(b).name.startswith("sit_")}

    def draw(self, look, frame):
        """Put the look's scene in the state of a replayed frame."""
        m, d = look.model, look.data
        for a, b, n in look._sync_qpos:
            d.qpos[b : b + n] = frame["qpos"][a : a + n]
        src, dst = look._sync_bodies[:, 0], look._sync_bodies[:, 1]
        m.body_pos[dst], m.body_quat[dst] = frame["body_pos"][src], frame["body_quat"][src]
        gs, gd = look._sync_geoms[:, 0], look._sync_geoms[:, 1]
        for field, values in zip(
            (m.geom_pos, m.geom_quat, m.geom_size, m.geom_rgba, m.geom_group), frame["geom"], strict=True
        ):
            field[gd] = values[gs]
        m.geom_dataid[gd] = frame["geom_dataid"][gs]
        if len(look._sync_mocap):
            ms, md = look._sync_mocap[:, 0], look._sync_mocap[:, 1]
            d.mocap_pos[md], d.mocap_quat[md] = frame["mocap_pos"][ms], frame["mocap_quat"][ms]
        mujoco.mj_forward(m, d)

    def touched(self, look):
        """Does anything but the table or the floor touch one of the look's own objects? (The distractors a look does
        not use are parked on the floor; counting that contact turned every placement down.)"""
        if not look._own:
            return False
        m, d = look.model, look.data
        ground = (m.geom("table").id, m.geom("floor").id)
        for i in range(d.ncon):
            g1, g2 = d.contact.geom1[i], d.contact.geom2[i]
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if (
                (b1 in look._own) != (b2 in look._own)
                and g1 not in ground
                and g2 not in ground
                and d.contact.dist[i] < 0
            ):
                return True
        return False

    def render(self, seed, scenario_seed, spec, frames):
        """The frames in one look, or None if the look's objects are in the way of the trajectory."""
        look = self.look_env(spec)
        look.reset(seed=seed, scenario_seed=scenario_seed)
        variation = dict(look.scenario_info)
        if look._own:
            for frame in frames:
                self.draw(look, frame)
                if self.touched(look):
                    return None, variation
        images = {c: [] for c in CAMERAS}
        for frame in frames:
            self.draw(look, frame)
            obs = look.get_obs()
            for c in CAMERAS:
                images[c].append(_jpeg(obs[c]))
        return images, variation


def collection_rounds(entries, info, workers):
    """The chunks of seeds collect.py's workers ran, round by round: [(worker id, [seeds])]. Every success of a round
    but the last was kept (the dataset holds the first `episodes` successes by seed)."""
    kept = sorted(e["seed"] for e in entries)
    target, attempted = len(kept), int(info["expert_attempted"])
    assert info.get("targets_per_layout", 1) == 1 and not info.get("takeover") and not info.get("disturb_prob")
    chunks, start, done_kept = [], 0, 0
    while start < attempted:
        need = target - done_kept
        seeds = list(range(start, start + max(need + need // 10, workers)))
        start = seeds[-1] + 1
        chunks += [(i, seeds[i::workers]) for i in range(workers)]
        done_kept = sum(s < start for s in kept)
    if start != attempted:
        raise ValueError(
            f"{workers} workers give {start} attempts, the dataset says {attempted}: wrong --collect_workers?"
        )
    return chunks


_REPLAYER = None


def _render_copies(index, entry, seed, frames, table, out, copies, looks):
    results = []
    for k in range(copies):
        images = None
        for shift in range(len(looks)):
            spec = looks[(index + k + shift) % len(looks)]
            for attempt in range(RETRIES):
                scenario_seed = 10_000_000 + 1_000 * seed + 100 * k + attempt
                images, variation = _REPLAYER.render(seed, scenario_seed, spec, frames)
                if images is not None:
                    break
            if images is not None:
                break
        if images is None:
            raise RuntimeError(f"seed {seed}: no look could be drawn clear of the trajectory")
        path = Path(out) / "tmp" / f"src{index:06d}_copy{k}.parquet"
        pq.write_table(
            pa.table(
                {
                    **{c: pa.array([{"bytes": b, "path": None} for b in images[c]]) for c in CAMERAS},
                    "state": table["state"],  # the recorded columns themselves: bit-identical
                    "action": table["action"],
                }
            ),
            path,
        )
        results.append(
            dict(
                seed=seed,
                instruction=entry["tasks"][0],
                path=str(path),
                rerender_of=index,
                look=spec,
                scenario_seed=scenario_seed,
                scenario=variation,
                attempts=attempt + 1,
            )
        )
    return results


def _chunk_job(args):
    src, out, info, worker_id, seeds, by_seed, copies, looks, image_size = args
    global _REPLAYER
    _REPLAYER = Replayer(info, image_size)
    results = []
    for seed, states, actions, frames in _REPLAYER.collect_chunk(worker_id, seeds, int(info.get("tail_steps", 0))):
        if seed not in by_seed:
            continue  # a success past the dataset's last episode (the last round makes more than it needs)
        entry = by_seed[seed]
        index = entry["episode_index"]
        table = pq.read_table(
            Path(src) / "data" / "chunk-000" / f"episode_{index:06d}.parquet", columns=["state", "action"]
        )
        recorded_s = np.array(table["state"].to_pylist(), dtype=np.float32)
        recorded_a = np.array(table["action"].to_pylist(), dtype=np.float32)
        if not (np.array_equal(states, recorded_s) and np.array_equal(actions.astype(np.float32), recorded_a)):
            raise RuntimeError(f"seed {seed}: the re-run collection does not reproduce the recorded episode")
        results += _render_copies(index, entry, seed, frames, table, out, copies, looks)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a dataset written by collect.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--copies", type=int, default=2, help="re-rendered copies of every source episode")
    ap.add_argument("--looks", nargs="+", default=list(LOOKS), help="scenario specs, one per copy (in rotation)")
    ap.add_argument("--include_source", action="store_true", help="the source episodes come first, unchanged")
    ap.add_argument("--episodes", type=int, default=None, help="only the first N source episodes (for checks)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--collect_workers",
        type=int,
        required=True,
        help="the --workers the source dataset was collected with (the episodes are re-run in the same chunks)",
    )
    ap.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="render the looks at this many pixels (square); with --looks plain, the plain scene at a new size",
    )
    args = ap.parse_args()
    if args.include_source and args.image_size != 224:
        ap.error("--include_source keeps the source's 224-pixel images: it cannot be mixed with another --image_size")

    src, out = Path(args.src), Path(args.out)
    info = json.loads((src / "meta" / "info.json").read_text())
    assert not info.get("scenario"), "re-render a dataset collected without scenarios"
    assert not info.get("action_noise"), "re-render a dataset collected without --action_noise (its replay skips it)"
    all_entries = [json.loads(line) for line in (src / "meta" / "episodes.jsonl").read_text().splitlines()]
    entries = all_entries[: args.episodes] if args.episodes else all_entries
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    by_seed = {e["seed"]: e for e in entries}
    chunks = collection_rounds(all_entries, info, args.collect_workers)
    jobs = [
        (str(src), str(out), info, w, seeds, by_seed, args.copies, args.looks, args.image_size) for w, seeds in chunks
    ]
    if args.episodes:  # a check on part of the dataset: only the chunks that hold the episodes asked for
        jobs = [j for j in jobs if any(s in by_seed for s in j[4])]
    new = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_chunk_job, jobs)):
            new.extend(res)
            print(f"[rerender] {i + 1}/{len(jobs)} chunks, {len(new)} copies, {time.time() - t0:.0f}s", flush=True)
    new.sort(key=lambda r: (r["rerender_of"], r["path"]))
    included = _included(src, info["sim_task"]) if args.include_source else []
    if args.episodes and included:
        included = included[: args.episodes]
    kept = included + new

    tasks, episodes, index = {}, [], 0
    for ep_idx, r in enumerate(kept):
        task_idx = tasks.setdefault(r["instruction"], len(tasks))
        table = pq.read_table(r["path"], columns=[*CAMERAS, "state", "action"])
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
                **(
                    dict(look="source", source_episode=r["episode_index"])
                    if "rerender_of" not in r
                    else {k: r[k] for k in ("look", "scenario_seed", "scenario", "attempts")}
                    | {"source_episode": r["rerender_of"]}
                ),
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
    counts = {}
    for e in episodes:
        counts[e["look"]] = counts.get(e["look"], 0) + 1
    features = {
        k: dict(v, shape=[args.image_size, args.image_size, *v["shape"][2:]]) if k in CAMERAS else v
        for k, v in info.get("features", {}).items()
    }
    meta = dict(
        info,
        **(dict(features=features) if features else {}),
        total_episodes=len(episodes),
        total_frames=index,
        total_tasks=len(tasks),
        rerender=dict(
            source=src.name,
            copies=args.copies,
            looks=args.looks,
            source_included=bool(args.include_source),
            image_size=args.image_size,
            episodes_by_look=counts,
            redrawn=sum(r["attempts"] > 1 for r in new),
        ),
    )
    (out / "meta" / "info.json").write_text(json.dumps(meta, indent=2))
    print(f"[rerender] wrote {len(episodes)} episodes / {index} frames to {out} ({time.time() - t0:.0f}s): {counts}")


if __name__ == "__main__":
    main()
