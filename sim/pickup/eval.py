"""Closed-loop evaluation of a policy served by `policy_server.py` in the pickup sim.

Evaluation seeds start at 1_000_000 and are disjoint from the demonstration seeds, so every
evaluated layout is unseen. Writes results.json (+ rollout videos) to --out.

  python -m sim.pickup.eval --task pick_cube --episodes 50 --port 6200 --out results/pickup/<run>
"""

import argparse
import io
import json
import time
from collections import defaultdict
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
from PIL import Image

from .diagnostics import GraspProbe, summarize
from .env import PickupEnv

AUTHKEY = b"pickup-sim"  # must match policy_server.AUTHKEY (not imported: that module needs torch)
EVAL_SEED_START = 1_000_000


def _jpeg(img, quality=95):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def connect(port, timeout_s=600):
    t0 = time.time()
    while True:
        try:
            return Client(("127.0.0.1", port), authkey=AUTHKEY)
        except (ConnectionRefusedError, OSError):
            if time.time() - t0 > timeout_s:
                raise
            time.sleep(1.0)


def write_video(path, frames, fps=20):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_cube")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed_start", type=int, default=EVAL_SEED_START)
    ap.add_argument("--port", type=int, default=6200)
    ap.add_argument("--exec_horizon", type=int, default=8, help="actions executed per predicted chunk")
    ap.add_argument("--flow_steps", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--videos", type=int, default=5, help="number of rollouts to save as mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shutdown_server", action="store_true")
    ap.add_argument("--expert", action="store_true", help="run the scripted expert instead of a served policy")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = PickupEnv(args.task, max_steps=args.max_steps)
    expert = conn = None
    if args.expert:
        from .expert import ScriptedExpert

        expert = ScriptedExpert(env, seed=0)
        server_info = dict(policy="scripted expert")
    else:
        conn = connect(args.port)
        conn.send({"cmd": "info"})
        server_info = conn.recv()
    cams = env.camera_names

    episodes, latencies = [], []
    per_target = defaultdict(lambda: [0, 0])
    t_start = time.time()
    for ep in range(args.episodes):
        seed = args.seed_start + ep
        obs = env.reset(seed=seed)
        prev_obs, queue, frames = obs, [], []
        wrong_object = False  # a non-target object was lifted at any point in the episode
        probe = GraspProbe(env)
        if expert is not None:
            expert.reset()
        done, info = False, {}
        while not done:
            if expert is not None:
                queue = [expert.act()]
            elif not queue:
                images = {}
                for cam in cams:
                    images[f"{cam}_t-1"] = _jpeg(prev_obs[cam])
                    images[f"{cam}_t0"] = _jpeg(obs[cam])
                conn.send(
                    dict(
                        cmd="infer",
                        images=images,
                        state=[prev_obs["state"].tolist(), obs["state"].tolist()],
                        prev_action=env.last_action.tolist(),
                        instruction=env.instruction,
                        num_flow_steps=args.flow_steps,
                    )
                )
                reply = conn.recv()
                latencies.append(reply["latency_s"])
                queue = list(reply["actions"][: args.exec_horizon])
            if ep < args.videos:
                frames.append(np.concatenate([obs["front"], obs["wrist"]], axis=1))
            prev_obs = obs
            action = queue.pop(0)
            probe.update(action)
            obs, done, info = env.step(action)
            wrong_object = wrong_object or bool(info["wrong_object"])
        episodes.append(
            dict(
                seed=seed,
                target=env.target,
                instruction=env.instruction,
                success=bool(info["success"]),
                steps=env.steps,
                wrong_object=wrong_object,
                **probe.result(),
            )
        )
        per_target[env.target][0] += int(info["success"])
        per_target[env.target][1] += 1
        if ep < args.videos:
            write_video(out / f"rollout_{ep:03d}_{env.target}_{'ok' if info['success'] else 'fail'}.mp4", frames)
        done_n = len(episodes)
        print(
            f"[eval] {done_n}/{args.episodes} seed={seed} target={env.target} success={info['success']} "
            f"steps={env.steps}  running={sum(e['success'] for e in episodes) / done_n:.2f}",
            flush=True,
        )

    n = len(episodes)
    k = sum(e["success"] for e in episodes)
    p = k / n
    z = 1.96  # Wilson 95% interval
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    results = dict(
        task=args.task,
        episodes=n,
        successes=k,
        success_rate=p,
        success_ci95=[max(0.0, centre - half), min(1.0, centre + half)],
        mean_steps_success=float(np.mean([e["steps"] for e in episodes if e["success"]])) if k else None,
        wrong_object_rate=sum(e["wrong_object"] for e in episodes) / n,
        per_target={t: dict(successes=s, episodes=m, success_rate=s / m) for t, (s, m) in sorted(per_target.items())},
        grasp=summarize(episodes),
        inference_latency_ms=dict(
            mean=1000 * float(np.mean(latencies)),
            p50=1000 * float(np.percentile(latencies, 50)),
            p95=1000 * float(np.percentile(latencies, 95)),
        )
        if latencies
        else None,
        exec_horizon=args.exec_horizon,
        flow_steps=args.flow_steps,
        seed_start=args.seed_start,
        wall_time_s=time.time() - t_start,
        server=server_info,
        per_episode=episodes,
    )
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    g = results["grasp"]
    print(
        f"[eval] {args.task}: {k}/{n} = {p:.1%} "
        f"(95% CI {results['success_ci95'][0]:.1%}-{results['success_ci95'][1]:.1%}) -> {out / 'results.json'}"
    )
    if g["median_xy_err_target_mm"] is not None:
        print(
            f"[eval] gripper closed in {g['closed_gripper_rate']:.0%} of episodes, "
            f"nearest object was the named one in {g['closed_nearest_is_target_rate']:.0%}; "
            f"median miss {g['median_xy_err_target_mm']:.0f} mm to the named object, "
            f"{g['median_xy_err_nearest_mm']:.0f} mm to the nearest object"
        )
    if conn is not None:
        if args.shutdown_server:
            conn.send({"cmd": "shutdown"})
            conn.recv()
        conn.close()
    env.close()


if __name__ == "__main__":
    main()
