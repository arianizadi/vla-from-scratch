"""Closed-loop evaluation of a policy served by `policy_server.py` in the pickup sim.

Evaluation seeds start at 1_000_000 and are disjoint from the demonstration seeds, so every
evaluated layout is unseen. Writes results.json (+ rollout videos) to --out.

  python -m sim.pickup.eval --task pick_cube --episodes 50 --port 6200 --out results/pickup/<run>
"""

import argparse
import io
import json
import re
import sys
import time
from collections import defaultdict
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
from PIL import Image

from .diagnostics import GraspProbe, summarize
from .disturb import Disturbance
from .env import BUTTONS, CONTROL_HZ, PRESS_TRAVEL, REACH_AZIMUTH, PickupEnv

AUTHKEY = b"pickup-sim"  # must match policy_server.AUTHKEY (not imported: that module needs torch)
EVAL_SEED_START = 1_000_000


def write_json(path, data, **kwargs):
    """Write through a temporary file, so that a monitor reading `path` never sees half of it."""
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w") as f:
        json.dump(data, f, **kwargs)
    partial.replace(path)


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


def trace_step(env, obs, action):
    """One control step of a rollout as plain numbers, recorded before `action` is applied."""

    def r(values):
        return [round(float(v), 4) for v in values]

    return dict(
        tcp=r(env.tcp_pos()),
        yaw=round(env.tcp_yaw(), 4),
        grip=round(float(obs["state"][7]), 4),
        q=r(obs["state"][:7]),
        action=r(action),
        objects={n: r([*env.object_pos(n), env.object_yaw(n)]) for n in env.active},
    )


def check_server(info, port, run, number, weights):
    """Stop if the server on `port` is not serving the checkpoint this evaluation is for.

    Two evaluations started on the same port both reach whichever server bound it first. Without this
    check the second one silently evaluates the other run's policy and files the result under its own
    name.
    """
    served = str(info.get("checkpoint", "")).replace("\\", "/")
    parts = served.split("/")
    m = re.fullmatch(r"(ema|checkpoint)_(\d+)\.pt", parts[-1])
    problems = []
    if run is not None and (len(parts) < 3 or parts[-3] != run):
        problems.append(f"run '{run}'")
    if number is not None and (m is None or int(m.group(2)) != number):
        problems.append(f"checkpoint {number}")
    if weights is not None and (m is None or (m.group(1) == "ema") != (weights == "ema")):
        problems.append(f"{weights} weights")
    if problems:
        sys.exit(
            f"[eval] the policy server on port {port} is serving {served}, not {', '.join(problems)}. "
            "Another evaluation is using this port; give each one its own (eval.sh picks a free port "
            "unless PICKUP_PORT is set)."
        )


def _wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _turning_summary(episodes):
    """Pruning: how efficiently the plant was turned. For one-shoot tasks, `needed` is the turn that brings the shoot
    to point at the robot the short way; a first press is `short_way` when it turns that way."""
    eps = [e for e in episodes if e.get("turned_deg") is not None]
    if not eps:
        return None
    pressed = [e for e in eps if e["first_press"] is not None and abs(e["start_offset_deg"]) > 25]
    ok = [e for e in eps if e["success"]]
    return dict(
        mean_turned_deg_success=float(np.mean([e["turned_deg"] for e in ok])) if ok else None,
        mean_needed_deg_success=float(np.mean([abs(e["start_offset_deg"]) for e in ok])) if ok else None,
        first_press_short_way=sum(e["first_press"] == np.sign(e["start_offset_deg"]) for e in pressed) / len(pressed)
        if pressed
        else None,
        pressed_when_out_of_reach=len(pressed) / max(1, sum(abs(e["start_offset_deg"]) > 25 for e in eps)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_cube")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed_start", type=int, default=EVAL_SEED_START)
    ap.add_argument("--port", type=int, default=6200)
    ap.add_argument("--exec_horizon", type=int, default=8, help="actions executed per predicted chunk")
    ap.add_argument("--flow_steps", type=int, default=10)
    ap.add_argument(
        "--noise",
        choices=["fresh", "episode"],
        default="fresh",
        help="noise the sampler starts each chunk from: fresh every time, or the same for a whole episode",
    )
    ap.add_argument("--max_steps", type=int, default=None, help="episode length limit (default: the task's own)")
    ap.add_argument("--videos", type=int, default=5, help="number of rollouts to save as mp4")
    ap.add_argument(
        "--traces",
        type=int,
        default=0,
        help="number of rollouts to save as JSON (per control step: joint angles, gripper, gripper position, "
        "object poses), for replays and plots",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--shutdown_server", action="store_true")
    ap.add_argument("--expert", action="store_true", help="run the scripted expert instead of a served policy")
    ap.add_argument("--expect_run", default=None, help="stop unless the server serves a checkpoint of this run")
    ap.add_argument("--expect_number", type=int, default=None, help="... and this checkpoint number")
    ap.add_argument("--expect_weights", choices=["raw", "ema"], default=None, help="... and this kind of weights")
    # Interventions for finding out what a policy's choices depend on.
    ap.add_argument("--expert_prefix", type=int, default=0, help="the scripted expert drives the first K steps")
    ap.add_argument(
        "--prefix_target",
        choices=["named", "distractor"],
        default="named",
        help="which object the expert prefix heads for: the named one, or another object in the layout",
    )
    ap.add_argument("--stale_images", action="store_true", help="keep sending the first frame's images")
    ap.add_argument(
        "--first_prev_action",
        choices=["hold", "expert"],
        default="hold",
        help="previous action sent with the first query: the hold command the robot is really under, or the "
        "expert's first command (what a training window padded by copying contains)",
    )
    # Disturbances: how does the policy cope when the scene changes under it? (disturb.py)
    ap.add_argument(
        "--disturb",
        choices=["none", "reach", "grasp", "drift"],
        default="none",
        help="move the named object once per episode: 'reach' at --disturb_step, while the gripper is on its way; "
        "'grasp' at the moment the gripper first closes, so that succeeding takes a second attempt; 'drift': the "
        "object keeps sliding on a smooth random path at --drift_cm_s until it is held",
    )
    ap.add_argument("--disturb_step", type=int, default=15)
    ap.add_argument("--disturb_mm", type=float, default=80.0, help="how far the object is moved")
    ap.add_argument(
        "--drift_cm_s",
        type=float,
        default=2.0,
        help="with --disturb drift: how fast the named object slides (cm/s), from the first step until it is held",
    )
    ap.add_argument(
        "--expert_side",
        choices=["clearance", "preferred"],
        default="clearance",
        help="with --expert, stems: which side the expert approaches from (see ScriptedExpert)",
    )
    ap.add_argument(
        "--expert_reactive",
        action="store_true",
        help="with --expert: the expert plans again when the object has moved or a grasp closed on nothing",
    )
    ap.add_argument(
        "--random_start",
        action="store_true",
        help="start every episode from a random gripper pose above the table instead of the home pose",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = PickupEnv(args.task, max_steps=args.max_steps)
    expert = conn = None
    if args.expert:
        from .expert import ScriptedExpert

        expert = ScriptedExpert(env, seed=0, reactive=args.expert_reactive, side=args.expert_side)
        server_info = dict(policy="scripted expert", reactive=args.expert_reactive, side=args.expert_side)
    else:
        conn = connect(args.port)
        conn.send({"cmd": "info"})
        server_info = conn.recv()
        check_server(server_info, args.port, args.expect_run, args.expect_number, args.expect_weights)
    pilot = None
    if expert is None and (args.expert_prefix > 0 or args.first_prev_action == "expert"):
        from .expert import ScriptedExpert

        pilot = ScriptedExpert(env, seed=0)
    starter = None
    if args.random_start:
        from .expert import ScriptedExpert, random_start

        starter = ScriptedExpert(env, seed=0)  # for its inverse kinematics only
    cams = env.camera_names

    episodes, latencies = [], []
    per_target = defaultdict(lambda: [0, 0])
    t_start = time.time()
    for ep in range(args.episodes):
        seed = args.seed_start + ep
        obs, start_yaw = env.reset(seed=seed), 0.0
        if starter is not None:
            obs, start_yaw = random_start(env, starter, seed)
        prev_obs, queue, frames, trace = obs, [], [], []
        wrong_object = False  # a non-target object was lifted at any point in the episode
        caught = False  # the named object was lifted (stems: held) at some step, if only briefly
        probe = GraspProbe(env)
        disturbance = Disturbance(
            env, args.disturb, args.disturb_step, args.disturb_mm / 1000, seed=seed, speed=args.drift_cm_s / 100
        )
        if expert is not None:
            expert.reset(yaw=start_yaw)
        prefix_goal = None
        if pilot is not None:  # plan the prefix toward the chosen object; the instruction is unchanged
            named = env.target
            if args.prefix_target == "distractor":
                env.target = next(n for n in env.active if n != named)
            prefix_goal = env.target
            pilot.reset(yaw=start_yaw)
            env.target = named
        first_obs, first_query = obs, True
        done, info = False, {}
        # pruning efficiency: how far the plate turned, against how far the first shoot had to turn into reach
        turn = None
        if env.plant_task:
            off0 = float(_wrap_pi(REACH_AZIMUTH - env.shoot_azimuth(env.active[0])))
            turn = dict(plate=env.plate_angle(), turned=0.0, offset0=off0, first_press=None)
        while not done:
            if expert is not None:
                queue = [expert.act()]
            elif pilot is not None and env.steps < args.expert_prefix:
                queue = [pilot.act()]
            elif not queue:
                shown_prev, shown = (first_obs, first_obs) if args.stale_images else (prev_obs, obs)
                images = {}
                for cam in cams:
                    images[f"{cam}_t-1"] = _jpeg(shown_prev[cam])
                    images[f"{cam}_t0"] = _jpeg(shown[cam])
                prev_action = env.last_action
                if first_query and args.first_prev_action == "expert":
                    prev_action = pilot.act()
                first_query = False
                conn.send(
                    dict(
                        cmd="infer",
                        images=images,
                        state=[prev_obs["state"].tolist(), obs["state"].tolist()],
                        prev_action=np.asarray(prev_action).tolist(),
                        instruction=env.instruction,
                        num_flow_steps=args.flow_steps,
                        noise_seed=seed if args.noise == "episode" else None,
                    )
                )
                reply = conn.recv()
                latencies.append(reply["latency_s"])
                queue = list(reply["actions"][: args.exec_horizon])
            if ep < args.videos:
                frames.append(np.concatenate([obs["front"], obs["wrist"]], axis=1))
            prev_obs = obs
            action = queue.pop(0)
            if ep < args.traces:
                trace.append(trace_step(env, obs, action))
            probe.update(action)  # before the disturbance: the first close is judged on the scene it was aimed at
            disturbance.update(action)
            obs, done, info = env.step(action)
            if turn is not None:
                angle = env.plate_angle()
                turn["turned"] += abs(angle - turn["plate"])
                turn["plate"] = angle
                if turn["first_press"] is None:
                    down = [b for b in BUTTONS if env.button_travel(b) > PRESS_TRAVEL]
                    turn["first_press"] = BUTTONS[down[0]][1] if down else None
            wrong_object = wrong_object or bool(info["wrong_object"])
            caught = caught or env.target in info["lifted"]
        episodes.append(
            dict(
                seed=seed,
                target=env.target,
                instruction=env.instruction,
                success=bool(info["success"]),
                steps=env.steps,
                wrong_object=wrong_object,
                damaged=info.get("damaged"),  # pruning: what was damaged, if anything
                # pruning: degrees the plate turned; how far the first shoot pointed from the robot at the start; the
                # turning direction of the first button press (+1 counter-clockwise, which brings a shoot with a
                # positive offset toward the robot the short way)
                turned_deg=round(float(np.degrees(turn["turned"])), 1) if turn else None,
                start_offset_deg=round(float(np.degrees(turn["offset0"])), 1) if turn else None,
                first_press=turn["first_press"] if turn else None,
                caught=caught,
                prefix_goal=prefix_goal,
                disturbed=disturbance.record,
                **probe.result(),
            )
        )
        per_target[env.target][0] += int(info["success"])
        per_target[env.target][1] += 1
        if ep < args.videos:
            write_video(out / f"rollout_{ep:03d}_{env.target}_{'ok' if info['success'] else 'fail'}.mp4", frames)
        if ep < args.traces:
            write_json(
                out / f"trace_{ep:03d}_{env.target}_{'ok' if info['success'] else 'fail'}.json",
                dict(
                    seed=seed,
                    target=env.target,
                    instruction=env.instruction,
                    objects=list(env.active),
                    control_hz=CONTROL_HZ,
                    success=bool(info["success"]),
                    close_step=episodes[-1]["close_step"],
                    disturbed=disturbance.record,
                    steps=trace,
                ),
            )
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
        turning=_turning_summary(episodes),
        caught_rate=sum(e["caught"] for e in episodes) / n,
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
        noise=args.noise,
        random_start=args.random_start,
        interventions=dict(
            expert_prefix=args.expert_prefix,
            prefix_target=args.prefix_target if args.expert_prefix else None,
            stale_images=args.stale_images,
            first_prev_action=args.first_prev_action,
            disturb=args.disturb,
            disturb_step=args.disturb_step if args.disturb == "reach" else None,
            disturb_mm=args.disturb_mm if args.disturb in ("reach", "grasp") else None,
            drift_cm_s=args.drift_cm_s if args.disturb == "drift" else None,
            disturbed_rate=sum(bool(e["disturbed"] and e["disturbed"]["to_xy"]) for e in episodes) / n,
        ),
        seed_start=args.seed_start,
        wall_time_s=time.time() - t_start,
        server=server_info,
        per_episode=episodes,
    )
    write_json(out / "results.json", results, indent=2)
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
