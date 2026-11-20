"""Where the fast level's cut is set (tutorial/lab-notebook.md, "The cut, diagnosed"). Runs the simulator and, where
asked, the fast level in one process, on `prune` layouts from clear starts, with the scripted expert choosing the shoot
and turning the plate, and measures each cut as eval.py's cut_places does: the gripper's centre from the shoot's axis
as grown, across it, at the control step that cut it, and that offset in the hand's frame (along the closing jaws, the
way the hand points, and the third axis).

  --mode expert     the scripted expert alone: how far its cuts land from the axis it aims at, and how they vary
  --mode handover   the expert drives until its open jaws are within --handover_cm of the middle of the cut zone of the
                    shoot it is going for, then the fast level, given that shoot's sub-goal, drives until the shoot is
                    cut (one cut per episode, the first). --handover_cm 100 hands over as soon as the shoot is chosen
                    and in reach. Inference settings: --exec_horizon, --flow_steps, --samples (several chunks drawn
                    from one look and averaged)
  --mode shadow     the expert drives; at every control step of each approach within --shadow_cm of the cut zone the
                    fast level is asked for a chunk from that state and its chunk is compared with the expert's next
                    commands (gripper positions from forward kinematics), by distance to the cut zone

  python -m sim.pickup.cut_diag --mode handover --checkpoint_dir experiments/pickup/<run> --episodes 100 --out x.json
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from . import reasoning
from .env import CUT_MID, PickupEnv
from .eval import infer_request

VALIDATION_SEED = 900000
CROSS_MM = (100, 80, 60, 40, 20)  # architecture v3's validation layouts (e13.sh)


def cut_offset(env, name) -> dict:
    """The gripper's centre against shoot `name`'s axis as grown: across it, along it, and the offset from the
    nearest point of the axis to the gripper's centre in the hand's frame (x; y, the way the jaws close; z, the way
    the hand points), in mm."""
    a, d = env.shoot_axis(name)
    tcp = env.tcp_pos()
    along = float((tcp - a) @ d)
    v = tcp - (a + along * d)
    R = env.data.site_xmat[env.tcp_id].reshape(3, 3)
    h = R.T @ v
    return dict(
        across_mm=round(1000 * float(np.linalg.norm(v)), 2),
        along_mm=round(1000 * along, 2),
        hand_x_mm=round(1000 * float(h[0]), 2),
        hand_y_mm=round(1000 * float(h[1]), 2),
        hand_z_mm=round(1000 * float(h[2]), 2),
    )


def hand_angles(env, name) -> dict:
    """The hand's orientation against the expert's for a cut of shoot `name` (prune_orientation): the jaws' closing
    axis from vertical (roll), the pointing direction from level (pitch) and from square across the shoot, seen from
    above (yaw), in degrees."""
    R = env.data.site_xmat[env.tcp_id].reshape(3, 3)
    _, d = env.shoot_axis(name)
    u = d[:2] / np.linalg.norm(d[:2])
    z = R[:, 2]
    zh = z[:2] / max(np.linalg.norm(z[:2]), 1e-9)
    return dict(
        roll_deg=round(float(np.degrees(np.arccos(min(1.0, abs(R[2, 1]))))), 1),
        pitch_deg=round(float(np.degrees(np.arcsin(np.clip(z[2], -1, 1)))), 1),
        yaw_deg=round(float(np.degrees(np.arcsin(min(1.0, abs(zh @ u))))), 1),
    )


def zone_distance(env, name) -> float:
    """Metres from the gripper's centre to the middle of shoot `name`'s cut zone."""
    a, d = env.shoot_axis(name)
    return float(np.linalg.norm(env.tcp_pos() - (a + d * CUT_MID)))


def subgoal_for(env, name) -> str:
    return f"cut the {reasoning.position_name(env.active.index(name), len(env.active))} shoot"


class Kinematics:
    """The gripper's centre for joint targets (forward kinematics on a copy of the model's data)."""

    def __init__(self, env):
        self.env, self.data = env, mujoco.MjData(env.model)

    def tcp(self, q):
        self.data.qpos[:7] = q[:7]
        mujoco.mj_kinematics(self.env.model, self.data)
        return self.data.site_xpos[self.env.tcp_id].copy()

    def across(self, q, a, d):
        p = self.tcp(q)
        v = p - a
        return float(np.linalg.norm(v - (v @ d) * d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["expert", "handover", "shadow"], required=True)
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--task", default="prune")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed_start", type=int, default=VALIDATION_SEED)
    ap.add_argument("--handover_cm", type=float, default=100.0)
    ap.add_argument(
        "--shift_mm",
        type=float,
        default=0.0,
        help="handover: the expert's plan for the shoot is moved this far up or down (the sign drawn per episode), so "
        "that the hand is handed over off its line by a known amount: does the fast level correct it?",
    )
    ap.add_argument("--shadow_cm", type=float, default=8.0)
    ap.add_argument("--exec_horizon", type=int, default=8)
    ap.add_argument("--flow_steps", type=int, default=10)
    ap.add_argument("--samples", type=int, default=1, help="handover: chunks drawn per look, averaged")
    ap.add_argument("--spread_every", type=int, default=4, help="shadow: draw 8 chunks every n-th look for the spread")
    ap.add_argument("--budget", type=int, default=240, help="handover: control steps the fast level has for the cut")
    ap.add_argument("--action_noise", type=float, default=0.0, help="expert mode: collect.py's DART noise, radians")
    ap.add_argument("--action_noise_quiet_cm", type=float, default=0.0, help="expert mode: as collect.py's")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from .collect import ActionNoise
    from .expert import PRUNE_OPEN, PRUNE_SECTOR, ScriptedExpert, clear_start

    env = PickupEnv(args.task, scenario=args.scenario)
    starter = ScriptedExpert(env, seed=0)
    kin = Kinematics(env)
    policy = None
    if args.mode != "expert":
        from .policy_server import Policy

        policy = Policy(args.checkpoint_dir, None, "cuda", args.weights)

    def ask(obs_prev, obs, prev_action, goal, samples=1):
        msg = infer_request(obs_prev, obs, prev_action, goal, args.flow_steps)
        chunks = [policy.infer(msg["images"], msg["state"], msg["prev_action"], goal, args.flow_steps)]
        chunks += [
            policy.infer(msg["images"], msg["state"], msg["prev_action"], goal, args.flow_steps)
            for _ in range(samples - 1)
        ]
        return np.asarray(chunks)  # [samples, 1 + future, 8]

    records, t0 = [], time.time()
    for ep in range(args.episodes):
        seed = args.seed_start + ep
        obs = env.reset(seed=seed)
        obs, yaw = clear_start(env, starter, seed)
        expert = ScriptedExpert(env, seed=seed)
        expert.reset(yaw=yaw)
        rec = dict(seed=seed, cuts=[], handover=None)
        noise = ActionNoise(seed, args.action_noise, args.action_noise_quiet_cm / 100)
        prev_obs, done, info = obs, False, {}
        driver, queue, close_at = "expert", [], {}
        shadow = []  # (step, shoot, distance, predicted chunk, spread)
        commands = []
        handed = None
        steps_after = 0
        crossings, crossed = [], set()  # the hand on its way in: at CROSS_MM from the cut zone's middle
        shifted = set()
        while not done:
            name = getattr(expert, "prune_target", None)
            live = name is not None and name not in env.cut
            in_reach = live and abs(expert._turn_needed(name)) <= PRUNE_SECTOR and abs(env.plate_speed()) < 0.05
            cutting = any(w[1] == PRUNE_OPEN for w in expert.waypoints)  # the plan of a cut, not of a press
            if args.shift_mm and driver == "expert" and in_reach and cutting and name not in shifted:
                shifted.add(name)
                dz = args.shift_mm / 1000 * (1 if np.random.default_rng([seed, 5]).random() < 0.5 else -1)
                lift = np.array([0.0, 0.0, dz])
                expert.waypoints = [
                    (w[0] + lift, *w[1:]) if i >= expert.wp else w for i, w in enumerate(expert.waypoints)
                ]
                rec["shift_mm"] = round(1000 * dz, 1)
            if driver == "expert" and args.mode == "handover" and in_reach and float(obs["state"][7]) > 0.3:
                dist = zone_distance(env, name)
                if dist < args.handover_cm / 100:
                    driver, handed = "policy", name
                    rec["handover"] = dict(
                        shoot=name,
                        step=env.steps,
                        distance_mm=round(1000 * dist, 1),
                        **{k: v for k, v in cut_offset(env, name).items() if k == "across_mm"},
                    )
            goal = handed if driver == "policy" else (name if in_reach else None)
            if goal is not None and goal not in env.cut and float(obs["state"][7]) > 0.3:
                dist = 1000 * zone_distance(env, goal)
                for mm in CROSS_MM:
                    if dist < mm and (goal, mm) not in crossed:
                        crossed.add((goal, mm))
                        crossings.append(
                            dict(shoot=goal, mm=mm, driver=driver, **cut_offset(env, goal), **hand_angles(env, goal))
                        )
            if driver == "expert":
                action = expert.act()
                if args.mode == "shadow" and live and zone_distance(env, name) < args.shadow_cm / 100:
                    k = 8 if (env.steps % args.spread_every == 0) else 1
                    chunk = ask(prev_obs, obs, env.last_action, subgoal_for(env, name), k)
                    a, d = env.shoot_axis(name)
                    ends = np.array([kin.tcp(c[args.exec_horizon - 1]) for c in chunk])
                    shadow.append(
                        dict(
                            step=len(commands),
                            shoot=name,
                            distance_mm=round(1000 * zone_distance(env, name), 1),
                            chunk=chunk[0].tolist(),
                            axis=(a.tolist(), d.tolist()),
                            spread_mm=round(1000 * float(np.linalg.norm(ends - ends.mean(0), axis=1).mean()), 2)
                            if k > 1
                            else None,
                        )
                    )
            else:
                if not queue:
                    chunk = ask(prev_obs, obs, env.last_action, subgoal_for(env, handed), args.samples)
                    queue = list(chunk.mean(axis=0)[: args.exec_horizon])
                action = np.asarray(queue.pop(0), dtype=np.float32)
                steps_after += 1
            commands.append(np.asarray(action, dtype=np.float64))
            for n in env.active:  # the step at which the jaws were last commanded shut while open near a shoot
                if (
                    n not in env.cut
                    and action[7] < 0.5
                    and float(obs["state"][7]) > 0.3
                    and zone_distance(env, n) < 0.03
                ):
                    close_at[n] = dict(step=env.steps, **cut_offset(env, n))
            before = set(env.cut)
            prev_obs = obs
            executed = noise.apply(action, env) if driver == "expert" and args.mode == "expert" else action
            obs, done, info = env.step(executed)
            for n in sorted(set(env.cut) - before):
                c = dict(shoot=n, step=env.steps, driver=driver, **cut_offset(env, n))
                if n in close_at:
                    c["close_cmd"] = close_at[n]
                rec["cuts"].append(c)
            if driver == "policy" and (handed in env.cut or steps_after >= args.budget):
                break
        rec["crossings"] = crossings
        rec["success"] = bool(info.get("success"))
        rec["damaged"] = info.get("damaged")
        rec["wrong"] = sorted(n for n in env.cut if env.targets and n not in env.targets)
        if args.mode == "handover":
            rec["handed_cut"] = handed is not None and handed in env.cut
            rec["policy_steps"] = steps_after
        if args.mode == "shadow":
            cmd = np.asarray(commands)
            rows = []
            for s in shadow:
                t = s["step"]
                pred = np.asarray(s["chunk"])
                h = min(args.exec_horizon, len(cmd) - t)
                a, d = (np.asarray(x) for x in s["axis"])
                err = [1000 * float(np.linalg.norm(kin.tcp(pred[k]) - kin.tcp(cmd[t + k]))) for k in range(h)]
                # where each would have the hand at the end of the executed part of the chunk, across the shoot
                k = h - 1
                close_pred = next((i for i in range(len(pred)) if pred[i][7] < 0.5), None)
                close_exp = next((i for i in range(min(len(pred), len(cmd) - t)) if cmd[t + i][7] < 0.5), None)
                rows.append(
                    dict(
                        step=t,
                        shoot=s["shoot"],
                        distance_mm=s["distance_mm"],
                        spread_mm=s["spread_mm"],
                        err_first_mm=round(err[0], 2),
                        err_mean_mm=round(float(np.mean(err)), 2),
                        across_pred_mm=round(1000 * kin.across(pred[k], a, d), 2),
                        across_expert_mm=round(1000 * kin.across(cmd[t + k], a, d), 2),
                        close_pred=close_pred,
                        close_expert=close_exp,
                    )
                )
            rec["shadow"] = rows
        records.append(rec)
        if (ep + 1) % 10 == 0:
            print(f"[cut_diag] {ep + 1}/{args.episodes} episodes, {time.time() - t0:.0f} s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(args=vars(args), episodes=records), indent=1))
    summary(records, args.mode)


def summary(records, mode):
    def stats(xs, label):
        if not xs:
            print(f"  {label}: none")
            return
        x = np.asarray(xs)
        print(
            f"  {label}: n {len(x)} median {np.median(x):.1f} mean {x.mean():.1f} sd {x.std():.1f} "
            f"p10 {np.percentile(x, 10):.1f} p90 {np.percentile(x, 90):.1f}"
        )

    def count(key):
        return sum(bool(key(r)) for r in records)

    def med(rows, key, absolute=False):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.median(np.abs(values) if absolute else values)) if values else float("nan")

    cuts = [c for r in records for c in r["cuts"]]
    if mode == "handover":
        cuts = [c for c in cuts if c["driver"] == "policy"]
    n = len(records)
    print(f"episodes {n}: solved {count(lambda r: r['success'])}, damaged {count(lambda r: r['damaged'] is not None)}")
    if mode == "handover":
        print(
            f"handed over {count(lambda r: r['handover'] is not None)}/{n}, cut {count(lambda r: r['handed_cut'])}, "
            f"damaged {count(lambda r: r['damaged'] is not None)}, wrong {count(lambda r: r['wrong'])}"
        )
    stats([c["across_mm"] for c in cuts], "across at the cut (mm)")
    for k in ("hand_x_mm", "hand_y_mm", "hand_z_mm", "along_mm"):
        stats([c[k] for c in cuts], k)
    stats([c["close_cmd"]["across_mm"] for c in cuts if "close_cmd" in c], "across when the jaws were commanded shut")
    stats([c["step"] - c["close_cmd"]["step"] for c in cuts if "close_cmd" in c], "steps from that command to the cut")
    xs = [x for r in records for x in r.get("crossings", []) if (mode != "handover" or x["driver"] == "policy")]
    if xs:
        print("  on the way in (median |.|, mm and degrees):")
        for mm in CROSS_MM:
            b = [x for x in xs if x["mm"] == mm]
            if b:
                print(
                    f"    {mm:3d} mm: n {len(b)} hand_x {med(b, 'hand_x_mm', True):.1f} "
                    f"hand_y {med(b, 'hand_y_mm', True):.1f} roll {med(b, 'roll_deg'):.1f} "
                    f"pitch {med(b, 'pitch_deg', True):.1f} yaw {med(b, 'yaw_deg'):.1f}"
                )
    if mode == "shadow":
        rows = [s for r in records for s in r["shadow"]]
        for lo, hi in ((0, 10), (10, 20), (20, 35), (35, 50), (50, 80)):
            b = [s for s in rows if lo <= s["distance_mm"] < hi]
            if b:
                print(
                    f"  {lo:2d}-{hi:2d} mm from the zone: n {len(b)} err first {med(b, 'err_first_mm'):.1f} "
                    f"mean {med(b, 'err_mean_mm'):.1f} | across at chunk end: policy {med(b, 'across_pred_mm'):.1f} "
                    f"expert {med(b, 'across_expert_mm'):.1f} | spread of 8 draws {med(b, 'spread_mm'):.1f}"
                )


if __name__ == "__main__":
    main()
