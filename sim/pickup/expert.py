"""Scripted top-down grasp expert used to generate demonstrations.

Plans in end-effector space (approach above -> descend -> close -> lift) and converts each
set-point to joint targets with damped-least-squares IK, so the recorded actions live in the
same 8-D joint-target space the policy is trained on.
"""

import mujoco
import numpy as np

from .env import BUTTONS, HOME_Q, OBJECTS, REACH_AZIMUTH, STEM_H, PickupEnv


def _wrap(a, period):
    return (a + period / 2) % period - period / 2


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _turn_toward(R, R_target, max_angle):
    """R rotated toward R_target by at most max_angle (radians), about the axis that joins them."""
    q, qt, dq = np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    mujoco.mju_mat2Quat(qt, R_target.reshape(-1))
    inv = np.zeros(4)
    mujoco.mju_negQuat(inv, q)
    mujoco.mju_mulQuat(dq, qt, inv)
    if dq[0] < 0:
        dq = -dq
    angle = 2 * np.arccos(np.clip(dq[0], -1, 1))
    if angle <= max_angle:
        return R_target.copy(), angle
    axis = dq[1:] / np.linalg.norm(dq[1:])
    step, out = np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(step, axis, max_angle)
    mujoco.mju_mulQuat(out, step, q)
    R_out = np.zeros(9)
    mujoco.mju_quat2Mat(R_out, out)
    return R_out.reshape(3, 3), angle


def _angle(R1, R2):
    return float(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1)))


def _segment_distance(points, a, b):
    """Distance of each of `points` (n x 3) to the segment a-b."""
    ab = b - a
    t = np.clip((points - a) @ ab / (ab @ ab), 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


TRACK_LIMIT = 0.04  # metres: a move of the object up to this is followed by shifting the plan, a larger one re-plans
SIDE_BLOCKED = 0.04  # metres: side="preferred" turns to the other side when its path passes this close to a stem
# Pruning
PRUNE_SECTOR = np.radians(25)  # a shoot pointing within this of REACH_AZIMUTH is cut without turning the plant
TURN_STOP = np.radians(4)  # let go of the button this far before the shoot points at REACH_AZIMUTH (the plate coasts)
PRESS_DEPTH = 0.0055  # metres the fingertips aim below the cap (they get about 4 of the 6 mm; the motor starts at 3)
FINGERTIP = 0.0098  # metres the fingertips reach beyond the tcp along the gripper's axis
SIDE_CLEAR = 0.02  # metres: an approach path this far from other shoots and the flower counts as clear
PRUNE_OPEN = 0.6  # gripper command while cutting: jaws 48 mm apart, enough for a rising shoot, clear of the pot
PRESS_TILT = np.radians(15)  # the gripper leans this far from vertical, fingertips away from the robot, to press
LANE_BACK = 0.20  # metres toward the robot from the stand's axis: the line the hand travels along between moves
# Taking over (take_over): a gripper this close to the main stem's axis, below the top of the flower plus this margin,
# is among the shoots; it backs out along its own axis, at most this far, before anything else
AMONG_SHOOTS, ABOVE_FLOWER, BACK_OUT = 0.17, 0.06, 0.15  # metres
# Points on the open fingers, the hand and the last link (tcp frame): the clearance of an approach path.
HAND_POINTS = np.array(
    [[x, y, z] for x in (-0.01, 0.01) for y in (-0.045, 0.045) for z in (-0.045, 0.01)]
    + [[x, y, z] for x in (-0.032, 0.032) for y in (-0.105, 0.0, 0.105) for z in (-0.13, -0.035)]
    + [[x, y, z] for x in (-0.044, 0.044) for y in (-0.055, 0.09) for z in (-0.16, -0.10)]  # the last link
)


class ScriptedExpert:
    """`reactive=False`: one plan per episode, made at reset. `reactive=True`: the expert also watches
    the named object. When it has moved a little (up to TRACK_LIMIT, as a drifting target does between
    two checks) the plan is shifted with it and the expert carries on where it was; when it has jumped
    further, or a grasp closed on nothing, the expert plans again, which gives demonstrations of
    recovery (see disturb.py).

    `side` (stems only) picks the side a stem is approached from: 'clearance', the side whose path in passes
    farthest from the other stems; 'preferred', from the outer edge of the table toward its middle, unless that
    path passes within SIDE_BLOCKED of another stem. The second makes the choice follow from where the stem
    stands, not from fine distances to the other stems."""

    def __init__(self, env: PickupEnv, seed: int = 0, reactive: bool = False, side: str = "clearance"):
        assert side in ("clearance", "preferred"), side
        self.env = env
        self.reactive = reactive
        self.side = side
        self.rng = np.random.default_rng(seed)
        self.ik_data = mujoco.MjData(env.model)
        # Reference top-down orientation and finger-axis yaw at the home configuration.
        self.ik_data.qpos[:7] = HOME_Q
        mujoco.mj_kinematics(env.model, self.ik_data)
        self.R_home = self.ik_data.site_xmat[env.tcp_id].reshape(3, 3).copy()
        self.yaw_home = float(np.arctan2(self.R_home[1, 1], self.R_home[0, 1]))
        # Pruning: pointing down with the fingers closing along the table's y, the hand's long side across the table
        self.R_down = _rotz(np.pi / 2) @ self.R_home
        # Pressing a button: leaning the fingertips away from the robot keeps the wrist back, out of the way of the
        # leaves of a shoot that the plate is turning toward the robot.
        c, s = np.cos(PRESS_TILT), np.sin(PRESS_TILT)
        self.R_press = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T @ self.R_down  # about y, by -PRESS_TILT

    # --------------------------------------------------------------------- IK
    def solve_ik(self, q_init, pos, yaw_offset, iters=30, damping=0.05, R_des=None):
        """Joint targets that put the gripper's centre at `pos`, pointing down with the yaw offset given, or in
        the orientation `R_des` when that is given (a grasp from the side)."""
        m, d, sid = self.env.model, self.ik_data, self.env.tcp_id
        if R_des is None:
            R_des = _rotz(yaw_offset) @ self.R_home
        q = np.array(q_init, dtype=float)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[:7] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.site_xmat[sid].reshape(3, 3)
            err_p = pos - d.site_xpos[sid]
            err_r = 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))
            err = np.concatenate([err_p, err_r])
            if np.linalg.norm(err_p) < 2e-4 and np.linalg.norm(err_r) < 2e-3:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))
            dq = J_pinv @ err + (np.eye(7) - J_pinv @ J) @ (0.05 * (HOME_Q - q))  # null-space posture bias
            q = np.clip(q + dq, self.env.ctrl_lo, self.env.ctrl_hi)
        return q

    # ------------------------------------------------------------------- plan
    def reset(self, yaw=0.0, R=None):
        """Plan for the current scene. `yaw`: the gripper's yaw offset from the home orientation at the
        start (0 at the home pose; random_start returns it for other starting poses). `R`: the gripper's
        orientation, when it does not point down (see take_over)."""
        env = self.env
        self.des_pos = env.tcp_pos()
        self.des_yaw = yaw
        self.des_R = R  # also set when a grasp from the side turns the gripper away from pointing down
        self.q_cmd = env.data.ctrl[:7].copy()
        self.replans = 0
        self.tracks = 0
        self.preamble = False  # take_over: still on the moves that only take_over makes (see there)
        self._plan()

    def take_over(self):
        """Plan from the arm as it is now, part way through an episode that another controller (a learned policy)
        has been driving: the set-point starts at the gripper's actual position and orientation, and the plan
        covers what is left of the task (pruning: the shoots not yet cut). Top-down grasps keep only the yaw of
        the actual orientation, as their plans assume the gripper points down.

        Pruning from some poses first needs moves that the expert makes nowhere else (backing out of the plant, a move
        back onto the lane); `preamble` is True until they are done and the ordinary plan begins. Demonstrations start
        after them (collect.hand_over): the same state must not be labelled one way here and another way in the
        expert's own demonstrations."""
        env = self.env
        R = env.data.site_xmat[env.tcp_id].reshape(3, 3).copy()
        yaw = float(_wrap(env.tcp_yaw() - self.yaw_home, 2 * np.pi))
        self.reset(yaw=yaw, R=R if env.plant_task else None)  # every pruning waypoint carries its orientation
        self.q_cmd = env.data.qpos[:7].copy()  # the joints as they are, not the last command another controller sent
        if env.plant_task and self._among_shoots():
            # A plan may start by turning the hand where it is, which among the shoots swings the fingers into them or
            # the flower. First open the jaws (they may hold a shoot) and back out the way the hand points (never
            # downward) until it is clear of the plant, or for at most 3 s; the next plan starts from there.
            out = -R[:, 2].copy()
            out[2] = max(out[2], 0.0)
            out = out / np.linalg.norm(out) if np.linalg.norm(out) > 0.3 else np.array([0.0, 0.0, 1.0])
            self.waypoints = [
                (self.des_pos.copy(), PRUNE_OPEN, 0.0, 3, R),
                (self.des_pos + out * BACK_OUT, PRUNE_OPEN, 0.15, 60, R, lambda: not self._among_shoots()),
            ]
            self.wp, self.dwell, self.preamble = 0, 0, True
        elif env.plant_task:
            # Far out or high up (outside the box random starts are drawn from), a hand pointing down may be out of
            # reach, and a plan that starts by turning it there never arrives: first move into the box, back on the
            # lane in front of the plant (clear of the flower), and turn.
            lane_x = env.data.xpos[env.stand_body][0] - LANE_BACK
            lo, hi = np.array([START_X[0], START_Y[0], START_Z[0]]), np.array([lane_x, START_Y[1], START_Z[1]])
            inside = np.clip(self.des_pos, lo, hi)
            if not np.all((self.des_pos >= lo) & (self.des_pos <= [START_X[1], START_Y[1], START_Z[1]])):
                self.waypoints = [
                    (inside, PRUNE_OPEN, 0.3, 100, self.R_down, lambda: np.linalg.norm(env.tcp_pos() - inside) < 0.02)
                ]
                self.wp, self.dwell, self.preamble = 0, 0, True

    def _among_shoots(self):
        """Is the gripper among the shoots: near the main stem's axis, below the top of the flower (plus a margin)?"""
        base, up = self.env.stem_axis()
        rel = self.env.tcp_pos() - base
        return bool(np.linalg.norm(rel - up * (rel @ up)) < AMONG_SHOOTS and rel @ up < STEM_H + ABOVE_FLOWER)

    def _plan(self, again=False):
        """Waypoints for the named object where it is now, starting from the current set-point."""
        env = self.env
        if env.plant_task:
            return self._plan_prune()
        obj = OBJECTS[env.target]
        if obj.get("grasp") == "side":
            return self._plan_side(again)
        p = env.object_pos(env.target)
        if obj["grasp_sym"] is None:
            yaw_off = self.rng.uniform(-0.3, 0.3)
        else:
            finger_yaw = env.object_yaw(env.target) + obj["grasp_axis"]
            yaw_off = _wrap(finger_yaw - self.yaw_home, obj["grasp_sym"])
        xy_noise = self.rng.normal(0, 0.003, 2)
        grasp = np.array([p[0] + xy_noise[0], p[1] + xy_noise[1], max(p[2], 0.018)])
        speed = self.rng.uniform(0.8, 1.2)
        approach = grasp + [0, 0, self.rng.uniform(0.10, 0.14)]
        # (position, gripper, max speed m/s, dwell steps after arrival)
        self.waypoints = [
            (approach, 1.0, 0.40 * speed, 0),
            (grasp, 1.0, 0.18 * speed, 2),
            (grasp, 0.0, 0.0, 8),
            (grasp + [0, 0, self.rng.uniform(0.18, 0.24)], 0.0, 0.30 * speed, 1000),
        ]
        if again and self.des_pos[2] < approach[2] - 0.02:
            # Planning again with the gripper low: go straight up first, so the fingers clear the objects.
            self.waypoints.insert(0, (np.array([self.des_pos[0], self.des_pos[1], approach[2]]), 1.0, 0.40 * speed, 0))
        self.wp = 0
        self.dwell = 0
        self.yaw_off = yaw_off
        self.planned_xy = p[:2].copy()

    def side_orientation(self, toward):
        """Gripper pointing horizontally along `toward` (unit, in the table plane), fingers closing sideways.
        Of the two such orientations, the one nearer the gripper pointing down (less wrist roll)."""
        z = np.array([toward[0], toward[1], 0.0])
        best = None
        for sign in (1, -1):
            y = sign * np.array([-toward[1], toward[0], 0.0])
            R = np.stack([np.cross(y, z), y, z], axis=1)
            angle = np.arccos(np.clip((np.trace(R @ self.R_home.T) - 1) / 2, -1, 1))
            if best is None or angle < best[0]:
                best = (angle, R)
        return best[1]

    def _plan_side(self, again=False):
        """Take a stem from the side: rise and turn the gripper horizontal well back from the stem, come level
        with the band, move in until the stem is between the fingers, close and hold."""
        env = self.env
        obj = OBJECTS[env.target]
        p = env.object_pos(env.target)
        # Across the table, at right angles to the line from the robot's base (at the origin): pointing straight
        # out from the base is out of reach for a horizontal gripper, across is in reach everywhere on the table.
        # Of the two sides, the one whose path in passes farthest from the other stems.
        radial = p[:2] / np.linalg.norm(p[:2])
        back = self.rng.uniform(0.12, 0.15)
        others = [env.object_pos(n)[:2] for n in env.active if n != env.target]

        def clearance(u):
            a, b = p[:2] - u * back, p[:2]
            t = [np.clip((o - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1) for o in others]
            gaps = [np.linalg.norm(o - (a + k * (b - a))) for o, k in zip(others, t, strict=True)]
            return min(gaps, default=1.0)

        # 'preferred': the hand comes in from the outer edge of the table (the side away from the robot's
        # middle line, y = 0) and moves toward the middle.
        clockwise, anticlockwise = np.array([radial[1], -radial[0]]), np.array([-radial[1], radial[0]])
        preferred, other = (clockwise, anticlockwise) if p[1] >= 0 else (anticlockwise, clockwise)
        if self.side == "preferred" and clearance(preferred) >= SIDE_BLOCKED:
            toward = preferred
        else:
            toward = max((other, preferred), key=clearance)
        R = self.side_orientation(toward)
        lo, hi = obj["band"]
        z = self.rng.uniform(lo + 0.01, hi - 0.01)
        xy_noise = self.rng.normal(0, 0.002, 2)
        grasp = np.array([p[0] + xy_noise[0], p[1] + xy_noise[1], z])
        speed = self.rng.uniform(0.8, 1.2)
        pre = grasp - np.array([toward[0], toward[1], 0.0]) * back
        self.waypoints = [
            (pre + [0, 0, 0.12], 1.0, 0.40 * speed, 0, R),
            (pre, 1.0, 0.25 * speed, 0, R),
            (grasp, 1.0, 0.12 * speed, 2, R),
            (grasp, 0.0, 0.0, 1000, R),
        ]
        if again:  # back out along the gripper's own axis before anything else
            axis = self.des_R[:, 2] if self.des_R is not None else np.array([toward[0], toward[1], 0.0])
            out = self.des_pos - np.array([axis[0], axis[1], 0.0]) * 0.10
            self.waypoints.insert(0, (out, 1.0, 0.25 * speed, 0, R))
        self.wp = 0
        self.dwell = 0
        self.yaw_off = self.des_yaw
        self.planned_xy = p[:2].copy()

    # ---------------------------------------------------------------- pruning
    def prune_orientation(self, toward):
        """Gripper pointing horizontally along `toward`, the jaws closing vertically, so the rising shoot crosses
        them. (Closing at right angles to the shoot instead tilts the jaws by the shoot's rise, and the open lower
        finger then swings in against the main stem.) Of the two, the one nearer pointing down (R_down)."""
        z = toward / np.linalg.norm(toward)
        candidates = []
        for s in (1, -1):
            y = np.array([0.0, 0.0, s])
            candidates.append(np.stack([np.cross(y, z), y, z], axis=1))
        return min(candidates, key=lambda R: _angle(R, self.R_down))

    def _turn_needed(self, name):
        return float(_wrap(REACH_AZIMUTH - self.env.shoot_azimuth(name), 2 * np.pi))

    def _clearance(self, name, cut, toward, R, back):
        """Least distance from the hand and open fingers, along the path in from `back` metres, to the other uncut
        shoots, their leaves and the flower."""
        env = self.env
        segments = []
        for other in env.active:
            if other != name and other not in env.cut:
                a, d = env.shoot_axis(other)
                segments.append((a, a + d * env.shoots[other]["length"]))
                for _, col in env.leaf_geoms[other]:  # each leaf in use, as the line along its blade
                    if env.model.geom_contype[col]:
                        centre, along = env.data.geom_xpos[col], env.data.geom_xmat[col].reshape(3, 3)[:, 0]
                        segments.append((centre - along * 0.022, centre + along * 0.022))
        base, up = env.stem_axis()
        segments.append((base + up * (STEM_H - 0.01), base + up * (STEM_H + 0.03)))  # the flower head
        points = np.concatenate([cut - toward * lam + HAND_POINTS @ R.T for lam in np.linspace(0, back, 8)])
        return min(_segment_distance(points, a, b).min() for a, b in segments)

    def _plan_prune(self):
        """The next step of pruning, from where the gripper is: if the shoot nearest to the reachable direction
        points away, press the button that turns it toward the robot the shorter way and hold it until the shoot is
        a few degrees from REACH_AZIMUTH; otherwise come in level from the side whose path is clearest, the jaws
        closing across the shoot inside its cut zone, close until it is cut, open and back out. The last waypoint
        of each plan calls for the next."""
        env = self.env
        speed = self.rng.uniform(0.8, 1.2)
        R_now = self.des_R if self.des_R is not None else _rotz(self.des_yaw) @ self.R_home
        remaining = [n for n in env.active if n not in env.cut]
        self.wp, self.dwell, self.preamble = 0, 0, False
        if not remaining:
            self.waypoints = [(self.des_pos.copy(), 1.0, 0.0, 10**6, R_now)]
            return
        name = min(remaining, key=lambda n: abs(self._turn_needed(n)))
        self.prune_target = name
        turn = self._turn_needed(name)
        lane_x = env.data.xpos[env.stand_body][0] - LANE_BACK
        if abs(turn) > PRUNE_SECTOR:
            button = next(b for b, (_, sign) in BUTTONS.items() if sign == np.sign(turn))
            top = env.button_top(button)
            R = self.R_press
            # the lowest point of the closed fingertips (a corner, with the hand leaning) goes PRESS_DEPTH below the
            # cap's top; the centre of the fingertips over the cap's centre, coming straight down
            tip = R[:, 2] * FINGERTIP
            lowest = min((tip + R[:, 0] * x)[2] for x in (-0.0105, 0.0105))
            press = top - tip + [0, 0, -PRESS_DEPTH - lowest + tip[2]]
            press[:2] = top[:2] - tip[:2]
            above = press + [0, 0, 0.09]

            sign = np.sign(turn)

            def faced(name=name, sign=sign):
                return self._turn_needed(name) * sign < TURN_STOP

            plan = [
                (above, 0.0, 0.40 * speed, 0, self.R_press),
                (press, 0.0, 0.08 * speed, 300, self.R_press, faced),
                (above, 0.0, 0.30 * speed, 0, self.R_press),
                # back toward the robot before anything else: the shoot now faces the arm, leaves and all
                (np.array([lane_x - 0.03, above[1], 0.20]), 0.0, 0.30 * speed, 0, self.R_down),
            ]
        else:
            a, d = env.shoot_axis(name)
            cut = a + d * self.rng.uniform(0.031, 0.039)
            u = d[:2] / np.linalg.norm(d[:2])
            back = self.rng.uniform(0.13, 0.15)
            options = []
            for side in (1, -1):
                toward = side * np.array([-u[1], u[0], 0.0])
                R = self.prune_orientation(toward)
                options.append((self._clearance(name, cut, toward, R, back), side, toward, R))
            # Pointing even slightly out from the robot's base, the hand cannot keep its jaws upright (the arm runs
            # out of reach and rolls the wrist, swinging the lower finger into the main stem): of two clear paths,
            # the one pointing less outward.
            clear = [o for o in options if o[0] >= SIDE_CLEAR]
            if clear:
                _, _, toward, R = min(clear, key=lambda o: o[2][0])
            else:
                _, _, toward, R = max(options, key=lambda o: o[0])
            pre = cut - toward * back
            lane = np.array([min(lane_x, pre[0]), pre[1], pre[2]])

            def is_cut(name=name):
                return name in env.cut

            ajar = PRUNE_OPEN
            plan = [
                (lane, ajar, 0.35 * speed, 0, R),
                (pre, ajar, 0.25 * speed, 0, R),
                (cut, ajar, 0.10 * speed, 2, R),
                (cut, 0.0, 0.0, 25, R, is_cut),  # close and hold until the shoot is cut
                (cut, 0.0, 0.0, 2, R),
                (cut, ajar, 0.0, 3, R),  # open: the cut part falls (on the plate, the pot or the table)
                (pre, ajar, 0.20 * speed, 0, R),
                (lane, ajar, 0.30 * speed, 0, R),
            ]
        stem = env.stem_axis()[0]
        if np.linalg.norm(self.des_pos[:2] - stem[:2]) < 0.15 and self.des_pos[2] > 0.2:  # above the plant (at home)
            plan.insert(
                0, (np.array([lane_x, self.des_pos[1], self.des_pos[2]]), plan[0][1], 0.35 * speed, 0, plan[0][4])
            )
        if _angle(R_now, plan[0][4]) > 0.6:  # turn the hand where it is first (it swings up, away from the plant)
            plan.insert(0, (self.des_pos.copy(), plan[0][1], 0.1, 0, self.R_down))
        self.waypoints = plan

    def _needs_new_plan(self):
        env = self.env
        if not env.on_table(env.target):
            return False
        moved = np.linalg.norm(env.object_pos(env.target)[:2] - self.planned_xy) > 0.015
        lifting = self.wp == len(self.waypoints) - 1
        return moved or (lifting and env.state()[7] < 0.05)  # the fingers met: nothing is in them

    def _track(self):
        """Shift the remaining waypoints with an object that has moved a little; True if that was done."""
        env = self.env
        delta = env.object_pos(env.target)[:2] - self.planned_xy
        closing = self.waypoints[self.wp][1] < 0.5
        if closing or np.linalg.norm(delta) > TRACK_LIMIT:
            return False
        self.waypoints = [(p + np.array([delta[0], delta[1], 0.0]), *rest) for p, *rest in self.waypoints]
        self.planned_xy = self.planned_xy + delta
        return True

    def act(self):
        env = self.env
        if self.reactive and not env.plant_task and self._needs_new_plan():
            if self._track():
                self.tracks += 1
            else:
                self.replans += 1
                self._plan(again=True)
        # (position, gripper, max speed m/s, dwell steps after arrival[, orientation[, condition]]): with a condition,
        # the waypoint is left once it holds after arrival, or when the dwell runs out
        target, grip, speed, dwell, *rest = self.waypoints[self.wp]
        R_target = rest[0] if rest else None
        until = rest[1] if len(rest) > 1 else None
        dt = 1.0 / 20.0
        delta = target - self.des_pos
        dist = np.linalg.norm(delta)
        step = speed * dt
        self.des_pos = target.copy() if dist <= step or speed == 0 else self.des_pos + delta / dist * step
        if R_target is None:
            dyaw = self.yaw_off - self.des_yaw
            self.des_yaw += np.clip(dyaw, -0.12, 0.12)
            turned = abs(self.yaw_off - self.des_yaw) < 1e-6
        else:  # turn toward the waypoint's orientation, at most 0.08 rad per control step
            if self.des_R is None:
                self.des_R = _rotz(self.des_yaw) @ self.R_home
            self.des_R, left = _turn_toward(self.des_R, R_target, 0.08)
            turned = left <= 0.08

        arrived = (
            np.linalg.norm(target - self.des_pos) < 1e-6 and np.linalg.norm(env.tcp_pos() - target) < 0.012 and turned
        )
        if arrived or speed == 0 or until is not None:  # a condition is watched (and its time limit runs) at once
            self.dwell += 1
            finished = self.dwell > dwell or (until is not None and until())
            if finished and self.wp < len(self.waypoints) - 1:
                self.wp += 1
                self.dwell = 0
            elif finished and env.plant_task:
                self._plan_prune()
        self.q_cmd = self.solve_ik(self.q_cmd, self.des_pos, self.des_yaw, R_des=self.des_R)
        return np.concatenate([self.q_cmd, [grip]]).astype(np.float32)


# Where an episode may start when it does not start at the home pose: gripper position (metres) and yaw.
START_X, START_Y, START_Z, START_YAW = (0.35, 0.65), (-0.25, 0.25), (0.22, 0.45), (-0.6, 0.6)


def random_start(env: PickupEnv, expert: ScriptedExpert, seed: int):
    """Move the arm to a random pose above the table, gripper pointing down, before the episode begins.

    From the home pose every reach is a straight line that starts at the same point, so a few steps in,
    the arm's own position says where it is heading and the cameras are no longer needed to know the
    goal. With a start that varies, the position of the arm alone says little. Returns (observation,
    yaw offset) and expects env.reset(seed) to have been called; pass the yaw to ScriptedExpert.reset.
    """
    rng = np.random.default_rng([seed, 13])  # its own stream: the layout of `seed` is unchanged
    pos = np.array([rng.uniform(*START_X), rng.uniform(*START_Y), rng.uniform(*START_Z)])
    yaw = float(rng.uniform(*START_YAW))
    q = expert.solve_ik(HOME_Q, pos, yaw, iters=200)
    return env.set_arm(q), yaw
