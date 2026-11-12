"""Scripted top-down grasp expert used to generate demonstrations.

Plans in end-effector space (approach above -> descend -> close -> lift) and converts each
set-point to joint targets with damped-least-squares IK, so the recorded actions live in the
same 8-D joint-target space the policy is trained on.
"""

import mujoco
import numpy as np

from .env import HOME_Q, OBJECTS, PickupEnv


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


TRACK_LIMIT = 0.04  # metres: a move of the object up to this is followed by shifting the plan, a larger one re-plans
SIDE_BLOCKED = 0.04  # metres: side="preferred" turns to the other side when its path passes this close to a stem


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
    def reset(self, yaw=0.0):
        """Plan for the current scene. `yaw`: the gripper's yaw offset from the home orientation at the
        start (0 at the home pose; random_start returns it for other starting poses)."""
        env = self.env
        self.des_pos = env.tcp_pos()
        self.des_yaw = yaw
        self.des_R = None  # set when a grasp from the side turns the gripper away from pointing down
        self.q_cmd = env.data.ctrl[:7].copy()
        self.replans = 0
        self.tracks = 0
        self._plan()

    def _plan(self, again=False):
        """Waypoints for the named object where it is now, starting from the current set-point."""
        env = self.env
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
        if self.reactive and self._needs_new_plan():
            if self._track():
                self.tracks += 1
            else:
                self.replans += 1
                self._plan(again=True)
        target, grip, speed, dwell, *R_target = self.waypoints[self.wp]
        R_target = R_target[0] if R_target else None
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
        if arrived or speed == 0:
            self.dwell += 1
            if self.dwell > dwell and self.wp < len(self.waypoints) - 1:
                self.wp += 1
                self.dwell = 0
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
