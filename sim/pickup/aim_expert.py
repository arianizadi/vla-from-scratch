"""Scripted solutions of the aim range's drills (aim.py), on the waypoint follower of the pruning expert.

ScriptedExpert(env) on an aim task returns an AimExpert. Each episode draws a speed factor, so the demonstrations
show the same drill done faster and slower; the moves themselves are straight lines at a steady speed (the
follower's), slowed down for the last few centimetres where a drill asks for a precise stop.
"""

import numpy as np

from . import aim as A
from .expert import FINGERTIP, PRESS_DEPTH, ScriptedExpert, _rotz, _wrap

SPEED = (0.7, 1.3)  # per-episode factor on the speeds below
CRUISE = 0.40  # m/s of the gripper's set-point between targets
LEAD = 2  # aim_track: control steps ahead on the dot's path the set-point aims at (the arm lags its set-point)
SAG_GAIN, SAG_LIMIT = 0.3, 0.02  # the set-point's correction for the arm's sag under gravity (see solve_ik)
GRASP_Z = 0.013  # aim_pinch: height of the gripper's centre when it closes on a small cube (fingertips 3 mm up)


def _minimal_turn(R, z):
    """R turned the least way (about one axis) so that its z axis points along the unit vector z."""
    a, b = R[:, 2], z / np.linalg.norm(z)
    v, c = np.cross(a, b), float(a @ b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return R.copy() if c > 0 else np.diag([1.0, -1.0, -1.0]) @ R
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    angle = np.arctan2(s, c)
    return (np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K) @ R


class AimExpert(ScriptedExpert):
    def __init__(self, env, seed=0, reactive=False, side="clearance"):
        self.sag = np.zeros(3)
        super().__init__(env, seed=seed, reactive=False, side=side)  # it watches its targets itself

    def solve_ik(self, q_init, pos, yaw_offset, iters=30, damping=0.05, R_des=None):
        """As the base class's, for the set-point plus the correction for sag. Held still under its position servos
        the arm sags about 8 mm below its set-point, which a precise stop (aim_hold) cannot afford: while the set-point
        stands still, the correction integrates what is left of the error."""
        return super().solve_ik(q_init, pos + self.sag, yaw_offset, iters=iters, damping=damping, R_des=R_des)

    def reset(self, yaw=0.0, R=None):
        self.sag = np.zeros(3)
        super().reset(yaw=yaw, R=R)

    def _plan(self, again=False):
        env = self.env
        self.factor = self.rng.uniform(*SPEED)
        self.wp, self.dwell, self.chase = 0, 0, False
        self.yaw_off = self.des_yaw + self.rng.uniform(-0.3, 0.3)
        self.flick_index = env.flick_index
        getattr(self, f"_plan_{env.drill}")()

    def _plan_touch(self):
        self.waypoints = [(self.env.dot_pos(), 1.0, CRUISE * self.factor, 10**6)]

    _plan_flick = _plan_touch

    def _plan_hold(self):
        dot = self.env.dot_pos()
        way = dot - self.des_pos
        slow = dot - way / max(np.linalg.norm(way), 1e-6) * min(0.03, np.linalg.norm(way))
        self.waypoints = [(slow, 1.0, CRUISE * self.factor, 0), (dot, 1.0, 0.06, 10**6)]

    def _plan_pose(self):
        env = self.env
        p = env.pin_dir
        R = _minimal_turn(_rotz(self.yaw_off) @ self.R_home, p)
        dot = env.dot_pos()
        pre = dot - p * self.rng.uniform(0.08, 0.11)
        high = pre + [0, 0, 0.06]  # turn the hand up there, clear of the table
        self.waypoints = [
            (high, 1.0, CRUISE * self.factor, 0, R),
            (pre, 1.0, 0.25 * self.factor, 0, R),
            (dot, 1.0, 0.10 * self.factor, 10**6, R),
        ]

    def _plan_track(self):
        self.chase = True
        self.waypoints = [(self._ahead(), 1.0, 0.5, 10**6)]

    def _ahead(self):
        env = self.env
        return env.path[min(env.steps + LEAD, len(env.path) - 1)].copy()

    def _plan_press(self):
        top = self.env.button_top()
        press = top + [0, 0, FINGERTIP - PRESS_DEPTH]  # the closed fingertips this far below the cap's top
        above = press + [0, 0, self.rng.uniform(0.06, 0.10)]
        self.waypoints = [(above, 0.0, CRUISE * self.factor, 0), (press, 0.0, 0.08 * self.factor, 10**6)]

    def _plan_pinch(self):
        env = self.env
        c = env.cube_pos()
        self.yaw_off = _wrap(env.cube_yaw() - self.yaw_home, np.pi / 2)
        xy = c[:2] + self.rng.normal(0, 0.002, 2)
        grasp = np.array([xy[0], xy[1], max(c[2] + GRASP_Z - A.CUBE_HALF, GRASP_Z)])
        approach = grasp + [0, 0, self.rng.uniform(0.08, 0.12)]
        self.waypoints = [
            (approach, 1.0, CRUISE * self.factor, 0),
            (grasp, 1.0, 0.15 * self.factor, 2),
            (grasp, 0.0, 0.0, 8),
            (grasp + [0, 0, 0.08], 0.0, 0.25 * self.factor, 10**6),
        ]

    def act(self):
        env = self.env
        if env.drill == "flick" and env.flick_index != self.flick_index and env.flick_index < len(env.flick_points):
            self.flick_index = env.flick_index  # the next dot has lit up
            self.wp, self.dwell = 0, 0
            self._plan_flick()
        if self.chase:
            self.waypoints[self.wp] = (self._ahead(), *self.waypoints[self.wp][1:])
        target = self.waypoints[self.wp][0]
        if np.linalg.norm(target - self.des_pos) < 1e-6:  # the set-point is standing still: correct for sag
            self.sag = np.clip(self.sag + SAG_GAIN * (self.des_pos - self.env.tcp_pos()), -SAG_LIMIT, SAG_LIMIT)
        return super().act()
