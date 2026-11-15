"""Scripted solutions of the avoidance drills (avoid.py): a known safe approach for each drill, checked for clearance.

Each drill's approach is an approach direction (the way the gripper points as it comes in, and how the jaws lie) and a
few waypoints: a staging point clear of everything, where the hand turns in place to the approach orientation; a point
just outside (in front of the opening, short of the gap, above the dot); the target; and for avoid_turn the way back
out. Translations never turn the hand and turns never move it, so the path the waypoint follower takes is exactly the
straight segments and in-place turns that `check` looks at: every centimetre and every 0.08 rad it solves the arm's
pose and measures the distance from every collision shape of the arm (links, hand, open or closed fingers) to every
obstacle, to the table and to the hot zone (MuJoCo's geom distance). A plan is safe when that distance stays above
AVOID_MARGIN and every pose is reachable. The environment draws layouts until the plan is safe (layouts with no safe
path are left out); for avoid_hot the expert tries several approach directions and takes the clearest.
"""

import mujoco
import numpy as np

from . import avoid as V
from .aim_expert import CRUISE, AimExpert
from .env import HOME_Q
from .expert import _angle, _rotz, _turn_toward

SAFE_ABOVE = 0.12  # metres above the highest obstacle for a move over everything
STEP, TURN_STEP = 0.01, 0.08  # metres, radians between checked poses
PRE_DWELL = 8  # control steps at the point outside: the sag correction settles before the slow move in


def _orient(p, jaws, R_ref):
    """Gripper orientation pointing along `p`, the jaws closing horizontally ('h': the hand's long side level) or
    vertically ('v': edge-on, the hand's long side upright); of the two such, the one nearer `R_ref`."""
    z = np.asarray(p, dtype=float) / np.linalg.norm(p)
    h = np.cross([0.0, 0.0, 1.0], z)
    if np.linalg.norm(h) < 1e-6:  # pointing straight down or up: keep R_ref's jaw direction
        h = R_ref[:, 1] - (R_ref[:, 1] @ z) * z
    h /= np.linalg.norm(h)
    if jaws == "h":
        y = h
        x = np.cross(y, z)
    else:
        x = h
        y = np.cross(z, x)
    cands = [np.stack([x, y, z], axis=1), np.stack([-x, -y, z], axis=1)]
    return min(cands, key=lambda R: _angle(R, R_ref))


class AvoidExpert(AimExpert):
    turn_rate = 0.04  # half the usual: in tight places the arm must follow its turning set-point closely

    def _plan(self, again=False):
        env = self.env
        self.factor = self.rng.uniform(0.8, 1.2)
        self.wp, self.dwell, self.chase = 0, 0, False
        self.yaw_off = self.des_yaw
        self.flick_index = env.flick_index
        R0 = self.des_R if self.des_R is not None else _rotz(self.des_yaw) @ self.R_home
        self.des_R = R0
        moves = self.plan(check=False) or self.plan(check=False, force=True)
        stage = moves[0][0]
        # from where the hand starts: straight to the staging point if that is clear, else up, over and down
        lead = [(stage.copy(), moves[0][1], CRUISE, 0, R0)]
        clear = self.check([lead[0]], self.des_pos, R0, env.data.qpos[:7].copy())
        if clear is None or clear < V.AVOID_MARGIN:
            top = self._top() + SAFE_ABOVE
            z = max(top, self.des_pos[2])
            lead = [
                (np.array([self.des_pos[0], self.des_pos[1], z]), moves[0][1], CRUISE, 0, R0),
                (np.array([stage[0], stage[1], z]), moves[0][1], CRUISE, 0, R0),
                lead[0],
            ]
        self.waypoints = [
            (pos, grip, speed * self.factor if speed > 0.0 else speed, dwell, *rest)
            for pos, grip, speed, dwell, *rest in lead + moves
        ]

    def _top(self):
        m, env = self.env.model, self.env
        tops = [m.body_pos[env.slab_body[i]][2] + m.geom_size[env.slab_geom[i]][2] for i in range(env.slabs_used)]
        if env.zone_r > 0:
            tops.append(m.body_pos[env.zone_body][2] + env.zone_r)
        return max(tops, default=0.0)

    # ------------------------------------------------------------------ plans
    def plan(self, check=True, force=False):
        """The drill's moves from the staging point on: [(position, gripper, speed, dwell, orientation[, until])],
        the first being the staging point itself (reached with the hand as it is, then turned in place). With `check`,
        None unless the moves are safe from the staging point on (self.last_clearance: the least distance). `force`:
        the best plan even if unsafe."""
        env = self.env
        R_ref = self.R_home
        options = getattr(self, f"_moves_{env.drill}")(R_ref)
        best, best_clear = None, -np.inf
        for moves in options:
            if not check and len(options) == 1:
                return moves
            stage = moves[0]
            clear = self.check(moves[1:], stage[0], stage[4], None)
            if clear is not None and clear > best_clear:
                best, best_clear = moves, clear
        self.last_clearance = best_clear if best is not None else None
        if best is not None and best_clear >= V.AVOID_MARGIN:
            return best
        return options[0] if force else None

    def _turn(self, pos, grip, R):
        """Turn in place: done once the set-point has turned and held still for 3 steps (at most 60 steps: a hand that
        cannot quite reach the point does not wait for ever)."""
        held = [0]

        def turned():
            held[0] = held[0] + 1 if _angle(self.des_R, R) < 1e-6 else 0
            return held[0] > 3

        return (pos.copy(), grip, 0.1, 60, R, turned)

    def _set_point_at(self, target):
        return lambda: np.linalg.norm(self.des_pos - target) < 1e-6

    def _moves_box(self, R_ref):
        env = self.env
        p = env.layout["point"]
        R = _orient(p, "h", R_ref)
        grasp = env.acube_pos() + [0, 0, 0.004]
        pre = grasp - p * 0.15
        stage = pre + [0, 0, 0.10]
        return [
            [
                (stage, 1.0, CRUISE, 0, R_ref),
                self._turn(stage, 1.0, R),
                (pre, 1.0, 0.30, PRE_DWELL, R),
                (grasp, 1.0, 0.08, 2, R),
                (grasp, 0.0, 0.0, 8, R),  # close (not checked: from here on the fingers are on the cube)
                (grasp + [0, 0, 0.012], 0.0, 0.03, 10**6, R),
            ]
        ]

    _moves_shelf = _moves_box

    def _moves_gap(self, R_ref):
        env, L = self.env, self.env.layout
        R_in = _orient(L["point"], "h", R_ref)  # level through the gap
        R_f = _orient(L["final"], "h", R_in)
        pre = L["gap_centre"] - L["point"] * 0.12
        near = L["dot"] - L["final"] * 0.05
        stage = pre + [0, 0, 0.08]
        return [
            [
                (stage, 0.0, CRUISE, 0, R_ref),
                self._turn(stage, 0.0, R_in),
                (pre, 0.0, 0.30, PRE_DWELL, R_in),
                (near, 0.0, 0.15, 0, R_in),
                self._turn(near, 0.0, R_f),  # past the posts: turn the wrist
                (env.dot_pos(), 0.0, 0.06, 10**6, R_f),
            ]
        ]

    def _moves_wall(self, R_ref):
        L = self.env.layout
        R = self._down_along(L["along"], R_ref)  # the hand's long side along the wall
        dot = L["dot"]
        stage = np.array([dot[0], dot[1], max(L["top"], dot[2]) + SAFE_ABOVE])
        above = dot + [0, 0, 0.05]
        return [
            [
                (stage, 0.0, CRUISE, 0, R_ref),
                self._turn(stage, 0.0, R),
                (above, 0.0, 0.30, PRE_DWELL, R),
                (dot.copy(), 0.0, 0.08, 10**6, R),
            ]
        ]

    def _down_along(self, along, R_ref):
        z = np.array([0.0, 0.0, -1.0])
        y = np.array([along[0], along[1], 0.0]) / np.linalg.norm(along[:2])
        cands = [np.stack([np.cross(s * y, z), s * y, z], axis=1) for s in (1, -1)]
        return min(cands, key=lambda R: _angle(R, R_ref))

    def _moves_hot(self, R_ref):
        L = self.env.layout
        dot, zone = L["dot"], L["zone"]
        away = dot[:2] - zone[:2]
        across = np.array([-away[1], away[0], 0.0]) if np.linalg.norm(away) > 1e-6 else np.array([0.0, 1.0, 0.0])
        options = []
        R = self._down_along(across, R_ref)  # from above, the hand's long side across the line to the zone
        stage = dot + [0, 0, 0.15]
        options.append(
            [
                (stage, 0.0, CRUISE, 0, R_ref),
                self._turn(stage, 0.0, R),
                (dot + [0, 0, 0.05], 0.0, 0.30, PRE_DWELL, R),
                (dot.copy(), 0.0, 0.08, 10**6, R),
            ]
        )
        radial = np.arctan2(dot[1], dot[0])
        for off in (60, -60, 85, -85, 110, -110):
            for pitch in (0.0, np.radians(35)):
                a = radial + np.radians(off)
                p = np.cos(pitch) * np.array([np.cos(a), np.sin(a), 0.0]) + [0, 0, -np.sin(pitch)]
                R = _orient(p, "h", R_ref)
                pre = dot - p * 0.12
                stage = pre + [0, 0, 0.10]
                options.append(
                    [
                        (stage, 0.0, CRUISE, 0, R_ref),
                        self._turn(stage, 0.0, R),
                        (pre, 0.0, 0.30, PRE_DWELL, R),
                        (dot.copy(), 0.0, 0.08, 10**6, R),
                    ]
                )
        return options

    def _moves_turn(self, R_ref):
        env, L = self.env, self.env.layout
        R_in = _orient(L["point"], "h", R_ref)
        R_f = _orient(L["final"], "h", R_in)
        dot = L["dot"]
        pre = dot - L["point"] * 0.12
        near = dot - L["point"] * 0.01
        stage = pre + [0, 0, 0.10]

        def touched():
            return env.touched

        return [
            [
                (stage, 0.0, CRUISE, 0, R_ref),
                self._turn(stage, 0.0, R_in),
                (pre, 0.0, 0.30, PRE_DWELL, R_in),
                (near, 0.0, 0.08, 0, R_in),
                self._turn(near, 0.0, R_f),  # inside the pocket: turn the hand
                (dot.copy(), 0.0, 0.05, 60, R_f, touched),
                (near, 0.0, 0.05, 40, R_f, self._set_point_at(near)),  # (the hand may lag a little: go on)
                self._turn(near, 0.0, R_in),
                (pre, 0.0, 0.15, 10**6, R_in),  # back out the way it came
            ]
        ]

    # --------------------------------------------------------------- checking
    def check(self, moves, pos, R, q0):
        """Least clearance (metres) of the arm to the obstacles and the hot zone along `moves` from `pos` with the hand
        at `R`, or None if a pose on the way cannot be reached. Moves after the gripper closes are not checked."""
        env, m, d = self.env, self.env.model, self.ik_data
        poses = []
        for move in moves:
            target, grip, _, _, Rm = move[:5]
            if grip < 0.5 and env.drill in ("box", "shelf"):
                break
            dist = np.linalg.norm(target - pos)
            if dist > 1e-9:
                n = int(np.ceil(dist / STEP))
                poses += [(pos + (target - pos) * k / n, R, grip) for k in range(1, n + 1)]
            while _angle(R, Rm) > 1e-6:
                R, _ = _turn_toward(R, Rm, TURN_STEP)
                poses.append((target.copy(), R, grip))
            pos = target
        q = np.array(q0 if q0 is not None else getattr(self, "q_cmd", HOME_Q), dtype=float)
        worst = np.inf
        for x, Rx, grip in poses:
            q = self.solve_ik(q, x, 0.0, iters=40, R_des=Rx)
            d.qpos[:7] = q
            d.qpos[7:9] = 0.04 * grip
            mujoco.mj_kinematics(m, d)
            err_p = np.linalg.norm(d.site_xpos[env.tcp_id] - x)
            if err_p > 0.004 or _angle(d.site_xmat[env.tcp_id].reshape(3, 3), Rx) > 0.05:
                return None
            clear, zone = env.clearance(d, table=True)
            worst = min(worst, clear, zone if env.zone_r > 0 else np.inf)
        return worst
