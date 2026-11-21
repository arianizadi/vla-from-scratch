"""The arm: what the plant scenarios need to know about the robot that works on the plant, kept in one place.

Everything here is specific to the arm and its gripper (a Franka Panda with its parallel jaws, as env.build_model
loads it): the gripper's frame (the tool centre point `tcp`, pointing along its z axis, the jaws closing along its y
axis), how wide the jaws open, the inverse kinematics and its joint limits, which collision shapes belong to the arm,
and how a cut is approached (the wrist's orientation for a stem segment and the line in). The scenarios (scenarios.py,
scenarios_reach.py) say what the plant looks like, what is in the way and what counts as success; they and the expert
ask the arm only `plan_cut` (a reachable, collision-free way to cut this shoot, or None).
Moving to another arm means writing another class with the same methods (`arm_for` picks it), not touching the
scenarios.

A cut, as this gripper makes it: the jaws close across the segment, so the segment lies in the fingers' plane (the
tcp's x-z plane) and crosses the closing axis y at right angles. Given the segment's direction d, the gripper may point
along any direction a at right angles to d, a = cos(phi) b1 + sin(phi) b2, where b1 is level and at right angles to d
and b2 is at right angles to both, pointing up (phi = 0 or pi: in from the side, level; phi < 0: the hand above the
segment, pointing down onto it; phi > 0: the hand below it, pointing up). It may also lean along d by psi (the hand
moved back toward the segment's base, psi > 0, or out toward its tip, psi < 0), the segment then crossing the fingers'
plane at that angle. The orientation the existing expert uses for prune (level, the jaws closing straight up and down,
expert.ScriptedExpert.prune_orientation) is one more candidate, tried first, so that where it is clear the new plans
look like the old ones.

A plan is checked as the expert will follow it (expert.ScriptedExpert._plan_reach), from where its set-point is: back
to the lane's plane in front of the plant, along it to the point level with the approach while turning to the cut's
orientation, to a point back along the approach, in to the cut, every control step as the expert's waypoint follower
moves its set-point (the speeds it uses, 0.08 rad of turn a step) and solved with the expert's own inverse kinematics
(damped least squares from the last pose, 30 iterations, a bias toward the home posture, the joints clipped to their
limits). Each pose must be reached to REACH_TOL and REACH_TOL_ROT with every joint LIMIT_MARGIN inside its limits, and
the arm's collision shapes (links 4 to 7, the hand and the open fingers) must keep each obstacle's margin (the caller
gives one per obstacle; MuJoCo's geom distance). The opening of the jaws is part of the plan: narrower jaws pass closer
to a stem or a leaf.

What is specific to this arm, and would be rewritten for another: the tcp frame and which axis the jaws close along
(cut_orientation, legacy_orientation), the jaws' travel (FINGER_TRAVEL, JAW_OPENINGS), the inverse kinematics and its
posture bias (ik), the joint limits (within_limits), which bodies are checked for collisions (CHECK_LINKS), the
candidate approaches (PHIS, PSIS: what this wrist can turn to), and how the hand settles before the jaws close (the
arm's sag and lag under its position servos: settle_step).
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .env import HOME_Q

REACH_TOL, REACH_TOL_ROT = 0.004, 0.05  # metres, radians: a pose solved to within these is reachable
MARGIN = 0.008  # metres the arm keeps from an obstacle on the way in (at the cut too), unless the caller says more
CHECK_LINKS = ("link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger")
JAW_OPENINGS = (0.6, 0.45, 0.35)  # gripper commands tried, widest first (0.6: 48 mm, as the existing expert cuts)
FINGER_TRAVEL = 0.04  # metres each finger moves from shut to the command 1.0
CROSS_MAX = 50  # degrees: the segment at most this far from square across the closing jaws (a cut needs within 45)
BRANCH_TOL = 0.3  # radians: a plan ends with every joint this near the canonical start's (canonical_start)
LIMIT_MARGIN = 0.1  # radians: a plan keeps every joint this far inside its limits (near them the arm cannot follow)
# Settling before the jaws close (the expert's `settle` option): held still under its position servos the arm sags and
# lags its set-point by millimetres, mostly downward, and the jaws used to shut while the hand was still on its way.
# The rule: at the cut, before closing, the set-point is corrected by SETTLE_GAIN of the hand's remaining error every
# control step (the sag taken out) until the hand's error along the closing jaws is under SETTLE_HEIGHT and its speed
# under SETTLE_SPEED, or SETTLE_STEPS have passed; the correction is held while the jaws close and let go on the way
# out.
SETTLE_HEIGHT, SETTLE_SPEED, SETTLE_STEPS, SETTLE_GAIN, SETTLE_MAX = 0.001, 0.005, 40, 0.3, 0.015
PHIS = np.radians([0, 180, -30, 210, -55, 235, 30, 150, -80, 260, 55, 125, 80, 100])  # the approach around d
PSIS = np.radians([0, -20, 20, -40])  # the lean along d


def _angle(R1, R2):
    return float(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1)))


def _turn_toward(R, R_target, max_angle):
    """R turned toward R_target by at most max_angle about the axis that joins them (expert._turn_toward)."""
    q, qt, dq, inv = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    mujoco.mju_mat2Quat(qt, R_target.reshape(-1))
    mujoco.mju_negQuat(inv, q)
    mujoco.mju_mulQuat(dq, qt, inv)
    if dq[0] < 0:
        dq = -dq
    angle = 2 * np.arccos(np.clip(dq[0], -1, 1))
    if angle <= max_angle:
        return R_target.copy()
    axis = dq[1:] / np.linalg.norm(dq[1:])
    step, out = np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(step, axis, max_angle)
    mujoco.mju_mulQuat(out, step, q)
    R_out = np.zeros(9)
    mujoco.mju_quat2Mat(R_out, out)
    return R_out.reshape(3, 3)


def segment_frame(d):
    """(b1, b2) for a segment along unit d: b1 level and at right angles to d, b2 at right angles to both, pointing up.
    (For a segment pointing straight up or down b1 is taken along the world's y.)"""
    b1 = np.cross(d, [0.0, 0.0, 1.0])
    if np.linalg.norm(b1) < 1e-6:
        b1 = np.array([0.0, 1.0, 0.0])
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(b1, d)
    return b1, b2 / np.linalg.norm(b2)


@dataclass
class CutPlan:
    """A way to cut a segment: the gripper's orientation R, the cut point, the point back along the approach (pre),
    the point on the lane in front of the plant (lane), the gripper command for the open jaws, and how it was found."""

    R: np.ndarray
    cut: np.ndarray
    pre: np.ndarray
    lane: np.ndarray
    opening: float
    phi: float  # radians (None for the existing expert's orientation)
    psi: float
    clearance: float  # least distance of the arm to the obstacles on the way, metres
    kind: str = "level"  # "legacy" (the existing expert's), "level" (|phi| or |phi - pi| < 45 deg), "above", "below"
    extra: dict = field(default_factory=dict)


class PandaArm:
    """The Franka Panda of env.build_model and its parallel gripper."""

    name = "panda"

    def __init__(self, env):
        self.env = env
        m = env.model
        self.data = mujoco.MjData(m)
        self.tcp = env.tcp_id
        self.lo, self.hi = env.ctrl_lo.copy(), env.ctrl_hi.copy()
        bodies = {m.body(n).id for n in CHECK_LINKS}
        self.geoms = [
            g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])
        ]
        self.data.qpos[:7] = HOME_Q
        mujoco.mj_kinematics(m, self.data)
        self.R_home = self.data.site_xmat[self.tcp].reshape(3, 3).copy()
        c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
        self.R_down = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ self.R_home  # as the expert's R_down
        self._fromto = np.zeros(6)
        self._eye6, self._eye7 = np.eye(6), np.eye(7)
        self._starts = {}

    # ------------------------------------------------------------------------------------------------ kinematics
    def ik(self, q_init, pos, R_des, iters=30, damping=0.05):
        """The expert's inverse kinematics (expert.ScriptedExpert.solve_ik with an orientation): damped least squares
        from q_init, a null-space bias toward the home posture, the joints clipped to their limits."""
        m, d, sid = self.env.model, self.data, self.tcp
        q = np.array(q_init, dtype=float)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[:7] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.site_xmat[sid].reshape(3, 3)
            err_p = pos - d.site_xpos[sid]
            # 0.5 * the sum of the columns' cross products (the expert's orientation error), written out: faster
            a, b = R, R_des
            err_r = 0.5 * np.array(
                [
                    (a[1] * b[2] - a[2] * b[1]).sum(),
                    (a[2] * b[0] - a[0] * b[2]).sum(),
                    (a[0] * b[1] - a[1] * b[0]).sum(),
                ]
            )
            if np.linalg.norm(err_p) < 2e-4 and np.linalg.norm(err_r) < 2e-3:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            J_pinv = J.T @ np.linalg.solve(J @ J.T + damping**2 * self._eye6, np.eye(6))
            dq = J_pinv @ np.concatenate([err_p, err_r]) + (self._eye7 - J_pinv @ J) @ (0.05 * (HOME_Q - q))
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    def pose(self, q):
        d = self.data
        d.qpos[:7] = q
        mujoco.mj_kinematics(self.env.model, d)
        return d.site_xpos[self.tcp].copy(), d.site_xmat[self.tcp].reshape(3, 3).copy()

    def within_limits(self, q, margin=LIMIT_MARGIN):
        return bool(np.all(q >= self.lo + margin) and np.all(q <= self.hi - margin))

    def tcp_velocity(self):
        """The gripper centre's linear velocity in the environment now (world frame, m/s)."""
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.env.model, self.env.data, mujoco.mjtObj.mjOBJ_SITE, self.tcp, v, 0)
        return v[3:]

    def settle_step(self, offset, target, R):
        """One control step of settling at `target` with the jaws closing along R's y axis: (the set-point's corrected
        offset, whether the hand has settled: its error along the closing axis under SETTLE_HEIGHT and its speed under
        SETTLE_SPEED). The offset grows by SETTLE_GAIN of the remaining error, at most SETTLE_MAX in all."""
        err = target - self.env.tcp_pos()
        settled = abs(err @ R[:, 1]) < SETTLE_HEIGHT and np.linalg.norm(self.tcp_velocity()) < SETTLE_SPEED
        if not settled:
            offset = offset + SETTLE_GAIN * err
            n = np.linalg.norm(offset)
            if n > SETTLE_MAX:
                offset = offset * (SETTLE_MAX / n)
        return offset, settled

    # ------------------------------------------------------------------------------------------------ the gripper
    @staticmethod
    def jaw_width(command):
        """Metres between the fingers at a gripper command (0 shut, 1 fully open)."""
        return 2 * FINGER_TRAVEL * command

    def cut_orientation(self, d, phi, psi, R_ref):
        """The gripper's orientation for a cut of a segment along d, approached at phi around it and leaning psi along
        it (see the module's docstring); of the two orientations (the jaws either way round), the nearer R_ref."""
        b1, b2 = segment_frame(d)
        a = np.cos(phi) * b1 + np.sin(phi) * b2
        z = np.cos(psi) * a + np.sin(psi) * d
        z /= np.linalg.norm(z)
        x = d - (d @ z) * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        cands = [np.stack([x, y, z], axis=1), np.stack([-x, -y, z], axis=1)]
        return min(cands, key=lambda R: _angle(R, R_ref))

    def legacy_orientation(self, toward):
        """The existing expert's cut orientation: level, pointing along `toward`, the jaws closing straight up and
        down; of the two, the nearer pointing down (expert.ScriptedExpert.prune_orientation)."""
        z = toward / np.linalg.norm(toward)
        cands = [np.stack([np.cross(y, z), y, z], axis=1) for y in (np.array([0.0, 0, 1]), np.array([0.0, 0, -1]))]
        return min(cands, key=lambda R: _angle(R, self.R_down))

    # ------------------------------------------------------------------------------------------------ checking
    def clearance(self, q, opening, obstacles, distmax=0.05):
        """Least clearance (metres, at most distmax) of the arm's collision shapes at joints q with the jaws open at
        `opening` to `obstacles`, {geom: margin}: the distance to each geom (MuJoCo's geom distance, negative inside)
        less its margin, so that 0 is the margin reached (the environment's current state for everything else)."""
        m, d = self.env.model, self.data
        d.qpos[:7] = q
        d.qpos[7] = d.qpos[8] = FINGER_TRAVEL * opening
        mujoco.mj_kinematics(m, d)
        best = distmax
        for o, margin in obstacles.items():
            for g in self.geoms:
                best = min(best, mujoco.mj_geomDistance(m, d, g, o, distmax + margin, self._fromto) - margin)
        return best

    def check(self, moves, pos, R, q0, opening, obstacles, stop_below=None, dt=0.05, turn_rate=0.08):
        """Follow `moves` [(target, R, speed m/s), ...] as the expert's waypoint follower does, from the set-point
        (pos, R) with the joints at q0: every control step (dt) the set-point moves up to speed * dt toward the target
        while its orientation turns up to turn_rate toward R, and the joints are solved from the last ones (30
        iterations, as the expert). Returns (least clearance on the way, the last joints), or None if a pose on the way
        is not reached or the clearance falls below `stop_below` (the check then stops early)."""
        q, worst = np.array(q0, dtype=float), np.inf
        self.data.qpos[:] = self.env.data.qpos  # the env's state (shoots, plate) once: only the arm moves
        pos, R = np.array(pos, dtype=float), R.copy()
        for target, Rm, speed in moves:
            for _ in range(2000):
                delta = target - pos
                dist = np.linalg.norm(delta)
                pos = target.copy() if dist <= speed * dt else pos + delta / dist * speed * dt
                R = _turn_toward(R, Rm, turn_rate)
                q = self.ik(q, pos, R, iters=30)
                px, Rq = self.pose(q)
                if np.linalg.norm(px - pos) > REACH_TOL or _angle(Rq, R) > REACH_TOL_ROT or not self.within_limits(q):
                    return None
                worst = min(worst, self.clearance(q, opening, obstacles))
                if stop_below is not None and worst < stop_below:
                    return None
                if dist <= speed * dt and _angle(R, Rm) < 1e-6:
                    break
        return worst, q

    # ------------------------------------------------------------------------------------------------ planning
    def plan_cut(
        self,
        a,
        d,
        cut_s,
        obstacles,
        lane_x,
        start,
        back=0.14,
        legacy_toward=None,
        kinds=None,
        pose_ok=None,
        slack=0.0,
    ):
        """A reachable, collision-free way to cut the segment from `a` along unit `d` at one of the distances `cut_s`
        (metres along it), keeping MARGIN from `obstacles`, or None. `start` = (position, orientation, joints) of the
        set-point the expert starts from. The way in, as the expert takes it (expert.ScriptedExpert._plan_reach): back
        to the lane's plane (x = lane_x, in front of the plant) at the start's height without turning, along the lane
        to the point level with the approach turning to the cut's orientation, to the point `back` metres back along
        the approach, in to the cut. Candidates, in order: the existing expert's orientation from either side
        (`legacy_toward`: the level directions it would take, best first), then every phi and psi of PHIS and PSIS; for
        each the widest opening of JAW_OPENINGS that keeps MARGIN at the cut, and the cut points in turn. `kinds`
        limits the candidates to those kinds ("legacy", "level", "above", "below"); `pose_ok(cut, R, opening)` may
        refuse a cut pose for reasons of the task's (the caller's). Returns the first candidate whose whole way keeps
        MARGIN, and `slack` more (a scenario's draw asks for some, so that the plan made later from the hand's own
        state, with the plant settled, finds a way too)."""
        pos0, R0, q0 = start
        cands = []
        for toward in legacy_toward or ():
            cands.append(("legacy", None, 0.0, self.legacy_orientation(toward)))
        for psi in PSIS:
            for phi in PHIS:
                cands.append((_kind(phi), phi, psi, self.cut_orientation(d, phi, psi, R0)))
        if kinds is not None:
            cands = [c for c in cands if c[0] in kinds]
        tried = 0
        for kind, phi, psi, R in cands:
            if abs(d @ R[:, 1]) > np.cos(np.radians(CROSS_MAX)):
                continue  # the jaws would close (nearly) along the segment
            z = R[:, 2]
            for s in cut_s:
                tried += 1
                cut = a + d * s
                q_cut = self.ik(q0, cut, R, iters=80)  # can the cut be reached at all, and is it clear?
                px, Rq = self.pose(q_cut)
                if (
                    np.linalg.norm(px - cut) > REACH_TOL
                    or _angle(Rq, R) > REACH_TOL_ROT
                    or not self.within_limits(q_cut)
                ):
                    break  # (the other cut points of this orientation are as far)
                self.data.qpos[:] = self.env.data.qpos
                openings = [o for o in JAW_OPENINGS if pose_ok is None or pose_ok(cut, R, o)]
                opening = next((o for o in openings if self.clearance(q_cut, o, obstacles) >= slack), None)
                if opening is None:
                    continue
                pre = cut - z * back
                lane = np.array([min(lane_x, pre[0]), pre[1], pre[2]])
                moves = []
                if pos0[0] > lane_x + 0.01:
                    moves.append((np.array([lane_x, pos0[1], pos0[2]]), R0, 0.30))
                moves += [(lane, R, 0.35), (pre, R, 0.25), (cut, R, 0.10)]
                # and back out, then to the canonical start: the arm must end in the posture it starts from there
                # (the inverse kinematics follows a branch from pose to pose; one plan that swings the elbow over
                # would leave every later plan, and the draw's promise, on another branch)
                home_pos, home_R, home_q = self.canonical_start(lane_x)
                moves += [(pre, R, 0.20), (lane, R, 0.30), (home_pos, home_R, 0.30)]
                out = self.check(moves, pos0, R0, q0, opening, obstacles, stop_below=slack)
                if out is None or np.max(np.abs(out[1] - home_q)) > BRANCH_TOL:
                    continue
                return CutPlan(
                    R=R,
                    cut=cut,
                    pre=pre,
                    lane=lane,
                    opening=opening,
                    phi=phi,
                    psi=psi,
                    clearance=out[0],
                    kind=kind,
                    extra=dict(s=float(s), tried=tried),
                )
        return None

    def canonical_start(self, lane_x):
        """A start for plans made without an arm to start from (a scenario's draw): on the lane's plane in front of the
        plant, 30 cm above the table, pointing down (the expert's R_down), the joints solved from the home posture."""
        key = round(float(lane_x), 6)
        if key not in self._starts:
            pos = np.array([lane_x, 0.0, 0.30])
            self._starts[key] = (pos, self.R_down.copy(), self.ik(HOME_Q, pos, self.R_down, iters=300))
        pos, R, q = self._starts[key]
        return pos.copy(), R.copy(), q.copy()


def _kind(phi):
    """'level' within 45 degrees of level, 'above' (the hand above the segment) or 'below'."""
    s = np.sin(phi)
    if abs(s) < np.sin(np.radians(45)):
        return "level"
    return "above" if s < 0 else "below"


ARMS = {"panda": PandaArm}


def arm_for(env) -> PandaArm:
    """The arm of an environment (built once per environment and kept on it)."""
    arm = getattr(env, "_arm", None)
    if arm is None:
        arm = env._arm = ARMS[getattr(env, "arm_name", "panda")](env)
    return arm
