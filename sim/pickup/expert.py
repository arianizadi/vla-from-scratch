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


class ScriptedExpert:
    def __init__(self, env: PickupEnv, seed: int = 0):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.ik_data = mujoco.MjData(env.model)
        # Reference top-down orientation and finger-axis yaw at the home configuration.
        self.ik_data.qpos[:7] = HOME_Q
        mujoco.mj_kinematics(env.model, self.ik_data)
        self.R_home = self.ik_data.site_xmat[env.tcp_id].reshape(3, 3).copy()
        self.yaw_home = float(np.arctan2(self.R_home[1, 1], self.R_home[0, 1]))

    # --------------------------------------------------------------------- IK
    def solve_ik(self, q_init, pos, yaw_offset, iters=30, damping=0.05):
        m, d, sid = self.env.model, self.ik_data, self.env.tcp_id
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
    def reset(self):
        env = self.env
        obj = OBJECTS[env.target]
        p = env.object_pos(env.target)
        if obj["grasp_sym"] is None:
            yaw_off = self.rng.uniform(-0.3, 0.3)
        else:
            finger_yaw = env.object_yaw(env.target) + obj["grasp_axis"]
            yaw_off = _wrap(finger_yaw - self.yaw_home, obj["grasp_sym"])
        xy_noise = self.rng.normal(0, 0.003, 2)
        grasp = np.array([p[0] + xy_noise[0], p[1] + xy_noise[1], max(p[2], 0.018)])
        speed = self.rng.uniform(0.8, 1.2)
        # (position, gripper, max speed m/s, dwell steps after arrival)
        self.waypoints = [
            (grasp + [0, 0, self.rng.uniform(0.10, 0.14)], 1.0, 0.40 * speed, 0),
            (grasp, 1.0, 0.18 * speed, 2),
            (grasp, 0.0, 0.0, 8),
            (grasp + [0, 0, self.rng.uniform(0.18, 0.24)], 0.0, 0.30 * speed, 1000),
        ]
        self.wp = 0
        self.dwell = 0
        self.yaw_off = yaw_off
        self.des_pos = env.tcp_pos()
        self.des_yaw = 0.0
        self.q_cmd = env.data.ctrl[:7].copy()

    def act(self):
        env = self.env
        target, grip, speed, dwell = self.waypoints[self.wp]
        dt = 1.0 / 20.0
        delta = target - self.des_pos
        dist = np.linalg.norm(delta)
        step = speed * dt
        self.des_pos = target.copy() if dist <= step or speed == 0 else self.des_pos + delta / dist * step
        dyaw = self.yaw_off - self.des_yaw
        self.des_yaw += np.clip(dyaw, -0.12, 0.12)

        arrived = (
            np.linalg.norm(target - self.des_pos) < 1e-6
            and np.linalg.norm(env.tcp_pos() - target) < 0.012
            and abs(self.yaw_off - self.des_yaw) < 1e-6
        )
        if arrived or speed == 0:
            self.dwell += 1
            if self.dwell > dwell and self.wp < len(self.waypoints) - 1:
                self.wp += 1
                self.dwell = 0
        self.q_cmd = self.solve_ik(self.q_cmd, self.des_pos, self.des_yaw)
        return np.concatenate([self.q_cmd, [grip]]).astype(np.float32)
