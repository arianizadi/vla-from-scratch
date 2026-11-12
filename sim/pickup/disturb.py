"""Disturbances: the named object is moved while the robot is working.

Two moments matter for a pick-up:

  reach  The object is moved while the gripper is still on its way. A policy that looks at the scene
         every time it plans can follow; one that committed to its first plan closes on empty table.
  grasp  The object is moved at the moment the gripper first closes, so the grasp finds nothing.
         Succeeding now takes a second attempt: open, go back up, approach again, close.

`Disturbance` applies one such move per episode. eval.py uses it to measure how a policy copes
(--disturb), collect.py to record demonstrations in which the expert recovers (--disturb_prob).
"""

import numpy as np

from .env import SPAWN_X, SPAWN_Y

MIN_OBJECT_GAP = 0.11  # metres between object centres, as in PickupEnv.reset
MIN_GRIPPER_GAP = 0.07  # the object is not put down under the open fingers


class Disturbance:
    def __init__(self, env, kind="none", step=15, distance=0.08, seed=0):
        assert kind in ("none", "reach", "grasp"), kind
        self.env, self.kind, self.step, self.distance = env, kind, step, distance
        self.rng = np.random.default_rng([seed, 11])  # its own stream, apart from the layout's
        self.record = None  # dict(step, kind, from_xy, to_xy) once the move has been attempted

    def update(self, action):
        """Call once per control step with the action about to be executed. True if the object moved."""
        env = self.env
        if self.kind == "none" or self.record is not None or not env.on_table(env.target):
            return False
        closing = float(action[7]) < 0.5
        if (self.kind == "reach" and env.steps < self.step) or (self.kind == "grasp" and not closing):
            return False
        old = env.object_pos(env.target)[:2]
        tcp = env.tcp_pos()[:2]
        others = [env.object_pos(n)[:2] for n in env.active if n != env.target]
        self.record = dict(step=int(env.steps), kind=self.kind, from_xy=[float(v) for v in old], to_xy=None)
        for _ in range(64):
            angle = self.rng.uniform(-np.pi, np.pi)
            new = old + self.distance * np.array([np.cos(angle), np.sin(angle)])
            inside = SPAWN_X[0] <= new[0] <= SPAWN_X[1] and SPAWN_Y[0] <= new[1] <= SPAWN_Y[1]
            clear = all(np.linalg.norm(new - p) >= MIN_OBJECT_GAP for p in others)
            if inside and clear and np.linalg.norm(new - tcp) >= MIN_GRIPPER_GAP:
                env.move_object(env.target, new, yaw=env.object_yaw(env.target) + self.rng.uniform(-0.5, 0.5))
                self.record["to_xy"] = [float(v) for v in new]
                return True
        return False  # no free place at this distance: the episode runs undisturbed
