"""Disturbances: the named object is moved while the robot is working.

Two moments matter for a pick-up, and one kind of motion:

  reach  The object is moved while the gripper is still on its way. A policy that looks at the scene
         every time it plans can follow; one that committed to its first plan closes on empty table.
  grasp  The object is moved at the moment the gripper first closes, so the grasp finds nothing.
         Succeeding now takes a second attempt: open, go back up, approach again, close.
  drift  The object keeps moving: from the first step it slides along a smooth random path at a set
         speed, as a stem sways, and stops while the gripper is closed. A target that never stands still
         has to be tracked, not found once.

`Disturbance` applies one jump per episode ('reach', 'grasp') or the drift. eval.py uses it to measure how
a policy copes (--disturb), collect.py to record demonstrations in which the expert recovers
(--disturb_prob, --drift_prob).
"""

import numpy as np

from .env import SPAWN_X, SPAWN_Y

MIN_OBJECT_GAP = 0.11  # metres between object centres, as in PickupEnv.reset
MIN_GRIPPER_GAP = 0.07  # the object is not put down under the open fingers


DRIFT_TURN = 1.2  # rad/s: standard deviation of the drift's change of heading, per second of motion


class Disturbance:
    def __init__(self, env, kind="none", step=15, distance=0.08, seed=0, speed=0.02):
        """`speed` (m/s) is used by 'drift' only; `step` and `distance` by 'reach' and 'grasp'."""
        assert kind in ("none", "reach", "grasp", "drift"), kind
        self.env, self.kind, self.step, self.distance, self.speed = env, kind, step, distance, speed
        self.rng = np.random.default_rng([seed, 11])  # its own stream, apart from the layout's
        # dict(step, kind, from_xy, to_xy) once the move has been attempted; for 'drift', also speed and
        # path_m (metres travelled), with to_xy updated as the object moves
        self.record = None
        self.heading = None

    def update(self, action):
        """Call once per control step with the action about to be executed. True if the object moved."""
        env = self.env
        if self.kind == "drift":
            return self._drift(action)
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

    def _drift(self, action):
        env = self.env
        if not env.on_table(env.target) or float(action[7]) < 0.5:
            return False  # lifted, or held: a stem in the gripper does not sway away
        old = env.object_pos(env.target)[:2]
        if self.record is None:
            self.heading = self.rng.uniform(-np.pi, np.pi)
            self.record = dict(
                step=int(env.steps),
                kind="drift",
                from_xy=[float(v) for v in old],
                to_xy=None,
                speed=float(self.speed),
                path_m=0.0,
            )
        dt = 1.0 / 20.0
        self.heading += self.rng.normal(0, DRIFT_TURN * np.sqrt(dt))
        others = [env.object_pos(n)[:2] for n in env.active if n != env.target]
        for turn in (0.0, np.pi / 2, -np.pi / 2, np.pi):  # turn away from the edge of the area and from the others
            heading = self.heading + turn
            new = old + self.speed * dt * np.array([np.cos(heading), np.sin(heading)])
            inside = SPAWN_X[0] <= new[0] <= SPAWN_X[1] and SPAWN_Y[0] <= new[1] <= SPAWN_Y[1]
            if inside and all(np.linalg.norm(new - p) >= MIN_OBJECT_GAP for p in others):
                self.heading = heading
                env.move_object(env.target, new)
                self.record["to_xy"] = [float(v) for v in new]
                self.record["path_m"] += float(self.speed * dt)
                return True
        return False  # boxed in this step: stand still
