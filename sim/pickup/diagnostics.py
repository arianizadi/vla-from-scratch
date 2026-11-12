"""Grasp diagnostics: where the gripper was, relative to the objects, when it first closed.

Success rates alone cannot tell "went to the wrong object" from "right object, missed by a
centimetre". `GraspProbe` records the gripper pose at the first close command of an episode;
`summarize` turns the per-episode records into rates and medians.
"""

import numpy as np

from .env import OBJECTS


def _wrap(a, period):
    return (a + period / 2) % period - period / 2


class GraspProbe:
    def __init__(self, env):
        self.env = env
        self.record = None

    def update(self, action):
        """Call with the action that is about to be executed."""
        if self.record is not None or float(action[7]) >= 0.5:
            return
        env = self.env
        tcp = env.tcp_pos()
        dist = {n: float(np.linalg.norm(tcp[:2] - env.object_pos(n)[:2])) for n in env.active}
        nearest = min(dist, key=dist.get)
        obj = OBJECTS[env.target]
        yaw_err = None
        if obj["grasp_sym"] is not None:
            finger_yaw = env.object_yaw(env.target) + obj["grasp_axis"]
            yaw_err = float(abs(_wrap(finger_yaw - env.tcp_yaw(), obj["grasp_sym"])))
        self.record = dict(
            close_step=int(env.steps),
            close_nearest=nearest,
            close_xy_err_target=dist[env.target],
            close_xy_err_nearest=dist[nearest],
            close_z_above_grasp=float(tcp[2] - max(env.object_pos(env.target)[2], 0.018)),
            close_yaw_err_target=yaw_err,
            # Table-plane positions (metres), enough to draw a top-down map of the attempt.
            close_tcp_xy=[float(tcp[0]), float(tcp[1])],
            close_object_xy={n: [float(v) for v in env.object_pos(n)[:2]] for n in env.active},
        )

    def result(self):
        if self.record is not None:
            return self.record
        return dict(
            close_step=None,
            close_nearest=None,
            close_xy_err_target=None,
            close_xy_err_nearest=None,
            close_z_above_grasp=None,
            close_yaw_err_target=None,
            close_tcp_xy=None,
            close_object_xy=None,
        )


def _fit(x, y):
    """Least-squares slope of y on x and the fraction of variance explained."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return None, None
    slope = float(np.polyfit(x, y, 1)[0])
    return slope, float(np.corrcoef(x, y)[0, 1] ** 2)


def summarize(episodes):
    """Aggregate GraspProbe records (dicts merged into the per-episode results)."""
    n = len(episodes)
    closed = [e for e in episodes if e["close_step"] is not None]
    at_target = [e for e in closed if e["close_nearest"] == e["target"]]

    def med(key, rows):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.median(vals)) if vals else None

    def frac_within(rows, limit):
        return sum(r["close_xy_err_target"] <= limit for r in rows) / n if n else None

    # Does the place where the gripper closed depend on where the named object was? Slope 1 and R2 near
    # 1: it goes to the object. Slope and R2 near 0: its position is independent of the scene.
    placed = [e for e in closed if e.get("close_tcp_xy") and e.get("close_object_xy")]
    fits = [
        _fit([e["close_object_xy"][e["target"]][axis] for e in placed], [e["close_tcp_xy"][axis] for e in placed])
        for axis in (0, 1)
    ]
    by_target = {}
    for target in sorted({e["target"] for e in closed}):
        rows = [e for e in closed if e["target"] == target]
        yaw = med("close_yaw_err_target", rows)
        by_target[target] = dict(
            episodes=len(rows),
            median_xy_err_mm=1000 * med("close_xy_err_target", rows),
            median_yaw_err_deg=None if yaw is None else float(np.degrees(yaw)),
            median_z_above_grasp_mm=1000 * med("close_z_above_grasp", rows),
        )

    return dict(
        closed_gripper_rate=len(closed) / n if n else None,
        closed_nearest_is_target_rate=len(at_target) / n if n else None,
        median_xy_err_target_mm=None if not closed else 1000 * med("close_xy_err_target", closed),
        median_xy_err_nearest_mm=None if not closed else 1000 * med("close_xy_err_nearest", closed),
        median_xy_err_when_at_target_mm=None if not at_target else 1000 * med("close_xy_err_target", at_target),
        median_z_above_grasp_mm=None if not closed else 1000 * med("close_z_above_grasp", closed),
        median_yaw_err_deg=None
        if med("close_yaw_err_target", at_target) is None
        else float(np.degrees(med("close_yaw_err_target", at_target))),
        within_10mm_of_target_rate=frac_within(closed, 0.010),
        within_20mm_of_target_rate=frac_within(closed, 0.020),
        within_30mm_of_target_rate=frac_within(closed, 0.030),
        close_vs_target_slope=[f[0] for f in fits],
        close_vs_target_r2=[f[1] for f in fits],
        by_target=by_target,
    )
