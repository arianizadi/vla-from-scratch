"""Situations that need the arm's every axis: the plant cannot be turned, the shoots sit at many heights and pitches,
and something may block one side of a cut (tutorial/test-situations.md, "Full 3D reach").

In the base task the turntable brings each shoot round to the front, so the arm repeats one level reach from the front
at a few heights. Here:

  locked    R1  no spinning: the buttons are gone and the plate is locked. The shoots point all round (each at least
                SHOOT_SPREAD from the next, as in prune), so the arm reaches round the main stem: from the sides,
                diagonally behind, from above or below, turning the wrist, without touching the flower or the stem.
  heights   R2  the plate locked as in `locked`, and the shoots spread from just above the pot to just under the flower,
                on plants a fifth shorter to a fifth taller; a cut counts only with the jaws' centre within HEIGHT_BAND
                of the shoot's axis in height (across the shoot, in its vertical plane).
  pitched   R3  the plate locked, and the shoots pitched steeply up or down (PITCH): jaws that close straight up and
                down would close along such a shoot, so the wrist must pitch with it.
  blocked   R4  the plate locked, and a large leaf lies flat just above or just below each shoot's cut zone: the arm
                comes in from the clear side (from below, or from above).
  height_named R5  the plant of `heights`, and the instruction names one shoot by its height: "cut the shoot just below
                the flower", "the one nearest the pot".

Every layout is drawn so that each shoot can be cut: the arm (arm.py, `arm_for(env).plan_cut`) must find a reachable,
collision-free way in to it with the plant as drawn; a draw where it cannot is drawn again from the same stream
(DRAW_TRIES), and how many draws it took is recorded. The scenarios here say only what the plant looks like, what is in
the way and what counts as success; reach, joint limits, the wrist's orientation and the approach are the arm's
(arm.py). They change the task (one at a time, like the A-situations): the instruction stays the task's own.
env.flags holds 'reach3d' (the expert plans every cut with the arm's planner and never presses a button) and
'locked'.
"""

import mujoco
import numpy as np

from . import env as E
from . import scenarios as S
from .arm import MARGIN, arm_for
from .scenarios import Scenario, _obj_pose, _show

DRAW_TRIES = 400
REDRAW_PLANT = 40  # draws after which a scenario that keeps prune's heights and elevations draws them afresh too
LOCK_DAMPING = 1e4  # N m s / rad on the plate's hinge: the plate does not turn under the arm
# The expert's cut points, metres along a shoot from the main stem's axis (the cut zone is CUT_ZONE, 20 to 50 mm): first
# the existing expert's draw (31 to 39 mm), then farther out, where a steep shoot leaves more room from the stem; at
# most 8 mm short of the zone's end, which the hand overshoots by up to 5 mm on a shoot pointing down (it sags)
CUT_S = (0.035, 0.042)


GROUND_MARGIN = 0.025  # metres from the table, the stand and the plate: the arm sags and lags low down
POT_MARGIN = 0.012  # metres from the pot and its rim


def plant_obstacles(env, target=None, extra=()):
    """What a way in to a cut of shoot `target` must keep clear of, {geom: margin}: the main stem and the flower, every
    other uncut shoot with its stub and leaves, the target's own stub and leaves, `extra` (a scenario's obstacles) at
    arm.MARGIN; the pot and its rim at POT_MARGIN; the table, the stand (and its buttons where they are there) and the
    plate at GROUND_MARGIN."""
    m = env.model
    ground = ["plate_col", "stand_col0", "stand_col1", "table"]
    out = {m.geom(n).id: GROUND_MARGIN for n in ground}
    out.update({m.geom(n).id: POT_MARGIN for n in ["pot_col0", "pot_col1", *(f"pot_rim{i}" for i in range(16))]})
    for b in E.BUTTONS:
        g = m.geom(f"button_{b}_col").id
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            out[g] = GROUND_MARGIN
    plant = [m.geom("stem_col").id, m.geom("flower_col").id]
    for n in env.active:
        leaves = [col for _, col in env.leaf_geoms[n] if m.geom_contype[col]]
        if n == target:
            plant += [env.stub_geoms[n][0], *leaves]
        elif n not in env.cut:
            plant += [env.outer_geoms[n][0], env.stub_geoms[n][0], *leaves]
        else:
            plant.append(env.stub_geoms[n][0])
    out.update({g: MARGIN for g in [*plant, *extra]})
    return out


def scenario_obstacles(env):
    """The obstacle geoms the scenarios in use add this episode (Scenario.obstacles)."""
    return [g for s in env.scenarios if s.applied for g in getattr(s, "obstacles", lambda env: [])(env)]


def lane_x(env):
    from .expert import LANE_BACK

    return float(env.data.xpos[env.stand_body][0] - LANE_BACK)


STEM_CLEAR = 0.004  # metres beyond the closing jaws' reach (and their depth) the main stem must stay at a cut


def stem_clear_of_jaws(env):
    """A check of a cut pose (pos, R, opening) for arm.plan_cut: the main stem is not between the jaws there, as
    env._segment_held would find it (its axis within half the opening along the closing axis and within the fingertips'
    depth, crossing the jaws), with STEM_CLEAR to spare. Jaws that shut on a shoot right next to the stem would count
    as cutting the stem."""
    base, up = env.stem_axis()

    def ok(pos, R, opening):
        s = float((pos - base) @ up)
        if not 0.0 <= s <= env.stem_h:
            return True
        rel = R.T @ (base + s * up - pos)
        width = arm_for(env).jaw_width(opening)
        crossing = abs(up @ R[:, 1]) <= np.cos(np.radians(45))
        return not (crossing and abs(rel[1]) < width / 2 + STEM_CLEAR and abs(rel[2]) < 0.015 + STEM_CLEAR)

    return ok


DRAW_SLACK = 0.003  # metres of clearance a draw asks for beyond what the expert's own plan needs


def plan_cut(env, name, start=None, cut_s=CUT_S, legacy=True, kinds=None, slack=0.0):
    """The arm's way in to a cut of shoot `name` with the plant as it is now, from `start` (set-point position,
    orientation, joints; default the arm's canonical start), or None (arm.PandaArm.plan_cut). With `legacy` the
    existing expert's level orientations are tried first (the one pointing less outward first, as it prefers)."""
    arm = arm_for(env)
    a, d = env.shoot_axis(name)
    toward = []
    if legacy:
        u = d[:2] / max(np.linalg.norm(d[:2]), 1e-9)
        toward = sorted((side * np.array([-u[1], u[0], 0.0]) for side in (1, -1)), key=lambda t: t[0])
    obstacles = plant_obstacles(env, name, scenario_obstacles(env))
    start = arm.canonical_start(lane_x(env)) if start is None else start
    return arm.plan_cut(
        a,
        d,
        cut_s,
        obstacles,
        lane_x(env),
        start,
        legacy_toward=toward,
        kinds=kinds,
        pose_ok=stem_clear_of_jaws(env),
        slack=slack,
    )


def regrow(env, name, height, azimuth, elevation, length=None):
    """Grow shoot `name` again: leaving the main stem `height` above its base, pointing to world `azimuth` (seen from
    above) at `elevation`, `length` long (default: as drawn). Its leaves keep their pose at the tip. Call
    env._place_shoots() once every shoot is grown."""
    m = env.model
    shoot = env.shoots[name]
    old_tip = shoot["length"] - E.CUT_MID
    a = azimuth - env.plate_angle()  # in the plant's frame (it turns with the plate)
    direction = np.array([np.cos(a) * np.cos(elevation), np.sin(a) * np.cos(elevation), np.sin(elevation)])
    shoot["origin"] = np.array([0.0, 0.0, E.STEM_BASE + height])
    shoot["dir"], shoot["frame"] = direction, E._shoot_frame(direction)
    if length is not None:
        shoot["length"] = float(length)
    env._shape_shoot(name, E.SHOOT_SPLIT, E.SHOOT_SPLIT)
    shift = shoot["length"] - E.CUT_MID - old_tip
    for vis, col in env.leaf_geoms[name]:
        m.geom_pos[vis][2] += shift
        m.geom_pos[col][2] += shift
    data = m.eq_data[env.shoot_weld[name]]
    data[:3], data[3:6], data[6:10] = shoot["origin"] + direction * E.CUT_MID, 0, E._quat(shoot["frame"].T)


def shoot_tip_clear(env, name, stem_h=E.STEM_H):
    """Is the outer end of shoot `name` (with its leaves' reach, about 6 cm beyond) clear of the pot and soil below and
    of the flower above? A rough check in the plant's frame."""
    s = env.shoots[name]
    tip = s["origin"] + s["dir"] * (s["length"] + 0.03)
    r = np.linalg.norm(tip[:2])
    soil = 0.0685 + 0.012
    if tip[2] < soil + 0.02 and r < 0.06:
        return False
    top = E.STEM_BASE + stem_h
    if tip[2] > top - 0.035 and r < 0.06:
        return False
    return tip[2] > 0.02


class Reach(Scenario):
    """Base of the situations here: the buttons taken away and the plate locked, the shoots drawn again from this
    scenario's stream until the arm can cut every one (reshape, then arm.plan_cut for each), the task's success rule
    (with a check of each cut, `cut_ok`)."""

    changes_task = True
    prune_program = True  # what to cut is prune's (program.from_scenario: Cut(All()))
    catalogue = ""

    def check(self, env):
        super().check(env)
        if env.named or env.reason or env.task != "prune":
            raise ValueError(f"{self.name} applies to prune, not {env.task}")

    def init(self, env):
        m = env.model
        self.button_geoms = {
            b: (m.geom(f"button_{b}_col").id, m.geom(f"button_{b}_vis").id, m.geom(f"button_{b}_arrow").id)
            for b in E.BUTTONS
        }
        self.button_bodies = {b: m.body(f"button_{b}").id for b in E.BUTTONS}
        self.hinge = m.joint("plate_hinge").dofadr[0]
        self.defaults = dict(
            damping=float(m.dof_damping[self.hinge]),
            contype={
                b: (int(m.geom_contype[g[0]]), int(m.geom_conaffinity[g[0]])) for b, g in self.button_geoms.items()
            },
        )

    def restore(self, env):
        m = env.model
        m.dof_damping[self.hinge] = self.defaults["damping"]
        for b, (col, vis, arrow) in self.button_geoms.items():
            m.geom_contype[col], m.geom_conaffinity[col] = self.defaults["contype"][b]
            _show(m, [vis, arrow], True)
        self.checks = {}

    def lock(self, env):
        """No buttons and no turning: the caps are taken away (not drawn, not solid), the plate held on its hinge."""
        m = env.model
        m.dof_damping[self.hinge] = LOCK_DAMPING
        for col, vis, arrow in self.button_geoms.values():
            m.geom_contype[col] = m.geom_conaffinity[col] = 0
            _show(m, [vis, arrow], False)

    def reshape(self, env, rng):
        """Draw this episode's shoots again (regrow each) and anything else of the plant; returns the variation."""
        raise NotImplementedError

    def draw(self, env, rng, mode):
        self.lock(env)
        env.flags |= {"reach3d", "locked"}
        self.mode = mode
        tries = self.tries = 0
        plans = {}
        while tries < DRAW_TRIES:
            tries = self.tries = tries + 1
            out = self.reshape(env, rng)
            env._place_shoots()
            mujoco.mj_kinematics(env.model, env.data)
            if getattr(self, "unreachable", False):  # the reshape found the layout of no use (blocked: no leaf)
                continue
            plans = {}
            # the shoots pointing farthest from the robot first: the likeliest to have no way in, so a draw fails fast
            for n in sorted(env.active, key=lambda n: -abs(E._wrap_angle(E.REACH_AZIMUTH - env.shoot_azimuth(n)))):
                plan = plan_cut(env, n, slack=DRAW_SLACK)
                if plan is None:
                    break
                plans[n] = plan
            if len(plans) == len(env.active):
                break
        else:
            # no draw in DRAW_TRIES has a way in to every shoot (one pitched layout in 100): keep the last draw and
            # take off the plant the shoots that have none (the plant keeps at least one shoot)
            if getattr(self, "unreachable", False):  # (the last draw was refused before it was planned: plan it now)
                plans = {n: p for n in env.active if (p := plan_cut(env, n, slack=DRAW_SLACK)) is not None}
            dropped = [n for n in env.active if n not in plans]
            if len(dropped) == len(env.active):
                raise RuntimeError(f"{self.name}: no shoot of the layout the arm can cut in {DRAW_TRIES} draws")
            for n in dropped:
                env._park_shoot(n)
                del env.shoots[n]
            env.active = [n for n in env.active if n in plans]
            env.target = env.active[0]
            if "targets" in env.flags:
                env.targets = [n for n in env.targets if n in plans] or env.targets
            env._set_exclusions()
            mujoco.mj_kinematics(env.model, env.data)
            out = dict(out, dropped=len(dropped))
        kinds = [plans[n].kind for n in env.active]
        return dict(out, draws=tries, approach=kinds)

    def cut_ok(self, env, name):
        """Does the cut of shoot `name` just made count? (Every cut does, unless a scenario says otherwise.)"""
        return True

    def result(self, env, done, info):
        for n in env.cut:
            if n not in self.checks:
                self.checks[n] = dict(cut_error(env, n), ok=bool(self.cut_ok(env, n)))
        bad = sorted(n for n, c in self.checks.items() if not c["ok"])
        if bad:
            info = dict(info, success=False, cut_off_band=bad)
            done = True
        return done, dict(info, cut_checks={n: dict(c) for n, c in self.checks.items()})


def cut_error(env, name) -> dict:
    """Where the jaws' centre is against shoot `name`'s axis as grown, in mm: along the shoot from the main stem's axis;
    across it, split into height (z: along b2, at right angles to the shoot in its vertical plane, + up) and lateral
    (lat: along b1, level and at right angles to the shoot); and the same offset in the hand's frame (cut_diag.py's):
    along the closing jaws (jaws), across them (side) and the way the hand points (depth)."""
    from .arm import segment_frame

    a, d = env.shoot_axis(name)
    v = env.tcp_pos() - a
    along = float(v @ d)
    v = v - along * d
    b1, b2 = segment_frame(d)
    h = env.data.site_xmat[env.tcp_id].reshape(3, 3).T @ v

    def mm(x):
        return round(1000 * float(x), 1)

    return dict(
        along_mm=mm(along), z_mm=mm(v @ b2), lat_mm=mm(v @ b1), jaws_mm=mm(h[1]), side_mm=mm(h[0]), depth_mm=mm(h[2])
    )


def _prune_shape(env, rng, tries):
    """(heights above the stem's base, elevations) of the shoots: the ones prune drew for the seed, or after
    REDRAW_PLANT failed draws new ones from prune's ranges (SHOOT_HEIGHTS, SHOOT_GAP, SHOOT_ELEVATION)."""
    if tries <= REDRAW_PLANT:
        hs = [float(env.shoots[n]["origin"][2] - E.STEM_BASE) for n in env.active]
        es = [float(np.arcsin(np.clip(env.shoots[n]["dir"][2], -1, 1))) for n in env.active]
        return np.array(hs), np.array(es)
    n = len(env.active)
    free = E.SHOOT_HEIGHTS[1] - E.SHOOT_HEIGHTS[0] - (n - 1) * E.SHOOT_GAP
    hs = E.SHOOT_HEIGHTS[0] + np.sort(rng.uniform(0, free, n)) + E.SHOOT_GAP * np.arange(n)
    return hs, rng.uniform(*E.SHOOT_ELEVATION, n)


def _spread_azimuths(rng, n):
    for _ in range(1000):
        az = rng.uniform(-np.pi, np.pi, n)
        gaps = [abs(E._wrap_angle(a - b)) for i, a in enumerate(az) for b in az[i + 1 :]]
        if min(gaps, default=np.pi) >= E.SHOOT_SPREAD:
            return az
    return az


# ------------------------------------------------------------------------------------------------- R1 locked plate
class Locked(Reach):
    """R1, no spinning: the plant as prune draws it (heights, elevations, lengths, leaves), each shoot pointing to a
    world direction drawn all round (at least SHOOT_SPREAD apart), the plate locked and the buttons gone."""

    name, catalogue = "locked", "R1"
    modes = ("locked",)

    def reshape(self, env, rng):
        az = _spread_azimuths(rng, len(env.active))
        heights, elevations = _prune_shape(env, rng, self.tries)
        for n, a, h, e in zip(env.active, az, heights, elevations, strict=True):
            regrow(env, n, h, a, e)
        return dict(azimuths_deg=np.degrees(az), heights_m=heights, elevations_deg=np.degrees(elevations))


# ----------------------------------------------------------------------------------------------- R2 height spread
STEM_SCALES = (0.8, 0.9, 1.0, 1.1, 1.2)  # the main stem's length against prune's (STEM_H, 24 cm): 19 to 29 cm
LOWEST, BELOW_FLOWER = 0.03, 0.065  # metres: the lowest shoot above the stem's base, the highest below its top
HEIGHT_BAND = 0.010  # metres: a cut counts with the jaws' centre this near the shoot's axis in height


class Heights(Reach):
    """R2, height spread: the plate locked and the buttons gone (as `locked`), the plant a fifth shorter to a fifth
    taller (STEM_SCALES: the stem and the flower on it), and the shoots anywhere from LOWEST above the stem's base (just
    above the pot) to BELOW_FLOWER under its top (just under the flower), at least SHOOT_GAP apart, pointing all round.
    A cut counts only with the jaws' centre within HEIGHT_BAND of the shoot's axis in height (across the shoot, in its
    vertical plane: cut_error's z); a cut outside the band ends the episode as a failure (info['cut_off_band'])."""

    name, catalogue = "heights", "R2"
    modes = ("heights",)

    def build(self, spec):
        for k in STEM_SCALES:
            if k != 1.0:
                mesh = spec.add_mesh()
                mesh.name, mesh.file = f"plant_main_stem_x{k:g}", str(E.ASSETS / "meshes" / "plant_main_stem.obj")
                mesh.scale = [1.0, 1.0, k]

    def init(self, env):
        super().init(env)
        m = env.model
        self.stem_vis, self.stem_col = m.geom("stem_vis").id, m.geom("stem_col").id
        self.meshes = {k: m.mesh("plant_main_stem" if k == 1.0 else f"plant_main_stem_x{k:g}").id for k in STEM_SCALES}
        self.flower = [m.geom(f"flower_{p}").id for p in ("petals", "centre", "sepals")] + [m.geom("flower_col").id]
        geoms = [self.stem_vis, self.stem_col, *self.flower]
        self.stem_defaults = {
            f: getattr(m, f)[geoms].copy() for f in ("geom_pos", "geom_size", "geom_rbound", "geom_aabb", "geom_dataid")
        }
        self.stem_geoms = geoms
        m.geom_sameframe[geoms] = 0

    def restore(self, env):
        super().restore(env)
        m = env.model
        for f, v in self.stem_defaults.items():
            getattr(m, f)[self.stem_geoms] = v
        env.stem_h = E.STEM_H

    def grow_stem(self, env, k):
        """The main stem k times prune's length, the flower on its top."""
        m = env.model
        for f, v in self.stem_defaults.items():
            getattr(m, f)[self.stem_geoms] = v
        h = k * E.STEM_H
        m.geom_dataid[self.stem_vis] = self.meshes[k]
        _obj_pose(m, self.stem_vis, np.array([0.0, 0.0, E.STEM_BASE]), np.eye(3))
        m.geom_rbound[self.stem_vis] *= k
        m.geom_aabb[self.stem_vis][3:] *= [1.0, 1.0, k]
        m.geom_aabb[self.stem_vis][2] *= k
        m.geom_size[self.stem_col][1] = h / 2
        m.geom_pos[self.stem_col][2] = E.STEM_BASE + h / 2
        m.geom_rbound[self.stem_col] = E.STEM_R + h / 2
        m.geom_aabb[self.stem_col][5] = h / 2 + E.STEM_R
        for g in self.flower:
            m.geom_pos[g][2] += h - E.STEM_H
        env.stem_h = h

    def reshape(self, env, rng):
        k = STEM_SCALES[int(rng.integers(len(STEM_SCALES)))]
        self.grow_stem(env, k)
        n = len(env.active)
        hi = env.stem_h - BELOW_FLOWER
        free = hi - LOWEST - (n - 1) * E.SHOOT_GAP
        heights = LOWEST + np.sort(rng.uniform(0, free, n)) + E.SHOOT_GAP * np.arange(n)
        az = _spread_azimuths(rng, n)
        for name, h, a in zip(env.active, heights, az, strict=True):
            e = float(np.arcsin(np.clip(env.shoots[name]["dir"][2], -1, 1)))  # the elevation prune drew
            regrow(env, name, h, a, e)
        return dict(stem_scale=k, heights_m=heights, azimuths_deg=np.degrees(az))

    def cut_ok(self, env, name):
        return abs(cut_error(env, name)["z_mm"]) <= 1000 * HEIGHT_BAND


# --------------------------------------------------------------------------------------- R5 a height in the instruction
HEIGHT_FRAMES = {
    "below_flower": (
        "cut the shoot just below the flower",
        "prune the side shoot right under the flower",
        "remove the shoot nearest the flower",
    ),
    "near_pot": (
        "cut the shoot nearest the pot",
        "prune the lowest side shoot, just above the pot",
        "remove the shoot closest to the soil",
    ),
}


class HeightNamed(Heights):
    """R5, a height in the instruction: the plant of `heights` (the plate locked, the shoots from just above the pot
    to just under the flower, the height band), and the instruction names one shoot by where it is on the stem: the
    one just below the flower (the highest) or the one nearest the pot (the lowest). Success: that shoot cut within
    the band, no other shoot cut, no damage; cutting another ends the episode (wrong_object). env.targets holds the
    shoot."""

    name, catalogue = "height_named", "R5"
    modes = ("below_flower", "near_pot")
    prune_program = False  # program.from_scenario: the named shoot's program

    def draw(self, env, rng, mode):
        out = super().draw(env, rng, mode)
        target = env.active[-1] if mode == "below_flower" else env.active[0]
        frames = HEIGHT_FRAMES[mode]
        env.instruction = frames[int(rng.integers(len(frames)))]
        env.targets, env.target = [target], target
        env.target_label = "highest" if mode == "below_flower" else "lowest"
        env.flags.add("targets")
        return dict(out, instruction=env.instruction, target=env.target_label)

    def result(self, env, done, info):
        done, info = super().result(env, done, info)
        wrong = sorted(n for n in env.cut if n not in env.targets)
        success = env.damaged is None and not wrong and all(n in env.cut for n in env.targets)
        success = success and not info.get("cut_off_band")
        done = done or success or bool(wrong) or env.damaged is not None or env.steps >= env.max_steps
        return done, dict(
            info, success=success, wrong_object=bool(wrong) or env.damaged is not None, wrong_shoots=wrong
        )


# ------------------------------------------------------------------------------------------------ R3 pitched shoots
PITCH = (np.radians(40), np.radians(60))  # the shoots' elevation, up or down (prune: 15 to 35 degrees, up)
PITCH_HEIGHTS = (0.05, 0.17)  # metres above the stem's base


class Pitched(Reach):
    """R3, pitched shoots: the plate locked and the buttons gone, every shoot pitched steeply up or down (PITCH, the
    sign drawn per shoot), at heights in PITCH_HEIGHTS (at least SHOOT_GAP apart) where its far end, leaves and all,
    stays clear of the pot and of the flower (shoot_tip_clear), pointing all round. Jaws that close straight up and
    down would close along such a shoot (env._segment_held needs it within 45 degrees of square across them), so the
    wrist pitches with the shoot."""

    name, catalogue = "pitched", "R3"
    modes = ("up_down",)

    def reshape(self, env, rng):
        n = len(env.active)
        for _ in range(200):
            free = PITCH_HEIGHTS[1] - PITCH_HEIGHTS[0] - (n - 1) * E.SHOOT_GAP
            heights = PITCH_HEIGHTS[0] + np.sort(rng.uniform(0, free, n)) + E.SHOOT_GAP * np.arange(n)
            signs = rng.choice([-1.0, 1.0], n)
            pitch = signs * rng.uniform(*PITCH, n)
            az = _spread_azimuths(rng, n)
            for name, h, a, e in zip(env.active, heights, az, pitch, strict=True):
                regrow(env, name, h, a, e)
            if all(shoot_tip_clear(env, name) for name in env.active):
                break
        return dict(heights_m=heights, pitch_deg=np.degrees(pitch), azimuths_deg=np.degrees(az))


# ---------------------------------------------------------------------------------------- R4 blocked above or below
BLOCK_LEAF = (0.040, 0.022, 0.0015)  # semi-axes of the blocking leaf's blade (an ellipsoid): along, across, thick
BLOCK_FROM = 0.015  # metres out along the shoot where the blade starts (it then covers the whole cut zone)
BLOCK_GAP = (0.024, 0.027)  # metres from the shoot's axis to the near face of the blade
BLOCK_RGBA = (0.16, 0.46, 0.12, 1)


class Blocked(Reach):
    """R4, blocked above or below: the plate locked and the buttons gone, the shoots as prune draws them pointing all
    round, and over or under a shoot a large leaf growing from the main stem, its blade lying along the shoot BLOCK_GAP
    from its axis over the whole cut zone: jaws closing up and down would meet it (their fingers reach 2 to 3 cm above
    and below the shoot), so the arm comes in from the clear side, the jaws closing level: from below a leaf above, from
    above a leaf below. The side is drawn per shoot among the sides the arm has such a way in from (arm.plan_cut with
    the hand below the shoot or above it, before the leaf is there); a shoot with neither has no leaf, and a draw with
    no leaf at all is drawn again. Touching a blocking leaf damages the plant ('leaf touched')."""

    name, catalogue = "blocked", "R4"
    modes = ("above_below",)

    def build(self, spec):
        plant = spec.body("plant")
        for k in range(E.MAX_SHOOTS):
            g = plant.add_geom(
                name=f"block{k}_blade",
                type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                size=list(BLOCK_LEAF),
                pos=[0.3, 0.0, 0.02 + 0.01 * k],
                rgba=list(BLOCK_RGBA),
                group=3,
                contype=0,
                conaffinity=0,
                density=300,
                friction=[1.5, 0.01, 0.0002],
                condim=4,
            )
            g.solref, g.solimp = list(E.CONTACT_SOLREF), list(E.CONTACT_SOLIMP)
            plant.add_geom(
                name=f"block{k}_petiole",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[0.0018, BLOCK_FROM / 2, 0],
                rgba=list(BLOCK_RGBA),
                group=3,
                contype=0,
                conaffinity=0,
                density=0,
            )

    def init(self, env):
        super().init(env)
        m = env.model
        self.blades = [m.geom(f"block{k}_blade").id for k in range(E.MAX_SHOOTS)]
        self.petioles = [m.geom(f"block{k}_petiole").id for k in range(E.MAX_SHOOTS)]
        m.geom_sameframe[self.blades + self.petioles] = 0
        self.robot = set(env.robot_bodies.tolist())

    def restore(self, env):
        super().restore(env)
        self.restore_leaves(env)

    def obstacles(self, env):
        return list(self.used)

    def reshape(self, env, rng):
        m = env.model
        self.restore_leaves(env)
        az = _spread_azimuths(rng, len(env.active))
        heights, elevations = _prune_shape(env, rng, self.tries)
        for name, a, h, e in zip(env.active, az, heights, elevations, strict=True):
            regrow(env, name, h, a, e)
        env._place_shoots()
        mujoco.mj_kinematics(m, env.data)
        sides, gaps = [], []
        for k, name in enumerate(env.active):
            # the leaf goes above the shoot if the arm still has a way in with it there (from below: the hand below
            # the shoot is more often clear of the plant than above it, where the stem and the flower are), else below
            # it, else the shoot has none
            gap = rng.uniform(*BLOCK_GAP)
            placed = None
            for side in (1.0, -1.0):
                self.place_leaf(env, k, name, side, gap)
                # (only the clear side's approaches: a level one would have to squeeze its jaws past the leaf)
                kinds = ("below",) if side > 0 else ("above",)
                if plan_cut(env, name, kinds=kinds, slack=DRAW_SLACK) is not None:
                    placed = side
                    break
                self.remove_leaf(env, k)
            sides.append("none" if placed is None else "above" if placed > 0 else "below")
            gaps.append(0.0 if placed is None else gap)
        if not self.used:  # no shoot the arm could reach from above or below: Reach.draw draws again
            self.unreachable = True
        return dict(
            azimuths_deg=np.degrees(az), heights_m=heights, elevations_deg=np.degrees(elevations), leaf=sides, gap_m=gaps
        )

    def place_leaf(self, env, k, name, side, gap):
        """Leaf k over (side +1) or under (-1) shoot `name`, its near face `gap` from the shoot's axis."""
        m = env.model
        s = env.shoots[name]
        d = s["dir"]  # in the plant's frame
        _, up = _plant_frame(d)
        offset = side * (gap + BLOCK_LEAF[2])
        centre = s["origin"] + d * (BLOCK_FROM + BLOCK_LEAF[0]) + up * offset
        R = np.stack([d, np.cross(up, d), up], axis=1)  # the blade's long axis along the shoot, flat across it
        b, p = self.blades[k], self.petioles[k]
        m.geom_pos[b], m.geom_quat[b] = centre, E._quat(R)
        m.geom_pos[p] = s["origin"] + d * (BLOCK_FROM / 2) + up * offset
        m.geom_quat[p] = E._quat(E._shoot_frame(d))
        m.geom_contype[b], m.geom_conaffinity[b] = 1, 1
        _show(m, [b, p], True)
        self.used.append(b)
        mujoco.mj_kinematics(m, env.data)

    def remove_leaf(self, env, k):
        m = env.model
        b, p = self.blades[k], self.petioles[k]
        m.geom_contype[b] = m.geom_conaffinity[b] = 0
        _show(m, [b, p], False)
        self.used.remove(b)

    def restore_leaves(self, env):
        m = env.model
        for b, p in zip(self.blades, self.petioles, strict=True):
            m.geom_pos[b], m.geom_pos[p] = [0.0, 0.0, 0.02], [0.0, 0.0, 0.02]
            m.geom_contype[b] = m.geom_conaffinity[b] = 0
            _show(m, [b, p], False)
        self.used, self.unreachable = [], False

    def substep(self, env):
        """Every physics step (the contacts of the one before): the arm on a blocking leaf damages the plant."""
        if env.damaged is not None or not self.used:
            return
        m, d = env.model, env.data
        n = d.ncon
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        robot1, robot2 = np.isin(m.geom_bodyid[g1], env.robot_bodies), np.isin(m.geom_bodyid[g2], env.robot_bodies)
        other = np.where(robot1, g2, np.where(robot2, g1, -1))
        if np.isin(other, self.used).any():
            env.damaged = "leaf touched"


def _plant_frame(d):
    """(b1, b2) of a shoot along d in the plant's frame (arm.segment_frame: level across it, and up across it)."""
    from .arm import segment_frame

    return segment_frame(d)


SCENARIOS = {s.name: s for s in (Locked, Heights, Pitched, Blocked, HeightNamed)}
CATALOGUE = {s.catalogue: s.name for s in SCENARIOS.values()}


def register():
    """Make these scenarios names that scenarios.parse knows (called from either module, whichever is imported last)."""
    S.SCENARIOS.update(SCENARIOS)
    S.CATALOGUE.update(CATALOGUE)


register()
