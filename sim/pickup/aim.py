"""Aim range: short motor drills for the arm, apart from any task (in the spirit of an aim trainer).

Each drill is a short episode on the bare table with a clear success rule and a time budget. The instruction names
the drill and the target, as a slow planner would hand a sub-goal to a fast controller ("touch the red dot", "press
the blue button"), and other targets of other colours stand around, so that the instruction has to be read.

  aim_touch   touch the named dot (one of four glowing dots, anywhere in the workspace): the gripper's centre within
              TOUCH_TOL of it for TOUCH_HOLD steps in a row
  aim_pose    the same, arriving along the dot's pin: the gripper points along the pin into the dot (within POSE_ANGLE),
              from straight down to level
  aim_hold    stop on the named dot: within HOLD_TOL for HOLD_STEPS steps in a row (a precise stop, no overshoot)
  aim_track   follow the named dot while it moves on a smooth or a jerky path (a swaying shoot): after ACQUIRE steps,
              within TRACK_TOL on at least TRACK_SHARE of the steps; the episode lasts as long as the path
  aim_flick   touch 3-5 dots of the named colour in turn: each lights up where the last one was hit (within FLICK_TOL)
  aim_press   press the named one of two buttons placed anywhere on the table: pushed past PRESS_TRAVEL for
              PRESS_STEPS steps in a row; pushing the other button is a failure
  aim_pinch   pinch the named one of two small cubes (20 mm) and lift it PINCH_LIFT; lifting the other is a failure
  aim_mix     one of the seven per episode (the training mixture, MIX_WEIGHTS)
  aim_each    the seven in turn, by seed (evaluation: equal numbers of each)

Every episode is scored for accuracy and time (AimEnv.aim_record). The drills are built from primitives: no meshes.
Built only for the aim tasks, so the model of every other task is unchanged; PickupEnv("aim_touch") returns an AimEnv,
and ScriptedExpert(env) on one returns an AimExpert (aim_expert.py), so collect.py and eval.py need no changes to run
them.
"""

import mujoco
import numpy as np

from . import env as E

DRILLS = ("touch", "pose", "hold", "track", "flick", "press", "pinch")
# control steps at 20 Hz; aim_track ends with its path (TRACK_STEPS) instead
BUDGET = dict(touch=160, pose=200, hold=200, track=200, flick=320, press=240, pinch=240)
# aim_mix: how often each drill is drawn. The expert's episodes last about 18 (touch), 44 (pose), 40 (hold), 171
# (track), 61 (flick), 44 (press) and 54 (pinch) control steps, so drawing them equally would give the tracking drill
# 39% of the frames and touching 4%; these weights give each drill 11-19% of the frames.
MIX_WEIGHTS = dict(touch=2.0, pose=1.5, hold=1.5, track=0.5, flick=1.0, press=1.0, pinch=1.0)

DOT_COLOURS = (
    ("red", (0.95, 0.12, 0.10, 1)),
    ("green", (0.15, 0.85, 0.20, 1)),
    ("blue", (0.15, 0.35, 0.98, 1)),
    ("yellow", (0.98, 0.85, 0.10, 1)),
)
DOT_R, PIN_R, PIN_LEN = 0.012, 0.0025, 0.06  # metres
# where dots appear (the gripper's centre must reach them pointing down; the lowest keeps the fingertips off the table)
DOT_X, DOT_Y, DOT_Z = (0.33, 0.65), (-0.25, 0.25), (0.05, 0.35)
POSE_X, POSE_Z = (0.36, 0.60), (0.10, 0.30)  # aim_pose: room for a level hand
DOT_GAP = 0.10  # least distance between two dots
TOUCH_TOL, TOUCH_HOLD = 0.015, 3
POSE_ANGLE = np.radians(20)
POSE_AZIMUTH = np.radians(60), np.radians(115)  # from the line out of the robot's base, either side
HOLD_TOL, HOLD_STEPS = 0.008, 15
TRACK_TOL, TRACK_SHARE, ACQUIRE = 0.03, 0.8, 30
TRACK_STEPS = (140, 200)
TRACK_SPEED = (0.02, 0.05)  # m/s, the path's top speed
TRACK_BOX = ((0.38, 0.60), (-0.18, 0.18), (0.10, 0.30))  # the path stays inside
FLICK_TOL, FLICK_COUNT, FLICK_GAP = 0.02, (3, 5), 0.12
PRESS_STEPS = 5
PAD_HALF = (0.03, 0.03, 0.013)  # a button's housing (half sizes); the cap sits on its top, as on the plant stand
PAD_COLOURS = (("blue", E.BUTTON_RGBA["ccw"]), ("orange", E.BUTTON_RGBA["cw"]))
PAD_X, PAD_Y, PAD_GAP = (0.36, 0.62), (-0.22, 0.22), 0.13
CUBE_HALF, PINCH_LIFT, PINCH_STEPS = 0.010, 0.03, 5
CUBE_COLOURS = (("yellow", (0.95, 0.75, 0.05, 1)), ("purple", (0.50, 0.20, 0.75, 1)))
CUBE_X, CUBE_Y, CUBE_GAP = (0.38, 0.62), (-0.22, 0.22), 0.10
HOUSING = (0.13, 0.13, 0.14, 1)
PARK = np.array([2.0, 3.0, -0.70])  # far outside both camera views

INSTRUCTIONS = dict(
    touch="touch the {c} dot",
    pose="touch the {c} dot along its pin",
    hold="stop on the {c} dot",
    track="follow the {c} dot",
    flick="touch the {c} dots one after another",
    press="press the {c} button",
    pinch="pinch the small {c} cube",
)


def add_to_spec(spec):
    """Four dots (mocap bodies; each with a pin along its +z, shown for aim_pose only), two push buttons on housings
    (static bodies, moved per episode) and two small cubes (free bodies). None of the dots or pins collide."""
    wb = spec.worldbody
    for i, (name, rgba) in enumerate(DOT_COLOURS):
        mat = spec.add_material(name=f"aim_{name}", rgba=list(rgba), emission=0.6)
        dot = wb.add_body(name=f"aim_dot_{name}", mocap=True, pos=list(PARK + [0.1 * i, 0, 0]))
        for geom, gtype, size, pos in (
            ("dot", mujoco.mjtGeom.mjGEOM_SPHERE, [DOT_R, 0, 0], [0, 0, 0]),
            ("pin", mujoco.mjtGeom.mjGEOM_CAPSULE, [PIN_R, PIN_LEN / 2, 0], [0, 0, DOT_R + PIN_LEN / 2]),
        ):
            dot.add_geom(
                name=f"aim_{geom}_{name}",
                type=gtype,
                size=size,
                pos=pos,
                material=mat.name,
                contype=0,
                conaffinity=0,
                group=2,
                density=0,
            )
    for i, (name, rgba) in enumerate(PAD_COLOURS):
        pad = wb.add_body(name=f"aim_pad_{name}", pos=list(PARK + [0.1 * i, 0.2, 0]))
        E._look(pad, f"aim_pad_{name}_vis", HOUSING, gtype="box", size=PAD_HALF, pos=(0, 0, PAD_HALF[2]))
        E._solid(pad, f"aim_pad_{name}_col", "box", PAD_HALF, (0, 0, PAD_HALF[2]))
        button = pad.add_body(name=f"aim_button_{name}", pos=[0, 0, 2 * PAD_HALF[2]])
        button.add_joint(
            name=f"aim_button_{name}",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0, 0, 1],
            range=[-E.BUTTON_TRAVEL, 0],
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            stiffness=400.0,  # as the plant stand's buttons
            springref=0.002,
            damping=1.0,
            armature=0.001,
            solref_limit=[0.004, 1],
        )
        size = (E.BUTTON_R, E.BUTTON_H / 2)
        E._solid(button, f"aim_button_{name}_col", "cylinder", size, (0, 0, E.BUTTON_H / 2), density=1000)
        E._look(button, f"aim_button_{name}_vis", rgba, gtype="cylinder", size=size, pos=(0, 0, E.BUTTON_H / 2))
        exclude = spec.add_exclude()
        exclude.bodyname1, exclude.bodyname2 = f"aim_pad_{name}", f"aim_button_{name}"
    for i, (name, rgba) in enumerate(CUBE_COLOURS):
        cube = wb.add_body(name=f"aim_cube_{name}", pos=list(PARK + [0.1 * i, 0.4, -0.05]))
        cube.add_freejoint()
        E._solid(cube, f"aim_cube_{name}_col", "box", (CUBE_HALF,) * 3, conaffinity=1)
        E._look(cube, f"aim_cube_{name}_vis", rgba, gtype="box", size=(CUBE_HALF,) * 3)


def _quat_z_to(v):
    """Quaternion that turns +z onto the unit vector v."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.asarray(v, dtype=float) / np.linalg.norm(v)
    axis, c = np.cross(z, v), float(z @ v)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1, 0, 0])
    angle = np.arctan2(s, c)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis / s])


def _spaced(rng, count, lo, hi, gap, avoid=(), tries=500):
    """`count` points drawn uniformly in the box lo-hi, each at least `gap` from the others and from `avoid`."""
    points = []
    for _ in range(count):
        for _ in range(tries):
            p = rng.uniform(lo, hi)
            if all(np.linalg.norm(p - q) >= gap for q in [*points, *avoid]):
                break
        points.append(p)
    return points


def track_path(rng, steps, kind):
    """Positions of a moving dot, one per control step: 'smooth' (two sine waves per axis, a sway) or 'jerky' (straight
    runs at a steady speed that change direction at random moments). Scaled so its top speed is in TRACK_SPEED."""
    lo, hi = np.array([b[0] for b in TRACK_BOX]), np.array([b[1] for b in TRACK_BOX])
    dt = 1.0 / E.CONTROL_HZ
    top = rng.uniform(*TRACK_SPEED)
    centre = rng.uniform(lo + 0.06, hi - 0.06)
    t = np.arange(steps + 1) * dt
    if kind == "smooth":
        path = np.zeros((steps + 1, 3))
        for axis in range(3):
            for _ in range(2):
                amp, period = rng.uniform(0.02, 0.06) * (0.5 if axis == 2 else 1.0), rng.uniform(2.5, 6.0)
                path[:, axis] += amp * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
        speed = np.linalg.norm(np.diff(path, axis=0), axis=1).max() / dt
        path = centre + path * (top / speed)
    else:
        path, p, v, left = [], centre.copy(), None, 0
        for _ in range(steps + 1):
            if left <= 0:
                d = rng.normal(size=3) * [1, 1, 0.5]
                v, left = d / np.linalg.norm(d) * top, int(rng.integers(8, 25))
            path.append(p.copy())
            p = p + v * dt
            out = (p < lo) | (p > hi)  # bounce off the box
            v[out] *= -1
            p = np.clip(p, lo, hi)
            left -= 1
        path = np.array(path)
    return np.clip(path, lo, hi)


class AimEnv(E.PickupEnv):
    """PickupEnv on an aim task (see the module's docstring). Built by PickupEnv(task) for any task with an `aim`
    entry; its reset and step replace the base class's (no objects of OBJECTS are used, all are parked)."""

    def __init__(self, task, *args, **kwargs):
        super().__init__(task, *args, **kwargs)
        m = self.model
        self.aim = E.TASKS[task]["aim"]
        if self._max_steps_given is None:  # until a reset draws the drill (aim_mix, aim_each): the longest budget
            self.max_steps = BUDGET.get(self.aim, max(BUDGET.values()))
        self.drill = ""
        self.dot_mocap = {c: int(m.body(f"aim_dot_{c}").mocapid[0]) for c, _ in DOT_COLOURS}
        self.pin_geom = {c: m.geom(f"aim_pin_{c}").id for c, _ in DOT_COLOURS}
        self.pad_body = {c: m.body(f"aim_pad_{c}").id for c, _ in PAD_COLOURS}
        self.button_qadr = {c: m.joint(f"aim_button_{c}").qposadr[0] for c, _ in PAD_COLOURS}
        self.button_body = {c: m.body(f"aim_button_{c}").id for c, _ in PAD_COLOURS}
        self.cube_qadr = {c: m.joint(m.body(f"aim_cube_{c}").jntadr[0]).qposadr[0] for c, _ in CUBE_COLOURS}
        self.cube_vadr = {c: m.joint(m.body(f"aim_cube_{c}").jntadr[0]).dofadr[0] for c, _ in CUBE_COLOURS}
        self.colour = ""

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int, target_index: int | None = None):
        """Draw the drill (aim_mix), its target and layout from `seed`. `target_index` is not used."""
        rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        q0 = E.HOME_Q + rng.normal(0, 0.03, 7)
        d.qpos[:7] = q0
        d.qpos[7:9] = 0.04
        d.ctrl[:7] = q0
        d.ctrl[7] = 255.0
        for i, name in enumerate(E.OBJECTS):  # nothing of the pick tasks is used
            adr = self.obj_qadr[name]
            d.qpos[adr : adr + 3] = [2.0 + 0.3 * i, 2.0, -0.76 + E.OBJECTS[name]["rest_z"]]
            d.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
        self._place_leaves(rng, [], 0)
        self.drill = self._choose_drill(seed)
        self.target_label = self.drill
        if self._max_steps_given is None:
            self.max_steps = self._budget(self.drill)
        # per episode: what was drawn and how the episode is going
        self.dots: dict[str, np.ndarray] = {}  # colour -> position of each dot in view
        self.pin_dir = None  # aim_pose: unit vector the gripper must point along (into the dot)
        self.path = None  # aim_track: the dot's position at every control step
        self.path_kind = ""
        self.flick_points: list[np.ndarray] = []
        self.flick_index = 0
        self.flick_steps: list[int] = []  # aim_flick: the step at which each dot was hit
        self.near, self.track_errors = 0, []
        self.press_count, self.wrong = 0, False
        self.cube_rest: dict[str, float] = {}
        self.result_info: dict = {}
        self.aim_success = False
        for c, mid in self.dot_mocap.items():
            d.mocap_pos[mid], d.mocap_quat[mid] = PARK + [0.1 * list(self.dot_mocap).index(c), 0, 0], [1, 0, 0, 0]
            m.geom_group[self.pin_geom[c]] = 3
        for c, body in self.pad_body.items():
            m.body_pos[body] = PARK + [0.1 * list(self.pad_body).index(c), 0.2, 0]
        for i, c in enumerate(self.cube_qadr):
            self._put_cube(c, PARK[:2] + [0.1 * i, 0.4], -0.75 + CUBE_HALF, 0.0)
        self._reset_drill(rng)
        self.instruction = self._instructions()[self.drill].format(c=self.colour)
        self.target = f"aim_{self.colour}"
        self.active = [self.target] + [f"aim_{c}" for c in self._others()]
        mujoco.mj_forward(m, d)
        for _ in range(100):  # let the cubes and buttons settle
            mujoco.mj_step(m, d)
        self.cube_rest = {c: float(d.qpos[self.cube_qadr[c] + 2]) for c in self.cube_qadr}
        self._after_settle()
        self.rest_z = {}
        self.last_action = np.concatenate([d.ctrl[:7], [1.0]]).astype(np.float32)
        self.steps = 0
        self.lift_count = 0
        return self.get_obs()

    # hooks for a family of drills built on this one (avoid.py)
    def _choose_drill(self, seed):
        if self.aim == "mix":  # from a stream of its own: the layout is that of the drill's own task
            p = np.array([MIX_WEIGHTS[k] for k in DRILLS])
            return str(np.random.default_rng([seed, 37]).choice(DRILLS, p=p / p.sum()))
        if self.aim == "each":
            return DRILLS[seed % len(DRILLS)]
        return self.aim

    def _budget(self, drill):
        return BUDGET[drill]

    def _instructions(self):
        return INSTRUCTIONS

    def _reset_drill(self, rng):
        getattr(self, f"_reset_{self.drill}")(rng)

    def _after_settle(self):
        pass

    def _substep_hook(self):
        pass

    def _score(self):
        getattr(self, f"_score_{self.drill}")()

    def _others(self):
        if self.drill == "press":
            return [c for c, _ in PAD_COLOURS if c != self.colour]
        if self.drill == "pinch":
            return [c for c, _ in CUBE_COLOURS if c != self.colour]
        return [c for c in self.dots if c != self.colour]

    def _show_dots(self, rng, target_pos, distractors=2, lo=None, hi=None):
        """The named dot at `target_pos`, and `distractors` dots of other colours elsewhere in the workspace."""
        lo = np.array([DOT_X[0], DOT_Y[0], DOT_Z[0]]) if lo is None else lo
        hi = np.array([DOT_X[1], DOT_Y[1], DOT_Z[1]]) if hi is None else hi
        others = [c for c, _ in DOT_COLOURS if c != self.colour]
        others = [others[i] for i in rng.permutation(len(others))[:distractors]]
        self.dots = {self.colour: np.asarray(target_pos, dtype=float)}
        for c, p in zip(others, _spaced(rng, distractors, lo, hi, DOT_GAP, avoid=[target_pos]), strict=True):
            self.dots[c] = p
        for c, p in self.dots.items():
            self.data.mocap_pos[self.dot_mocap[c]] = p

    def _dot_box(self):
        return np.array([DOT_X[0], DOT_Y[0], DOT_Z[0]]), np.array([DOT_X[1], DOT_Y[1], DOT_Z[1]])

    def _pick_colour(self, rng, palette=DOT_COLOURS):
        self.colour = palette[int(rng.integers(len(palette)))][0]

    def _reset_touch(self, rng):
        self._pick_colour(rng)
        self._show_dots(rng, rng.uniform(*self._dot_box()))

    _reset_hold = _reset_touch

    def _reset_pose(self, rng):
        self._pick_colour(rng)
        pos = np.array([rng.uniform(*POSE_X), rng.uniform(*DOT_Y), rng.uniform(*POSE_Z)])
        # the way the gripper points: from straight down to level; seen from above, roughly across the line from the
        # robot's base (pointing out, away from the base, or back toward it, the arm runs out of reach)
        elevation = rng.uniform(0, np.pi / 2)
        radial = np.arctan2(pos[1], pos[0])
        azimuth = radial + rng.choice([-1, 1]) * rng.uniform(*POSE_AZIMUTH)
        self.pin_dir = np.array(
            [np.cos(azimuth) * np.cos(elevation), np.sin(azimuth) * np.cos(elevation), -np.sin(elevation)]
        )
        self._show_dots(rng, pos)
        mid = self.dot_mocap[self.colour]
        self.data.mocap_quat[mid] = _quat_z_to(-self.pin_dir)  # the pin sticks out the way the gripper comes from
        self.model.geom_group[self.pin_geom[self.colour]] = 2

    def _reset_track(self, rng):
        self._pick_colour(rng)
        steps = int(rng.integers(TRACK_STEPS[0], TRACK_STEPS[1] + 1))
        self.path_kind = str(rng.choice(["smooth", "jerky"]))
        self.path = track_path(rng, steps, self.path_kind)
        self.max_steps = steps if self._max_steps_given is None else self.max_steps
        self._show_dots(rng, self.path[0], distractors=1)

    def _reset_flick(self, rng):
        self._pick_colour(rng)
        count = int(rng.integers(FLICK_COUNT[0], FLICK_COUNT[1] + 1))
        lo, hi = self._dot_box()
        self._show_dots(rng, rng.uniform(lo, hi))
        avoid = [p for c, p in self.dots.items() if c != self.colour]
        self.flick_points = [self.dots[self.colour]]
        for _ in range(count - 1):  # each at least FLICK_GAP from the last and clear of the distractors
            self.flick_points += _spaced(rng, 1, lo, hi, DOT_GAP, avoid=avoid + self.flick_points[-1:])
            while np.linalg.norm(self.flick_points[-1] - self.flick_points[-2]) < FLICK_GAP:
                self.flick_points[-1] = _spaced(rng, 1, lo, hi, DOT_GAP, avoid=avoid)[0]

    def _reset_press(self, rng):
        self._pick_colour(rng, PAD_COLOURS)
        lo, hi = np.array([PAD_X[0], PAD_Y[0]]), np.array([PAD_X[1], PAD_Y[1]])
        for (c, _), xy in zip(PAD_COLOURS, _spaced(rng, 2, lo, hi, PAD_GAP), strict=True):
            self.model.body_pos[self.pad_body[c]] = [xy[0], xy[1], 0.0]

    def _reset_pinch(self, rng):
        self._pick_colour(rng, CUBE_COLOURS)
        lo, hi = np.array([CUBE_X[0], CUBE_Y[0]]), np.array([CUBE_X[1], CUBE_Y[1]])
        for (c, _), xy in zip(CUBE_COLOURS, _spaced(rng, 2, lo, hi, CUBE_GAP), strict=True):
            self._put_cube(c, xy, CUBE_HALF + 0.001, rng.uniform(-np.pi, np.pi))

    def _put_cube(self, colour, xy, z, yaw):
        adr, vadr = self.cube_qadr[colour], self.cube_vadr[colour]
        self.data.qpos[adr : adr + 3] = [xy[0], xy[1], z]
        self.data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[vadr : vadr + 6] = 0

    # ---------------------------------------------------------------- queries
    def dot_pos(self, colour=None):
        """Where the named dot (or the dot of `colour`) is now."""
        return self.data.mocap_pos[self.dot_mocap[colour or self.colour]].copy()

    def button_top(self, colour=None):
        """World position of the centre of a button's cap top, at rest."""
        p = self.data.xpos[self.pad_body[colour or self.colour]]
        return p + [0, 0, 2 * PAD_HALF[2] + E.BUTTON_H]

    def aim_button_travel(self, colour):
        return float(-self.data.qpos[self.button_qadr[colour]])

    def cube_pos(self, colour=None):
        adr = self.cube_qadr[colour or self.colour]
        return self.data.qpos[adr : adr + 3].copy()

    def cube_yaw(self, colour=None):
        adr = self.cube_qadr[colour or self.colour]
        w, x, y, z = self.data.qpos[adr + 3 : adr + 7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def object_pos(self, name):
        """For the diagnostics of eval.py: where the dot, button or cube `aim_<colour>` is."""
        colour = name.removeprefix("aim_")
        if self.drill == "press":
            return self.button_top(colour)
        if self.drill == "pinch":
            return self.cube_pos(colour)
        return self.dot_pos(colour)

    def object_yaw(self, name):
        return self.cube_yaw(name.removeprefix("aim_")) if self.drill == "pinch" else 0.0

    def on_table(self, name):
        return True

    def pointing(self):
        """The way the gripper points (unit vector from the hand toward the fingertips)."""
        return self.data.site_xmat[self.tcp_id].reshape(3, 3)[:, 2].copy()

    # ------------------------------------------------------------------- step
    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        d = self.data
        d.ctrl[:7] = np.clip(action[:7], self.ctrl_lo, self.ctrl_hi)
        d.ctrl[7] = 255.0 * float(np.clip(action[7], 0.0, 1.0))
        if self.path is not None:  # the dot moves on to where its path is at the end of this step
            d.mocap_pos[self.dot_mocap[self.colour]] = self.path[min(self.steps + 1, len(self.path) - 1)]
        for _ in range(E.SUBSTEPS):
            mujoco.mj_step(self.model, d)
            self._substep_hook()
        self.last_action = np.concatenate([d.ctrl[:7], [d.ctrl[7] / 255.0]]).astype(np.float32)
        self.steps += 1
        if not self.aim_success and not self.wrong:  # once ended, an episode's outcome stays (collect.py's tail)
            self._score()
        done = self.aim_success or self.wrong or self.steps >= self.max_steps
        info = dict(
            success=self.aim_success,
            lifted=[self.target] if self.aim_success else [],
            wrong_object=self.wrong,
            damaged=None,
            drill=self.drill,
            **self.result_info,
        )
        return self.get_obs(), done, info

    def _error(self):
        return float(np.linalg.norm(self.tcp_pos() - self.dot_pos()))

    def _hold_near(self, tol, steps, extra=True):
        self.near = self.near + 1 if self._error() < tol and extra else 0
        self.result_info = dict(error_mm=round(1000 * self._error(), 1))
        self.aim_success = self.near >= steps

    def _score_touch(self):
        self._hold_near(TOUCH_TOL, TOUCH_HOLD)

    def _score_hold(self):
        self._hold_near(HOLD_TOL, HOLD_STEPS)

    def _score_pose(self):
        angle = float(np.arccos(np.clip(self.pointing() @ self.pin_dir, -1, 1)))
        self._hold_near(TOUCH_TOL, TOUCH_HOLD, extra=angle < POSE_ANGLE)
        self.result_info["angle_deg"] = round(float(np.degrees(angle)), 1)

    def _score_track(self):
        if self.steps > ACQUIRE:
            self.track_errors.append(self._error())
        within = float(np.mean(np.array(self.track_errors) < TRACK_TOL)) if self.track_errors else 0.0
        self.result_info = dict(
            error_mm=round(1000 * float(np.mean(self.track_errors)), 1) if self.track_errors else None,
            within=round(within, 3),
        )
        self.aim_success = self.steps >= self.max_steps and within >= TRACK_SHARE

    def _score_flick(self):
        if self._error() < FLICK_TOL:
            self.flick_steps.append(self.steps)
            self.flick_index += 1
            if self.flick_index < len(self.flick_points):  # the next dot lights up
                self.data.mocap_pos[self.dot_mocap[self.colour]] = self.flick_points[self.flick_index]
        self.result_info = dict(hits=self.flick_index, targets=len(self.flick_points), hit_steps=list(self.flick_steps))
        self.aim_success = self.flick_index >= len(self.flick_points)

    def _score_press(self):
        other = self._others()[0]
        self.wrong = bool(self.aim_button_travel(other) > E.PRESS_TRAVEL)
        self.press_count = self.press_count + 1 if self.aim_button_travel(self.colour) > E.PRESS_TRAVEL else 0
        xy = float(np.linalg.norm(self.tcp_pos()[:2] - self.button_top()[:2]))
        self.result_info = dict(error_mm=round(1000 * xy, 1))
        self.aim_success = self.press_count >= PRESS_STEPS and not self.wrong

    def _score_pinch(self):
        lift = {c: self.cube_pos(c)[2] - self.cube_rest[c] for c in self.cube_qadr}
        other = self._others()[0]
        self.wrong = bool(lift[other] > PINCH_LIFT)
        self.lift_count = self.lift_count + 1 if lift[self.colour] > PINCH_LIFT else 0
        xy = float(np.linalg.norm(self.tcp_pos()[:2] - self.cube_pos()[:2]))
        self.result_info = dict(error_mm=round(1000 * xy, 1))
        self.aim_success = self.lift_count >= PINCH_STEPS and not self.wrong

    # ------------------------------------------------------------- reporting
    def aim_record(self, info):
        """One episode's score, for results.json: the drill, the colour named and how close the gripper came."""
        record = dict(drill=self.drill, colour=self.colour, **{k: info[k] for k in self.result_info if k in info})
        if self.drill == "track":
            record["path"] = self.path_kind
        return record


def summary(episodes):
    """Per drill: episodes, successes, wrong targets, the mean steps of successful episodes, and the median error
    (millimetres; for aim_track the mean tracking error, for aim_flick the steps per dot)."""
    out = {}
    present = {e.get("aim", {}).get("drill") for e in episodes}
    for drill in [*DRILLS, *sorted(present - set(DRILLS) - {None})]:
        eps = [e for e in episodes if e.get("aim", {}).get("drill") == drill]
        if not eps:
            continue
        ok = [e for e in eps if e["success"]]
        errors = [e["aim"]["error_mm"] for e in eps if e["aim"].get("error_mm") is not None]
        row = dict(
            episodes=len(eps),
            successes=len(ok),
            success_rate=len(ok) / len(eps),
            wrong=sum(e["wrong_object"] for e in eps),
            mean_steps_success=float(np.mean([e["steps"] for e in ok])) if ok else None,
            median_error_mm=float(np.median(errors)) if errors else None,
        )
        if drill == "flick":
            hits = [e for e in ok if e["aim"].get("hit_steps")]
            row["steps_per_dot_success"] = (
                float(np.mean([e["aim"]["hit_steps"][-1] / len(e["aim"]["hit_steps"]) for e in hits])) if hits else None
            )
        if drill == "track":
            row["by_path"] = {
                k: dict(episodes=len(g), successes=sum(e["success"] for e in g))
                for k in ("smooth", "jerky")
                for g in [[e for e in eps if e["aim"].get("path") == k]]
                if g
            }
        if any(e["aim"].get("angle_deg") is not None for e in eps):
            angles = [e["aim"]["angle_deg"] for e in eps if e["aim"].get("angle_deg") is not None]
            row["median_angle_deg"] = float(np.median(angles))
        if any("hit" in e["aim"] for e in eps):  # avoidance drills (avoid.py)
            clear = [e["aim"]["clearance_mm"] for e in eps if e["aim"].get("clearance_mm") is not None]
            zone = [e["aim"]["zone_mm"] for e in eps if e["aim"].get("zone_mm") is not None]
            row.update(
                hit_obstacle=sum(e["aim"]["hit"] for e in eps),
                entered_zone=sum(e["aim"]["in_zone"] for e in eps),
                contact_steps=sum(e["aim"]["contacts"] for e in eps),
                median_clearance_mm=float(np.median(clear)) if clear else None,
                median_zone_mm=float(np.median(zone)) if zone else None,
            )
        out[drill] = row
    return out
