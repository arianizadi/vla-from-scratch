"""Test situations (scenarios): the same robot and plant in a changed world or with a changed task.

A scenario is an option of the environment, applied on top of a task: `PickupEnv("prune", scenario="wind")`, or
`--scenario wind` for collect.py (training data) and eval.py (testing). Several can be combined with commas. Each one
draws its own variation for every episode from a stream of its own, seeded by the episode's seed, so that

  - the plant, the stand and everything else the task draws are the ones the plain task draws for that seed (a
    scenario run and a plain run on the same seeds compare the same layouts);
  - the variation is the same every time the seed is run (a dataset can be rebuilt, a failure replayed).

What was drawn is in `env.scenario_info` after every reset ({name: variation}); eval.py and collect.py record it per
episode. Without a scenario the environment is exactly what it was before scenarios existed.

The spec of one scenario is `name[:mode][@p]`: `mode` pins one of its variations (`lighting:dusk`), `p` applies it to
that share of the episodes only (`nothing_to_do@0.2`, for training data that mixes it in). The catalogue numbers
(B1, A5, ...) are accepted as names. See tutorial/test-situations.md for the catalogue.

Robustness situations (the task is unchanged; held out by default):

  lighting   B1  dusk, harsh noon sun with hard shadows, purple grow lights, flickering light
  camera     B2  the front camera knocked a few cm and a few degrees; sometimes slight blur or sensor noise
  wind       B5  the shoots sway gently, or a gust at a random moment
  bump       B6  someone turns the plate 30-120 degrees mid-task, while the gripper is clear of the plant
  sticky     B7  a button whose first press does nothing, or whose motor runs slowly

Reasoning situations (the task changes; the expert solves them):

  nothing_to_do  A5  an already pruned plant: success is no damage and the arm back home
  order          A8  cut the shoots in a given order ("from the top down", "the lowest, then the highest")
  order_heldout  A8  the same with sentence frames no demonstration uses
"""

import json
import zlib
from functools import cache

import mujoco
import numpy as np
from PIL import Image, ImageFilter

from . import env as E

CATALOGUE = {
    "A1": "ties",
    "A3": "sick",
    "A4": "buds",
    "B1": "lighting",
    "B2": "camera",
    "B3": "bench",
    "B4": "distractors",
    "B5": "wind",
    "B6": "bump",
    "B7": "sticky",
    "A5": "nothing_to_do",
    "A8": "order",
}


def _r(x):
    """Plain, rounded numbers for the record (JSON)."""
    if isinstance(x, dict):
        return {k: _r(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


class Scenario:
    """One situation. Subclasses set `name`, `modes` (the kinds of variation, drawn uniformly unless pinned) and
    override the hooks they need. Every hook acts only in the episodes the scenario applies to (`self.applied`)."""

    name = ""
    catalogue = ""
    modes: tuple[str, ...] = ("default",)
    needs_plant = True
    changes_task = False  # gives its own instruction and success rule (at most one such scenario at a time)
    priority = 0  # order of the hooks within a step (lower first)

    def __init__(self, mode=None, prob=1.0):
        assert mode is None or mode in self.modes, f"{self.name}: unknown mode {mode!r}; choose from {self.modes}"
        assert 0.0 < prob <= 1.0, f"{self.name}: probability {prob} not in (0, 1]"
        self.mode, self.prob = mode, prob
        self.applied, self.variation = False, {}

    def spec(self):
        return self.name + (f":{self.mode}" if self.mode else "") + (f"@{self.prob:g}" if self.prob < 1 else "")

    # --------------------------------------------------------------------------------------------------- hooks
    def check(self, env):
        """Raise if the scenario cannot apply to env's task."""
        if self.needs_plant and not env.plant_task:
            raise ValueError(f"scenario {self.name} needs a plant task, not {env.task}")

    def build(self, spec):
        """Change the model before it is compiled (only models built with this scenario)."""

    def init(self, env):
        """Once, after the model is built: look up ids, keep the defaults that restore() puts back."""

    def restore(self, env):
        """At every reset, before the draw: undo what the last episode changed."""

    def draw(self, env, rng, mode):
        """Apply this episode's variation (the plant is drawn and placed, not yet settled). Returns the variation."""
        return {}

    def before_step(self, env, action):
        """Before the physics of a control step."""

    def substep(self, env):
        """Every physics substep, after the turntable's motor has been set by the buttons."""

    def after_step(self, env):
        """After the physics of a control step, before the cameras render."""

    def image(self, env, camera, img):
        return img

    def result(self, env, done, info):
        """Change the outcome of a step (tasks whose success rule the scenario changes). Returns (done, info)."""
        return done, info

    # ---------------------------------------------------------------------------------------------- per episode
    def reset(self, env, seed):
        self.restore(env)
        rng = np.random.default_rng([int(seed), 53, zlib.crc32(self.name.encode()) & 0xFFFFFFF])
        self.applied = bool(rng.random() < self.prob)  # always drawn, so a pinned probability keeps the stream
        mode = self.mode or self.modes[int(rng.integers(len(self.modes)))]
        self.rng = rng
        self.variation = {"applied": self.applied}
        if self.applied:
            self.variation.update(mode=mode, **{k: _r(v) for k, v in self.draw(env, rng, mode).items()})
        return self.variation


# ---------------------------------------------------------------------------------------------------- B1 lighting
KEY_POS, FILL_POS = (0.4, -0.3, 1.6), (0.9, 0.6, 1.2)  # the two lights of env.build_model
LIGHT_TARGET = np.array([0.55, 0.0, 0.15])  # about the middle of the plant


class Lighting(Scenario):
    name, catalogue = "lighting", "B1"
    modes = ("dusk", "noon", "grow", "flicker")
    needs_plant = False

    def init(self, env):
        m = env.model
        self.key = int(np.argmin(np.linalg.norm(m.light_pos - KEY_POS, axis=1)))
        self.fill = int(np.argmin(np.linalg.norm(m.light_pos - FILL_POS, axis=1)))
        fields = ("light_pos", "light_dir", "light_diffuse", "light_ambient", "light_specular")
        self.defaults = {f: getattr(m, f).copy() for f in fields}
        h = m.vis.headlight
        self.head = (h.ambient.copy(), h.diffuse.copy(), h.specular.copy())
        self.factor = 1.0

    def restore(self, env):
        m = env.model
        for f, v in self.defaults.items():
            getattr(m, f)[:] = v
        h = m.vis.headlight
        h.ambient[:], h.diffuse[:], h.specular[:] = self.head
        self.flicker = None

    def _aim(self, env, light, azimuth, elevation, distance=1.6):
        """Put a light on a sphere around the plant, pointing at it."""
        u = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
        env.model.light_pos[light] = LIGHT_TARGET + distance * u
        env.model.light_dir[light] = -u

    def draw(self, env, rng, mode):
        m, h = env.model, env.model.vis.headlight
        key, fill = self.key, self.fill
        out = {}
        if mode == "dusk":  # low, warm sun from one side; dim blue sky; little light from the camera's side
            k, az, el = rng.uniform(0.35, 0.6), rng.uniform(-np.pi, np.pi), np.radians(rng.uniform(8, 25))
            self._aim(env, key, az, el)
            m.light_diffuse[key] = np.array([1.0, 0.55, 0.30]) * k
            m.light_diffuse[fill] = np.array([0.15, 0.18, 0.32]) * rng.uniform(0.6, 1.2)
            h.ambient[:] = np.array([0.12, 0.12, 0.17]) * rng.uniform(0.8, 1.3)
            h.diffuse[:] = np.array([0.10, 0.10, 0.12]) * rng.uniform(0.8, 1.3)
            out = dict(key=k, azimuth=az, elevation=el)
        elif mode == "noon":  # one hard, bright light from nearly overhead; hard shadows, little fill
            k, az, el = rng.uniform(0.85, 1.0), rng.uniform(-np.pi, np.pi), np.radians(rng.uniform(65, 85))
            self._aim(env, key, az, el)
            m.light_diffuse[key] = np.array([1.0, 0.98, 0.92]) * k
            m.light_specular[key] = 0.3
            m.light_diffuse[fill] = 0.0
            h.ambient[:] = rng.uniform(0.06, 0.15)
            h.diffuse[:] = rng.uniform(0.08, 0.2)
            out = dict(key=k, azimuth=az, elevation=el)
        elif mode == "grow":  # purple LED grow lights: red and blue, almost no green
            k = rng.uniform(0.6, 1.0)
            m.light_diffuse[key] = np.array([0.80, 0.22, 0.95]) * k
            m.light_diffuse[fill] = np.array([0.55, 0.12, 0.70]) * k * rng.uniform(0.5, 1.0)
            h.ambient[:] = np.array([0.28, 0.08, 0.34]) * rng.uniform(0.7, 1.2)
            h.diffuse[:] = np.array([0.22, 0.07, 0.28]) * rng.uniform(0.7, 1.2)
            out = dict(key=k)
        else:  # flicker: the usual lights dip now and then, for a few control steps
            self.flicker = dict(p=rng.uniform(0.04, 0.12), lo=rng.uniform(0.2, 0.5), left=0, level=1.0)
            out = dict(dip_prob=self.flicker["p"], dip_low=self.flicker["lo"])
            self._flicker(env)
        return out

    def _flicker(self, env):
        f = self.flicker
        if f["left"] > 0:
            f["left"] -= 1
        elif self.rng.random() < f["p"]:
            f["left"], f["level"] = int(self.rng.integers(1, 4)), self.rng.uniform(f["lo"], 0.8)
        level = f["level"] if f["left"] > 0 else self.rng.uniform(0.95, 1.05)
        m, h = env.model, env.model.vis.headlight
        for light in (self.key, self.fill):
            m.light_diffuse[light] = self.defaults["light_diffuse"][light] * level
        h.diffuse[:] = self.head[1] * level
        h.ambient[:] = self.head[0] * (0.5 + 0.5 * level)

    def after_step(self, env):
        if self.flicker is not None:
            self._flicker(env)


# ------------------------------------------------------------------------------------------------ B2 camera knocked
class Camera(Scenario):
    name, catalogue = "camera", "B2"
    modes = ("moved", "moved_blur", "moved_noise")
    needs_plant = False

    def init(self, env):
        self.cam = env.model.camera("front").id
        self.pos, self.quat = env.model.cam_pos[self.cam].copy(), env.model.cam_quat[self.cam].copy()

    def restore(self, env):
        env.model.cam_pos[self.cam], env.model.cam_quat[self.cam] = self.pos, self.quat
        self.blur = self.noise = 0.0

    def draw(self, env, rng, mode):
        direction = rng.normal(size=3)
        shift = direction / np.linalg.norm(direction) * rng.uniform(0.01, 0.04)  # metres
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.radians(rng.uniform(1.0, 5.0))
        turn, quat = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, axis, angle)
        mujoco.mju_mulQuat(quat, self.quat, turn)  # turned about an axis in the camera's own frame
        env.model.cam_pos[self.cam] = self.pos + shift
        env.model.cam_quat[self.cam] = quat
        if mode == "moved_blur":
            self.blur = rng.uniform(0.6, 1.5)  # pixels (Gaussian radius)
        if mode == "moved_noise":
            self.noise = rng.uniform(3.0, 10.0)  # standard deviation, in 0-255 units
        return dict(shift_m=shift, axis=axis, angle_deg=np.degrees(angle), blur_px=self.blur, noise=self.noise)

    def image(self, env, camera, img):
        if camera != "front":
            return img
        if self.blur:
            img = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(self.blur)))
        if self.noise:
            img = np.clip(img + self.rng.normal(0, self.noise, img.shape), 0, 255).astype(np.uint8)
        return img


# -------------------------------------------------------------------------------------------------------- B5 wind
class Wind(Scenario):
    """The shoots sway: each shoot's direction is turned about the main stem (and up and down by half as much) along
    a smooth curve, and its outer part follows through its weld; a cut shoot's stub sways too. 'sway': a steady sway
    that fades in over the first second. 'gust': a little sway, and one gust at a random moment that swings every shoot
    the same way and dies down."""

    name, catalogue = "wind", "B5"
    modes = ("sway", "gust")

    def restore(self, env):
        self.base = None

    def draw(self, env, rng, mode):
        self.base = {n: s["dir"].copy() for n, s in env.shoots.items()}
        self.phase = {n: rng.uniform(0, 2 * np.pi) for n in env.shoots}
        if mode == "sway":
            self.amp, self.freq = np.radians(rng.uniform(4, 10)), rng.uniform(0.4, 1.0)
            self.gust = None
            return dict(amp_deg=np.degrees(self.amp), freq_hz=self.freq)
        self.amp, self.freq = np.radians(2.0), rng.uniform(0.4, 1.0)
        limit = env.max_steps / E.CONTROL_HZ
        self.gust = dict(
            t0=rng.uniform(2.0, max(2.5, 0.5 * limit)),
            amp=np.radians(rng.uniform(15, 25)),
            decay=rng.uniform(0.6, 1.5),
            freq=rng.uniform(1.0, 1.8),
            sign=float(rng.choice([-1.0, 1.0])),
        )
        return dict(amp_deg=2.0, freq_hz=self.freq, **{f"gust_{k}": v for k, v in self.gust.items()})

    def offset(self, name, t):
        """(azimuth, elevation) offsets of a shoot at time t (seconds into the episode), radians."""
        a = self.amp * min(1.0, t) * np.sin(2 * np.pi * self.freq * t + self.phase[name])
        g = self.gust
        if g is not None and t >= g["t0"]:
            s = t - g["t0"]
            a += g["sign"] * g["amp"] * np.exp(-s / g["decay"]) * np.sin(2 * np.pi * g["freq"] * s)
        return a, 0.5 * a

    def before_step(self, env, action):
        t = env.steps / E.CONTROL_HZ
        m = env.model
        for name, shoot in env.shoots.items():
            d0 = self.base[name]
            da, de = self.offset(name, t)
            az, el = np.arctan2(d0[1], d0[0]) + da, np.arcsin(np.clip(d0[2], -1, 1)) + de
            d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            shoot["dir"], shoot["frame"] = d, E._shoot_frame(d)
            if name in env.cut:
                s, end = shoot["cut_s"], E.CUT_GAP + E.SHOOT_R
                env._shape_shoot(name, s - end, s + end)
            else:
                env._shape_shoot(name, E.SHOOT_SPLIT, E.SHOOT_SPLIT)
                data = m.eq_data[env.shoot_weld[name]]
                data[:3], data[6:10] = shoot["origin"] + d * E.CUT_MID, E._quat(shoot["frame"].T)


# ------------------------------------------------------------------------------------------------- B6 bump the plate
BUMP_SPEED = 3.0  # rad/s: how fast a bump turns the plate
# The bump waits until the gripper is clear of the plant: this far from the main stem's axis (seen from above), or this
# far above the top of the flower (metres)
BUMP_CLEAR, BUMP_ABOVE = 0.15, 0.06


class Bump(Scenario):
    """Someone turns the plate by hand: at a drawn moment (8-45% into the episode's step limit), once the gripper is
    clear of the plant (beside it or above the flower), the plate is driven round by 30-120 degrees, either way, at
    BUMP_SPEED. When it stops, env.events holds 'bumped' for one step (the expert plans again from there)."""

    name, catalogue = "bump", "B6"
    modes = ("bump",)
    priority = 20  # after the buttons' motor and a sticky button: a bump overrides both

    def init(self, env):
        a = env.turntable
        self.ranges = (env.model.actuator_ctrlrange[a].copy(), env.model.actuator_forcerange[a].copy())

    def restore(self, env):
        a = env.turntable
        env.model.actuator_ctrlrange[a], env.model.actuator_forcerange[a] = self.ranges
        self.state = None

    def draw(self, env, rng, mode):
        step = int(rng.uniform(0.08, 0.45) * env.max_steps)
        angle = np.radians(rng.uniform(30, 120)) * rng.choice([-1.0, 1.0])
        self.state = dict(due=step, angle=angle, start=None, at=None)
        return dict(due_step=step, angle_deg=np.degrees(angle))

    def _clear(self, env):
        base, up = env.stem_axis()
        rel = env.tcp_pos() - base
        return np.linalg.norm(rel[:2]) >= BUMP_CLEAR or rel @ up > E.STEM_H + BUMP_ABOVE

    def before_step(self, env, action):
        s = self.state
        if s is None or s["start"] is not None or env.steps < s["due"] or not self._clear(env):
            return
        s["start"], s["at"] = env.plate_angle(), env.steps
        a = env.turntable
        env.model.actuator_ctrlrange[a] = [-2 * BUMP_SPEED, 2 * BUMP_SPEED]
        env.model.actuator_forcerange[a] = [-10, 10]

    def substep(self, env):
        s = self.state
        if s is None or s["start"] is None or "done" in s:
            return
        turned = env.plate_angle() - s["start"]
        if turned * np.sign(s["angle"]) >= abs(s["angle"]):
            s["done"] = True
            self.restore_ranges(env)
            env.events.add("bumped")
            self.variation.update(bumped_step=int(s["at"]), turned_deg=_r(np.degrees(turned)))
            return
        env.data.ctrl[env.turntable] = np.sign(s["angle"]) * BUMP_SPEED

    def restore_ranges(self, env):
        a = env.turntable
        env.model.actuator_ctrlrange[a], env.model.actuator_forcerange[a] = self.ranges


# ------------------------------------------------------------------------------------------------- B7 sticky button
STICKY_RELEASE = 0.001  # metres of travel below which a button counts as let go (between two presses)


class Sticky(Scenario):
    """A faulty button: 'dead' - its first one or two presses do nothing (let go and press again); 'slow' - its motor
    turns the plate at 35-60% of the usual speed. One button or both. env.flags holds 'sticky' (the expert lets go and
    presses again when the plate does not move)."""

    name, catalogue = "sticky", "B7"
    modes = ("dead", "slow")
    priority = 10

    def restore(self, env):
        self.faulty = {}

    def draw(self, env, rng, mode):
        buttons = list(E.BUTTONS) if rng.random() < 0.5 else [str(rng.choice(list(E.BUTTONS)))]
        if mode == "dead":
            dead = int(rng.integers(1, 3))
            self.faulty = {b: dict(dead=dead, presses=0, down=False) for b in buttons}
            out = dict(buttons=buttons, dead_presses=dead)
        else:
            factor = rng.uniform(0.35, 0.6)
            self.faulty = {b: dict(factor=factor) for b in buttons}
            out = dict(buttons=buttons, speed_factor=factor)
        env.flags.add("sticky")
        return out

    def substep(self, env):
        if not self.faulty:
            return
        turn = 0.0
        for name, (_, sign) in E.BUTTONS.items():
            travel = env.button_travel(name)
            pressed = travel > E.PRESS_TRAVEL
            f = self.faulty.get(name)
            gain = 1.0
            if f is not None and "dead" in f:
                if pressed and not f["down"]:
                    f["presses"] += 1
                    f["down"] = True
                elif travel < STICKY_RELEASE:
                    f["down"] = False
                gain = 0.0 if f["presses"] <= f["dead"] else 1.0
            elif f is not None:
                gain = f["factor"]
            turn += sign * gain if pressed else 0.0
        env.data.ctrl[env.turntable] = E.TURN_SPEED * turn


# -------------------------------------------------------------------------------------------- A5 nothing to do
HOME_TOL = 0.05  # metres: the gripper counts as home this close to where it is at the home pose
HOME_HOLD = 20  # control steps at home that end the episode as a success
NOTHING_MAX_STEPS = 300


class NothingToDo(Scenario):
    """An already pruned plant: every side shoot was cut earlier (stubs of 2-5 cm remain, the cut parts are gone).
    The instruction is the task's own. Success: no damage, and the gripper back at the home pose (within HOME_TOL) and
    held there for HOME_HOLD steps; the episode limit is NOTHING_MAX_STEPS. env.flags holds 'home' (the expert goes
    home when nothing is left to cut)."""

    name, catalogue = "nothing_to_do", "A5"
    modes = ("pruned",)
    changes_task = True

    def check(self, env):
        super().check(env)
        if env.named or env.reason:
            raise ValueError(f"nothing_to_do applies to prune_one, prune_turn and prune, not {env.task}")

    def init(self, env):
        d = mujoco.MjData(env.model)
        d.qpos[:7] = E.HOME_Q
        mujoco.mj_kinematics(env.model, d)
        env.home_tcp = d.site_xpos[env.tcp_id].copy()

    def restore(self, env):
        self.home = 0

    def draw(self, env, rng, mode):
        d = env.data
        cuts = {}
        for k, name in enumerate(env.shoot_names):
            if name not in env.shoots:
                continue
            s = rng.uniform(*E.CUT_ZONE)
            end = E.CUT_GAP + E.SHOOT_R
            env.shoots[name]["cut_s"] = s
            env._shape_shoot(name, s - end, s + end)
            adr, vadr = env.obj_qadr[name], env.obj_vadr[name]
            d.qpos[adr : adr + 7] = [2.0 + 0.3 * k, -2.5, -0.75, 1, 0, 0, 0]  # cleared away, out of every view
            d.qvel[vadr : vadr + 6] = 0
            d.eq_active[env.shoot_weld[name]] = 0
            env.cut.add(name)
            cuts[name] = s
        env._set_exclusions()
        env.max_steps = NOTHING_MAX_STEPS if env._max_steps_given is None else env.max_steps
        env.flags.add("home")
        self.precut = set(env.cut)
        return dict(stubs=len(cuts), stub_lengths_m=list(cuts.values()))

    def result(self, env, done, info):
        new_cuts = sorted(set(env.cut) - self.precut)
        home = float(np.linalg.norm(env.tcp_pos() - env.home_tcp))
        self.home = self.home + 1 if home < HOME_TOL else 0
        success = env.damaged is None and not new_cuts and self.home >= HOME_HOLD
        done = success or env.damaged is not None or bool(new_cuts) or env.steps >= env.max_steps
        info = dict(
            info, success=success, wrong_object=env.damaged is not None or bool(new_cuts), home_m=round(home, 4)
        )
        return done, info


# ----------------------------------------------------------------------------------------------- A8 order matters
ORDER_ALL = ("top_down", "bottom_up")
ORDER_FRAMES = {
    "top_down": (
        "cut the side shoots from the top down",
        "prune the shoots starting with the highest and working down",
        "cut the highest shoot first, then the next one down",
    ),
    "bottom_up": (
        "cut the side shoots from the bottom up",
        "prune the shoots starting with the lowest and working up",
        "cut the lowest shoot first, then the next one up",
    ),
    "pair": (
        "cut the {a} shoot, then the {b} one",
        "first cut the {a} shoot, after that the {b} one",
        "start with the {a} shoot, then cut the {b} shoot",
    ),
}
ORDER_HELDOUT_FRAMES = {
    "top_down": ("begin at the top and work your way down the shoots", "snip the branches one by one, top to bottom"),
    "bottom_up": ("begin at the bottom and work your way up the shoots", "snip the branches one by one, bottom to top"),
    "pair": ("the {a} shoot goes first and the {b} shoot second", "remove the {a} branch before the {b} branch"),
}


class Order(Scenario):
    """The instruction gives an order: every shoot from the top down or from the bottom up, or two named shoots one
    after the other ('pair'; any other shoot stays on the plant). Success: the shoots meant, cut in that order, nothing
    else cut, no damage. Cutting a shoot before one that should come first, or a shoot not meant, ends the episode as a
    failure (wrong_object; info['out_of_order'] tells the two apart). env.targets lists the shoots in order and
    env.flags holds 'ordered' (the expert cuts them in that order)."""

    name, catalogue = "order", "A8"
    modes = ("top_down", "bottom_up", "pair")
    changes_task = True
    frames = ORDER_FRAMES
    split = "train"

    def check(self, env):
        super().check(env)
        if env.named or env.reason or E.TASKS[env.task]["shoots"][0] < 2:
            raise ValueError(f"{self.name} applies to prune (2-3 shoots), not {env.task}")

    def restore(self, env):
        self.order, self.cut_seq = [], []

    def draw(self, env, rng, mode):
        shoots = list(env.active)  # from the lowest up
        names = [E.phrasings.position_name(i, len(shoots)) for i in range(len(shoots))]
        frame = self.frames[mode][int(rng.integers(len(self.frames[mode])))]
        if mode == "top_down":
            order = shoots[::-1]
            instruction = frame
        elif mode == "bottom_up":
            order = shoots
            instruction = frame
        else:
            i, j = rng.choice(len(shoots), size=2, replace=False)
            order = [shoots[i], shoots[j]]
            instruction = frame.format(a=names[i], b=names[j])
        self.order = order
        env.targets = list(order)
        env.target = order[0]
        env.instruction = instruction
        env.target_label = ",".join(names[shoots.index(n)] for n in order)
        env.phrasing_kind = self.split
        env.flags.add("ordered")
        return dict(instruction=instruction, order=env.target_label, shoots=len(shoots))

    def result(self, env, done, info):
        for n in sorted(set(env.cut) - set(self.cut_seq)):
            self.cut_seq.append(n)
        wrong = sorted(n for n in env.cut if n not in self.order)
        meant = [n for n in self.cut_seq if n in self.order]
        in_order = meant == self.order[: len(meant)]
        success = env.damaged is None and not wrong and in_order and len(meant) == len(self.order)
        done = success or bool(wrong) or not in_order or env.damaged is not None or env.steps >= env.max_steps
        info = dict(
            info,
            success=success,
            wrong_object=bool(wrong) or not in_order,
            wrong_shoots=wrong,
            out_of_order=not in_order,
            cut_order=list(self.cut_seq),
        )
        return done, info


class OrderHeldout(Order):
    name = "order_heldout"
    frames = ORDER_HELDOUT_FRAMES
    split = "heldout"


# ======================================================================================== situations with new assets
# The meshes and textures of sim/pickup/assets/meshes/situations/ (built by assets/build_situations_blender.py, which
# writes manifest.json: collision shapes, colours, where each asset attaches). MuJoCo moves a mesh's frame to its centre
# of mass and principal axes when it compiles, so a geom whose OBJ frame should be at (p, R) gets
# geom_pos = p + R @ mesh_pos and geom_quat = R * mesh_quat (_obj_pose).
SITUATIONS = E.ASSETS / "meshes" / "situations"
LEAF_SCALE = E.LEAF_SCALE


@cache
def manifest():
    return json.loads((SITUATIONS / "manifest.json").read_text())


def _add_mesh(spec, name, scale=1.0):
    if spec.mesh(name) is not None:
        return
    mesh = spec.add_mesh()
    mesh.name, mesh.file = name, str(SITUATIONS / f"{name}.obj")
    if scale != 1.0:
        mesh.scale = [scale] * 3


def _mat(quat):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    return R.reshape(3, 3)


def _obj_pose(m, g, p, R):
    """Draw mesh geom g with its OBJ frame at (p, R) in its body's frame."""
    mid = m.geom_dataid[g]
    m.geom_pos[g] = p + R @ m.mesh_pos[mid]
    m.geom_quat[g] = E._quat(R @ _mat(m.mesh_quat[mid]))


def _drawn_frame(m, g):
    """Where mesh geom g's OBJ frame is drawn (p, R) in its body's frame, whatever pose was written to it."""
    mid = m.geom_dataid[g]
    R = _mat(m.geom_quat[g]) @ _mat(m.mesh_quat[mid]).T
    return m.geom_pos[g] - R @ m.mesh_pos[mid], R


def _visual(body, name, mesh, rgba, group=3):
    """A visual-only mesh geom (built hidden: group 3)."""
    return body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh,
        rgba=list(rgba),
        contype=0,
        conaffinity=0,
        group=group,
        density=0,
    )


def _show(m, geoms, on):
    m.geom_group[list(geoms)] = 2 if on else 3


def _leaf_pose(env, name, j, yaw, pitch, roll):
    """Put leaf j of a shoot at the tip as env._reset_plant does (same convention for the pose it writes), in use."""
    m = env.model
    vis, col = env.leaf_geoms[name][j]
    base = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=float)
    R = base @ E._rot("z", yaw) @ E._rot("y", pitch) @ E._rot("x", roll)
    pos = np.array([0.0, 0.0, env.shoots[name]["length"] - E.CUT_MID - 0.001])
    m.geom_pos[vis], m.geom_quat[vis] = pos, E._quat(R)
    m.geom_pos[col], m.geom_quat[col] = pos + R @ [0.028, 0, 0.002], E._quat(R)
    m.geom_group[vis] = 2
    m.geom_contype[col], m.geom_conaffinity[col] = 2, 1


def _leaves_in_use(env, name):
    return [vis for vis, _ in env.leaf_geoms[name] if env.model.geom_group[vis] == 2]


class ChooseShoots(Scenario):
    """Base of the situations whose instruction picks shoots by something seen on them (a tie, sick leaves, a bud).
    Success: exactly the shoots meant are cut (any order), nothing else, no damage. When no shoot is meant, success is
    as in nothing_to_do (no cut, no damage, the arm home). env.targets lists the shoots meant; env.flags holds 'targets'
    and 'home' (the expert cuts those, then goes home if there were none)."""

    changes_task = True
    frames: dict = {}
    split = "train"

    def check(self, env):
        super().check(env)
        if env.named or env.reason:
            raise ValueError(f"{self.name} applies to prune_one, prune_turn and prune, not {env.task}")

    def init(self, env):
        NothingToDo.init(self, env)

    def restore(self, env):
        self.home = 0

    def choose(self, env, rng, mode):
        """Mark the shoots; returns (shoots meant, instruction, variation)."""
        raise NotImplementedError

    def draw(self, env, rng, mode):
        targets, instruction, out = self.choose(env, rng, mode)
        env.targets = list(targets)
        env.target = targets[0] if targets else env.active[0]
        env.instruction, env.phrasing_kind = instruction, self.split
        env.target_label = ",".join(E.phrasings.position_name(env.active.index(n), len(env.active)) for n in targets)
        env.flags |= {"targets", "home"}
        return dict(out, instruction=instruction, targets=env.target_label, shoots=len(env.active))

    def _frame(self, rng, kind):
        frames = self.frames[kind]
        return frames[int(rng.integers(len(frames)))]

    def result(self, env, done, info):
        wrong = sorted(n for n in env.cut if n not in env.targets)
        if env.targets:
            success = env.damaged is None and not wrong and all(n in env.cut for n in env.targets)
        else:
            home = float(np.linalg.norm(env.tcp_pos() - env.home_tcp))
            self.home = self.home + 1 if home < HOME_TOL else 0
            success = env.damaged is None and not wrong and self.home >= HOME_HOLD
            info = dict(info, home_m=round(home, 4))
        done = success or bool(wrong) or env.damaged is not None or env.steps >= env.max_steps
        info = dict(info, success=success, wrong_object=bool(wrong), wrong_shoots=wrong)
        return done, info


# ---------------------------------------------------------------------------------------------------------- A1 ties
TIE_SCALE = 1.8  # the tie as modelled is 2-3 pixels wide in the policy's 224-pixel view; scaled, about 6
TIE_AT = 0.058  # metres from the main stem's axis along the shoot: beyond the cut zone, on the part that falls
TIE_FRAMES = {
    "colour": ("cut the shoot with the {c} tie", "prune the shoot tied with {c}", "remove the shoot with the {c} tie"),
    "one": ("cut the shoot with the tie", "prune the tied shoot", "remove the shoot that has a tie"),
}
TIE_HELDOUT_FRAMES = {
    "colour": ("snip the branch wearing the {c} ribbon", "the shoot with the {c} tie has to go"),
    "one": ("snip the branch wearing a ribbon", "the tied shoot has to go"),
}


class Ties(ChooseShoots):
    """A1, gardener's ties: 'colour' - every shoot has a tie, each of another colour (red, blue, yellow, white), and the
    instruction names one colour; 'one' - one shoot has a tie, and the instruction asks for the tied shoot."""

    name, catalogue = "ties", "A1"
    modes = ("colour", "one")
    frames = TIE_FRAMES

    def build(self, spec):
        _add_mesh(spec, "sit_tie", TIE_SCALE)
        for k in range(E.MAX_SHOOTS):
            g = _visual(spec.body(f"shoot{k}"), f"shoot{k}_tie", "sit_tie", (1, 1, 1, 1))
            g.pos = [0, 0, TIE_AT - E.CUT_MID]  # (the OBJ frame is set at init, once compiled)

    def init(self, env):
        super().init(env)
        m = env.model
        self.ties = {n: m.geom(f"{n}_tie").id for n in env.shoot_names}
        for g in self.ties.values():
            _obj_pose(m, g, np.array([0.0, 0.0, TIE_AT - E.CUT_MID]), np.eye(3))
        m.geom_sameframe[list(self.ties.values())] = 0

    def restore(self, env):
        super().restore(env)
        _show(env.model, self.ties.values(), False)

    def choose(self, env, rng, mode):
        colours = manifest()["assets"]["tie"]["colours"]
        names = list(colours)
        shoots = list(env.active)
        target = shoots[int(rng.integers(len(shoots)))]
        tied = shoots if mode == "colour" else [target]
        picked = [names[i] for i in rng.choice(len(names), size=len(tied), replace=False)]
        for n, c in zip(tied, picked, strict=True):
            env.model.geom_rgba[self.ties[n]] = colours[c]
            _show(env.model, [self.ties[n]], True)
        colour = picked[tied.index(target)]
        instruction = self._frame(rng, mode).format(c=colour)
        return [target], instruction, dict(colour=colour, ties={n: c for n, c in zip(tied, picked, strict=True)})


class TiesHeldout(Ties):
    name, catalogue = "ties_heldout", "A1"
    frames = TIE_HELDOUT_FRAMES
    split = "heldout"


# --------------------------------------------------------------------------------------------------- A3 sick shoots
SICK_FRAMES = ("cut the sick shoots, keep the healthy ones", "prune every sick shoot", "remove the diseased shoots")
SICK_HELDOUT_FRAMES = ("snip off the ailing branches and spare the healthy ones", "the unhealthy shoots have to go")


class Sick(ChooseShoots):
    """A3, sick shoots only: 0 to all of the shoots are sick, each with one look on all its leaves: brown 'spots',
    brown 'edges' (both over a yellowish leaf) or 'wilted' (a drooping yellowish leaf). Cut the sick ones."""

    name, catalogue = "sick", "A3"
    modes = ("sick",)
    frames = {"sick": SICK_FRAMES}
    looks = ("spots", "edges", "wilted")

    def build(self, spec):
        for mesh in ("sit_leaf_spots", "sit_leaf_edges", "sit_leaf_wilted"):
            _add_mesh(spec, mesh, LEAF_SCALE)
        a = manifest()["meshes"]
        for k in range(E.MAX_SHOOTS):
            body = spec.body(f"shoot{k}")
            for j in range(2):
                for layer in ("spots", "edges"):
                    _visual(body, f"shoot{k}_leaf{j}_{layer}", f"sit_leaf_{layer}", a[f"sit_leaf_{layer}"]["rgba"])

    def init(self, env):
        super().init(env)
        m = env.model
        self.layers = {
            (n, j, layer): m.geom(f"{n}_leaf{j}_{layer}").id
            for n in env.shoot_names
            for j in range(2)
            for layer in ("spots", "edges")
        }
        m.geom_sameframe[list(self.layers.values())] = 0
        self.leaf_mesh, self.wilted_mesh = m.mesh("plant_leaf").id, m.mesh("sit_leaf_wilted").id
        self.leaf_rgba = m.geom_rgba[env.leaf_geoms[env.shoot_names[0]][0][0]].copy()

    def restore(self, env):
        super().restore(env)
        m = env.model
        _show(m, self.layers.values(), False)
        for n in env.shoot_names:
            for vis, _ in env.leaf_geoms[n]:
                m.geom_dataid[vis], m.geom_rgba[vis] = self.leaf_mesh, self.leaf_rgba

    def make_sick(self, env, name, look):
        m = env.model
        sick = manifest()["assets"]["sick_leaves"]["colours"]["sick_leaf"]
        for j, (vis, _) in enumerate(env.leaf_geoms[name]):
            if m.geom_group[vis] != 2:
                continue
            p, R = _drawn_frame(m, vis)
            m.geom_rgba[vis] = sick
            if look == "wilted":
                m.geom_dataid[vis] = self.wilted_mesh
                _obj_pose(m, vis, p, R)
            else:
                g = self.layers[(name, j, look)]
                _obj_pose(m, g, p, R)
                _show(m, [g], True)

    def choose(self, env, rng, mode):
        shoots = list(env.active)
        count = int(rng.integers(len(shoots) + 1))
        sick = [shoots[i] for i in sorted(rng.choice(len(shoots), size=count, replace=False))]
        looks = {}
        for n in sick:
            looks[n] = self.looks[int(rng.integers(len(self.looks)))]
            self.make_sick(env, n, looks[n])
        return sick, self._frame(rng, "sick"), dict(sick=count, looks=looks)


class SickHeldout(Sick):
    name, catalogue = "sick_heldout", "A3"
    frames = {"sick": SICK_HELDOUT_FRAMES}
    split = "heldout"


# --------------------------------------------------------------------------------------------------------- A4 buds
BUD_FRAMES = (
    "cut every shoot without a bud",
    "prune the shoots that have no bud",
    "keep the buds, cut the other shoots",
)
BUD_HELDOUT_FRAMES = ("spare the budding branches and snip the rest", "only the shoots carrying no bud have to go")


class Buds(ChooseShoots):
    """A4, spare the buds: 0 to all of the shoots carry a flower bud at the tip (and then two leaves, turned away from
    it to either side). Cut every shoot without a bud; with a bud on every shoot there is nothing to do."""

    name, catalogue = "buds", "A4"
    modes = ("buds",)
    frames = {"buds": BUD_FRAMES}

    def build(self, spec):
        a = manifest()
        for mesh in ("sit_bud_body", "sit_bud_sepals"):
            _add_mesh(spec, mesh)
        (shape,) = a["assets"]["bud"]["collision"]
        for k in range(E.MAX_SHOOTS):
            body = spec.body(f"shoot{k}")
            for part in ("body", "sepals"):
                _visual(body, f"shoot{k}_bud_{part}", f"sit_bud_{part}", a["meshes"][f"sit_bud_{part}"]["rgba"])
            E._solid(body, f"shoot{k}_bud_col", shape["type"], shape["size"], density=300, contype=0, conaffinity=0)

    def init(self, env):
        super().init(env)
        m = env.model
        self.buds = {n: [m.geom(f"{n}_bud_{p}").id for p in ("body", "sepals", "col")] for n in env.shoot_names}
        m.geom_sameframe[[g for gs in self.buds.values() for g in gs]] = 0

    def restore(self, env):
        super().restore(env)
        m = env.model
        for vis_body, vis_sepals, col in self.buds.values():
            _show(m, [vis_body, vis_sepals], False)
            m.geom_contype[col] = m.geom_conaffinity[col] = 0

    def choose(self, env, rng, mode):
        m = env.model
        shoots = list(env.active)
        count = int(rng.integers(len(shoots) + 1))
        budding = [shoots[i] for i in sorted(rng.choice(len(shoots), size=count, replace=False))]
        (shape,) = manifest()["assets"]["bud"]["collision"]
        for n in budding:
            tip = np.array([0.0, 0.0, env.shoots[n]["length"] - E.CUT_MID])
            vis_body, vis_sepals, col = self.buds[n]
            for g in (vis_body, vis_sepals):
                _obj_pose(m, g, tip, np.eye(3))
            _show(m, [vis_body, vis_sepals], True)
            m.geom_pos[col] = tip + shape["pos"]
            m.geom_contype[col], m.geom_conaffinity[col] = 2, 1
            for j, side in enumerate((-1, 1)):  # two leaves, turned away from the bud
                _leaf_pose(env, n, j, side * rng.uniform(0.5, 0.85), rng.uniform(0.25, 0.5), rng.uniform(-0.25, 0.25))
        targets = [n for n in shoots if n not in budding]
        return targets, self._frame(rng, "buds"), dict(buds=count)


class BudsHeldout(Buds):
    name, catalogue = "buds_heldout", "A4"
    frames = {"buds": BUD_HELDOUT_FRAMES}
    split = "heldout"


# --------------------------------------------------------------------------------------------- B3 bench and backdrop
TABLE_POS = (0.35, 0.0, -0.02)  # the table's visual geom (env.build_model)
BACKDROPS = (
    ((-0.75, 0.0, -0.76), np.pi / 2),
    ((0.2, 0.95, -0.76), 0.0),
)  # behind the scene, seen from the front camera


class Bench(Scenario):
    """B3, a new workbench: a wooden potting bench ('wood'), a soil-stained one ('stained'), the usual table in a
    greenhouse ('greenhouse': glass walls with plants behind, on two sides), or either bench in the greenhouse. The
    surfaces' colours are tinted a little per episode. Only what is seen changes (the table's collision box stays)."""

    name, catalogue = "bench", "B3"
    modes = ("wood", "stained", "greenhouse", "wood_greenhouse", "stained_greenhouse")
    needs_plant = False

    def build(self, spec):
        for mesh in ("sit_bench_top", "sit_backdrop_frame", "sit_backdrop_glass"):
            _add_mesh(spec, mesh)
        for tex in ("bench_wood", "bench_stained", "backdrop_greenhouse"):
            t = spec.add_texture()
            t.name, t.file, t.type = f"sit_{tex}", str(SITUATIONS / f"sit_{tex}.png"), mujoco.mjtTexture.mjTEXTURE_2D
            t.colorspace = mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR
            mat = spec.add_material()
            mat.name = f"sit_{tex}"
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"sit_{tex}"
        wb = spec.worldbody
        g = _visual(wb, "sit_bench", "sit_bench_top", (1, 1, 1, 1))
        g.pos, g.material = list(TABLE_POS), "sit_bench_wood"
        frame = manifest()["meshes"]["sit_backdrop_frame"]["rgba"]
        for i, (pos, yaw) in enumerate(BACKDROPS):
            q = list(E._quat(E._rot("z", yaw)))
            g = _visual(wb, f"sit_backdrop{i}_frame", "sit_backdrop_frame", frame)
            g.pos, g.quat = list(pos), q
            g = _visual(wb, f"sit_backdrop{i}_glass", "sit_backdrop_glass", (1, 1, 1, 1))
            g.pos, g.quat, g.material = list(pos), q, "sit_backdrop_greenhouse"

    def init(self, env):
        m = env.model
        self.table = m.geom("table_visual").id
        self.bench = m.geom("sit_bench").id
        self.backdrop = [m.geom(f"sit_backdrop{i}_{p}").id for i in range(2) for p in ("frame", "glass")]
        self.mats = {k: m.material(f"sit_{k}").id for k in ("bench_wood", "bench_stained", "backdrop_greenhouse")}
        self.mat_rgba = m.mat_rgba.copy()

    def restore(self, env):
        m = env.model
        _show(m, [self.table], True)
        _show(m, [self.bench, *self.backdrop], False)
        m.mat_rgba[:] = self.mat_rgba

    def draw(self, env, rng, mode):
        m = env.model
        out = {}
        if mode != "greenhouse":
            surface = "bench_stained" if mode.startswith("stained") else "bench_wood"
            m.geom_matid[self.bench] = self.mats[surface]
            tint = rng.uniform(0.85, 1.1, 3)
            m.mat_rgba[self.mats[surface], :3] = np.clip(tint, 0, 1)
            _show(m, [self.table], False)
            _show(m, [self.bench], True)
            out["tint"] = tint
        if mode.endswith("greenhouse"):
            _show(m, self.backdrop, True)
            light = rng.uniform(0.8, 1.0)
            m.mat_rgba[self.mats["backdrop_greenhouse"], :3] = light
            out["backdrop_light"] = light
        return out


# ------------------------------------------------------------------------------------------------- B4 distractors
DISTRACTORS = ("watering_can", "gloves", "trowel", "seed_packet", "pot_small")
DISTRACTOR_RING = (0.12, 0.40)  # metres from the stand's axis: clear of its base (radius 0.11) and on the table
DISTRACTOR_SIDE = np.radians(100)  # placed within this of the stand's +x (away from the robot's working side)
MOVED = 0.01  # metres: a distractor that has moved this far was disturbed


class Distractors(Scenario):
    """B4, distractors: two to four of a watering can, gardening gloves, a trowel, a seed packet and a small empty pot
    stand or lie on the table beside and behind the plant (solid, free to be pushed), and in half the episodes a bee
    sits on a leaf. The task is unchanged; info['distractors_touched'] lists those the robot has touched and
    info['distractors_moved'] those that moved more than MOVED."""

    name, catalogue = "distractors", "B4"
    modes = ("table", "table_bee")

    def build(self, spec):
        a = manifest()
        for i, name in enumerate(DISTRACTORS):
            asset = a["assets"][name]
            body = spec.worldbody.add_body(name=f"sit_{name}", pos=[2.0 + 0.4 * i, -3.0, -0.75])
            body.add_freejoint()
            for mesh in asset["meshes"]:
                _add_mesh(spec, mesh)
                _visual(body, f"sit_{name}_{mesh}", mesh, a["meshes"][mesh]["rgba"], group=2)
            shapes = asset["collision"]
            for j, c in enumerate(shapes):
                g = E._solid(body, f"sit_{name}_col{j}", c["type"], c["size"], c["pos"], c["quat"])
                g.mass = asset["mass"] / len(shapes)
        for mesh in a["assets"]["bee"]["meshes"]:
            _add_mesh(spec, mesh)
        for k in range(E.MAX_SHOOTS):
            for mesh in a["assets"]["bee"]["meshes"]:
                _visual(spec.body(f"shoot{k}"), f"shoot{k}_{mesh}", mesh, a["meshes"][mesh]["rgba"])

    def init(self, env):
        m = env.model
        self.bodies = {n: m.body(f"sit_{n}").id for n in DISTRACTORS}
        self.qadr = {n: m.joint(m.body(f"sit_{n}").jntadr[0]).qposadr[0] for n in DISTRACTORS}
        self.vadr = {n: m.joint(m.body(f"sit_{n}").jntadr[0]).dofadr[0] for n in DISTRACTORS}
        bee = manifest()["assets"]["bee"]["meshes"]
        self.bee = {n: [m.geom(f"{n}_{mesh}").id for mesh in bee] for n in env.shoot_names}
        m.geom_sameframe[[g for gs in self.bee.values() for g in gs]] = 0
        # the footprint radius of each, from its meshes' bounds (seen from above)
        meshes = manifest()["meshes"]
        self.radius = {
            n: max(
                np.linalg.norm(np.maximum(np.abs(meshes[x]["min"][:2]), np.abs(meshes[x]["max"][:2])))
                for x in manifest()["assets"][n]["meshes"]
            )
            for n in DISTRACTORS
        }
        self.robot = set(env.robot_bodies.tolist())

    def restore(self, env):
        d = env.data
        for i, n in enumerate(DISTRACTORS):
            adr, vadr = self.qadr[n], self.vadr[n]
            d.qpos[adr : adr + 7] = [2.0 + 0.4 * i, -3.0, -0.75, 1, 0, 0, 0]
            d.qvel[vadr : vadr + 6] = 0
        for gs in self.bee.values():
            _show(env.model, gs, False)
        self.placed, self.touched = {}, set()

    def draw(self, env, rng, mode):
        d = env.data
        stand = env.model.body_pos[env.stand_body][:2]
        count = int(rng.integers(2, 5))
        chosen = [DISTRACTORS[i] for i in rng.choice(len(DISTRACTORS), size=count, replace=False)]
        placed = {}
        for n in chosen:
            r_obj = self.radius[n]
            for _ in range(200):
                angle = rng.uniform(-DISTRACTOR_SIDE, DISTRACTOR_SIDE)
                dist = rng.uniform(DISTRACTOR_RING[0] + r_obj, DISTRACTOR_RING[1])
                xy = stand + dist * np.array([np.cos(angle), np.sin(angle)])
                on_table = xy[0] + r_obj <= 0.74 and abs(xy[1]) + r_obj <= 0.58
                apart = all(np.linalg.norm(xy - p) >= r_obj + self.radius[o] for o, (p, _) in placed.items())
                if on_table and apart:
                    placed[n] = (xy, rng.uniform(-np.pi, np.pi))
                    break
        for n, (xy, yaw) in placed.items():
            adr = self.qadr[n]
            d.qpos[adr : adr + 7] = [xy[0], xy[1], 0.001, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.placed = {n: np.array([xy[0], xy[1]]) for n, (xy, _) in placed.items()}
        out = dict(objects={n: [*xy, yaw] for n, (xy, yaw) in placed.items()})
        if mode == "table_bee":
            leaves = [(n, vis) for n in env.active for vis in _leaves_in_use(env, n)]
            n, vis = leaves[int(rng.integers(len(leaves)))]
            p, R = _drawn_frame(env.model, vis)
            R_bee = R @ E._rot("z", rng.uniform(-0.6, 0.6))
            p_bee = p + R @ (np.array([0.03, 0.0, 0.004]) * LEAF_SCALE)
            for g in self.bee[n]:
                _obj_pose(env.model, g, p_bee, R_bee)
            _show(env.model, self.bee[n], True)
            out["bee_on"] = n
        return out

    def after_step(self, env):
        m, d = env.model, env.data
        n = d.ncon
        if n:
            b1, b2 = m.geom_bodyid[d.contact.geom1[:n]], m.geom_bodyid[d.contact.geom2[:n]]
            for name in self.placed:
                body = self.bodies[name]
                hit = (np.isin(b1, list(self.robot)) & (b2 == body)) | (np.isin(b2, list(self.robot)) & (b1 == body))
                if hit.any():
                    self.touched.add(name)

    def result(self, env, done, info):
        moved = sorted(
            n
            for n, xy in self.placed.items()
            if np.linalg.norm(env.data.qpos[self.qadr[n] : self.qadr[n] + 2] - xy) > MOVED
        )
        return done, dict(info, distractors_touched=sorted(self.touched), distractors_moved=moved)


SCENARIOS = {
    s.name: s
    for s in (
        Lighting,
        Camera,
        Wind,
        Bump,
        Sticky,
        NothingToDo,
        Order,
        OrderHeldout,
        Ties,
        TiesHeldout,
        Sick,
        SickHeldout,
        Buds,
        BudsHeldout,
        Bench,
        Distractors,
    )
}


def parse(spec) -> list[Scenario]:
    """Scenario instances from a spec: None or "" (none), "name[:mode][@p],..." or a list of such strings."""
    if not spec:
        return []
    parts = spec if isinstance(spec, (list, tuple)) else [p for p in str(spec).split(",")]
    out = []
    for part in (p.strip() for p in parts):
        if not part:
            continue
        part, _, prob = part.partition("@")
        name, _, mode = part.partition(":")
        name = CATALOGUE.get(name.upper(), name)
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)} (or {sorted(CATALOGUE)})")
        out.append(SCENARIOS[name](mode or None, float(prob) if prob else 1.0))
    names = [s.name for s in out]
    if len(set(names)) != len(names):
        raise ValueError(f"a scenario is given twice: {spec}")
    if sum(s.changes_task for s in out) > 1:
        raise ValueError(f"these scenarios change the task in ways that cannot be combined: {spec}")
    return sorted(out, key=lambda s: s.priority)
