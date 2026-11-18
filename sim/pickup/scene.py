"""Scene records: what is on the stand now, as a short list of typed fields, and a memory of it over an episode.

Architecture v2 (tutorial/architecture-v2.md) splits the slow level into eyes that turn the camera images into a
description of the scene, a scene memory without weights, and a reasoner that reads text. This module holds the
description, the simulator's truth for it and the memory.

A *scene* (Scene) is what the stand shows now: the side shoots on the plant, lowest first (Part: where it leaves the
main stem, how far the plate has to turn to bring it in front of the robot, whether it has been cut, whether a camera
shows it), the flower's colour and the card in the pot (LABELS: no card, a card that does not show its face to the
front camera, or the colour of its face). The simulator's truth (scene_truth) and what a perception model reads from
the images (a *percept*) have the same shape. The eyes are trained to write the truth's fields, and the memory and
everything after it take either without knowing which it was given, so a part of the system can be tested on the truth
first and on the eyes after. A percept adds how sure the model is (p, p_cut, p_visible, p_flower, p_label); the truth
adds which camera shows each part (views), found by casting rays from each camera to points along the part. As plain
dicts (to_dict, to_json) they are what a dataset stores and a server sends.

The memory (SceneMemory) keeps the plant's inventory over an episode: every part seen, each with an id that is the
order in which it was found and does not change. A part is known by its height on the main stem, because that is
what does not change: its bearing changes whenever the plate turns, it leaves the view whenever the stem, a leaf or
the arm is in the way, and when it is cut its stub stays at that height. The memory is updated only from what is seen,
and only from evidence that is confident and repeated: a part is added once it has been seen `stable` updates in a
row, and marked cut once it has been seen cut that often. Nothing is removed, and nothing is marked cut because it went
out of view or because an action was meant to cut it: a plan made from the memory follows what happened, not what was
intended. Two options, off by default, answer errors the eyes make (SceneMemory, rules 1 to 3): `unseen` takes a part
they are sure of but judge no camera to show as being there, and `revoke` unmarks a cut when enough confident
sightings in a row say the part is not cut.

The memory writes itself as the expert's notes write what they see (reasoning.note_text, field `see`), so a reasoner
reads the same line whether the notes or the memory wrote it, with the shoots cut added:

  3 shoots, purple flower, label hidden, read cyan; cut: the lowest

The records and the memory are pure Python with numpy. scene_truth and PlantCrop (the front camera's view narrowed onto
the plant, as a crop of a higher-resolution image would show it) import the simulator when they are used.
"""

import json
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import reasoning

# The card in the pot: none, one that does not show its face to the front camera, or the colour of its face
LABELS = ("none", "unreadable", "white", "yellow", "cyan")
NO_COLOUR = ("none", "unreadable")
# Architecture v3: what a program can ask about a shoot (program.DOMAINS), each with a class of its own for "cannot be
# read in this view", which is never a value of the shoot. The truth writes tie, sick and bud (tag belongs to rung 6,
# whose situation is not built); a plant without a situation that draws one has no tie, no sick shoot and no bud.
UNREADABLE = "unreadable"
TRUTH_ATTRIBUTES = ("tie", "sick", "bud")
NO_ATTRIBUTES = {"tie": "none", "sick": "no", "bud": "no"}
# SceneMemory with `unseen`: a sighting founds a new part only at least this many `match` from every remembered
# part. Shoots are at least 2 * match apart; the half between is left for the error of a height
SEPARATE = 1.5
# SceneMemory's rules for an attribute (architecture v3): "row" (stable readings in a row, as every fact of v2), "last"
# (the last reading), "angles" (a row whose readings were taken with the plate at least ANGLES_APART apart: looks from
# one angle are one witness)
ATTRIBUTE_RULES = ("row", "last", "angles")
ANGLES_APART = float(np.radians(30))


def _r(x) -> float:
    """A plain float rounded to 4 decimals, as the dict form stores numbers."""
    return round(float(x), 4)


# ------------------------------------------------------------------------------------------------------ records
@dataclass
class Part:
    """One side shoot. In the truth `p` is 1 and `views` says which cameras show it; in a percept `views` is empty and
    the probabilities say how sure the eyes are."""

    height: float  # metres above the base of the main stem (env.STEM_BASE); 0.07 to 0.155 on these plants
    bearing: float  # radians: the turn of the plate (+ counter-clockwise, the blue button) that brings it to face the
    # robot; 0 when it faces the robot. As notes.py works it out: _wrap_angle(REACH_AZIMUTH - env.shoot_azimuth(name))
    cut: bool  # cut off: only its stub is on the plant
    visible: bool  # a camera shows it now (front or wrist)
    p: float = 1.0  # percepts: the probability that the part exists; truth: 1.0
    p_cut: float | None = None  # percepts: the probability that it is cut
    p_visible: float | None = None  # percepts: the probability that a camera shows it
    views: tuple[bool, ...] = ()  # truth: seen by (front, wrist); empty in percepts
    # e.g. {"tie": "red", "sick": "no", "bud": UNREADABLE}: a value of program.DOMAINS, or UNREADABLE where this view
    # cannot show it (scene_truth with attributes)
    attributes: dict[str, str] = field(default_factory=dict)
    p_attributes: dict[str, float] = field(default_factory=dict)  # percepts: the probability of each class given
    # truth with attribute_views: which cameras (front, wrist) can read each attribute now
    attribute_views: dict[str, tuple[bool, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """The plain form: a truth part has "vis" [front, wrist]; a percept part has "vis" (one boolean) and "p",
        "p_cut", "p_vis" (those that are set)."""
        out = {"h": _r(self.height), "b": _r(self.bearing), "cut": bool(self.cut)}
        if self.views:
            out["vis"] = [bool(v) for v in self.views]
        else:
            out["vis"] = bool(self.visible)
            out["p"] = _r(self.p)
        if self.p_cut is not None:
            out["p_cut"] = _r(self.p_cut)
        if self.p_visible is not None:
            out["p_vis"] = _r(self.p_visible)
        if self.attributes:
            out["attr"] = dict(self.attributes)
        if self.p_attributes:
            out["p_attr"] = {k: _r(v) for k, v in self.p_attributes.items()}
        if self.attribute_views:
            out["attr_vis"] = {k: [bool(x) for x in v] for k, v in self.attribute_views.items()}
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Part":
        vis = d.get("vis", False)
        views = tuple(bool(v) for v in vis) if isinstance(vis, (list, tuple)) else ()
        return cls(
            height=float(d["h"]),
            bearing=float(d["b"]),
            cut=bool(d["cut"]),
            visible=any(views) if views else bool(vis),
            p=float(d.get("p", 1.0)),
            p_cut=None if d.get("p_cut") is None else float(d["p_cut"]),
            p_visible=None if d.get("p_vis") is None else float(d["p_vis"]),
            views=views,
            attributes={str(k): str(v) for k, v in d.get("attr", d.get("attributes", {})).items()},
            p_attributes={str(k): float(v) for k, v in d.get("p_attr", {}).items()},
            attribute_views={str(k): tuple(bool(x) for x in v) for k, v in d.get("attr_vis", {}).items()},
        )


@dataclass
class Scene:
    """What is on the stand now."""

    parts: list[Part]  # lowest first
    flower: str | None  # one of reasoning.FLOWERS, or None when not known
    label: str  # one of LABELS
    p_flower: float = 1.0  # percepts: the probability of the class given
    p_label: float = 1.0

    def __post_init__(self):
        assert self.label in LABELS, f"label {self.label!r} is not one of {LABELS}"
        assert self.flower is None or self.flower in reasoning.FLOWERS, f"flower {self.flower!r}"

    def to_dict(self) -> dict:
        """The plain form: what datasets store and servers send. Floats are rounded to 4 decimals; a percept also
        carries "p_flower" and "p_label"."""
        out = {"parts": [p.to_dict() for p in self.parts], "flower": self.flower, "label": self.label}
        percept = any(not p.views for p in self.parts) or self.p_flower != 1.0 or self.p_label != 1.0
        if percept:
            out["p_flower"], out["p_label"] = _r(self.p_flower), _r(self.p_label)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        """Reads the truth form and the percept form alike."""
        return cls(
            parts=[Part.from_dict(p) for p in d.get("parts", [])],
            flower=d.get("flower"),
            label=d.get("label", "none"),
            p_flower=float(d.get("p_flower", 1.0)),
            p_label=float(d.get("p_label", 1.0)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "Scene":
        return cls.from_dict(json.loads(text))


# ----------------------------------------------------------------------------------------- truth from the simulator
# Which camera shows a part, without rendering: a camera sees a point when the point lies in its field of view and a
# ray from the camera to the point meets nothing else first. Rays are cast against what the cameras draw, geom groups
# 0-2 (the collision geoms are group 3 and are not drawn). The points lie along what is on the plant: an attached shoot
# from where its outer part begins (SHOOT_SPLIT) out to its tip, a cut shoot along its stub.
DRAWN_GROUPS = (1, 1, 1, 0, 0, 0)  # mjvOption.geomgroup of the renderers: groups 0, 1 and 2
SHOOT_POINTS = 5  # points along an attached shoot
STUB_POINTS = 3  # points along the stub of a cut shoot
STUB_START = 0.01  # metres from the main stem's axis: the first point on a stub, beyond the stem as drawn (radius 4 to
# 5.5 mm, its centre line wandering by up to 2.5 mm), which would otherwise hide it from every side
SEEN_POINTS = 2  # a part is seen by a camera when at least this many of its points are
# A ray aimed at a point on a part's axis meets the part's own surface before it: SHOOT_R (3 mm) before the point, more
# where the ray runs along the shoot. It reaches the point when its first hit is one of the part's own geoms (the stub,
# the outer part, its leaves and whatever a scenario hangs on it), or anything else no more than this short of it.
SURFACE_TOL = 0.01  # metres
CROP_HEIGHT = 0.10  # metres above the main stem's base: the plant crop's centre, about the middle of the shoots


class _Camera:
    """A camera of the model as it is now: projects points into its image and casts rays to them."""

    def __init__(self, env, name: str):
        import mujoco

        m, d = env.model, env.data
        cam = m.camera(name).id
        self.model, self.data, self._ray = m, d, mujoco.mj_ray
        self.pos = d.cam_xpos[cam].copy()
        self.R = d.cam_xmat[cam].reshape(3, 3).copy()  # columns: image right, image up, backward (it looks along -z)
        self.focal = 1.0 / np.tan(np.radians(m.cam_fovy[cam]) / 2)  # the image's edges at -1 and 1 (square images)
        self.near = m.vis.map.znear * m.stat.extent  # the renderer draws nothing nearer than this
        body = int(m.cam_bodyid[cam])
        self.exclude = body if body > 0 else -1  # a camera on the hand (the wrist camera) does not see the hand itself
        self._groups = np.array(DRAWN_GROUPS, dtype=np.uint8)
        self._geom = np.zeros(1, dtype=np.int32)

    def project(self, point) -> tuple[float, float, float]:
        """Image coordinates (x to the right, y down, both in [-1, 1] when in view) and depth (metres)."""
        rel = self.R.T @ (np.asarray(point, dtype=float) - self.pos)
        depth = float(-rel[2])
        scale = self.focal / max(depth, 1e-9)
        return float(rel[0] * scale), float(-rel[1] * scale), depth

    def sees(self, point, own: set[int], tol: float = SURFACE_TOL) -> bool:
        """Is `point` in view, with nothing in front of it but `own` geoms (or anything within `tol` of it)?"""
        x, y, depth = self.project(point)
        if depth <= self.near or abs(x) > 1.0 or abs(y) > 1.0:
            return False
        vec = np.asarray(point, dtype=float) - self.pos
        dist = float(np.linalg.norm(vec))
        hit = self._ray(self.model, self.data, self.pos, vec / dist, self._groups, 1, self.exclude, self._geom)
        return hit < 0 or int(self._geom[0]) in own or hit >= dist - tol


def _part_points(env, name: str) -> np.ndarray:
    """Points along what is on the plant of shoot `name` (world frame, n x 3)."""
    from .env import CUT_GAP, CUT_MID, SHOOT_R, SHOOT_SPLIT

    shoot = env.shoots[name]
    if name in env.cut:  # the stub, from the main stem's axis to the cut (its rounded end reaches SHOOT_R further)
        a, d = env.shoot_axis(name)
        end = max(shoot["cut_s"] - (CUT_GAP + SHOOT_R), STUB_START)
        return a + np.linspace(STUB_START, end, STUB_POINTS)[:, None] * d
    # The outer part, where it is drawn: its body's z axis runs along the shoot, its origin CUT_MID out from the stem
    body = env.obj_body[name]
    s = np.linspace(SHOOT_SPLIT, shoot["length"], SHOOT_POINTS)
    return env.data.xpos[body] + (s - CUT_MID)[:, None] * env.data.xmat[body].reshape(3, 3)[:, 2]


def _own_geoms(env, name: str) -> set[int]:
    """The geoms of shoot `name` that are on the plant: its stub, and while it is attached its outer part's body (the
    shoot, its leaves and anything a scenario hangs on it)."""
    own = {int(g) for g in env.stub_geoms[name]}
    if name not in env.cut:
        own |= {int(g) for g in np.flatnonzero(env.model.geom_bodyid == env.obj_body[name])}
    return own


def _label(env, front: _Camera) -> str:
    """The card: "none", its colour when it faces the front camera (env.label_visible) and nothing hides the centre of
    its face, else "unreadable"."""
    if env.label_colour is None:
        return "none"
    face = env.label_geoms[1]  # (collision box, coloured face, brown back)
    shown = env.label_visible() and front.sees(env.data.geom_xpos[face].copy(), {int(face)}, tol=0.0)
    return env.label_colour if shown else "unreadable"


# Where an attribute of an attached shoot is to be seen (architecture v3), by the same rays: a camera reads it when it
# sees enough points of one of its places. Only what carries the attribute counts as the shoot's own there (a leaf in
# front of the tip hides a bud as anything else would):
#   tie   the tie's place on the shoot (scenarios.TIE_AT): TIE_POINTS along the axis around it, SEEN_POINTS of them;
#         own: the shoot and the tie
#   bud   the shoot's tip, where a bud would stand: BUD_POINTS out along the axis from it, SEEN_POINTS of them; own:
#         the shoot and the bud
#   sick  any leaf: the centre of a leaf as drawn; own: that leaf and what a scenario draws on it
TIE_POINTS = (-0.008, 0.0, 0.008)  # metres along the shoot from TIE_AT (the tie is a band about 20 mm wide)
BUD_POINTS = (0.010, 0.018, 0.026)  # metres out along the shoot from its tip (a bud reaches 29 mm beyond it)


def _geom_ids(env, names) -> set[int]:
    import mujoco

    ids = (mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names)
    return {int(g) for g in ids if g >= 0}


def _attribute_places(env, name: str) -> dict[str, list[tuple[np.ndarray, set[int], int]]]:
    """For each attribute of attached shoot `name`, its places: (points, own geoms, points needed)."""
    from .env import CUT_MID
    from .scenarios import TIE_AT

    shoot, body = env.shoots[name], env.obj_body[name]
    origin, axis = env.data.xpos[body], env.data.xmat[body].reshape(3, 3)[:, 2]  # axis: along the shoot

    def along(s):
        return origin + (np.asarray(s) - CUT_MID)[:, None] * axis

    stem = _geom_ids(env, [f"{name}_vis"])
    tie = along(TIE_AT + np.array(TIE_POINTS)), stem | _geom_ids(env, [f"{name}_tie"]), SEEN_POINTS
    bud = (
        along(shoot["length"] + np.array(BUD_POINTS)),
        stem | _geom_ids(env, [f"{name}_bud_body", f"{name}_bud_sepals"]),
    )
    leaves = []
    for j, (vis, _) in enumerate(env.leaf_geoms[name]):
        if env.model.geom_group[vis] == 2:  # in use
            own = {int(vis)} | _geom_ids(env, [f"{name}_leaf{j}_spots", f"{name}_leaf{j}_edges"])
            leaves.append((env.data.geom_xpos[vis][None, :].copy(), own, 1))
    return {"tie": [tie], "bud": [(*bud, SEEN_POINTS)], "sick": leaves}


def attribute_views(env, name: str, cameras=None) -> dict[str, tuple[bool, ...]]:
    """Which cameras (default: front, wrist) can read each attribute of shoot `name` now (TRUTH_ATTRIBUTES). Nothing of
    a cut shoot can be read: its tie, leaves and bud fell with it."""
    cameras = cameras or (_Camera(env, "front"), _Camera(env, "wrist"))
    if name in env.cut:
        return {attr: tuple(False for _ in cameras) for attr in TRUTH_ATTRIBUTES}
    places = _attribute_places(env, name)
    return {
        attr: tuple(
            any(sum(camera.sees(p, own) for p in points) >= needed for points, own, needed in places[attr])
            for camera in cameras
        )
        for attr in TRUTH_ATTRIBUTES
    }


def drawn_attributes(env) -> dict[str, dict[str, str]]:
    """Every attribute of every shoot of the plant as the episode drew it (Scenario.attributes), seen or not: per shoot
    NO_ATTRIBUTES with what the situations drew over it."""
    out = {name: dict(NO_ATTRIBUTES) for name in env.active}
    for scenario in env.scenarios:
        if scenario.applied:
            for name, values in scenario.attributes(env).items():
                out[name].update(values)
    return out


def scene_truth(env, attributes: bool | None = None, full: bool = False, views: bool = False) -> Scene:
    """The scene as the simulator knows it (env: a PickupEnv of a plant task, in the state the cameras show now): one
    Part per shoot of the plant, lowest first, cut or not, with which cameras show it (views: front, wrist), the
    flower's colour and the card.

    `attributes` (architecture v3): each part's tie, sick and bud (drawn_attributes), UNREADABLE where no camera can
    read it (attribute_views) and for every attribute of a cut part. None, the default: only where a situation that
    draws attributes applies (ties, sick, buds), so that every other scene is what it was. `full`: the simulator's
    whole state, hidden facts included (architecture v3's second oracle): every part is shown by both cameras, every
    attribute is its value, cut or not, and the card's colour is given whichever way it faces. `views` (with
    attributes, not full): each part also says which cameras can read each attribute (Part.attribute_views), for
    measuring eyes that look through one camera."""
    from .env import REACH_AZIMUTH, STEM_BASE, _wrap_angle

    assert env.plant_task, f"{env.task} has no plant"
    if attributes is None:
        attributes = full or any(s.applied and s.attributes(env) for s in env.scenarios)
    front, wrist = _Camera(env, "front"), _Camera(env, "wrist")
    drawn = drawn_attributes(env) if attributes else {}
    views_asked = bool(views)  # (`views` is each part's cameras below)
    parts = []
    for name in env.active:  # from the lowest shoot up
        if full:
            views = (True, True)
        else:
            points, own = _part_points(env, name), _own_geoms(env, name)
            views = tuple(sum(camera.sees(p, own) for p in points) >= SEEN_POINTS for camera in (front, wrist))
        values, readable = {}, {}
        if attributes and full:
            values = dict(drawn[name])
        elif attributes:
            readable = attribute_views(env, name, (front, wrist))
            values = {attr: drawn[name][attr] if any(readable[attr]) else UNREADABLE for attr in TRUTH_ATTRIBUTES}
        parts.append(
            Part(
                height=float(env.shoots[name]["origin"][2] - STEM_BASE),
                bearing=float(_wrap_angle(REACH_AZIMUTH - env.shoot_azimuth(name))),
                cut=name in env.cut,
                visible=any(views),
                views=views,
                attributes=values,
                attribute_views=readable if views_asked else {},
            )
        )
    label = (env.label_colour or "none") if full else _label(env, front)  # full: the card's colour, faced or not
    return Scene(parts, env.flower_colour, label)


class PlantCrop:
    """The front camera's view narrowed onto the plant: what a crop of a higher-resolution camera image shows.

    The front camera is rendered at size * zoom pixels (by a renderer of its own, made once; plant models have an
    offscreen buffer of 960 x 540), and the size x size window centred on the main stem, CROP_HEIGHT above its base, is
    cut out, moved inward where it would leave the image. With zoom 2 the plant is about twice as large as in the front
    image. No camera is added to the model, so everything else renders as without it. The scenarios' image hooks are
    applied as to the front camera's own image (env.get_obs), before the crop."""

    def __init__(self, env, size: int = 224, zoom: int = 2):
        import mujoco

        assert env.plant_task, f"{env.task} has no plant"
        self.env, self.size, self.zoom = env, size, zoom
        self.renderer = mujoco.Renderer(env.model, size * zoom, size * zoom)

    def window(self) -> tuple[int, int]:
        """(top, left) of the crop in the full image, for the scene now."""
        full = self.size * self.zoom
        base, up = self.env.stem_axis()
        x, y, _ = _Camera(self.env, "front").project(base + CROP_HEIGHT * up)
        col, row = (x + 1) / 2 * full, (y + 1) / 2 * full
        left = int(np.clip(round(col - self.size / 2), 0, full - self.size))
        top = int(np.clip(round(row - self.size / 2), 0, full - self.size))
        return top, left

    def render(self) -> np.ndarray:
        """uint8 [size, size, 3]."""
        env = self.env
        self.renderer.update_scene(env.data, camera="front")
        img = self.renderer.render().copy()
        for s in env.scenarios:
            if s.applied:
                img = s.image(env, "front", img)
        top, left = self.window()
        return np.ascontiguousarray(img[top : top + self.size, left : left + self.size])

    def close(self):
        self.renderer.close()


# ------------------------------------------------------------------------------------------------------- memory
@dataclass
class Remembered:
    """A part of the inventory."""

    id: int  # stable for the episode: the order of discovery
    height: float  # the mean of its seen sightings (rule 2: of the unseen ones that founded it, while it has none)
    cut: bool = False
    bearing: float | None = None  # its last seen sighting's, or the last founding unseen one's (meaningless once cut)
    sightings: int = 0  # seen sightings only
    seen_step: int = -1
    attributes: dict[str, str] = field(default_factory=dict)
    # What the memory holds about the part rests on a look: its last seen sighting, and the sighting at which it was
    # marked cut (kept from then on, cleared if the cut is unmarked), each {"step", "p", "p_cut", "cut"} as that
    # percept said them
    evidence: dict | None = None
    cut_evidence: dict | None = None
    revocations: int = 0  # how often its cut was unmarked (rule 3)
    whole_step: int | None = None  # the first seen sighting that said it is not cut (rule 9, witness)


def _sighting(sighting: Part, step: int) -> dict:
    """A seen sighting as plain data: the look's control step, and what the percept said of the part."""
    p_cut = None if sighting.p_cut is None else _r(sighting.p_cut)
    return dict(step=int(step), p=_r(sighting.p), p_cut=p_cut, cut=bool(sighting.cut))


def _spread(bearings: list[float]) -> float:
    """The largest angle between two of the bearings."""
    wrapped = [(a - b + np.pi) % (2 * np.pi) - np.pi for a in bearings for b in bearings]
    return float(max((abs(x) for x in wrapped), default=0.0))


def _position_name(index: int, count: int) -> str:
    """reasoning.position_name for any count: an inventory still being made can hold one part (or, misled, four)."""
    if 2 <= count <= 3:
        return reasoning.position_name(index, count)
    return "lowest" if index == 0 else "highest" if index == count - 1 else "middle"


def _nearest_pairs(a: list[float], b: list[float], limit: float) -> list[tuple[int, int]]:
    """Pairs (i, j) with |a[i] - b[j]| <= limit, one to one, the nearest pairs first."""
    pairs = sorted((abs(x - y), i, j) for i, x in enumerate(a) for j, y in enumerate(b) if abs(x - y) <= limit)
    used_a, used_b, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            out.append((i, j))
    return out


class SceneMemory:
    """The scene over an episode, from scene records (percepts or the truth), by rules that need no training:

    1. Only confident evidence counts: a part when p >= p_min and it is visible (a seen sighting), the flower when
       p_flower >= p_min, the label when p_label >= p_min. With `unseen`, a part with p >= p_min that is not visible
       counts too (an unseen sighting), as evidence that it exists and of nothing else.
    2. A part is known by its height on the main stem. A counted sighting, seen or unseen, is matched to the nearest
       remembered part within `match` metres (one to one, the nearest pairs first): a seen one updates the part (its
       height is the mean of its seen sightings, its bearing, seen_step and evidence the last one's, `sightings` their
       number), an unseen one leaves it as it is. One that matches none is a candidate, and a candidate counted in
       `stable` updates in a row (by seen sightings, unseen ones or both) becomes a remembered part with the next id.
       A part founded on unseen sightings alone has their mean height and the last one's bearing, sightings 0,
       seen_step -1 and no evidence; its first seen sighting replaces that height, and until then a seen sighting
       that matched nothing is taken for it up to SEPARATE * match away. With `unseen`, a sighting that matches none
       starts or continues a candidate only at least SEPARATE * match from every remembered part (no two shoots are
       nearer than 2 * match, and the rest is left for the error of a height); one nearer is ambiguous and is dropped
       for the update. Parts are never removed.
    3. A part is marked cut after `stable` seen sightings in a row that say cut (a sighting says cut when its `cut` is
       set and its p_cut, if any, is at least p_min). A part that goes out of view is not cut, and an unseen sighting
       neither counts toward the row nor breaks it. With revoke 0 a cut is never unmarked. With `revoke` > 0 it is
       unmarked after `revoke` seen sightings in a row that say, with confidence, that it is not cut (its `cut` is not
       set and its p_cut, if any, is at most 1 - p_min); any other seen sighting breaks that row, and an update that
       does not see the part neither counts nor breaks it. Unmarking clears cut_evidence, starts the cut row again and
       is counted in revocations.
    4. The flower's colour is known after `stable` confident percepts of one colour in a row and changes only after
       `stable` of another. label_now is debounced the same way; label_read is its first colour and is kept when the
       card leaves view, replaced only by `stable` confident readings in a row of another colour. Percepts that are not
       confident neither count toward a row nor break it.
    5. `settled`: at least `stable` updates were made and the inventory did not grow in the last `stable` updates in
       which a part was counted, seen or unseen (updates that count no part, the arm hiding the plant, say nothing
       about it).
    6. Positions are named over the whole inventory, cut or not, as the expert's notes name them.
    7. With stable=1, p_min=0 and the simulator's truth as the percept, the memory is the truth after one update (of
       the parts a camera shows: rule 1 counts nothing else; with `unseen` the others are listed too, uncut).
    8. An attribute of a part (architecture v3): a reading counts when it comes with a seen sighting (rule 1) that does
       not say the part is cut, its class is not UNREADABLE and its probability (p_attributes, if any) is at least
       p_min. Under `attribute_rule` "row" (the default) the attribute is known after `stable` readings in a row of one
       value and changes only after `stable` of another; "last" takes the last reading; "angles" asks of the row that
       its readings were taken at bearings at least ANGLES_APART apart. A look that cannot read it neither counts nor
       breaks the row: what a tie showed is kept when it has turned away. Once a part is marked cut its attributes do
       not change (until a revoke unmarks it): a stub read as "no tie" would otherwise move the plan's selection. Every
       counted reading is kept with the part's bearing then (`readings`).
    9. With `witness`, a part first seen whole is marked cut only if the jaws have shut at its height (within `match`)
       since (`jaws_shut`, which the runner's cut gate calls); a part found cut at its first sighting needs no witness.
    """

    def __init__(
        self,
        stable: int = 3,
        p_min: float = 0.7,
        match: float = 0.0175,
        unseen: bool = False,
        revoke: int = 0,
        attribute_rule: str = "row",
        witness: bool = False,
    ):
        assert stable >= 1, stable
        assert revoke >= 0, revoke
        assert attribute_rule in ATTRIBUTE_RULES, f"attribute_rule {attribute_rule!r}: one of {ATTRIBUTE_RULES}"
        self.stable, self.p_min, self.match = stable, p_min, match
        self.unseen, self.revoke = unseen, revoke
        self.attribute_rule, self.witness = attribute_rule, bool(witness)
        self.reset()

    def reset(self):
        """Forget everything (a new episode)."""
        self.parts: list[Remembered] = []  # sorted by height, lowest first: the inventory
        self.flower: str | None = None
        self.label_read: str | None = None  # a colour, once read; kept when the card leaves view
        self.label_now: str = "none"  # the last debounced label class
        self.updates = 0
        self._next_id = 0
        self._candidates: list[list[tuple[Part, int]]] = []  # each: its sightings (part, step), one per update in a row
        self._cut_rows: dict[int, int] = {}  # part id -> seen sightings in a row that say cut
        self._uncut_rows: dict[int, int] = {}  # part id -> seen sightings in a row sure it is not cut (rule 3, revoke)
        # (part id, attribute) -> (value, its readings in a row, their bearings)
        self._attribute_rows: dict[tuple[int, str], tuple[str, int, list]] = {}
        self.readings: dict[tuple[int, str], list[tuple[int, str, float | None]]] = {}  # -> [(step, value, bearing)]
        self.shut: list[tuple[int, float]] = []  # (control step, height): the jaws shut among the shoots (rule 9)
        self._flower_row: tuple[str | None, int] = (None, 0)
        self._label_row: tuple[str | None, int] = (None, 0)
        self._grew: deque[bool] = deque(maxlen=self.stable)  # per update that counted a part: did the inventory grow

    # ----------------------------------------------------------------------------------------------- updating
    def update(self, scene: Scene, step: int) -> bool:
        """Take in what is seen now (`step`: the control step, kept as each part's seen_step). True when a fact a plan
        depends on changed: a part was added, marked cut or unmarked, an attribute became known, the flower, either
        label field or `settled` changed. A bearing or a height moving is not such a fact."""
        before = self._facts()
        self.updates += 1
        counted = [p for p in scene.parts if (p.visible or self.unseen) and p.p >= self.p_min]  # rule 1
        grew = self._update_parts(counted, step)
        if counted:
            self._grew.append(grew)
        if scene.flower is not None and scene.p_flower >= self.p_min:
            self._flower_row = self._row(self._flower_row, scene.flower)
            if self._flower_row[1] >= self.stable:
                self.flower = scene.flower
        if scene.p_label >= self.p_min:
            self._label_row = self._row(self._label_row, scene.label)
            if self._label_row[1] >= self.stable:
                self.label_now = scene.label
                if scene.label not in NO_COLOUR:
                    self.label_read = scene.label
        return self._facts() != before

    @staticmethod
    def _row(row: tuple, value) -> tuple:
        """One more in a row of `value`, or the first of a new row."""
        return (value, row[1] + 1) if row[0] == value else (value, 1)

    def _update_parts(self, counted: list[Part], step: int) -> bool:
        """Match the counted sightings to the inventory, then to the candidates; True if the inventory grew."""
        matched = _nearest_pairs([p.height for p in counted], [r.height for r in self.parts], self.match)
        for i, j in matched:
            if counted[i].visible:  # an unseen sighting only accounts for the part it matches
                self._observe(self.parts[j], counted[i], step)
        taken = {i for i, _ in matched}
        if self.unseen:
            far = SEPARATE * self.match  # the separation of rule 2
            # A part no camera has shown yet has only an estimate of its height: a seen sighting that matched nothing
            # is taken for it up to that separation, and from then on the part is where it was seen
            held = {j for _, j in matched}
            loose = [j for j, r in enumerate(self.parts) if not r.sightings and j not in held]
            seen = [i for i, p in enumerate(counted) if i not in taken and p.visible]
            pairs = _nearest_pairs([counted[i].height for i in seen], [self.parts[j].height for j in loose], far)
            for a, b in pairs:
                self._observe(self.parts[loose[b]], counted[seen[a]], step)
                taken.add(seen[a])
        rest = [p for i, p in enumerate(counted) if i not in taken]
        if self.unseen:
            rest = [p for p in rest if all(abs(p.height - r.height) >= far for r in self.parts)]
        heights = [sum(p.height for p, _ in c) / len(c) for c in self._candidates]
        pairs = _nearest_pairs([p.height for p in rest], heights, self.match)
        rows = [[*self._candidates[j], (rest[i], step)] for i, j in pairs]
        paired = {i for i, _ in pairs}
        rows += [[(p, step)] for i, p in enumerate(rest) if i not in paired]
        # a candidate not counted in this update has lost its row and is dropped; parts found in the same update get
        # their ids from the lowest up
        self._candidates, grew = [], False
        for sightings in sorted(rows, key=lambda row: sum(p.height for p, _ in row) / len(row)):
            if len(sightings) < self.stable:
                self._candidates.append(sightings)
                continue
            part = Remembered(id=self._next_id, height=sightings[0][0].height)
            self._next_id += 1
            for sighting, at in sightings:
                if sighting.visible:
                    self._observe(part, sighting, at)
            if not part.sightings:  # founded on unseen sightings alone: the best estimates there are
                part.height = sum(p.height for p, _ in sightings) / len(sightings)
                part.bearing = float(sightings[-1][0].bearing)
            self.parts.append(part)
            grew = True
        self.parts.sort(key=lambda r: r.height)
        return grew

    def _observe(self, part: Remembered, sighting: Part, step: int):
        """One seen sighting of a remembered part."""
        n = part.sightings
        part.height = (part.height * n + sighting.height) / (n + 1)  # the first replaces an unseen estimate
        part.sightings, part.bearing, part.seen_step = n + 1, float(sighting.bearing), step
        part.evidence = _sighting(sighting, step)
        if not sighting.cut and part.whole_step is None:
            part.whole_step = step
        says_cut = sighting.cut and (sighting.p_cut is None or sighting.p_cut >= self.p_min)
        self._cut_rows[part.id] = self._cut_rows.get(part.id, 0) + 1 if says_cut else 0
        if self._cut_rows[part.id] >= self.stable and (part.cut or self._witnessed(part)):
            if not part.cut:
                part.cut_evidence = dict(part.evidence)  # the sighting that made it count
            part.cut = True  # and never back, but by revoke
        if self.revoke:
            sure_uncut = not sighting.cut and (sighting.p_cut is None or sighting.p_cut <= 1 - self.p_min)
            self._uncut_rows[part.id] = self._uncut_rows.get(part.id, 0) + 1 if sure_uncut else 0
            if part.cut and self._uncut_rows[part.id] >= self.revoke:
                part.cut, part.cut_evidence, part.revocations = False, None, part.revocations + 1
                self._cut_rows[part.id] = self._uncut_rows[part.id] = 0
        if sighting.cut or part.cut:  # rule 8: nothing is read of a stub, and a cut part's attributes are frozen
            return
        for key, value in sighting.attributes.items():
            p = sighting.p_attributes.get(key)
            if value == UNREADABLE or (p is not None and p < self.p_min):
                continue
            self.readings.setdefault((part.id, key), []).append((int(step), value, part.bearing))
            then, n, bearings = self._attribute_rows.get((part.id, key), (None, 0, []))
            n, bearings = (n + 1, [*bearings, part.bearing]) if then == value else (1, [part.bearing])
            self._attribute_rows[(part.id, key)] = (value, n, bearings)
            known = n >= self.stable
            if self.attribute_rule == "last":
                known = True
            elif self.attribute_rule == "angles":
                known = known and _spread([b for b in bearings if b is not None]) >= ANGLES_APART
            if known:
                part.attributes[key] = value

    def _witnessed(self, part: Remembered) -> bool:
        """Rule 9: a part first seen whole is cut only if the jaws have shut at its height since."""
        if not self.witness or part.whole_step is None:
            return True
        return any(at >= part.whole_step and abs(h - part.height) <= self.match for at, h in self.shut)

    def jaws_shut(self, height: float, step: int):
        """The jaws shut among the shoots at `height` (metres above the stem's base) at control step `step` (rule 9)."""
        self.shut.append((int(step), float(height)))

    def _facts(self) -> tuple:
        parts = tuple((r.id, r.cut, tuple(sorted(r.attributes.items()))) for r in self.parts)
        return parts, self.flower, self.label_read, self.label_now, self.settled

    # ----------------------------------------------------------------------------------------------- reading
    @property
    def settled(self) -> bool:
        """The inventory is believed complete (rule 5)."""
        return self.updates >= self.stable and len(self._grew) == self.stable and not any(self._grew)

    def position(self, id: int) -> str:
        """Where part `id` is in the whole inventory, cut or not: "lowest", "middle" or "highest"."""
        for index, part in enumerate(self.parts):
            if part.id == id:
                return _position_name(index, len(self.parts))
        raise KeyError(f"no part {id} in the inventory")

    def by_position(self, name: str) -> Remembered | None:
        """The part `position` names `name`, or None (no middle part on a plant with two)."""
        return next((r for r in self.parts if self.position(r.id) == name), None)

    def see_text(self) -> str:
        """The notes' `see` field (reasoning.note_text) for what is remembered, e.g.
        "3 shoots, purple flower, label hidden, read cyan". The flower not known yet: "flower not seen"."""
        text = f"{len(self.parts)} shoots, " + (f"{self.flower} flower" if self.flower else "flower not seen")
        if self.label_read is not None:
            shown = self.label_now == self.label_read
            text += f", label {self.label_read}" if shown else f", label hidden, read {self.label_read}"
        elif self.label_now != "none":  # a card, not read yet
            text += ", label not seen"
        return text

    def text(self) -> str:
        """see_text, with the parts cut: "...; cut: the lowest" (or "the lowest and the middle")."""
        cut = [self.position(r.id) for r in self.parts if r.cut]
        return self.see_text() + (f"; cut: {reasoning._join(cut)}" if cut else "")

    def snapshot(self) -> dict:
        """Plain types, for logs. The evidence of each part (Remembered.evidence, cut_evidence) is listed apart, by id,
        in the inventory's order, with its revocations when the memory can unmark a cut (revoke)."""
        return dict(
            parts=[
                dict(
                    id=int(r.id),
                    position=self.position(r.id),
                    height=_r(r.height),
                    cut=bool(r.cut),
                    bearing=None if r.bearing is None else _r(r.bearing),
                    sightings=int(r.sightings),
                    seen_step=int(r.seen_step),
                    attributes=dict(r.attributes),
                )
                for r in self.parts
            ],
            flower=self.flower,
            label_read=self.label_read,
            label_now=self.label_now,
            settled=self.settled,
            updates=self.updates,
            text=self.text(),
            evidence=[
                dict(
                    id=int(r.id),
                    evidence=None if r.evidence is None else dict(r.evidence),
                    cut_evidence=None if r.cut_evidence is None else dict(r.cut_evidence),
                    **(dict(revocations=int(r.revocations)) if self.revoke else {}),  # without revoke, as it always was
                )
                for r in self.parts
            ],
            # rule 8, where there are any: every reading counted, by part and attribute, with the bearing then
            **(
                dict(
                    readings=[
                        dict(
                            id=int(i), attribute=a, readings=[[s, v, None if b is None else _r(b)] for s, v, b in rows]
                        )
                        for (i, a), rows in sorted(self.readings.items())
                    ]
                )
                if self.readings
                else {}
            ),
        )
