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
intended.

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
    attributes: dict[str, str] = field(default_factory=dict)  # e.g. {"tie": "red"}; no task here sets any yet

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


def scene_truth(env) -> Scene:
    """The scene as the simulator knows it (env: a PickupEnv of a plant task, in the state the cameras show now): one
    Part per shoot of the plant, lowest first, cut or not, with which cameras show it (views: front, wrist), the
    flower's colour and the card."""
    from .env import REACH_AZIMUTH, STEM_BASE, _wrap_angle

    assert env.plant_task, f"{env.task} has no plant"
    front, wrist = _Camera(env, "front"), _Camera(env, "wrist")
    parts = []
    for name in env.active:  # from the lowest shoot up
        points, own = _part_points(env, name), _own_geoms(env, name)
        views = tuple(sum(camera.sees(p, own) for p in points) >= SEEN_POINTS for camera in (front, wrist))
        parts.append(
            Part(
                height=float(env.shoots[name]["origin"][2] - STEM_BASE),
                bearing=float(_wrap_angle(REACH_AZIMUTH - env.shoot_azimuth(name))),
                cut=name in env.cut,
                visible=any(views),
                views=views,
            )
        )
    return Scene(parts, env.flower_colour, _label(env, front))


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
    height: float  # the mean of its counted sightings
    cut: bool = False
    bearing: float | None = None  # its last counted sighting (meaningless once cut)
    sightings: int = 0
    seen_step: int = -1
    attributes: dict[str, str] = field(default_factory=dict)


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

    1. Only confident evidence counts: a part when p >= p_min and it is visible, the flower when p_flower >= p_min,
       the label when p_label >= p_min.
    2. A part is known by its height on the main stem. A counted part is matched to the nearest remembered part within
       `match` metres (one to one, the nearest pairs first); one that matches none is a candidate, and a candidate
       counted in `stable` updates in a row becomes a remembered part with the next id. Parts are never removed.
    3. A part is marked cut after `stable` counted sightings in a row that say cut (a sighting says cut when its `cut`
       is set and its p_cut, if any, is at least p_min) and is never unmarked. A part that goes out of view is not cut.
    4. The flower's colour is known after `stable` confident percepts of one colour in a row and changes only after
       `stable` of another. label_now is debounced the same way; label_read is its first colour and is kept when the
       card leaves view, replaced only by `stable` confident readings in a row of another colour. Percepts that are not
       confident neither count toward a row nor break it.
    5. `settled`: at least `stable` updates were made and the inventory did not grow in the last `stable` updates in
       which a part was counted (updates that count no part, the arm hiding the plant, say nothing about it).
    6. Positions are named over the whole inventory, cut or not, as the expert's notes name them.
    7. With stable=1, p_min=0 and the simulator's truth as the percept, the memory is the truth after one update (of
       the parts a camera shows: rule 1 counts nothing else).
    """

    def __init__(self, stable: int = 3, p_min: float = 0.7, match: float = 0.0175):
        assert stable >= 1, stable
        self.stable, self.p_min, self.match = stable, p_min, match
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
        self._cut_rows: dict[int, int] = {}  # part id -> counted sightings in a row that say cut
        self._attribute_rows: dict[tuple[int, str], tuple[str, int]] = {}  # (part id, attribute) -> (value, row)
        self._flower_row: tuple[str | None, int] = (None, 0)
        self._label_row: tuple[str | None, int] = (None, 0)
        self._grew: deque[bool] = deque(maxlen=self.stable)  # per update that counted a part: did the inventory grow

    # ----------------------------------------------------------------------------------------------- updating
    def update(self, scene: Scene, step: int) -> bool:
        """Take in what is seen now (`step`: the control step, kept as each part's seen_step). True when a fact a plan
        depends on changed: a part was added or marked cut, an attribute became known, the flower, either label field
        or `settled` changed. A bearing or a height moving is not such a fact."""
        before = self._facts()
        self.updates += 1
        counted = [p for p in scene.parts if p.visible and p.p >= self.p_min]
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
        """Match the counted parts to the inventory, then to the candidates; True if the inventory grew."""
        matched = _nearest_pairs([p.height for p in counted], [r.height for r in self.parts], self.match)
        for i, j in matched:
            self._observe(self.parts[j], counted[i], step)
        taken = {i for i, _ in matched}
        rest = [p for i, p in enumerate(counted) if i not in taken]
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
                self._observe(part, sighting, at)
            self.parts.append(part)
            grew = True
        self.parts.sort(key=lambda r: r.height)
        return grew

    def _observe(self, part: Remembered, sighting: Part, step: int):
        """One counted sighting of a remembered part."""
        n = part.sightings
        part.height = (part.height * n + sighting.height) / (n + 1)
        part.sightings, part.bearing, part.seen_step = n + 1, float(sighting.bearing), step
        says_cut = sighting.cut and (sighting.p_cut is None or sighting.p_cut >= self.p_min)
        self._cut_rows[part.id] = self._cut_rows.get(part.id, 0) + 1 if says_cut else 0
        if self._cut_rows[part.id] >= self.stable:
            part.cut = True  # and never back
        for key, value in sighting.attributes.items():
            row = self._row(self._attribute_rows.get((part.id, key), (None, 0)), value)
            self._attribute_rows[(part.id, key)] = row
            if row[1] >= self.stable:
                part.attributes[key] = value

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
        """Plain types, for logs."""
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
        )
