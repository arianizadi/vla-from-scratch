"""Avoidance drills: the aim range's drills with things in the way.

Reach into awkward places, come in from the right direction, turn the wrist on the way, and grab or touch without
hitting anything. The pruning policy's other failures besides running out of time are touching the flower, and the
test situations add a thorny rose, a big leaf hiding a shoot and cuttings left on the plate: all are about moving the
hand near things it must not touch.

  avoid_box    grab the cube in a box open on one side only (a cubby on a plinth): come in level through the opening
  avoid_shelf  grab the cube under a low shelf (on a short post): come in level and low, under the board
  avoid_gap    touch the dot through a gap between two posts (the level hand and the wrist behind it pass between
               them), then turn the wrist to arrive along the dot's pin
  avoid_wall   touch the dot behind a short wall: the straight way runs into the wall, so go over it
  avoid_hot    touch the dot next to a red keep-out zone (a bubble, as around a flower): no part of the arm may enter it
  avoid_turn   touch the dot just inside a pocket, arriving along its pin (the hand turns inside the pocket), then back
               out the way it came
  avoid_mix    one of the six per episode (training data); avoid_each takes them in turn by seed (evaluation)

Rules for every drill: touching an obstacle (any part of the arm against any of its slabs), or entering the hot zone,
ends the episode as a failure; the episode records how many physics steps had such a contact, and the closest the arm
came to the obstacles (clearance) and to the zone. Layouts are drawn until the scripted expert's approach for that
drill (aim_expert_avoid.plan) clears every obstacle by AVOID_MARGIN all the way: layouts with no safe path are left out.
The drills are built from primitives (boxes, a translucent sphere); none of it is in the model of any other task.
"""

import mujoco
import numpy as np

from . import aim as A
from . import env as E

DRILLS = ("box", "shelf", "gap", "wall", "hot", "turn")
BUDGET = dict(box=300, shelf=300, gap=300, wall=240, hot=240, turn=360)
INSTRUCTIONS = dict(
    box="grab the cube in the box",
    shelf="grab the cube under the shelf",
    gap="touch the {c} dot through the gap",
    wall="touch the {c} dot behind the wall",
    hot="touch the {c} dot, keep out of the red zone",
    turn="touch the {c} dot in the pocket, then back out",
)
DOT_COLOURS = ("green", "blue", "yellow")  # red is the hot zone's colour
N_SLABS = 6
SLAB_RGBA = (0.50, 0.54, 0.60, 1)
HOT_RGBA = (0.95, 0.12, 0.08, 0.35)
CUBE_HALF = 0.015  # the cube to grab (30 mm: a level grasp needs the fingertips clear of the floor under it)
CUBE_RGBA = (0.95, 0.75, 0.05, 1)
WALL_T = 0.01  # thickness of every wall, board and post
AVOID_MARGIN = 0.006  # metres: least clearance of the expert's planned approach (these are tight places)
TRIES = 60  # layouts drawn before giving up (the last one is kept)
GRAB_LIFT, GRAB_STEPS = 0.004, 5
TURN_OUT = 0.06  # avoid_turn: how far out of the pocket the gripper's centre must come back
PARK = np.array([2.5, 3.0, -0.70])


def add_to_spec(spec):
    """N_SLABS static boxes (each moved and resized per episode), the hot zone (a sphere, seen but not solid) and the
    cube to grab (a free body)."""
    wb = spec.worldbody
    for i in range(N_SLABS):
        slab = wb.add_body(name=f"avoid_slab{i}", pos=list(PARK + [0.1 * i, 0, 0]))
        slab.add_geom(
            name=f"avoid_slab{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.01, 0.01, 0.01],
            rgba=list(SLAB_RGBA),
            friction=[1.0, 0.005, 0.0001],
            contype=1,
            conaffinity=1,
            group=2,
        )
    zone = wb.add_body(name="avoid_zone", pos=list(PARK + [0, 0.3, 0]))
    zone.add_geom(
        name="avoid_zone",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.05, 0, 0],
        rgba=list(HOT_RGBA),
        contype=0,
        conaffinity=0,
        group=2,
        density=0,
    )
    cube = wb.add_body(name="avoid_cube", pos=list(PARK + [0, 0.5, -0.05]))
    cube.add_freejoint()
    E._solid(cube, "avoid_cube_col", "box", (CUBE_HALF,) * 3, conaffinity=1)
    E._look(cube, "avoid_cube_vis", CUBE_RGBA, gtype="box", size=(CUBE_HALF,) * 3)


def _horizontal(azimuth):
    return np.array([np.cos(azimuth), np.sin(azimuth), 0.0])


class AvoidEnv(A.AimEnv):
    """AimEnv on an avoidance task (see the module's docstring)."""

    def __init__(self, task, *args, **kwargs):
        super().__init__(task, *args, **kwargs)
        m = self.model
        self.avoid = E.TASKS[task]["avoid"]
        if self._max_steps_given is None:
            self.max_steps = BUDGET.get(self.avoid, max(BUDGET.values()))
        self.slab_body = [m.body(f"avoid_slab{i}").id for i in range(N_SLABS)]
        self.slab_geom = [m.geom(f"avoid_slab{i}").id for i in range(N_SLABS)]
        self.zone_body, self.zone_geom = m.body("avoid_zone").id, m.geom("avoid_zone").id
        self.acube_qadr = m.joint(m.body("avoid_cube").jntadr[0]).qposadr[0]
        self.acube_vadr = m.joint(m.body("avoid_cube").jntadr[0]).dofadr[0]
        root = m.body("link0").id
        self.robot_geoms = [
            g
            for g in range(m.ngeom)
            if m.body_rootid[m.geom_bodyid[g]] == root and (m.geom_contype[g] or m.geom_conaffinity[g])
        ]
        self.robot_body_set = {int(m.geom_bodyid[g]) for g in self.robot_geoms}
        self.finger_bodies = (m.body("left_finger").id, m.body("right_finger").id)
        self.table_geom = m.geom("table").id
        base = {m.body(n).id for n in ("link0", "link1", "link2")}  # stand on the table, or always near it
        self.table_check_geoms = [g for g in self.robot_geoms if m.geom_bodyid[g] not in base]
        self._planner = None
        self._fromto = np.zeros(6)

    # ------------------------------------------------------------------ reset
    # AimEnv.reset draws the drill (_choose_drill), then its layout (_reset_drill), until the expert's approach is safe
    def _choose_drill(self, seed):
        if self.avoid == "mix":  # a stream of its own: the layout is that of the drill's own task
            return DRILLS[int(np.random.default_rng([seed, 43]).integers(len(DRILLS)))]
        if self.avoid == "each":
            return DRILLS[seed % len(DRILLS)]
        return self.avoid

    def _budget(self, drill):
        return BUDGET[drill]

    def _instructions(self):
        return INSTRUCTIONS

    def _after_settle(self):
        self.acube_rest = float(self.acube_pos()[2])

    def _reset_drill(self, rng):
        drill = self.drill
        self.slabs_used = 0
        self.zone_r = 0.0
        self.touched = False
        self.touch_count = 0
        self.contact_steps = 0
        self.hit = False
        self.min_clearance = np.inf
        self.min_zone = np.inf
        self.in_zone = False
        self.layout_tries = 0
        if self._planner is None:
            from .aim_expert_avoid import AvoidExpert

            self._planner = AvoidExpert(self, seed=0)
        for tries in range(1, TRIES + 1):
            self._clear_layout()
            getattr(self, f"_layout_{drill}")(rng)
            mujoco.mj_forward(self.model, self.data)
            self.layout_tries = tries
            plan = self._planner.plan(check=True)
            if plan is not None:
                break
        self.plan_clearance = getattr(self._planner, "last_clearance", None)

    def _clear_layout(self):
        m, d = self.model, self.data
        for i, body in enumerate(self.slab_body):
            m.body_pos[body], m.body_quat[body] = PARK + [0.1 * i, 0, 0], [1, 0, 0, 0]
            self._size_slab(i, (0.01, 0.01, 0.01))
        m.body_pos[self.zone_body] = PARK + [0, 0.3, 0]
        m.geom_group[self.zone_geom] = 3
        self._put_acube(PARK[:2] + [0, 0.5], -0.75 + CUBE_HALF)
        for c, mid in self.dot_mocap.items():
            d.mocap_pos[mid], d.mocap_quat[mid] = PARK + [0.1 * list(self.dot_mocap).index(c), 0.8, 0], [1, 0, 0, 0]
            m.geom_group[self.pin_geom[c]] = 3
        self.dots, self.pin_dir, self.layout, self.slabs_used = {}, None, {}, 0

    def _size_slab(self, i, half):
        m, g = self.model, self.slab_geom[i]
        m.geom_size[g] = half
        m.geom_rbound[g] = float(np.linalg.norm(half))
        m.geom_aabb[g] = [0, 0, 0, *half]
        # the body's bounding volume, built at compile time and used by collision detection: resize it as well
        adr = m.body_bvhadr[self.slab_body[i]]
        if adr >= 0:
            m.bvh_aabb[adr] = [0, 0, 0, *half]

    def _slab(self, centre, half, yaw=0.0):
        """Place the next slab: a box of half sizes `half` (x, y, z in its own frame) at `centre`, turned by `yaw`."""
        i = self.slabs_used
        self.slabs_used += 1
        self.model.body_pos[self.slab_body[i]] = centre
        self.model.body_quat[self.slab_body[i]] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self._size_slab(i, half)

    def _local(self, origin, a, b, yaw):
        """Point `a` along yaw's direction and `b` across it (to its left), from `origin` (world)."""
        return origin + a * _horizontal(yaw) + b * _horizontal(yaw + np.pi / 2)

    def _put_acube(self, xy, z, yaw=0.0):
        adr, vadr = self.acube_qadr, self.acube_vadr
        self.data.qpos[adr : adr + 3] = [xy[0], xy[1], z]
        self.data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[vadr : vadr + 6] = 0

    def _show_dot(self, rng, pos, pin=None):
        self.colour = DOT_COLOURS[int(rng.integers(len(DOT_COLOURS)))]
        self.dots = {self.colour: np.asarray(pos, dtype=float)}
        mid = self.dot_mocap[self.colour]
        self.data.mocap_pos[mid] = pos
        self.pin_dir = pin
        if pin is not None:
            self.data.mocap_quat[mid] = A._quat_z_to(-pin)
            self.model.geom_group[self.pin_geom[self.colour]] = 2

    def _approach_yaw(self, rng, pos):
        """A level approach direction the arm can reach at `pos`: across the line from the robot's base, or partly
        out along it (yaw of the direction the gripper points)."""
        radial = np.arctan2(pos[1], pos[0])
        return radial + rng.choice([-1, 1]) * rng.uniform(np.radians(20), np.radians(115))

    # --- the six layouts. Each sets self.layout with what the expert's plan needs.
    def _layout_box(self, rng):
        self.colour = "cube"
        floor = rng.uniform(0.05, 0.10)
        W, H, D, t = 0.17, 0.09, 0.11, WALL_T  # the open fingers span about 0.11 m
        o = np.array([rng.uniform(0.40, 0.58), rng.uniform(-0.18, 0.18), 0.0])  # middle of the opening, at the table
        yaw = self._approach_yaw(rng, o)  # the gripper points into the box along yaw
        L = lambda a, b, z: self._local(o, a, b, yaw) + [0, 0, z]  # noqa: E731
        self._slab(L(D / 2, 0, floor / 2), (D / 2 + t, W / 2 + t, floor / 2), yaw)  # plinth, flush with the opening
        self._slab(L(D + t / 2, 0, floor + H / 2), (t / 2, W / 2 + t, H / 2), yaw)  # back
        for side in (-1, 1):
            self._slab(L(D / 2, side * (W / 2 + t / 2), floor + H / 2), (D / 2, t / 2, H / 2), yaw)
        self._slab(L(D / 2, 0, floor + H + t / 2), (D / 2 + t, W / 2 + t, t / 2), yaw)  # roof
        depth = rng.uniform(0.018, 0.026)  # the palm stays outside: the box holds the fingers only
        cube = L(depth, rng.uniform(-0.01, 0.01), floor + CUBE_HALF + 0.001)
        self._put_acube(cube[:2], cube[2], yaw + rng.uniform(-0.15, 0.15))
        self.layout = dict(point=_horizontal(yaw), opening=o + [0, 0, floor], grab=True)

    def _layout_shelf(self, rng):
        self.colour = "cube"
        post = rng.uniform(0.035, 0.06)  # the cube stands on a short post, so a level hand clears the table
        gap = rng.uniform(0.035, 0.05)  # between the cube's top and the board
        board = post + 2 * CUBE_HALF + gap
        W, D, t = 0.26, 0.13, WALL_T
        o = np.array([rng.uniform(0.40, 0.58), rng.uniform(-0.18, 0.18), 0.0])  # middle of the board's front edge
        yaw = self._approach_yaw(rng, o)
        L = lambda a, b, z: self._local(o, a, b, yaw) + [0, 0, z]  # noqa: E731
        self._slab(L(D / 2, 0, board + t / 2), (D / 2, W / 2, t / 2), yaw)  # the board
        for side in (-1, 1):  # its legs, at the back corners (the level hand is 0.21 m long across)
            self._slab(L(D - 0.01, side * (W / 2 - 0.01), board / 2), (0.01, 0.01, board / 2), yaw)
        depth = rng.uniform(0.035, 0.045)
        cube = L(depth, rng.uniform(-0.02, 0.02), 0.0)
        self._slab(cube + [0, 0, post / 2], (0.011, 0.011, post / 2), yaw)  # the post
        self._put_acube(cube[:2], post + CUBE_HALF + 0.001, yaw + rng.uniform(-0.15, 0.15))
        self.layout = dict(point=_horizontal(yaw), opening=o + [0, 0, post], grab=True)

    def _layout_gap(self, rng):
        gap = rng.uniform(
            0.24, 0.27
        )  # between the posts' inner faces: the level hand is 0.21 m long, the wrist follows
        height = 0.20  # below the lowest random start of the gripper (0.22 m), so no start pose meets a post
        centre = np.array([rng.uniform(0.40, 0.55), rng.uniform(-0.15, 0.15), 0.0])
        radial = np.arctan2(centre[1], centre[0])
        yaw = radial + rng.choice([-1, 1]) * rng.uniform(np.radians(60), np.radians(110))  # across: in reach
        for side in (-1, 1):
            self._slab(
                self._local(centre, 0, side * (gap / 2 + 0.01), yaw) + [0, 0, height / 2], (0.01, 0.01, height / 2)
            )
        beyond = rng.uniform(0.08, 0.11)
        dot = self._local(centre, beyond, rng.uniform(-0.02, 0.02), yaw) + [0, 0, rng.uniform(0.10, 0.16)]
        turn = rng.choice([-1, 1]) * rng.uniform(np.radians(20), np.radians(40))
        if rng.random() < 0.5:  # turn the wrist sideways
            pin = _horizontal(yaw + turn)
        else:  # or down
            a = abs(turn)
            pin = np.cos(a) * _horizontal(yaw) + [0, 0, -np.sin(a)]
        self._show_dot(rng, dot, pin)
        self.layout = dict(point=_horizontal(yaw), gap_centre=centre + [0, 0, dot[2]], dot=dot, final=pin)

    def _layout_wall(self, rng):
        dot = np.array([rng.uniform(0.42, 0.62), rng.uniform(-0.20, 0.20), rng.uniform(0.03, 0.07)])
        radial = dot[:2] / np.linalg.norm(dot[:2])
        height, width = rng.uniform(0.10, 0.16), rng.uniform(0.24, 0.32)
        back = rng.uniform(0.05, 0.08)  # the wall stands this far toward the robot from the dot
        yaw = float(np.arctan2(radial[1], radial[0]))
        centre = dot - np.array([radial[0], radial[1], 0.0]) * back
        self._slab([centre[0], centre[1], height / 2], (WALL_T / 2, width / 2, height / 2), yaw)
        self._show_dot(rng, dot)
        self.layout = dict(dot=dot, along=_horizontal(yaw + np.pi / 2), top=height)

    def _layout_hot(self, rng):
        dot = np.array([rng.uniform(0.38, 0.60), rng.uniform(-0.20, 0.20), rng.uniform(0.04, 0.12)])
        r = rng.uniform(0.04, 0.06)
        u = rng.normal(size=3)
        u[2] = abs(u[2]) + 0.3  # the zone sits beside or above the dot, as a flower above a shoot
        u /= np.linalg.norm(u)
        zone = dot + u * (r + rng.uniform(0.04, 0.06))
        zone[2] = np.clip(zone[2], r * 0.5, 0.20 - r)  # below the lowest start pose of the gripper (0.22 m)
        self.model.body_pos[self.zone_body] = zone
        self.model.geom_size[self.zone_geom][0] = r
        self.model.geom_rbound[self.zone_geom] = r
        self.model.geom_group[self.zone_geom] = 2
        self.zone_r = r
        self._show_dot(rng, dot)
        self.layout = dict(dot=dot, zone=zone, r=r)

    def _layout_turn(self, rng):
        floor = rng.uniform(0.06, 0.10)
        W, H, D, t = 0.09, 0.09, 0.12, WALL_T
        o = np.array([rng.uniform(0.40, 0.56), rng.uniform(-0.18, 0.18), 0.0])
        yaw = self._approach_yaw(rng, o)
        L = lambda a, b, z: self._local(o, a, b, yaw) + [0, 0, z]  # noqa: E731
        self._slab(L(D / 2, 0, floor / 2), (D / 2 + t, W / 2 + t, floor / 2), yaw)  # plinth
        self._slab(L(D + t / 2, 0, floor + H / 2), (t / 2, W / 2 + t, H / 2), yaw)  # back
        for side in (-1, 1):
            self._slab(L(D / 2, side * (W / 2 + t / 2), floor + H / 2), (D / 2, t / 2, H / 2), yaw)
        self._slab(L(D / 2, 0, floor + H + t / 2), (D / 2 + t, W / 2 + t, t / 2), yaw)  # roof
        dot = L(rng.uniform(0.02, 0.028), rng.uniform(-0.01, 0.01), floor + H / 2 + rng.uniform(-0.01, 0.01))
        a = rng.choice([-1, 1]) * rng.uniform(np.radians(15), np.radians(30))  # the hand pitches up or down inside
        pin = np.cos(a) * _horizontal(yaw) + [0, 0, -np.sin(a)]
        self._show_dot(rng, dot, pin)
        self.layout = dict(point=_horizontal(yaw), opening=o + [0, 0, floor + H / 2], dot=dot, final=pin)

    # ---------------------------------------------------------------- queries
    def acube_pos(self):
        return self.data.qpos[self.acube_qadr : self.acube_qadr + 3].copy()

    def obstacle_geoms(self):
        return self.slab_geom[: self.slabs_used]

    def clearance(self, data=None, table=False):
        """(least distance from the arm to the obstacles, least distance to the hot zone's surface), up to 5 cm each;
        negative inside. `data`: another MjData of the same model (the expert's planning copy). `table`: count the
        table as an obstacle too (for planning: touching it fails no drill, but a hand pressed on it is stuck)."""
        m, d = self.model, data if data is not None else self.data
        best = 0.05
        for o in self.obstacle_geoms():
            for g in self.robot_geoms:
                best = min(best, mujoco.mj_geomDistance(m, d, g, o, 0.05, self._fromto))
        if table:
            for g in self.table_check_geoms:
                best = min(best, mujoco.mj_geomDistance(m, d, g, self.table_geom, 0.05, self._fromto))
        zone = 0.05
        if self.zone_r > 0:
            for g in self.robot_geoms:
                zone = min(zone, mujoco.mj_geomDistance(m, d, g, self.zone_geom, 0.05, self._fromto))
        return best, zone

    def object_pos(self, name):
        if self.drill in ("box", "shelf"):
            return self.acube_pos()
        return self.dot_pos()

    def object_yaw(self, name):
        return 0.0

    # ------------------------------------------------------------------- step
    def _substep_hook(self):
        """After every physics step: count steps with the arm against an obstacle."""
        d, m = self.data, self.model
        n = d.ncon
        if n == 0 or not self.slabs_used:
            return
        slabs = set(self.obstacle_geoms())
        for i in range(n):
            g1, g2 = d.contact.geom1[i], d.contact.geom2[i]
            if (g1 in slabs and m.geom_bodyid[g2] in self.robot_body_set) or (
                g2 in slabs and m.geom_bodyid[g1] in self.robot_body_set
            ):
                self.contact_steps += 1
                return

    def _score_common(self):
        before = self.contact_steps
        if before and not self.hit:
            self.hit = True
        clearance, zone = self.clearance()
        self.min_clearance, self.min_zone = min(self.min_clearance, clearance), min(self.min_zone, zone)
        if zone <= 0:
            self.in_zone = True
        self.wrong = self.hit or self.in_zone
        self.result_info = dict(
            contacts=int(self.contact_steps),
            clearance_mm=round(1000 * float(self.min_clearance), 1) if self.slabs_used else None,
            zone_mm=round(1000 * float(self.min_zone), 1) if self.zone_r > 0 else None,
            hit=bool(self.hit),
            in_zone=bool(self.in_zone),
        )

    def _score(self):
        self._score_common()
        if self.wrong:
            return
        drill = self.drill
        if drill in ("box", "shelf"):
            self.result_info["error_mm"] = round(1000 * float(np.linalg.norm(self.tcp_pos() - self.acube_pos())), 1)
            self.lift_count = self.lift_count + 1 if self._held() else 0
            self.aim_success = self.lift_count >= GRAB_STEPS
            return
        error = self._error()
        self.result_info["error_mm"] = round(1000 * error, 1)
        ok = error < A.TOUCH_TOL
        if self.pin_dir is not None:
            angle = float(np.arccos(np.clip(self.pointing() @ self.pin_dir, -1, 1)))
            self.result_info["angle_deg"] = round(float(np.degrees(angle)), 1)
            ok = ok and angle < A.POSE_ANGLE
        if not self.touched:
            self.touch_count = self.touch_count + 1 if ok else 0
            self.touched = self.touch_count >= A.TOUCH_HOLD
        if drill != "turn":
            self.aim_success = self.touched
            return
        out = float((self.layout["opening"] - self.tcp_pos()) @ self.layout["point"])  # metres out of the pocket
        self.result_info["touched"] = bool(self.touched)
        self.aim_success = self.touched and out >= TURN_OUT

    def _held(self):
        """The cube is held: the gripper is commanded shut, both fingers touch it, and it is off its support."""
        d, m = self.data, self.model
        if d.ctrl[7] >= 0.5 * 255:
            return False
        body = m.body("avoid_cube").id
        touching = set()
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact.geom1[i]], m.geom_bodyid[d.contact.geom2[i]]
            for f in self.finger_bodies:
                if {b1, b2} == {f, body}:
                    touching.add(f)
        return len(touching) == 2 and self.acube_pos()[2] - self.acube_rest > GRAB_LIFT

    def aim_record(self, info):
        record = dict(drill=self.drill, colour=self.colour, **{k: info[k] for k in self.result_info if k in info})
        record["layout_tries"] = self.layout_tries
        return record
