"""MuJoCo tabletop pick-up environment: Franka Panda + Blender-authored objects.

Observation: two RGB cameras (front, wrist) + 8-D state (7 joint angles, gripper opening in [0, 1]).
Action:      8-D absolute target (7 joint position targets, gripper command 1=open / 0=closed).
Control runs at 20 Hz (25 physics substeps of 2 ms).

Tasks
  pick_cube    one red cube; "pick up the red cube"
  pick_target  three of four objects on the table; "pick up the <object>" names the target
"""

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
PANDA_XML = ASSETS / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"

STEEL = (0.75, 0.76, 0.78, 1)
# Body frame conventions match the Blender export origins (see assets/build_assets_blender.py).
# visual: (mesh, rgba, z offset). collision: (type, size, z offset). rest_z: body-origin height when resting.
# grasp_sym: yaw symmetry of a valid top-down grasp (None = any yaw); grasp_axis: finger closing axis offset.
OBJECTS = {
    "cube": dict(
        label="red cube",
        visual=[("cube", (0.80, 0.08, 0.06, 1), 0.0)],
        collision=[("box", (0.0225, 0.0225, 0.0225), 0.0)],
        rest_z=0.0225,
        grasp_sym=np.pi / 2,
        grasp_axis=0.0,
    ),
    "can": dict(
        label="green can",
        visual=[("can_body", (0.10, 0.55, 0.18, 1), 0.0), ("can_lid", STEEL, 0.040), ("can_base", STEEL, -0.040)],
        collision=[("cylinder", (0.023, 0.042), 0.0)],
        rest_z=0.042,
        grasp_sym=None,
        grasp_axis=0.0,
    ),
    "hex": dict(
        label="blue hex nut",
        visual=[("hex", (0.08, 0.22, 0.80, 1), 0.0)],
        collision=[("cylinder", (0.0235, 0.020), 0.0)],
        rest_z=0.020,
        grasp_sym=None,
        grasp_axis=0.0,
    ),
    "tblock": dict(
        label="yellow T block",
        visual=[("tblock", (0.95, 0.72, 0.05, 1), 0.0)],
        collision=[("box", (0.015, 0.015, 0.0225), 0.0), ("box", (0.035, 0.015, 0.0075), 0.030)],
        rest_z=0.0225,
        grasp_sym=np.pi,
        grasp_axis=np.pi / 2,  # fingers close along the object's local y (the 30 mm stem width)
    ),
}
# Look-alikes: the red cube's shape in colours a small step apart, as flowers of one plant are. Only the
# instruction's colour word tells them apart.
for _name, _label, _rgba in (
    ("cube_orange", "orange cube", (0.88, 0.32, 0.04, 1)),
    ("cube_pink", "pink cube", (0.85, 0.20, 0.42, 1)),
    ("cube_dark", "dark red cube", (0.42, 0.03, 0.03, 1)),
):
    OBJECTS[_name] = dict(OBJECTS["cube"], label=_label, visual=[("cube", _rgba, 0.0)])
# Flowers on stems (flower-trimming step 4): a heavy base, a thin stem and a flower head, built from primitives.
# The flower on top blocks a grasp from above, so the stem is taken from the side ("side"), inside a band of
# heights on the stem ("band": the gripper's centre point, metres above the table).
STEM_GREEN = (0.20, 0.50, 0.15, 1)
POT = (0.35, 0.22, 0.14, 1)
for _colour, _rgba in (
    ("red", (0.86, 0.10, 0.12, 1)),
    ("orange", (0.93, 0.47, 0.08, 1)),
    ("pink", (0.93, 0.45, 0.70, 1)),
    ("purple", (0.46, 0.22, 0.72, 1)),
):
    OBJECTS[f"flower_{_colour}"] = dict(
        label=f"{_colour} flower",
        visual=[],
        prim_visual=[
            ("cylinder", (0.035, 0.008), POT, 0.0),
            ("capsule", (0.005, 0.07), STEM_GREEN, 0.083),
            ("sphere", (0.021,), _rgba, 0.172),
        ],
        collision=[
            ("cylinder", (0.035, 0.008), 0.0, 8000),
            ("capsule", (0.005, 0.07), 0.083, 600),
            ("sphere", (0.021,), 0.172, 150),
        ],
        rest_z=0.008,
        grasp="side",
        band=(0.07, 0.11),
        grasp_sym=None,
        grasp_axis=0.0,
    )
TASKS = {
    "pick_cube": dict(pool=["cube"], num_objects=1),
    "pick_target": dict(pool=["cube", "can", "hex", "tblock"], num_objects=3),
    "pick_lookalike": dict(pool=["cube", "cube_orange", "cube_pink", "cube_dark"], num_objects=3),
    # Clutter and occlusion: four objects, look-alikes among them, and upright "leaves" that hide them from
    # the front camera. The leaves are seen but not touched (no collision): this task is about seeing.
    "pick_clutter": dict(
        pool=["cube", "can", "hex", "tblock", "cube_orange", "cube_pink", "cube_dark"], num_objects=4, leaves=3
    ),
    # Step 5: the same scene; closing the jaws on a stem inside its band cuts it, and the part above falls.
    # Cutting any other stem ends the episode as a failure.
    "cut_stem": dict(
        pool=["flower_red", "flower_orange", "flower_pink", "flower_purple"],
        num_objects=3,
        instruction="cut the stem of the {label}",
        cut=True,
    ),
    # Step 5a: the motor half of step 5 on its own. One flower, so there is no wrong stem to pick.
    "cut_one": dict(
        pool=["flower_red", "flower_orange", "flower_pink", "flower_purple"],
        num_objects=1,
        instruction="cut the stem of the {label}",
        cut=True,
    ),
    # Control for the look-alike result: the clutter scenes without the leaves.
    "pick_mixed": dict(pool=["cube", "can", "hex", "tblock", "cube_orange", "cube_pink", "cube_dark"], num_objects=4),
    # Step 4: three flowers on stems; grasp the named one by its stem, from the side, inside its band.
    "grasp_stem": dict(
        pool=["flower_red", "flower_orange", "flower_pink", "flower_purple"],
        num_objects=3,
        instruction="grasp the stem of the {label}",
    ),
}
MAX_LEAVES = 4
LEAF_HALF = (0.035, 0.0015, 0.065)  # metres: a plate 7 cm wide and 13 cm tall
LEAF_RGBA = (0.22, 0.52, 0.16, 1)
FRONT_CAMERA_XY = np.array([1.15, -0.55])

HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
CONTROL_HZ = 20
TIMESTEP = 0.002
SUBSTEPS = int(round(1.0 / (CONTROL_HZ * TIMESTEP)))
SPAWN_X = (0.38, 0.62)
SPAWN_Y = (-0.22, 0.22)
LIFT_HEIGHT = 0.08  # metres above rest height that counts as "lifted"
HOLD_STEPS = 10  # consecutive lifted control steps required for success
CONTACT_SOLREF = [0.004, 1.0]
CONTACT_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]
CUT_GAP = 0.0015  # cut_stem: half the gap a cut leaves between the ends (rounded caps included) of the two stem pieces
CUT_STEPS = 3  # cut_stem: consecutive control steps of env.holds_stem that cut a stem
SPLIT_Z = 0.09  # cut_stem: where a flower's top part meets its base until a cut (metres above the base frame)


def _look_at_quat(pos, target, up=(0, 0, 1)):
    """Quaternion for a MuJoCo camera (looks along -z, +y up) at `pos` aimed at `target`."""
    pos, target, up = (np.asarray(v, dtype=float) for v in (pos, target, up))
    f = target - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    mat = np.stack([r, u, -f], axis=1).reshape(-1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


def _split_stem(entries, z_at, split):
    """Split a flower's geoms at height `split`: (base part, top part). The stem capsule is cut in two; anything
    whose centre lies above the split goes to the top. `z_at` is the index of the height in each entry."""
    base, top = [], []
    for entry in entries:
        entry = list(entry)
        if entry[0] == "capsule":
            radius, half = entry[1]
            lo, hi = entry[z_at] - half, entry[z_at] + half
            base.append([*entry[:1], (radius, (split - lo) / 2), *entry[2:z_at], (lo + split) / 2, *entry[z_at + 1 :]])
            top.append([*entry[:1], (radius, (hi - split) / 2), *entry[2:z_at], (split + hi) / 2, *entry[z_at + 1 :]])
        else:
            (top if entry[z_at] > split else base).append(entry)
    return base, top


def build_model(cut_stems: bool = False) -> mujoco.MjModel:
    """`cut_stems`: build each flower as a base and a top part held together by a weld that a cut releases."""
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.option.timestep = TIMESTEP
    spec.option.impratio = 10
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    # Stiff contacts: with MuJoCo's default soft contacts (time constant 0.02 s) the arm's position actuators
    # pressed the hand 11 mm into the table and a finger 24 mm into a flower. 0.004 s is the stiffest the 2 ms
    # step allows (at least two steps).
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
    spec.option.o_solref = CONTACT_SOLREF
    spec.option.o_solimp = CONTACT_SOLIMP
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]

    mesh_names = {v[0] for o in OBJECTS.values() for v in o["visual"]} | {"table_top"}
    for name in sorted(mesh_names):
        mesh = spec.add_mesh()
        mesh.name = name
        mesh.file = str(ASSETS / "meshes" / f"{name}.obj")

    wb = spec.worldbody
    wb.add_light(pos=[0.4, -0.3, 1.6], dir=[0, 0.15, -1], diffuse=[0.5, 0.5, 0.5], castshadow=True)
    wb.add_light(pos=[0.9, 0.6, 1.2], dir=[-0.4, -0.4, -1], diffuse=[0.25, 0.25, 0.25], castshadow=False)
    wb.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], pos=[0, 0, -0.76], rgba=[0.30, 0.32, 0.36, 1]
    )
    # Table: top surface is z=0, the robot base sits on it.
    wb.add_geom(
        name="table_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="table_top",
        pos=[0.35, 0, -0.02],
        rgba=[0.62, 0.45, 0.28, 1],
        contype=0,
        conaffinity=0,
        group=2,
        density=0,
    )
    wb.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.4, 0.6, 0.02],
        pos=[0.35, 0, -0.02],
        group=3,
        friction=[1.0, 0.005, 0.0001],
    )
    wb.add_camera(
        name="front", pos=[1.15, -0.55, 0.62], quat=_look_at_quat([1.15, -0.55, 0.62], [0.42, 0.0, 0.10]), fovy=45
    )

    for i in range(MAX_LEAVES):  # moved into place at reset by tasks that use them, parked out of view otherwise
        leaf = wb.add_body(name=f"leaf{i}", mocap=True, pos=[2.0 + 0.3 * i, -2.0, -0.7])
        leaf.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(LEAF_HALF),
            rgba=list(LEAF_RGBA),
            contype=0,
            conaffinity=0,
            group=2,
        )

    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.004, 0.004, 0.004], rgba=[0, 0, 0, 0])
    hand.add_camera(
        name="wrist", pos=[0.075, 0, 0.035], quat=_look_at_quat([0.075, 0, 0.035], [0, 0, 0.17], up=(1, 0, 0)), fovy=80
    )

    geom_types = {
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    }

    def add_collision(body, prefix, entries):
        for j, (gtype, size, z, *density) in enumerate(entries):
            body.add_geom(
                name=f"{prefix}_col{j}",
                type=geom_types[gtype],
                size=list(size) + [0] * (3 - len(size)),
                pos=[0, 0, z],
                group=3,
                density=density[0] if density else 600,
                friction=[1.5, 0.01, 0.0002],
                condim=4,
                solref=[0.004, 1],
                solimp=[0.95, 0.99, 0.001, 0.5, 2],
            )

    def add_prim_visual(body, prefix, entries):
        for j, (gtype, size, rgba, z) in enumerate(entries):
            body.add_geom(
                name=f"{prefix}_vis{j}",
                type=geom_types[gtype],
                size=list(size) + [0] * (3 - len(size)),
                pos=[0, 0, z],
                rgba=list(rgba),
                contype=0,
                conaffinity=0,
                group=2,
                density=0,
            )

    for i, (name, obj) in enumerate(OBJECTS.items()):
        pos = [2.0 + 0.3 * i, 2.0, -0.76 + obj["rest_z"]]
        body = wb.add_body(name=name, pos=pos)
        body.add_freejoint()
        collision, prim_visual = obj["collision"], obj.get("prim_visual", [])
        split = cut_stems and obj.get("grasp") == "side"
        if split:
            collision, top_collision = _split_stem(collision, 2, SPLIT_Z)
            prim_visual, top_visual = _split_stem(prim_visual, 3, SPLIT_Z)
        add_collision(body, name, collision)
        add_prim_visual(body, name, prim_visual)
        if split:
            top = wb.add_body(name=f"{name}_top", pos=pos)  # same frame as the base: the weld's pose is identity
            top.add_freejoint()
            add_collision(top, f"{name}_top", top_collision)
            add_prim_visual(top, f"{name}_top", top_visual)
            exclude = spec.add_exclude()
            exclude.bodyname1, exclude.bodyname2 = name, f"{name}_top"
            weld = spec.add_equality()
            weld.name, weld.type, weld.objtype = f"{name}_weld", mujoco.mjtEq.mjEQ_WELD, mujoco.mjtObj.mjOBJ_BODY
            weld.name1, weld.name2 = f"{name}_top", name
            weld.data = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
            weld.solref = [0.005, 1]
        for mesh_name, rgba, z in obj["visual"]:
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                pos=[0, 0, z],
                rgba=list(rgba),
                contype=0,
                conaffinity=0,
                group=2,
                density=0,
            )
    return spec.compile()


class PickupEnv:
    camera_names = ("front", "wrist")
    state_dim = 8
    action_dim = 8

    def __init__(self, task: str = "pick_cube", image_size: int = 224, max_steps: int = 200):
        assert task in TASKS, f"unknown task {task}; choose from {list(TASKS)}"
        self.task = task
        self.image_size = image_size
        self.max_steps = max_steps
        self.cut_task = bool(TASKS[task].get("cut"))
        self.model = build_model(cut_stems=self.cut_task)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, image_size, image_size)
        self.tcp_id = self.model.site("tcp").id
        self.ctrl_lo = self.model.actuator_ctrlrange[:7, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:7, 1].copy()
        self.obj_qadr = {n: self.model.joint(self.model.body(n).jntadr[0]).qposadr[0] for n in OBJECTS}
        self.obj_vadr = {n: self.model.joint(self.model.body(n).jntadr[0]).dofadr[0] for n in OBJECTS}
        self.leaf_mocap = [int(self.model.body(f"leaf{i}").mocapid[0]) for i in range(MAX_LEAVES)]
        self.finger_bodies = (self.model.body("left_finger").id, self.model.body("right_finger").id)
        self.obj_body = {n: self.model.body(n).id for n in OBJECTS}
        # cut_stem: each flower's top part, its weld, and its two stem pieces (resized at a cut)
        self.stems = [n for n in OBJECTS if self.cut_task and OBJECTS[n].get("grasp") == "side"]
        m = self.model
        self.top_body = {n: m.body(f"{n}_top").id for n in self.stems}
        self.top_qadr = {n: m.joint(m.body(f"{n}_top").jntadr[0]).qposadr[0] for n in self.stems}
        self.top_vadr = {n: m.joint(m.body(f"{n}_top").jntadr[0]).dofadr[0] for n in self.stems}
        self.weld = {n: m.equality(f"{n}_weld").id for n in self.stems}
        self.stem_pieces = {
            n: [
                g
                for g in range(m.ngeom)
                if m.geom_bodyid[g] in (self.obj_body[n], self.top_body[n])
                and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
            ]
            for n in self.stems
        }
        self._geom_defaults = (m.geom_size.copy(), m.geom_pos.copy(), m.geom_rbound.copy(), m.geom_aabb.copy())
        # While a flower is whole its two stem pieces overlap and must not collide (an exclusion); once cut they
        # are apart and do collide, so the top cannot slide down through the base.
        self._exclude_defaults = m.exclude_signature.copy()
        self.exclude_index = {
            n: int(np.flatnonzero(m.exclude_signature == (min(b, t) << 16) + max(b, t))[0])
            for n in self.stems
            for b, t in [(self.obj_body[n], self.top_body[n])]
        }
        self.cut: set[str] = set()
        self.hold_counts: dict[str, int] = {}
        self.leaves: list[list[float]] = []  # [x, y, yaw] of each leaf in use
        self.active: list[str] = []
        self.target: str = ""
        self.instruction: str = ""
        self.rest_z: dict[str, float] = {}

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int, target_index: int | None = None):
        """Build the layout of `seed`. `target_index` names one of the layout's objects (0, 1, ...) as
        the target instead of the seed's own choice, which gives the same scene with another instruction."""
        rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        q0 = HOME_Q + rng.normal(0, 0.03, 7)
        d.qpos[:7] = q0
        d.qpos[7:9] = 0.04
        d.ctrl[:7] = q0
        d.ctrl[7] = 255.0

        cfg = TASKS[self.task]
        self.active = list(rng.choice(cfg["pool"], size=cfg["num_objects"], replace=False))
        self.target = str(rng.choice(self.active))  # always drawn, so the layout does not depend on target_index
        if target_index is not None:
            self.target = str(self.active[target_index])
        self.instruction = cfg.get("instruction", "pick up the {label}").format(label=OBJECTS[self.target]["label"])

        placed = []
        for i, name in enumerate(OBJECTS):
            adr = self.obj_qadr[name]
            if name in self.active:
                for _ in range(200):
                    xy = np.array([rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y)])
                    if all(np.linalg.norm(xy - p) > 0.11 for p in placed):
                        break
                placed.append(xy)
                yaw = rng.uniform(-np.pi, np.pi)
                d.qpos[adr : adr + 3] = [xy[0], xy[1], OBJECTS[name]["rest_z"] + 0.002]
                d.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            else:  # parked on the floor, far outside both camera views
                d.qpos[adr : adr + 3] = [2.0 + 0.3 * i, 2.0, -0.76 + OBJECTS[name]["rest_z"]]
                d.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
        self._place_leaves(rng, placed, cfg.get("leaves", 0))  # draws only for tasks with leaves
        if self.cut_task:  # whole flowers again: stem pieces back to their first length, welds on, tops in place
            m.geom_size[:], m.geom_pos[:], m.geom_rbound[:], m.geom_aabb[:] = self._geom_defaults
            m.exclude_signature[:] = self._exclude_defaults
            for n in self.stems:
                d.eq_active[self.weld[n]] = 1
                self._sync_top(n)
            self.cut, self.hold_counts = set(), {}
        mujoco.mj_forward(m, d)
        for _ in range(100):  # let objects settle
            mujoco.mj_step(m, d)
        self.rest_z = {n: float(d.qpos[self.obj_qadr[n] + 2]) for n in self.active}
        self.last_action = np.concatenate([d.ctrl[:7], [1.0]]).astype(np.float32)
        self.steps = 0
        self.lift_count = 0
        return self.get_obs()

    # ------------------------------------------------------------------- step
    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.data.ctrl[:7] = np.clip(action[:7], self.ctrl_lo, self.ctrl_hi)
        self.data.ctrl[7] = 255.0 * float(np.clip(action[7], 0.0, 1.0))
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self.last_action = np.concatenate([self.data.ctrl[:7], [self.data.ctrl[7] / 255.0]]).astype(np.float32)
        self.steps += 1

        if OBJECTS[self.target].get("grasp") == "side":
            # A stem counts as taken while env.holds_stem holds; success needs HOLD_STEPS steps in a row.
            held = [n for n in self.active if self.holds_stem(n)]
            target_up = self.target in held
            if self.cut_task:
                for n in self.active:
                    self.hold_counts[n] = self.hold_counts.get(n, 0) + 1 if n in held else 0
                    if self.hold_counts[n] >= CUT_STEPS and n not in self.cut:
                        self._cut(n)
                wrong = any(n != self.target for n in self.cut)
                success = self.target in self.cut and not wrong
                done = success or wrong or self.steps >= self.max_steps
                info = dict(success=success, lifted=held, wrong_object=wrong, cut=sorted(self.cut))
                return self.get_obs(), done, info
            self.lift_count = self.lift_count + 1 if target_up else 0
            success = self.lift_count >= HOLD_STEPS
            done = success or self.steps >= self.max_steps
            info = dict(success=success, lifted=held, wrong_object=bool(held) and not target_up)
            return self.get_obs(), done, info
        lifted = self.lifted_objects()
        target_up = self.target in lifted
        self.lift_count = self.lift_count + 1 if target_up else 0
        success = self.lift_count >= HOLD_STEPS
        done = success or self.steps >= self.max_steps
        info = dict(success=success, lifted=lifted, wrong_object=bool(lifted) and not target_up)
        return self.get_obs(), done, info

    # ---------------------------------------------------------------- queries
    def holds_stem(self, name):
        """The gripper has closed on `name`: the gripper is commanded shut, the stem lies between the fingers and
        inside the fingertips' depth, and a finger presses on it; the gripper's centre is inside the stem's band,
        and the stem still stands (< 20 degrees). The fingers are coupled, so on a stem that cannot slide to the
        middle they stop when the first one touches it; for a cut, a stem caught between the jaws is enough."""
        d = self.data
        bodies = (self.obj_body[name], self.top_body[name]) if name in self.top_body else (self.obj_body[name],)
        touching = any(
            {self.model.geom_bodyid[d.contact[i].geom1], self.model.geom_bodyid[d.contact[i].geom2]} == {f, body}
            for i in range(d.ncon)
            for f in self.finger_bodies
            for body in bodies
        )
        lo, hi = OBJECTS[name]["band"]
        tcp = self.tcp_pos()
        R = d.site_xmat[self.tcp_id].reshape(3, 3)
        base = self.object_pos(name)
        rel = R.T @ (np.array([base[0], base[1], tcp[2]]) - tcp)  # the stem's axis at the gripper's height, hand frame
        opening = float(self.state()[7]) * 0.08
        between = abs(rel[1]) < opening / 2 and abs(rel[2]) < 0.015
        closed = self.data.ctrl[7] < 0.5 * 255
        adr = self.obj_qadr[name]
        x, y = d.qpos[adr + 4], d.qpos[adr + 5]
        upright = 1 - 2 * (x * x + y * y) > np.cos(np.radians(20))  # the body's z axis against the world's
        return bool(touching and between and closed and lo <= tcp[2] <= hi and upright)

    def lifted_objects(self):
        return [n for n in self.active if self.object_pos(n)[2] - self.rest_z[n] > LIFT_HEIGHT]

    def object_pos(self, name):
        adr = self.obj_qadr[name]
        return self.data.qpos[adr : adr + 3].copy()

    def object_yaw(self, name):
        adr = self.obj_qadr[name]
        w, x, y, z = self.data.qpos[adr + 3 : adr + 7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def set_arm(self, q):
        """Put the arm in joint configuration `q` at rest, holding it (used to start an episode somewhere
        other than the home pose). Returns the new observation."""
        self.data.qpos[:7] = q
        self.data.qvel[:9] = 0
        self.data.ctrl[:7] = q
        mujoco.mj_forward(self.model, self.data)
        self.last_action = np.concatenate([self.data.ctrl[:7], [self.data.ctrl[7] / 255.0]]).astype(np.float32)
        return self.get_obs()

    def _place_leaves(self, rng, placed, count):
        """Stand `count` leaves on the table, each between an object and the front camera, roughly facing it."""
        d = self.data
        self.leaves = []
        for i in range(MAX_LEAVES):
            mid = self.leaf_mocap[i]
            if i >= count:
                d.mocap_pos[mid] = [2.0 + 0.3 * i, -2.0, -0.7]
                d.mocap_quat[mid] = [1, 0, 0, 0]
                continue
            obj = placed[int(rng.integers(len(placed)))]
            to_cam = FRONT_CAMERA_XY - obj
            to_cam /= np.linalg.norm(to_cam)
            side = np.array([-to_cam[1], to_cam[0]])
            xy = obj + to_cam * rng.uniform(0.045, 0.08) + side * rng.uniform(-0.035, 0.035)
            yaw = float(np.arctan2(to_cam[1], to_cam[0]) + np.pi / 2 + rng.uniform(-0.5, 0.5))
            d.mocap_pos[mid] = [xy[0], xy[1], LEAF_HALF[2]]
            d.mocap_quat[mid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            self.leaves.append([float(xy[0]), float(xy[1]), yaw])

    def move_object(self, name, xy, yaw=None):
        """Put an object down somewhere else on the table, upright and at rest: a disturbance, used to
        test how a policy copes and to record demonstrations of recovery. Yaw is kept unless given."""
        adr, vadr = self.obj_qadr[name], self.obj_vadr[name]
        if yaw is None:
            yaw = self.object_yaw(name)
        self.data.qpos[adr : adr + 2] = xy
        self.data.qpos[adr + 2] = self.rest_z[name] + 0.002
        self.data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[vadr : vadr + 6] = 0
        if name in self.top_body and name not in self.cut:
            self._sync_top(name)
        mujoco.mj_forward(self.model, self.data)

    def _sync_top(self, name):
        """Put a flower's top part back on its base (same pose, at rest)."""
        adr, top = self.obj_qadr[name], self.top_qadr[name]
        self.data.qpos[top : top + 7] = self.data.qpos[adr : adr + 7]
        self.data.qvel[self.top_vadr[name] : self.top_vadr[name] + 6] = 0

    def _cut(self, name):
        """Cut a stem where the jaws are: the base keeps the stem below, the top part keeps the stem above and
        is let go (its weld is switched off)."""
        m = self.model
        lo, hi = OBJECTS[name]["band"]
        z = float(np.clip(self.tcp_pos()[2], lo, hi) - self.object_pos(name)[2])  # in the flower's frame
        for g in self.stem_pieces[name]:
            radius, half, centre = m.geom_size[g][0], m.geom_size[g][1], m.geom_pos[g][2]
            # collision capsules end CUT_GAP from the cut including their rounded caps, so the pieces never overlap;
            # the visible stem keeps a thin gap
            end = CUT_GAP + radius if m.geom_contype[g] or m.geom_conaffinity[g] else 0.002
            if m.geom_bodyid[g] == self.obj_body[name]:
                bottom, top = centre - half, z - end
            else:
                bottom, top = z + end, centre + half
            half = max((top - bottom) / 2, 0.001)
            m.geom_size[g][1], m.geom_pos[g][2] = half, (top + bottom) / 2
            m.geom_rbound[g] = radius + half
            m.geom_aabb[g] = [0, 0, 0, radius, radius, half + radius]
        self.data.eq_active[self.weld[name]] = 0
        # the pieces collide from now on: point the exclusion at a body pair that does not exist (body 65535),
        # keeping the list sorted for MuJoCo's lookup
        i = self.exclude_index[name]
        m.exclude_signature[i] = (m.exclude_signature[i] >> 16 << 16) + 0xFFFF
        self.cut.add(name)
        mujoco.mj_forward(m, self.data)

    def on_table(self, name):
        """Is the object still resting at table height (not lifted, not carried)?"""
        return bool(self.object_pos(name)[2] - self.rest_z[name] < 0.01)

    def tcp_pos(self):
        return self.data.site_xpos[self.tcp_id].copy()

    def tcp_yaw(self):
        """World yaw of the finger closing axis (the tcp site's y axis)."""
        R = self.data.site_xmat[self.tcp_id].reshape(3, 3)
        return float(np.arctan2(R[1, 1], R[0, 1]))

    def state(self):
        d = self.data
        return np.concatenate([d.qpos[:7], [(d.qpos[7] + d.qpos[8]) / 0.08]]).astype(np.float32)

    def render(self, camera: str, size: int | None = None):
        if size is not None and size != self.image_size:
            with mujoco.Renderer(self.model, size, size) as r:
                r.update_scene(self.data, camera=camera)
                return r.render().copy()
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def get_obs(self):
        obs = {cam: self.render(cam) for cam in self.camera_names}
        obs["state"] = self.state()
        return obs

    def close(self):
        self.renderer.close()
