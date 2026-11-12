"""MuJoCo tabletop pick-up environment: Franka Panda + Blender-authored objects.

Observation: two RGB cameras (front, wrist) + 8-D state (7 joint angles, gripper opening in [0, 1]).
Action:      8-D absolute target (7 joint position targets, gripper command 1=open / 0=closed).
Control runs at 20 Hz (25 physics substeps of 2 ms).

Tasks
  pick_cube    one red cube; "pick up the red cube"
  pick_target  three of four objects on the table; "pick up the <object>" names the target
  prune_one / prune_turn / prune  a potted plant on a turntable with two push buttons; cut the side shoots off
               the main stem without damaging the flower or the stem (see TASKS)
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
FLOWER_COLOURS = (
    ("red", (0.86, 0.10, 0.12, 1)),
    ("orange", (0.93, 0.47, 0.08, 1)),
    ("pink", (0.93, 0.45, 0.70, 1)),
    ("purple", (0.46, 0.22, 0.72, 1)),
)
for _colour, _rgba in FLOWER_COLOURS:
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
    # Pruning (toward steps 6-8): a potted plant on a motorised turntable with two push buttons. Cut every side
    # shoot off the main stem, keep the main stem and its flower. Holding a shoot between the shut jaws inside its
    # cut zone for CUT_STEPS steps cuts it; touching the flower, pushing hard on the main stem or cutting it is
    # damage and fails the episode. prune_one: one shoot already facing the robot. prune_turn: one shoot facing
    # anywhere, so the arm presses a button to turn the plant first when it faces away. prune: 2-3 shoots.
    "prune_one": dict(
        plant=True,
        shoots=(1, 1),
        facing=True,
        num_objects=1,
        max_steps=300,
        instruction="cut the side shoot off the stem, keep the flower",
    ),
    "prune_turn": dict(
        plant=True,
        shoots=(1, 1),
        num_objects=1,
        max_steps=450,
        instruction="cut the side shoot off the stem, keep the flower",
    ),
    "prune": dict(
        plant=True,
        shoots=(2, 3),
        num_objects=3,
        max_steps=1000,
        instruction="cut the side shoots off the stem, keep the flower",
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

# Pruning: the plant stand, the plant and its side shoots (meshes and their frames: assets/build_plant_blender.py).
PLANT_MESHES = (
    "plant_pot",
    "plant_soil",
    "plant_main_stem",
    "plant_flower_petals",
    "plant_flower_centre",
    "plant_flower_sepals",
    "plant_leaf",
    "plant_stand_base",
    "plant_stand_plate",
    "plant_stand_plate_marks",
    "plant_button",
    "plant_arrow_cw",
    "plant_arrow_ccw",
)
STAND_XY = (0.55, 0.0)  # the stand's axis on the table; its control panel faces the robot (-x)
STAND_JITTER = 0.015  # metres, uniform in x and y per episode
PLATE_Z, PLATE_TOP = 0.030, 0.012  # plate bottom in the stand frame; plate thickness (the pot stands on top)
STEM_BASE, STEM_H, STEM_R = 0.068, 0.240, 0.005  # main stem: base above the pot bottom, length, collision radius
# Buttons in the stand frame: (x, y) of the cap centre and the plate's turning direction while held down.
# Seen from the robot, the left (+y, blue) button turns the plant counter-clockwise (seen from above).
BUTTONS = {"ccw": ((-0.145, 0.045), +1), "cw": ((-0.145, -0.045), -1)}
BUTTON_Z, BUTTON_H, BUTTON_R = 0.026, 0.010, 0.016  # recess floor (cap bottom at rest), cap height, cap radius
BUTTON_TRAVEL = 0.006  # metres a button can be pushed down
PRESS_TRAVEL = 0.003  # metres of travel that switch the motor on
TURN_SPEED = 0.7  # rad/s of the plate while a button is held down
MAX_SHOOTS = 3
SHOOT_R = 0.003
SHOOT_SPLIT = 0.015  # where a shoot's stub (on the plant) meets its outer part until a cut, from the stem's axis
CUT_ZONE = (0.02, 0.05)  # a shoot is cut where the jaws hold it, this far out along it from the stem's axis
CUT_MID = 0.035  # the outer part's frame: on the shoot's axis, this far out (env.object_pos of a shoot)
# where shoots leave the main stem (above its base), and their least spacing. The highest stays low enough for the
# open jaws to cut it without touching the flower, the lowest high enough for them to clear the pot's rim.
SHOOT_HEIGHTS, SHOOT_GAP = (0.07, 0.155), 0.035
SHOOT_ELEVATION, SHOOT_LENGTH = (np.radians(15), np.radians(35)), (0.07, 0.09)
SHOOT_SPREAD = np.radians(100)  # least angle between two shoots, seen from above
REACH_AZIMUTH = np.pi  # a shoot pointing this way (toward the robot) is in reach without turning the plant
FACING_SPREAD = np.radians(20)  # prune_one: the shoot points within this of REACH_AZIMUTH
LEAF_SCALE = 0.8
STEM_PUSH_LIMIT = 5.0  # newtons: summed normal force of the robot on the main stem above which it is damaged
TERRACOTTA, SOIL = (0.72, 0.36, 0.20, 1), (0.22, 0.15, 0.09, 1)
HOUSING, PLATE_RGBA, MARKS = (0.13, 0.13, 0.14, 1), (0.40, 0.40, 0.43, 1), (0.92, 0.92, 0.92, 1)
BUTTON_RGBA = {"ccw": (0.10, 0.32, 0.88, 1), "cw": (0.95, 0.48, 0.06, 1)}
FLOWER_CENTRE, LEAF_GREEN = (0.36, 0.20, 0.05, 1), (0.18, 0.50, 0.13, 1)


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


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(-1))
    return q


def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _rot(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def _shoot_frame(d):
    """Frame of a shoot pointing along `d` (unit): z along the shoot, y horizontal, x upward (perpendicular)."""
    y = np.cross(d, [0.0, 0.0, 1.0])
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, d), y, d], axis=1)


GEOM_TYPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "mesh": mujoco.mjtGeom.mjGEOM_MESH,
}


def _solid(body, name, gtype, size, pos=(0, 0, 0), quat=(1, 0, 0, 0), density=600, contype=1, conaffinity=3):
    """A collision geom (hidden: group 3). conaffinity bit 2 lets fallen leaves rest on it."""
    return body.add_geom(
        name=name,
        type=GEOM_TYPES[gtype],
        size=list(size) + [0] * (3 - len(size)),
        pos=list(pos),
        quat=list(quat),
        group=3,
        density=density,
        friction=[1.5, 0.01, 0.0002],
        condim=4,
        contype=contype,
        conaffinity=conaffinity,
        solref=list(CONTACT_SOLREF),
        solimp=list(CONTACT_SOLIMP),
    )


def _look(body, name, rgba, mesh=None, gtype="mesh", size=(0, 0, 0), pos=(0, 0, 0), quat=(1, 0, 0, 0)):
    """A visual-only geom (no contacts, no mass)."""
    return body.add_geom(
        name=name,
        type=GEOM_TYPES[gtype],
        meshname=mesh or "",
        size=list(size) + [0] * (3 - len(size)),
        pos=list(pos),
        quat=list(quat),
        rgba=list(rgba),
        contype=0,
        conaffinity=0,
        group=2,
        density=0,
    )


def _add_plant(spec):
    """The plant stand (fixed housing with two spring-loaded push buttons, a plate on a motorised hinge), the potted
    plant on the plate, and MAX_SHOOTS side shoots. Each shoot is a stub on the plant and an outer part (a free body
    welded to the plant, not colliding with it while attached). Shoot geometry is set per episode (env.reset)."""
    wb = spec.worldbody
    stand = wb.add_body(name="stand", pos=[STAND_XY[0], STAND_XY[1], 0])  # static: welded to the table
    _look(stand, "stand_vis", HOUSING, "plant_stand_base")
    _solid(stand, "stand_col0", "cylinder", (0.11, 0.015), (0, 0, 0.015))
    _solid(stand, "stand_col1", "box", (0.06, 0.066, 0.0115), (-0.12, 0, 0.0145))  # panel, top at the recess floor
    for name, ((bx, by), _) in BUTTONS.items():
        button = stand.add_body(name=f"button_{name}", pos=[bx, by, BUTTON_Z])
        button.add_joint(
            name=f"button_{name}",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0, 0, 1],
            range=[-BUTTON_TRAVEL, 0],
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            stiffness=400.0,  # N/m, preloaded 2 mm against its upper stop: 0.8 N at rest, 3.2 N fully down
            springref=0.002,
            damping=1.0,
            armature=0.001,
            solref_limit=[0.004, 1],
        )
        _solid(button, f"button_{name}_col", "cylinder", (BUTTON_R, BUTTON_H / 2), (0, 0, BUTTON_H / 2), density=1000)
        _look(button, f"button_{name}_vis", BUTTON_RGBA[name], "plant_button")
        _look(button, f"button_{name}_arrow", MARKS, f"plant_arrow_{name}", pos=(0, 0, 0.0094))
    plate = stand.add_body(name="plate", pos=[0, 0, PLATE_Z])
    plate.add_joint(name="plate_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=0.02, armature=0.005)
    _look(plate, "plate_vis", PLATE_RGBA, "plant_stand_plate")
    _look(plate, "plate_marks", MARKS, "plant_stand_plate_marks")
    _solid(plate, "plate_col", "cylinder", (0.1, PLATE_TOP / 2), (0, 0, PLATE_TOP / 2), density=1500)
    motor = spec.add_actuator(name="turntable", target="plate_hinge", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    kv = 1.0  # a velocity servo: holds the plate still (control 0) against the arm, up to the force range
    motor.gaintype, motor.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
    motor.gainprm[0], motor.biasprm[2] = kv, -kv
    motor.ctrlrange, motor.forcerange = [-1, 1], [-3, 3]
    motor.ctrllimited = motor.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE

    plant = plate.add_body(name="plant", pos=[0, 0, PLATE_TOP])  # origin: the pot's bottom centre
    _look(plant, "pot_vis", TERRACOTTA, "plant_pot")
    _look(plant, "soil_vis", SOIL, "plant_soil")
    _solid(plant, "pot_col0", "cylinder", (0.041, 0.020), (0, 0, 0.020))
    _solid(plant, "pot_col1", "cylinder", (0.0475, 0.0143), (0, 0, 0.0543))  # up to the soil, z = 0.0685
    for i in range(16):  # the rim above the soil, as a ring of blocks
        a = 2 * np.pi * i / 16
        _solid(
            plant,
            f"pot_rim{i}",
            "box",
            (0.003, 0.0095, 0.0033),
            (0.0468 * np.cos(a), 0.0468 * np.sin(a), 0.0717),
            _quat(_rot("z", a)),
        )
    _look(plant, "stem_vis", STEM_GREEN, "plant_main_stem", pos=(0, 0, STEM_BASE))
    _solid(plant, "stem_col", "capsule", (STEM_R, STEM_H / 2), (0, 0, STEM_BASE + STEM_H / 2))
    top = STEM_BASE + STEM_H
    for part, rgba in (("petals", FLOWER_COLOURS[0][1]), ("centre", FLOWER_CENTRE), ("sepals", STEM_GREEN)):
        _look(plant, f"flower_{part}", rgba, f"plant_flower_{part}", pos=(0, 0, top))
    tilt = _rot("y", np.radians(15))  # the head leans 15 degrees toward the plant's +x
    _solid(plant, "flower_col", "cylinder", (0.034, 0.011), [0, 0, top] + tilt @ [0, 0, 0.011], _quat(tilt))
    for k in range(MAX_SHOOTS):
        _solid(plant, f"stub{k}_col", "capsule", (SHOOT_R, 0.005))
        _look(plant, f"stub{k}_vis", STEM_GREEN, gtype="capsule", size=(SHOOT_R, 0.005))
    for k in range(MAX_SHOOTS):
        shoot = wb.add_body(name=f"shoot{k}", pos=[2.0 + 0.3 * k, -2.5, -0.75])
        shoot.add_freejoint()
        _solid(shoot, f"shoot{k}_col", "capsule", (SHOOT_R, 0.03), (0, 0, 0.01), density=500, conaffinity=1)
        _look(shoot, f"shoot{k}_vis", STEM_GREEN, gtype="capsule", size=(SHOOT_R, 0.03), pos=(0, 0, 0.01))
        for j in range(2):  # leaves: a thin box under each blade; solid to the arm, the table and the plant, but
            # not to each other (contype 2, conaffinity 1)
            _look(shoot, f"shoot{k}_leaf{j}", LEAF_GREEN, "plant_leaf")
            _solid(shoot, f"shoot{k}_leaf{j}_col", "box", (0.019, 0.008, 0.0015), density=300, contype=2, conaffinity=1)
        exclude = spec.add_exclude()
        exclude.bodyname1, exclude.bodyname2 = "plant", f"shoot{k}"
        weld = spec.add_equality()
        weld.name, weld.type, weld.objtype = f"shoot{k}_weld", mujoco.mjtEq.mjEQ_WELD, mujoco.mjtObj.mjOBJ_BODY
        weld.name1, weld.name2 = f"shoot{k}", "plant"
        weld.data = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]  # the relative pose is set per episode
        weld.solref = [0.005, 1]
        weld.active = False
    for other in ("plate", "button_ccw", "button_cw"):  # parts of the stand that touch the housing
        exclude = spec.add_exclude()
        exclude.bodyname1, exclude.bodyname2 = "stand", other


def build_model(cut_stems: bool = False, plant: bool = False) -> mujoco.MjModel:
    """`cut_stems`: build each flower as a base and a top part held together by a weld that a cut releases.
    `plant`: add the plant stand and the potted plant of the pruning tasks."""
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
    if plant:
        mesh_names |= set(PLANT_MESHES)
        # Shoot geometry moves per episode, so the static bounding-volume tree MuJoCo builds at compile time for
        # bodies with several geoms would be stale: no mid-phase (the broad phase and exact tests remain).
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 960, 540  # room for a wide 960x540 view
    for name in sorted(mesh_names):
        mesh = spec.add_mesh()
        mesh.name = name
        mesh.file = str(ASSETS / "meshes" / f"{name}.obj")
        if name == "plant_leaf":
            mesh.scale = [LEAF_SCALE] * 3

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
    table = wb.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.4, 0.6, 0.02],
        pos=[0.35, 0, -0.02],
        group=3,
        friction=[1.0, 0.005, 0.0001],
    )
    if plant:
        table.conaffinity = 3  # fallen leaves (contype 2) rest on it
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
    if plant:
        _add_plant(spec)
    return spec.compile()


class PickupEnv:
    camera_names = ("front", "wrist")
    state_dim = 8
    action_dim = 8

    def __init__(self, task: str = "pick_cube", image_size: int = 224, max_steps: int | None = None):
        """`max_steps`: episode length limit; by default the task's own (200 unless TASKS says otherwise)."""
        assert task in TASKS, f"unknown task {task}; choose from {list(TASKS)}"
        self.task = task
        self.image_size = image_size
        self.max_steps = max_steps if max_steps is not None else TASKS[task].get("max_steps", 200)
        self.cut_task = bool(TASKS[task].get("cut"))
        self.plant_task = bool(TASKS[task].get("plant"))
        self.model = build_model(cut_stems=self.cut_task, plant=self.plant_task)
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
        if self.plant_task:
            self._init_plant()
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
        if self.plant_task:
            self.active = []
            self._reset_plant(rng)
        else:
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
        if self.plant_task:
            self._place_shoots()
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
        if self.plant_task and self.target in self.cut:  # the next shoot is named once the step after a cut begins
            self.target = next((n for n in self.active if n not in self.cut), self.target)
        for _ in range(SUBSTEPS):
            if self.plant_task:
                self._drive_turntable()
            mujoco.mj_step(self.model, self.data)
            if self.plant_task:
                self._check_damage()
        self.last_action = np.concatenate([self.data.ctrl[:7], [self.data.ctrl[7] / 255.0]]).astype(np.float32)
        self.steps += 1
        if self.plant_task:
            return self._plant_result()

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

    # ---------------------------------------------------------------- pruning
    def _init_plant(self):
        m = self.model
        self.stand_body, self.plant_body = m.body("stand").id, m.body("plant").id
        self.plate_qadr = m.joint("plate_hinge").qposadr[0]
        self.plate_vadr = m.joint("plate_hinge").dofadr[0]
        self.button_qadr = {n: m.joint(f"button_{n}").qposadr[0] for n in BUTTONS}
        self.turntable = m.actuator("turntable").id
        self.petals_geom, self.stem_geom = m.geom("flower_petals").id, m.geom("stem_col").id
        self.flower_geom = m.geom("flower_col").id
        root = m.body("link0").id
        self.robot_bodies = np.array([b for b in range(m.nbody) if m.body_rootid[b] == root])
        self.shoot_names = [f"shoot{k}" for k in range(MAX_SHOOTS)]
        for n in self.shoot_names:
            body = m.body(n)
            self.obj_qadr[n] = m.joint(body.jntadr[0]).qposadr[0]
            self.obj_vadr[n] = m.joint(body.jntadr[0]).dofadr[0]
            self.obj_body[n] = body.id
        self.stub_geoms = {
            n: (m.geom(f"stub{k}_col").id, m.geom(f"stub{k}_vis").id) for k, n in enumerate(self.shoot_names)
        }
        self.outer_geoms = {n: (m.geom(f"{n}_col").id, m.geom(f"{n}_vis").id) for n in self.shoot_names}
        self.leaf_geoms = {
            n: [(m.geom(f"{n}_leaf{j}").id, m.geom(f"{n}_leaf{j}_col").id) for j in range(2)] for n in self.shoot_names
        }
        self.shoot_weld = {n: m.equality(f"{n}_weld").id for n in self.shoot_names}
        # Geoms moved per episode: the compiler marks a geom placed at its body's frame as "same frame" and the
        # kinematics then ignores its geom_pos; likewise a body's inertial frame. Clear those marks.
        for n in self.shoot_names:
            moved = [*self.stub_geoms[n], *self.outer_geoms[n], *(g for pair in self.leaf_geoms[n] for g in pair)]
            m.geom_sameframe[moved] = 0
            m.body_sameframe[self.obj_body[n]] = 0
        plant = self.plant_body
        self._shoot_exclusion = {
            n: (min(plant, self.obj_body[n]) << 16) + max(plant, self.obj_body[n]) for n in self.shoot_names
        }
        self._fixed_exclusions = [s for s in m.exclude_signature if s not in self._shoot_exclusion.values()]
        assert len(self._fixed_exclusions) + MAX_SHOOTS == m.nexclude
        self.shoots: dict[str, dict] = {}  # per episode: shoot geometry in the plant frame
        self.damaged: str | None = None
        self.stem_hold = 0
        self.max_stem_push = 0.0  # largest summed normal force of the robot on the main stem this episode (N)
        self.step_stem_push = 0.0  # the same, within the current control step

    def _reset_plant(self, rng):
        """Draw this episode's stand position, plate angle, flower colour and shoots (geometry only; the outer parts
        are put in place by _place_shoots once the plate's pose is known)."""
        m, d = self.model, self.data
        cfg = TASKS[self.task]
        m.body_pos[self.stand_body][:2] = np.array(STAND_XY) + rng.uniform(-STAND_JITTER, STAND_JITTER, 2)
        theta = rng.uniform(-np.pi, np.pi)
        d.qpos[self.plate_qadr] = theta
        self.flower_colour, rgba = FLOWER_COLOURS[int(rng.integers(len(FLOWER_COLOURS)))]
        m.geom_rgba[self.petals_geom] = rgba
        n = int(rng.integers(cfg["shoots"][0], cfg["shoots"][1] + 1))
        # uniform over the spacings allowed: sorted draws on a shortened range, then spread by the least gap
        free = SHOOT_HEIGHTS[1] - SHOOT_HEIGHTS[0] - (n - 1) * SHOOT_GAP
        heights = SHOOT_HEIGHTS[0] + np.sort(rng.uniform(0, free, n)) + SHOOT_GAP * np.arange(n)
        for _ in range(1000):  # seen from above, in the world
            if cfg.get("facing"):
                azimuths = REACH_AZIMUTH + rng.uniform(-FACING_SPREAD, FACING_SPREAD, n)
            else:
                azimuths = rng.uniform(-np.pi, np.pi, n)
            gaps = [abs(_wrap_angle(a - b)) for i, a in enumerate(azimuths) for b in azimuths[i + 1 :]]
            if min(gaps, default=np.pi) >= SHOOT_SPREAD:
                break
        self.shoots = {}
        for k, name in enumerate(self.shoot_names):
            if k >= n:
                self._park_shoot(name)
                continue
            a, e = azimuths[k] - theta, rng.uniform(*SHOOT_ELEVATION)
            direction = np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
            shoot = dict(
                origin=np.array([0.0, 0.0, STEM_BASE + heights[k]]),
                dir=direction,
                frame=_shoot_frame(direction),
                length=float(rng.uniform(*SHOOT_LENGTH)),
            )
            self.shoots[name] = shoot
            self._shape_shoot(name, SHOOT_SPLIT, SHOOT_SPLIT)
            count = int(rng.integers(1, 3))
            yaws = [rng.uniform(-0.25, 0.25)] if count == 1 else [s * rng.uniform(0.5, 0.85) for s in (-1, 1)]
            tip = shoot["length"] - CUT_MID
            base = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=float)  # leaf blade along the shoot, face up
            for j, (vis, col) in enumerate(self.leaf_geoms[name]):
                used = j < count
                m.geom_group[vis] = 2 if used else 3
                m.geom_contype[col], m.geom_conaffinity[col] = (2, 1) if used else (0, 0)
                if not used:
                    continue
                R = base @ _rot("z", yaws[j]) @ _rot("y", rng.uniform(0.25, 0.5)) @ _rot("x", rng.uniform(-0.25, 0.25))
                pos = np.array([0.0, 0.0, tip - 0.001])
                m.geom_pos[vis], m.geom_quat[vis] = pos, _quat(R)
                m.geom_pos[col], m.geom_quat[col] = pos + R @ [0.028, 0, 0.002], _quat(R)
            # The weld: anchored at the outer part's origin (CUT_MID out along the shoot, given in the plant's frame),
            # where the outer part's own origin must stay; its orientation relative to the plant is the shoot's frame.
            # (Anchored at the plant's origin, 25 cm away, a small angular give of the weld moved the shoot by mm.)
            data = m.eq_data[self.shoot_weld[name]]
            data[:3], data[3:6], data[6:10] = shoot["origin"] + direction * CUT_MID, 0, _quat(shoot["frame"].T)
        self.active = [n for n in self.shoot_names if n in self.shoots]  # from the lowest shoot up
        self.target = self.active[0]
        self.instruction = cfg["instruction"]
        self.cut, self.hold_counts = set(), {}
        self.damaged, self.stem_hold, self.max_stem_push, self.step_stem_push = None, 0, 0.0, 0.0
        d.ctrl[self.turntable] = 0.0
        self._set_exclusions()

    def _place_shoots(self):
        """Outer parts on their stubs (welded, at rest), at the plant's pose for this episode."""
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        Rp, pp = d.xmat[self.plant_body].reshape(3, 3), d.xpos[self.plant_body]
        for name, shoot in self.shoots.items():
            adr, vadr = self.obj_qadr[name], self.obj_vadr[name]
            d.qpos[adr : adr + 3] = pp + Rp @ (shoot["origin"] + shoot["dir"] * CUT_MID)
            d.qpos[adr + 3 : adr + 7] = _quat(Rp @ shoot["frame"])
            d.qvel[vadr : vadr + 6] = 0
            d.eq_active[self.shoot_weld[name]] = 1

    def _park_shoot(self, name):
        m, d = self.model, self.data
        k = self.shoot_names.index(name)
        adr, vadr = self.obj_qadr[name], self.obj_vadr[name]
        d.qpos[adr : adr + 7] = [2.0 + 0.3 * k, -2.5, -0.75, 1, 0, 0, 0]
        d.qvel[vadr : vadr + 6] = 0
        d.eq_active[self.shoot_weld[name]] = 0
        for g in self.stub_geoms[name]:  # a tiny stub hidden inside the main stem
            self._set_capsule(g, [0, 0, STEM_BASE + 0.05], np.eye(3), 0.001)
        m.geom_group[self.stub_geoms[name][1]] = 3

    def _set_capsule(self, g, centre, R, half):
        m = self.model
        m.geom_pos[g], m.geom_quat[g] = centre, _quat(R)
        m.geom_size[g][:2] = SHOOT_R, half
        m.geom_rbound[g] = SHOOT_R + half
        m.geom_aabb[g] = [0, 0, 0, SHOOT_R, SHOOT_R, half + SHOOT_R]

    def _shape_shoot(self, name, stub_end, outer_start):
        """The stub along the shoot from the stem's axis to `stub_end`, the outer part from `outer_start` to the tip
        (distances along the shoot's axis; the capsules' rounded caps reach SHOOT_R beyond)."""
        shoot = self.shoots[name]
        o, d, F, length = shoot["origin"], shoot["dir"], shoot["frame"], shoot["length"]
        for g in self.stub_geoms[name]:
            self._set_capsule(g, o + d * stub_end / 2, F, stub_end / 2)
        self.model.geom_group[self.stub_geoms[name][1]] = 2
        centre = (outer_start + length) / 2 - CUT_MID
        for g in self.outer_geoms[name]:
            self._set_capsule(g, [0, 0, centre], np.eye(3), (length - outer_start) / 2)
        self.model.body_ipos[self.obj_body[name]] = [0, 0, centre]

    def _set_exclusions(self):
        """The plant and an attached outer part overlap where they meet and must not collide; a cut or unused shoot
        has no exclusion (its slot points at a body pair that does not exist). MuJoCo looks pairs up by bisection,
        so the list stays sorted."""
        attached = {n for n in self.shoots if n not in self.cut}
        signatures = list(self._fixed_exclusions) + [
            self._shoot_exclusion[n] if n in attached else 0x7FFFFFFF for n in self.shoot_names
        ]
        self.model.exclude_signature[:] = np.sort(np.array(signatures, dtype=self.model.exclude_signature.dtype))

    def plate_angle(self):
        return float(self.data.qpos[self.plate_qadr])

    def button_travel(self, name):
        """How far (metres) a button is pushed down."""
        return float(-self.data.qpos[self.button_qadr[name]])

    def button_top(self, name):
        """World position of the centre of a button's cap top, at rest."""
        (bx, by), _ = BUTTONS[name]
        base = self.data.xpos[self.stand_body]
        return np.array([base[0] + bx, base[1] + by, base[2] + BUTTON_Z + BUTTON_H])

    def _drive_turntable(self):
        """The motor turns the plate at TURN_SPEED while exactly one button is held down past PRESS_TRAVEL."""
        turn = sum(sign for name, (_, sign) in BUTTONS.items() if self.button_travel(name) > PRESS_TRAVEL)
        self.data.ctrl[self.turntable] = TURN_SPEED * turn

    def stem_axis(self):
        """World position of the main stem's base and its (unit) direction."""
        R, p = self.data.xmat[self.plant_body].reshape(3, 3), self.data.xpos[self.plant_body]
        return p + R @ [0, 0, STEM_BASE], R[:, 2].copy()

    def shoot_axis(self, name):
        """World position where a shoot leaves the main stem's axis and its (unit) direction, as grown."""
        shoot = self.shoots[name]
        R, p = self.data.xmat[self.plant_body].reshape(3, 3), self.data.xpos[self.plant_body]
        return p + R @ shoot["origin"], R @ shoot["dir"]

    def shoot_azimuth(self, name):
        """World direction a shoot points to, seen from above (radians)."""
        _, d = self.shoot_axis(name)
        return float(np.arctan2(d[1], d[0]))

    def _segment_held(self, bodies, a, d, s_range):
        """The jaws hold a stem segment (axis from `a` along unit `d`): the gripper is commanded shut, the axis
        passes between the fingers within the fingertips' depth at a point s in `s_range` along it, the segment
        crosses the jaws (not along the closing axis), and a finger touches one of `bodies`."""
        d_ = self.data
        if d_.ctrl[7] >= 0.5 * 255:
            return False
        tcp = self.tcp_pos()
        R = d_.site_xmat[self.tcp_id].reshape(3, 3)
        s = float((tcp - a) @ d)
        if not s_range[0] <= s <= s_range[1]:
            return False
        rel = R.T @ (a + s * d - tcp)
        opening = float(self.state()[7]) * 0.08
        if abs(rel[1]) >= opening / 2 or abs(rel[2]) >= 0.015 or abs(d @ R[:, 1]) > np.cos(np.radians(45)):
            return False
        n = d_.ncon
        b1 = self.model.geom_bodyid[d_.contact.geom1[:n]]
        b2 = self.model.geom_bodyid[d_.contact.geom2[:n]]
        fingers = np.array(self.finger_bodies)
        return bool(
            np.any(np.isin(b1, fingers) & np.isin(b2, bodies)) or np.any(np.isin(b2, fingers) & np.isin(b1, bodies))
        )

    def holds_shoot(self, name):
        a, d = self.shoot_axis(name)
        return self._segment_held([self.obj_body[name]], a, d, CUT_ZONE)

    def holds_main_stem(self):
        a, d = self.stem_axis()
        return self._segment_held([self.plant_body], a, d, (0.0, STEM_H))

    def _check_damage(self):
        """After every physics step: the robot touching the flower head damages the plant at once; the normal force
        the robot presses on the main stem with (summed over its contacts) is recorded for _plant_result."""
        m, d = self.model, self.data
        n = d.ncon
        if n == 0:
            return
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        robot1, robot2 = np.isin(m.geom_bodyid[g1], self.robot_bodies), np.isin(m.geom_bodyid[g2], self.robot_bodies)
        other = np.where(robot1, g2, np.where(robot2, g1, -1))
        if np.any(other == self.flower_geom) and self.damaged is None:
            self.damaged = "flower touched"
        push = 0.0
        force = np.zeros(6)
        for i in np.flatnonzero(other == self.stem_geom):
            mujoco.mj_contactForce(m, d, int(i), force)
            push += force[0]
        self.max_stem_push = max(self.max_stem_push, push)
        self.step_stem_push = max(self.step_stem_push, push)

    def _cut_shoot(self, name):
        """Cut a shoot where the jaws are: the stub keeps the shoot up to the cut, the outer part the rest, and the
        outer part is let go (its weld is switched off, it collides with the plant from now on)."""
        a, d = self.shoot_axis(name)
        s = float(np.clip((self.tcp_pos() - a) @ d, *CUT_ZONE))
        end = CUT_GAP + SHOOT_R  # the capsules' caps end CUT_GAP from the cut, so the pieces never overlap
        self.shoots[name]["cut_s"] = s
        self._shape_shoot(name, s - end, s + end)
        self.data.eq_active[self.shoot_weld[name]] = 0
        self.cut.add(name)
        self._set_exclusions()
        mujoco.mj_forward(self.model, self.data)

    def _plant_result(self):
        held = [n for n in self.active if n not in self.cut and self.holds_shoot(n)]
        stem_held = self.holds_main_stem()
        # Pressing on the main stem with more than STEM_PUSH_LIMIT newtons during this control step damages it, unless
        # the stem is between the shut jaws: that is a cut of the main stem, counted below after CUT_STEPS steps.
        if self.step_stem_push > STEM_PUSH_LIMIT and not stem_held and self.damaged is None:
            self.damaged = "main stem pushed"
        self.step_stem_push = 0.0
        for n in self.active:
            self.hold_counts[n] = self.hold_counts.get(n, 0) + 1 if n in held else 0
            if self.hold_counts[n] >= CUT_STEPS and n not in self.cut:
                self._cut_shoot(n)
        self.stem_hold = self.stem_hold + 1 if stem_held else 0
        if self.stem_hold >= CUT_STEPS and self.damaged is None:
            self.damaged = "main stem cut"
        success = self.damaged is None and len(self.cut) == len(self.active)
        done = success or self.damaged is not None or self.steps >= self.max_steps
        info = dict(
            success=success,
            lifted=held + (["stem"] if stem_held else []),
            wrong_object=self.damaged is not None,
            cut=sorted(self.cut),
            damaged=self.damaged,
        )
        return self.get_obs(), done, info

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
