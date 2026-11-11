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
TASKS = {
    "pick_cube": dict(pool=["cube"], num_objects=1),
    "pick_target": dict(pool=["cube", "can", "hex", "tblock"], num_objects=3),
}

HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
CONTROL_HZ = 20
TIMESTEP = 0.002
SUBSTEPS = int(round(1.0 / (CONTROL_HZ * TIMESTEP)))
SPAWN_X = (0.38, 0.62)
SPAWN_Y = (-0.22, 0.22)
LIFT_HEIGHT = 0.08  # metres above rest height that counts as "lifted"
HOLD_STEPS = 10  # consecutive lifted control steps required for success


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


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.option.timestep = TIMESTEP
    spec.option.impratio = 10
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
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

    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.004, 0.004, 0.004], rgba=[0, 0, 0, 0])
    hand.add_camera(
        name="wrist", pos=[0.075, 0, 0.035], quat=_look_at_quat([0.075, 0, 0.035], [0, 0, 0.17], up=(1, 0, 0)), fovy=80
    )

    geom_types = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}
    for i, (name, obj) in enumerate(OBJECTS.items()):
        body = wb.add_body(name=name, pos=[2.0 + 0.3 * i, 2.0, -0.76 + obj["rest_z"]])
        body.add_freejoint()
        for j, (gtype, size, z) in enumerate(obj["collision"]):
            body.add_geom(
                name=f"{name}_col{j}",
                type=geom_types[gtype],
                size=list(size) + [0] * (3 - len(size)),
                pos=[0, 0, z],
                group=3,
                density=600,
                friction=[1.5, 0.01, 0.0002],
                condim=4,
                solref=[0.004, 1],
                solimp=[0.95, 0.99, 0.001, 0.5, 2],
            )
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
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, image_size, image_size)
        self.tcp_id = self.model.site("tcp").id
        self.ctrl_lo = self.model.actuator_ctrlrange[:7, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:7, 1].copy()
        self.obj_qadr = {n: self.model.joint(self.model.body(n).jntadr[0]).qposadr[0] for n in OBJECTS}
        self.obj_vadr = {n: self.model.joint(self.model.body(n).jntadr[0]).dofadr[0] for n in OBJECTS}
        self.active: list[str] = []
        self.target: str = ""
        self.instruction: str = ""
        self.rest_z: dict[str, float] = {}

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int):
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
        self.target = str(rng.choice(self.active))
        self.instruction = f"pick up the {OBJECTS[self.target]['label']}"

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

        lifted = self.lifted_objects()
        target_up = self.target in lifted
        self.lift_count = self.lift_count + 1 if target_up else 0
        success = self.lift_count >= HOLD_STEPS
        done = success or self.steps >= self.max_steps
        info = dict(success=success, lifted=lifted, wrong_object=bool(lifted) and not target_up)
        return self.get_obs(), done, info

    # ---------------------------------------------------------------- queries
    def lifted_objects(self):
        return [n for n in self.active if self.object_pos(n)[2] - self.rest_z[n] > LIFT_HEIGHT]

    def object_pos(self, name):
        adr = self.obj_qadr[name]
        return self.data.qpos[adr : adr + 3].copy()

    def object_yaw(self, name):
        adr = self.obj_qadr[name]
        w, x, y, z = self.data.qpos[adr + 3 : adr + 7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

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
