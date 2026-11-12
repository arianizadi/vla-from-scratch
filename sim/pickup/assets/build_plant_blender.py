"""Build the potted plant and the rotating plant stand in Blender.

Run inside Blender (through the Blender MCP, or
`blender --background --python sim/pickup/assets/build_plant_blender.py`).

Writes one OBJ per part to sim/pickup/assets/meshes/ (Z up, metres, triangulated, smooth vertex
normals) and saves plant_assets.blend with the parts assembled as a preview. Each OBJ is exported
with the object at the world origin, so the mesh frame is the MuJoCo geom frame. Colours are
assigned in MuJoCo (rgba per geom); the Blender materials are for the preview only.

Paths come from this file's location. Two environment variables override that when the code is
executed without a file (e.g. through the MCP): PLANT_ASSETS_OUT (the assets directory) and
PLANT_PREVIEW_DIR (if set, plant_preview_front.png and plant_preview_top.png are rendered there).

Parts, final dimensions and origins:

  plant_pot           terracotta pot, bottom radius 38 mm, top outer radius 50 mm, height 75 mm, with
                      an 18 mm collar and a rolled rim; closed inside at z = 64 mm (the soil covers it).
                      Origin: centre of the bottom face.
  plant_soil          lumpy soil disc, radius 44.8 mm, surface between z = 66 and 70 mm in the pot
                      frame (68.5 mm at the centre). Origin: the pot's bottom centre (same frame as
                      plant_pot).
  plant_main_stem     stem, radius 5.5 mm at the base to 4.0 mm at the top, z = 0 to 240 mm, centreline
                      waviness up to 2.5 mm (zero at both ends), small nodes at z = 32, 82 and 132 mm
                      (0.10, 0.15, 0.20 m above the pot bottom when the stem stands at z = 68 mm in
                      the pot frame). Origin: centre of the base.
  plant_flower_petals gerbera-like head, 65 mm across: 20 outer and 14 inner petals, cupped, tilted 15
  plant_flower_centre degrees toward +X. Centre disc 20 mm across. Sepals: receptacle with a 4.3 mm
  plant_flower_sepals collar that sleeves the stem top, and 12 bracts under the petals. All three
                      share one origin: the receptacle bottom, where the head meets the top of the
                      main stem (the head spans z = -3 to about +27 mm).
  plant_leaf          ovate blade 50 mm long and 24 mm wide with a midrib fold and a drooping tip, on
                      a 10 mm petiole (60 mm overall). Origin: petiole base; blade along +X, lying in
                      the XY plane, upper side +Z.
  plant_stand_base    fixed housing: round base, radius 110 mm, top at z = 30 mm, on six 3 mm rubber
                      feet; control panel 132 mm wide (Y) reaching X = -180 mm, its top sloping from
                      z = 26 mm at the front to 35 mm at the back (31 mm at the buttons); two button
                      recesses, radius 17.5 mm, flat floor at z = 26 mm, centred at X = -145 mm,
                      Y = +/-45 mm. Origin: centre of the bottom, on the rotation axis.
  plant_stand_plate   rotating plate, radius 100 mm, 12 mm thick, 40 V-grooves round the edge and a
                      ring groove on top. Origin: bottom centre, on the axis (sits at z = 30 mm).
  plant_stand_plate_marks  twelve raised radial marks for the plate top (one long index mark at +X,
                      three medium, eight short), 0.7 mm proud, as a separate mesh so they can take
                      a contrasting colour. Origin: the plate's origin.
  plant_button        push-button cap, diameter 32 mm, height 10 mm, domed. Origin: bottom centre
                      (sits on the recess floor, z = 26 mm in the base frame).
  plant_arrow_cw      raised curved arrows, 19 mm across, 1.5 mm thick, clockwise / counter-clockwise
  plant_arrow_ccw     seen from +Z. Origin: bottom centre; place 9.4 mm above the button's origin.

Preview assembly (base frame): plate z = 0.030, pot and soil z = 0.042, stem z = 0.110, flower
z = 0.350, buttons (-0.145, +/-0.045, 0.026), arrows 0.0094 above the buttons.
"""

import contextlib
import math
import os
import random

import bmesh
import bpy
from mathutils import Matrix, Vector, noise

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # executed without a file
    _HERE = os.getcwd()
OUT = os.environ.get("PLANT_ASSETS_OUT") or _HERE
PREVIEW_DIR = os.environ.get("PLANT_PREVIEW_DIR")
os.makedirs(os.path.join(OUT, "meshes"), exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
for blocks in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
    for block in list(blocks):
        blocks.remove(block)

rng = random.Random(7)
TAU = 2 * math.pi


# ---------------------------------------------------------------- helpers
def material(name, rgba, rough=0.5, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    return m


class Geo:
    """Vertex and face lists that several shells can be appended to."""

    def __init__(self):
        self.verts, self.faces = [], []

    def add(self, verts, faces, matrix=None):
        n = len(self.verts)
        for v in verts:
            v = Vector(v)
            self.verts.append(matrix @ v if matrix else v)
        self.faces += [tuple(i + n for i in f) for f in faces]
        return self


def lathe(profile, segments, radial=None):
    """Surface of revolution about Z. profile: (r, z) or (r, z, tag); r = 0 makes a pole vertex."""
    verts, rings = [], []
    for p in profile:
        r, z, tag = p[0], p[1], (p[2] if len(p) > 2 else 0)
        if r < 1e-9:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
            continue
        ring = []
        for i in range(segments):
            a = TAU * i / segments
            rr = radial(i, r, tag) if radial else r
            ring.append(len(verts))
            verts.append((rr * math.cos(a), rr * math.sin(a), z))
        rings.append(ring)
    faces = []
    for a, b in zip(rings, rings[1:], strict=False):
        for i in range(segments):
            j = (i + 1) % segments
            if len(a) == 1 and len(b) > 1:
                faces.append((a[0], b[i], b[j]))
            elif len(b) == 1 and len(a) > 1:
                faces.append((a[i], a[j], b[0]))
            elif len(a) > 1:
                faces.append((a[i], a[j], b[j], b[i]))
    return verts, faces


def sheet(grid, thickness):
    """Thin closed solid from a rows x cols grid of points (the upper surface)."""
    rows, cols = len(grid), len(grid[0])
    P = [[Vector(p) for p in row] for row in grid]

    def normal(i, j):
        du = P[min(i + 1, rows - 1)][j] - P[max(i - 1, 0)][j]
        dv = P[i][min(j + 1, cols - 1)] - P[i][max(j - 1, 0)]
        n = du.cross(dv)
        return n.normalized() if n.length > 1e-12 else Vector((0, 0, 1))

    top = [P[i][j] for i in range(rows) for j in range(cols)]
    bottom = [P[i][j] - normal(i, j) * thickness for i in range(rows) for j in range(cols)]
    nt = len(top)
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a, b, c, d = i * cols + j, (i + 1) * cols + j, (i + 1) * cols + j + 1, i * cols + j + 1
            faces.append((a, b, c, d))
            faces.append((nt + d, nt + c, nt + b, nt + a))
    loop = (
        [j for j in range(cols)]
        + [i * cols + cols - 1 for i in range(1, rows)]
        + [(rows - 1) * cols + j for j in range(cols - 2, -1, -1)]
        + [i * cols for i in range(rows - 2, 0, -1)]
    )
    for a, b in zip(loop, loop[1:] + loop[:1], strict=False):
        faces.append((a, b, nt + b, nt + a))
    return top + bottom, faces


def new_object(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def finish(obj, mat, sharp_deg=40.0):
    """Consistent outward normals, triangles, smooth shading with sharp edges above sharp_deg."""
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.set_sharp_from_angle(angle=math.radians(sharp_deg))
    me.update()
    me.materials.clear()
    me.materials.append(mat)
    return obj


def build(name, geo, mat, sharp_deg=40.0):
    return finish(new_object(name, geo.verts, geo.faces), mat, sharp_deg)


def activate(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


M = dict(
    terracotta=material("terracotta", (0.62, 0.27, 0.14, 1), 0.85),
    soil=material("soil", (0.085, 0.055, 0.035, 1), 0.95),
    stem=material("stem", (0.16, 0.36, 0.10, 1), 0.6),
    leaf=material("leaf", (0.10, 0.40, 0.09, 1), 0.5),
    petal=material("petal", (0.95, 0.36, 0.03, 1), 0.55),
    centre=material("flower_centre", (0.30, 0.16, 0.03, 1), 0.8),
    housing=material("stand_housing", (0.10, 0.10, 0.11, 1), 0.45),
    plate=material("stand_plate", (0.23, 0.23, 0.25, 1), 0.35, 0.3),
    white=material("white", (0.92, 0.92, 0.92, 1), 0.5),
    button_blue=material("button_blue", (0.05, 0.25, 0.85, 1), 0.35),
    button_orange=material("button_orange", (0.95, 0.42, 0.03, 1), 0.35),
    table=material("table", (0.62, 0.50, 0.36, 1), 0.7),
)

# ---------------------------------------------------------------- pot and soil
POT_PROFILE = [
    (0.0, 0.0),
    (0.0365, 0.0),
    (0.0380, 0.0015),
    (0.0403, 0.020),
    (0.0428, 0.040),
    (0.0450, 0.057),  # tapered wall
    (0.0486, 0.0574),
    (0.0500, 0.0590),  # underside of the collar
    (0.0500, 0.0715),
    (0.0495, 0.0735),
    (0.0481, 0.0748),  # collar and rolled rim
    (0.0465, 0.0750),
    (0.0450, 0.0744),
    (0.0442, 0.0730),
    (0.0440, 0.0700),
    (0.0440, 0.0640),
    (0.0, 0.0640),
]
pot = build("plant_pot", Geo().add(*lathe(POT_PROFILE, 48)), M["terracotta"])

SOIL_SEG = 32
soil_verts, soil_rings = [(0.0, 0.0, 0.0685)], []
for r in (0.006, 0.012, 0.019, 0.026, 0.033, 0.039, 0.0432, 0.0448):
    ring = []
    for i in range(SOIL_SEG):
        a = TAU * (i + 0.5 * (len(soil_rings) % 2)) / SOIL_SEG
        x, y = r * math.cos(a), r * math.sin(a)
        lump = noise.noise(Vector((x * 85, y * 85, 1.7))) + 0.5 * noise.noise(Vector((x * 210, y * 210, 4.1)))
        z = 0.0682 + 0.0019 * lump * min(1.0, r / 0.012)
        if r > 0.040:
            z = min(z, 0.0672) if r < 0.044 else 0.0660
        ring.append(len(soil_verts))
        soil_verts.append((x, y, max(0.0660, min(0.0700, z))))
    soil_rings.append(ring)
ring = [len(soil_verts) + i for i in range(SOIL_SEG)]  # skirt, hidden inside the pot wall
soil_verts += [(soil_verts[i][0], soil_verts[i][1], 0.0630) for i in soil_rings[-1]]
soil_rings.append(ring)
soil_faces = [(0, soil_rings[0][i], soil_rings[0][(i + 1) % SOIL_SEG]) for i in range(SOIL_SEG)]
for a, b in zip(soil_rings, soil_rings[1:], strict=False):
    soil_faces += [(a[i], a[(i + 1) % SOIL_SEG], b[(i + 1) % SOIL_SEG], b[i]) for i in range(SOIL_SEG)]
soil_faces.append(tuple(reversed(soil_rings[-1])))
soil = build("plant_soil", Geo().add(soil_verts, soil_faces), M["soil"], sharp_deg=50)

# ---------------------------------------------------------------- main stem
STEM_H, STEM_SIDES = 0.240, 12
STEM_NODES = (0.032, 0.082, 0.132)


def stem_centre(z):
    t = z / STEM_H
    return Vector((0.0025 * math.sin(TAU * t), 0.0018 * math.sin(3 * math.pi * t), z))


def stem_radius(z):
    r = 0.0055 + (0.0040 - 0.0055) * z / STEM_H
    return r + sum(0.0007 * math.exp(-(((z - n) / 0.0035) ** 2)) for n in STEM_NODES)


stem_z = sorted(
    {round(STEM_H * i / 24, 5) for i in range(25)} | {round(n + d, 5) for n in STEM_NODES for d in (-0.004, 0.0, 0.004)}
)
stem_geo, prev = Geo(), None
stem_verts, stem_faces = [], []
for _k, z in enumerate(stem_z):
    c, r = stem_centre(z), stem_radius(z)
    ring = []
    for i in range(STEM_SIDES):
        a = TAU * i / STEM_SIDES + 0.35 * z / STEM_H  # slight twist
        ring.append(len(stem_verts))
        stem_verts.append((c.x + r * math.cos(a), c.y + r * math.sin(a), z))
    if prev:
        stem_faces += [
            (prev[i], prev[(i + 1) % STEM_SIDES], ring[(i + 1) % STEM_SIDES], ring[i]) for i in range(STEM_SIDES)
        ]
    else:
        stem_faces.append(tuple(reversed(ring)))
    prev = ring
stem_faces.append(tuple(prev))
stem = build("plant_main_stem", Geo().add(stem_verts, stem_faces), M["stem"], sharp_deg=50)

# ---------------------------------------------------------------- flower head
TILT = Matrix.Rotation(math.radians(15), 4, "Y")  # +Z leans toward +X
PETAL_T = (0.0, 0.22, 0.50, 0.78, 0.94, 1.0)
PETAL_W = (0.45, 0.82, 1.00, 0.90, 0.60, 0.22)


def petal(angle, r0, z0, tip_r, width, elev, droop, cup, twist):
    u = Vector((math.cos(angle), math.sin(angle), 0))
    v = Vector((-math.sin(angle), math.cos(angle), 0))
    length = (tip_r - r0) / math.cos(elev)
    grid = []
    for t, wf in zip(PETAL_T, PETAL_W, strict=False):
        d = length * t
        w = 0.5 * width * wf
        r = r0 + d * math.cos(elev)
        z = z0 + d * math.sin(elev) - droop * length * t * t
        grid.append([u * r + v * (s * w) + Vector((0, 0, z + cup * w * s * s + twist * s * w * t)) for s in (-1, 0, 1)])
    return sheet(grid, 0.0004)


petal_geo = Geo()
for i in range(20):  # outer ring
    petal_geo.add(
        *petal(
            TAU * (i + rng.uniform(-0.12, 0.12)) / 20,
            0.006,
            0.0098 + 0.0004 * (i % 2),
            0.0325 * rng.uniform(0.94, 1.0),
            0.0088 * rng.uniform(0.9, 1.08),
            math.radians(12 + rng.uniform(-3, 3) + 3 * (i % 2)),
            rng.uniform(0.06, 0.16),
            0.35,
            rng.uniform(-0.25, 0.25),
        ),
        TILT,
    )
for i in range(14):  # inner ring, shorter and more upright
    petal_geo.add(
        *petal(
            TAU * (i + 0.5 + rng.uniform(-0.12, 0.12)) / 14,
            0.006,
            0.0108,
            0.0235 * rng.uniform(0.92, 1.0),
            0.0076 * rng.uniform(0.9, 1.08),
            math.radians(30 + rng.uniform(-4, 4)),
            rng.uniform(0.02, 0.10),
            0.45,
            rng.uniform(-0.25, 0.25),
        ),
        TILT,
    )
petals = build("plant_flower_petals", petal_geo, M["petal"], sharp_deg=45)

CENTRE_PROFILE = [
    (0.0, 0.0090),
    (0.0098, 0.0090),
    (0.0100, 0.0118),
    (0.0090, 0.0134),
    (0.0066, 0.0144),
    (0.0035, 0.0141),
    (0.0, 0.0134),
]
centre = build("plant_flower_centre", Geo().add(*lathe(CENTRE_PROFILE, 20), TILT), M["centre"], sharp_deg=50)

SEPAL_PROFILE = [
    (0.0, -0.0030),
    (0.0043, -0.0030),
    (0.0046, 0.0),
    (0.0066, 0.0030),
    (0.0104, 0.0064),
    (0.0120, 0.0090),
    (0.0, 0.0095),
]
sepal_geo = Geo().add(*lathe(SEPAL_PROFILE, 16), TILT)
for i in range(12):
    a = TAU * (i + 0.5) / 12
    u, v = Vector((math.cos(a), math.sin(a), 0)), Vector((-math.sin(a), math.cos(a), 0))
    grid = [
        [u * r + v * (s * w) + Vector((0, 0, z)) for s in (-1, 0, 1)]
        for r, z, w in ((0.0085, 0.0052, 0.0030), (0.0135, 0.0082, 0.0025), (0.0178, 0.0096, 0.0004))
    ]
    sepal_geo.add(*sheet(grid, 0.0005), TILT)
sepals = build("plant_flower_sepals", sepal_geo, M["stem"], sharp_deg=45)

# ---------------------------------------------------------------- leaf
LEAF_X0, LEAF_L, LEAF_HALF_W = 0.010, 0.050, 0.012
leaf_grid = []
for t in (0.0, 0.04, 0.10, 0.20, 0.32, 0.45, 0.58, 0.70, 0.80, 0.89, 0.96, 1.0):
    w = max(0.0005, LEAF_HALF_W * math.sin(math.pi * t**0.7) ** 0.8)
    arch = 0.006 * math.sin(0.9 * math.pi * t) - 0.006 * t * t  # rises, then the tip droops
    row = []
    for s in (-1.0, -0.6, -0.25, 0.0, 0.25, 0.6, 1.0):
        y = s * w
        z = arch + 0.30 * abs(y) - 0.10 * w * s**4 + 0.0005 * math.sin(16 * t + 2.0 * s) * abs(s)
        row.append((LEAF_X0 + LEAF_L * t, y, z))
    leaf_grid.append(row)
leaf_geo = Geo().add(*sheet(leaf_grid, 0.0005))
pet_verts, pet_faces, prev = [], [], None
for x, r, z in ((0.0, 0.0012, 0.0), (0.006, 0.0010, 0.0002), (0.0125, 0.0008, 0.0002)):  # petiole
    ring = [len(pet_verts) + i for i in range(6)]
    pet_verts += [(x, r * math.cos(TAU * i / 6), z + r * math.sin(TAU * i / 6) - 0.0003) for i in range(6)]
    if prev:
        pet_faces += [(prev[i], prev[(i + 1) % 6], ring[(i + 1) % 6], ring[i]) for i in range(6)]
    else:
        pet_faces.append(tuple(reversed(ring)))
    prev = ring
pet_faces.append(tuple(prev))
leaf_geo.add(pet_verts, pet_faces)
leaf = build("plant_leaf", leaf_geo, M["leaf"], sharp_deg=45)

# ---------------------------------------------------------------- stand: fixed housing
BUTTON_X, BUTTON_Y, RECESS_Z, RECESS_R = -0.145, 0.045, 0.026, 0.0175
PANEL_HALF_W, PANEL_FRONT_X = 0.066, -0.180

BASE_PROFILE = [
    (0.0, 0.003),
    (0.105, 0.003),
    (0.1085, 0.0045),
    (0.110, 0.0075),
    (0.110, 0.0125),
    (0.1088, 0.0133),
    (0.1088, 0.0152),
    (0.110, 0.016),  # accent groove
    (0.110, 0.0268),
    (0.1092, 0.0288),
    (0.1072, 0.030),
    (0.0, 0.030),
]
base_geo = Geo().add(*lathe(BASE_PROFILE, 72))
FOOT_PROFILE = [(0.0, 0.0), (0.0085, 0.0), (0.0095, 0.0008), (0.0095, 0.0035), (0.0, 0.0035)]
for fx, fy in [
    (0.085 * math.cos(a), 0.085 * math.sin(a)) for a in (0.25 * math.pi, 0.75 * math.pi, 1.25 * math.pi, 1.75 * math.pi)
] + [(-0.165, 0.05), (-0.165, -0.05)]:
    base_geo.add(*lathe(FOOT_PROFILE, 14), Matrix.Translation((fx, fy, 0)))
stand_base = new_object("plant_stand_base", base_geo.verts, base_geo.faces)

# Control panel: an X-Z profile extruded along Y, front corners rounded, edges eased, recesses cut.
PANEL_PROFILE = [
    (PANEL_FRONT_X, 0.003),
    (PANEL_FRONT_X, 0.026),
    (-0.116, 0.035),
    (-0.1105, 0.035),
    (-0.1085, 0.0296),
    (-0.060, 0.0296),
    (-0.060, 0.003),
]
n = len(PANEL_PROFILE)
panel_verts = [(x, -PANEL_HALF_W, z) for x, z in PANEL_PROFILE] + [(x, PANEL_HALF_W, z) for x, z in PANEL_PROFILE]
panel_faces = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
panel_faces += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
panel = new_object("plant_stand_panel", panel_verts, panel_faces)
bm = bmesh.new()
bm.from_mesh(panel.data)
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
corners = [
    e
    for e in bm.edges
    if all(abs(v.co.x - PANEL_FRONT_X) < 1e-6 for v in e.verts) and abs(e.verts[0].co.y - e.verts[1].co.y) < 1e-6
]
bmesh.ops.bevel(bm, geom=corners, offset=0.014, offset_type="OFFSET", segments=6, profile=0.5, affect="EDGES")
bm.to_mesh(panel.data)
bm.free()
activate(panel)
mod = panel.modifiers.new("bevel", "BEVEL")
mod.width, mod.segments, mod.limit_method, mod.angle_limit = 0.002, 2, "ANGLE", math.radians(30)
bpy.ops.object.modifier_apply(modifier="bevel")
for side in (1, -1):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=40, radius=RECESS_R, depth=0.030, location=(BUTTON_X, side * BUTTON_Y, RECESS_Z + 0.015)
    )
    cutter = bpy.context.object
    mod = panel.modifiers.new("cut", "BOOLEAN")
    mod.object, mod.operation = cutter, "DIFFERENCE"
    activate(panel)
    bpy.ops.object.modifier_apply(modifier="cut")
    bpy.data.objects.remove(cutter)
activate(stand_base)
panel.select_set(True)
bpy.ops.object.join()
finish(stand_base, M["housing"], sharp_deg=32)

# ---------------------------------------------------------------- stand: rotating plate
PLATE_PROFILE = [
    (0.0, 0.0),
    (0.0955, 0.0),
    (0.0985, 0.0012, 1),
    (0.100, 0.0026, 1),
    (0.100, 0.0094, 1),
    (0.0985, 0.0108, 1),
    (0.0968, 0.012),
    (0.0940, 0.012),
    (0.0934, 0.0113),
    (0.0926, 0.0113),
    (0.0920, 0.012),
    (0.0, 0.012),
]
# 120 segments, every third vertex pulled in: 40 V-grooves round the edge.
plate = build(
    "plant_stand_plate",
    Geo().add(*lathe(PLATE_PROFILE, 120, radial=lambda i, r, tag: r - (0.0016 if tag and i % 3 == 0 else 0.0))),
    M["plate"],
    sharp_deg=14,
)

marks_geo = Geo()
for k in range(12):
    r1, half_w = (0.058, 0.0032) if k == 0 else (0.071, 0.0016) if k % 3 == 0 else (0.081, 0.0011)
    r2, z1, z2 = 0.0905, 0.0116, 0.0127
    box = [(x, y, z) for z in (z1, z2) for x, y in ((r1, -half_w), (r2, -half_w), (r2, half_w), (r1, half_w))]
    quads = [(0, 1, 2, 3), (7, 6, 5, 4)] + [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)]
    marks_geo.add(box, quads, Matrix.Rotation(TAU * k / 12, 4, "Z"))
marks = build("plant_stand_plate_marks", marks_geo, M["white"], sharp_deg=30)

# ---------------------------------------------------------------- button and arrows
BUTTON_PROFILE = [
    (0.0, 0.0),
    (0.0155, 0.0),
    (0.016, 0.0005),
    (0.016, 0.0066),
    (0.0156, 0.0078),
    (0.0146, 0.0086),
    (0.0125, 0.0092),
    (0.0090, 0.0096),
    (0.0050, 0.0099),
    (0.0, 0.010),
]
button = build("plant_button", Geo().add(*lathe(BUTTON_PROFILE, 32)), M["button_blue"])
ARROW_Z_ON_BUTTON = 0.0094


def arrow(ccw):
    r_mid, half_w, thick, steps = 0.0070, 0.0015, 0.0015, 20
    a0, a1 = math.radians(55), math.radians(285)
    verts, faces = [], []
    for k in range(steps + 1):
        a = a0 + (a1 - a0) * k / steps
        for r in (r_mid - half_w, r_mid + half_w):
            for z in (0.0, thick):
                verts.append((r * math.cos(a), r * math.sin(a), z))
        if k:
            p, q = 4 * (k - 1), 4 * k  # per station: inner-bottom, inner-top, outer-bottom, outer-top
            faces += [
                (p + 1, p + 3, q + 3, q + 1),
                (p, q, q + 2, p + 2),
                (p, p + 1, q + 1, q),
                (p + 2, q + 2, q + 3, p + 3),
            ]
    faces += [(0, 2, 3, 1), (4 * steps, 4 * steps + 1, 4 * steps + 3, 4 * steps + 2)]
    # arrowhead: a triangular prism at the leading end, pointing along the tangent
    c = Vector((r_mid * math.cos(a1), r_mid * math.sin(a1), 0))
    radial, tangent = Vector((math.cos(a1), math.sin(a1), 0)), Vector((-math.sin(a1), math.cos(a1), 0))
    tri = [c - radial * 0.0036 - tangent * 0.0003, c + radial * 0.0036 - tangent * 0.0003, c + tangent * 0.0062]
    m = len(verts)
    verts += [tuple(p) for p in tri] + [(p.x, p.y, thick) for p in tri]
    faces += [(m, m + 2, m + 1), (m + 3, m + 4, m + 5)] + [
        (m + i, m + (i + 1) % 3, m + 3 + (i + 1) % 3, m + 3 + i) for i in range(3)
    ]
    if not ccw:
        verts = [(x, -y, z) for x, y, z in verts]
    return verts, faces


arrow_cw = build("plant_arrow_cw", Geo().add(*arrow(False)), M["white"], sharp_deg=30)
arrow_ccw = build("plant_arrow_ccw", Geo().add(*arrow(True)), M["white"], sharp_deg=30)

# ---------------------------------------------------------------- export
PARTS = [
    pot,
    soil,
    stem,
    petals,
    centre,
    sepals,
    leaf,
    stand_base,
    plate,
    marks,
    button,
    arrow_cw,
    arrow_ccw,
]
TRIANGLES = {}
for obj in PARTS:
    obj.location = (0, 0, 0)
    bpy.context.view_layer.update()
    activate(obj)
    bpy.ops.wm.obj_export(
        filepath=os.path.join(OUT, "meshes", f"{obj.name}.obj"),
        export_selected_objects=True,
        forward_axis="Y",
        up_axis="Z",
        export_materials=False,
        export_normals=True,
        export_uv=False,
        export_triangulated_mesh=True,
    )
    TRIANGLES[obj.name] = len(obj.data.polygons)
    print(f"{obj.name}: {TRIANGLES[obj.name]} triangles, size {tuple(round(d, 4) for d in obj.dimensions)}")

# ---------------------------------------------------------------- preview assembly
PLATE_Z, POT_Z = 0.030, 0.042
STEM_Z = POT_Z + 0.068
FLOWER_Z = STEM_Z + STEM_H


def instance(obj, name, mat=None):
    dup = obj.copy()
    dup.name = name
    bpy.context.scene.collection.objects.link(dup)
    if mat:  # material on the object, so the shared mesh keeps its own
        dup.material_slots[0].link = "OBJECT"
        dup.material_slots[0].material = mat
    return dup


plate.location = marks.location = (0, 0, PLATE_Z)
pot.location = soil.location = (0, 0, POT_Z)
stem.location = (0, 0, STEM_Z)
for part in (petals, centre, sepals):
    part.location = (0, 0, FLOWER_Z)
button.location = (BUTTON_X, BUTTON_Y, RECESS_Z)
button_2 = instance(button, "plant_button_orange", M["button_orange"])
button_2.location = (BUTTON_X, -BUTTON_Y, RECESS_Z)
arrow_ccw.location = (BUTTON_X, BUTTON_Y, RECESS_Z + ARROW_Z_ON_BUTTON)
arrow_cw.location = (BUTTON_X, -BUTTON_Y, RECESS_Z + ARROW_Z_ON_BUTTON)

# Three side shoots (plain cylinders; in the sim these are the parts that get cut) with leaves.
SHOOT_LEN, SHOOT_R = 0.080, 0.0022
leaf.hide_render = leaf.hide_viewport = True  # the origin copy; instances are placed below
for k, (height, azimuth, yaws) in enumerate(((0.10, 200, (-50, 0, 50)), (0.15, 75, (-35, 35)), (0.20, 315, (0,)))):
    az, el = math.radians(azimuth), math.radians(35)
    d = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
    start = stem_centre(POT_Z + height - STEM_Z) + Vector((0, 0, STEM_Z))
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=10, radius=SHOOT_R, depth=SHOOT_LEN, location=start + d * (SHOOT_LEN / 2)
    )
    shoot = bpy.context.object
    shoot.name = f"preview_shoot_{k + 1}"
    shoot.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    shoot.data.materials.append(M["stem"])
    bpy.ops.object.shade_smooth()
    y_axis = Vector((0, 0, 1)).cross(d).normalized()
    frame = Matrix((d, y_axis, d.cross(y_axis))).transposed().to_4x4()
    tip = start + d * (SHOOT_LEN - 0.001)
    for j, yaw in enumerate(yaws):
        lf = instance(leaf, f"preview_leaf_{k + 1}_{j + 1}")
        lf.hide_render = lf.hide_viewport = False
        lf.matrix_world = (
            Matrix.Translation(tip)
            @ frame
            @ Matrix.Rotation(math.radians(yaw), 4, "Z")
            @ Matrix.Rotation(math.radians(25 if yaw else 12), 4, "Y")
            @ Matrix.Rotation(math.radians(rng.uniform(-15, 15)), 4, "X")
        )

bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
bpy.context.object.name = "preview_table"
bpy.context.object.data.materials.append(M["table"])

scene = bpy.context.scene
bpy.ops.object.light_add(type="SUN", location=(-0.5, -0.8, 1.5))
sun = bpy.context.object
sun.name = "preview_sun"
sun.data.energy, sun.data.angle = 2.2, math.radians(8)
sun.rotation_euler = (math.radians(38), math.radians(-12), math.radians(-28))
bpy.ops.object.light_add(type="AREA", location=(-0.7, 0.5, 0.7))
fill = bpy.context.object
fill.name = "preview_fill"
fill.data.energy, fill.data.size = 6, 0.8
fill.rotation_euler = (Vector((0, 0, 0.15)) - fill.location).to_track_quat("-Z", "Y").to_euler()
world = scene.world or bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
background = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
background.inputs[0].default_value = (0.75, 0.80, 0.88, 1)
background.inputs[1].default_value = 0.35


def camera(name, location, target, lens=50.0, ortho=None):
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.object
    cam.name = name
    cam.rotation_euler = (Vector(target) - Vector(location)).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens, cam.data.clip_start = lens, 0.01
    if ortho:
        cam.data.type, cam.data.ortho_scale = "ORTHO", ortho
    return cam


cam_front = camera("preview_cam_front", (-0.50, -0.50, 0.44), (-0.02, 0.0, 0.185))  # front-left, 45 degrees
cam_top = camera("preview_cam_top", (-0.035, 0.0, 1.2), (-0.035, 0.0, 0.0), ortho=0.40)
scene.camera = cam_front
scene.render.resolution_x = scene.render.resolution_y = 1000
with contextlib.suppress(TypeError):
    scene.view_settings.view_transform = "Standard"

if PREVIEW_DIR:
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    with contextlib.suppress(TypeError):
        scene.render.image_settings.file_format = "PNG"
    for cam, filename in ((cam_front, "plant_preview_front.png"), (cam_top, "plant_preview_top.png")):
        scene.camera = cam
        scene.render.filepath = os.path.join(PREVIEW_DIR, filename)
        bpy.ops.render.render(write_still=True)
    scene.camera = cam_front

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "plant_assets.blend"))
