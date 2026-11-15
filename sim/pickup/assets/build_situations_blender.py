"""Build the assets of the test situations in Blender: ties, sick leaves, a bud, distractors, a bench, a backdrop.

Run inside Blender (through the Blender MCP, or
`blender --background --python sim/pickup/assets/build_situations_blender.py`).

Writes one OBJ per part to sim/pickup/assets/meshes/situations/ (Z up, metres, triangulated, smooth vertex
normals; no MTL), three PNG textures and manifest.json (triangle counts, bounds, suggested colours, collision
primitives and attachment frames, for the code that puts the parts into the simulation). Each OBJ is exported with
the object at the world origin, so the mesh frame is the MuJoCo geom frame. Colours are assigned in MuJoCo (rgba per
geom): a part that needs two colours is two meshes in one frame. Only sit_bench_top and sit_backdrop_glass carry
texture coordinates; their textures are optional (they also look right with a plain rgba).

Paths come from this file's location. Two environment variables override that when the code is executed without a
file (e.g. through the MCP): SITUATIONS_ASSETS_OUT (the assets directory) and SITUATIONS_PREVIEW_DIR (if set, preview
images are rendered there; they show the new parts on the existing plant, read back from meshes/*.obj).

The parts fit the plant of build_plant_blender.py as sim/pickup/env.py places it: side shoots of radius 3 mm
(SHOOT_R) whose frame has z along the shoot and x upward, leaves scaled by 0.8 (LEAF_SCALE), the main stem's top
sleeved by a 4.3 mm collar, the soil disc of radius 44.8 mm at z = 64 to 70 mm in the pot.

Tier 1. Parts, final dimensions and origins:

  sit_tie             gardener's tie: a band 12 mm long, inner radius 2.9 mm (inside the shoot's surface), outer
                      5.6 mm, a knot and two twisted ribbon tails (8 to 11 mm wide, 29 mm long). Overall 39 x 11 x
                      49 mm (x = -5.6 to +33.5 mm). Origin: on the shoot's axis, at the middle of the band; the
                      shoot runs along Z, the knot and tails point to +X (upward in the shoot's frame). One mesh:
                      red, blue, yellow or white by rgba. Visual only.
  sit_leaf_wilted     the plant's leaf, wilted: the same 24 mm x 50 mm blade (narrowed to 20 mm by its curl) on
                      the same petiole, hanging down in a curve, edges drooping, twisted. Spans x = 0 to 36 mm
                      and z = -35 to +1 mm. Origin and axes as plant_leaf: petiole base, blade toward +X, upper
                      side +Z.
  sit_leaf_spots      brown blotches for plant_leaf (the healthy shape): six patches, 5 to 11 mm across, that
                      wrap the blade (0.35 mm proud of both faces). Same frame as plant_leaf; use the same scale.
  sit_leaf_edges      brown margins and tip for plant_leaf, wrapping the blade the same way. Same frame.
  sit_bud_body        closed bud, 14.6 mm across (16 mm with the sepals); with its calyx it spans z = -4 to
  sit_bud_sepals      +29 mm. Origin: on the shoot's axis at the shoot's tip, the bud along +Z. The calyx (in
                      sit_bud_sepals, with five sepals that hug the bud) has a collar of radius 3.7 mm reaching
                      4 mm back over the shoot.
  sit_watering_can    small can: body 88 mm across at the bottom and 90 mm tall, a strap handle toward -X, a
  sit_watering_can_trim  spout toward +X; the trim is the rose on the spout and a band round the top. Overall
                      253 x 88 x 131 mm (x = -96 to +157 mm). Origin: centre of the body's bottom.
  sit_gloves          a pair of gardening gloves lying flat, one on the other, fingers toward +X; the cuffs are
  sit_gloves_cuffs    a second mesh. Overall 254 x 159 x 27 mm (x = -114 to +141 mm). Origin: on the table
                      under the lower palm.
  sit_trowel_blade    hand trowel lying flat along X: steel blade (150 x 74 mm, scooped) with its neck, and a
  sit_trowel_handle   wooden handle 108 mm long, 29 mm across. Overall 303 x 74 x 29 mm. Origin: on the table,
                      under the neck (x = -148 to +155 mm).
  sit_seed_packet     paper packet lying flat, 82 x 120 x 4 mm; the art (a header band and a flower, 0.6 mm
  sit_seed_packet_art proud) is a second mesh. Origin: centre of the underside.
  sit_pot_small       empty terracotta pot, hollow: bottom radius 27 mm, top 37 mm, height 58 mm. Origin: centre of
                      the bottom.
  sit_bee_body        bee, 15.6 mm long, 7 mm across (15.5 mm with the wings), 8 mm tall: yellow abdomen and
  sit_bee_dark        thorax; head, stripes and tail; two wings. Origin: on the surface it sits on, under the
  sit_bee_wings       thorax; head toward +X. Visual only.
  sit_bench_top       potting-bench top of eight planks, 0.8 x 1.2 x 0.04 m, the planks along Y: the size and
                      origin (centre) of table_top, so it replaces that mesh in place. Texture coordinates:
                      u = (y + 0.6) / 1.2, v = (x + 0.4) / 0.8, for sit_bench_wood.png or sit_bench_stained.png
                      (1024 x 512, the same planks soil-stained). The planks are 0.5 mm apart.
  sit_backdrop_frame  greenhouse wall, 2.4 m wide (X) and 1.6 m tall (Z), seen from -Y: white frame (five posts,
  sit_backdrop_glass  a kick board, two rails) and one glass sheet behind it. Origin: bottom centre, the glass's
                      front face at y = 0. Texture coordinates of the glass: u = (x + 1.2) / 2.4, v = z / 1.6,
                      for sit_backdrop_greenhouse.png (768 x 512: sky over blurred foliage).

Tier 2:

  sit_symbol_leaf     raised symbols, 1.5 mm thick, like plant_arrow_cw / _ccw: a leaf 16 mm across, a sun 19 mm
  sit_symbol_sun      across. Origin: bottom centre; on a button, place 9.4 mm above the button's origin (in place
                      of the arrow).
  sit_legend_card     table card on a prop, leaning back 25 degrees, face 110 x 75 mm toward +X; the marks are two
  sit_legend_card_marks  "=" signs. Overall 53 x 110 x 70 mm. Origin: on the table, centre of the footprint. The
                      manifest gives the frames of four slots on the face (symbol and arrow, two rows).
  sit_rose_petals     flower heads with the origin and collar of plant_flower_*: the receptacle bottom, on top of
  sit_rose_sepals     the main stem, tilted 15 degrees toward +X. Rose 65 mm across, z = -8 (sepal tips) to
  sit_daisy_petals    +32 mm; daisy 61 mm across, z = -4 to +19 mm, with a yellow centre; tulip 35 mm across,
                      z = -4 to +48 mm.
  sit_daisy_centre
  sit_daisy_sepals
  sit_tulip_petals
  sit_tulip_sepals
  sit_pot_square      square pot with rounded corners, 96 mm across at the top and 75 mm tall, closed inside at
  sit_soil_square     z = 64 mm, with its own soil (surface z = 66 to 70 mm). Origin: bottom centre (the frame of
                      plant_pot and plant_soil).
  sit_pot_glazed      round glazed pot with a belly, 102 mm across, 75 mm tall, closed inside at z = 64 mm; inner
                      radius 44.8 mm, so plant_soil fits. Origin: bottom centre.
  sit_leaf_large      large leaf to hide things behind: blade 130 x 72 mm on a 30 mm petiole (160 mm overall).
                      Origin and axes as plant_leaf (no scale).
  sit_bin             open bin, 124 mm square with rounded corners at the rim, 110 mm tall, hollow (walls 3.5 mm).
                      One mesh for both bins (brown compost, green). Origin: bottom centre.

Not built: a gravel tray.

Two things to know when these go into MuJoCo (checked with MuJoCo 4.0.1):
  - The compiler moves each mesh to its centre of mass and principal axes. A geom's pos and quat given to the
    spec are in the OBJ's frame (the frames above), but model.geom_pos / geom_quat of the compiled model are those
    of the moved mesh. To place a mesh geom at run time at pose (p, R) of its OBJ frame: geom_pos = p + R @
    mesh_pos, geom_quat = R * mesh_quat (model.mesh_pos, model.mesh_quat; they differ from mesh to mesh, so a
    layer does not get the leaf's geom_pos).
  - The PNGs carry an sRGB tag, which makes MuJoCo darken them: set the texture's colorspace to
    mjCOLORSPACE_LINEAR to get the colours as painted.
"""

import contextlib
import json
import math
import os
import random

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector, noise

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # executed without a file
    _HERE = os.getcwd()
OUT = os.environ.get("SITUATIONS_ASSETS_OUT") or _HERE
PREVIEW_DIR = os.environ.get("SITUATIONS_PREVIEW_DIR")
MESH_DIR = os.path.join(OUT, "meshes", "situations")
os.makedirs(MESH_DIR, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
for blocks in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
    for block in list(blocks):
        blocks.remove(block)
for block in [i for i in bpy.data.images if i.name.startswith("sit_")]:
    bpy.data.images.remove(block)

rng = random.Random(11)
TAU = 2 * math.pi

# The plant as sim/pickup/env.py builds it (for the fit and the previews).
SHOOT_R, LEAF_SCALE, STEM_BASE, STEM_H = 0.003, 0.8, 0.068, 0.240
STEM_GREEN, LEAF_GREEN = (0.20, 0.50, 0.15, 1), (0.18, 0.50, 0.13, 1)
TERRACOTTA, SOIL = (0.72, 0.36, 0.20, 1), (0.22, 0.15, 0.09, 1)
TIE_COLOURS = {
    "red": (0.85, 0.10, 0.10, 1),
    "blue": (0.10, 0.30, 0.90, 1),
    "yellow": (0.98, 0.82, 0.08, 1),
    "white": (0.95, 0.95, 0.93, 1),
}
SICK_LEAF = (0.56, 0.60, 0.18, 1)  # a sick leaf's blade, under the brown layers
BIN_COLOURS = {"compost": (0.40, 0.27, 0.14, 1), "green": (0.16, 0.45, 0.22, 1)}
# Suggested colours (rgba in MuJoCo); the previews use them.
RGBA = {
    "sit_tie": TIE_COLOURS["red"],
    "sit_leaf_wilted": (0.55, 0.47, 0.16, 1),
    "sit_leaf_spots": (0.34, 0.20, 0.07, 1),
    "sit_leaf_edges": (0.45, 0.28, 0.10, 1),
    "sit_bud_body": (0.45, 0.66, 0.22, 1),
    "sit_bud_sepals": STEM_GREEN,
    "sit_watering_can": (0.22, 0.48, 0.46, 1),
    "sit_watering_can_trim": (0.80, 0.80, 0.82, 1),
    "sit_gloves": (0.80, 0.73, 0.60, 1),
    "sit_gloves_cuffs": (0.28, 0.42, 0.30, 1),
    "sit_trowel_blade": (0.62, 0.64, 0.67, 1),
    "sit_trowel_handle": (0.55, 0.33, 0.15, 1),
    "sit_seed_packet": (0.93, 0.90, 0.80, 1),
    "sit_seed_packet_art": (0.85, 0.35, 0.12, 1),
    "sit_pot_small": TERRACOTTA,
    "sit_bee_body": (0.95, 0.72, 0.10, 1),
    "sit_bee_dark": (0.10, 0.08, 0.06, 1),
    "sit_bee_wings": (0.90, 0.93, 0.97, 0.6),
    "sit_bench_top": (0.62, 0.45, 0.28, 1),
    "sit_backdrop_frame": (0.90, 0.91, 0.90, 1),
    "sit_backdrop_glass": (0.72, 0.84, 0.86, 1),
    "sit_symbol_leaf": (0.15, 0.50, 0.15, 1),
    "sit_symbol_sun": (0.96, 0.72, 0.06, 1),
    "sit_legend_card": (0.93, 0.91, 0.85, 1),
    "sit_legend_card_marks": (0.15, 0.15, 0.16, 1),
    "sit_rose_petals": (0.80, 0.08, 0.12, 1),
    "sit_rose_sepals": STEM_GREEN,
    "sit_daisy_petals": (0.96, 0.96, 0.94, 1),
    "sit_daisy_centre": (0.95, 0.75, 0.10, 1),
    "sit_daisy_sepals": STEM_GREEN,
    "sit_tulip_petals": (0.93, 0.50, 0.72, 1),
    "sit_tulip_sepals": STEM_GREEN,
    "sit_pot_square": TERRACOTTA,
    "sit_soil_square": SOIL,
    "sit_pot_glazed": (0.12, 0.30, 0.62, 1),
    "sit_leaf_large": (0.14, 0.42, 0.12, 1),
    "sit_bin": BIN_COLOURS["compost"],
}


# ---------------------------------------------------------------- helpers
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


def ring_lathe(profile, segments):
    """A closed (r, z) profile revolved about Z: a ring."""
    n = len(profile)
    verts = [
        (r * math.cos(TAU * i / segments), r * math.sin(TAU * i / segments), z)
        for r, z in profile
        for i in range(segments)
    ]
    faces = []
    for k in range(n):
        m = (k + 1) % n
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((k * segments + i, k * segments + j, m * segments + j, m * segments + i))
    return verts, faces


def squircle(angle, power):
    """Radius factor that turns a circle into a square with rounded corners (half-width = the radius)."""
    return (abs(math.cos(angle)) ** power + abs(math.sin(angle)) ** power) ** (-1.0 / power)


def squircle_lathe(profile, segments, power):
    return lathe(profile, segments, radial=lambda i, r, tag: r * squircle(TAU * i / segments, power))


def loft(rings, cap_start=True, cap_end=True):
    """Skin between rings of equal vertex count, with flat caps."""
    n = len(rings[0])
    verts = [v for ring in rings for v in ring]
    faces = []
    for k in range(len(rings) - 1):
        a, b = k * n, (k + 1) * n
        faces += [(a + i, a + (i + 1) % n, b + (i + 1) % n, b + i) for i in range(n)]
    if cap_start:
        faces.append(tuple(range(n - 1, -1, -1)))
    if cap_end:
        faces.append(tuple(range((len(rings) - 1) * n, len(rings) * n)))
    return verts, faces


def tube(path, radius, sides=8, flatten=1.0):
    """A round (or, with flatten, oval) section swept along a path. radius: a number or a function of 0..1."""
    pts = [Vector(p) for p in path]
    n = len(pts)
    rings, normal = [], None
    for i, p in enumerate(pts):
        t = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized()
        if normal is None:
            normal = Vector((0, 1, 0)).cross(t)
            if normal.length < 1e-6:
                normal = Vector((1, 0, 0)).cross(t)
        normal = (normal - t * normal.dot(t)).normalized()
        b = t.cross(normal)
        r = radius(i / (n - 1)) if callable(radius) else radius
        rings.append(
            [
                p + (normal * math.cos(TAU * k / sides) + b * (math.sin(TAU * k / sides) * flatten)) * r
                for k in range(sides)
            ]
        )
    return loft(rings)


def chaikin(points, times=2):
    """Round a polyline's corners (the end points stay)."""
    pts = [Vector(p) for p in points]
    for _ in range(times):
        out = [pts[0]]
        for a, b in zip(pts, pts[1:], strict=False):
            out += [a.lerp(b, 0.25), a.lerp(b, 0.75)]
        pts = [*out, pts[-1]]
    return pts


def ellipsoid(centre, radii, segments=12, stacks=6):
    profile = [(math.sin(math.pi * k / stacks), -math.cos(math.pi * k / stacks)) for k in range(1, stacks)]
    verts, faces = lathe([(0.0, -1.0), *profile, (0.0, 1.0)], segments)
    m = Matrix.Translation(centre) @ Matrix.Diagonal((*radii, 1.0))
    return [m @ Vector(v) for v in verts], faces


def prism(outline, z0, z1):
    """A 2D outline (x, y) extruded from z0 to z1."""
    n = len(outline)
    verts = [(x, y, z0) for x, y in outline] + [(x, y, z1) for x, y in outline]
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    faces += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return verts, faces


def box(centre, half):
    (cx, cy, cz), (hx, hy, hz) = centre, half
    return prism([(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)], cz - hz, cz + hz)


def disc(cx, cy, r, n=14):
    return [(cx + r * math.cos(TAU * k / n), cy + r * math.sin(TAU * k / n)) for k in range(n)]


def rounded_rect(hx, hy, r, n=4):
    pts = []
    for (cx, cy), a0 in (
        ((hx - r, hy - r), 0.0),
        ((r - hx, hy - r), 0.25),
        ((r - hx, r - hy), 0.5),
        ((hx - r, r - hy), 0.75),
    ):
        pts += [
            (cx + r * math.cos(TAU * (a0 + 0.25 * k / n)), cy + r * math.sin(TAU * (a0 + 0.25 * k / n)))
            for k in range(n + 1)
        ]
    return pts


def stadium(length, width, n=6):
    """Outline of a finger: from x = 0 to x = length, round at both ends."""
    r = width / 2
    tip = [
        (length - r + r * math.cos(-math.pi / 2 + math.pi * k / n), r * math.sin(-math.pi / 2 + math.pi * k / n))
        for k in range(n + 1)
    ]
    base = [
        (r + r * math.cos(math.pi / 2 + math.pi * k / n), r * math.sin(math.pi / 2 + math.pi * k / n))
        for k in range(n + 1)
    ]
    return tip + base


def move2d(outline, x, y, degrees=0.0, mirror=False):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    out = [(x + c * px - s * py, y + s * px + c * py) for px, py in outline]
    return [(px, -py) for px, py in out] if mirror else out


PILLOW = ((0.0, 0.94), (0.2, 1.0), (0.6, 1.0), (0.87, 0.9), (1.0, 0.66))  # (height fraction, scale about the centre)


def pillow(outline, height, layers=PILLOW):
    """A soft slab: the outline lofted upward, pulled in toward the top."""
    cx = sum(p[0] for p in outline) / len(outline)
    cy = sum(p[1] for p in outline) / len(outline)
    return loft([[(cx + (x - cx) * k, cy + (y - cy) * k, f * height) for x, y in outline] for f, k in layers])


def shell(top, bottom):
    """Thin closed solid between two rows x cols grids of points (upper and lower surface)."""
    rows, cols = len(top), len(top[0])
    nt = rows * cols
    verts = [Vector(p) for row in top for p in row] + [Vector(p) for row in bottom for p in row]
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

    return shell(P, [[P[i][j] - normal(i, j) * thickness for j in range(cols)] for i in range(rows)])


def material(name, rgba, rough=0.55, image=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*rgba[:3], 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    if image:
        tex = m.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        m.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return m


PAINTS = {}


def paint(rgba):
    key = tuple(round(c, 3) for c in rgba)
    if key not in PAINTS:
        PAINTS[key] = material(f"paint_{len(PAINTS)}", rgba)
    return PAINTS[key]


def new_object(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def finish(obj, sharp_deg=40.0):
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
    me.materials.append(paint(RGBA.get(obj.name, (0.6, 0.6, 0.6, 1))))
    return obj


PARTS = []


def build(name, geo, sharp_deg=40.0, uv=None):
    """A finished part, queued for export. uv: a function of a vertex position, for texture coordinates."""
    obj = finish(new_object(name, geo.verts, geo.faces), sharp_deg)
    if uv:
        layer = obj.data.uv_layers.new(name="uv")
        for loop in obj.data.loops:
            layer.data[loop.index].uv = uv(obj.data.vertices[loop.vertex_index].co)
    PARTS.append(obj)
    return obj


def activate(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def prim(kind, size, pos=(0, 0, 0), axis=None, yaw=0.0):
    """A collision primitive in the asset's frame, as MuJoCo takes it: box (half sizes), cylinder and capsule
    (radius, half length; along `axis`, default Z), quat (w, x, y, z)."""
    q = Quaternion((0, 0, 1), yaw)
    if axis is not None:
        q = Vector((0, 0, 1)).rotation_difference(Vector(axis).normalized())
    return {
        "type": kind,
        "size": [round(v, 5) for v in size],
        "pos": [round(v, 5) for v in pos],
        "quat": [round(v, 5) for v in q],
    }


def frame(pos, rotation):
    q = rotation.to_quaternion()
    return {"pos": [round(v, 5) for v in pos], "quat": [round(v, 5) for v in q]}


ASSETS = {}  # asset -> meshes, origin, how to attach, collision primitives

# ---------------------------------------------------------------- tier 1: gardener's tie
tie_geo = Geo().add(
    *ring_lathe(
        [(0.0029, -0.006), (0.0050, -0.006), (0.0056, -0.0045), (0.0056, 0.0045), (0.0050, 0.006), (0.0029, 0.006)],
        20,
    )
)
tie_geo.add(*ellipsoid((0.0064, 0, 0), (0.0030, 0.0034, 0.0038), 10, 6))  # the knot
for side in (-1, 1):  # two ribbon tails, twisted so that some of each shows its face from any side
    start, end = Vector((0.0072, 0, side * 0.0018)), Vector((0.0300, 0, side * 0.0200))
    along = (end - start).normalized()
    across = along.cross(Vector((0, 1, 0)))
    grid = []
    for k in range(8):
        u = k / 7
        c = start.lerp(end, u) + across * (side * 0.0026 * math.sin(math.pi * u))
        twist = side * math.radians(85) * u
        w = (Vector((0, 1, 0)) * math.cos(twist) + across * math.sin(twist)) * (0.0038 + 0.0018 * u)
        notch = along * (-0.0045 if k == 7 else 0.0)  # swallow-tail end
        grid.append([c - w, c + notch, c + w])
    tie_geo.add(*sheet(grid, 0.0011))
tie = build("sit_tie", tie_geo, sharp_deg=50)
ASSETS["tie"] = {
    "meshes": ["sit_tie"],
    "origin": "on the shoot's axis, middle of the band; shoot along +Z, knot and tails toward +X",
    "attach": "in a shoot's frame (z along the shoot, x upward) at (0, 0, d), identity rotation, d = distance from "
    "the main stem's axis: 0.012 puts the band on the stub, which stays on the plant after the cut; 0.058 or more "
    "puts it beyond the cut zone, on the part that falls. Visual only.",
    "colours": {k: list(v) for k, v in TIE_COLOURS.items()},
    "collision": [],
}

# ---------------------------------------------------------------- tier 1: sick leaves
LEAF_X0, LEAF_L, LEAF_HALF_W, LEAF_T = 0.010, 0.050, 0.012, 0.0005  # plant_leaf (build_plant_blender.py)
LEAF_ROWS = (0.0, 0.04, 0.10, 0.20, 0.32, 0.45, 0.58, 0.70, 0.80, 0.89, 0.96, 1.0)


def leaf_half_width(t):
    return max(0.0005, LEAF_HALF_W * max(0.0, math.sin(math.pi * t**0.7)) ** 0.8)


def leaf_point(t, s):
    """The upper surface of plant_leaf: t along the blade (0..1), s across it (-1..1)."""
    t = min(1.0, max(0.0, t))
    w = leaf_half_width(t)
    arch = 0.006 * math.sin(0.9 * math.pi * t) - 0.006 * t * t
    y = s * w
    z = arch + 0.30 * abs(y) - 0.10 * w * s**4 + 0.0005 * math.sin(16 * t + 2.0 * s) * abs(s)
    return Vector((LEAF_X0 + LEAF_L * t, y, z))


def leaf_normal(t, s, e=1e-3):
    du = leaf_point(t + e, s) - leaf_point(t - e, s)
    dv = leaf_point(t, s + e) - leaf_point(t, s - e)
    n = du.cross(dv).normalized()
    return -n if n.z < 0 else n


def leaf_wrap(grid, lift=0.00035):
    """A layer round the blade of plant_leaf over a grid of (t, s): `lift` proud of the upper and the lower face."""
    top = [[leaf_point(t, s) + leaf_normal(t, s) * lift for t, s in row] for row in grid]
    bottom = [[leaf_point(t, s) - leaf_normal(t, s) * (LEAF_T + lift) for t, s in row] for row in grid]
    return shell(top, bottom)


spots_geo = Geo()
for t0, s0, rt, rs in (
    (0.22, -0.45, 0.085, 0.36),
    (0.34, 0.52, 0.10, 0.40),
    (0.50, -0.30, 0.075, 0.38),
    (0.62, 0.40, 0.085, 0.45),
    (0.76, -0.35, 0.070, 0.50),
    (0.47, 0.08, 0.045, 0.20),
):
    grid = []
    for i in range(5):
        tau = -1 + 0.5 * i
        chord = max(0.35, math.sqrt(1 - tau * tau)) * rng.uniform(0.85, 1.1)
        row = []
        for j in range(5):
            sigma = -1 + 0.5 * j
            s = max(-1.03, min(1.03, s0 + rs * chord * sigma))
            row.append((t0 + rt * tau + 0.012 * math.sin(3 * sigma + i), s))
        grid.append(row)
    spots_geo.add(*leaf_wrap(grid))
spots = build("sit_leaf_spots", spots_geo, sharp_deg=50)

edges_geo = Geo()
EDGE_T = [0.06 + 0.80 * k / 12 for k in range(13)]
for side in (-1, 1):
    grid = []
    for t in EDGE_T:
        inner = 0.60 + 0.13 * math.sin(11 * t + side) + 0.25 * max(0.0, 0.2 - t)
        grid.append([(t, side * (inner + (1.05 - inner) * k / 2)) for k in range(3)])
    edges_geo.add(*leaf_wrap(grid))
edges_geo.add(*leaf_wrap([[(t, -1.05 + 2.1 * j / 4) for j in range(5)] for t in (0.83, 0.88, 0.93, 0.97, 1.0)]))
edges = build("sit_leaf_edges", edges_geo, sharp_deg=50)


def petiole(points, r0, r1):
    return tube(points, lambda u: r0 + (r1 - r0) * u, 6)


WILT_STEPS = 240
wilt_line, wilt_angle = [], []
p = Vector((LEAF_X0, 0.0, 0.0))
for k in range(WILT_STEPS + 1):  # the midrib: bends down more and more toward the tip
    t = k / WILT_STEPS
    angle = -math.radians(6 + 108 * t**1.15)
    wilt_line.append(p.copy())
    wilt_angle.append(angle)
    p += Vector((math.cos(angle), 0.0, math.sin(angle))) * (LEAF_L / WILT_STEPS)
wilt_grid = []
for t in LEAF_ROWS:
    k = round(t * WILT_STEPS)
    c, angle = wilt_line[k], wilt_angle[k]
    up = Vector((-math.sin(angle), 0.0, math.cos(angle)))
    w = leaf_half_width(t)
    roll = math.radians(28) * t
    row = []
    for s in (-1.0, -0.6, -0.25, 0.0, 0.25, 0.6, 1.0):
        y = s * w * (1 - 0.17 * s * s)  # the curled edges make it narrower
        z = -(0.25 + 0.45 * t) * abs(s) * w + 0.0011 * math.sin(21 * t + 3.0 * s) * (0.3 + abs(s))
        row.append(
            c
            + Vector((0, 1, 0)) * (y * math.cos(roll) - z * math.sin(roll))
            + up * (y * math.sin(roll) + z * math.cos(roll))
        )
    wilt_grid.append(row)
wilted_geo = Geo().add(*sheet(wilt_grid, LEAF_T))
wilted_geo.add(*petiole([(0.0, 0, -0.0003), (0.006, 0, -0.0002), (0.0125, 0, -0.0006)], 0.0012, 0.0008))
wilted = build("sit_leaf_wilted", wilted_geo, sharp_deg=45)
ASSETS["sick_leaves"] = {
    "meshes": ["sit_leaf_wilted", "sit_leaf_spots", "sit_leaf_edges"],
    "origin": "as plant_leaf: petiole base, blade toward +X, upper side +Z",
    "attach": "sit_leaf_wilted replaces plant_leaf in a leaf geom (same pose, same mesh scale 0.8). sit_leaf_spots "
    "and sit_leaf_edges are extra geoms whose OBJ frame has exactly the pose and scale of a plant_leaf geom's "
    "(see 'about': not the same geom_pos), coloured brown; give the leaf under them the colour 'sick_leaf'. "
    "The leaf's collision box (env.py) suits all of them except that "
    "the wilted blade hangs up to 30 mm below it.",
    "colours": {"sick_leaf": list(SICK_LEAF)},
    "collision": [],
}

# ---------------------------------------------------------------- tier 1: flower bud
BUD_BODY = [
    (0.0, 0.0060),
    (0.0050, 0.0065),
    (0.0068, 0.0100),
    (0.0073, 0.0140),
    (0.0068, 0.0190),
    (0.0052, 0.0240),
    (0.0028, 0.0278),
    (0.0, 0.0290),
]
BUD_CALYX = [
    (0.0, -0.004),
    (0.0037, -0.004),
    (0.0039, 0.0),
    (0.0052, 0.003),
    (0.0066, 0.007),
    (0.0069, 0.0095),
    (0.0, 0.010),
]


def bud_radius(z):
    for (r0, z0), (r1, z1) in zip(BUD_BODY[1:], BUD_BODY[2:], strict=False):
        if z0 <= z <= z1:
            return r0 + (r1 - r0) * (z - z0) / (z1 - z0)
    return BUD_BODY[1][0]


bud_body = build("sit_bud_body", Geo().add(*lathe(BUD_BODY, 20)), sharp_deg=60)
bud_geo = Geo().add(*lathe(BUD_CALYX, 16))
for i in range(5):
    a = TAU * (i + 0.5) / 5
    grid = []
    for z, lift, half_w in (
        (0.0075, 0.0004, 0.0040),
        (0.012, 0.0004, 0.0044),
        (0.017, 0.0005, 0.0034),
        (0.0215, 0.0009, 0.0018),
        (0.0255, 0.0024, 0.0004),
    ):
        r = bud_radius(z) + lift
        grid.append([(r * math.cos(a + s * half_w / r), r * math.sin(a + s * half_w / r), z) for s in (-1, 0, 1)])
    bud_geo.add(*sheet(grid, 0.0005))
bud_sepals = build("sit_bud_sepals", bud_geo, sharp_deg=45)
ASSETS["bud"] = {
    "meshes": ["sit_bud_body", "sit_bud_sepals"],
    "origin": "on the shoot's axis at the shoot's tip; the bud points along +Z; the collar reaches back to z = -0.004",
    "attach": "in a shoot's frame at (0, 0, shoot length), identity rotation (both meshes). The leaves sit at the "
    "same tip: with a bud use two leaves, yawed at least 0.5 rad to either side, so that none runs through it. "
    "For a collision shape: a capsule of radius 0.006 from z = 0.006 to 0.022.",
    "collision": [prim("capsule", (0.006, 0.008), (0, 0, 0.014))],
}

# ---------------------------------------------------------------- tier 1: distractors
# watering can
CAN_BODY = [
    (0.0, 0.0),
    (0.0425, 0.0),
    (0.0440, 0.0015),
    (0.0405, 0.0870),
    (0.0412, 0.0885),
    (0.0405, 0.0900),
    (0.0380, 0.0900),
    (0.0375, 0.0860),
    (0.0405, 0.0120),
    (0.0, 0.0120),
]
can_geo = Geo().add(*lathe(CAN_BODY, 32))
SPOUT = chaikin([(0.036, 0, 0.026), (0.075, 0, 0.052), (0.110, 0, 0.088), (0.134, 0, 0.108)], 1)
can_geo.add(*tube(SPOUT, lambda u: 0.0088 - 0.0030 * u, 10))
HANDLE = chaikin(
    [
        (-0.036, 0, 0.082),
        (-0.066, 0, 0.096),
        (-0.090, 0, 0.082),
        (-0.093, 0, 0.048),
        (-0.074, 0, 0.022),
        (-0.040, 0, 0.017),
    ],
    2,
)
can_geo.add(*tube(HANDLE, 0.0042, 8, flatten=2.2))
can = build("sit_watering_can", can_geo, sharp_deg=45)
spout_dir = (Vector(SPOUT[-1]) - Vector(SPOUT[-2])).normalized()
spout_end = Vector(SPOUT[-1])
trim_geo = Geo().add(
    *tube(
        [
            spout_end - spout_dir * 0.004,
            spout_end + spout_dir * 0.012,
            spout_end + spout_dir * 0.0165,
            spout_end + spout_dir * 0.0175,
        ],
        lambda u: (0.0064, 0.0150, 0.0156, 0.0135)[round(u * 3)],
        14,
    )
)
trim_geo.add(*ring_lathe([(0.0400, 0.0780), (0.0418, 0.0784), (0.0416, 0.0836), (0.0398, 0.0840)], 32))
can_trim = build("sit_watering_can_trim", trim_geo, sharp_deg=45)
ASSETS["watering_can"] = {
    "meshes": ["sit_watering_can", "sit_watering_can_trim"],
    "origin": "centre of the body's bottom; spout toward +X, handle toward -X",
    "attach": "stands on the table at z = 0; any yaw",
    "mass": 0.25,
    "collision": [
        prim("cylinder", (0.044, 0.045), (0, 0, 0.045)),
        prim("capsule", (0.009, 0.066), (0.089, 0, 0.070), axis=(0.105, 0, 0.082)),
        prim("box", (0.028, 0.009, 0.040), (-0.066, 0, 0.057)),
    ],
}


# gardening gloves
def glove(geo, cuffs, matrix, mirror):
    parts = [(rounded_rect(0.050, 0.047, 0.018), 0.0140)]  # palm
    for y, length, splay in ((-0.034, 0.062, -7), (-0.0115, 0.076, -2), (0.0115, 0.080, 2), (0.034, 0.068, 7)):
        parts.append((move2d(stadium(length + 0.022, 0.0215), 0.028, y, splay), 0.0120))
    parts.append((move2d(stadium(0.066, 0.025), -0.030, 0.026, 48), 0.0125))  # thumb
    for outline, height in parts:
        geo.add(*pillow(move2d(outline, 0, 0, 0, mirror), height), matrix)
    cuffs.add(*pillow(move2d(rounded_rect(0.034, 0.0435, 0.008), -0.076, 0, 0, mirror), 0.0132), matrix)


gloves_geo, cuffs_geo = Geo(), Geo()
glove(gloves_geo, cuffs_geo, Matrix.Rotation(math.radians(-7), 4, "Z"), False)
GLOVE_2 = Matrix.Translation((0.010, -0.006, 0.0125)) @ Matrix.Rotation(math.radians(9), 4, "Z")
glove(gloves_geo, cuffs_geo, GLOVE_2, True)
gloves = build("sit_gloves", gloves_geo, sharp_deg=50)
gloves_cuffs = build("sit_gloves_cuffs", cuffs_geo, sharp_deg=50)
ASSETS["gloves"] = {
    "meshes": ["sit_gloves", "sit_gloves_cuffs"],
    "origin": "on the table, under the lower glove's palm; fingers toward +X",
    "attach": "lies on the table at z = 0; any yaw",
    "mass": 0.12,
    "collision": [prim("box", (0.127, 0.080, 0.01325), (0.0136, -0.003, 0.01325))],
}

# hand trowel
blade_grid = []
for t in (0.0, 0.05, 0.13, 0.28, 0.45, 0.62, 0.78, 0.90, 0.97, 1.0):
    w = 0.036 * min(1.0, (t / 0.13) ** 0.5 + 0.25) * (1 - max(0.0, (t - 0.30) / 0.70) ** 1.9)
    w = max(w, 0.0012)
    z0 = 0.0032 + 0.0070 * (1 - t) ** 2.5
    blade_grid.append([(0.005 + 0.150 * t, s * w, z0 + 0.42 * w * s * s) for s in (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0)])
blade_geo = Geo().add(*sheet(blade_grid, 0.0015))
NECK = chaikin([(-0.044, 0, 0.0147), (-0.020, 0, 0.0147), (-0.004, 0, 0.0128), (0.016, 0, 0.0090)], 1)
blade_geo.add(*tube(NECK, lambda u: 0.0080 if u < 0.2 else 0.0046, 10))
trowel_blade = build("sit_trowel_blade", blade_geo, sharp_deg=45)
HANDLE_PROFILE = [
    (0.0, 0.0),
    (0.0090, 0.0006),
    (0.0130, 0.0045),
    (0.0145, 0.0150),
    (0.0132, 0.0450),
    (0.0116, 0.0750),
    (0.0122, 0.1000),
    (0.0100, 0.1080),
    (0.0, 0.1080),
]
ALONG_X = Matrix.Translation((-0.148, 0, 0.0147)) @ Matrix.Rotation(math.radians(90), 4, "Y")
trowel_handle = build("sit_trowel_handle", Geo().add(*lathe(HANDLE_PROFILE, 16), ALONG_X), sharp_deg=45)
ASSETS["trowel"] = {
    "meshes": ["sit_trowel_blade", "sit_trowel_handle"],
    "origin": "on the table, under the neck; handle toward -X, blade toward +X",
    "attach": "lies on the table at z = 0; any yaw",
    "mass": 0.20,
    "collision": [
        prim("capsule", (0.0135, 0.041), (-0.094, 0, 0.0147), axis=(1, 0, 0)),
        prim("box", (0.076, 0.034, 0.007), (0.079, 0, 0.008)),
    ],
}

# seed packet
PACKET = rounded_rect(0.041, 0.060, 0.003, 3)
packet = build(
    "sit_seed_packet",
    Geo().add(*pillow(PACKET, 0.004, ((0.0, 0.985), (0.2, 1.0), (0.8, 1.0), (1.0, 0.985)))),
    sharp_deg=30,
)
ART_Z = (0.0036, 0.0046)
art_geo = Geo().add(*prism(move2d(rounded_rect(0.035, 0.0105, 0.002, 2), 0, 0.043), *ART_Z))
art_geo.add(*prism(disc(0, -0.008, 0.0062), *ART_Z))
for k in range(6):
    art_geo.add(*prism(disc(0.0125 * math.cos(TAU * k / 6), -0.008 + 0.0125 * math.sin(TAU * k / 6), 0.0068), *ART_Z))
art_geo.add(*box((0, -0.0375, 0.0041), (0.0016, 0.0145, 0.0005)))
for side in (-1, 1):
    art_geo.add(*prism(move2d(stadium(0.017, 0.008, 4), 0, -0.043, 90 - side * 55), *ART_Z))
packet_art = build("sit_seed_packet_art", art_geo, sharp_deg=30)
ASSETS["seed_packet"] = {
    "meshes": ["sit_seed_packet", "sit_seed_packet_art"],
    "origin": "centre of the underside; the long side along Y, the header band at +Y",
    "attach": "lies on the table at z = 0; any yaw",
    "mass": 0.01,
    "collision": [prim("box", (0.041, 0.060, 0.002), (0, 0, 0.002))],
}

# empty pot
POT_SMALL = [
    (0.0, 0.0),
    (0.0258, 0.0),
    (0.0270, 0.0012),
    (0.0328, 0.0450),
    (0.0358, 0.0455),
    (0.0370, 0.0468),
    (0.0370, 0.0560),
    (0.0362, 0.0576),
    (0.0348, 0.0580),
    (0.0334, 0.0572),
    (0.0330, 0.0540),
    (0.0240, 0.0050),
    (0.0, 0.0050),
]
pot_small = build("sit_pot_small", Geo().add(*lathe(POT_SMALL, 40)))
ASSETS["pot_small"] = {
    "meshes": ["sit_pot_small"],
    "origin": "centre of the bottom",
    "attach": "stands on the table at z = 0 (or lies on its side)",
    "mass": 0.09,
    "collision": [prim("cylinder", (0.031, 0.0225), (0, 0, 0.0225)), prim("cylinder", (0.037, 0.0065), (0, 0, 0.0515))],
}

# bee
BEE_Z = 0.0036
bee_geo = Geo().add(*ellipsoid((-0.0030, 0, BEE_Z), (0.0048, 0.0031, 0.0030), 12, 8))
bee_geo.add(*ellipsoid((0.0028, 0, BEE_Z + 0.0002), (0.0026, 0.0027, 0.0026), 10, 6))
bee_body = build("sit_bee_body", bee_geo, sharp_deg=60)
dark_geo = Geo().add(*ellipsoid((0.0060, 0, BEE_Z), (0.0017, 0.0020, 0.0019), 10, 6))
dark_geo.add(*ellipsoid((-0.0068, 0, BEE_Z), (0.0011, 0.0019, 0.0018), 10, 6))  # tail
TO_X = Matrix.Translation((-0.0030, 0, BEE_Z)) @ Matrix.Rotation(math.radians(90), 4, "Y")
for dx in (-0.0020, 0.0006):  # stripes round the abdomen
    stripe = [
        (0.0031 * math.sqrt(1 - ((dx + d) / 0.0048) ** 2) + lift, dx + d)
        for d, lift in ((-0.0006, -0.0005), (-0.0006, 0.0003), (0.0006, 0.0003), (0.0006, -0.0005))
    ]
    dark_geo.add(*ring_lathe(stripe, 12), TO_X)
bee_dark = build("sit_bee_dark", dark_geo, sharp_deg=60)
wings_geo = Geo()
for side in (-1, 1):
    root = Vector((0.0030, side * 0.0010, BEE_Z + 0.0024))
    out = Vector((-0.70, side * 0.66, 0.20)).normalized()
    wide = Vector((0.66, side * 0.70, 0.0)).normalized()
    grid = [
        [root + out * (0.0095 * u) + wide * (s * 0.0019 * math.sin(math.pi * (0.12 + 0.80 * u))) for s in (-1, 0, 1)]
        for u in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    wings_geo.add(*sheet(grid, 0.0002))
bee_wings = build("sit_bee_wings", wings_geo, sharp_deg=60)
ASSETS["bee"] = {
    "meshes": ["sit_bee_body", "sit_bee_dark", "sit_bee_wings"],
    "origin": "on the surface it sits on, under the thorax; head toward +X, up +Z",
    "attach": "all three meshes in one frame; on a leaf, use the leaf geom's frame moved about 0.03 m along the "
    "blade and 0.004 m up (times the leaf's scale). Visual only.",
    "collision": [],
}

# ---------------------------------------------------------------- tier 1: bench top and backdrop
PLANKS, PLANK_GAP = 8, 0.0005
bench_geo = Geo()
PLANK_SECTION = [
    (-0.05 + PLANK_GAP, -0.02),
    (0.05 - PLANK_GAP, -0.02),
    (0.05 - PLANK_GAP, 0.0184),
    (0.0484, 0.02),
    (-0.0484, 0.02),
    (-0.05 + PLANK_GAP, 0.0184),
]
SECTION_TO_XZ = Matrix.Rotation(math.radians(90), 4, "X")  # a prism's (x, y, z) -> (x, -z, y)
for k in range(PLANKS):
    bench_geo.add(*prism(PLANK_SECTION, -0.6, 0.6), Matrix.Translation((-0.35 + 0.1 * k, 0, 0)) @ SECTION_TO_XZ)
bench = build("sit_bench_top", bench_geo, sharp_deg=30, uv=lambda co: ((co.y + 0.6) / 1.2, (co.x + 0.4) / 0.8))

BACK_W, BACK_H = 2.4, 1.6
frame_geo = Geo()
for x in (-1.18, -0.59, 0.0, 0.59, 1.18):
    frame_geo.add(*box((x, -0.008, BACK_H / 2), (0.02, 0.018, BACK_H / 2)))
for z, half in ((0.07, 0.07), (0.86, 0.018), (1.58, 0.02)):
    frame_geo.add(*box((0, -0.010, z), (BACK_W / 2, 0.020, half)))
backdrop_frame = build("sit_backdrop_frame", frame_geo, sharp_deg=30)
backdrop_glass = build(
    "sit_backdrop_glass",
    Geo().add(*box((0, 0.003, BACK_H / 2), (BACK_W / 2 - 0.005, 0.003, BACK_H / 2 - 0.005))),
    sharp_deg=30,
    uv=lambda co: ((co.x + BACK_W / 2) / BACK_W, co.z / BACK_H),
)
ASSETS["bench"] = {
    "meshes": ["sit_bench_top"],
    "origin": "centre of the slab (the top surface is z = +0.02), planks along Y",
    "attach": "replaces the mesh table_top in the geom table_visual (same position); the collision box stays",
    "textures": ["sit_bench_wood.png", "sit_bench_stained.png"],
    "collision": [],
}
ASSETS["backdrop"] = {
    "meshes": ["sit_backdrop_frame", "sit_backdrop_glass"],
    "origin": "bottom centre; 2.4 m along X, 1.6 m up Z, to be seen from -Y",
    "attach": "stands on the floor (z = -0.76 in the world) behind the scene as the front camera sees it: e.g. at "
    "(-0.75, 0, -0.76) turned +90 degrees about Z (it then faces +x) and a second one at (0.2, 0.95, -0.76) not "
    "turned. Visual only.",
    "textures": ["sit_backdrop_greenhouse.png"],
    "collision": [],
}

# ---------------------------------------------------------------- tier 2: symbols and the legend card
SYMBOL_T = 0.0015
leaf_outline = []
for side in (1, -1):
    for k in range(10):
        u = k / 10 if side == 1 else 1 - k / 10
        w = 0.0050 * max(0.0, math.sin(math.pi * u**0.8)) ** 0.9
        leaf_outline.append((-0.0085 + 0.0180 * u, side * w))
symbol_leaf_geo = Geo().add(*prism(move2d(leaf_outline, 0.0008, 0.0008, 45), 0, SYMBOL_T))
symbol_leaf_geo.add(*prism(move2d(rounded_rect(0.0024, 0.0008, 0.0004, 1), -0.0068, -0.0068, 45), 0, SYMBOL_T))  # stalk
symbol_leaf = build("sit_symbol_leaf", symbol_leaf_geo, sharp_deg=30)
symbol_sun_geo = Geo().add(*prism(disc(0, 0, 0.0044, 18), 0, SYMBOL_T))
for k in range(8):
    ray = [(0.0057, -0.0013), (0.0095, 0.0), (0.0057, 0.0013)]
    symbol_sun_geo.add(*prism(move2d(ray, 0, 0, 45 * k), 0, SYMBOL_T))
symbol_sun = build("sit_symbol_sun", symbol_sun_geo, sharp_deg=30)

CARD_LEAN = math.radians(25)
CARD_UP = Vector((-math.sin(CARD_LEAN), 0, math.cos(CARD_LEAN)))
CARD_OUT = Vector((math.cos(CARD_LEAN), 0, math.sin(CARD_LEAN)))
CARD_FOOT = Vector((0.0265, 0, 0.0017))  # the bottom edge of the face
CARD_ROT = Matrix((Vector((0, 1, 0)), CARD_UP, CARD_OUT)).transposed()  # face frame: x right, y up, z out
CARD_FACE = Matrix.Translation(CARD_FOOT + CARD_UP * 0.0375) @ CARD_ROT.to_4x4()
card_geo = Geo().add(*box((0, 0, -0.002), (0.055, 0.0375, 0.002)), CARD_FACE)
prop_top = CARD_FOOT + CARD_UP * 0.058 - CARD_OUT * 0.004
prop_low = prop_top - CARD_UP * 0.008
card_geo.add(
    *prism([(prop_top.x, prop_top.z), (prop_low.x, prop_low.z), (-0.0195, 0.0), (-0.0265, 0.0)], -0.006, 0.006),
    SECTION_TO_XZ,
)
legend_card = build("sit_legend_card", card_geo, sharp_deg=30)
marks_geo = Geo()
CARD_ROWS = (0.017, -0.017)
for row in CARD_ROWS:
    for dy in (-0.003, 0.003):
        marks_geo.add(*box((0, row + dy, 0.0006), (0.006, 0.0011, 0.0006)), CARD_FACE)
legend_marks = build("sit_legend_card_marks", marks_geo, sharp_deg=30)
ASSETS["legend_card"] = {
    "meshes": ["sit_legend_card", "sit_legend_card_marks"],
    "origin": "on the table, centre of the footprint; the face looks toward +X, leaning back 25 degrees",
    "attach": "stands on the table at z = 0, turned to face the camera. 'slots' are frames on the face (in the "
    "card's frame) for a symbol (sit_symbol_leaf or sit_symbol_sun, mesh scale 1.4) and the arrow it means "
    "(plant_arrow_cw or plant_arrow_ccw, mesh scale 1.4), one pair per row. On the buttons, a symbol takes the "
    "arrow's place: 0.0094 above the button's origin, scale 1.",
    "slots": {
        f"row{i}_{name}": frame(CARD_FACE @ Vector((x, row, 0.0)), CARD_ROT)
        for i, row in enumerate(CARD_ROWS)
        for name, x in (("symbol", -0.030), ("arrow", 0.030))
    },
    "mass": 0.03,
    "collision": [prim("box", (0.0265, 0.055, 0.034), (0, 0, 0.034))],
}

# ---------------------------------------------------------------- tier 2: flower heads
TILT = Matrix.Rotation(math.radians(15), 4, "Y")  # +Z leans toward +X, as the gerbera head
RECEPTACLE = [  # plant_flower_sepals: a 4.3 mm collar that sleeves the stem top
    (0.0, -0.0030),
    (0.0043, -0.0030),
    (0.0046, 0.0),
    (0.0066, 0.0030),
    (0.0104, 0.0064),
    (0.0120, 0.0090),
    (0.0, 0.0095),
]
SLIM_RECEPTACLE = [(0.0, -0.0030), (0.0043, -0.0030), (0.0046, 0.0), (0.0058, 0.0030), (0.0074, 0.0062), (0.0, 0.0068)]
PETAL_T = (0.0, 0.22, 0.50, 0.78, 0.94, 1.0)
PETAL_W = (0.45, 0.82, 1.00, 0.90, 0.60, 0.22)


def ray_petal(angle, r0, z0, tip_r, width, elev, droop, cup, twist):
    """A flat petal that radiates from the centre (as the gerbera's)."""
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


def cup_petal(angle, spread, radius, height, z0, tip, power, lip=0.0, rows=7, cols=5):
    """A petal that stands in a cup round the axis. radius: a function of 0..1 (base to tip); spread: its half
    angle at its widest; tip < 1 leaves a broad top edge, 1 a point; lip rolls the top edge outward."""
    grid = []
    for i in range(rows):
        t = i / (rows - 1)
        half = spread * max(0.03, math.sin(math.pi * tip * t**power)) ** 0.6
        r = radius(t) + (lip if i == rows - 1 else 0.35 * lip if i == rows - 2 else 0.0)
        z = z0 + height * t - (0.5 * lip if i == rows - 1 else 0.0)
        row = []
        for j in range(cols):
            s = -1 + 2 * j / (cols - 1)
            row.append((r * math.cos(angle + s * half), r * math.sin(angle + s * half), z - 0.06 * height * s * s * t))
        grid.append(row)
    return sheet(grid, 0.0005)


def bracts(geo, count, rows):
    for i in range(count):
        a = TAU * (i + 0.5) / count
        u, v = Vector((math.cos(a), math.sin(a), 0)), Vector((-math.sin(a), math.cos(a), 0))
        geo.add(
            *sheet([[u * r + v * (s * w) + Vector((0, 0, z)) for s in (-1, 0, 1)] for r, z, w in rows], 0.0005), TILT
        )
    return geo


# rose: five rings of cupped petals, opening outward
rose_geo = Geo()
for count, r0, r1, height, spread, flare, lip in (
    (3, 0.0020, 0.0045, 0.0250, 78, 0.0, 0.0),
    (4, 0.0040, 0.0090, 0.0245, 64, 0.001, 0.0008),
    (5, 0.0060, 0.0140, 0.0225, 52, 0.002, 0.0015),
    (5, 0.0080, 0.0190, 0.0190, 50, 0.004, 0.0025),
    (6, 0.0090, 0.0230, 0.0150, 44, 0.006, 0.0035),
):
    phase = rng.uniform(0, TAU)
    for i in range(count):
        rose_geo.add(
            *cup_petal(
                phase + TAU * (i + rng.uniform(-0.08, 0.08)) / count,
                math.radians(spread),
                lambda t, r0=r0, r1=r1, flare=flare: r0 + (r1 - r0) * t**0.8 + flare * t**3,
                height * rng.uniform(0.95, 1.05),
                0.0060,
                0.72,
                0.5,
                lip,
            ),
            TILT,
        )
rose_petals = build("sit_rose_petals", rose_geo, sharp_deg=50)
rose_sepals = build(
    "sit_rose_sepals",
    bracts(
        Geo().add(*lathe(RECEPTACLE, 16), TILT),
        5,
        ((0.0090, 0.0050, 0.0036), (0.0160, 0.0042, 0.0028), (0.0230, 0.0012, 0.0013), (0.0270, -0.0025, 0.0003)),
    ),
    sharp_deg=45,
)

# daisy: one ring of narrow white petals round a yellow dome
daisy_geo = Geo()
for i in range(21):
    daisy_geo.add(
        *ray_petal(
            TAU * (i + rng.uniform(-0.12, 0.12)) / 21,
            0.0075,
            0.0100 + 0.0004 * (i % 2),
            0.0310 * rng.uniform(0.94, 1.0),
            0.0066 * rng.uniform(0.9, 1.08),
            math.radians(5 + rng.uniform(-3, 3)),
            rng.uniform(0.04, 0.12),
            0.25,
            rng.uniform(-0.2, 0.2),
        ),
        TILT,
    )
daisy_petals = build("sit_daisy_petals", daisy_geo, sharp_deg=45)
DAISY_CENTRE = [
    (0.0, 0.0090),
    (0.0100, 0.0090),
    (0.0104, 0.0112),
    (0.0092, 0.0136),
    (0.0062, 0.0153),
    (0.0030, 0.0159),
    (0.0, 0.0161),
]
daisy_centre = build("sit_daisy_centre", Geo().add(*lathe(DAISY_CENTRE, 20), TILT), sharp_deg=50)
daisy_sepals = build(
    "sit_daisy_sepals",
    bracts(
        Geo().add(*lathe(RECEPTACLE, 16), TILT),
        10,
        ((0.0085, 0.0052, 0.0030), (0.0135, 0.0082, 0.0025), (0.0178, 0.0096, 0.0004)),
    ),
    sharp_deg=45,
)

# tulip: three inner and three outer petals in a tall cup
tulip_geo = Geo()
for ring, (scale, height) in enumerate(((1.0, 0.0440), (0.90, 0.0420))):
    for i in range(3):
        tulip_geo.add(
            *cup_petal(
                TAU * (i + 0.5 * ring) / 3 + rng.uniform(-0.06, 0.06),
                math.radians(66),
                lambda t, scale=scale: scale * (0.0035 + 0.0135 * math.sin(math.pi * 0.80 * t**0.75)),
                height,
                0.0040,
                1.0,
                0.8,
                rows=9,
            ),
            TILT,
        )
tulip_petals = build("sit_tulip_petals", tulip_geo, sharp_deg=50)
tulip_sepals = build("sit_tulip_sepals", Geo().add(*lathe(SLIM_RECEPTACLE, 16), TILT), sharp_deg=45)
HEAD_ATTACH = (
    "in the plant's frame at (0, 0, 0.308), the top of the main stem, as the plant_flower_* geoms (all meshes of a "
    "head in one frame). The head leans 15 degrees toward the plant's +X; the collision cylinder is given in the "
    "head's frame."
)
for name, parts, radius, half, lift in (
    ("rose", ("petals", "sepals"), 0.031, 0.014, 0.016),
    ("daisy", ("petals", "centre", "sepals"), 0.032, 0.008, 0.012),
    ("tulip", ("petals", "sepals"), 0.018, 0.023, 0.025),
):
    ASSETS[f"flower_{name}"] = {
        "meshes": [f"sit_{name}_{part}" for part in parts],
        "origin": "the receptacle's bottom, where the head meets the top of the main stem (as plant_flower_*)",
        "attach": HEAD_ATTACH,
        "collision": [prim("cylinder", (radius, half), TILT @ Vector((0, 0, lift)), axis=TILT @ Vector((0, 0, 1)))],
    }

# ---------------------------------------------------------------- tier 2: pots
SQUARE_POWER = 5.0
POT_SQUARE = [
    (0.0, 0.0),
    (0.0345, 0.0),
    (0.0360, 0.0015),
    (0.0445, 0.0570),
    (0.0468, 0.0575),
    (0.0480, 0.0590),
    (0.0480, 0.0715),
    (0.0475, 0.0738),
    (0.0462, 0.0750),
    (0.0448, 0.0750),
    (0.0438, 0.0738),
    (0.0436, 0.0700),
    (0.0436, 0.0640),
    (0.0, 0.0640),
]
pot_square = build("sit_pot_square", Geo().add(*squircle_lathe(POT_SQUARE, 64, SQUARE_POWER)))


def soil(radius, power=None, segments=32):
    """The lumpy soil of plant_soil, to another radius (and, with power, square with rounded corners)."""
    k = radius / 0.0448
    verts, rings = [(0.0, 0.0, 0.0685)], []
    for r0 in (0.006, 0.012, 0.019, 0.026, 0.033, 0.039, 0.0432, 0.0448):
        ring = []
        for i in range(segments):
            a = TAU * (i + 0.5 * (len(rings) % 2)) / segments
            r = r0 * k * (squircle(a, power) if power else 1.0)
            x, y = r * math.cos(a), r * math.sin(a)
            lump = noise.noise(Vector((x * 85, y * 85, 3.3))) + 0.5 * noise.noise(Vector((x * 210, y * 210, 6.2)))
            z = 0.0682 + 0.0019 * lump * min(1.0, r0 / 0.012)
            if r0 > 0.040:
                z = min(z, 0.0672) if r0 < 0.044 else 0.0660
            ring.append(len(verts))
            verts.append((x, y, max(0.0660, min(0.0700, z))))
        rings.append(ring)
    skirt = [len(verts) + i for i in range(segments)]  # hidden inside the pot wall
    verts += [(verts[i][0], verts[i][1], 0.0630) for i in rings[-1]]
    rings.append(skirt)
    faces = [(0, rings[0][i], rings[0][(i + 1) % segments]) for i in range(segments)]
    for a, b in zip(rings, rings[1:], strict=False):
        faces += [(a[i], a[(i + 1) % segments], b[(i + 1) % segments], b[i]) for i in range(segments)]
    faces.append(tuple(reversed(rings[-1])))
    return verts, faces


soil_square = build("sit_soil_square", Geo().add(*soil(0.0444, SQUARE_POWER, 48)), sharp_deg=50)
POT_GLAZED = [
    (0.0, 0.0),
    (0.0300, 0.0),
    (0.0330, 0.0020),
    (0.0450, 0.0200),
    (0.0510, 0.0400),
    (0.0500, 0.0560),
    (0.0472, 0.0690),
    (0.0482, 0.0725),
    (0.0476, 0.0745),
    (0.0462, 0.0750),
    (0.0450, 0.0742),
    (0.0448, 0.0700),
    (0.0448, 0.0640),
    (0.0, 0.0640),
]
pot_glazed = build("sit_pot_glazed", Geo().add(*lathe(POT_GLAZED, 48)), sharp_deg=30)
ASSETS["pot_square"] = {
    "meshes": ["sit_pot_square", "sit_soil_square"],
    "origin": "bottom centre: the frame of plant_pot and plant_soil",
    "attach": "replace plant_pot and plant_soil in the geoms pot_vis and soil_vis; the stem stands at z = 0.068 as "
    "before. Collision in place of pot_col0, pot_col1 and the rim blocks.",
    "collision": [
        prim("box", (0.040, 0.040, 0.0285), (0, 0, 0.0285)),
        prim("box", (0.0475, 0.0475, 0.0058), (0, 0, 0.0628)),
        *(
            prim("box", (0.0022, 0.048, 0.0033), (0.0458 * math.cos(a), 0.0458 * math.sin(a), 0.0717), yaw=a)
            for a in (0, TAU / 4, TAU / 2, 3 * TAU / 4)
        ),
    ],
}
ASSETS["pot_glazed"] = {
    "meshes": ["sit_pot_glazed"],
    "origin": "bottom centre: the frame of plant_pot",
    "attach": "replaces plant_pot in the geom pot_vis; plant_soil stays (inner radius 44.8 mm). The plant's "
    "existing collision (pot_col0, pot_col1, rim blocks) fits to within 4 mm; the belly is 51 mm at z = 0.040.",
    "collision": [prim("cylinder", (0.046, 0.020), (0, 0, 0.020)), prim("cylinder", (0.051, 0.0143), (0, 0, 0.0543))],
}

# ---------------------------------------------------------------- tier 2: large leaf
BIG_X0, BIG_L, BIG_HALF_W = 0.030, 0.130, 0.036
big_grid = []
for i in range(15):
    t = (i / 14) ** 1.15
    w = max(0.0008, BIG_HALF_W * max(0.0, math.sin(math.pi * t**0.55)) ** 0.75)
    arch = 0.016 * math.sin(0.9 * math.pi * t) - 0.022 * t * t
    row = []
    for j in range(9):
        s = -1 + j / 4
        y = s * w
        z = arch + 0.22 * abs(y) - 0.10 * w * s**4 + 0.0016 * math.sin(13 * t + 2.5 * s) * abs(s)
        row.append((BIG_X0 + BIG_L * t, y, z))
    big_grid.append(row)
big_geo = Geo().add(*sheet(big_grid, 0.0008))
big_geo.add(*petiole([(0.0, 0, 0.0), (0.012, 0, 0.0004), (0.024, 0, 0.0002), (0.034, 0, -0.0004)], 0.0022, 0.0015))
leaf_large = build("sit_leaf_large", big_geo, sharp_deg=45)
ASSETS["leaf_large"] = {
    "meshes": ["sit_leaf_large"],
    "origin": "as plant_leaf: petiole base, blade toward +X, upper side +Z (no mesh scale)",
    "attach": "petiole base on the main stem or on a shoot, the blade between the camera and what it hides. For "
    "a collision shape that the arm can push: a thin box under the blade.",
    "collision": [prim("box", (0.060, 0.030, 0.002), (0.095, 0, 0.012))],
}

# ---------------------------------------------------------------- tier 2: bin
BIN = [
    (0.0, 0.0),
    (0.0480, 0.0),
    (0.0500, 0.0020),
    (0.0585, 0.1040),
    (0.0612, 0.1050),
    (0.0620, 0.1075),
    (0.0610, 0.1100),
    (0.0575, 0.1100),
    (0.0565, 0.1060),
    (0.0478, 0.0060),
    (0.0, 0.0060),
]
bin_mesh = build("sit_bin", Geo().add(*squircle_lathe(BIN, 64, 6.0)))
ASSETS["bin"] = {
    "meshes": ["sit_bin"],
    "origin": "bottom centre; the sides face +/-X and +/-Y",
    "attach": "stands on the table at z = 0. Hollow: the collision is a floor and four walls, so that cuttings "
    "can be dropped in (inside about 96 mm square at the floor, 113 mm at the rim). 'slots': a frame on the +X "
    "side for a symbol (sit_symbol_leaf, mesh scale 2.5).",
    "colours": {k: list(v) for k, v in BIN_COLOURS.items()},
    "slots": {"front": frame((0.0548, 0, 0.060), Matrix(((0, 0, 1), (1, 0, 0), (0, 1, 0))))},
    "mass": 0.15,
    "collision": [
        prim("box", (0.049, 0.049, 0.003), (0, 0, 0.003)),
        *(
            prim("box", (0.0025, 0.057, 0.052), (0.0545 * math.cos(a), 0.0545 * math.sin(a), 0.058), yaw=a)
            for a in (0, TAU / 4, TAU / 2, 3 * TAU / 4)
        ),
    ],
}


# ---------------------------------------------------------------- textures
def blur(a, k, axis):
    """Box blur over 2k samples along an axis (wraps round)."""
    n = a.shape[axis]
    c = np.cumsum(
        np.concatenate([a.take(range(n - k, n), axis=axis), a, a.take(range(k), axis=axis)], axis=axis), axis=axis
    )
    return (c.take(range(2 * k, 2 * k + n), axis=axis) - c.take(range(n), axis=axis)) / (2 * k)


def wood_texture(stained):
    """Eight planks along u (1024 x 512): v = 0..1 crosses the planks."""
    rs = np.random.default_rng(11)
    w, h = 1024, 512
    ph = h // PLANKS
    u, v = np.arange(w)[None, :], np.arange(ph)[:, None]
    img = np.zeros((h, w, 3))
    for p in range(PLANKS):
        tint = (
            np.array([0.62, 0.45, 0.28])
            * rs.uniform(0.86, 1.08)
            * np.array([1.0, rs.uniform(0.96, 1.03), rs.uniform(0.90, 1.05)])
        )
        phase = rs.uniform(0, TAU)
        wave = 4 * np.sin(u / w * TAU * rs.uniform(0.6, 1.6) + phase) + 1.5 * np.sin(u / w * TAU * 5 + phase)
        rings = np.sin((v + wave) * rs.uniform(0.35, 0.6) + 2.0 * np.sin((v + wave) * 0.11 + phase))
        shade = 1 + 0.07 * rings + 0.22 * blur(rs.normal(0, 1, (ph, w)), 24, 1)
        for _ in range(int(rs.integers(0, 3))):  # knots
            r = np.sqrt(((u - rs.uniform(0, w)) / 22) ** 2 + ((v - rs.uniform(0.2, 0.8) * ph) / 8) ** 2)
            shade = shade * (1 - 0.45 * np.exp(-r * r) + 0.06 * np.sin(r * 5) * np.exp(-r / 2.5))
        edge = np.minimum(v, ph - 1 - v)
        shade = shade * np.where(edge < 1, 0.55, np.where(edge < 2, 0.85, 1.0))
        img[p * ph : (p + 1) * ph] = tint[None, None, :] * shade[:, :, None]
    if stained:
        yy, xx = np.mgrid[0:h, 0:w]
        stain = np.zeros((h, w))
        for _ in range(26):
            r = rs.uniform(18, 80)
            stain += rs.uniform(0.25, 0.7) * np.exp(
                -(((xx - rs.uniform(0, w)) / r) ** 2 + ((yy - rs.uniform(0, h)) / (r * rs.uniform(0.5, 1.0))) ** 2)
            )
        specks = blur(blur((rs.random((h, w)) > 0.985).astype(float), 2, 0), 2, 1) * 6
        k = np.clip(stain + specks, 0, 0.85)[:, :, None]
        img = img * (1 - k) + np.array([0.16, 0.11, 0.07]) * k
    return np.clip(img, 0, 1)


def greenhouse_texture():
    """What is seen through the glass (768 x 512): sky over blurred foliage; v = 0 is the bottom."""
    rs = np.random.default_rng(23)
    w, h = 768, 512
    yy, xx = np.mgrid[0:h, 0:w]
    v = (yy / (h - 1))[:, :, None]
    img = np.array([0.88, 0.93, 0.90]) * (1 - v) + np.array([0.60, 0.77, 0.92]) * v
    mask = np.clip((0.40 - yy / h) * 5, 0, 1)
    shade = np.zeros((h, w))
    for _ in range(60):
        r = rs.uniform(25, 85)
        blob = np.exp(
            -(((xx - rs.uniform(0, w)) / r) ** 2 + ((yy - rs.uniform(0, 0.62) * h) / (r * rs.uniform(0.7, 1.3))) ** 2)
        )
        mask = np.maximum(mask, 0.95 * blob)
        shade += rs.uniform(-0.3, 0.3) * blob
    mask = blur(blur(mask, 8, 0), 8, 1)[:, :, None]
    green = np.array([0.30, 0.48, 0.24]) * (1 + np.clip(shade, -0.45, 0.45))[:, :, None]
    return np.clip(img * (1 - mask) + green * mask, 0, 1)


IMAGES = {}
for name, pixels in (
    ("sit_bench_wood", wood_texture(False)),
    ("sit_bench_stained", wood_texture(True)),
    ("sit_backdrop_greenhouse", greenhouse_texture()),
):
    h, w = pixels.shape[:2]
    image = bpy.data.images.new(name, w, h, alpha=False)
    image.pixels.foreach_set(np.concatenate([pixels, np.ones((h, w, 1))], axis=2).astype(np.float32).ravel())
    image.filepath_raw = os.path.join(MESH_DIR, f"{name}.png")
    image.file_format = "PNG"
    image.save()
    IMAGES[name] = image

# ---------------------------------------------------------------- export
MESHES = {}
for obj in PARTS:
    activate(obj)
    has_uv = bool(obj.data.uv_layers)
    bpy.ops.wm.obj_export(
        filepath=os.path.join(MESH_DIR, f"{obj.name}.obj"),
        export_selected_objects=True,
        forward_axis="Y",
        up_axis="Z",
        export_materials=False,
        export_normals=True,
        export_uv=has_uv,
        export_triangulated_mesh=True,
    )
    co = np.array([v.co for v in obj.data.vertices])
    MESHES[obj.name] = {
        "file": f"{obj.name}.obj",
        "triangles": len(obj.data.polygons),
        "min": [round(float(c), 4) for c in co.min(axis=0)],
        "max": [round(float(c), 4) for c in co.max(axis=0)],
        "rgba": list(RGBA[obj.name]),
        "uv": has_uv,
    }
    print(f"{obj.name}: {len(obj.data.polygons)} triangles, size {tuple(round(d, 4) for d in obj.dimensions)}")
MANIFEST = {
    "about": "Assets of the test situations, written by build_situations_blender.py. Metres, Z up. Collision "
    "primitives as MuJoCo takes them (box: half sizes; cylinder, capsule: radius, half length), with pos and quat "
    "(w, x, y, z) in the asset's frame. rgba: suggested colours. Files are relative to this folder. Poses are "
    "those of the OBJ frame: MuJoCo's compiler moves a mesh to its centre of mass and principal axes, so at run "
    "time geom_pos = p + R @ mesh_pos and geom_quat = R * mesh_quat. Textures: set colorspace to linear in MuJoCo "
    "(the PNGs' sRGB tag otherwise darkens them).",
    "meshes": MESHES,
    "textures": {
        "sit_bench_wood.png": {"size": [1024, 512], "mesh": "sit_bench_top"},
        "sit_bench_stained.png": {"size": [1024, 512], "mesh": "sit_bench_top"},
        "sit_backdrop_greenhouse.png": {"size": [768, 512], "mesh": "sit_backdrop_glass"},
    },
    "assets": ASSETS,
}
with open(os.path.join(MESH_DIR, "manifest.json"), "w") as f:
    json.dump(MANIFEST, f, indent=1)
    f.write("\n")

# ---------------------------------------------------------------- previews
for obj in PARTS:
    obj.hide_render = obj.hide_viewport = True  # the origin copies; the previews place instances


def place(obj, matrix, rgba=None, mat=None):
    dup = obj.copy()
    dup.name = f"preview_{obj.name}"
    bpy.context.scene.collection.objects.link(dup)
    dup.hide_render = dup.hide_viewport = False
    dup.matrix_world = matrix
    dup.material_slots[0].link = "OBJECT"
    dup.material_slots[0].material = mat or paint(rgba or RGBA.get(obj.name, (0.6, 0.6, 0.6, 1)))
    return dup


def reference(name):
    """An existing part of the plant, read back from its OBJ (None if it has not been built)."""
    path = os.path.join(OUT, "meshes", f"{name}.obj")
    if not os.path.exists(path):
        return None
    bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    obj = bpy.context.selected_objects[0]
    obj.name = f"reference_{name}"
    obj.data.materials.clear()
    obj.data.materials.append(paint((0.6, 0.6, 0.6, 1)))
    obj.hide_render = obj.hide_viewport = True
    return obj


def at(x=0.0, y=0.0, z=0.0, yaw=0.0):
    return Matrix.Translation((x, y, z)) @ Matrix.Rotation(math.radians(yaw), 4, "Z")


def shoot_frame(origin, azimuth, elevation):
    """env._shoot_frame: z along the shoot, y horizontal, x upward."""
    a, e = math.radians(azimuth), math.radians(elevation)
    d = Vector((math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)))
    y = d.cross(Vector((0, 0, 1))).normalized()
    return Matrix.Translation(origin) @ Matrix((y.cross(d), y, d)).transposed().to_4x4()


LEAF_BASE = Matrix(((0, 0, 1), (0, -1, 0), (1, 0, 0))).to_4x4()  # env: the blade along the shoot, face up


def leaf_pose(length, yaw, pitch=0.35):
    return (
        Matrix.Translation((0, 0, length - 0.001))
        @ LEAF_BASE
        @ Matrix.Rotation(yaw, 4, "Z")
        @ Matrix.Rotation(pitch, 4, "Y")
        @ Matrix.Scale(LEAF_SCALE, 4)
    )


REF = {
    name: reference(name)
    for name in (
        "plant_pot",
        "plant_soil",
        "plant_main_stem",
        "plant_flower_petals",
        "plant_flower_centre",
        "plant_flower_sepals",
        "plant_leaf",
        "plant_button",
        "plant_arrow_cw",
        "plant_arrow_ccw",
    )
}
shoot_obj = finish(new_object("preview_shoot", *tube([(0, 0, 0), (0, 0, 1)], SHOOT_R, 10)))
shoot_obj.hide_render = shoot_obj.hide_viewport = True
stalk_obj = finish(new_object("preview_stalk", *tube([(0, 0, 0), (0, 0, 1)], 0.004, 10)))
stalk_obj.hide_render = stalk_obj.hide_viewport = True


def shoot(F, length, rgba=STEM_GREEN):
    return place(shoot_obj, F @ Matrix.Diagonal((1, 1, length, 1)), rgba)


def ground(G, size=3.0, z=0.0, rgba=(0.62, 0.50, 0.36, 1)):
    bpy.ops.mesh.primitive_plane_add(size=size, location=G @ Vector((0, 0, z)))
    bpy.context.object.name = "preview_ground"
    bpy.context.object.data.materials.append(paint(rgba))


def plant(G, pot=None, soil_part=None, head=None, pot_rgba=TERRACOTTA):
    """The plant in its frame (the pot's bottom centre) at G; pot / soil / head: parts that replace the usual."""
    pot = pot or REF["plant_pot"]
    soil_part = soil_part or REF["plant_soil"]
    if not (pot and soil_part and REF["plant_main_stem"]):
        return
    place(pot, G, pot_rgba)
    place(soil_part, G, SOIL)
    place(REF["plant_main_stem"], G @ at(z=STEM_BASE), STEM_GREEN)
    top = G @ at(z=STEM_BASE + STEM_H)
    if head:
        for part in head:
            place(part, top)
    else:
        for part, rgba in (
            ("petals", (0.95, 0.42, 0.05, 1)),
            ("centre", (0.36, 0.20, 0.05, 1)),
            ("sepals", STEM_GREEN),
        ):
            place(REF[f"plant_flower_{part}"], top, rgba)


# A: the plant with ties, sick and wilted leaves and a bud, seen as the front camera sees it
GA = Matrix.Identity(4)
ground(GA)
plant(GA)
if REF["plant_leaf"]:
    for height, azimuth, colour, kind in (
        (0.075, -75, "red", "healthy"),
        (0.105, -150, "blue", "sick"),
        (0.135, 15, "yellow", "wilted"),
        (0.160, -20, "white", "bud"),
    ):
        F = GA @ shoot_frame((0, 0, STEM_BASE + height), azimuth, 25)
        shoot(F, 0.08)
        place(tie, F @ at(z=0.012), TIE_COLOURS[colour])
        for yaw in (-0.7, 0.7):
            pose = F @ leaf_pose(0.08, yaw)
            if kind == "wilted":
                place(wilted, pose)
                continue
            place(REF["plant_leaf"], pose, SICK_LEAF if kind == "sick" else LEAF_GREEN)
            if kind == "sick":
                place(spots, pose)
                place(edges, pose)
        if kind == "bud":
            place(bud_body, F @ at(z=0.08))
            place(bud_sepals, F @ at(z=0.08))

# B: the distractors on the bench, the greenhouse behind
GB = Matrix.Translation((4, 0, 0))
ground(GB, 6.0, -0.76, (0.30, 0.32, 0.36, 1))
WOOD = material("bench_wood", (0.6, 0.45, 0.28, 1), 0.7, IMAGES["sit_bench_wood"])
place(bench, GB @ at(z=-0.02, yaw=90), mat=WOOD)
place(backdrop_frame, GB @ at(y=0.62, z=-0.76))
place(
    backdrop_glass,
    GB @ at(y=0.62, z=-0.76),
    mat=material("glass", (0.7, 0.8, 0.8, 1), 0.2, IMAGES["sit_backdrop_greenhouse"]),
)
for parts, pose in (
    ((can, can_trim), at(-0.33, 0.10, yaw=-20)),
    ((gloves, gloves_cuffs), at(0.30, -0.02, yaw=25)),
    ((trowel_blade, trowel_handle), at(-0.08, -0.20, yaw=-12)),
    ((packet, packet_art), at(0.06, 0.13, yaw=15)),
    ((pot_small,), at(-0.12, 0.02)),
    ((bee_body, bee_dark, bee_wings), at(0.33, -0.01, 0.0265, yaw=200)),
):
    for part in parts:
        place(part, GB @ pose)

# C: tier 2
GC = Matrix.Translation((8, 0, 0))
ground(GC)
for x, head in (
    (-0.32, (rose_petals, rose_sepals)),
    (-0.22, (daisy_petals, daisy_centre, daisy_sepals)),
    (-0.12, (tulip_petals, tulip_sepals)),
):
    place(stalk_obj, GC @ at(x, 0.14) @ Matrix.Diagonal((1, 1, 0.10, 1)), STEM_GREEN)
    for part in head:
        place(part, GC @ at(x, 0.14, 0.10, yaw=-90))
plant(GC @ at(0.04, 0.14), pot_square, soil_square, (rose_petals, rose_sepals))
plant(GC @ at(0.20, 0.14), pot_glazed, None, (tulip_petals, tulip_sepals), RGBA["sit_pot_glazed"])
for x, colour, symbol in ((0.36, "compost", symbol_leaf), (0.36 + 0.15, "green", symbol_sun)):
    B = GC @ at(x, 0.05, yaw=-90)
    place(bin_mesh, B, BIN_COLOURS[colour])
    slot = ASSETS["bin"]["slots"]["front"]
    place(
        symbol,
        B @ Matrix.Translation(slot["pos"]) @ Quaternion(slot["quat"]).to_matrix().to_4x4() @ Matrix.Scale(2.5, 4),
        (0.95, 0.95, 0.93, 1),
    )
CARD = GC @ at(-0.30, -0.08, yaw=-90)
place(legend_card, CARD)
place(legend_marks, CARD)
for key, slot in ASSETS["legend_card"]["slots"].items():
    row, kind = key.split("_")
    part = (
        (symbol_leaf, symbol_sun)[row == "row1"]
        if kind == "symbol"
        else REF[("plant_arrow_cw", "plant_arrow_ccw")[row == "row1"]]
    )
    if part:
        pose = (
            CARD
            @ Matrix.Translation(slot["pos"])
            @ Quaternion(slot["quat"]).to_matrix().to_4x4()
            @ Matrix.Scale(1.4, 4)
        )
        place(part, pose, RGBA.get(part.name, (0.15, 0.15, 0.16, 1)))
if REF["plant_button"]:
    for x, rgba, symbol in ((-0.17, (0.10, 0.32, 0.88, 1), symbol_leaf), (-0.12, (0.95, 0.48, 0.06, 1), symbol_sun)):
        place(REF["plant_button"], GC @ at(x, -0.08), rgba)
        place(symbol, GC @ at(x, -0.08, 0.0094), (0.95, 0.95, 0.93, 1))
place(leaf_large, GC @ at(-0.02, -0.10, 0.004, yaw=20))

# D: close-ups of the small parts (D1: tie, bud, bee; D2: the leaves)
GD = Matrix.Translation((12, 0, 0))
ground(GD, 1.0)
for x, colour in ((-0.075, "red"), (-0.035, "white")):
    F = GD @ shoot_frame((x, 0, 0.012), 0, 35)
    shoot(F, 0.045)
    place(tie, F @ at(z=0.020), TIE_COLOURS[colour])
F = GD @ shoot_frame((0.012, 0, 0.012), 0, 35)
shoot(F, 0.03)
place(bud_body, F @ at(z=0.03))
place(bud_sepals, F @ at(z=0.03))
for part in (bee_body, bee_dark, bee_wings):
    place(part, GD @ at(0.075, 0, yaw=-30))
GD2 = Matrix.Translation((12, 1, 0))
ground(GD2, 1.0)
if REF["plant_leaf"]:
    for x, y, layers in (
        (-0.09, 0.03, ()),
        (-0.02, 0.03, (spots,)),
        (0.05, 0.03, (edges,)),
        (-0.09, -0.03, (spots, edges)),
    ):
        pose = GD2 @ at(x, y, 0.004)
        place(REF["plant_leaf"], pose, SICK_LEAF if layers else LEAF_GREEN)
        for layer in layers:
            place(layer, pose)
    place(wilted, GD2 @ at(-0.02, -0.03, 0.040))
    place(wilted, GD2 @ at(0.05, -0.03, 0.040, yaw=-50))

scene = bpy.context.scene
bpy.ops.object.light_add(type="SUN", location=(-0.5, -0.8, 1.5))
sun = bpy.context.object
sun.name = "preview_sun"
sun.data.energy, sun.data.angle = 2.4, math.radians(8)
sun.rotation_euler = (math.radians(40), math.radians(-10), math.radians(-25))
world = scene.world or bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
background = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
background.inputs[0].default_value = (0.80, 0.84, 0.90, 1)
background.inputs[1].default_value = 0.6
with contextlib.suppress(TypeError):
    scene.view_settings.view_transform = "Standard"


def camera(name, G, location, target, fov=30.0):
    location, target = G @ Vector(location), G @ Vector(target)
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.object
    cam.name = name
    cam.rotation_euler = (target - location).to_track_quat("-Z", "Y").to_euler()
    cam.data.sensor_fit = "VERTICAL"
    cam.data.angle_y, cam.data.clip_start = math.radians(fov), 0.01
    return cam


VIEWS = [
    # the stand's frame is the plate's bottom, 0.042 below the pot; the camera of env.py relative to the stand
    (
        "situations_plant_front_camera.png",
        camera("cam_plant_env", GA, (0.60, -0.55, 0.578), (-0.13, 0.0, 0.058), 45),
        (960, 540),
    ),
    (
        "situations_plant_close.png",
        camera("cam_plant_close", GA, (0.16, -0.42, 0.30), (0.0, 0.0, 0.17), 34),
        (1100, 1100),
    ),
    ("situations_bench.png", camera("cam_bench", GB, (0.15, -1.25, 0.62), (0.0, 0.0, 0.03), 40), (1200, 800)),
    ("situations_backdrop.png", camera("cam_backdrop", GB, (0.9, -2.3, 0.75), (0.0, 0.3, 0.0), 50), (1200, 800)),
    ("situations_tier2.png", camera("cam_tier2", GC, (0.08, -0.95, 0.52), (0.08, 0.02, 0.07), 42), (1400, 800)),
    (
        "situations_close_tie_bud_bee.png",
        camera("cam_small", GD, (0.0, -0.26, 0.14), (0.0, 0.0, 0.022), 32),
        (1200, 700),
    ),
    (
        "situations_close_leaves.png",
        camera("cam_leaves", GD2, (-0.02, -0.20, 0.22), (-0.02, 0.0, 0.01), 40),
        (1200, 800),
    ),
]
scene.camera = VIEWS[1][1]
if PREVIEW_DIR:
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    with contextlib.suppress(TypeError):
        scene.render.image_settings.file_format = "PNG"
    for filename, cam, (width, height) in VIEWS:
        scene.camera = cam
        scene.render.resolution_x, scene.render.resolution_y = width, height
        scene.render.filepath = os.path.join(PREVIEW_DIR, filename)
        bpy.ops.render.render(write_still=True)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "situations_assets.blend"))
