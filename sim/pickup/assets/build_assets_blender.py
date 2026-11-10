"""Rebuild the pickup-sim meshes in Blender.

Run inside Blender (through the Blender MCP, or
`blender --background --python sim/pickup/assets/build_assets_blender.py`).

Writes one OBJ per part to sim/pickup/assets/meshes/ (Z up, metres) and saves pickup_assets.blend.
Each OBJ is exported with the object at the world origin, so the mesh frame is the MuJoCo
body/geom frame used in sim/pickup/env.py:

  cube       45 mm beveled cube, origin at centre
  can_body   46 mm x 76 mm cylinder; can_lid / can_base are 4 mm steel discs placed at z = +/-40 mm
  hex        hex nut, 46 mm across flats, 40 mm tall, 18 mm bore, origin at centre
  tblock     30x30x45 mm stem (origin at stem centre) with a 70x30x15 mm cap on top
  table_top  0.8 x 1.2 x 0.04 m slab, origin at centre
  tray       0.2 x 0.2 x 0.025 m shallow tray (not used by the current tasks)

Colours are assigned in env.py (MuJoCo rgba), not taken from the Blender materials.
"""

import math
import os

import bpy
from mathutils import Vector

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT, "meshes"), exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
for m in list(bpy.data.meshes):
    bpy.data.meshes.remove(m)


def material(name, rgba, rough=0.5, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    return m


def finish(obj, mat, bevel=None, segments=3):
    obj.data.materials.append(mat)
    bpy.context.view_layer.objects.active = obj
    if bevel:
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width, mod.segments, mod.limit_method = bevel, segments, "ANGLE"
    obj.modifiers.new("tri", "TRIANGULATE")
    for mod in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=mod.name)
    return obj


def box(name, size, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def cylinder(name, radius, depth, vertices=48, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=location)
    obj = bpy.context.object
    obj.name = name
    return obj


def boolean_cut(obj, cutter):
    mod = obj.modifiers.new("cut", "BOOLEAN")
    mod.object, mod.operation = cutter, "DIFFERENCE"
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="cut")
    bpy.data.objects.remove(cutter)


M = dict(
    red=material("red", (0.80, 0.08, 0.06, 1), 0.45),
    green=material("green", (0.10, 0.55, 0.18, 1), 0.35, 0.3),
    blue=material("blue", (0.08, 0.22, 0.80, 1), 0.4),
    yellow=material("yellow", (0.95, 0.72, 0.05, 1), 0.5),
    steel=material("steel", (0.75, 0.76, 0.78, 1), 0.25, 0.9),
    wood=material("wood", (0.62, 0.45, 0.28, 1), 0.7),
    tray=material("tray", (0.85, 0.85, 0.82, 1), 0.6),
)

finish(box("cube", (0.045, 0.045, 0.045)), M["red"], bevel=0.004)
finish(cylinder("can_body", 0.023, 0.076), M["green"], bevel=0.002, segments=2)
finish(cylinder("can_lid", 0.0235, 0.004), M["steel"], bevel=0.001, segments=2)
finish(cylinder("can_base", 0.0235, 0.004), M["steel"], bevel=0.001, segments=2)

hex_nut = cylinder("hex", 0.023 / math.cos(math.pi / 6), 0.040, vertices=6)
boolean_cut(hex_nut, cylinder("hex_bore", 0.009, 0.06, vertices=32))
finish(hex_nut, M["blue"], bevel=0.003)

stem = box("tblock", (0.030, 0.030, 0.045))
cap = box("tblock_cap", (0.070, 0.030, 0.015), location=(0, 0, 0.030))
bpy.ops.object.select_all(action="DESELECT")
cap.select_set(True)
stem.select_set(True)
bpy.context.view_layer.objects.active = stem
bpy.ops.object.join()
finish(stem, M["yellow"], bevel=0.003)

finish(box("table_top", (0.8, 1.2, 0.04)), M["wood"], bevel=0.006)
tray = box("tray", (0.20, 0.20, 0.025))
boolean_cut(tray, box("tray_cut", (0.184, 0.184, 0.025), location=(0, 0, 0.006)))
finish(tray, M["tray"], bevel=0.002, segments=2)

for obj in [o for o in bpy.data.objects if o.type == "MESH"]:
    obj.location = (0, 0, 0)
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=os.path.join(OUT, "meshes", f"{obj.name}.obj"),
        export_selected_objects=True,
        forward_axis="Y",
        up_axis="Z",
        export_materials=False,
        export_normals=True,
        export_uv=False,
    )

# Lay the parts out on the table for a readable .blend.
layout = dict(
    table_top=(0.35, 0.0, -0.02),
    cube=(0.45, -0.24, 0.0225),
    can_body=(0.45, -0.08, 0.042),
    can_lid=(0.45, -0.08, 0.082),
    can_base=(0.45, -0.08, 0.002),
    hex=(0.45, 0.08, 0.020),
    tblock=(0.45, 0.24, 0.0225),
    tray=(0.62, 0.0, 0.0125),
)
for name, loc in layout.items():
    bpy.data.objects[name].location = Vector(loc)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "pickup_assets.blend"))
