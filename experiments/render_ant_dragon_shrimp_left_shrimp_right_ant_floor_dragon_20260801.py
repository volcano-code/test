#!/usr/bin/env python3
"""Render the locked StructShadow case in the requested gallery receiver order.

Requested display semantics:
  left wall  = shrimp (source Z)
  right wall = ant    (source X)
  floor      = dragon (source Y)

This renderer is isolated from the accepted four-case assets and does not
modify the existing centered render.
"""

from __future__ import annotations

import json
from math import atan
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
GALLERY = ROOT / "paper_aaai_submission/supplement_gallery_one_case_20260801"
BASE_BLEND = GALLERY / "parrot_snail/renders/parrot_snail_gallery_evidence.blend"
STRUCT_CASE = Path(
    "/data/xvxinyao/xvxinyao/StructShadow/print_ready_stls_diversity/by_group/"
    "n3_g5_ant_dragon_shrimp"
)
MESH = STRUCT_CASE / "g5_bugs_ant_x_dragon_sea_shrimp_MINIMAL.stl"
OUT = (
    GALLERY
    / "ant_dragon_shrimp"
    / "renders_left_shrimp_right_ant_floor_dragon"
)
MODEL_DISPLAY_EXTENT = 46.0
PROJECTOR_RECEIVER_RADIUS = 44.0

# Use orientation 20 from the locked 24-orientation audit verbatim.
BASIS = Matrix(
    (
        (0.0, 0.0, -1.0),
        (0.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0),
    )
)


def bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    low = Vector(tuple(min(point[i] for point in corners) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in corners) for i in range(3)))
    return low, high


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def replace_model() -> bpy.types.Object:
    old = bpy.data.objects.get("PCRAFT_parrot_snail")
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)

    bpy.ops.wm.stl_import(filepath=str(MESH))
    imported = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not imported:
        raise RuntimeError(f"No mesh imported from {MESH}")
    if len(imported) > 1:
        bpy.context.view_layer.objects.active = imported[0]
        bpy.ops.object.join()
    model = bpy.context.view_layer.objects.active
    model.name = "GalleryEvidenceModel"
    model.rotation_euler = BASIS.to_euler()
    bpy.context.view_layer.objects.active = model
    model.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    low, high = bounds(model)
    display_scale = MODEL_DISPLAY_EXTENT / max(high - low)
    model.scale = (display_scale, display_scale, display_scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    low, high = bounds(model)
    model.location.x -= (low.x + high.x) * 0.5
    model.location.y -= (low.y + high.y) * 0.5
    model.location.z -= low.z
    bpy.context.view_layer.update()

    model.data.materials.clear()
    material = bpy.data.materials.get("printed model")
    if material is None:
        raise RuntimeError("Approved template is missing material 'printed model'")
    model.data.materials.append(material)
    return model


def projected_centroid(model: bpy.types.Object, axis: int) -> Vector:
    normal_matrix = model.matrix_world.to_3x3().inverted().transposed()
    weighted = Vector((0.0, 0.0, 0.0))
    total = 0.0
    for polygon in model.data.polygons:
        normal = normal_matrix @ polygon.normal
        weight = polygon.area * abs(normal[axis])
        if weight <= 1.0e-12:
            continue
        weighted += (model.matrix_world @ polygon.center) * weight
        total += weight
    if total > 1.0e-12:
        return weighted / total
    low, high = bounds(model)
    return (low + high) * 0.5


def configure_projector(
    light: bpy.types.Object,
    source: Vector,
    receiver_center: Vector,
    receiver_radius: float = PROJECTOR_RECEIVER_RADIUS,
) -> None:
    light.location = source
    distance = (receiver_center - source).length
    light.data.energy = 165_000_000.0
    light.data.spot_size = 2.0 * atan(receiver_radius / distance)
    light.data.spot_blend = 0.64
    light.data.shadow_soft_size = 0.12
    look_at(light, receiver_center)


def configure_projectors(model: bpy.types.Object) -> dict[str, object]:
    left = bpy.data.objects.get("left projection light")
    right = bpy.data.objects.get("right projection light")
    if left is None or right is None:
        raise RuntimeError("Approved template is missing its wall projectors")

    center_x = projected_centroid(model, 0)
    center_y = projected_centroid(model, 1)
    configure_projector(
        left,
        Vector((-2000.0, center_x.y, center_x.z)),
        Vector((62.0, center_x.y, center_x.z)),
    )
    configure_projector(
        right,
        Vector((center_y.x, -2000.0, center_y.z)),
        Vector((center_y.x, 62.0, center_y.z)),
    )

    old_floor = bpy.data.objects.get("floor projection light")
    if old_floor is not None:
        bpy.data.objects.remove(old_floor, do_unlink=True)
    center_z = projected_centroid(model, 2)
    bpy.ops.object.light_add(type="SPOT", location=(center_z.x, center_z.y, 2000.0))
    floor = bpy.context.object
    floor.name = "floor projection light"
    floor.data.color = (0.90, 0.96, 1.0)
    configure_projector(
        floor,
        Vector((center_z.x, center_z.y, 2000.0)),
        Vector((center_z.x, center_z.y, -46.0)),
    )
    return {
        "requested_receiver_order": {
            "left_wall": "shrimp (source Z)",
            "right_wall": "ant (source X)",
            "floor": "dragon (source Y)",
        },
        "left_wall": [62.0, center_x.y, center_x.z],
        "right_wall": [center_y.x, 62.0, center_y.z],
        "floor": [center_z.x, center_z.y, -46.0],
    }


def main() -> None:
    if not BASE_BLEND.is_file():
        raise FileNotFoundError(BASE_BLEND)
    if not MESH.is_file():
        raise FileNotFoundError(MESH)
    OUT.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(BASE_BLEND))
    model = replace_model()
    centers = configure_projectors(model)

    camera = bpy.data.objects.get("SimulationCamera")
    if camera is None:
        raise RuntimeError("Approved template is missing SimulationCamera")
    camera.data.ortho_scale = 280.0
    look_at(camera, Vector((10.0, 10.0, -5.0)))

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.filepath = str(OUT / "simulation_left_shrimp_right_ant_floor_dragon.png")

    bpy.ops.wm.save_as_mainfile(
        filepath=str(OUT / "ant_dragon_shrimp_left_shrimp_right_ant_floor_dragon.blend")
    )
    bpy.ops.render.render(write_still=True)
    (OUT / "projection_centers.json").write_text(
        json.dumps(centers, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(scene.render.filepath)


if __name__ == "__main__":
    main()
