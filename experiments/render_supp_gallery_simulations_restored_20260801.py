#!/usr/bin/env python3
"""Restore the approved dark-corner gallery simulation style.

The accepted parrot/snail Blender scene is used as the immutable visual
template.  Each case only replaces the evidence STL, applies a rigid protocol
axis transform, and adjusts the camera framing.  No silhouette textures are
composited onto the receivers.
"""

from __future__ import annotations

from math import atan
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
GALLERY = ROOT / "paper_aaai_submission/supplement_gallery_one_case_20260801"
BASE_BLEND = GALLERY / "parrot_snail/renders/parrot_snail_gallery_evidence.blend"


CASES = (
    {
        "name": "dark_mage_samurai",
        "mesh": ROOT
        / "results/anime_two_wall_fixed_no_floor_physical_axes_qualityfirst_10cm_256_20260710"
        / "dark_mage_left__samurai_right"
        / "dark_mage_left__samurai_right_upright_saving_10cm_256.stl",
        "out": GALLERY / "dark_mage_samurai/renders",
        # Native Y -> left receiver; native X -> right receiver.
        "basis": Matrix(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        "floor_shadow": False,
    },
    {
        "name": "parrot_snail",
        "mesh": ROOT
        / "results/full_pipeline_batch_2v3v_20260730/2v_simple_parrot_snail"
        / "selected_display_fixed.stl",
        "out": GALLERY / "parrot_snail/renders",
        # Restore the display STL to the protocol axes: native Y projects to
        # the left (+X) receiver and native X to the right (+Y) receiver.
        "basis": Matrix(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        "floor_shadow": False,
    },
    {
        "name": "diverse5_thick1_13_21_23",
        "mesh": ROOT
        / "results/anime30_diverse_fixed_50_60_10cm_256_20260707"
        / "diverse5_thick1_13_21_23"
        / "diverse5_thick1_13_21_23_carve_pf4_10cm_256.stl",
        "out": GALLERY / "diverse5_thick1_13_21_23/renders",
        # Approved fixed-TFS layout: native Y -> left wall, native Z -> right
        # wall, native X -> floor.  Both wall silhouettes remain upright.
        "basis": Matrix(((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0))),
        "floor_shadow": True,
    },
)


def bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    low = Vector(tuple(min(point[i] for point in corners) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in corners) for i in range(3)))
    return low, high


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def replace_model(mesh_path: Path, basis: Matrix) -> bpy.types.Object:
    old = bpy.data.objects.get("PCRAFT_parrot_snail")
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)

    bpy.ops.wm.stl_import(filepath=str(mesh_path))
    imported = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not imported:
        raise RuntimeError(f"No mesh imported from {mesh_path}")
    if len(imported) > 1:
        bpy.context.view_layer.objects.active = imported[0]
        bpy.ops.object.join()
    model = bpy.context.view_layer.objects.active
    model.name = "GalleryEvidenceModel"

    model.rotation_euler = basis.to_euler()
    model.scale = (0.32, 0.32, 0.32)
    bpy.context.view_layer.objects.active = model
    model.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    low, high = bounds(model)
    model.location.x -= (low.x + high.x) * 0.5
    model.location.y -= (low.y + high.y) * 0.5
    model.location.z -= low.z

    model.data.materials.clear()
    material = bpy.data.materials.get("printed model")
    if material is None:
        raise RuntimeError("Approved template is missing material 'printed model'")
    model.data.materials.append(material)
    return model


def configure_projector(
    light: bpy.types.Object,
    source: Vector,
    receiver_center: Vector,
    receiver_radius: float = 43.0,
) -> None:
    """Configure a distant spot as a bounded near-orthographic projector."""
    light.location = source
    distance = (receiver_center - source).length
    light.data.energy = 165_000_000.0
    light.data.spot_size = 2.0 * atan(receiver_radius / distance)
    light.data.spot_blend = 0.64
    light.data.shadow_soft_size = 0.12
    look_at(light, receiver_center)


def configure_wall_projectors(model: bpy.types.Object) -> None:
    low, high = bounds(model)
    center = (low + high) * 0.5
    left = bpy.data.objects.get("left projection light")
    right = bpy.data.objects.get("right projection light")
    if left is None or right is None:
        raise RuntimeError("Approved template is missing its wall projectors")

    # Long source-to-receiver distances keep the rays nearly parallel while
    # retaining the approved bounded light pools on the two receiver walls.
    configure_projector(
        left,
        Vector((-2000.0, center.y, center.z)),
        Vector((62.0, center.y, center.z)),
    )
    configure_projector(
        right,
        Vector((center.x, -2000.0, center.z)),
        Vector((center.x, 62.0, center.z)),
    )


def add_floor_projector(model: bpy.types.Object) -> None:
    low, high = bounds(model)
    center = (low + high) * 0.5
    bpy.ops.object.light_add(type="SPOT", location=(center.x, center.y, 2000.0))
    light = bpy.context.object
    light.name = "floor projection light"
    light.data.color = (0.90, 0.96, 1.0)
    configure_projector(
        light,
        Vector((center.x, center.y, 2000.0)),
        Vector((center.x, center.y, -46.0)),
    )


def render_case(case: dict) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(BASE_BLEND))
    model = replace_model(case["mesh"], case["basis"])
    configure_wall_projectors(model)
    if case["floor_shadow"]:
        add_floor_projector(model)

    # Retain the approved camera direction and dark-corner atmosphere.  The
    # slightly wider/lower framing exposes more floor and prevents the lower
    # silhouette extremities from visually merging at the crop boundary.
    camera = bpy.data.objects.get("SimulationCamera")
    if camera is None:
        raise RuntimeError("Approved template is missing SimulationCamera")
    if case["floor_shadow"]:
        camera.data.ortho_scale = 216.0
        look_at(camera, Vector((10.0, 10.0, -18.0)))
    else:
        camera.data.ortho_scale = 198.0
        look_at(camera, Vector((10.0, 10.0, -2.0)))

    scene = bpy.context.scene
    scene.camera = camera
    # Match the wide Simulation cell directly so the gallery never needs to
    # crop away receiver walls or the third-view floor projection.
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 808
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    out = case["out"]
    out.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out / "simulation_restored.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(out / f"{case['name']}_simulation_restored.blend"))
    bpy.ops.render.render(write_still=True)
    print(scene.render.filepath)


def main() -> None:
    if not BASE_BLEND.is_file():
        raise FileNotFoundError(BASE_BLEND)
    for case in CASES:
        if not case["mesh"].is_file():
            raise FileNotFoundError(case["mesh"])
        render_case(case)


if __name__ == "__main__":
    main()
