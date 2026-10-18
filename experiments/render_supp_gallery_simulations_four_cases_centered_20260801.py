#!/usr/bin/env python3
"""Render four supplementary-gallery simulations with centered real shadows.

The accepted parrot/snail Blender scene remains the visual template.  Each
spotlight axis passes through an area-weighted projection centroid of the
evidence mesh, so each physical shadow is centered in its light pool without
compositing or translating a silhouette image.
"""

from __future__ import annotations

import json
import os
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
ORIGINAL_STRUCT_DELIVERY = (
    STRUCT_CASE / "g5_bugs_ant_x_dragon_sea_shrimp_MINIMAL.stl"
)
MODEL_DISPLAY_EXTENT = 46.0
PROJECTOR_RECEIVER_RADIUS = 44.0


CASES = (
    {
        "name": "dark_mage_samurai",
        "mesh": ROOT
        / "results/anime_two_wall_fixed_no_floor_physical_axes_qualityfirst_10cm_256_20260710"
        / "dark_mage_left__samurai_right"
        / "dark_mage_left__samurai_right_upright_saving_10cm_256.stl",
        "basis": Matrix(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        "floor_shadow": False,
    },
    {
        "name": "parrot_snail",
        "mesh": ROOT
        / "results/full_pipeline_batch_2v3v_20260730/2v_simple_parrot_snail"
        / "selected_display_fixed.stl",
        "basis": Matrix(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        "floor_shadow": False,
    },
    {
        "name": "diverse5_thick1_13_21_23",
        "mesh": ROOT
        / "results/anime30_diverse_fixed_50_60_10cm_256_20260707"
        / "diverse5_thick1_13_21_23"
        / "diverse5_thick1_13_21_23_carve_pf4_10cm_256.stl",
        "basis": Matrix(((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0))),
        "floor_shadow": True,
    },
    {
        "name": "ant_dragon_shrimp",
        "mesh": ORIGINAL_STRUCT_DELIVERY,
        # Source protocol axes are X=ant, Y=dragon, Z=shrimp. Map them to
        # gallery receivers floor, left wall, and right wall, respectively.
        "basis": Matrix(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))),
        "floor_shadow": True,
    },
    {
        "name": "arcane_casters",
        "mesh": ROOT
        / "results/fixed_xyz_carve_anime_gallery_20260731"
        / "arcane_casters"
        / "selected_display.stl",
        # The fixed-XYZ evidence is already stored in the protocol frame:
        # X/Y are the two walls and Z is the floor receiver.
        "basis": Matrix.Identity(3),
        "floor_shadow": True,
    },
    {
        "name": "urban_allies",
        "mesh": ROOT
        / "results/fixed_xyz_carve_anime_gallery_3_final_20260731"
        / "urban_allies"
        / "selected_display.stl",
        # Independent final-STL registration maps target order to source Z/X/Y.
        # Map source Z to the left wall, source X to the right wall, and source Y
        # to the floor while preserving one physical model coordinate frame.
        "basis": Matrix(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
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


def replace_model(
    mesh_path: Path,
    basis: Matrix,
    display_extent: float = MODEL_DISPLAY_EXTENT,
) -> bpy.types.Object:
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
    bpy.context.view_layer.objects.active = model
    model.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Source STLs use different authoring units. Normalize the world-space
    # maximum extent so every evidence object has comparable visual weight.
    low, high = bounds(model)
    display_scale = display_extent / max(high - low)
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
    """Area-weighted centroid of the mesh silhouette for one projection axis."""
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
    energy: float = 165_000_000.0,
) -> None:
    light.location = source
    distance = (receiver_center - source).length
    light.data.energy = energy
    light.data.spot_size = 2.0 * atan(receiver_radius / distance)
    light.data.spot_blend = 0.64
    light.data.shadow_soft_size = 0.12
    look_at(light, receiver_center)


def configure_projectors(
    model: bpy.types.Object,
    include_floor: bool,
    receiver_radius: float = PROJECTOR_RECEIVER_RADIUS,
    wall_source_distance: float = 2000.0,
    floor_source_height: float = 2000.0,
    light_energy: float = 165_000_000.0,
) -> dict[str, list[float]]:
    left = bpy.data.objects.get("left projection light")
    right = bpy.data.objects.get("right projection light")
    if left is None or right is None:
        raise RuntimeError("Approved template is missing its wall projectors")

    center_x = projected_centroid(model, 0)
    center_y = projected_centroid(model, 1)
    configure_projector(
        left,
        Vector((-wall_source_distance, center_x.y, center_x.z)),
        Vector((62.0, center_x.y, center_x.z)),
        receiver_radius=receiver_radius,
        energy=light_energy,
    )
    configure_projector(
        right,
        Vector((center_y.x, -wall_source_distance, center_y.z)),
        Vector((center_y.x, 62.0, center_y.z)),
        receiver_radius=receiver_radius,
        energy=light_energy,
    )
    centers = {
        "left_wall": [62.0, center_x.y, center_x.z],
        "right_wall": [center_y.x, 62.0, center_y.z],
    }

    old_floor = bpy.data.objects.get("floor projection light")
    if old_floor is not None:
        bpy.data.objects.remove(old_floor, do_unlink=True)
    if include_floor:
        center_z = projected_centroid(model, 2)
        bpy.ops.object.light_add(
            type="SPOT",
            location=(center_z.x, center_z.y, floor_source_height),
        )
        floor = bpy.context.object
        floor.name = "floor projection light"
        floor.data.color = (0.90, 0.96, 1.0)
        configure_projector(
            floor,
            Vector((center_z.x, center_z.y, floor_source_height)),
            Vector((center_z.x, center_z.y, -46.0)),
            receiver_radius=receiver_radius,
            energy=light_energy,
        )
        centers["floor"] = [center_z.x, center_z.y, -46.0]
    return centers


def split_camera_model(
    model: bpy.types.Object,
    factor: float,
    camera_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    """Shrink the camera-visible model while preserving its full-size shadows."""
    if not 0.0 < factor < 1.0:
        raise ValueError(f"camera_model_scale must be in (0, 1), got {factor}")

    low, high = bounds(model)
    anchor = Vector(
        (
            0.5 * (low.x + high.x),
            0.5 * (low.y + high.y),
            low.z,
        )
    )
    display = model.copy()
    display.data = model.data
    bpy.context.scene.collection.objects.link(display)
    display.name = "GalleryDisplayModel"
    model.name = "GalleryShadowProxy"

    # Eevee honors these independent visibility channels: the original mesh
    # remains the full-size shadow caster, while the smaller duplicate is the
    # only copy visible to the camera.
    model.visible_camera = False
    display.visible_shadow = False
    display.scale = tuple(value * factor for value in model.scale)
    display.location = (
        anchor
        + factor * (model.location - anchor)
        + Vector(camera_offset)
    )
    bpy.context.view_layer.update()
    return display


def render_case(case: dict) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(BASE_BLEND))
    model = replace_model(
        case["mesh"],
        case["basis"],
        display_extent=float(case.get("display_extent", MODEL_DISPLAY_EXTENT)),
    )
    centers = configure_projectors(
        model,
        case["floor_shadow"],
        receiver_radius=float(
            case.get("receiver_radius", PROJECTOR_RECEIVER_RADIUS)
        ),
        wall_source_distance=float(case.get("wall_source_distance", 2000.0)),
        floor_source_height=float(case.get("floor_source_height", 2000.0)),
        light_energy=float(case.get("light_energy", 165_000_000.0)),
    )
    if "camera_model_scale" in case:
        split_camera_model(
            model,
            float(case["camera_model_scale"]),
            tuple(case.get("camera_model_offset", (0.0, 0.0, 0.0))),
        )

    camera = bpy.data.objects.get("SimulationCamera")
    if camera is None:
        raise RuntimeError("Approved template is missing SimulationCamera")
    if case["floor_shadow"]:
        default_ortho_scale = 280.0
        default_target = (10.0, 10.0, -5.0)
    else:
        default_ortho_scale = 230.0
        default_target = (10.0, 10.0, 10.0)
    camera.data.ortho_scale = float(
        case.get("camera_ortho_scale", default_ortho_scale)
    )
    look_at(camera, Vector(case.get("camera_target", default_target)))

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.resolution_x = 1600
    # Match the gallery cell aspect ratio so the evidence fills its row
    # without letterboxing or distorting the physical projection geometry.
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    out = GALLERY / case["name"] / case.get(
        "output_subdir", "renders_four_case_centered"
    )
    out.mkdir(parents=True, exist_ok=True)
    output_name = case.get("output_name", "simulation_centered.png")
    scene.render.filepath = str(out / output_name)
    bpy.ops.wm.save_as_mainfile(
        filepath=str(
            out
            / case.get(
                "blend_name", f"{case['name']}_simulation_centered.blend"
            )
        )
    )
    bpy.ops.render.render(write_still=True)
    (out / "projection_centers.json").write_text(
        json.dumps(centers, indent=2) + "\n",
        encoding="utf-8",
    )
    print(scene.render.filepath)


def main() -> None:
    if not BASE_BLEND.is_file():
        raise FileNotFoundError(BASE_BLEND)
    selected = os.environ.get("PCRAFT_GALLERY_CASE")
    for case in CASES:
        if selected and case["name"] != selected:
            continue
        if not case["mesh"].is_file():
            raise FileNotFoundError(case["mesh"])
        render_case(case)


if __name__ == "__main__":
    main()
