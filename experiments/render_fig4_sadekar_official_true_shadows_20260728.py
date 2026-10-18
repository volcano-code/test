#!/usr/bin/env python3
"""Render genuine three-receiver shadows from archived Sadekar meshes.

This is an isolated Figure 4 evidence renderer. It does not import silhouette
textures and does not modify any delivery or two-view pipeline code.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


METHODS = {
    "sadekar_mesh": (
        Path(
            "/data/xvxinyao/ShadowArt-Revisited/mesh_results/"
            "fixed3_mr1_connected10_sadekar_20260722/mesh/"
            "connected_06/seed_17/sample_0_output.obj"
        ),
        (0.20, 0.52, 0.88, 1.0),
    ),
    "sadekar_voxel": (
        Path(
            "/data/xvxinyao/ShadowArt-Revisited/voxel_results/"
            "fixed3_mr1_connected10_sadekar_20260722/voxel/"
            "connected_06/seed_17/sample_0_output.obj"
        ),
        (0.16, 0.66, 0.58, 1.0),
    ),
}

RECEIVER_SIZE = 3.4
RECEIVER_OFFSET = RECEIVER_SIZE * 0.5
MODEL_CENTER = Vector((0.0, 0.0, 0.96))


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS),
        default=list(METHODS),
    )
    parser.add_argument("--resolution-x", type=int, default=1600)
    parser.add_argument("--resolution-y", type=int, default=1200)
    parser.add_argument(
        "--axis-map",
        choices=("legacy", "upright"),
        default="legacy",
        help="Keep the archived mapping or use the corrected upright wall mapping.",
    )
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for blocks in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.worlds,
    ):
        for block in list(blocks):
            if block.users == 0:
                blocks.remove(block)


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.75,
) -> bpy.types.Material:
    result = bpy.data.materials.new(name)
    result.diffuse_color = color
    result.use_nodes = True
    principled = result.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Metallic"].default_value = 0.0
    return result


def import_official_obj(path: Path) -> bpy.types.Object:
    bpy.ops.wm.obj_import(filepath=str(path))
    objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh imported from {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    result = bpy.context.view_layer.objects.active
    result.name = "OfficialSadekarGeometry"

    # Blender's OBJ importer adds an X-up conversion. Sadekar/PyTorch3D reads
    # the stored OBJ coordinates directly, so discard only that import transform.
    result.matrix_world = Matrix.Identity(4)
    return result


def map_protocol_axes(obj: bpy.types.Object, axis_map: str) -> None:
    """Map native Sadekar views to left wall, right wall, and floor.

    Native +X shows the detective, +Z the airborne character, and +Y the
    spider. Rotate the prior scene mapping by 90 degrees about scene Z so the
    wall projections follow the same left-to-right order as the silhouette
    rows while preserving the floor projection and a right-handed frame.
    """
    for vertex in obj.data.vertices:
        x, y, z = vertex.co
        if axis_map == "upright":
            # Left wall: detective; right wall: airborne character; floor: spider.
            vertex.co = (x, -z, y)
        else:
            vertex.co = (-z, x, -y)
    obj.data.update()


def normalize_uniformly(obj: bpy.types.Object, method: str) -> None:
    """Uniformly frame the complete archived object without deleting geometry."""
    coordinates = np.asarray([vertex.co[:] for vertex in obj.data.vertices])
    # Mesh output contains a few very long official triangles. Robust framing
    # keeps the reconstructed body inspectable while retaining every triangle.
    quantile = 0.005 if method == "sadekar_mesh" else 0.0
    low = np.quantile(coordinates, quantile, axis=0)
    high = np.quantile(coordinates, 1.0 - quantile, axis=0)
    center = (low + high) * 0.5
    extent = np.maximum(high - low, 1e-8)
    scale = 1.36 / float(np.linalg.norm(extent))
    for vertex in obj.data.vertices:
        source = np.asarray(vertex.co[:], dtype=np.float64)
        target = (source - center) * scale
        vertex.co = tuple(float(value) for value in target)
    obj.data.update()

    full = np.asarray([vertex.co[:] for vertex in obj.data.vertices])
    full_low = full.min(axis=0)
    full_high = full.max(axis=0)
    robust_low = np.quantile(full, quantile, axis=0)
    robust_high = np.quantile(full, 1.0 - quantile, axis=0)
    robust_center = (robust_low + robust_high) * 0.5
    obj.location = (
        MODEL_CENTER.x - robust_center[0],
        MODEL_CENTER.y - robust_center[1],
        MODEL_CENTER.z - robust_center[2],
    )
    print(
        f"FRAME {method}: vertices={len(obj.data.vertices)} "
        f"faces={len(obj.data.polygons)} "
        f"full_extent={tuple(np.round(full_high - full_low, 5))} "
        f"robust_extent={tuple(np.round(robust_high - robust_low, 5))}"
    )


def add_receiver(
    name: str,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(
        size=RECEIVER_SIZE,
        location=location,
        rotation=rotation,
    )
    receiver = bpy.context.object
    receiver.name = name
    receiver.data.materials.append(material(name + "Material", color, 0.92))
    return receiver


def add_spot(
    name: str,
    location: tuple[float, float, float],
    target: tuple[float, float, float],
    color: tuple[float, float, float],
    energy: float,
    spot_size: float,
) -> None:
    bpy.ops.object.light_add(type="SPOT", location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.color = color
    light.data.spot_size = math.radians(spot_size)
    light.data.spot_blend = 0.06
    light.data.shadow_soft_size = 0.01
    point_at(light, Vector(target))


def setup_scene(
    method: str,
    geometry: bpy.types.Object,
    geometry_color: tuple[float, float, float, float],
    output: Path,
    resolution_x: int,
    resolution_y: int,
) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.18
    scene.world = world

    geometry.data.materials.clear()
    geometry.data.materials.append(
        material("ArchivedGeometryMaterial", geometry_color, 0.58)
    )

    add_receiver(
        "FloorReceiver",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.88, 0.90, 0.93, 1.0),
    )
    add_receiver(
        "LeftReceiver",
        (-RECEIVER_OFFSET, 0.0, RECEIVER_OFFSET),
        (0.0, math.radians(90.0), 0.0),
        (0.94, 0.91, 0.70, 1.0),
    )
    add_receiver(
        "RightReceiver",
        (0.0, RECEIVER_OFFSET, RECEIVER_OFFSET),
        (math.radians(90.0), 0.0, 0.0),
        (0.84, 0.77, 0.92, 1.0),
    )

    # Each narrow cone passes through the model center and reaches exactly one
    # receiver. All visible silhouettes are genuine cast shadows.
    add_spot(
        "DetectiveProjection",
        (7.0, 0.0, MODEL_CENTER.z),
        (-RECEIVER_OFFSET, 0.0, MODEL_CENTER.z),
        (1.0, 0.94, 0.66),
        1500.0,
        12.0,
    )
    add_spot(
        "AirborneProjection",
        (0.0, -7.0, MODEL_CENTER.z),
        (0.0, RECEIVER_OFFSET, MODEL_CENTER.z),
        (0.84, 0.74, 1.0),
        1500.0,
        12.0,
    )
    add_spot(
        "SpiderProjection",
        (0.0, 0.0, 7.0),
        (0.0, 0.0, 0.0),
        (0.86, 0.90, 0.96),
        1300.0,
        15.0,
    )

    bpy.ops.object.light_add(type="AREA", location=(3.3, -4.0, 4.8))
    fill = bpy.context.object
    fill.name = "ShadowlessModelFill"
    fill.data.energy = 120.0
    fill.data.shape = "DISK"
    fill.data.size = 2.3
    if hasattr(fill.data, "use_shadow"):
        fill.data.use_shadow = False
    point_at(fill, MODEL_CENTER)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "FigureCamera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 4.55
    camera.location = (4.35, -5.25, 3.45)
    point_at(camera, Vector((-0.03, 0.03, 1.03)))
    scene.camera = camera

    scene.render.filepath = str(output / f"{method}_true_shadows.png")
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(
        filepath=str(output / f"{method}_true_shadows.blend")
    )


def render_method(
    method: str,
    source: Path,
    color: tuple[float, float, float, float],
    output: Path,
    resolution_x: int,
    resolution_y: int,
    axis_map: str,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    clear_scene()
    geometry = import_official_obj(source)
    map_protocol_axes(geometry, axis_map)
    normalize_uniformly(geometry, method)
    setup_scene(
        method,
        geometry,
        color,
        output,
        resolution_x,
        resolution_y,
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        source, color = METHODS[method]
        render_method(
            method,
            source,
            color,
            args.output,
            args.resolution_x,
            args.resolution_y,
            args.axis_map,
        )
    print(args.output)


if __name__ == "__main__":
    main()
