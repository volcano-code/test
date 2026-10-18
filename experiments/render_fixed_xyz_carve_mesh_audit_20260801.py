#!/usr/bin/env python3
"""Render axis-aligned alpha masks for fixed-XYZ carved-cube STL audits.

Run with Blender in background mode. This script is intentionally isolated
from the P-CRAFT generation code and never edits source artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def import_mesh(path: Path) -> bpy.types.Object:
    bpy.ops.wm.stl_import(filepath=str(path))
    objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh imported from {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def normalize_mesh(obj: bpy.types.Object, extent: float = 1.6) -> None:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    points = [vertex.co.copy() for vertex in obj.data.vertices]
    low = Vector(tuple(min(point[i] for point in points) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in points) for i in range(3)))
    center = (low + high) * 0.5
    scale = extent / max(float(value) for value in (high - low))
    for vertex in obj.data.vertices:
        vertex.co = (vertex.co - center) * scale
    bpy.context.view_layer.update()


def point_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    obj.rotation_euler = (
        Vector(target) - obj.location
    ).to_track_quat("-Z", "Y").to_euler()


def configure_scene() -> bpy.types.Scene:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    world = bpy.data.worlds.new("AuditWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    scene.world = world
    return scene


def render_case(manifest_path: Path, output_root: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mesh_path = Path(manifest["selected"]["artifact_stl"])
    case_dir = output_root / manifest_path.parent.name
    case_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    scene = configure_scene()
    obj = import_mesh(mesh_path)
    normalize_mesh(obj)

    material = bpy.data.materials.new("AuditObject")
    material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    obj.data.materials.clear()
    obj.data.materials.append(material)

    directions = {
        "x": (4.0, 0.0, 0.0),
        "y": (0.0, 4.0, 0.0),
        "z": (0.0, 0.0, 4.0),
    }
    for axis, location in directions.items():
        bpy.ops.object.camera_add(location=location)
        camera = bpy.context.object
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = 1.92
        point_at(camera, (0.0, 0.0, 0.0))
        scene.camera = camera
        scene.render.filepath = str(case_dir / f"mesh_axis_{axis}.png")
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(camera, do_unlink=True)

    metadata = {
        "manifest": str(manifest_path),
        "mesh": str(mesh_path),
        "targets": [
            str(manifest_path.parent / "inputs" / f"target_view_{index}.png")
            for index in range(3)
        ],
    }
    (case_dir / "render_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    manifests = [
        Path(line.strip())
        for line in args.manifest_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        render_case(manifest, args.output)
        print(manifest.parent.name, flush=True)


if __name__ == "__main__":
    main()
