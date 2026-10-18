#!/usr/bin/env python3
"""Render matched fixed-camera 3D views for the Figure 4 comparison."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = (
    ROOT
    / "paper_aaai_submission/fig4_table1_connected06_soft_preview_20260726"
    / "blender_renders"
)
PROJECTION_TEXTURES = OUT / "projection_textures"
PROTOCOL_PROJECTIONS = OUT / "protocol_projections"

METHODS = {
    "mitra": (
        ROOT
        / "paper_aaai_submission/fig4_table1_connected06_20260726"
        / "mitra_connected06_surface_for_visualization.obj",
        (0.12, 0.23, 0.29, 1.0),
    ),
    "sadekar_mesh": (
        OUT / "display_meshes/sadekar_mesh_display_only.obj",
        (0.08, 0.27, 0.56, 1.0),
    ),
    "sadekar_voxel": (
        OUT / "display_meshes/sadekar_voxel_display_only.obj",
        (0.00, 0.31, 0.26, 1.0),
    ),
    "pcraft": (
        ROOT
        / "results/fixed3_mr1_connected10_pcraft_20260722/"
        "connected_06/seed_17/selected_display_fixed.stl",
        (0.55, 0.22, 0.01, 1.0),
    ),
}

VIEWS = {
    "view_a": (4.25, -5.10, 3.35),
}

DISPLAY_SCALE = {
    "mitra": 0.82,
    "sadekar_mesh": 0.78,
    "sadekar_voxel": 0.82,
    "pcraft": 0.82,
}

RECEIVER_SIZE = 3.20
# Plane centers sit exactly one half-width from the shared edges. Keeping this
# equality exact removes the dark cracks previously visible between receivers.
RECEIVER_OFFSET = RECEIVER_SIZE * 0.5


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def material(name: str, rgba: tuple[float, float, float, float], roughness: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    mat.use_nodes = True
    principled = mat.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = rgba
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Metallic"].default_value = 0.02
    return mat


def projection_material(
    name: str,
    image_path: Path,
    light_color: tuple[float, float, float, float],
) -> bpy.types.Material:
    """Create an evidence-driven receiver: colored field plus dark silhouette."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(str(image_path), check_existing=True)
    texture.interpolation = "Closest"
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"

    links.new(texture.outputs["Color"], mix.inputs["Fac"])
    # Projection evidence is stored as a white foreground on black. Render the
    # black background as the pale receiver and the white foreground as a dark
    # shadow. This is a deterministic visualization of the evaluated
    # orthographic projection, not a second learned prediction.
    mix.inputs["Color1"].default_value = (0.035, 0.040, 0.045, 1.0)
    mix.inputs["Color2"].default_value = light_color
    links.new(mix.outputs["Color"], emission.inputs["Color"])
    emission.inputs["Strength"].default_value = 1.85
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def import_geometry(path: Path) -> bpy.types.Object:
    if path.suffix.lower() == ".stl":
        bpy.ops.wm.stl_import(filepath=str(path))
    elif path.suffix.lower() == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported geometry: {path}")
    objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh imported from {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "MethodGeometry"
    return obj


def world_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    maximum = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return minimum, maximum


def normalize_geometry(obj: bpy.types.Object, display_scale: float = 1.0) -> None:
    # Keep every stored vertex, but use robust bounds for matched framing.
    # Sadekar Mesh contains a few very long spikes; full min/max bounds would
    # shrink its coherent body until neither the model nor its shadows can be
    # inspected. The outliers remain in the render and can still cast shadows.
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.location = (0.0, 0.0, 0.0)
    coordinates = np.asarray([vertex.co[:] for vertex in obj.data.vertices], dtype=np.float64)
    low = np.quantile(coordinates, 0.01, axis=0)
    high = np.quantile(coordinates, 0.99, axis=0)
    center = (low + high) * 0.5
    extent = np.maximum(high - low, 1e-8)
    # Normalize all methods by the largest physical extent, then place the
    # object center at the same floating position. This preserves aspect ratio
    # while making apparent scale and shadow geometry directly comparable.
    # Match the robust 3D diagonal rather than a single dominant axis. This
    # prevents a thin reconstruction from appearing much smaller than a
    # volumetric one under the shared oblique camera.
    scale = display_scale * 1.78 / float(np.linalg.norm(extent))
    for vertex in obj.data.vertices:
        source = Vector(vertex.co)
        vertex.co = Vector(
            (
                (source.x - center[0]) * scale,
                (source.y - center[1]) * scale,
                (source.z - center[2]) * scale,
            )
        )
    bpy.context.view_layer.update()
    minimum, maximum = world_bounds(obj)
    obj.location.z += 0.92 - (minimum.z + maximum.z) * 0.5
    bpy.context.view_layer.update()
    minimum, maximum = world_bounds(obj)
    print(f"NORMALIZED {obj.name}: min={tuple(minimum)}, max={tuple(maximum)}")


def canonicalize_display_frame(obj: bpy.types.Object, slug: str) -> None:
    """Map archived method coordinates into the shared display frame.

    The Mitra archive stores its first two orthographic axes in the opposite
    horizontal order from the common X/Y display convention. This operation is
    applied only to the paper-render copy; evaluated geometry is untouched.
    """
    if slug != "mitra":
        return
    for vertex in obj.data.vertices:
        x, y, z = vertex.co
        vertex.co = (y, -x, z)
    obj.data.update()


def remove_degenerate_display_faces(obj: bpy.types.Object) -> None:
    """Remove only extreme numerical triangles from the display copy.

    The archived Sadekar mesh contains a small number of triangles whose edges
    are orders of magnitude longer than the coherent reconstruction. They are
    retained in the evaluated source OBJ, but omitted from the paper render so
    that the actual reconstructed body remains inspectable.
    """
    mesh = obj.data
    coordinates = np.asarray([vertex.co[:] for vertex in mesh.vertices], dtype=np.float64)
    low = np.quantile(coordinates, 0.01, axis=0)
    high = np.quantile(coordinates, 0.99, axis=0)
    robust_extent = max(float((high - low).max()), 1e-8)
    threshold = 0.55 * robust_extent
    remove = []
    for polygon in mesh.polygons:
        vertices = coordinates[list(polygon.vertices)]
        max_edge = max(
            np.linalg.norm(vertices[i] - vertices[(i + 1) % len(vertices)])
            for i in range(len(vertices))
        )
        if max_edge > threshold:
            remove.append(polygon.index)
    if not remove:
        return
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for index in remove:
        mesh.polygons[index].select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.delete(type="FACE")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_loose()
    bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"DISPLAY_SANITIZE {obj.name}: removed {len(remove)} degenerate faces")


def point_camera(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def emission_material(
    name: str,
    rgba: tuple[float, float, float, float],
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = rgba
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def render_protocol_projections(
    geometry: bpy.types.Object,
    slug: str,
) -> dict[str, Path]:
    """Render X/Y/Z silhouettes in the shared fixed-surface coordinate frame.

    The camera frame spans the complete receiver. No crop, resize, or
    image-space translation is applied, so projection centers and scales are
    inherited directly from the normalized model coordinates.
    """
    scene = bpy.context.scene
    PROTOCOL_PROJECTIONS.mkdir(parents=True, exist_ok=True)

    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.look = "None"

    world = bpy.data.worlds.new("ProjectionWorld")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0

    original_materials = list(geometry.data.materials)
    geometry.data.materials.clear()
    geometry.data.materials.append(
        emission_material("ProjectionSilhouette", (0.0, 0.0, 0.0, 1.0))
    )

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "ProtocolProjectionCamera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = RECEIVER_SIZE
    camera.data.lens = 50.0
    scene.camera = camera

    # Fixed surfaces use the common object frame: X and Y are the two wall
    # normals, and Z is the floor normal. Wall cameras share the receiver
    # center height; the floor camera is centered on the same X/Y origin.
    frames = {
        "x": ((4.5, 0.0, RECEIVER_OFFSET), (0.0, 0.0, RECEIVER_OFFSET)),
        "y": ((0.0, -4.5, RECEIVER_OFFSET), (0.0, 0.0, RECEIVER_OFFSET)),
        "z": ((0.0, 0.0, 4.5), (0.0, 0.0, 0.0)),
    }
    paths: dict[str, Path] = {}
    for axis, (location, target) in frames.items():
        camera.location = location
        point_camera(camera, Vector(target))
        output = PROTOCOL_PROJECTIONS / f"{slug}_{axis}.png"
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        paths[axis] = output

    bpy.data.objects.remove(camera, do_unlink=True)
    geometry.data.materials.clear()
    for mat in original_materials:
        geometry.data.materials.append(mat)
    return paths


def add_receiver(
    name: str,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float],
    image_path: Path,
    rgba: tuple[float, float, float, float],
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(
        size=RECEIVER_SIZE,
        location=location,
        rotation=rotation,
    )
    receiver = bpy.context.object
    receiver.name = name
    receiver.data.materials.append(
        projection_material(name + "Material", image_path, rgba)
    )
    return receiver


def add_projection_light(
    name: str,
    location: tuple[float, float, float],
    target: tuple[float, float, float],
    color: tuple[float, float, float],
    energy: float,
) -> None:
    # A broad spot cone covers the complete receiver but does not illuminate
    # the other two receivers. This preserves physical cast shadows without
    # showing spotlight circles or washing shadows out with cross-lighting.
    bpy.ops.object.light_add(type="SPOT", location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.color = color
    light.data.spot_size = math.radians(92.0)
    light.data.spot_blend = 0.08
    light.data.shadow_soft_size = 0.025
    point_camera(light, Vector(target))


def setup_scene(
    slug: str,
    geometry_color: tuple[float, float, float, float],
    protocol_projections: dict[str, Path],
) -> tuple[bpy.types.Object, bpy.types.Object]:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 900
    scene.render.resolution_y = 675
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.image_settings.color_mode = "RGB"

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.85

    # A matched three-receiver scene. Receiver textures are the evaluated
    # orthographic projections for the same stored model; the imported geometry
    # is shown in their shared canonical frame.
    add_receiver(
        "FloorReceiver",
        (0.0, 0.0, -0.025),
        (0.0, 0.0, 0.0),
        protocol_projections["z"],
        (0.94, 0.95, 0.96, 1.0),
    )
    add_receiver(
        "LeftReceiver",
        (-RECEIVER_OFFSET, 0.0, RECEIVER_OFFSET),
        (0.0, math.radians(90.0), 0.0),
        protocol_projections["x"],
        (1.00, 0.98, 0.76, 1.0),
    )
    add_receiver(
        "RightReceiver",
        (0.0, RECEIVER_OFFSET, RECEIVER_OFFSET),
        (math.radians(90.0), 0.0, 0.0),
        protocol_projections["y"],
        (0.94, 0.86, 1.00, 1.0),
    )

    bpy.ops.object.light_add(type="AREA", location=(2.2, -2.6, 3.0))
    fill = bpy.context.object
    fill.name = "ModelFill"
    fill.data.energy = 700.0
    fill.data.shape = "DISK"
    fill.data.size = 2.0
    point_camera(fill, Vector((0.0, 0.0, 0.92)))

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 4.85
    scene.camera = camera

    geometry = bpy.data.objects.get("MethodGeometry")
    geometry.data.materials.clear()
    geometry.data.materials.append(
        material("MethodGeometryMaterial", geometry_color, 0.58)
    )
    return geometry, camera


def render_method(slug: str, path: Path, color: tuple[float, float, float, float]) -> None:
    clear_scene()
    geometry = import_geometry(path)
    canonicalize_display_frame(geometry, slug)
    if slug == "sadekar_mesh":
        remove_degenerate_display_faces(geometry)
    normalize_geometry(geometry, DISPLAY_SCALE[slug])
    # Derive every receiver silhouette from the exact mesh shown in the same
    # panel after its canonical display transform. This keeps model pose,
    # projection center, scale, and all three receiver silhouettes physically
    # coupled. These images are orthographic shadow masks, not target or
    # evaluation textures copied from the comparison rows above.
    protocol_projections = render_protocol_projections(geometry, slug)
    geometry, camera = setup_scene(slug, color, protocol_projections)
    # The normalized object is centered on the common X/Y projection axes.
    # Aim at the same center so the object and all three receiver masks remain
    # registered in the compact three-surface composition.
    target = Vector((0.0, 0.0, 0.92))
    for view_name, location in VIEWS.items():
        camera.location = location
        point_camera(camera, target)
        bpy.context.scene.render.filepath = str(OUT / f"{slug}_{view_name}.png")
        bpy.ops.render.render(write_still=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path, _ in METHODS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Figure 4 geometry:\n" + "\n".join(missing))
    for slug, (path, color) in METHODS.items():
        render_method(slug, path, color)
    print(OUT)


if __name__ == "__main__":
    main()
