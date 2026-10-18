"""Render strict receiver-aligned orthographic views for the gallery case."""

from __future__ import annotations

from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
CASE = (
    ROOT
    / "results/anime_two_wall_fixed_no_floor_physical_axes_qualityfirst_10cm_256_20260710"
    / "dark_mage_left__samurai_right"
)
STL = CASE / "dark_mage_left__samurai_right_upright_saving_10cm_256.stl"
OUT = (
    ROOT
    / "paper_aaai_submission/supplement_gallery_one_case_20260801"
    / "dark_mage_samurai"
    / "renders"
)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def material(name: str, color: tuple[float, float, float], roughness: float = 0.46):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = (*color, 1.0)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    specular = bsdf.inputs.get("Specular IOR Level")
    if specular is not None:
        specular.default_value = 0.34
    return mat


def look_at(obj, point: Vector) -> None:
    obj.rotation_euler = (point - obj.location).to_track_quat("-Z", "Y").to_euler()


def world_bounds(obj) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    lower = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    upper = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return lower, upper


def add_area_light(
    name: str,
    location: tuple[float, float, float],
    energy: float,
    size: float,
    color: tuple[float, float, float],
    target: Vector,
) -> None:
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.shape = "DISK"
    light.data.size = size
    light.data.color = color
    light.data.use_shadow = True
    look_at(light, target)


def render_view(
    camera_location: tuple[float, float, float],
    target: tuple[float, float, float],
    ortho_scale: float,
    output: Path,
) -> None:
    scene = bpy.context.scene
    camera = bpy.data.objects.get("GalleryCamera")
    camera.location = camera_location
    camera.data.ortho_scale = ortho_scale
    look_at(camera, Vector(target))
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clear_scene()

    bpy.ops.wm.stl_import(filepath=str(STL))
    model = bpy.context.object
    model.name = "PCRAFT_dark_mage_samurai"
    # STL facet normals are not consistently oriented. Recompute only the
    # shading normals so Eevee reveals the existing surface relief faithfully.
    bpy.context.view_layer.objects.active = model
    model.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    model.data.materials.append(material("PCRAFT_teal", (0.16, 0.58, 0.57), 0.46))

    lower, upper = world_bounds(model)
    center = (lower + upper) / 2.0
    model.location -= center
    lower, upper = world_bounds(model)
    model.location.z -= lower.z

    bpy.ops.object.camera_add(location=(185.0, 0.0, 40.0))
    camera = bpy.context.object
    camera.name = "GalleryCamera"
    camera.data.type = "ORTHO"
    camera.data.lens = 55
    bpy.context.scene.camera = camera

    lower, upper = world_bounds(model)
    light_target = Vector((0.0, 0.0, (lower.z + upper.z) * 0.52))
    add_area_light("Key", (-105.0, -125.0, 165.0), 1250.0, 72.0, (1.0, 0.93, 0.84), light_target)
    add_area_light("Fill", (125.0, -25.0, 95.0), 430.0, 92.0, (0.72, 0.86, 1.0), light_target)
    add_area_light("Rim", (35.0, 135.0, 150.0), 980.0, 58.0, (0.78, 1.0, 0.96), light_target)

    scene = bpy.context.scene
    # Workbench studio shading keeps the exact evidence geometry and aligned
    # cameras while making the layered surface relief readable at paper scale.
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.studio_light = "studio.sl"
    shading.color_type = "MATERIAL"
    shading.show_shadows = True
    shading.show_cavity = True
    shading.cavity_type = "BOTH"
    shading.show_specular_highlight = True
    scene.render.resolution_x = 900
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.35
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.035, 0.045, 0.05, 1.0)
    background.inputs["Strength"].default_value = 0.42

    # Match the receiver/light metadata used by the silhouette projections.
    # The Y receiver uses rays along -Y; the X receiver uses rays along -X.
    # Dynamic orthographic scales keep the complete object visible without
    # introducing a transverse camera offset that would change the view.
    lower, upper = world_bounds(model)
    z_center = (lower.z + upper.z) / 2.0
    distance = 220.0
    y_view_scale = 1.12 * max(upper.x - lower.x, upper.z - lower.z)
    x_view_scale = 1.12 * max(upper.y - lower.y, upper.z - lower.z)
    render_view(
        (0.0, -distance, z_center),
        (0.0, 0.0, z_center),
        y_view_scale,
        OUT / "mesh_view_y.png",
    )
    render_view(
        (-distance, 0.0, z_center),
        (0.0, 0.0, z_center),
        x_view_scale,
        OUT / "mesh_view_x.png",
    )

    blend = OUT / "dark_mage_samurai_gallery_views.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))


if __name__ == "__main__":
    main()
