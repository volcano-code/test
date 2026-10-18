"""Render evidence views for the full-pipeline parrot/snail two-view case.

The orthographic thumbnails and the corner simulation are generated directly
from the selected P-CRAFT STL. No silhouette texture is composited onto a wall.
"""

from __future__ import annotations

from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
CASE = ROOT / "results/full_pipeline_batch_2v3v_20260730/2v_simple_parrot_snail"
STL = CASE / "selected_display_fixed.stl"
OUT = (
    ROOT
    / "paper_aaai_submission/supplement_gallery_one_case_20260801"
    / "parrot_snail"
    / "renders"
)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def material(
    name: str,
    color: tuple[float, float, float],
    roughness: float = 0.55,
    metallic: float = 0.0,
):
    value = bpy.data.materials.new(name)
    value.use_nodes = True
    value.diffuse_color = (*color, 1.0)
    bsdf = value.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    specular = bsdf.inputs.get("Specular IOR Level")
    if specular is not None:
        specular.default_value = 0.32
    return value


def look_at(obj, point: Vector) -> None:
    obj.rotation_euler = (point - obj.location).to_track_quat("-Z", "Y").to_euler()


def bounds(obj) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    lower = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    upper = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return lower, upper


def import_model(mat):
    bpy.ops.wm.stl_import(filepath=str(STL))
    obj = bpy.context.object
    obj.name = "PCRAFT_parrot_snail"
    obj.data.materials.append(mat)
    lower, upper = bounds(obj)
    center = (lower + upper) / 2.0
    obj.location -= center
    lower, _ = bounds(obj)
    obj.location.z -= lower.z
    return obj


def add_cube(name, location, dimensions, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    return obj


def render_orthographic_views() -> None:
    clear_scene()
    model = import_model(material("PCRAFT teal", (0.12, 0.56, 0.55), 0.48))

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "GalleryCamera"
    camera.data.type = "ORTHO"
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "studio.sl"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "BOTH"
    scene.display.shading.show_specular_highlight = True
    scene.render.resolution_x = 900
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.25

    lower, upper = bounds(model)
    z_center = (lower.z + upper.z) / 2.0
    distance = 220.0

    # Native fixed-two-wall protocol: Y receiver is viewed along -Y and X
    # receiver along -X. Dynamic scales retain the complete selected object.
    views = (
        (
            Vector((0.0, -distance, z_center)),
            1.12 * max(upper.x - lower.x, upper.z - lower.z),
            OUT / "mesh_view_y.png",
        ),
        (
            Vector((-distance, 0.0, z_center)),
            1.12 * max(upper.y - lower.y, upper.z - lower.z),
            OUT / "mesh_view_x.png",
        ),
    )
    target = Vector((0.0, 0.0, z_center))
    for location, ortho_scale, output in views:
        camera.location = location
        camera.data.ortho_scale = ortho_scale
        look_at(camera, target)
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)


def add_spot(name: str, location, target, color) -> None:
    bpy.ops.object.light_add(type="SPOT", location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = 5200.0
    light.data.color = color
    light.data.spot_size = 0.60
    light.data.spot_blend = 0.66
    light.data.shadow_soft_size = 0.12
    light.data.use_shadow = False
    look_at(light, Vector(target))


def add_sun(name: str, direction, color) -> None:
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 0.0))
    light = bpy.context.object
    light.name = name
    light.data.energy = 2.3
    light.data.angle = 0.025
    light.data.color = color
    look_at(light, Vector(direction))


def render_simulation() -> None:
    clear_scene()
    # Reuse the exact calibrated Fig. 1(b) scene so that gallery cases differ
    # only in their evidence STL, never in camera, receiver, or lighting style.
    wall_left_mat = material("warm white left", (0.70, 0.70, 0.64), 0.78)
    wall_right_mat = material("cool white right", (0.68, 0.70, 0.74), 0.78)
    floor_mat = material("neutral floor", (0.42, 0.44, 0.47), 0.78)
    model_mat = material("printed model", (1.0, 0.55, 0.015), 0.48)

    add_cube("left wall X", (62.0, 0.0, 28.0), (1.2, 260.0, 220.0), wall_left_mat)
    add_cube("right wall Y", (0.0, 62.0, 28.0), (260.0, 1.2, 220.0), wall_right_mat)
    add_cube("floor", (0.0, 0.0, -46.0), (260.0, 260.0, 1.2), floor_mat)

    model = import_model(model_mat)
    # Match the visual object-to-scene ratio used by the first gallery case.
    model.scale = (0.32, 0.32, 0.32)
    bpy.context.view_layer.objects.active = model
    model.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    lower, upper = bounds(model)
    model.location.x -= (lower.x + upper.x) / 2.0
    model.location.y -= (lower.y + upper.y) / 2.0
    model.location.z -= lower.z

    # These two real spot lights generate the visible pools and the shadows.
    # No wall mask or post-render silhouette compositing is used.
    for name, location, target in (
        ("left projection light", (-60.0, 0.0, 20.0), (62.0, 0.0, 20.0)),
        ("right projection light", (0.0, -60.0, 20.0), (0.0, 62.0, 20.0)),
    ):
        bpy.ops.object.light_add(type="SPOT", location=location)
        light = bpy.context.object
        light.name = name
        light.data.energy = 600000.0
        light.data.spot_size = 0.92
        light.data.spot_blend = 0.72
        light.data.shadow_soft_size = 0.18
        look_at(light, Vector(target))

    bpy.ops.object.camera_add(location=(-175.0, -175.0, 108.0))
    camera = bpy.context.object
    camera.name = "SimulationCamera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 190.0
    look_at(camera, Vector((10.0, 10.0, 18.0)))

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1108
    scene.render.resolution_y = 820
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.filepath = str(OUT / "simulation.png")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.94, 0.94, 0.94, 1.0)
    background.inputs["Strength"].default_value = 0.08
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.45
    bpy.ops.render.render(write_still=True)

    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "parrot_snail_gallery_evidence.blend"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    render_orthographic_views()
    render_simulation()
    print(OUT / "mesh_view_y.png")
    print(OUT / "mesh_view_x.png")
    print(OUT / "simulation.png")


if __name__ == "__main__":
    main()
