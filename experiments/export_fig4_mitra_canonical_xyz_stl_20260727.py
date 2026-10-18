#!/usr/bin/env python3
"""Export the canonical Fig. 4 Mitra hull in the Blender receiver frame."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    if len(argv) != 2:
        raise SystemExit("usage: blender --background --python SCRIPT -- INPUT.obj OUTPUT.stl")
    source, output = map(Path, argv)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.obj_import(filepath=str(source))
    objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh imported from {source}")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    geometry = bpy.context.view_layer.objects.active

    # NumPy mask rows/columns and Blender receiver coordinates differ by Y/Z.
    for vertex in geometry.data.vertices:
        x, y, z = vertex.co
        vertex.co = (x, z, y)
    geometry.data.update()

    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    geometry.select_set(True)
    bpy.context.view_layer.objects.active = geometry
    bpy.ops.wm.stl_export(
        filepath=str(output),
        export_selected_objects=True,
        apply_modifiers=True,
    )
    print(output)


if __name__ == "__main__":
    main()
