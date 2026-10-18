#!/usr/bin/env python3
"""Render a Mitra reconstruction with native protocol axes.

This isolated Figure 4 audit reuses the established Sadekar-style Blender
scene while leaving all pipeline and paper-rendering scripts unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
BASE_SCRIPT = (
    ROOT / "experiments/render_fig4_sadekar_official_true_shadows_20260728.py"
)


def load_base():
    spec = importlib.util.spec_from_file_location("sadekar_scene", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sign-x", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--sign-y", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--sign-z", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--resolution-x", type=int, default=1600)
    parser.add_argument("--resolution-y", type=int, default=1200)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    args.output.mkdir(parents=True, exist_ok=True)

    base = load_base()
    base.clear_scene()
    geometry = base.import_official_obj(args.source)
    for vertex in geometry.data.vertices:
        vertex.co = (
            args.sign_x * vertex.co.x,
            args.sign_y * vertex.co.y,
            args.sign_z * vertex.co.z,
        )
    geometry.data.update()

    base.normalize_uniformly(geometry, "mitra")
    base.setup_scene(
        "mitra_native_axes",
        geometry,
        (0.20, 0.52, 0.88, 1.0),
        args.output,
        args.resolution_x,
        args.resolution_y,
    )
    print(
        "NATIVE_AXIS_RENDER "
        f"signs=({args.sign_x},{args.sign_y},{args.sign_z}) "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
