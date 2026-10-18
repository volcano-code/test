#!/usr/bin/env python3
"""Render the accepted TSF=1 three-anime case for the supplement gallery."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
TEMPLATE = ROOT / "experiments/render_supp_gallery_arcane_casters_20260801.py"
CASE = ROOT / "results/fig4_anime_multiview_v2_gatepass_20260728_run3"
OUT = ROOT / "paper_aaai_submission/supplement_gallery_one_case_20260801/tsf1_anime/renders"


def main() -> None:
    spec = importlib.util.spec_from_file_location("gallery_render_template", TEMPLATE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Blender rendering template: {TEMPLATE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CASE = CASE
    module.MESH = CASE / "selected_display_external_shell.stl"
    module.OUT = OUT
    module.main()


if __name__ == "__main__":
    main()
