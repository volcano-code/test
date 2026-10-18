#!/usr/bin/env python3
"""Package complex NSA-style models under the same fixed-required gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
BASE = ROOT / "experiments/package_nsa_fixed_required_10cm_256.py"
spec = importlib.util.spec_from_file_location("pkg_base", str(BASE))
pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pkg)

pkg.SRC = ROOT / "results/nsa_style_complex_12_10cm_256"
pkg.OUT = ROOT / "results/nsa_style_complex_fixed_required_10cm_256"


if __name__ == "__main__":
    pkg.main()
