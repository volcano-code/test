#!/usr/bin/env python3
"""Run humanoid-only anime30 fixed 10cm/256 exports, excluding animal-like silhouettes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
BASE = ROOT / "experiments/build_anime30_diverse_fixed_50_60_10cm_256_20260707.py"

spec = importlib.util.spec_from_file_location("base_anime30_50_60", str(BASE))
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

mod.OUT = ROOT / "results/anime30_humanoid_fixed_50_60_10cm_256_20260707"
mod.EXAMPLES = [
    ("humanoid1_close1_05_13_15", "close1", ["anime30_05", "anime30_13", "anime30_15"]),
    ("humanoid2_thick1_05_15_28", "thick1", ["anime30_05", "anime30_15", "anime30_28"]),
    ("humanoid3_thick1_15_20_28", "thick1", ["anime30_15", "anime30_20", "anime30_28"]),
    ("humanoid4_close1_13_14_15", "close1", ["anime30_13", "anime30_14", "anime30_15"]),
]


if __name__ == "__main__":
    mod.main()
