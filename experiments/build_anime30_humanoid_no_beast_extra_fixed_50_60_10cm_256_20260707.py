#!/usr/bin/env python3
"""Extra no-beast fixed 10cm/256 anime30 exports to reach five accepted cases."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
BASE = ROOT / "experiments/build_anime30_diverse_fixed_50_60_10cm_256_20260707.py"

spec = importlib.util.spec_from_file_location("base_anime30_50_60", str(BASE))
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

mod.OUT = ROOT / "results/anime30_humanoid_no_beast_extra_fixed_50_60_10cm_256_20260707"
mod.EXAMPLES = [
    ("extra1_thick1_15_22_28", "thick1", ["anime30_15", "anime30_22", "anime30_28"]),
    ("extra2_close1_09_13_15", "close1", ["anime30_09", "anime30_13", "anime30_15"]),
    ("extra3_thick1_15_23_28", "thick1", ["anime30_15", "anime30_23", "anime30_28"]),
    ("extra4_thick1_13_21_23", "thick1", ["anime30_13", "anime30_21", "anime30_23"]),
    ("extra5_thick1_13_15_23", "thick1", ["anime30_13", "anime30_15", "anime30_23"]),
    ("extra6_thick1_13_14_22", "thick1", ["anime30_13", "anime30_14", "anime30_22"]),
    ("extra7_thick1_15_27_28", "thick1", ["anime30_15", "anime30_27", "anime30_28"]),
    ("extra8_thick1_13_15_28", "thick1", ["anime30_13", "anime30_15", "anime30_28"]),
]


if __name__ == "__main__":
    mod.main()
