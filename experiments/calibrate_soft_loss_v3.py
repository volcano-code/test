#!/usr/bin/env python3
"""Executable entry point for gradient-only soft-loss-v3 calibration."""

from __future__ import annotations

from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadowforge_pipeline.soft_voxel_v3.calibration import calibration_main


if __name__ == "__main__":
    raise SystemExit(calibration_main())
