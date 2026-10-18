"""Thin CLI entry point for authenticated soft-loss-v4 calibration."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadowforge_pipeline.soft_voxel_v4.calibration import calibration_main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(calibration_main())
