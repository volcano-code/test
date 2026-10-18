"""Thin executable for immutable v7 suite binding."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadowforge_pipeline.soft_voxel_v7.suite import binding_main, write_v7_suite_bundle


if __name__ == "__main__":
    raise SystemExit(binding_main())
