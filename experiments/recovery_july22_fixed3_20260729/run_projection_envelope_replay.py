#!/usr/bin/env python3
"""Replay the fixed-view delivery path with a projection-only envelope.

The verified July 22 artifacts contain a single watertight exterior shell,
whereas the current generic adapter merges the fragmented V256 soft proposal
into the delivery envelope.  This isolated runner keeps the soft proposal as
candidate guidance but builds the delivery envelope only from the declared
camera masks.  It deliberately leaves the production two-wall path and the
generic implementation unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadowforge_pipeline import unified_run as pipeline


_production_visual_hull = pipeline._generic_visual_hull


def projection_only_visual_hull(
    targets: list[np.ndarray],
    args,
    resolution: int,
    soft_info: dict | None = None,
    preserve_xyz: np.ndarray | None = None,
) -> np.ndarray:
    """Construct the protocol envelope without reinserting soft fragments."""
    del preserve_xyz
    return _production_visual_hull(
        targets,
        args,
        resolution,
        soft_info=soft_info,
        preserve_xyz=None,
    )


def main() -> None:
    pipeline._generic_visual_hull = projection_only_visual_hull
    pipeline.main()


if __name__ == "__main__":
    main()
