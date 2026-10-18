#!/usr/bin/env python3
"""Isolated compatibility runner for the verified 2026-07-22 delivery path.

The current generic adapter performs another declared-ray restoration and a
global bridge repair after lifting geometry to V256. Historical fixed-three-
view artifacts instead combined each repaired core with one common projection
envelope shell. This runner restores that final composition without modifying
the mature two-wall path or the current generic implementation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadowforge_pipeline import unified_run as pipeline


def recovered_connected_delivery(
    core: np.ndarray,
    hull_256: np.ndarray,
    shell: np.ndarray,
    include_shell: bool,
    guidance: np.ndarray | None = None,
    targets: list[np.ndarray] | None = None,
    args=None,
    soft_info: dict | None = None,
    restore_rays: bool = True,
) -> np.ndarray:
    """Compose a repaired core and the shared V256 envelope shell once."""
    del guidance, targets, soft_info, restore_rays

    envelope = np.asarray(hull_256, dtype=bool)
    geometry = np.asarray(core, dtype=bool)
    if geometry.shape == envelope.shape:
        lifted = geometry.copy()
    elif geometry.ndim == 3 and len(set(geometry.shape)) == 1:
        lifted = pipeline.upsample_nearest_to(geometry, envelope.shape[0])
    else:
        raise ValueError(
            f"Unsupported recovered geometry shape {geometry.shape}; "
            f"expected a cubic core or {envelope.shape}."
        )

    delivered = lifted & envelope
    shell_voxels = 0
    if include_shell:
        envelope_shell = np.asarray(shell, dtype=bool)
        if envelope_shell.shape != envelope.shape:
            raise ValueError(
                f"Shell shape {envelope_shell.shape} does not match "
                f"delivery envelope {envelope.shape}."
            )
        delivered |= envelope_shell
        shell_voxels = int(envelope_shell.sum())

    _, components = label(delivered, structure=pipeline.S6)
    print(
        json.dumps(
            {
                "stage": "recovered_july22_delivery",
                "geometry_resolution": int(geometry.shape[0]),
                "delivery_resolution": int(envelope.shape[0]),
                "shell_voxels": shell_voxels,
                "components": int(components),
                "post_delivery_ray_restoration": False,
                "post_delivery_global_bridge": False,
            }
        ),
        flush=True,
    )
    return delivered.astype(bool)


def main() -> None:
    pipeline._generic_connected_delivery = recovered_connected_delivery
    pipeline.main()


if __name__ == "__main__":
    main()
