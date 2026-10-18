#!/usr/bin/env python3
"""Replay fixed-view delivery from the principal soft-proposal body.

This recovery runner is intentionally isolated from the production pipeline.
It tests the behavior inferred from the verified July 22 artifacts:

1. the soft proposal defines one high-fidelity V256 principal body;
2. thresholded C48 states provide the structural/material core;
3. one shared V256 boundary shell preserves the declared projections; and
4. SDF meshing is performed only after the core and shell are combined.

The mature two-wall implementation and the production generic implementation
are not modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, label

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadowforge_pipeline import unified_run as pipeline


def _principal_component(volume: np.ndarray) -> np.ndarray:
    """Return the largest 6-connected foreground component."""
    source = np.asarray(volume, dtype=bool)
    labels, count = label(source, structure=pipeline.S6)
    if count == 0:
        raise RuntimeError("Cannot recover a principal body from an empty volume")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    principal = labels == int(np.argmax(sizes))
    print(
        json.dumps(
            {
                "stage": "principal_body",
                "resolution": int(source.shape[0]),
                "components_before": int(count),
                "source_voxels": int(source.sum()),
                "principal_voxels": int(principal.sum()),
                "retained_fraction": float(
                    principal.sum() / max(int(source.sum()), 1)
                ),
            }
        ),
        flush=True,
    )
    return principal


def principal_body_envelope(
    targets: list[np.ndarray],
    args,
    resolution: int,
    soft_info: dict | None = None,
    preserve_xyz: np.ndarray | None = None,
) -> np.ndarray:
    """Use the validated soft principal body as the shared view envelope."""
    del targets, args, soft_info
    if preserve_xyz is None:
        raise RuntimeError(
            "Principal-body recovery requires the frozen soft proposal"
        )
    preserve = np.asarray(preserve_xyz, dtype=bool)
    if preserve.shape != (resolution, resolution, resolution):
        raise ValueError(
            f"Expected a {resolution}^3 proposal, got {preserve.shape}"
        )
    return _principal_component(preserve)


def principal_body_delivery(
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
    """Combine one lifted C48 core with one fixed V256 principal-body shell."""
    del guidance, targets, soft_info, restore_rays
    envelope = np.asarray(hull_256, dtype=bool)
    geometry = np.asarray(core, dtype=bool)
    if geometry.shape == envelope.shape:
        lifted = geometry.copy()
    elif geometry.shape[0] < envelope.shape[0]:
        lifted = pipeline.upsample_nearest_to(
            geometry, envelope.shape[0]
        )
    else:
        raise ValueError(
            f"Geometry {geometry.shape} exceeds envelope {envelope.shape}"
        )

    delivery = lifted & envelope
    shell_voxels = 0
    if include_shell:
        boundary = np.asarray(shell, dtype=bool)
        shell_voxels = int(boundary.sum())
        delivery |= boundary

    # A one-voxel attachment band avoids numerical separation between a
    # nearest-neighbor lifted core and the fixed high-resolution boundary.
    delivery |= (
        binary_dilation(delivery, structure=pipeline.S6, iterations=1)
        & envelope
    )
    before_fill = int(delivery.sum())
    delivery = binary_fill_holes(delivery)
    filled_voxels = int(delivery.sum()) - before_fill
    _, components = label(delivery, structure=pipeline.S6)
    print(
        json.dumps(
            {
                "stage": "principal_body_delivery",
                "components": int(components),
                "core_voxels": int(lifted.sum()),
                "shell_voxels": shell_voxels,
                "delivered_voxels": int(delivery.sum()),
                "filled_internal_voxels": filled_voxels,
            }
        ),
        flush=True,
    )
    return delivery


def main() -> None:
    pipeline._generic_visual_hull = principal_body_envelope
    pipeline._generic_connected_delivery = principal_body_delivery
    pipeline.main()


if __name__ == "__main__":
    main()
