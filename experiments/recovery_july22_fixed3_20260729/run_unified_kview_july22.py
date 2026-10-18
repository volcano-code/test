#!/usr/bin/env python3
"""Replay the verified July 22 unified K-view delivery semantics.

This compatibility runner is intentionally isolated from the production
pipeline. It restores the fixed-view implementation used by the verified
July 22 artifacts while reusing the current CLI, metrics, selection, and
export code. The mature two-wall implementation is never modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadowforge_pipeline import unified_run as pipeline


def july22_visual_hull(
    targets: list[np.ndarray],
    args,
    resolution: int,
    soft_info: dict | None = None,
    preserve_xyz: np.ndarray | None = None,
) -> np.ndarray:
    """Build the historical direct-TFS projection envelope in physical XYZ."""
    del soft_info, preserve_xyz
    if not args.tsf:
        return pipeline._production_generic_visual_hull(
            targets,
            args,
            resolution,
            soft_info=None,
            preserve_xyz=None,
        )

    n_base = len(args.silhouettes)
    if n_base not in (2, 3):
        raise ValueError(
            "The verified direct-TFS implementation supports K=2 or K=3, "
            f"received K={n_base}"
        )

    masks: list[np.ndarray] = []
    for index, target in enumerate(targets):
        base = index % n_base
        image = np.fliplr(target) if index >= n_base else np.asarray(target)
        if base == 0:
            image = np.flipud(image)
            mask = np.repeat(image[:, :, None], resolution, axis=2)
        elif base == 1:
            image = np.flipud(image)
            mask = np.repeat(image[:, None, :], resolution, axis=1)
        else:
            mask = np.repeat(image[None, :, :], resolution, axis=0)
        masks.append(mask)

    hull_renderer_order = np.logical_and.reduce(masks)
    if not hull_renderer_order.any():
        raise RuntimeError("The July 22 direct-TFS envelope is empty")
    return np.transpose(hull_renderer_order, (2, 1, 0)).astype(bool)


def july22_seed_trajectory(
    guidance: np.ndarray,
    hull: np.ndarray,
    thresholds: tuple[float, ...],
    bridge_radius: int,
    bridge_passes: int,
    bridge: bool = True,
    targets: list[np.ndarray] | None = None,
    args=None,
    soft_info: dict | None = None,
    restore_rays: bool = True,
) -> list[tuple[str, np.ndarray]]:
    """Threshold and minimally connect each core exactly once."""
    del targets, args, soft_info, restore_rays
    from voxel_ops import connect6_min_bridges

    # The July 22 runner transposed the downsampled canonical probability once
    # more before intersecting it with the direct-TFS envelope. Preserve that
    # historical frame mapping here; the current runner no longer performs it.
    guidance_historical = np.transpose(np.asarray(guidance), (2, 1, 0))
    result: list[tuple[str, np.ndarray]] = []
    envelope = np.asarray(hull, dtype=bool)
    for threshold in thresholds:
        seed = (guidance_historical >= threshold) & envelope
        if not seed.any():
            continue
        if bridge:
            repaired, _ = connect6_min_bridges(
                seed,
                envelope,
                max_passes=bridge_passes,
                bridge_radius=bridge_radius,
            )
            label_name = f"soft_seed_t{threshold:.2f}:bridge6r{bridge_radius}"
        else:
            repaired = seed
            label_name = f"soft_seed_t{threshold:.2f}:raw"
        result.append((label_name, np.asarray(repaired, dtype=bool)))

    if not result:
        raise RuntimeError("No July 22 threshold seed remained in the envelope")
    return result


def july22_connected_delivery(
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
    """Lift C48, add the shared thin V256 shell, then minimally connect once."""
    del guidance, targets, soft_info, restore_rays
    from voxel_ops import connect6_min_bridges

    envelope = np.asarray(hull_256, dtype=bool)
    geometry = np.asarray(core, dtype=bool)
    lifted = pipeline.upsample_nearest_to(geometry, envelope.shape[0])
    delivery = lifted & envelope
    if include_shell:
        delivery |= np.asarray(shell, dtype=bool)
    connected, _ = connect6_min_bridges(
        delivery,
        envelope,
        max_passes=8,
        bridge_radius=1,
    )
    return np.asarray(connected, dtype=bool)


def july22_render(
    volume_xyz: np.ndarray,
    args,
    soft_info: dict | None = None,
) -> list[np.ndarray]:
    """Render direct-TFS views with the axis convention used on July 22."""
    if not args.tsf:
        return pipeline._production_generic_render(volume_xyz, args, soft_info)

    from renderer_voxel import render_tsf_orthographic

    density = pipeline.torch.from_numpy(
        np.ascontiguousarray(
            np.transpose(np.asarray(volume_xyz, dtype=np.float32), (2, 1, 0))
        )
    )[None, None]
    density = (density * 0.999).to(args.device)
    with pipeline.torch.no_grad():
        images = render_tsf_orthographic(
            density,
            mirror=bool(args.mr),
            num_base=len(args.silhouettes),
        )
    return [
        (image >= 0.5).detach().cpu().numpy().astype(bool)
        for image in images
    ]


def july22_export_stl_at_print_scale(
    volume: np.ndarray,
    path: Path,
    print_mm: float,
):
    """Export through the exact July 22 two-wall exterior-mesh adapter."""
    mesh = pipeline.n2.export_stl(
        np.transpose(np.asarray(volume, dtype=bool), (2, 1, 0)),
        path,
    )
    extent = float(np.max(mesh.extents))
    if extent <= 0:
        raise ValueError("Cannot scale an empty delivery mesh")
    mesh.apply_scale(float(print_mm) / extent)
    mesh.export(path)
    return mesh


def _append_flag(flag: str) -> None:
    if flag not in sys.argv:
        sys.argv.append(flag)


def _append_option(option: str, value: str) -> None:
    if option not in sys.argv:
        sys.argv.extend((option, value))


def main() -> None:
    # Retain handles only for the non-TFS fallback; production functions are
    # otherwise untouched outside this process.
    pipeline._production_generic_visual_hull = pipeline._generic_visual_hull
    pipeline._production_generic_render = pipeline._generic_render
    pipeline._generic_visual_hull = july22_visual_hull
    pipeline._generic_seed_trajectory = july22_seed_trajectory
    pipeline._generic_connected_delivery = july22_connected_delivery
    pipeline._generic_render = july22_render
    pipeline.export_stl_at_print_scale = july22_export_stl_at_print_scale

    _append_flag("--disable-guarded-carve")
    _append_flag("--disable-ray-restoration")
    _append_flag("--full-envelope-delivery-shell")
    _append_option("--delivery-shortlist-size", "0")
    pipeline.main()


if __name__ == "__main__":
    main()
