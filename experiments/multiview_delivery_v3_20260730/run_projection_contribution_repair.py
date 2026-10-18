#!/usr/bin/env python3
"""Projection-contribution repair for isolated multi-view V256 delivery.

This adapter leaves the mature two-view pipeline and the v2 multi-view runner
untouched. It replaces only v2's bounded repair policy:

1. rank detached proposal components by recovered target pixels;
2. reject components that introduce disproportionate false-positive support;
3. reconnect accepted components with a strictly 6-connected path whose
   projections remain inside the declared target support whenever possible.
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
V2_PATH = (
    ROOT
    / "experiments"
    / "multiview_delivery_v2_20260728"
    / "run_multiview_delivery_v2.py"
)


def load_v2():
    spec = importlib.util.spec_from_file_location("pcraft_multiview_v2", V2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import isolated v2 runner: {V2_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v2 = load_v2()


def view_specs(
    targets: list[np.ndarray],
    soft_info: dict,
) -> list[tuple[int, np.ndarray]]:
    if not v2.pipeline._uses_sadekar_direct_tfs(soft_info):
        raise RuntimeError(
            "Projection-contribution repair requires recorded direct-TFS cameras."
        )
    specs: list[tuple[int, np.ndarray]] = []
    for index, declared_target in enumerate(targets):
        axis, transform = v2.pipeline._sadekar_tfs_view_spec(index)
        target = v2.pipeline._transform_sadekar_tfs_target(
            declared_target, transform
        )
        specs.append((axis, np.asarray(target, dtype=bool)))
    return specs


def orthogonal_path(
    start: np.ndarray,
    end: np.ndarray,
    axis_order: tuple[int, int, int],
) -> np.ndarray:
    """Return a Manhattan path with exactly one-axis moves."""
    point = np.asarray(start, dtype=np.int64).copy()
    destination = np.asarray(end, dtype=np.int64)
    path = [point.copy()]
    for axis in axis_order:
        delta = int(destination[axis] - point[axis])
        direction = 1 if delta > 0 else -1
        for _ in range(abs(delta)):
            point = point.copy()
            point[axis] += direction
            path.append(point)
    return np.asarray(path, dtype=np.int64)


def projected_support_counts(
    coords: np.ndarray,
    specs: list[tuple[int, np.ndarray]],
    root_covers: dict[int, np.ndarray],
) -> tuple[int, int, int]:
    """Count unique missing-target, target, and new false-positive pixels."""
    recovered = 0
    target_support = 0
    false_positive = 0
    for axis, target in specs:
        plane_dims = [dimension for dimension in range(3) if dimension != axis]
        flat_indices = np.ravel_multi_index(
            (coords[:, plane_dims[0]], coords[:, plane_dims[1]]),
            tuple(target.shape),
        )
        flat_indices = np.unique(flat_indices)
        target_flat = target.ravel()
        cover_flat = root_covers[axis].ravel()
        on_target = target_flat[flat_indices]
        already_covered = cover_flat[flat_indices]
        recovered += int(np.count_nonzero(on_target & ~already_covered))
        target_support += int(np.count_nonzero(on_target))
        false_positive += int(
            np.count_nonzero(~on_target & ~already_covered)
        )
    return recovered, target_support, false_positive


def update_covers(
    root_covers: dict[int, np.ndarray],
    coords: np.ndarray,
) -> None:
    for axis, cover in root_covers.items():
        plane_dims = [dimension for dimension in range(3) if dimension != axis]
        cover[
            coords[:, plane_dims[0]],
            coords[:, plane_dims[1]],
        ] = True


def best_orthogonal_bridge(
    start: np.ndarray,
    end: np.ndarray,
    specs: list[tuple[int, np.ndarray]],
    root_covers: dict[int, np.ndarray],
    probability: np.ndarray,
) -> tuple[np.ndarray, dict]:
    candidates: list[tuple[tuple[int, float], np.ndarray, dict]] = []
    for order in itertools.permutations((0, 1, 2)):
        path = orthogonal_path(start, end, order)
        recovered, target_support, false_positive = projected_support_counts(
            path, specs, root_covers
        )
        mean_probability = float(probability[tuple(path.T)].mean())
        audit = {
            "axis_order": list(order),
            "path_voxels": int(len(path)),
            "recovered_pixels": int(recovered),
            "target_support_pixels": int(target_support),
            "false_positive_pixels": int(false_positive),
            "mean_probability": mean_probability,
        }
        candidates.append(
            ((false_positive, -mean_probability), path, audit)
        )
    _, path, audit = min(candidates, key=lambda item: item[0])
    return path, audit


def contribution_repair(
    current: np.ndarray,
    proposal: np.ndarray,
    probability: np.ndarray,
    targets: list[np.ndarray],
    protocol,
    soft_info: dict,
    args,
) -> tuple[np.ndarray, dict]:
    del protocol
    specs = view_specs(targets, soft_info)
    source = v2.pipeline._trainer_zyx_from_canonical_xyz(
        np.asarray(proposal, dtype=bool)
    )
    root = v2.pipeline._trainer_zyx_from_canonical_xyz(
        np.asarray(current, dtype=bool)
    ).copy()
    priority = v2.pipeline._trainer_zyx_from_canonical_xyz(
        np.asarray(probability, dtype=np.float32)
    )

    labels, sizes = v2.component_labels(source)
    count = max(int(sizes.size - 1), 0)
    if not root.any() or count <= 1:
        return current.copy(), {
            "policy": "projection_contribution",
            "components_before": count,
            "considered_components": 0,
            "attached_components": 0,
            "added_voxels": 0,
        }

    root_labels = labels[root]
    root_labels = root_labels[root_labels > 0]
    root_label = (
        int(np.bincount(root_labels).argmax()) if root_labels.size else -1
    )
    distances, nearest = ndimage.distance_transform_edt(
        ~root, return_indices=True
    )
    label_ids = np.arange(1, sizes.size, dtype=np.int64)
    minimum_distances = ndimage.minimum(
        distances, labels=labels, index=label_ids
    )
    eligible = label_ids[
        (sizes[1:] > 0)
        & (minimum_distances <= float(args.max_repair_gap))
        & (label_ids != root_label)
    ]

    score_limit = max(int(args.max_repair_components) * 16, 512)
    if len(eligible) > score_limit:
        order = np.argsort(sizes[eligible])[::-1][:score_limit]
        eligible = eligible[order]

    objects = ndimage.find_objects(labels)
    root_covers = {axis: root.any(axis=axis) for axis, _ in specs}
    scored: list[tuple[tuple[int, int, float, int], int, dict]] = []
    component_coords: dict[int, np.ndarray] = {}
    for label_id in eligible:
        component_slice = objects[int(label_id) - 1]
        if component_slice is None:
            continue
        local = np.argwhere(labels[component_slice] == int(label_id))
        if not len(local):
            continue
        offsets = np.asarray(
            [axis_slice.start for axis_slice in component_slice],
            dtype=np.int64,
        )
        coords = local + offsets
        recovered, target_support, false_positive = projected_support_counts(
            coords, specs, root_covers
        )
        if recovered <= 0:
            continue
        component_coords[int(label_id)] = coords
        audit = {
            "label": int(label_id),
            "voxels": int(len(coords)),
            "distance": float(minimum_distances[int(label_id) - 1]),
            "recovered_pixels": int(recovered),
            "target_support_pixels": int(target_support),
            "false_positive_pixels": int(false_positive),
        }
        scored.append(
            (
                (
                    int(recovered),
                    -int(false_positive),
                    -float(minimum_distances[int(label_id) - 1]),
                    int(target_support),
                ),
                int(label_id),
                audit,
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)

    before = int(root.sum())
    budget = max(
        1,
        int(
            np.asarray(proposal, dtype=bool).sum()
            * args.max_repair_fraction
        ),
    )
    attached_records: list[dict] = []
    rejected_records: list[dict] = []
    for _, label_id, initial_audit in scored:
        if len(attached_records) >= int(args.max_repair_components):
            break
        coords = component_coords[label_id]
        recovered, target_support, false_positive = projected_support_counts(
            coords, specs, root_covers
        )
        if recovered <= 0:
            continue
        allowed_false_positive = max(1, int(np.ceil(recovered * 0.25)))
        if false_positive > allowed_false_positive:
            rejected_records.append(
                {
                    **initial_audit,
                    "reason": "excess_projection_false_positive",
                    "updated_recovered_pixels": int(recovered),
                    "updated_false_positive_pixels": int(false_positive),
                }
            )
            continue

        component_distances = distances[tuple(coords.T)]
        start = coords[int(np.argmin(component_distances))]
        end = nearest[:, start[0], start[1], start[2]]
        bridge, bridge_audit = best_orthogonal_bridge(
            start, end, specs, root_covers, priority
        )
        bridge_fp_limit = max(1, int(np.ceil(recovered * 0.10)))
        if bridge_audit["false_positive_pixels"] > bridge_fp_limit:
            rejected_records.append(
                {
                    **initial_audit,
                    "reason": "bridge_leaves_target_support",
                    "bridge": bridge_audit,
                }
            )
            continue

        added_mask = np.zeros_like(root, dtype=bool)
        added_mask[tuple(coords.T)] = True
        added_mask[tuple(bridge.T)] = True
        added = int(np.count_nonzero(added_mask & ~root))
        if int(root.sum()) - before + added > budget:
            rejected_records.append(
                {
                    **initial_audit,
                    "reason": "repair_budget",
                    "proposed_added_voxels": int(added),
                }
            )
            continue

        root |= added_mask
        update_covers(root_covers, np.argwhere(added_mask))
        attached_records.append(
            {
                **initial_audit,
                "updated_recovered_pixels": int(recovered),
                "updated_target_support_pixels": int(target_support),
                "updated_false_positive_pixels": int(false_positive),
                "added_voxels": int(added),
                "bridge": bridge_audit,
            }
        )

    connected, component_audit = v2.largest_component(root)
    result = v2.pipeline._canonical_xyz_from_trainer_zyx(connected)
    return result, {
        "policy": "projection_contribution",
        "components_before": int(count),
        "root_label": int(root_label),
        "eligible_nearby_components": int(len(eligible)),
        "scored_components": int(len(scored)),
        "considered_components": int(len(scored)),
        "attached_components": int(len(attached_records)),
        "added_voxels": int(connected.sum()) - before,
        "repair_budget_voxels": int(budget),
        "attached": attached_records,
        "rejected": rejected_records[:128],
        **component_audit,
    }


v2.attempt_bounded_repair = contribution_repair


if __name__ == "__main__":
    v2.main()
