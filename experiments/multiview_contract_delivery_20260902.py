#!/usr/bin/env python3
"""Staging-only, immutable-contract multiview delivery backend."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

GENERIC = "custom_orthographic"
DIRECT = "sadekar_voxel_tfs_mirror_orthographic"
FRAME = "flip(transpose(zyx,(2,1,0)),axes=(0,2))"
EXTERNAL = Path(
    "/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel"
)
FACE6 = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
CONTRACT_FIELDS = (
    "protocol_id",
    "status",
    "requested_tsf",
    "effective_tsf",
    "mirror",
    "trainer_tsf",
    "trainer_tsf4",
    "views",
    "renderer_family",
    "direct_view_roles",
    "image_size",
    "zdist",
    "volume_extent",
    "ray_samples",
    "input_sha256",
    "normalized_target_sha256",
    "target_order",
    "soft_zyx_to_canonical_xyz",
    "preserve_target_aspect",
    "no_registration",
    "artifact_sha256",
)


@dataclass(frozen=True)
class DeliveryConfig:
    core_resolution: int = 48
    delivery_resolution: int = 256
    mean_iou_gate: float = 0.95
    worst_iou_gate: float = 0.90
    minimum_saving: float = 1e-6
    maximum_saving: float = 0.85
    minimum_erosion_survival: float = 0.08
    max_checkpoints: int = 256
    shell_voxels: int = 2
    sdf_pad: int = 2
    sdf_sigma: float = 0.4
    sdf_level: float = 0.0
    max_faces: int = 2_000_000
    max_dimension_mm: float = 100.0

    def __post_init__(self) -> None:
        integer_fields = (
            "core_resolution",
            "delivery_resolution",
            "max_checkpoints",
            "shell_voxels",
            "sdf_pad",
            "max_faces",
        )
        for name in integer_fields:
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact int")
        float_fields = (
            "mean_iou_gate",
            "worst_iou_gate",
            "minimum_saving",
            "maximum_saving",
            "minimum_erosion_survival",
            "sdf_sigma",
            "sdf_level",
            "max_dimension_mm",
        )
        for name in float_fields:
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be an exact float")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not (0 < self.core_resolution <= self.delivery_resolution):
            raise ValueError("invalid resolutions")
        if not 1 <= self.max_checkpoints <= 256:
            raise ValueError("max_checkpoints must be in [1,256]")
        absolute = (
            self.mean_iou_gate,
            self.worst_iou_gate,
            self.minimum_saving,
            self.maximum_saving,
            self.minimum_erosion_survival,
            self.shell_voxels,
            self.max_faces,
            self.max_dimension_mm,
        )
        if absolute != (0.95, 0.90, 1e-6, 0.85, 0.08, 2, 2_000_000, 100.0):
            raise ValueError("absolute delivery gates are frozen")
        if (
            self.sdf_pad < 1
            or self.sdf_sigma < 0.0
            or not 0.0 <= self.mean_iou_gate <= 1.0
            or not 0.0 <= self.worst_iou_gate <= 1.0
            or not 0.0 <= self.minimum_saving <= self.maximum_saving <= 1.0
            or not 0.0 <= self.minimum_erosion_survival <= 1.0
            or self.max_dimension_mm <= 0.0
        ):
            raise ValueError("invalid frozen delivery configuration")


class DeliveryFailure(RuntimeError):
    status: str
    reason: str
    details: dict[str, object]

    def __init__(
        self, status: str, reason: str, details: Mapping[str, object] | None = None
    ) -> None:
        super().__init__(reason)
        self.status, self.reason = str(status), str(reason)
        self.details = dict(details or {})


def _cube(value: Any, dtype: Any, name: str) -> np.ndarray:
    out = np.ascontiguousarray(value, dtype=dtype)
    if out.ndim != 3 or len(set(out.shape)) != 1:
        raise ValueError(f"{name} must be a contiguous cube")
    return out


def trainer_zyx_to_canonical_xyz(volume_zyx: Any) -> np.ndarray:
    value = np.asarray(volume_zyx)
    if value.ndim != 3:
        raise ValueError("volume must be 3-D")
    return np.ascontiguousarray(np.flip(np.transpose(value, (2, 1, 0)), (0, 2)))


def canonical_xyz_to_trainer_zyx(volume_xyz: Any) -> np.ndarray:
    value = np.asarray(volume_xyz)
    if value.ndim != 3:
        raise ValueError("volume must be 3-D")
    return np.ascontiguousarray(np.transpose(np.flip(value, (0, 2)), (2, 1, 0)))


def _validate_contract(contract: object) -> None:
    missing = [name for name in CONTRACT_FIELDS if not hasattr(contract, name)]
    if missing:
        raise DeliveryFailure(
            "CONTRACT_INVALID", "contract fields missing", {"missing": missing}
        )
    views, roles = tuple(contract.views), tuple(contract.direct_view_roles)  # type: ignore[attr-defined]
    if not views or len(views) != len(roles):
        raise DeliveryFailure("CONTRACT_INVALID", "view/role count mismatch")
    family = contract.renderer_family  # type: ignore[attr-defined]
    if family not in (GENERIC, DIRECT):
        raise DeliveryFailure("CONTRACT_INVALID", "unknown renderer family")
    if family == GENERIC and any(tuple(role) != (None, None) for role in roles):
        raise DeliveryFailure("CONTRACT_INVALID", "generic views declare direct roles")
    if family == DIRECT:
        transforms = {"ud", "tudlr", "udlr", "tlr"}
        if any(
            len(role) != 2 or role[0] not in (0, 1, 2) or role[1] not in transforms
            for role in roles
        ):
            raise DeliveryFailure("CONTRACT_INVALID", "direct ray roles malformed")
        if any(
            getattr(contract, name) is None
            for name in ("zdist", "volume_extent", "ray_samples")
        ):
            raise DeliveryFailure("CONTRACT_INVALID", "direct camera scalars missing")
    if contract.soft_zyx_to_canonical_xyz != FRAME:  # type: ignore[attr-defined]
        raise DeliveryFailure("CONTRACT_INVALID", "frame mapping mismatch")
    if contract.preserve_target_aspect is not True or contract.no_registration is not True:  # type: ignore[attr-defined]
        raise DeliveryFailure(
            "CONTRACT_INVALID", "aspect/registration contract mismatch"
        )


def _external(name: str, identity: str) -> Any:
    path = EXTERNAL / name
    spec = importlib.util.spec_from_file_location(identity, path)
    if not path.is_file() or spec is None or spec.loader is None:
        raise DeliveryFailure(
            "RENDERER_UNAVAILABLE", "external renderer unavailable", {"path": str(path)}
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_contract(
    volume_xyz: Any,
    contract: object,
    *,
    renderer: Callable[..., object] | None = None,
    device: str = "cpu",
) -> list[np.ndarray]:
    volume = _cube(volume_xyz, bool, "volume")
    _validate_contract(contract)
    family, views = contract.renderer_family, tuple(contract.views)  # type: ignore[attr-defined]
    if renderer is not None:
        output = []
        for i, (az, el) in enumerate(views):
            mask = np.ascontiguousarray(
                renderer(
                    volume_xyz=volume,
                    family=family,
                    view_index=i,
                    azimuth_deg=float(az),
                    elevation_deg=float(el),
                    contract=contract,
                    device=device,
                ),
                dtype=bool,
            )
            if mask.shape != volume.shape[:2]:
                raise DeliveryFailure("RENDER_FAILED", "renderer returned wrong shape")
            output.append(mask)
        return output
    import torch

    zyx = canonical_xyz_to_trainer_zyx(volume)
    density = (
        torch.from_numpy(np.ascontiguousarray(zyx, dtype=np.float32))[None, None].to(
            device
        )
        * 0.999
    )
    output = []
    with torch.no_grad():
        if family == GENERIC:
            fn = _external("renderer_voxel.py", "contract_renderer").render_orthographic
            output = [
                (fn(density, float(a), float(e)) >= 0.5).cpu().numpy().astype(bool)
                for a, e in views
            ]
        else:
            cls = _external(
                "sadekar_protocol.py", "contract_sadekar"
            ).SadekarTFSProtocol
            protocol = cls(
                torch.device(device),
                volume.shape[0],
                float(contract.zdist),
                float(contract.volume_extent),
                int(contract.ray_samples),
            )  # type: ignore[attr-defined]
            output = [
                protocol.binary_mask(protocol.render(density, i))
                .cpu()
                .numpy()
                .astype(bool)
                for i in range(len(views))
            ]
    return [np.ascontiguousarray(item) for item in output]


def _camera_matrix(az: float, el: float, device: Any) -> Any:
    import torch

    a, e = np.deg2rad(az), np.deg2rad(el)
    cz, sz, cx, sx = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    return torch.tensor(
        [
            [cz, -sz, 0.0, 0.0],
            [cx * sz, cx * cz, -sx, 0.0],
            [sx * sz, sx * cz, cx, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def _direct_target(target: Any, transform: str) -> np.ndarray:
    image = np.asarray(target, dtype=bool)
    if transform.startswith("t"):
        image = image.T
    if "ud" in transform:
        image = np.flipud(image)
    if "lr" in transform:
        image = np.fliplr(image)
    return np.ascontiguousarray(image)


def _direct_target_planes(targets: Sequence[Any], contract: object, resolution: int):
    """Map image pixels to voxel centres using the actual orthographic cameras.

    The image covers NDC [-1,1], not the volume extent. Flip/transpose alone
    silently applies an additional .85 scale for the extent=1.7 protocol.
    These are geometric centre-ray constraints, not an optical-depth inverse.
    """
    import cv2
    import torch
    from pytorch3d.structures import Volumes

    cls = _external("sadekar_protocol.py", "contract_backprojection").SadekarTFSProtocol
    protocol = cls(torch.device("cpu"), resolution, float(contract.zdist),
                   float(contract.volume_extent), int(contract.ray_samples))
    world = Volumes(
        densities=torch.ones(1, 1, resolution, resolution, resolution),
        voxel_size=float(contract.volume_extent) / resolution,
    ).get_coord_grid(world_coordinates=True)[0]
    planes = []
    for i, (target, (axis, _)) in enumerate(zip(targets, contract.direct_view_roles)):
        image = np.asarray(target, dtype=np.uint8)
        if image.shape != (resolution, resolution):
            raise DeliveryFailure("TARGET_MISMATCH", "target shape mismatch")
        points = world.select(axis, resolution // 2)
        ndc = protocol.cameras[i].transform_points_ndc(points.reshape(-1, 3))
        xy = ((1 - ndc[:, :2]) * resolution / 2 - .5).reshape(resolution, resolution, 2)
        xy = xy.numpy().astype(np.float32)
        planes.append(cv2.remap(image, xy[..., 0], xy[..., 1],
                                interpolation=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT) > 0)
    return planes


def backproject_contract(
    targets: Sequence[Any], contract: object, resolution: int, *, device: str = "cpu"
) -> np.ndarray:
    _validate_contract(contract)
    if len(targets) != len(contract.views):  # type: ignore[attr-defined]
        raise DeliveryFailure("TARGET_MISMATCH", "target/view count mismatch")
    masks = []
    if contract.renderer_family == DIRECT:  # type: ignore[attr-defined]
        planes = _direct_target_planes(targets, contract, resolution)
        for image, (axis, _) in zip(planes, contract.direct_view_roles):
            masks.append(
                np.repeat(np.expand_dims(image, axis=axis), resolution, axis=axis)
            )
        hull = trainer_zyx_to_canonical_xyz(np.logical_and.reduce(masks))
    else:
        import torch
        import torch.nn.functional as F

        dev = torch.device(device)
        for target, (az, el) in zip(targets, contract.views):  # type: ignore[attr-defined]
            image = np.ascontiguousarray(target, dtype=bool)
            if image.shape != (resolution, resolution):
                raise DeliveryFailure("TARGET_MISMATCH", "target shape mismatch")
            camera = torch.from_numpy(
                np.repeat(image[:, :, None], resolution, 2).astype(np.float32)
            )[None, None].to(dev)
            inverse = torch.linalg.inv(_camera_matrix(float(az), float(el), dev))[:3][
                None
            ]
            grid = F.affine_grid(inverse, camera.shape, align_corners=False)
            masks.append(
                F.grid_sample(
                    camera,
                    grid,
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=False,
                )[0, 0]
                .cpu()
                .numpy()
                >= 0.5
            )
        hull = trainer_zyx_to_canonical_xyz(np.logical_and.reduce(masks))
    if not hull.any():
        raise DeliveryFailure("EMPTY_ENVELOPE", "contract envelope empty")
    return np.ascontiguousarray(hull, dtype=bool)


def restore_contract_rays(
    volume_xyz: Any,
    envelope_xyz: Any,
    guidance_xyz: Any,
    targets: Sequence[Any],
    contract: object,
    *,
    device: str = "cpu",
    rounds: int = 2,
) -> np.ndarray:
    volume, hull = _cube(volume_xyz, bool, "volume").copy(), _cube(
        envelope_xyz, bool, "hull"
    )
    score = _cube(guidance_xyz, np.float32, "guidance")
    if volume.shape != hull.shape or score.shape != hull.shape:
        raise ValueError("restore shapes differ")
    _validate_contract(contract)
    volume &= hull
    if contract.renderer_family == DIRECT:  # type: ignore[attr-defined]
        occ, adm, guide = map(canonical_xyz_to_trainer_zyx, (volume, hull, score))
        planes = _direct_target_planes(targets, contract, volume.shape[0])
        for target, (axis, _) in zip(planes, contract.direct_view_roles):
            missing = target & ~occ.any(axis=axis) & adm.any(axis=axis)
            winners = np.argmax(np.where(adm, guide, -np.inf), axis=axis)
            selected = np.zeros_like(occ)
            np.put_along_axis(
                selected,
                np.expand_dims(winners, axis),
                np.expand_dims(missing, axis),
                axis=axis,
            )
            occ |= selected & adm
        return trainer_zyx_to_canonical_xyz(occ) & hull
    if not 0 <= rounds <= 2:
        raise ValueError("generic restore is limited to two rounds")
    import torch
    import torch.nn.functional as F

    dev = torch.device(device)

    def convert(array: np.ndarray) -> Any:
        return torch.from_numpy(canonical_xyz_to_trainer_zyx(array.astype(np.float32)))[
            None, None
        ].to(dev)

    world, camera_hull, guidance = convert(volume), convert(hull), convert(score)
    with torch.no_grad():
        for _ in range(rounds):
            changed = False
            for target_np, (az, el) in zip(targets, contract.views):  # type: ignore[attr-defined]
                matrix = _camera_matrix(float(az), float(el), dev)[:3][None]
                grid = F.affine_grid(matrix, world.shape, align_corners=False)
                occ = F.grid_sample(
                    world,
                    grid,
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=False,
                )
                adm = F.grid_sample(
                    camera_hull,
                    grid,
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=False,
                )
                scores = F.grid_sample(
                    guidance,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                target = torch.from_numpy(
                    np.ascontiguousarray(target_np, dtype=bool)
                ).to(dev)
                missing = target & ~occ[0, 0].bool().any(2) & adm[0, 0].bool().any(2)
                if not bool(missing.any()):
                    continue
                winners = (
                    scores[0, 0]
                    .masked_fill(~adm[0, 0].bool(), -torch.inf)
                    .argmax(2, keepdim=True)
                )
                selected = torch.zeros_like(occ)
                selected[0, 0].scatter_(2, winners, missing.unsqueeze(2).to(occ.dtype))
                inverse = torch.linalg.inv(_camera_matrix(float(az), float(el), dev))[
                    :3
                ][None]
                inverse_grid = F.affine_grid(
                    inverse, selected.shape, align_corners=False
                )
                selected = F.grid_sample(
                    selected,
                    inverse_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                before = int(world.count_nonzero())
                world = torch.maximum(world, (selected > 1e-6) * camera_hull)
                changed |= int(world.count_nonzero()) > before
            if not changed:
                break
    return np.ascontiguousarray(
        trainer_zyx_to_canonical_xyz(world[0, 0].cpu().numpy() >= 0.5) & hull
    )


def _components(volume: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy.ndimage import generate_binary_structure, label

    return label(volume, structure=generate_binary_structure(3, 1))


def bridge6_restricted(
    volume_xyz: Any, envelope_xyz: Any, *, max_passes: int, radius: int
) -> tuple[np.ndarray, dict[str, object]]:
    from scipy.ndimage import binary_dilation, generate_binary_structure

    volume, hull = _cube(volume_xyz, bool, "volume").copy(), _cube(
        envelope_xyz, bool, "hull"
    )
    if volume.shape != hull.shape or np.any(volume & ~hull):
        raise DeliveryFailure("BRIDGE_FAILED", "bridge input outside envelope")
    added = passes = 0
    for _ in range(max_passes):
        labels, count = _components(volume)
        if count <= 1:
            break
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        root = int(np.argmax(sizes[1:]) + 1)
        target = next(i for i in range(1, count + 1) if i != root)
        queue = deque(
            int(np.ravel_multi_index(tuple(p), volume.shape))
            for p in np.argwhere(labels == target)
        )
        parent = np.full(volume.size, -1, dtype=np.int64)
        seen = np.zeros(volume.size, dtype=bool)
        seen[list(queue)] = True
        found = -1
        while queue and found < 0:
            flat = queue.popleft()
            point = np.unravel_index(flat, volume.shape)
            for delta in FACE6:
                q = tuple(point[d] + delta[d] for d in range(3))
                if (
                    any(q[d] < 0 or q[d] >= volume.shape[d] for d in range(3))
                    or not hull[q]
                ):
                    continue
                qflat = int(np.ravel_multi_index(q, volume.shape))
                if seen[qflat]:
                    continue
                seen[qflat] = True
                parent[qflat] = flat
                if labels[q] == root:
                    found = qflat
                    break
                queue.append(qflat)
        if found < 0:
            raise DeliveryFailure(
                "BRIDGE_FAILED",
                "no in-envelope Manhattan route connects the components",
                {"component_count_6": int(count)},
            )
        bridge = np.zeros_like(volume)
        cursor = found
        while cursor >= 0:
            bridge[np.unravel_index(cursor, volume.shape)] = True
            cursor = int(parent[cursor])
        if radius:
            bridge = (
                binary_dilation(
                    bridge, generate_binary_structure(3, 1), iterations=radius
                )
                & hull
            )
        gained = bridge & ~volume
        added += int(gained.sum())
        volume |= bridge
        passes += 1
    count = _components(volume)[1]
    if count != 1:
        raise DeliveryFailure(
            "BRIDGE_FAILED",
            "restricted Bridge6 exhausted pass cap",
            {"component_count_6": int(count)},
        )
    return volume, {
        "added_voxels": added,
        "passes": passes,
        "component_count_6": 1,
        "envelope_contained": not bool(np.any(volume & ~hull)),
    }


def _simple(volume: np.ndarray, p: Sequence[int]) -> bool:
    x, y, z = map(int, p)
    x0, y0, z0 = max(x - 1, 0), max(y - 1, 0), max(z - 1, 0)
    block = volume[x0 : x + 2, y0 : y + 2, z0 : z + 2].copy()
    c = (x - x0, y - y0, z - z0)
    block[c] = False
    labels, _ = _components(block)
    adjacent = set()
    for d in FACE6:
        q = tuple(c[i] + d[i] for i in range(3))
        if all(0 <= q[i] < block.shape[i] for i in range(3)) and block[q]:
            adjacent.add(int(labels[q]))
    return len(adjacent) <= 1


def _direct_covered(
    volume: np.ndarray, targets: Sequence[Any], contract: object
) -> bool:
    trainer = canonical_xyz_to_trainer_zyx(volume)
    planes = _direct_target_planes(targets, contract, volume.shape[0])
    for declared, (axis, _) in zip(planes, contract.direct_view_roles):
        if not np.all(trainer.any(axis=axis)[declared]):
            return False
    return True


def _carve(
    volume: np.ndarray,
    probability: np.ndarray,
    targets: Sequence[Any],
    contract: object,
    budget: int,
) -> tuple[np.ndarray, int]:
    from scipy.ndimage import binary_erosion, generate_binary_structure

    out = volume.copy()
    boundary = np.argwhere(out & ~binary_erosion(out, generate_binary_structure(3, 1)))
    if len(boundary):
        boundary = boundary[np.argsort(probability[tuple(boundary.T)], kind="stable")]
    removed = 0
    for p in boundary:
        q = tuple(map(int, p))
        if removed >= budget:
            break
        if not out[q] or not _simple(out, p):
            continue
        out[q] = False
        if contract.renderer_family == DIRECT and not _direct_covered(
            out, targets, contract
        ):
            out[q] = True
            continue  # type: ignore[attr-defined]
        removed += 1
    return out, removed


def evaluate_contract(
    volume_xyz: Any,
    targets: Sequence[Any],
    contract: object,
    *,
    renderer: Callable[..., object] | None = None,
    device: str = "cpu",
    hull_voxels: int | None = None,
) -> dict[str, object]:
    from scipy.ndimage import binary_erosion, generate_binary_structure

    volume = _cube(volume_xyz, bool, "volume")
    predictions = render_contract(volume, contract, renderer=renderer, device=device)
    ious = []
    for pred, target in zip(predictions, targets):
        target = np.asarray(target, dtype=bool)
        inter = int(np.logical_and(pred, target).sum())
        union = int(np.logical_or(pred, target).sum())
        ious.append(inter / max(union, 1))
    eroded = binary_erosion(volume, generate_binary_structure(3, 1))
    occupied = int(volume.sum())
    result = {
        "mean_iou": float(np.mean(ious)),
        "worst_iou": float(min(ious)),
        "view_ious": [float(x) for x in ious],
        "component_count_6": int(_components(volume)[1]),
        "occupied_voxels": occupied,
        "erosion_survival": float(eroded.sum() / max(occupied, 1)),
        "surface_fraction": float((volume & ~eroded).sum() / max(occupied, 1)),
        "surface_fraction_role": "diagnostic_only",
    }
    if hull_voxels is not None:
        result["saving"] = float(1 - occupied / max(hull_voxels, 1))
    return result


def _core_pass(
    m: Mapping[str, object], mean: float = 0.95, worst: float = 0.90
) -> bool:
    return (
        int(m["component_count_6"]) == 1
        and float(m["mean_iou"]) >= mean
        and float(m["worst_iou"]) >= worst
    )


def carve_core_trajectory(
    seed_xyz: Any,
    probability_xyz: Any,
    envelope_xyz: Any,
    targets: Sequence[Any],
    contract: object,
    *,
    renderer: Callable[..., object] | None = None,
    device: str = "cpu",
    max_checkpoints: int = 256,
    deletion_budget: int | None = None,
    bridge_max_passes: int,
    bridge_radius: int,
    mean_iou_gate: float = 0.95,
    worst_iou_gate: float = 0.90,
) -> tuple[np.ndarray, dict[str, object]]:
    accepted = _cube(seed_xyz, bool, "seed")
    probability = _cube(probability_xyz, np.float32, "probability")
    hull = _cube(envelope_xyz, bool, "hull")
    checkpoints = []
    accepted_states = [{"checkpoint": "seed", "volume": accepted.copy()}]
    reason = "checkpoint_limit"
    accepted_n = 0
    if not 0 <= max_checkpoints <= 256:
        raise ValueError("max_checkpoints must be in [0,256]")
    for i in range(max_checkpoints):
        budget = (
            int(deletion_budget)
            if deletion_budget is not None
            else max(32, round(0.01 * int(accepted.sum())))
        )
        proposed, deleted = _carve(accepted, probability, targets, contract, budget)
        if not deleted:
            reason = "no_simple_boundary_deletions"
            break
        proposed = restore_contract_rays(
            proposed, hull, probability, targets, contract, device=device
        )
        try:
            proposed, bridge = bridge6_restricted(
                proposed, hull, max_passes=bridge_max_passes, radius=bridge_radius
            )
        except DeliveryFailure as e:
            checkpoints.append({"checkpoint": i, "accepted": False, "reason": e.reason})
            reason = "bridge_failed"
            break
        metrics = evaluate_contract(
            proposed, targets, contract, renderer=renderer, device=device
        )
        passed = _core_pass(metrics, mean_iou_gate, worst_iou_gate)
        checkpoints.append(
            {
                "checkpoint": i,
                "deleted": deleted,
                "accepted": passed,
                "metrics": metrics,
                "bridge": bridge,
            }
        )
        if not passed:
            reason = "core_gate_failed"
            break
        accepted = proposed
        accepted_n += 1
        accepted_states.append({"checkpoint": i, "volume": accepted.copy()})
    return accepted, {
        "accepted_checkpoints": accepted_n,
        "rollback_reason": reason,
        "checkpoints": checkpoints,
        "accepted_states": accepted_states,
        "carving_shape": list(accepted.shape),
    }


def _nearest(volume: np.ndarray, resolution: int) -> np.ndarray:
    idx = np.minimum(
        np.arange(resolution) * volume.shape[0] // resolution, volume.shape[0] - 1
    )
    return np.ascontiguousarray(volume[np.ix_(idx, idx, idx)])


def core_shell_deliver(
    core_xyz: Any,
    envelope_xyz: Any,
    probability_xyz: Any,
    targets: Sequence[Any],
    contract: object,
    *,
    renderer: Callable[..., object] | None = None,
    device: str = "cpu",
    shell_voxels: int = 2,
    bridge_max_passes: int,
    bridge_radius: int,
) -> tuple[np.ndarray, dict[str, object]]:
    from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

    if shell_voxels != 2:
        raise ValueError("CoreShellDeliver requires two shell voxels")
    core = _cube(core_xyz, bool, "core")
    hull = _cube(envelope_xyz, bool, "hull")
    p = _cube(probability_xyz, np.float32, "probability")
    lifted = _nearest(core, hull.shape[0]).astype(bool) & hull
    structure = generate_binary_structure(3, 1)
    local_h_boundary = hull & ~binary_erosion(hull, structure, iterations=shell_voxels)
    local_shell = local_h_boundary & binary_dilation(
        lifted, structure, iterations=shell_voxels
    )
    local_shell_voxels = int((local_shell & ~lifted).sum())
    delivered = lifted | local_shell
    before_restore = delivered.copy()
    delivered = restore_contract_rays(
        delivered, hull, p, targets, contract, device=device
    )
    ray_restored_voxels = int((delivered & ~before_restore).sum())
    bridge = None
    bridge_added_voxels = 0
    if _components(delivered)[1] != 1:
        delivered, bridge = bridge6_restricted(
            delivered, hull, max_passes=bridge_max_passes, radius=bridge_radius
        )
        bridge_added_voxels = int(bridge["added_voxels"])
    return delivered, {
        "lifted_voxels": int(lifted.sum()),
        "shell_added_voxels": local_shell_voxels,
        "local_shell_voxels": local_shell_voxels,
        "ray_restored_voxels": ray_restored_voxels,
        "bridge_added_voxels": bridge_added_voxels,
        "total_added_voxels": int((delivered & ~lifted).sum()),
        "deleted_voxels": 0,
        "delivery_shape": list(delivered.shape),
        "bridge": bridge,
    }


def _edge_counts(part: Any) -> np.ndarray:
    edges = np.sort(
        np.concatenate(
            (part.faces[:, [0, 1]], part.faces[:, [1, 2]], part.faces[:, [2, 0]])
        ),
        1,
    )
    return np.unique(edges, axis=0, return_counts=True)[1]


def _contains_points(mesh: Any, samples: np.ndarray) -> np.ndarray:
    """BVH-backed bounded-memory containment, including thousands of cavities."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

    vertices = vtk.vtkPoints()
    vertices.SetData(numpy_to_vtk(np.asarray(mesh.vertices, np.float64), deep=True))
    cells = np.column_stack((np.full(len(mesh.faces), 3, np.int64), mesh.faces))
    triangles = vtk.vtkCellArray()
    triangles.SetCells(len(cells), numpy_to_vtkIdTypeArray(cells.ravel(), deep=True))
    surface = vtk.vtkPolyData()
    surface.SetPoints(vertices)
    surface.SetPolys(triangles)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(samples, np.float64), deep=True))
    query = vtk.vtkPolyData()
    query.SetPoints(points)
    selector = vtk.vtkSelectEnclosedPoints()
    selector.SetSurfaceData(surface)
    selector.SetInputData(query)
    selector.SetTolerance(1e-7)
    selector.Update()
    return vtk_to_numpy(selector.GetOutput().GetPointData().GetArray("SelectedPoints")).astype(bool)


def _shells(mesh: Any) -> tuple[list[Any], list[int]]:
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise DeliveryFailure("SHELL_QA_FAILED", "empty mesh")
    global_counts = _edge_counts(mesh)
    if np.any(global_counts != 2):
        raise DeliveryFailure(
            "SHELL_QA_FAILED",
            "mesh is not a closed manifold",
            {
                "part": -1,
                "boundary_edges": int((global_counts == 1).sum()),
                "nonmanifold_edges": int((global_counts > 2).sum()),
            },
        )
    # ``repair=False`` is load-bearing: trimesh otherwise seals small boundary
    # defects while splitting, which would turn this audit into an implicit
    # geometry repair and hide the defect from the fail-closed gate.
    parts = list(mesh.split(only_watertight=False, repair=False))
    for i, part in enumerate(parts):
        counts = _edge_counts(part)
        if (
            not part.is_watertight
            or not part.is_winding_consistent
            or np.any(counts != 2)
        ):
            raise DeliveryFailure(
                "SHELL_QA_FAILED",
                "shell is not closed manifold",
                {
                    "part": i,
                    "boundary_edges": int((counts == 1).sum()),
                    "nonmanifold_edges": int((counts > 2).sum()),
                },
            )
    parent = [None] * len(parts)
    volumes = [abs(float(p.volume)) for p in parts]
    bounds = np.asarray([p.bounds for p in parts])
    samples = np.asarray([p.vertices[0] * .999 + p.vertices.mean(0) * .001 for p in parts])
    # One ray query batch per enclosing shell, not one expensive query per
    # cavity. Real 256^3 meshes may contain thousands of legitimate cavities.
    for j, p in enumerate(parts):
        children = np.flatnonzero(np.all(bounds[j, 0] < bounds[:, 0], axis=1)
                                  & np.all(bounds[j, 1] > bounds[:, 1], axis=1))
        if not len(children):
            continue
        inside = _contains_points(p, samples[children])
        for i in children[inside]:
            if parent[i] is None or volumes[j] < volumes[parent[i]]:
                parent[i] = j
    depths = []
    for i in range(len(parts)):
        d = 0
        cursor = parent[i]
        seen = {i}
        while cursor is not None:
            if cursor in seen:
                raise DeliveryFailure("SHELL_QA_FAILED", "containment cycle")
            seen.add(cursor)
            d += 1
            cursor = parent[cursor]
        depths.append(d)
    if depths.count(0) != 1:
        raise DeliveryFailure(
            "SHELL_QA_FAILED",
            "mesh must have exactly one exterior root",
            {"exterior_root_count": depths.count(0)},
        )
    return parts, depths


def _oriented(mesh: Any) -> Any:
    import trimesh

    parts, depths = _shells(mesh)
    result = []
    for part, depth in zip(parts, depths):
        part = part.copy()
        if (float(part.volume) > 0) != (depth % 2 == 0):
            part.invert()
        result.append(part)
    return trimesh.util.concatenate(result)


def audit_shell_containment(mesh: Any) -> dict[str, object]:
    parts, depths = _shells(mesh)
    orientations_valid = all(
        (float(p.volume) > 0) == (d % 2 == 0) for p, d in zip(parts, depths)
    )
    if not orientations_valid:
        raise DeliveryFailure(
            "SHELL_ORIENTATION_FAILED",
            "shell orientation does not match containment parity",
        )
    return {
        "shell_count": len(parts),
        "exterior_root_count": depths.count(0),
        "cavity_shell_count": sum(d % 2 for d in depths),
        "nested_material_shell_count": sum(d > 0 and d % 2 == 0 for d in depths),
        "shell_depths": depths,
        "all_watertight": all(p.is_watertight for p in parts),
        "all_winding_consistent": all(p.is_winding_consistent for p in parts),
        "zero_boundary_edges": True,
        "zero_nonmanifold_edges": True,
        "cavity_orientations_valid": True,
        "parts": [
            {
                "faces": len(p.faces),
                "vertices": len(p.vertices),
                "depth": d,
                "signed_volume": float(p.volume),
            }
            for p, d in zip(parts, depths)
        ],
    }


def padded_sdf_mesh(
    volume_xyz: Any, *, pad: int = 2, sigma: float = 0.4, level: float = 0.0
) -> Any:
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    from skimage.measure import marching_cubes
    import trimesh

    volume = _cube(volume_xyz, bool, "volume")
    if not volume.any() or volume.all():
        raise DeliveryFailure("MESH_FAILED", "invalid SDF volume")
    padded = np.pad(volume, pad)
    sdf = distance_transform_edt(~padded).astype(np.float32) - distance_transform_edt(
        padded
    ).astype(np.float32)
    if sigma:
        sdf = gaussian_filter(sdf, sigma)
    vertices, faces, normals, _ = marching_cubes(sdf, level=level)
    vertices = vertices.astype(np.float64) - pad
    return _oriented(
        trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_normals=normals, process=False
        )
    )


def _raster_part(part: Any, shape: tuple[int, int, int]) -> np.ndarray:
    import vtk
    from vtk.util.numpy_support import (
        numpy_to_vtk,
        numpy_to_vtkIdTypeArray,
        vtk_to_numpy,
    )

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(part.vertices, np.float64), deep=True))
    faces = np.asarray(part.faces, np.int64)
    cells = np.empty((len(faces), 4), np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = faces
    polys = vtk.vtkCellArray()
    polys.SetCells(len(faces), numpy_to_vtkIdTypeArray(cells.ravel(), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(polys)
    image = vtk.vtkImageData()
    image.SetDimensions(*shape)
    image.SetSpacing(1, 1, 1)
    image.SetOrigin(0, 0, 0)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    vtk_to_numpy(image.GetPointData().GetScalars())[:] = 1
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(poly)
    stencil.SetOutputOrigin(image.GetOrigin())
    stencil.SetOutputSpacing(image.GetSpacing())
    stencil.SetOutputWholeExtent(image.GetExtent())
    stencil.Update()
    apply = vtk.vtkImageStencil()
    apply.SetInputData(image)
    apply.SetStencilConnection(stencil.GetOutputPort())
    apply.ReverseStencilOff()
    apply.SetBackgroundValue(0)
    apply.Update()
    return (
        vtk_to_numpy(apply.GetOutput().GetPointData().GetScalars())
        .reshape((shape[2], shape[1], shape[0]))
        .transpose(2, 1, 0)
        .astype(bool)
    )


def rasterize_mesh_with_cavities(mesh: Any, shape: tuple[int, int, int]) -> np.ndarray:
    _shells(mesh)
    # VTK scan conversion applies crossing parity to the entire surface.
    # Passing all shells together preserves voids and avoids allocating a full
    # 256^3 image separately for every microscopic internal cavity.
    return _raster_part(mesh, shape)


def enforce_triangle_cap(
    mesh: Any,
    *,
    max_faces: int = 2_000_000,
    simplifier: Callable[[Any, int], Any] | None,
) -> tuple[Any, dict[str, object]]:
    before = audit_shell_containment(mesh)
    before_signature = _shell_topology_signature(before)
    source = len(mesh.faces)
    if source <= max_faces:
        return mesh, {
            "simplification_applied": False,
            "source_faces": source,
            "result_faces": source,
            "max_faces": max_faces,
        }
    if simplifier is None:
        raise DeliveryFailure(
            "TRIANGLE_CAP_FAILED",
            "triangle cap requires the Open3D simplifier",
            {"source_faces": source, "max_faces": max_faces},
        )
    try:
        result = simplifier(mesh, max_faces)
    except Exception as e:
        raise DeliveryFailure(
            "TRIANGLE_CAP_FAILED",
            "Open3D simplification failed",
            {"error": str(e), "source_faces": source, "max_faces": max_faces},
        ) from e
    if not 0 < len(result.faces) <= max_faces:
        raise DeliveryFailure("TRIANGLE_CAP_FAILED", "simplified mesh exceeds cap")
    result = _oriented(result)
    after = audit_shell_containment(result)
    after_signature = _shell_topology_signature(after)
    if after_signature != before_signature:
        raise DeliveryFailure(
            "TRIANGLE_TOPOLOGY_CHANGED",
            "simplification changed shell containment topology",
            {"before": before_signature, "after": after_signature},
        )
    return result, {
        "simplification_applied": True,
        "source_faces": source,
        "result_faces": len(result.faces),
        "max_faces": max_faces,
    }


def _shell_topology_signature(audit: Mapping[str, object]) -> dict[str, object]:
    return {
        "shell_count": int(audit["shell_count"]),
        "exterior_root_count": int(audit["exterior_root_count"]),
        "cavity_shell_count": int(audit["cavity_shell_count"]),
        "nested_material_shell_count": int(audit["nested_material_shell_count"]),
        "shell_depths": list(audit["shell_depths"]),
    }


def _open3d(mesh: Any, target: int) -> Any:
    import open3d as o3d
    import trimesh

    source = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, np.int32)),
    )
    result = source.simplify_quadric_decimation(target)
    return trimesh.Trimesh(
        np.asarray(result.vertices), np.asarray(result.triangles), process=False
    )


def _mesh_hash(mesh: Any) -> str:
    h = hashlib.sha256()
    for a, dtype in ((mesh.vertices, "<f8"), (mesh.faces, "<i8")):
        value = np.ascontiguousarray(a, dtype=dtype)
        h.update(np.asarray(value.shape, dtype="<i8").tobytes())
        h.update(value.tobytes())
    return h.hexdigest()


def _display(mesh: Any, mm: float) -> tuple[Any, dict[str, object]]:
    out = mesh.copy()
    scale = float(mm / max(out.extents))
    out.apply_scale(scale)
    center = out.bounds.mean(0)
    translation = np.asarray((-center[0], -center[1], -out.bounds[0, 2]))
    out.apply_translation(translation)
    return out, {
        "uniform_scale": scale,
        "translation": [float(value) for value in translation],
        "inverse": "(display_xyz - translation) / uniform_scale",
    }


def _reload(path: Path) -> dict[str, object]:
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=True)
    shell = audit_shell_containment(mesh)
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "extents": [float(x) for x in mesh.extents],
        "shell_audit": shell,
    }


def export_same_source_mesh(
    mesh: Any, staging_output: Path, *, max_dimension_mm: float = 100.0
) -> dict[str, object]:
    output = Path(staging_output)
    output.mkdir(parents=True, exist_ok=True)
    audit_shell_containment(mesh)
    fabrication, transform = _display(mesh, max_dimension_mm)
    parts, depths = _shells(fabrication)
    preview = parts[depths.index(0)]
    files = {
        "fabrication_stl": output / "candidate_fabrication.stl",
        "fabrication_obj": output / "candidate_fabrication.obj",
        "outer_preview_stl": output / "candidate_outer_preview.stl",
        "outer_preview_obj": output / "candidate_outer_preview.obj",
    }
    fabrication.export(files["fabrication_stl"])
    fabrication.export(files["fabrication_obj"])
    preview.export(files["outer_preview_stl"])
    preview.export(files["outer_preview_obj"])
    fr = {
        "stl": _reload(files["fabrication_stl"]),
        "obj": _reload(files["fabrication_obj"]),
    }
    pr = {
        "stl": _reload(files["outer_preview_stl"]),
        "obj": _reload(files["outer_preview_obj"]),
    }
    if any(x["faces"] != len(fabrication.faces) for x in fr.values()) or any(
        x["faces"] != len(preview.faces) for x in pr.values()
    ):
        raise DeliveryFailure("EXPORT_FAILED", "reload face mismatch")
    file_sha256 = {name: _hash(path) for name, path in files.items()}
    return {
        "files": {k: str(v) for k, v in files.items()},
        "source_vertices_faces_sha256": _mesh_hash(fabrication),
        "outer_preview_source_vertices_faces_sha256": _mesh_hash(preview),
        "source_vertices": len(fabrication.vertices),
        "source_faces": len(fabrication.faces),
        "transform": transform,
        "file_sha256": file_sha256,
        "fabrication_reload": fr,
        "outer_preview_reload": pr,
    }


def audit_reloaded_fabrication_contract(
    exports: Mapping[str, object],
    shape: tuple[int, int, int],
    targets: Sequence[Any],
    contract: object,
    device: str,
    hull_voxels: int,
    config: DeliveryConfig,
) -> dict[str, object]:
    import trimesh

    files = exports["files"]
    transform = exports["transform"]
    scale = float(transform["uniform_scale"])
    translation = np.asarray(transform["translation"], dtype=np.float64)
    audits = {}
    for format_name, file_key in (
        ("stl", "fabrication_stl"),
        ("obj", "fabrication_obj"),
    ):
        path = Path(files[file_key])
        if _hash(path) != exports["file_sha256"][file_key]:
            raise DeliveryFailure(
                "EXPORT_RELOAD_FAILED", "serialized file hash changed"
            )
        mesh = trimesh.load(path, force="mesh", process=True)
        mesh.apply_translation(-translation)
        mesh.apply_scale(1.0 / scale)
        shell = audit_shell_containment(mesh)
        raster = rasterize_mesh_with_cavities(mesh, shape)
        metrics = evaluate_contract(
            raster,
            targets,
            contract,
            renderer=None,
            device=device,
            hull_voxels=hull_voxels,
        )
        if not _delivery_pass(metrics, config):
            raise DeliveryFailure(
                "EXPORT_RELOAD_GATE_FAILED",
                f"reloaded fabrication {format_name} failed contract gates",
                {"format": format_name, "metrics": metrics},
            )
        audits[format_name] = {
            "inverse_transform": transform["inverse"],
            "shell_audit": shell,
            "raster_metrics": metrics,
            "raster_shape": list(raster.shape),
        }
    return audits


def select_delivery_candidate(
    candidates: Sequence[tuple[np.ndarray, Mapping[str, object]]]
) -> tuple[np.ndarray, Mapping[str, object]]:
    if not candidates:
        raise DeliveryFailure("NO_VALID_CANDIDATE", "absolute valid pool is empty")
    fidelity_max = max(float(m["mean_iou"]) for _, m in candidates)
    pool = [
        item for item in candidates if float(item[1]["mean_iou"]) >= fidelity_max - 0.01
    ]
    return min(
        pool, key=lambda x: (int(x[1]["occupied_voxels"]), -float(x[1]["saving"]))
    )


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot(path: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (p.relative_to(path).as_posix(), _hash(p))
        for p in sorted(path.rglob("*"))
        if p.is_file()
    )


def _targets(path: Path, contract: object, resolution: int) -> list[np.ndarray]:
    import cv2

    count = len(contract.views)  # type: ignore[attr-defined]
    if {p.name for p in path.glob("gt_view_*.png")} != {
        f"gt_view_{i}.png" for i in range(count)
    }:
        raise DeliveryFailure("TARGET_MISMATCH", "target set mismatch")
    result = []
    for i in range(count):
        image = cv2.imread(str(path / f"gt_view_{i}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (resolution, resolution):
            raise DeliveryFailure("TARGET_MISMATCH", "target shape mismatch")
        result.append(
            cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
        )
    actual = tuple(
        (f"gt_view_{i}.png", _hash(path / f"gt_view_{i}.png")) for i in range(count)
    )
    if actual != tuple(contract.normalized_target_sha256):
        raise DeliveryFailure("ARTIFACT_MUTATED", "target hashes changed")  # type: ignore[attr-defined]
    return result


def _maxpool(volume: np.ndarray, resolution: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    return F.adaptive_max_pool3d(
        torch.from_numpy(np.ascontiguousarray(volume, np.float32))[None, None],
        (resolution,) * 3,
    )[0, 0].numpy()


def _mask_resize(mask: np.ndarray, resolution: int) -> np.ndarray:
    idx = np.minimum(
        np.arange(resolution) * mask.shape[0] // resolution, mask.shape[0] - 1
    )
    return mask[np.ix_(idx, idx)]


def _profile(contract: object) -> tuple[tuple[float, ...], int, int]:
    name = str(contract.protocol_id).lower()  # type: ignore[attr-defined]
    if "dark_knight" in name or "two_wall" in name:
        return ((0.15, 0.25, 0.35, 0.45, 0.55), 1, 16)
    return ((0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.45, 0.55, 0.65), 2, 48)


def _delivery_pass(m: Mapping[str, object], c: DeliveryConfig) -> bool:
    return (
        int(m["occupied_voxels"]) > 0
        and int(m["component_count_6"]) == 1
        and float(m["mean_iou"]) >= c.mean_iou_gate
        and float(m["worst_iou"]) >= c.worst_iou_gate
        and c.minimum_saving <= float(m["saving"]) <= c.maximum_saving
        and float(m["erosion_survival"]) >= c.minimum_erosion_survival
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write(path: Path, data: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(_jsonable(data), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def evaluate_delivery_trajectory_states(
    accepted_states: Sequence[Mapping[str, object]],
    *,
    threshold: float,
    envelope_xyz: Any,
    probability_xyz: Any,
    targets: Sequence[Any],
    contract: object,
    config: DeliveryConfig,
    renderer: Callable[..., object] | None,
    device: str,
    bridge_max_passes: int,
    bridge_radius: int,
) -> tuple[
    list[dict[str, object]],
    list[tuple[np.ndarray, dict[str, object], dict[str, object], np.ndarray]],
]:
    """Run the absolute delivery gate for every C48-accepted checkpoint."""
    envelope = _cube(envelope_xyz, bool, "envelope")
    probability = _cube(probability_xyz, np.float32, "probability")
    rows: list[dict[str, object]] = []
    valid: list[tuple[np.ndarray, dict[str, object], dict[str, object], np.ndarray]] = (
        []
    )
    for state in accepted_states:
        core = _cube(state["volume"], bool, "accepted core")
        checkpoint = state["checkpoint"]
        delivered, shell = core_shell_deliver(
            core,
            envelope,
            probability,
            targets,
            contract,
            renderer=renderer,
            device=device,
            shell_voxels=config.shell_voxels,
            bridge_max_passes=bridge_max_passes,
            bridge_radius=bridge_radius,
        )
        metrics = evaluate_contract(
            delivered,
            targets,
            contract,
            renderer=renderer,
            device=device,
            hull_voxels=int(envelope.sum()),
        )
        ok = _delivery_pass(metrics, config)
        row: dict[str, object] = {
            "threshold": threshold,
            "checkpoint": checkpoint,
            "status": "absolute_valid" if ok else "absolute_gate_failed",
            "core_shell": shell,
            "delivery_metrics": metrics,
        }
        rows.append(row)
        if ok:
            valid.append((delivered, metrics, row, core.copy()))
    return rows, valid


def _run(
    contract: object,
    artifact: Path,
    output: Path,
    device: str,
    config: DeliveryConfig,
    renderer: Callable[..., object] | None,
) -> dict[str, object]:
    _validate_contract(contract)
    if int(contract.image_size) != config.delivery_resolution:
        raise DeliveryFailure("CONTRACT_INVALID", "image size mismatch")  # type: ignore[attr-defined]
    snapshot = tuple(tuple(x) for x in contract.artifact_sha256)  # type: ignore[attr-defined]
    if not snapshot or _snapshot(artifact) != snapshot:
        raise DeliveryFailure("ARTIFACT_MUTATED", "artifact snapshot mismatch")
    ppath, opath = (
        artifact / "occupancy_probability_core.npy",
        artifact / "occupancy_soft_proposal.npy",
    )
    if not ppath.is_file() or not opath.is_file():
        raise DeliveryFailure("ARTIFACT_MISSING", "soft arrays missing")
    p = trainer_zyx_to_canonical_xyz(np.load(ppath).astype(np.float32))
    soft = trainer_zyx_to_canonical_xyz(np.load(opath).astype(bool))
    expected = (config.delivery_resolution,) * 3
    if p.shape != expected or soft.shape != expected:
        raise DeliveryFailure("ARTIFACT_INVALID", "soft array shape mismatch")
    targets = _targets(artifact, contract, config.delivery_resolution)
    p48 = _maxpool(p, config.core_resolution)
    t48 = [_mask_resize(t, config.core_resolution) for t in targets]
    h48 = backproject_contract(t48, contract, config.core_resolution, device=device)
    h256 = backproject_contract(
        targets, contract, config.delivery_resolution, device=device
    )
    thresholds, radius, passes = _profile(contract)
    trajectories = []
    valid = []
    for threshold in thresholds:
        seed = (p48 >= threshold) & h48
        if not seed.any():
            trajectories.append({"threshold": threshold, "status": "empty_seed"})
            continue
        seed = restore_contract_rays(seed, h48, p48, t48, contract, device=device)
        try:
            seed, bridge = bridge6_restricted(
                seed, h48, max_passes=passes, radius=radius
            )
        except DeliveryFailure as e:
            trajectories.append(
                {
                    "threshold": threshold,
                    "status": "seed_bridge_failed",
                    "reason": e.reason,
                }
            )
            continue
        seedm = evaluate_contract(seed, t48, contract, renderer=renderer, device=device)
        if not _core_pass(seedm, config.mean_iou_gate, config.worst_iou_gate):
            trajectories.append(
                {
                    "threshold": threshold,
                    "status": "seed_core_gate_failed",
                    "metrics": seedm,
                }
            )
            continue
        _, carve = carve_core_trajectory(
            seed,
            p48,
            h48,
            t48,
            contract,
            renderer=renderer,
            device=device,
            max_checkpoints=config.max_checkpoints,
            bridge_max_passes=passes,
            bridge_radius=radius,
        )
        accepted_states = carve["accepted_states"]
        carve_manifest = dict(carve)
        del carve_manifest["accepted_states"]
        rows, state_valid = evaluate_delivery_trajectory_states(
            accepted_states,
            threshold=threshold,
            envelope_xyz=h256,
            probability_xyz=p,
            targets=targets,
            contract=contract,
            config=config,
            renderer=renderer,
            device=device,
            bridge_max_passes=passes,
            bridge_radius=radius,
        )
        for row in rows:
            row["seed_bridge"] = bridge
            row["carve"] = carve_manifest
        trajectories.extend(rows)
        valid.extend(state_valid)
    if not valid:
        np.save(output / "diagnostic_hull48.npy", h48)
        raise DeliveryFailure(
            "NO_VALID_CANDIDATE",
            "no absolute-valid candidate",
            {"trajectories": trajectories},
        )
    selected, selected_metrics = select_delivery_candidate(
        [(v, m) for v, m, _, _ in valid]
    )
    selected_row, selected_core = next(
        (row, core) for v, _, row, core in valid if v is selected
    )
    mesh = padded_sdf_mesh(
        selected, pad=config.sdf_pad, sigma=config.sdf_sigma, level=config.sdf_level
    )
    pre = audit_shell_containment(mesh)
    mesh, cap = enforce_triangle_cap(
        mesh,
        max_faces=config.max_faces,
        simplifier=_open3d if len(mesh.faces) > config.max_faces else None,
    )
    post = audit_shell_containment(mesh)
    raster = rasterize_mesh_with_cavities(mesh, expected)
    rm = evaluate_contract(
        raster,
        targets,
        contract,
        renderer=renderer,
        device=device,
        hull_voxels=int(h256.sum()),
    )
    if not _delivery_pass(rm, config):
        np.save(output / "diagnostic_mesh_raster.npy", raster)
        raise DeliveryFailure(
            "MESH_RASTER_GATE_FAILED",
            "mesh reraster failed full delivery gates",
            {"metrics": rm},
        )
    renderer_identity = (
        "builtin_exact_contract_dispatch"
        if renderer is None
        else f"{getattr(renderer, '__module__', '<unknown>')}."
        f"{getattr(renderer, '__qualname__', getattr(renderer, '__name__', type(renderer).__name__))}"
    )
    test_only = renderer is not None
    exports = None
    serialized_reload_qa = None
    if not test_only:
        exports = export_same_source_mesh(
            mesh, output, max_dimension_mm=config.max_dimension_mm
        )
        serialized_reload_qa = audit_reloaded_fabrication_contract(
            exports,
            expected,
            targets,
            contract,
            device,
            int(h256.sum()),
            config,
        )
        exports["files"] = {
            name: Path(path).name for name, path in exports["files"].items()
        }
    if _snapshot(artifact) != snapshot:
        raise DeliveryFailure("ARTIFACT_MUTATED", "artifact mutated during delivery")
    np.save(output / "candidate_delivery_voxels.npy", raster)
    contract_data = (
        asdict(contract)
        if is_dataclass(contract)
        else {f: getattr(contract, f) for f in CONTRACT_FIELDS}
    )
    return {
        "status": "STAGING_TEST_ONLY" if test_only else "STAGING_CANDIDATE_READY",
        "publication_eligible": not test_only,
        "renderer": {
            "mode": "injected_test_only" if test_only else "production_exact_dispatch",
            "identity": renderer_identity,
        },
        "mode": "shadowart_geometry_only",
        "structural_status": "NOT_REQUESTED",
        "publication_status": "NOT_PUBLISHED",
        "camera_contract": contract_data,
        "stages": {
            "probability256": {
                "shape": list(p.shape),
                "frame": "canonical_xyz",
                "dtype": str(p.dtype),
                "nonzero": int(np.count_nonzero(p)),
            },
            "soft_proposal256_read_only": {
                "shape": list(soft.shape),
                "frame": "canonical_xyz",
                "dtype": str(soft.dtype),
                "occupied": int(soft.sum()),
                "mutated": False,
            },
            "core_probability": {
                "shape": list(p48.shape),
                "frame": "canonical_xyz",
                "dtype": str(p48.dtype),
                "nonzero": int(np.count_nonzero(p48)),
            },
            "hull48": {
                "shape": list(h48.shape),
                "frame": "canonical_xyz",
                "dtype": str(h48.dtype),
                "occupied": int(h48.sum()),
            },
            "hull256": {
                "shape": list(h256.shape),
                "frame": "canonical_xyz",
                "dtype": str(h256.dtype),
                "occupied": int(h256.sum()),
            },
            "mesh_raster": {
                "shape": list(raster.shape),
                "frame": "canonical_xyz",
                "dtype": str(raster.dtype),
                "occupied": int(raster.sum()),
            },
            "selected_core": {
                "shape": list(selected_core.shape),
                "frame": "canonical_xyz",
                "dtype": str(selected_core.dtype),
                "occupied": int(selected_core.sum()),
            },
            "premesh_delivery": {
                "shape": list(selected.shape),
                "frame": "canonical_xyz",
                "dtype": str(selected.dtype),
                "occupied": int(selected.sum()),
            },
        },
        "trajectories": trajectories,
        "selection": {
            "absolute_valid_count": len(valid),
            "near_best_pool_count": sum(
                float(m["mean_iou"])
                >= max(float(item[1]["mean_iou"]) for item in valid) - 0.01
                for _, m, _, _ in valid
            ),
            "selected_threshold": selected_row["threshold"],
            "selected_checkpoint": selected_row["checkpoint"],
            "selected_metrics": selected_metrics,
        },
        "gates": {
            "mean_iou": config.mean_iou_gate,
            "worst_iou": config.worst_iou_gate,
            "saving_range": [config.minimum_saving, config.maximum_saving],
            "erosion_survival": config.minimum_erosion_survival,
            "component_count_6": 1,
            "max_faces": config.max_faces,
            "surface_fraction": "diagnostic_only",
        },
        "mesh": {
            "pre_cap_shell_audit": pre,
            "triangle_cap": cap,
            "post_cap_shell_audit": post,
            "reraster_metrics": rm,
            "source_vertices_faces_sha256": (
                _mesh_hash(mesh)
                if exports is None
                else exports["source_vertices_faces_sha256"]
            ),
            "serialized_reload_contract_qa": serialized_reload_qa,
            "exports": exports,
        },
    }


def run_contract_delivery(
    *,
    contract: object,
    soft_artifact: Path,
    staging_output: Path,
    device: str,
    config: DeliveryConfig = DeliveryConfig(),
    renderer: Callable[..., object] | None = None,
    delivery_mode: str = "core48",
) -> dict[str, object]:
    artifact = Path(soft_artifact).resolve()
    output = Path(staging_output).resolve()
    if delivery_mode not in ("auto", "core48", "native256"):
        raise DeliveryFailure("PRODUCTION_CONFIG_INVALID", "unknown delivery mode")
    if renderer is not None and delivery_mode != "core48":
        raise DeliveryFailure("PRODUCTION_CONFIG_INVALID", "native recovery cannot use an injected renderer")
    if _paths_overlap(artifact, output):
        raise DeliveryFailure(
            "PATH_OVERLAP",
            "soft artifact and staging output must be disjoint",
            {"soft_artifact": str(artifact), "staging_output": str(output)},
        )
    if renderer is None and (
        config.core_resolution != 48
        or config.delivery_resolution != 256
        or config.sdf_pad != 2
        or config.sdf_sigma != 0.4
        or config.sdf_level != 0.0
    ):
        raise DeliveryFailure(
            "PRODUCTION_CONFIG_INVALID",
            "production delivery requires frozen 48/256 and SDF configuration",
        )
    if output.exists():
        raise DeliveryFailure(
            "STAGING_CONFLICT", "staging ready directory already exists"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        if delivery_mode == "native256":
            from experiments.multiview_native_delivery_20260905 import run_native
            result = run_native(contract, artifact, temporary, device, config)
        else:
            try:
                result = _run(contract, artifact, temporary, device, config, renderer)
            except DeliveryFailure as error:
                if delivery_mode != "auto" or error.status != "NO_VALID_CANDIDATE":
                    raise
                from experiments.multiview_native_delivery_20260905 import run_native
                print("Core48 has no valid candidate; trying experimental native256 recovery", flush=True)
                result = run_native(contract, artifact, temporary, device, config,
                                    fallback={"status": error.status, "reason": error.reason,
                                              "details": error.details})
        _write(temporary / "staging_delivery_manifest.json", result)
        os.replace(temporary, output)
    except DeliveryFailure as e:
        try:
            _write_failure_directory(output, e, diagnostics_source=temporary)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    except Exception as e:
        failure = DeliveryFailure(
            "DELIVERY_INTERNAL_ERROR", "unexpected delivery error", {"error": str(e)}
        )
        try:
            _write_failure_directory(output, failure, diagnostics_source=temporary)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        raise failure from e
    except (KeyboardInterrupt, SystemExit):
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _copy_failure_diagnostics(source: Path, destination: Path) -> None:
    allowed_suffixes = {".npy", ".png", ".json"}
    for item in source.iterdir():
        if (
            item.is_file()
            and item.name.startswith("diagnostic_")
            and item.suffix.lower() in allowed_suffixes
        ):
            shutil.copy2(item, destination / item.name)
        elif item.is_dir() and item.name == "diagnostics":
            for diagnostic in item.rglob("*"):
                if (
                    not diagnostic.is_file()
                    or diagnostic.suffix.lower() not in allowed_suffixes
                ):
                    continue
                relative = diagnostic.relative_to(item)
                target = destination / "diagnostics" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(diagnostic, target)


def _write_failure_directory(
    output: Path,
    failure: DeliveryFailure,
    *,
    diagnostics_source: Path | None = None,
) -> None:
    if output.exists():
        return
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.failure-", dir=output.parent)
    )
    try:
        if diagnostics_source is not None and diagnostics_source.is_dir():
            _copy_failure_diagnostics(diagnostics_source, temporary)
        _write(
            temporary / "staging_delivery_failure.json",
            {
                "status": failure.status,
                "reason": failure.reason,
                "details": failure.details,
                "mode": "shadowart_geometry_only",
                "structural_status": "NOT_REQUESTED",
                "publication_status": "NOT_PUBLISHED",
                "publication_eligible": False,
            },
        )
        os.replace(temporary, output)
    except Exception as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise DeliveryFailure(
            "STAGING_IO_FAILED",
            "failed to write truthful staging failure diagnostics",
            {"error": str(error), "original_status": failure.status},
        ) from error


__all__ = [
    "DeliveryConfig",
    "DeliveryFailure",
    "trainer_zyx_to_canonical_xyz",
    "canonical_xyz_to_trainer_zyx",
    "render_contract",
    "backproject_contract",
    "restore_contract_rays",
    "bridge6_restricted",
    "evaluate_contract",
    "evaluate_delivery_trajectory_states",
    "carve_core_trajectory",
    "core_shell_deliver",
    "padded_sdf_mesh",
    "audit_shell_containment",
    "rasterize_mesh_with_cavities",
    "enforce_triangle_cap",
    "export_same_source_mesh",
    "audit_reloaded_fabrication_contract",
    "select_delivery_candidate",
    "run_contract_delivery",
]
