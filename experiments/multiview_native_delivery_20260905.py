"""Experimental native-grid geometry recovery; NOT the paper's Core48 method.

Soft artifacts are immutable. Binary boundary edits are accepted only after an
exact same-camera render and a global 6-connectivity check. This is a geometry
postprocess, not a new soft artifact, material-carving run, or FEM validation.
"""
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from experiments import multiview_contract_delivery_20260902 as d

ALGORITHM = "experimental_native256_connected_projection_repair_v1"


def connected_seed(soft, *, max_components=16):
    source = d._cube(soft, bool, "soft")
    labels, count = d._components(source)
    if not count:
        raise d.DeliveryFailure("NATIVE_EMPTY_SOURCE", "empty native source")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1][:min(count, max_components)]
    seed = np.isin(labels, order)
    paths = np.zeros_like(seed)
    if len(order) > 1:
        distance, nearest = ndi.distance_transform_edt(labels != order[0], return_indices=True)
        boxes = ndi.find_objects(labels)
        for part in order[1:]:
            box = boxes[part - 1]
            points = np.argwhere(labels[box] == part) + np.array([b.start for b in box])
            cursor = points[np.argmin(distance[tuple(points.T)])].copy()
            end = nearest[(slice(None), *cursor)]
            paths[tuple(cursor)] = True
            for _ in range(int(np.abs(end - cursor).sum())):
                axis = int(np.argmax(np.abs(end - cursor)))
                cursor[axis] += np.sign(end[axis] - cursor[axis])
                paths[tuple(cursor)] = True
        seed |= ndi.binary_dilation(paths, ndi.generate_binary_structure(3, 1))
    if d._components(seed)[1] != 1:
        raise d.DeliveryFailure("NATIVE_SEED_FAILED", "native seed is disconnected")
    return seed, {
        "source_components": count, "retained_components": len(order),
        "dropped_source_voxels": int((source & ~seed).sum()),
        "bridge_added_voxels": int((seed & ~source).sum()),
        "max_components": max_components,
    }


def _gradient(volume, targets, contract, device):
    import torch

    direct = contract.renderer_family == d.DIRECT
    zyx = d.canonical_xyz_to_trainer_zyx(volume)
    tensor = torch.tensor(np.ascontiguousarray(zyx, np.float32)[None, None],
                          device=device, requires_grad=True)
    gradient = torch.zeros_like(tensor)
    loss_total = 0.
    if direct:
        cls = d._external("sadekar_protocol.py", "native_sadekar").SadekarTFSProtocol
        protocol = cls(torch.device(device), volume.shape[0], float(contract.zdist),
                       float(contract.volume_extent), int(contract.ray_samples))
    else:
        render = d._external("renderer_voxel.py", "native_renderer").render_orthographic
    for i, (az, el) in enumerate(contract.views):
        if direct:
            # Sadekar uses a constant red feature sigmoid(1); divide it out to
            # optimize alpha, then always accept using its real binary_mask.
            pred = protocol.render(tensor * .999, i) / torch.sigmoid(torch.tensor(1., device=device))
        else:
            pred = render(tensor * .999, float(az), float(el))
        target = torch.tensor(targets[i], device=device, dtype=torch.float32)
        loss = ((pred - target) ** 2).mean()
        gradient += torch.autograd.grad(loss, tensor)[0]
        loss_total += float(loss.detach())
    g = gradient[0, 0].cpu().numpy()
    return d.trainer_zyx_to_canonical_xyz(g), loss_total


def refine_connected(seed, targets, contract, *, device, steps=60,
                     protected=None, acceptance_gate=None):
    volume = d._cube(seed, bool, "seed").copy()
    protected = (np.zeros_like(volume) if protected is None
                 else d._cube(protected, bool, "protected"))
    if protected.shape != volume.shape or np.any(protected & ~volume):
        raise ValueError("protected material must be a subset of the seed")
    best = d.evaluate_contract(volume, targets, contract, device=device)
    if best["component_count_6"] != 1:
        raise d.DeliveryFailure("NATIVE_SEED_FAILED", "native seed is disconnected")
    history = [{"step": -1, "budget": 0, "edit": "seed", "metrics": best}]
    for step in range(steps):
        gradient, loss = _gradient(volume, targets, contract, device)
        additions = np.flatnonzero((ndi.binary_dilation(volume) & ~volume & (gradient < 0)).ravel())
        removals = np.flatnonzero((volume & ~protected & ~ndi.binary_erosion(volume)
                                  & (gradient > 0)).ravel())
        additions = additions[np.argsort(gradient.ravel()[additions])]
        removals = removals[np.argsort(-gradient.ravel()[removals])]
        accepted = False
        for budget in (1024, 256, 64, 16, 4, 1):
            for edit in ("joint", "remove", "add"):
                candidate = volume.copy()
                if edit != "remove":
                    candidate.ravel()[additions[:budget]] = True
                if edit != "add":
                    for flat in removals[:budget]:
                        point = np.unravel_index(flat, candidate.shape)
                        if d._simple(candidate, point):
                            candidate[point] = False
                metrics = d.evaluate_contract(candidate, targets, contract, device=device)
                if (metrics["component_count_6"] == 1
                        and metrics["mean_iou"] > best["mean_iou"]
                        and (acceptance_gate is None or acceptance_gate(candidate))):
                    volume, best = candidate, metrics
                    history.append({"step": step, "budget": budget, "edit": edit, "metrics": metrics})
                    print(f"native256 step={step} mean={best['mean_iou']:.6f} "
                          f"worst={best['worst_iou']:.6f} components=1", flush=True)
                    accepted = True
                    break
            if accepted:
                break
        if not accepted:
            break
    return volume, history


def finalize_native(volume, targets, contract, output, device, config, *, hull_voxels):
    """Use the unchanged absolute mesh/reload gates, never publish a failed mesh."""
    selected_metrics = d.evaluate_contract(volume, targets, contract, device=device,
                                           hull_voxels=hull_voxels)
    if not d._delivery_pass(selected_metrics, config):
        np.save(output / "diagnostic_native_candidate.npy", volume)
        raise d.DeliveryFailure("NATIVE_GATE_FAILED", "native candidate failed absolute gates",
                                {"metrics": selected_metrics})
    mesh = d.padded_sdf_mesh(volume, pad=config.sdf_pad, sigma=config.sdf_sigma, level=config.sdf_level)
    pre = d.audit_shell_containment(mesh)
    mesh, cap = d.enforce_triangle_cap(mesh, max_faces=config.max_faces,
                                      simplifier=d._open3d if len(mesh.faces) > config.max_faces else None)
    post = d.audit_shell_containment(mesh)
    raster = d.rasterize_mesh_with_cavities(mesh, volume.shape)
    rm = d.evaluate_contract(raster, targets, contract, device=device, hull_voxels=hull_voxels)
    if not d._delivery_pass(rm, config):
        np.save(output / "diagnostic_mesh_raster.npy", raster)
        raise d.DeliveryFailure("MESH_RASTER_GATE_FAILED", "native mesh reraster failed", {"metrics": rm})
    exports = d.export_same_source_mesh(mesh, output, max_dimension_mm=config.max_dimension_mm)
    reload_qa = d.audit_reloaded_fabrication_contract(exports, volume.shape, targets, contract,
                                                    device, hull_voxels, config)
    exports["files"] = {name: Path(path).name for name, path in exports["files"].items()}
    return raster, selected_metrics, {
        "pre_cap_shell_audit": pre, "triangle_cap": cap, "post_cap_shell_audit": post,
        "reraster_metrics": rm,
        "source_vertices_faces_sha256": exports["source_vertices_faces_sha256"],
        "serialized_reload_contract_qa": reload_qa, "exports": exports,
    }


def run_native(contract, artifact, output, device, config, *, fallback=None):
    d._validate_contract(contract)
    snapshot = tuple(tuple(row) for row in contract.artifact_sha256)
    if not snapshot or d._snapshot(artifact) != snapshot:
        raise d.DeliveryFailure("ARTIFACT_MUTATED", "native artifact snapshot mismatch")
    if int(contract.image_size) != 256 or config.delivery_resolution != 256:
        raise d.DeliveryFailure("CONTRACT_INVALID", "native production delivery requires 256")
    targets = d._targets(artifact, contract, 256)
    soft = d.trainer_zyx_to_canonical_xyz(np.load(artifact / "occupancy_soft_proposal.npy").astype(bool))
    probability = d.trainer_zyx_to_canonical_xyz(np.load(artifact / "occupancy_probability_core.npy").astype(np.float32))
    if soft.shape != (256,) * 3 or probability.shape != soft.shape:
        raise d.DeliveryFailure("ARTIFACT_INVALID", "native source shape mismatch")
    # Freeze the reference before any recovery. Centre-ray visual hulls are
    # approximate for opacity rendering, so retain the immutable soft support
    # (the project's original graphics exporter uses the same preserve rule).
    # This saving is a diagnostic support-envelope metric, NOT paper Table 1.
    reference = d.backproject_contract(targets, contract, 256, device=device) | soft
    hull_voxels = int(reference.sum())
    seed, seed_audit = connected_seed(soft)
    volume, history = refine_connected(seed, targets, contract, device=device)
    np.save(output / "diagnostic_native_candidate.npy", volume)
    d._write(output / "diagnostic_native_trace.json", {"seed": seed_audit, "history": history})
    import cv2
    for i, prediction in enumerate(d.render_contract(volume, contract, device=device)):
        cv2.imwrite(str(output / f"diagnostic_pred_view_{i}.png"), prediction.astype(np.uint8) * 255)
    raster, selected_metrics, mesh = finalize_native(volume, targets, contract, output, device,
                                                    config, hull_voxels=hull_voxels)
    if d._snapshot(artifact) != snapshot:
        raise d.DeliveryFailure("ARTIFACT_MUTATED", "native artifact changed during delivery")
    np.save(output / "candidate_delivery_voxels.npy", raster)
    def stage(array, *, probability=False, read_only=False):
        record = {"shape": list(array.shape), "frame": "canonical_xyz", "dtype": str(array.dtype),
                  "nonzero" if probability else "occupied": int(np.count_nonzero(array))}
        if read_only:
            record["mutated"] = False
        return record
    return {
        "status": "STAGING_CANDIDATE_READY", "publication_eligible": True,
        "renderer": {"mode": "production_exact_dispatch", "identity": "builtin_exact_contract_dispatch"},
        "mode": "shadowart_geometry_only", "structural_status": "NOT_REQUESTED",
        "publication_status": "NOT_PUBLISHED",
        "camera_contract": asdict(contract) if is_dataclass(contract) else {f: getattr(contract, f) for f in d.CONTRACT_FIELDS},
        "native_repair": {
            "algorithm": ALGORITHM, "paper_method_equivalence": False,
            "source_resolution": 256, "max_steps": 60, "seed": seed_audit,
            "saving_reference": "geometric_center_ray_hull_union_immutable_soft_support",
            "saving_comparable_to_paper": False, "reference_voxels": hull_voxels,
            "core48_failure": fallback,
        },
        "stages": {
            "probability256": stage(probability, probability=True),
            "soft_proposal256_read_only": stage(soft, read_only=True),
            "reference_envelope256": stage(reference), "selected_native_seed": stage(seed),
            "premesh_delivery": stage(volume), "mesh_raster": stage(raster),
        },
        "trajectories": history,
        "selection": {"selected_metrics": selected_metrics},
        "gates": {
            "mean_iou": config.mean_iou_gate, "worst_iou": config.worst_iou_gate,
            "saving_range": [config.minimum_saving, config.maximum_saving],
            "erosion_survival": config.minimum_erosion_survival, "component_count_6": 1,
            "max_faces": config.max_faces, "surface_fraction": "diagnostic_only",
        },
        "mesh": mesh,
    }
