"""Native XYZ geometry candidates and fail-closed mesh audits for ShadowArt.

No coarse visual hull is imposed. Source arrays and enclosed voids are immutable;
all finite local edits are rerendered before acceptance. Results remain candidates
until independent serialized-mesh and manufacturing checks have passed.
"""
from __future__ import annotations

import importlib
import math

import numpy as np
from scipy import ndimage as ndi

from .manufacturing import FACE6, _block_cover, _positive, _thin_mask, _volume, thickness_audit


STAGE_SWITCH_SEMANTICS = {
    "reinforcement": ("physical_component_bridge", "local_fin_thickening"),
    "fin_removal": ("redundant_component_batch", "guarded_surface_batch",
                    "fin_removal", "fin_shortening"),
}
MAX_THIN_REGIONS = 8
MAX_BRIDGE_ENDPOINTS = 8


def occupancy_to_mesh(volume_xyz, extent=1.7):
    """Extract a padded, unsmoothed signed-distance surface without filling voids."""
    v = _volume(volume_xyz)
    extent = _positive(extent, "extent")
    if not v.any():
        raise ValueError("cannot mesh an empty occupancy")
    padded = np.pad(v, 1, constant_values=False)
    sdf = ndi.distance_transform_edt(padded) - ndi.distance_transform_edt(~padded)
    return _sdf_mesh(sdf, extent, v.shape[0])


def _sdf_mesh(sdf, extent, resolution):
    from skimage.measure import marching_cubes
    import trimesh

    pitch = extent / resolution
    vertices, faces, _, _ = marching_cubes(sdf.astype(np.float32), level=0.,
                                          spacing=(pitch,) * 3, allow_degenerate=False)
    vertices += -extent / 2 - pitch / 2
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    # Invert ALL shells together: individually fixing normals would fill cavities.
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def refine_surface_mesh(volume_xyz, targets, cameras, *, extent=1.7, longest_mm=100.,
                        max_candidates=3, min_wall_mm=.9, min_neck_mm=1.35):
    """Try bounded local SDF saddle regularization, audited as actual meshes.

    Only a three-cell neighborhood of open/nonmanifold edges may change. SDF
    signs and the entire independently revoxelized occupancy must stay unchanged;
    this is not global smoothing or hole filling. Accepted meshes require real
    intersection/containment QA, nonregressing full-image mesh IoU and physical
    checks at their actual uniform scale. Downstream serialized 256/512/1024 QA
    remains mandatory; this helper does not publish or certify printability.
    """
    from .rendering import render_mesh, voxelize_mesh

    volume = _volume(volume_xyz)
    extent = _positive(extent, "extent")
    longest_mm = _positive(longest_mm, "longest_mm")
    wall, neck = _positive(min_wall_mm, "min_wall_mm"), _positive(min_neck_mm, "min_neck_mm")
    if type(max_candidates) is not int or not 0 <= max_candidates <= 3:
        raise ValueError("max_candidates must be an integer from 0 to 3")
    targets = np.asarray(targets)
    cameras = tuple(cameras)
    if (targets.ndim != 3 or targets.shape[0] != len(cameras) or not cameras
            or targets.shape[1] != targets.shape[2] or not targets.shape[1]
            or not np.isin(targets, [0, 1]).all()):
        raise ValueError("targets must be one binary square image per camera")
    targets = targets.astype(bool)
    mesh = occupancy_to_mesh(volume, extent)
    trace = {"status": "UNRESOLVED", "accepted": False, "attempts": [],
             "algorithm": "local_sign_preserving_sdf_saddle_regularization_v1",
             "max_candidates": max_candidates, "target_image_size": int(targets.shape[1]),
             "publication_eligible": False, "serialized_mesh_validated": False,
             "global_smoothing": False, "global_hole_fill": False,
             "source_occupancy_mutated": False}
    if ndi.label(volume, FACE6)[1] != 1:
        trace["reason"] = "DISCONNECTED_MATERIAL_REPAIR_REQUIRED"
        trace["initial_mesh_audit"] = {"status": "NOT_RUN", "passed": False,
                                       "reason": trace["reason"]}
        return mesh, trace
    initial_audit = mesh_audit(mesh)
    trace["initial_mesh_audit"] = initial_audit
    if initial_audit["passed"]:
        trace["status"] = "NOT_NEEDED"
        return mesh, trace
    edges, counts = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
    bad_vertices = np.unique(edges[counts != 2])
    if not len(bad_vertices):
        trace["reason"] = "NO_LOCAL_EDGE_DEFECT_SEED_CAPABILITY_MISSING"
        return mesh, trace
    initial_render = render_mesh(mesh, cameras, targets.shape[1])
    initial_quality = _quality(initial_render, targets, .98, .97)
    _, transform = uniform_print_transform(mesh, longest_mm)
    initial_physical = thickness_audit(volume, extent / volume.shape[0] * np.linalg.norm(transform[:3, 0]), wall, neck)
    trace["initial_quality"] = initial_quality
    trace["initial_manufacturing"] = initial_physical
    padded = np.pad(volume, 1, constant_values=False)
    sdf = ndi.distance_transform_edt(padded) - ndi.distance_transform_edt(~padded)
    local = np.zeros_like(padded)
    grid_vertices = np.rint((mesh.vertices[bad_vertices] + extent / 2) / (extent / volume.shape[0]) + .5).astype(int)
    grid_vertices = np.clip(grid_vertices, 0, np.array(padded.shape) - 1)
    local[tuple(grid_vertices.T)] = True
    local = ndi.binary_dilation(local, iterations=3) & (np.abs(sdf) <= 2.)
    blurred = ndi.gaussian_filter(sdf, sigma=.65, mode="nearest")
    trace["local_sdf_samples"] = int(local.sum())
    for blend in (.1, .25, .5)[:max_candidates]:
        candidate_sdf = sdf.copy()
        candidate_sdf[local] = (1. - blend) * sdf[local] + blend * blurred[local]
        candidate_sdf = np.where(padded, np.maximum(candidate_sdf, .1), np.minimum(candidate_sdf, -.1))
        candidate = _sdf_mesh(candidate_sdf, extent, volume.shape[0])
        audit = mesh_audit(candidate)
        row = {"blend": blend, "mesh_audit": audit, "accepted": False}
        trace["attempts"].append(row)
        if not audit["passed"]:
            row["reason"] = "MESH_QA"
            continue
        try:
            replay = voxelize_mesh(candidate, volume.shape[0], extent)
        except Exception as error:
            row["reason"], row["error"] = "VOXELIZATION_UNVERIFIED", str(error)
            continue
        row["changed_voxel_centers"] = int(np.count_nonzero(replay != volume))
        if row["changed_voxel_centers"]:
            row["reason"] = "OCCUPANCY_OR_CAVITY_CHANGED"
            continue
        rendered = render_mesh(candidate, cameras, targets.shape[1])
        quality = _quality(rendered, targets, .98, .97)
        row["quality"] = quality
        if (quality["mean_iou"] < initial_quality["mean_iou"] - 1e-12
                or quality["worst_iou"] < initial_quality["worst_iou"] - 1e-12
                or (initial_quality["passed"] and not quality["passed"])):
            row["reason"] = "MESH_FIDELITY_REGRESSION"
            continue
        _, matrix = uniform_print_transform(candidate, longest_mm)
        physical = thickness_audit(replay, extent / volume.shape[0] * np.linalg.norm(matrix[:3, 0]), wall, neck)
        row["manufacturing"] = physical
        if _physical_rank(physical) < _physical_rank(initial_physical):
            row["reason"] = "MANUFACTURING_REGRESSION"
            continue
        row["accepted"], row["reason"] = True, "TOPOLOGY_IMPROVEMENT_WITH_REAL_MESH_QA"
        trace.update(status="CANDIDATE_ONLY", accepted=True, final_mesh_audit=audit,
                     final_quality=quality, final_manufacturing=physical,
                     required_next="independent serialized multi-resolution mesh and manufacturing QA")
        return candidate, trace
    trace["reason"] = "NO_VERIFIED_NONREGRESSING_LOCAL_CANDIDATE"
    return mesh, trace


def mesh_audit(mesh):
    """Check manifold topology, real triangle intersections and cavity orientation.

    Missing or failing intersection/containment capabilities fail closed. Nested
    positive solids are disconnected material and cannot pass as legal cavities.
    """
    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices)
    reasons = []
    finite = bool(np.isfinite(vertices).all())
    valid = bool(vertices.ndim == 2 and vertices.shape[1:] == (3,) and finite
                 and faces.ndim == 2 and faces.shape[1:] == (3,) and len(faces)
                 and faces.min() >= 0 and faces.max() < len(vertices))
    if not valid:
        return {"passed": False, "status": "FAIL", "reasons": ["INVALID_MESH"],
                "self_intersection": {"status": "REVIEW_REQUIRED", "reason": "invalid mesh"},
                "outer_shells": 0, "legal_cavities": 0}
    _, counts = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
    boundary, nonmanifold = int((counts == 1).sum()), int((counts > 2).sum())
    watertight, winding = bool(mesh.is_watertight), bool(mesh.is_winding_consistent)
    if boundary or nonmanifold or not watertight:
        reasons.append("NONMANIFOLD_OR_OPEN")
    if not winding:
        reasons.append("INCONSISTENT_WINDING")
    if len(faces) > 2_000_000:
        reasons.append("TRIANGLE_CAP")
    if np.any(np.asarray(mesh.area_faces) <= 1e-15):
        reasons.append("DEGENERATE_TRIANGLES")

    try:
        ml = importlib.import_module("pymeshlab")
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(vertex_matrix=np.asarray(vertices, np.float64),
                            face_matrix=np.asarray(faces, np.int32)))
        ms.compute_selection_by_self_intersections_per_face()
        selected = int(ms.current_mesh().selected_face_number())
        intersections = {"status": "FAIL" if selected else "PASS",
                         "selected_faces": selected,
                         "backend": "pymeshlab.compute_selection_by_self_intersections_per_face"}
        if selected:
            reasons.append("SELF_INTERSECTION")
    except Exception as error:
        intersections = {"status": "REVIEW_REQUIRED", "reason": str(error),
                         "backend": "pymeshlab"}
        reasons.append("SELF_INTERSECTION_UNVERIFIED")

    outer, cavities, shell_rows = 0, 0, []
    containment = "PASS"
    try:
        parts = list(mesh.split(only_watertight=False))
        for index, part in enumerate(parts):
            if not part.is_watertight:
                raise ValueError("containment requires closed shells")
            point = np.asarray(part.vertices[:1], np.float64)
            containers = [j for j, other in enumerate(parts) if j != index
                          and np.all(point[0] > other.bounds[0])
                          and np.all(point[0] < other.bounds[1])
                          and bool(other.contains(point)[0])]
            depth = len(containers)
            outer += int(depth == 0)
            cavities += int(depth == 1 and part.volume < 0)
            shell_rows.append({"shell": index, "depth": depth,
                               "signed_volume": float(part.volume), "containers": containers})
            if (depth == 0 and part.volume <= 0) or (depth == 1 and part.volume >= 0):
                reasons.append("SHELL_ORIENTATION")
            if depth > 1:
                reasons.append("NESTED_DISCONNECTED_SOLID")
        if outer != 1:
            reasons.append("MULTIPLE_OUTER_SHELLS")
    except Exception as error:
        containment = "REVIEW_REQUIRED"
        reasons.append("CONTAINMENT_UNVERIFIED")
        shell_rows.append({"error": str(error)})
    reasons = list(dict.fromkeys(reasons))
    return {
        "passed": not reasons, "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons, "vertices": int(len(vertices)), "triangles": int(len(faces)),
        "watertight": watertight, "winding_consistent": winding,
        "boundary_edges": boundary, "nonmanifold_edges": nonmanifold,
        "self_intersection": intersections, "containment_status": containment,
        "outer_shells": outer, "legal_cavities": cavities, "shells": shell_rows,
        "structural_validation": "NOT_REQUESTED", "physical_print_validated": False,
    }


def uniform_print_transform(mesh, longest_mm=100.):
    """Rotate canonical Y-up to print Z-up, uniformly scale and place on the bed."""
    longest_mm = _positive(longest_mm, "longest_mm")
    if not len(mesh.vertices) or not np.isfinite(mesh.vertices).all() or max(mesh.extents) <= 0:
        raise ValueError("mesh must have finite positive extent")
    scale = longest_mm / float(max(mesh.extents))
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    transformed = np.asarray(mesh.vertices) @ (scale * rotation).T
    lo, hi = transformed.min(0), transformed.max(0)
    translation = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = scale * rotation, translation
    result = mesh.copy()
    result.apply_transform(matrix)
    return result, matrix


def _enclosed_voids(volume):
    labels, _ = ndi.label(~volume, FACE6)
    exterior = np.unique(np.concatenate([labels[0].ravel(), labels[-1].ravel(),
                                        labels[:, 0].ravel(), labels[:, -1].ravel(),
                                        labels[:, :, 0].ravel(), labels[:, :, -1].ravel()]))
    return ~volume & ~np.isin(labels, exterior)


def _pitch_from_occupied_bounds(volume, longest_mm):
    points = np.argwhere(volume)
    span = int(np.max(np.ptp(points, axis=0) + 1)) if len(points) else volume.shape[0]
    return longest_mm / span


def _quality(predictions, targets, mean_gate, worst_gate):
    axes = (1, 2)
    tp = np.count_nonzero(predictions & targets, axis=axes)
    union = np.count_nonzero(predictions | targets, axis=axes)
    ious = np.divide(tp, union, out=np.ones(len(tp), float), where=union != 0)
    mean, worst = float(ious.mean()), float(ious.min())
    return {"mean_iou": mean, "worst_iou": worst, "view_ious": ious.tolist(),
            "false_positive_pixels": np.count_nonzero(predictions & ~targets, axis=axes).tolist(),
            "false_negative_pixels": np.count_nonzero(~predictions & targets, axis=axes).tolist(),
            "passed": mean > mean_gate and worst >= worst_gate,
            "mean_gate_strict": mean_gate, "worst_gate_inclusive": worst_gate}


def _physical_rank(report):
    return (bool(report["passed"]), -max(0, report["component_count_6"] - 1),
            -len(report["reasons"]), -report["thin_voxels"], -len(report["review_reasons"]))


def _quality_deficit(quality, mean_gate, worst_gate):
    return max(0., np.nextafter(mean_gate, 1.) - quality["mean_iou"]) + max(0., worst_gate - quality["worst_iou"])


def ray_constraints(volume_xyz, targets, cameras, *, extent=1.7):
    """Exact occupied-cell counts and full-ray incidence, never centre proxies.

    Count reductions reuse the renderer's exact cardinal map. Other cameras use
    its complete streamed Siddon segments. A protected voxel is the sole hit of
    at least one required foreground ray. Background/missing votes count all
    pixel rays crossing each cell, including multiple pixels through one cell.
    Returned arrays are internal numerical data, not a JSON report.
    """
    from .rendering import _axis_map, _key, iter_ray_segments
    v = _volume(volume_xyz)
    targets = np.asarray(targets, bool)
    size, resolution = targets.shape[1], v.shape[0]
    missing_votes = np.zeros(v.shape, np.int32)
    background_votes = np.zeros(v.shape, np.int32)
    exclusive = np.zeros(v.shape, bool)
    counts = np.zeros(targets.shape, np.int32)
    flat = v.ravel()
    for index, camera in enumerate(cameras):
        target = targets[index].ravel()
        mapping = _axis_map(_key(camera), size, resolution, extent)
        if mapping is not None:
            axis, indices, valid = mapping
            ray_count = v.sum(axis=axis, dtype=np.int32).ravel()[indices]
            ray_count = np.where(valid, ray_count, 0)
            counts[index] = ray_count.reshape(size, size)
            for selection, destination in ((valid & ~target, background_votes),
                                           (valid & target & (ray_count == 0), missing_votes)):
                plane = np.bincount(indices[selection], minlength=resolution ** 2).astype(np.int32)
                destination += np.expand_dims(plane.reshape(resolution, resolution), axis)
            protected = np.zeros(resolution ** 2, bool)
            protected[indices[valid & target & (ray_count == 1)]] = True
            exclusive |= v & np.expand_dims(protected.reshape(resolution, resolution), axis)
        else:
            for chunk in iter_ray_segments(camera, size, resolution, extent):
                occupied = flat[chunk.voxel_indices]
                ray_count = np.bincount(chunk.ray_indices, weights=occupied,
                                        minlength=chunk.stop - chunk.start).astype(np.int32)
                counts[index].ravel()[chunk.start:chunk.stop] = ray_count
                desired = target[chunk.start:chunk.stop]
                background = ~desired[chunk.ray_indices]
                missing = (desired & (ray_count == 0))[chunk.ray_indices]
                np.add.at(background_votes.ravel(), chunk.voxel_indices[background], 1)
                np.add.at(missing_votes.ravel(), chunk.voxel_indices[missing], 1)
                sole = (desired & (ray_count == 1))[chunk.ray_indices] & occupied
                exclusive.ravel()[chunk.voxel_indices[sole]] = True
    return {"counts": counts, "exclusive_hits": exclusive,
            "missing_votes": missing_votes, "background_votes": background_votes}


def _box_ray_pixels(lo, hi, cameras, image_size, resolution, extent):
    """Exact positive-length ray/solid-cell-box hits on only its projected ROI."""
    from .rendering import _parallel_cell_indices

    lower = np.asarray(lo) * extent / resolution - extent / 2
    upper = np.asarray(hi) * extent / resolution - extent / 2
    corners = np.array([[x, y, z] for x in (lower[0], upper[0])
                        for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
    result = []
    for camera in cameras:
        origin, right, up, direction = map(lambda key: np.asarray(getattr(camera, key), float),
                                           ("origin", "right", "up", "direction"))
        delta = corners - origin
        xy = np.column_stack([(delta @ right / camera.scale + .5) * image_size - .5,
                              (.5 - delta @ up / camera.scale) * image_size - .5])
        first = np.maximum(np.ceil(xy.min(0) - 1e-10).astype(int), 0)
        last = np.minimum(np.floor(xy.max(0) + 1e-10).astype(int), image_size - 1)
        if np.any(last < first):
            result.append(np.empty(0, np.int64))
            continue
        columns, rows = np.meshgrid(np.arange(first[0], last[0] + 1),
                                    np.arange(first[1], last[1] + 1))
        columns, rows = columns.ravel(), rows.ravel()
        origins = (origin + ((columns + .5) / image_size - .5)[:, None] * camera.scale * right
                   + (.5 - (rows + .5) / image_size)[:, None] * camera.scale * up)
        near, far = np.zeros(len(rows)), np.full(len(rows), np.inf)
        for axis in range(3):
            if abs(direction[axis]) <= 1e-14:
                column = _parallel_cell_indices(origins[:, axis], resolution, extent)
                far[(column < lo[axis]) | (column >= hi[axis])] = -1
            else:
                a = (lower[axis] - origins[:, axis]) / direction[axis]
                b = (upper[axis] - origins[:, axis]) / direction[axis]
                near = np.maximum(near, np.minimum(a, b))
                far = np.minimum(far, np.maximum(a, b))
        hit = far > near + np.finfo(float).eps * max(1., extent) * 16
        result.append(rows[hit] * image_size + columns[hit])
    return result


def _batch_restore(volume, guidance, targets, cameras, constraints, width, extent, budget, protected_voids):
    boundary = ndi.binary_dilation(volume, FACE6) & ~volume
    eligible = boundary & (constraints["missing_votes"] > 0)
    points = np.argwhere(eligible)
    if not len(points):
        return None, 0
    score = (8 * constraints["missing_votes"][tuple(points.T)]
             - 12 * constraints["background_votes"][tuple(points.T)]
             + .25 * np.clip(guidance[tuple(points.T)], 0, 1))
    limit = min(len(points), max(4096, budget * 32))
    chosen = np.argpartition(-score, limit - 1)[:limit]
    chosen = chosen[np.argsort(-score[chosen], kind="stable")]
    candidate = volume.copy(); reserved = np.zeros_like(volume)
    prediction = constraints["counts"] > 0
    count = 0
    for index in chosen:
        point = points[index]
        lo = np.maximum(point - (width - 1) // 2, 0)
        hi = np.minimum(lo + width, volume.shape)
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        if reserved[box].any() or protected_voids[box].any():
            continue
        touched = _box_ray_pixels(lo, hi, cameras, targets.shape[1], volume.shape[0], extent)
        gain = sum(int(np.count_nonzero(targets[i].ravel()[pixels] & ~prediction[i].ravel()[pixels]))
                   for i, pixels in enumerate(touched))
        leakage = sum(int(np.count_nonzero(~targets[i].ravel()[pixels] & ~prediction[i].ravel()[pixels]))
                      for i, pixels in enumerate(touched))
        if gain == 0 or gain <= leakage:
            continue
        candidate[box] = True; reserved[box] = True
        for i, pixels in enumerate(touched):
            prediction[i].ravel()[pixels] = True
        count += 1
        if count >= budget:
            break
    return candidate if count else None, count


def _deletion_counts(points, cameras, size, resolution, extent):
    touched = [dict() for _ in cameras]
    for point in points:
        for i, pixels in enumerate(_box_ray_pixels(point, point + 1, cameras, size, resolution, extent)):
            for pixel in pixels:
                key = int(pixel)
                touched[i][key] = touched[i].get(key, 0) + 1
    return touched


def _can_delete(touched, counts, targets):
    return all(not np.any(targets[i].ravel()[list(pixels)]
                          & (counts[i].ravel()[list(pixels)] <= np.fromiter(pixels.values(), int)))
               for i, pixels in enumerate(touched) if pixels)


def _subtract_counts(touched, counts):
    for i, pixels in enumerate(touched):
        if pixels:
            counts[i].ravel()[list(pixels)] -= np.fromiter(pixels.values(), int)


def _locally_simple_deletion(volume, point):
    """Conservative local six-connectivity guard; global checks remain mandatory."""
    lo = np.maximum(point - 1, 0)
    hi = np.minimum(point + 2, volume.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    neighbors = volume[box].copy()
    neighbors[tuple(point - lo)] = False
    return ndi.label(neighbors, FACE6)[1] <= 1


def _batch_prune(volume, guidance, targets, cameras, constraints, extent, budget):
    """Greedy count updates prevent deleting all hits of any foreground ray."""
    candidate = volume.copy(); counts = constraints["counts"].copy()
    labels, component_count = ndi.label(volume, FACE6)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    boxes = ndi.find_objects(labels)
    removed_components = 0
    for part in np.argsort(sizes, kind="stable"):
        if not part or not sizes[part] or sizes[part] > 512 or sizes[part] == sizes.max():
            continue
        box = boxes[part - 1]
        points = np.argwhere(labels[box] == part) + np.array([b.start for b in box])
        if constraints["exclusive_hits"][tuple(points.T)].any():
            continue
        touched = _deletion_counts(points, cameras, targets.shape[1], volume.shape[0], extent)
        if _can_delete(touched, counts, targets):
            candidate[tuple(points.T)] = False
            _subtract_counts(touched, counts)
            removed_components += 1
        if removed_components >= budget:
            break
    if removed_components:
        return candidate, removed_components, "redundant_component_batch"
    surface = volume & ~ndi.binary_erosion(volume, FACE6) & ~constraints["exclusive_hits"]
    points = np.argwhere(surface)
    if not len(points):
        return None, 0, "guarded_surface_batch"
    score = constraints["background_votes"][tuple(points.T)] - .25 * guidance[tuple(points.T)]
    chosen = np.argsort(-score, kind="stable")[:max(1024, budget * 8)]
    removed = 0
    for index in chosen:
        point = points[index]
        if not _locally_simple_deletion(candidate, point):
            continue
        touched = _deletion_counts(point[None], cameras, targets.shape[1], volume.shape[0], extent)
        if _can_delete(touched, counts, targets):
            candidate[tuple(point)] = False
            _subtract_counts(touched, counts)
            removed += 1
        if removed >= budget:
            break
    return candidate if removed else None, removed, "guarded_surface_batch"


def _thicken_mask(mask, width):
    width = max(1, int(width))
    return ndi.maximum_filter(mask, size=width, origin=-1 if width % 2 == 0 else 0,
                               mode="constant", cval=0).astype(bool)


def _thick_bridge(volume, width, *, guidance=None, background_votes=None):
    """Compatibility helper for the nearest single component endpoint."""
    first = next(_thick_bridges(volume, width, guidance=guidance,
                                background_votes=background_votes, max_candidates=1), None)
    return None if first is None else first[0]


def _thick_bridges(volume, width, *, guidance=None, background_votes=None,
                   max_candidates=MAX_BRIDGE_ENDPOINTS):
    """Reuse one distance field for at most eight distinct component endpoints."""
    if type(max_candidates) is not int or not 1 <= max_candidates <= MAX_BRIDGE_ENDPOINTS:
        raise ValueError("bridge endpoint budget must be an integer from 1 to 8")
    labels, count = ndi.label(volume, FACE6)
    if count < 2:
        return
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    order = np.argsort(-sizes, kind="stable")
    order = order[sizes[order] > 0]
    centres, _ = _block_cover(volume, width)
    root = (labels == order[0]) & centres
    if not root.any():
        root = labels == order[0]
    distances, nearest = ndi.distance_transform_edt(~root, return_indices=True)
    choices = []
    boxes = ndi.find_objects(labels)
    for part in order[1:]:
        box = boxes[part - 1]
        component = labels[box] == part
        eligible = component & centres[box]
        points = (np.argwhere(eligible if eligible.any() else component)
                  + np.array([axis.start for axis in box]))
        cursor = points[np.argmin(distances[tuple(points.T)])]
        choices.append((float(distances[tuple(cursor)]), int(part), cursor))
    from skimage.graph import route_through_array
    margin = max(8, 4 * width)
    for rank, (distance, component, cursor) in enumerate(sorted(choices, key=lambda x: x[0])[:max_candidates]):
        end = nearest[(slice(None), *cursor)]
        lo = np.maximum(np.minimum(cursor, end) - margin, 0)
        hi = np.minimum(np.maximum(cursor, end) + margin + 1, volume.shape)
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        probability = np.zeros(volume[box].shape) if guidance is None else np.clip(guidance[box], 0, 1)
        leakage = np.zeros(volume[box].shape) if background_votes is None else background_votes[box]
        cost = 1. + 24. * leakage + 2. * (1. - probability)
        # Penalize the whole proposed thick footprint, not just its centreline.
        cost = ndi.maximum_filter(cost, size=width, mode="nearest")
        cost[volume[box] & centres[box]] = .25
        route, _ = route_through_array(cost, tuple(cursor - lo), tuple(end - lo),
                                       fully_connected=False, geometric=False)
        coordinates = np.asarray(route, int) + lo
        path = np.zeros_like(volume)
        path[tuple(coordinates.T)] = True
        yield volume | _thicken_mask(path, width), {
            "bridge_component_id": component, "bridge_endpoint_rank": rank,
            "bridge_endpoint_distance_voxels": distance,
            "path_cost": "thick_footprint_background_votes_and_guidance"}


def _candidates(volume, guidance, targets, cameras, predictions, extent, pitch, wall, neck,
                constraints, budget, protected_voids, disabled_stages=()):
    """A finite, local set; there is no global fill or coarse-envelope clipping."""
    neck_width = max(1, int(math.ceil(neck / pitch - 1e-12)))
    wall_width = max(1, int(math.ceil(wall / pitch - 1e-12)))
    if "fin_removal" not in disabled_stages:
        pruned, pruned_count, pruning_kind = _batch_prune(volume, guidance, targets, cameras, constraints, extent, budget)
        if pruned is not None:
            yield pruning_kind, pruned, {"patches": pruned_count, "exclusive_hit_accounting": True,
                "local_six_connectivity_guard": pruning_kind == "guarded_surface_batch"}
    if "reinforcement" not in disabled_stages:
        for bridge, extra in _thick_bridges(volume, neck_width, guidance=guidance,
                                            background_votes=constraints["background_votes"]):
            yield "physical_component_bridge", bridge, extra

    restored, patches = _batch_restore(volume, guidance, targets, cameras, constraints,
                                       max(2, wall_width), extent, budget, protected_voids)
    if restored is not None:
        yield "projection_restoration", restored, {"patches": patches, "full_ray_incidence": True}

    if len(disabled_stages) == len(STAGE_SWITCH_SEMANTICS):
        return
    thin, _, core_cover = _thin_mask(volume, pitch, wall)
    thin_labels, count = ndi.label(thin, FACE6)
    if count:
        sizes = np.bincount(thin_labels.ravel())
        sizes[0] = 0
        order = np.argsort(-sizes, kind="stable")
        selected = order[sizes[order] > 0][:MAX_THIN_REGIONS]
        distance = ndi.distance_transform_edt(~core_cover) if "fin_removal" not in disabled_stages else None
        for rank, region in enumerate(selected):
            part = thin_labels == region
            extra = {"thin_region_id": int(region), "thin_region_rank": rank,
                     "thin_region_voxels": int(sizes[region])}
            if "fin_removal" not in disabled_stages:
                yield "fin_removal", volume & ~part, {
                    **extra, "exclusive_hit_voxels_in_region": int((part & constraints["exclusive_hits"]).sum())}
                values = distance[part]
                shorten = part & (distance >= np.median(values))
                yield "fin_shortening", volume & ~shorten, extra
            if "reinforcement" not in disabled_stages:
                yield "local_fin_thickening", volume | _thicken_mask(part, wall_width), extra


def repair_geometry(occupancy_xyz, guidance_xyz, targets, cameras, *, extent=1.7,
                    longest_mm=100., max_rounds=32, min_wall_mm=.9, min_neck_mm=1.35,
                    mean_iou_gate=.98, worst_iou_gate=.97, max_patches_per_round=256,
                    disabled_stages=()):
    """Return a bounded native-resolution candidate and an honest JSON trace.

    Quality is measured using the shared complete-ray binary renderer. A passing
    state can never accept an edit failing either fidelity gate. Below the gates,
    edits must improve the measured deficit, or preserve it while improving a
    physical defect. Material-only carving requires identical rendered masks and
    nonworsening physical checks: there is no 0.01-IoU saving allowance.

    ``CANDIDATE_PASSED`` is only an occupied-cell result. The caller must measure
    actual pitch from the final uniform mesh transform, audit that mesh, reload
    serialized files, and perform independent rendering and slicer checks.
    """
    from .rendering import render_binary

    volume = _volume(occupancy_xyz).copy()
    guidance = np.asarray(guidance_xyz, np.float64)
    if guidance.shape != volume.shape or not np.isfinite(guidance).all():
        raise ValueError("guidance must be finite and match occupancy")
    extent = _positive(extent, "extent")
    if (not isinstance(disabled_stages, (tuple, list))
            or any(not isinstance(stage, str) or stage not in STAGE_SWITCH_SEMANTICS
                   for stage in disabled_stages)
            or len(set(disabled_stages)) != len(disabled_stages)):
        raise ValueError("disabled_stages must be a tuple/list of unique reinforcement or fin_removal switches")
    disabled_stages = tuple(disabled_stages)
    longest_mm = _positive(longest_mm, "longest_mm")
    wall = _positive(min_wall_mm, "min_wall_mm")
    neck = _positive(min_neck_mm, "min_neck_mm")
    if type(max_rounds) is not int or not 0 <= max_rounds <= 64:
        raise ValueError("max_rounds must be an integer from 0 to 64")
    if type(max_patches_per_round) is not int or not 1 <= max_patches_per_round <= 256:
        raise ValueError("max_patches_per_round must be an integer from 1 to 256")
    if not (.98 <= mean_iou_gate < 1 and .97 <= worst_iou_gate <= 1):
        raise ValueError("quality gates cannot be weakened below the approved values")
    target_array = np.asarray(targets)
    cameras = tuple(cameras)
    if (target_array.ndim != 3 or target_array.shape[0] != len(cameras)
            or not cameras or target_array.shape[1] != target_array.shape[2]
            or not target_array.shape[1] or not np.isin(target_array, [0, 1]).all()):
        raise ValueError("targets must be one binary square image per camera")
    target_array = np.ascontiguousarray(target_array, dtype=bool)
    image_size = target_array.shape[1]
    protected_voids = _enclosed_voids(volume)
    predictions = np.asarray(render_binary(volume, cameras, image_size, extent=extent), bool)

    def state(v, rendered):
        pitch = _pitch_from_occupied_bounds(v, longest_mm)
        return {"quality": _quality(rendered, target_array, mean_iou_gate, worst_iou_gate),
                "manufacturing": thickness_audit(v, pitch, wall, neck),
                "occupied_voxels": int(v.sum()), "pitch_mm": pitch,
                "pitch_source": "occupied_cell_bbox_uniform_scale_RECHECK_AFTER_MESH_TRANSFORM"}

    best = state(volume, predictions)
    initial = best
    history = [{"round": -1, "edit": "initial", "metrics": best}]
    attempts = []
    for round_index in range(max_rounds):
        if not volume.any():
            break
        constraints = ray_constraints(volume, target_array, cameras, extent=extent)
        if not np.array_equal(constraints["counts"] > 0, predictions):
            raise RuntimeError("ray incidence and renderer disagree")
        accepted = []
        for kind, candidate, extra in _candidates(volume, guidance, target_array, cameras,
                                           predictions, extent, best["pitch_mm"], wall, neck,
                                           constraints, max_patches_per_round, protected_voids, disabled_stages):
            row = {"round": round_index, "edit": kind,
                   "added_voxels": int((candidate & ~volume).sum()),
                   "removed_voxels": int((volume & ~candidate).sum()), "accepted": False, **extra}
            if not row["added_voxels"] and not row["removed_voxels"]:
                continue
            if not candidate.any() or np.any(candidate & protected_voids):
                row["reason"] = "EMPTY_OR_PROTECTED_CAVITY"
                attempts.append(row)
                continue
            if protected_voids.any() and np.any(protected_voids & ~_enclosed_voids(candidate)):
                row["reason"] = "ORIGINAL_CAVITY_OPENED"
                attempts.append(row)
                continue
            components = int(ndi.label(candidate, FACE6)[1])
            if components > best["manufacturing"]["component_count_6"]:
                row["reason"] = "CONNECTIVITY_REGRESSION"
                attempts.append(row)
                continue
            rendered = np.asarray(render_binary(candidate, cameras, image_size, extent=extent), bool)
            quality = _quality(rendered, target_array, mean_iou_gate, worst_iou_gate)
            row["quality"] = quality
            old_deficit = _quality_deficit(best["quality"], mean_iou_gate, worst_iou_gate)
            new_deficit = _quality_deficit(quality, mean_iou_gate, worst_iou_gate)
            if (best["quality"]["passed"] and not quality["passed"]) or new_deficit > old_deficit + 1e-12:
                row["reason"] = "QUALITY_GATE"
                attempts.append(row)
                continue
            metrics = state(candidate, rendered)
            physical = _physical_rank(metrics["manufacturing"])
            old_physical = _physical_rank(best["manufacturing"])
            same_masks = np.array_equal(rendered, predictions)
            improves = (new_deficit < old_deficit - 1e-12
                        or physical > old_physical
                        or (physical >= old_physical
                            and quality["mean_iou"] > best["quality"]["mean_iou"] + 1e-12
                            and quality["worst_iou"] >= best["quality"]["worst_iou"])
                        or (same_masks and physical >= old_physical
                            and int(candidate.sum()) < best["occupied_voxels"]))
            if not improves:
                row["reason"] = "NO_QUALITY_OR_PHYSICAL_IMPROVEMENT"
                attempts.append(row)
                continue
            row["reason"] = "ELIGIBLE"
            attempts.append(row)
            # Physical constraints and fidelity outrank material among eligible edits.
            rank = (quality["passed"], -new_deficit, physical,
                    quality["worst_iou"], quality["mean_iou"], -int(candidate.sum()))
            accepted.append((rank, candidate, rendered, metrics, row))
        if not accepted:
            break
        _, volume, predictions, best, winner = max(accepted, key=lambda item: item[0])
        winner["accepted"], winner["reason"] = True, "ACCEPTED"
        history.append({"round": round_index, "edit": winner["edit"], "metrics": best})

    passed = best["quality"]["passed"] and best["manufacturing"]["passed"]
    trace = {"status": "CANDIDATE_PASSED" if passed else "UNRESOLVED",
             "final_mesh_validated": False, "publication_eligible": False,
             "algorithm": "native_batched_full_ray_count_guarded_repair_v3",
             "extent": extent, "longest_mm": longest_mm, "max_rounds": max_rounds,
             "max_patches_per_round": max_patches_per_round,
             "candidate_budgets": {"thin_regions_per_round": MAX_THIN_REGIONS,
                 "bridge_endpoints_per_round": MAX_BRIDGE_ENDPOINTS,
                 "surface_points_considered_per_round": max(1024, max_patches_per_round * 8),
                 "accepted_patches_per_batch": max_patches_per_round},
             "disabled_stages": list(disabled_stages),
             "stage_switch_semantics": {key: list(value) for key, value in STAGE_SWITCH_SEMANTICS.items()},
             "initial": initial, "final": best, "history": history, "attempts": attempts,
             "protected_cavity_voxels": int(protected_voids.sum()),
             "global_hole_fill": False, "hard_coarse_hull": False,
             "source_occupancy_mutated": False,
             "structural_validation": "NOT_REQUESTED", "physical_print_validated": False}
    return np.ascontiguousarray(volume, dtype=bool), trace
