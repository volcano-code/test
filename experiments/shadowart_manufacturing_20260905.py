"""Manufacturing-aware discrete ShadowArt conversion helpers (experiment layer).

Physical-scale neck checks and local Bridge6 reinforcement supplement, never
replace, same-camera fidelity and mesh gates. Slicer success is not a physical
print certificate. No soft probability, PINO, or FEM code is modified here.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy import ndimage as ndi

from experiments import multiview_contract_delivery_20260902 as d

S6 = ndi.generate_binary_structure(3, 1)


def neck_audit(volume, *, pitch_mm, minimum_lobe_mm3=10.):
    v = d._cube(volume, bool, "volume")
    if pitch_mm <= 0 or minimum_lobe_mm3 <= 0:
        raise ValueError("physical scales must be positive")
    eroded = ndi.binary_erosion(v, S6)
    labels, count = ndi.label(eroded, S6)
    sizes = np.bincount(labels.ravel())[1:]
    cutoff = max(1, int(np.ceil(minimum_lobe_mm3 / pitch_mm ** 3)))
    significant = int(np.count_nonzero(sizes >= cutoff))
    connected = int(ndi.label(v, S6)[1])
    return {
        "component_count_6": connected,
        "erosion_radius_mm": float(pitch_mm),
        "minimum_lobe_mm3": float(minimum_lobe_mm3),
        "erosion_survival": float(eroded.sum() / max(int(v.sum()), 1)),
        "eroded_components": int(count),
        "significant_eroded_components": significant,
        "significant_lobe_volumes_mm3": sorted((sizes[sizes >= cutoff] * pitch_mm ** 3).tolist(), reverse=True),
        "robust_neck_gate": bool(connected == 1 and significant == 1),
        "scope": "one_voxel_erosion_significant_lobe_connectivity_proxy_not_FEM",
    }


def fortify_necks(volume, *, pitch_mm, minimum_lobe_mm3=10.):
    """Reinforce paths joining surviving eroded lobes; never global-fill voids.

    All generated additions are candidates, NOT automatically shadow-valid.
    Callers must re-render them and reject/rollback failed fidelity gates.
    """
    from skimage.graph import route_through_array

    v = d._cube(volume, bool, "volume").copy()
    before = neck_audit(v, pitch_mm=pitch_mm, minimum_lobe_mm3=minimum_lobe_mm3)
    eroded = ndi.binary_erosion(v, S6)
    labels, count = ndi.label(eroded, S6)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    cutoff = max(1, int(np.ceil(minimum_lobe_mm3 / pitch_mm ** 3)))
    selected = np.flatnonzero(sizes >= cutoff)
    paths = np.zeros_like(v)
    if len(selected) > 1:
        root = int(sizes.argmax())
        distances, nearest = ndi.distance_transform_edt(labels != root, return_indices=True)
        boxes = ndi.find_objects(labels)
        for part in sorted(set(selected) - {root}, key=lambda i: -sizes[i]):
            box = boxes[part - 1]
            points = np.argwhere(labels[box] == part) + np.array([b.start for b in box])
            start = points[np.argmin(distances[tuple(points.T)])]
            end = nearest[(slice(None), *start)]
            lo = np.maximum(np.minimum(start, end) - 8, 0)
            hi = np.minimum(np.maximum(start, end) + 9, np.array(v.shape))
            crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            cost = np.where(v[crop], 1., 8.)
            route, _ = route_through_array(cost, tuple(start - lo), tuple(end - lo),
                                          fully_connected=False, geometric=False)
            coords = np.asarray(route) + lo
            paths[tuple(coords.T)] = True
    locked = ndi.binary_dilation(paths, S6) if paths.any() else paths
    result = v | locked
    after = neck_audit(result, pitch_mm=pitch_mm, minimum_lobe_mm3=minimum_lobe_mm3)
    return result, locked, {"before": before, "after": after,
                            "added_voxels": int((result & ~v).sum()),
                            "deleted_voxels": 0,
                            "global_hole_fill": False}


def resolve_profile(directory, name):
    """Resolve bundled Bambu profiles including G-code templates, read-only."""
    directory = Path(directory).resolve()
    hashes = {}
    def visit(key, stack):
        if key in stack:
            raise ValueError(f"profile inheritance cycle: {stack + [key]}")
        path = (directory / (key + ".json")).resolve()
        if path.parent != directory:
            raise ValueError("profile path escapes its directory")
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        result = {}
        if data.get("inherits"):
            result.update(visit(data["inherits"], stack + [key]))
        for include in data.get("include", []):
            result.update(visit(include, stack + [key]))
        result.update({k: v for k, v in data.items() if k not in ("inherits", "include")})
        return result
    return visit(name, []), hashes


def prepare_slicer_profiles(profile_root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = {"machine": "Bambu Lab A1 0.4 nozzle", "process": "0.20mm Standard @BBL A1",
             "filament": "Bambu PLA Basic @BBL A1"}
    paths, source_hashes = {}, {}
    for kind, name in names.items():
        config, hashes = resolve_profile(Path(profile_root) / kind, name)
        source_hashes[kind] = hashes
        if kind == "process":
            config.update({"enable_support": "1", "support_type": "tree(auto)",
                           "support_on_build_plate_only": "1", "layer_height": "0.2",
                           "brim_type": "outer_only", "brim_width": "8",
                           "wall_loops": "3", "wall_generator": "arachne"})
        if kind == "machine":
            config["printer_settings_id"] = name
        elif kind == "filament":
            config["filament_settings_id"] = [name]
        else:
            config["print_settings_id"] = name
        paths[kind] = output / (kind + ".json")
        d._write(paths[kind], config)
    return paths, {"source_sha256": source_hashes,
                   "support_policy": "external_build_plate_only_removable",
                   "nozzle_mm": .4, "layer_height_mm": .2,
                   "physical_print_validated": False}


def slicing_audit(report):
    plates = report.get("sliced_plates", [])
    warnings = [str(p.get("warning_message", "")) for p in plates]
    complete = report.get("return_code") == 0 and bool(plates)
    blocked = any(any(token in w.lower() for token in ("floating", "empty", "collision", "non-manifold")) for w in warnings)
    return {"slicing_completed": complete, "warning_messages": warnings,
            "print_preparation_passed": bool(complete and not blocked),
            "physical_print_validated": False}


def select_manufacturing_candidate(rows):
    valid = [r for r in rows if r["mean_iou"] >= .95 and r["worst_iou"] >= .90
             and r["robust_neck_gate"] is True]
    if not valid:
        raise d.DeliveryFailure("NO_MANUFACTURING_CANDIDATE", "no candidate passes fidelity and local neck gates")
    best = max(r["mean_iou"] for r in valid)
    pool = [r for r in valid if r["mean_iou"] >= best - .01]
    return min(pool, key=lambda r: (r["voxels"], -r["mean_iou"]))


def audit_effective_settings(config):
    """Check exported effective settings, not just requested preset filenames."""
    def scalar(name):
        value = config.get(name)
        return value[0] if isinstance(value, list) and value else value
    expected = {"printer_model": "A1", "filament_type": "PLA", "enable_support": "1",
                "support_on_build_plate_only": "1", "support_type": "tree(auto)"}
    checks = {key: str(scalar(key)) == value for key, value in expected.items()}
    checks["printer_model"] = scalar("printer_model") in ("A1", "Bambu Lab A1")
    for key, value in {"nozzle_diameter": .4, "layer_height": .2, "wall_loops": 3.}.items():
        try:
            checks[key] = abs(float(scalar(key)) - value) < 1e-8
        except (ValueError, TypeError):
            checks[key] = False
    for key in ("filament_density", "machine_max_acceleration_x"):
        try:
            checks[key] = float(scalar(key)) > 0
        except (ValueError, TypeError):
            checks[key] = False
    return {"passed": all(checks.values()), "checks": checks,
            "observed": {key: config.get(key) for key in checks}}


def audit_sliced_archive(path):
    import re
    import zipfile
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise d.DeliveryFailure("SLICED_ARCHIVE_INVALID", "3MF checksum failure")
        gcode = archive.read("Metadata/plate_1.gcode")
        md5 = archive.read("Metadata/plate_1.gcode.md5").decode().strip().lower()
    if hashlib.md5(gcode).hexdigest() != md5:
        raise d.DeliveryFailure("SLICED_ARCHIVE_INVALID", "G-code MD5 mismatch")
    text = gcode.decode("utf-8")
    sections = re.findall(r"^; FEATURE: Support(?: interface)?\n(.*?)(?=^; FEATURE:|\Z)",
                          text, re.M | re.S)
    extrusion_sections = sum(bool(re.search(r"^G1 [^\n]*\bE(?:[1-9]\d*(?:\.\d+)?|0\.\d*[1-9]\d*)\b", s, re.M))
                             for s in sections)
    if not extrusion_sections:
        raise d.DeliveryFailure("SUPPORT_TOOLPATH_MISSING", "no support extrusion in sliced 3MF")
    return {"zip_crc_passed": True, "gcode_md5_passed": True,
            "gcode_sha256": hashlib.sha256(gcode).hexdigest(),
            "support_toolpaths_present": True, "support_extrusion_sections": extrusion_sections}


def publish_preparation(stage, output, report):
    """Publish only after mesh reload, physical-scale proxy, and real slicing."""
    import shutil
    from experiments.multiview_export_adapter_20260830 import CANONICAL_EXPORTS
    if (report.get("status") != "PRINT_PREPARATION_VALIDATED"
            or report.get("mesh_neck", {}).get("robust_neck_gate") is not True
            or report.get("slicing", {}).get("print_preparation_passed") is not True
            or report.get("effective_settings", {}).get("passed") is not True
            or not report.get("mesh", {}).get("serialized_reload_contract_qa")):
        raise d.DeliveryFailure("PREPARATION_NOT_VALIDATED", "refusing canonical publication")
    stage, output = Path(stage), Path(output)
    exports = report["mesh"]["exports"]
    sources = {canonical: stage / exports["files"][role]
               for role, canonical in CANONICAL_EXPORTS.items()}
    expected = {CANONICAL_EXPORTS[role]: value for role, value in exports["file_sha256"].items()}
    sources["shadowart_A1_PLA_supported.gcode.3mf"] = stage / "slicer" / "shadowart_supported.gcode.3mf"
    expected["shadowart_A1_PLA_supported.gcode.3mf"] = report["sliced_3mf_sha256"]
    for name, source in sources.items():
        if (output / name).exists() or not source.is_file() or d._hash(source) != expected[name]:
            raise d.DeliveryFailure("PUBLICATION_SOURCE_INVALID", f"invalid or occupied export: {name}")
    manifest = output / "MANUFACTURING.json"
    pending = output / ".MANUFACTURING.pending.json"
    if manifest.exists() or pending.exists():
        raise d.DeliveryFailure("PUBLICATION_SOURCE_INVALID", "manifest destination occupied")
    created = []
    try:
        for name, source in sources.items():
            destination = output / name
            created.append(destination)
            shutil.copy2(source, destination)
            if d._hash(destination) != expected[name]:
                raise d.DeliveryFailure("PUBLICATION_HASH_MISMATCH", name)
        created.append(pending)
        d._write(pending, {**report, "canonical_sha256": expected})
        created.append(manifest)
        pending.replace(manifest)
    except BaseException:
        # A user interrupt during a copy must not leave a partial delivery.
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return expected
