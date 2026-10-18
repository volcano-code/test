#!/usr/bin/env python3
"""Audit the unified ShadowForge hard gate against existing recorded evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PRIMARY_FILES = {
    "free": RESULTS / "candidate_selection_batch_free50_20260624.json",
    "fixed": RESULTS / "candidate_selection_batch_fixed50_20260624.json",
}
GALLERY_GLOB = "gallery_FINAL/*/candidate_scores.json"
FINAL_MANIFEST = RESULTS / "nsa_style_fixed_required_10cm_256/MANIFEST.json"
OUT_JSON = RESULTS / "unified_hard_gate_audit_20260703.json"
OUT_MD = RESULTS / "UNIFIED_HARD_GATE_AUDIT_20260703.md"


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "mean": None, "max": None}
    return {"n": len(values), "min": min(values), "mean": mean(values), "max": max(values)}


def load_primary() -> list[dict]:
    rows = []
    for branch, path in PRIMARY_FILES.items():
        payload = load_json(path)
        for case in payload["cases"]:
            for candidate in case["candidates"]:
                row = dict(candidate)
                row.update(branch=branch, case_uid=f"{branch}:{case['case_id']}")
                rows.append(row)
    return rows


def load_gallery() -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob(GALLERY_GLOB)):
        for candidate in load_json(path):
            row = dict(candidate)
            row["case_uid"] = path.parent.name
            rows.append(row)
    return rows


def current_policy_safe(row: dict) -> bool:
    if row["acs_certified_safe"]:
        return True
    if row["acs_certified_unsafe"]:
        return False
    return bool(row["fem_safe"])


def structural_safe(row: dict, mode: str) -> bool:
    if mode == "current_policy":
        return current_policy_safe(row)
    if mode == "fem_reference":
        return bool(row["fem_safe"])
    raise ValueError(f"unknown structural mode: {mode}")


def gate_predicates(
    row: dict,
    iou_min: float = 0.95,
    saving_min: float = 0.40,
    saving_max: float = 0.85,
    erosion_min: float | None = None,
    structural_mode: str = "current_policy",
) -> dict[str, bool]:
    predicates = {
        "iou": row["mean_iou"] >= iou_min,
        "connected": row["component_count_6"] == 1,
        "watertight": bool(row["watertight"]),
        "saving": saving_min <= row["material_saved_vs_hull"] <= saving_max,
        "structural": structural_safe(row, structural_mode),
    }
    if erosion_min is not None:
        predicates["erosion"] = row["erosion_survival"] >= erosion_min
    return predicates


def pass_all(predicates: dict[str, bool], omit: str | None = None) -> bool:
    return all(value for key, value in predicates.items() if key != omit)


def summarize_gate(rows: list[dict], **thresholds) -> dict:
    passed = [row for row in rows if pass_all(gate_predicates(row, **thresholds))]
    return {
        "candidate_pass_count": len(passed),
        "candidate_pass_rate": len(passed) / len(rows) if rows else None,
        "candidate_pass_rate_wilson95": wilson(len(passed), len(rows)),
        "covered_cases": len({row["case_uid"] for row in passed}),
        "total_cases": len({row["case_uid"] for row in rows}),
        "mean_iou": mean([row["mean_iou"] for row in passed]) if passed else None,
        "mean_saving": mean([row["material_saved_vs_hull"] for row in passed]) if passed else None,
        "branch_counts": dict(Counter(row.get("branch", "unknown") for row in passed)),
    }


def threshold_sweeps(rows: list[dict]) -> dict:
    sweeps = {}
    configurations = {
        "iou_min": [(value, {"iou_min": value}) for value in (0.90, 0.925, 0.95, 0.975, 0.99)],
        "saving_min": [(value, {"saving_min": value}) for value in (0.0, 0.2, 0.4, 0.5, 0.6)],
        "saving_max": [(value, {"saving_max": value}) for value in (0.70, 0.80, 0.85, 0.90, 0.95)],
    }
    for name, settings in configurations.items():
        sweeps[name] = [{"threshold": value, **summarize_gate(rows, **kwargs)} for value, kwargs in settings]
    return sweeps


def leave_one_out(rows: list[dict], erosion_min: float | None = None) -> dict:
    predicates_by_row = [gate_predicates(row, erosion_min=erosion_min) for row in rows]
    full = [pass_all(predicates) for predicates in predicates_by_row]
    report = {}
    for gate in predicates_by_row[0]:
        admitted = [
            row
            for row, predicates, full_pass in zip(rows, predicates_by_row, full)
            if not full_pass and pass_all(predicates, omit=gate)
        ]
        report[gate] = {
            "newly_admitted": len(admitted),
            "fem_unsafe_among_new": sum(not row["fem_safe"] for row in admitted),
            "cases_affected": len({row["case_uid"] for row in admitted}),
            "iou": describe([row["mean_iou"] for row in admitted]),
            "saving": describe([row["material_saved_vs_hull"] for row in admitted]),
            "erosion": describe([row["erosion_survival"] for row in admitted]) if erosion_min is not None else None,
        }
    return report


def acs_audit(rows: list[dict]) -> dict:
    safe = [row for row in rows if row["acs_certified_safe"]]
    unsafe = [row for row in rows if row["acs_certified_unsafe"]]
    unresolved = [row for row in rows if not row["acs_certified_safe"] and not row["acs_certified_unsafe"]]
    false_safe = sum(not row["fem_safe"] for row in safe)
    false_unsafe = sum(row["fem_safe"] for row in unsafe)
    hybrid_errors = sum(current_policy_safe(row) != bool(row["fem_safe"]) for row in rows)
    manufacturing = [
        row for row in rows
        if pass_all({key: value for key, value in gate_predicates(row).items() if key != "structural"})
    ]
    policy_pass = [row for row in manufacturing if current_policy_safe(row)]
    return {
        "certified_safe": len(safe),
        "certified_safe_fem_unsafe": false_safe,
        "certified_safe_precision_vs_fem": (len(safe) - false_safe) / len(safe) if safe else None,
        "certified_safe_precision_wilson95": wilson(len(safe) - false_safe, len(safe)),
        "certified_unsafe": len(unsafe),
        "certified_unsafe_fem_safe": false_unsafe,
        "unresolved_fem_fallback": len(unresolved),
        "fallback_rate": len(unresolved) / len(rows),
        "fallback_rate_wilson95": wilson(len(unresolved), len(rows)),
        "hybrid_disagreement_with_stored_fem": hybrid_errors,
        "manufacturing_gate_candidates": len(manufacturing),
        "manufacturing_gate_fem_unsafe": sum(not row["fem_safe"] for row in manufacturing),
        "current_policy_pass_candidates": len(policy_pass),
        "current_policy_pass_fem_unsafe": sum(not row["fem_safe"] for row in policy_pass),
        "current_policy_pass_fem_unsafe_rate": sum(not row["fem_safe"] for row in policy_pass) / len(policy_pass),
        "current_policy_pass_fem_unsafe_rate_wilson95": wilson(
            sum(not row["fem_safe"] for row in policy_pass), len(policy_pass)
        ),
        "scope_note": "FEM is the stored numerical reference, not physical-test ground truth.",
    }


def gallery_erosion_audit(rows: list[dict]) -> dict:
    sweep = []
    for threshold in (0.0, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        sweep.append({"threshold": threshold, **summarize_gate(rows, erosion_min=threshold)})
    return {
        "n_candidates": len(rows),
        "n_cases": len({row["case_uid"] for row in rows}),
        "erosion_distribution": describe([row["erosion_survival"] for row in rows]),
        "threshold_sweep": sweep,
        "baseline_leave_one_out": leave_one_out(rows, erosion_min=0.15),
        "scope_note": "Erosion survival is a digital wall-thickness proxy; no measured print-success labels exist in this dataset.",
    }


def final_qa() -> dict:
    payload = load_json(FINAL_MANIFEST)
    examples = payload["flat_examples"]
    missing_files = []
    for example in examples:
        for key in ("stl", "voxels"):
            if not Path(example[key]).is_file():
                missing_files.append(example[key])
    return {
        "n_models": len(examples),
        "qa_pass": sum(bool(example["qa_pass"]) for example in examples),
        "all_files_present": not missing_files,
        "missing_files": missing_files,
        "iou": describe([example["iou"] for example in examples]),
        "saving": describe([example["saving"] for example in examples]),
        "erosion_survival": describe([example["qa"]["erosion_survival_recheck"] for example in examples]),
        "mesh_parts": dict(Counter(example["qa"]["mesh_parts_recheck"] for example in examples)),
        "voxel_components6": dict(Counter(example["qa"]["voxel_components6_recheck"] for example in examples)),
        "mesh_watertight_recheck_pass": sum(example["qa"]["mesh_watertight_recheck"] for example in examples),
        "scope_note": "These models were selected by the same gate; this is export verification, not independent threshold validation or physical printing evidence.",
    }


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_markdown(report: dict) -> None:
    primary = report["primary_100_case_audit"]
    baseline = primary["baseline_gate"]
    acs = primary["acs_vs_fem"]
    gallery = report["gallery_erosion_audit"]
    qa = report["selected_12_model_export_qa"]
    erosion_baseline = next(item for item in gallery["threshold_sweep"] if item["threshold"] == 0.15)
    lines = [
        "# Unified Hard-Gate Evidence Audit",
        "",
        "## Evidence-backed findings",
        "",
        f"- Primary evidence: {primary['n_cases']} cases and {primary['n_candidates']} candidates. The current-policy gate passes {baseline['candidate_pass_count']} candidates across {baseline['covered_cases']} cases; {acs['current_policy_pass_fem_unsafe']} of those conflict with stored FEM safety labels.",
        f"- ACS certifies {acs['certified_safe']} candidates safe; {acs['certified_safe_fem_unsafe']} conflict with stored FEM labels. It certifies {acs['certified_unsafe']} unsafe; {acs['certified_unsafe_fem_safe']} conflict with stored FEM labels. {acs['unresolved_fem_fallback']} candidates ({fmt(acs['fallback_rate'] * 100, 1)}%) require FEM fallback.",
        f"- Replacing ACS decisions with stored FEM labels yields {primary['fem_reference_gate']['candidate_pass_count']} passing candidates across {primary['fem_reference_gate']['covered_cases']} cases.",
        f"- Gallery erosion evidence: {gallery['n_candidates']} candidates across {gallery['n_cases']} cases. At erosion survival >= 0.15, {erosion_baseline['candidate_pass_count']} candidates across {erosion_baseline['covered_cases']} cases pass the full gate.",
        f"- Selected-export verification: {qa['qa_pass']}/{qa['n_models']} models pass recorded QA; all files present = {qa['all_files_present']}; all rechecked meshes are watertight = {qa['mesh_watertight_recheck_pass'] == qa['n_models']}.",
        "",
        "## What is and is not proved",
        "",
        "- Supported: IoU, connectivity, saving, structural-reference, and erosion gates each block recorded failure modes; selected exports satisfy the recorded digital invariants.",
        "- Refuted by current evidence: ACS decisive regions are not fully consistent with stored FEM labels. The current policy cannot be described as a sound structural certificate until recalibrated and validated on held-out cases.",
        "- Not yet proved: watertight-gate utility on a dataset containing failures, 0.15 erosion survival as a physical printability boundary, FEM labels as real-world structural truth, or print success without supports. These claims require suitable negative meshes and blinded physical outcomes.",
        "- IoU and material-saving bounds are design requirements. Sensitivity sweeps establish robustness/coverage trade-offs, not a universal mathematical optimum.",
        "",
        "## Reproducibility",
        "",
        f"- Machine-readable report: `{OUT_JSON.relative_to(ROOT)}`",
        f"- Script: `{Path(__file__).relative_to(ROOT)}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    primary = load_primary()
    gallery = load_gallery()
    source_paths = [*PRIMARY_FILES.values(), *sorted(RESULTS.glob(GALLERY_GLOB)), FINAL_MANIFEST]
    report = {
        "schema": "shadowforge_unified_hard_gate_audit_v1",
        "baseline_thresholds": {
            "mean_iou_min": 0.95,
            "component_count_6": 1,
            "watertight": True,
            "material_saving": [0.40, 0.85],
            "erosion_survival_min_where_available": 0.15,
            "structural": "ACS certified decision; unresolved cases use stored FEM label",
        },
        "primary_100_case_audit": {
            "n_candidates": len(primary),
            "n_cases": len({row["case_uid"] for row in primary}),
            "baseline_gate": summarize_gate(primary),
            "fem_reference_gate": summarize_gate(primary, structural_mode="fem_reference"),
            "acs_vs_fem": acs_audit(primary),
            "threshold_sweeps": threshold_sweeps(primary),
            "leave_one_gate_out": leave_one_out(primary),
            "unavailable_fields": ["erosion_survival", "mesh_parts"],
        },
        "gallery_erosion_audit": gallery_erosion_audit(gallery),
        "selected_12_model_export_qa": final_qa(),
        "claim_status": {
            "digital_geometry_invariants": "supported by selected-export rechecks",
            "acs_structural_screening": "not supported as a certificate: 42/700 ACS decisive labels conflict with stored FEM",
            "threshold_robustness": "quantified by sensitivity sweeps",
            "physical_printability_boundary": "not established: no print outcome labels",
            "real_world_structural_validity": "not established: no physical load/handling ground truth",
        },
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report)
    print(json.dumps({
        "json": str(OUT_JSON),
        "markdown": str(OUT_MD),
        "primary_baseline": report["primary_100_case_audit"]["baseline_gate"],
        "acs": report["primary_100_case_audit"]["acs_vs_fem"],
        "final_qa": report["selected_12_model_export_qa"],
    }, indent=2))


if __name__ == "__main__":
    main()
