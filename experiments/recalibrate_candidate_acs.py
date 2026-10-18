#!/usr/bin/env python3
"""Recalibrate ACS on candidate-domain PINO/FEM pairs with case-held-out testing."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "free": ROOT / "results/candidate_selection_batch_free50_20260624.json",
    "fixed": ROOT / "results/candidate_selection_batch_fixed50_20260624.json",
}
ARTIFACT = ROOT / "certification/acs_candidate_recalibrated_20260703.json"
REPORT = ROOT / "results/ACS_RECALIBRATION_20260703.md"
TAU = 1.3
SEED = 20260703


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_quantile(values: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(int(np.ceil((len(ordered) + 1) * (1 - alpha))) - 1, len(ordered) - 1)
    return float(ordered[max(rank, 0)])


def load_rows() -> list[dict]:
    rows = []
    for branch, path in SOURCES.items():
        for case in json.loads(path.read_text())["cases"]:
            for candidate in case["candidates"]:
                rows.append({**candidate, "branch": branch, "group": int(case["case_id"])})
    return rows


def group_split(rows: list[dict]) -> dict[str, list[int]]:
    groups = sorted({row["group"] for row in rows})
    has_unsafe = {
        group: any(not row["fem_safe"] for row in rows if row["group"] == group)
        for group in groups
    }
    train_groups, holdout = train_test_split(
        groups,
        test_size=0.40,
        random_state=SEED,
        stratify=[has_unsafe[group] for group in groups],
    )
    validation_groups, test_groups = train_test_split(
        holdout,
        test_size=0.50,
        random_state=SEED + 1,
        stratify=[has_unsafe[group] for group in holdout],
    )
    iso_groups, conformal_groups = train_test_split(
        train_groups,
        test_size=0.50,
        random_state=SEED + 2,
        stratify=[has_unsafe[group] for group in train_groups],
    )
    return {
        "isotonic_fit": sorted(iso_groups),
        "conformal": sorted(conformal_groups),
        "validation": sorted(validation_groups),
        "test": sorted(test_groups),
    }


def subset(rows: list[dict], groups: list[int]) -> list[dict]:
    allowed = set(groups)
    return [row for row in rows if row["group"] in allowed]


def xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    pino = np.asarray([row["sf_pino"] for row in rows], dtype=float)
    fem = np.asarray([row["sf_fem"] for row in rows], dtype=float)
    valid = np.isfinite(pino) & np.isfinite(fem) & (pino > 0) & (fem > 0)
    return np.log10(pino[valid]), np.log10(fem[valid])


def manufacturing(row: dict) -> bool:
    return bool(
        row["mean_iou"] >= 0.95
        and row["component_count_6"] == 1
        and row["watertight"]
        and 0.40 <= row["material_saved_vs_hull"] <= 0.85
    )


def metrics(rows: list[dict], iso, q_safe: float, q_unsafe: float, safe_margin: float, unsafe_margin: float) -> dict:
    x, y = xy(rows)
    pred = iso.predict(x)
    threshold = math.log10(TAU)
    safe = y >= threshold
    cert_safe = pred - q_safe >= math.log10(TAU * safe_margin)
    cert_unsafe = pred + q_unsafe < math.log10(TAU / unsafe_margin)
    decided = cert_safe | cert_unsafe
    false_safe = cert_safe & ~safe
    false_unsafe = cert_unsafe & safe
    return {
        "n": int(len(y)),
        "safe": int(safe.sum()),
        "unsafe": int((~safe).sum()),
        "certified_safe": int(cert_safe.sum()),
        "certified_unsafe": int(cert_unsafe.sum()),
        "fallback": int((~decided).sum()),
        "autonomous_rate": float(decided.mean()),
        "fallback_rate": float((~decided).mean()),
        "false_safe": int(false_safe.sum()),
        "false_unsafe": int(false_unsafe.sum()),
    }


def evaluate(rows: list[dict], *args) -> dict:
    return {
        "all_candidates": metrics(rows, *args),
        "manufacturing_eligible": metrics([row for row in rows if manufacturing(row)], *args),
    }


def log_correlation(rows: list[dict]) -> float | None:
    x, y = xy(rows)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else None


def main() -> None:
    rows = load_rows()
    splits = group_split(rows)
    iso_rows = subset(rows, splits["isotonic_fit"])
    conformal_rows = subset(rows, splits["conformal"])
    validation_rows = subset(rows, splits["validation"])
    test_rows = subset(rows, splits["test"])

    x_fit, y_fit = xy(iso_rows)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x_fit, y_fit)
    x_cal, y_cal = xy(conformal_rows)
    pred_cal = iso.predict(x_cal)

    candidates = []
    for alpha in (0.10, 0.05, 0.02, 0.01):
        q_safe = finite_quantile(pred_cal - y_cal, alpha)
        q_unsafe = finite_quantile(y_cal - pred_cal, alpha)
        for safe_margin in (1.0, 1.05, 1.10, 1.20, 1.30, 1.50, 2.0):
            for unsafe_margin in (1.0, 1.05, 1.10, 1.20, 1.30, 1.50, 2.0):
                result = evaluate(validation_rows, iso, q_safe, q_unsafe, safe_margin, unsafe_margin)
                all_metrics = result["all_candidates"]
                candidates.append({
                    "alpha": alpha,
                    "q_safe": q_safe,
                    "q_unsafe": q_unsafe,
                    "safe_margin": safe_margin,
                    "unsafe_margin": unsafe_margin,
                    "validation": result,
                    "valid": all_metrics["false_safe"] == 0 and all_metrics["false_unsafe"] == 0,
                })
    feasible = [candidate for candidate in candidates if candidate["valid"]]
    if not feasible:
        raise RuntimeError("No zero-error validation calibration found")
    selected = max(
        feasible,
        key=lambda candidate: (
            candidate["validation"]["all_candidates"]["autonomous_rate"],
            -candidate["alpha"],
            -candidate["safe_margin"],
            -candidate["unsafe_margin"],
        ),
    )
    args = (
        iso,
        selected["q_safe"],
        selected["q_unsafe"],
        selected["safe_margin"],
        selected["unsafe_margin"],
    )
    test = evaluate(test_rows, *args)
    old_cert_safe = [row for row in rows if row["acs_certified_safe"]]
    old_cert_unsafe = [row for row in rows if row["acs_certified_unsafe"]]
    effective_q_safe = selected["q_safe"] + math.log10(selected["safe_margin"])
    effective_q_unsafe = selected["q_unsafe"] + math.log10(selected["unsafe_margin"])
    artifact = {
        "schema": "shadowforge_candidate_domain_acs_v1",
        "created": "2026-07-03",
        "threshold_sf": TAU,
        "method": "case-grouped isotonic regression plus split conformal residual bounds",
        "selection_rule": "zero false-safe and false-unsafe decisions on validation; maximize autonomous rate",
        "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in SOURCES.values()},
        "split_seed": SEED,
        "splits_by_case_id": splits,
        "split_row_counts": {
            "isotonic_fit": len(iso_rows),
            "conformal": len(conformal_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "candidate_domain_diagnostics": {
            "log_sf_correlation_all": log_correlation(rows),
            "log_sf_correlation_manufacturing_eligible": log_correlation(
                [row for row in rows if manufacturing(row)]
            ),
            "old_recorded_acs_certified_safe": len(old_cert_safe),
            "old_recorded_acs_false_safe_vs_fem": sum(not row["fem_safe"] for row in old_cert_safe),
            "old_recorded_acs_certified_unsafe": len(old_cert_unsafe),
            "old_recorded_acs_false_unsafe_vs_fem": sum(row["fem_safe"] for row in old_cert_unsafe),
        },
        "selected": {
            "alpha": selected["alpha"],
            "q_safe_raw_log10": selected["q_safe"],
            "q_unsafe_raw_log10": selected["q_unsafe"],
            "safe_margin": selected["safe_margin"],
            "unsafe_margin": selected["unsafe_margin"],
            "q_safe_effective_log10": effective_q_safe,
            "q_unsafe_effective_log10": effective_q_unsafe,
        },
        "isotonic": {
            "x_thresholds_log10_pino_sf": iso.X_thresholds_.tolist(),
            "y_thresholds_log10_fem_sf": iso.y_thresholds_.tolist(),
        },
        "validation_metrics": selected["validation"],
        "heldout_test_metrics": test,
        "limitations": [
            "FEM labels are numerical references, not physical-test ground truth.",
            "The held-out test contains 10 case groups; confidence remains limited by sample size.",
            "Calibration is valid for the recorded candidate-generation and loading distribution only.",
        ],
    }
    ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n")

    val = selected["validation"]["all_candidates"]
    tst = test["all_candidates"]
    tst_gate = test["manufacturing_eligible"]
    REPORT.write_text(
        "\n".join([
            "# ACS Recalibration",
            "",
            f"- Data: {len(rows)} candidate-domain PINO/FEM pairs from 50 grouped cases.",
            "- Split: 15 isotonic-fit cases, 15 conformal cases, 10 validation cases, 10 untouched test cases; free/fixed variants of one case never cross splits.",
            f"- Selected: alpha={selected['alpha']}, safe margin={selected['safe_margin']}, unsafe margin={selected['unsafe_margin']}.",
            f"- Candidate-domain correlation: all={log_correlation(rows):.3f}; manufacturing-eligible={log_correlation([row for row in rows if manufacturing(row)]):.3f}.",
            f"- Validation: {val['certified_safe']} safe decisions, {val['certified_unsafe']} unsafe decisions, {val['false_safe']} false-safe, {val['false_unsafe']} false-unsafe, {val['fallback_rate']:.1%} fallback.",
            f"- Held-out test: {tst['certified_safe']} safe decisions, {tst['certified_unsafe']} unsafe decisions, {tst['false_safe']} false-safe, {tst['false_unsafe']} false-unsafe, {tst['fallback_rate']:.1%} fallback.",
            f"- Held-out manufacturing-eligible subset: n={tst_gate['n']}, false-safe={tst_gate['false_safe']}, false-unsafe={tst_gate['false_unsafe']}, fallback={tst_gate['fallback_rate']:.1%}.",
            "- Verdict: fail-closed recalibration removes observed automatic decision errors but provides no autonomous decisions; PINO retraining or richer uncertainty features are required before claiming FEM replacement.",
            "- Scope: evidence is relative to stored FEM labels; physical validity is not established.",
        ]) + "\n"
    )
    print(json.dumps({"artifact": str(ARTIFACT), "report": str(REPORT), "selected": artifact["selected"], "heldout_test": test}, indent=2))


if __name__ == "__main__":
    main()
