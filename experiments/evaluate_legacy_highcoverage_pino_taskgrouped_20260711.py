#!/usr/bin/env python3
"""Evaluate the historical high-coverage PINO threshold on task-held-out rows."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from acs_task_level_risk_coverage_20260711 import metrics, subset


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/acs_task_level_risk_coverage_20260711"
TASKS = json.loads((OUT / "task_split_manifest.json").read_text())["splits"]
SOURCES = {
    "free": OUT / "rescored_best85/free_best85_rescored.json",
    "fixed": OUT / "rescored_best85/fixed_best85_rescored.json",
}
TAU = 1.3


def rows() -> list[dict]:
    result = []
    for branch, path in SOURCES.items():
        data = json.loads(path.read_text())
        for case in data["cases"]:
            task = "|".join(case["triple"])
            for candidate in case["candidates"]:
                p, y = float(candidate["sf_pino"]), float(candidate["sf_fem"])
                if np.isfinite(p) and np.isfinite(y) and p > 0 and y > 0:
                    result.append({"task": task, "branch": branch,
                                   "pino_log_sf": math.log10(p), "fem_log_sf": math.log10(y)})
    return result


def evaluate_threshold(name: str, threshold: float, calibration: list[dict], test: list[dict]) -> dict:
    p_cal = np.asarray([r["pino_log_sf"] for r in calibration])
    y_cal = np.asarray([r["fem_log_sf"] for r in calibration])
    p_test = np.asarray([r["pino_log_sf"] for r in test])
    y_test = np.asarray([r["fem_log_sf"] for r in test])
    truth = y_test >= math.log10(TAU)
    safe = p_test >= threshold
    # FF-relaxed policy never auto-rejects. It sends every non-accepted row
    # to FEM, so false-unsafe becomes an intentional defer rather than error.
    return {
        "name": name,
        "threshold_log_sf": threshold,
        "threshold_sf": 10 ** threshold,
        "calibration": metrics(p_cal >= threshold, np.zeros(len(p_cal), dtype=bool),
                               y_cal >= math.log10(TAU), fallback=True),
        "sealed_test": metrics(safe, np.zeros(len(safe), dtype=bool), truth, fallback=True),
    }


def main() -> None:
    all_rows = rows()
    calibration = subset(all_rows, TASKS["calibration"])
    test = subset(all_rows, TASKS["sealed_test"])
    cal_p = np.asarray([r["pino_log_sf"] for r in calibration])
    cal_y = np.asarray([r["fem_log_sf"] for r in calibration])
    unsafe_p = cal_p[cal_y < math.log10(TAU)]
    frozen_zero_fp = float(np.nextafter(np.max(unsafe_p), np.inf))
    # 0.05579774 is the historical 70/75=93.3% candidate-heldout threshold.
    records = [
        evaluate_threshold("legacy_high_coverage_93p3_candidate_protocol", 0.05579774023689765, calibration, test),
        evaluate_threshold("legacy_audit_zero_fp_candidate_protocol", 0.1718017304102375, calibration, test),
        evaluate_threshold("task_calibrated_zero_fp_ff_relaxed", frozen_zero_fp, calibration, test),
    ]
    payload = {
        "schema": "legacy_pino_taskgrouped_recheck_v1",
        "checkpoint": "pino_safe_accept90_cal300_m08_20260704/safe_side_external_hardneg_scale.pt",
        "split_unit": "ordered silhouette triple; both branches and all candidates grouped",
        "calibration_rows": len(calibration), "sealed_test_rows": len(test),
        "ff_policy": "No automatic unsafe rejection; every non-auto-safe row is deferred to FEM.",
        "records": records,
    }
    (OUT / "legacy_highcoverage_pino_taskgrouped.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
