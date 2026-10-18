"""
Calibration helpers for aligning experimental mesh-FEM summaries with the
project's voxel-FEM safety-factor convention.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def percentile_sf(von_mises: np.ndarray, yield_stress: float, percentile: float) -> float:
    value = float(np.percentile(np.asarray(von_mises).ravel(), percentile))
    return float(yield_stress / max(value, 1e-12))


def sf_metrics(
    von_mises: np.ndarray,
    yield_stress: float,
    percentiles: Iterable[float] = (95.0, 96.0, 96.5, 97.0, 98.0, 99.0),
) -> Dict[str, float]:
    vm = np.asarray(von_mises).ravel()
    metrics = {
        "sf_max": float(yield_stress / max(float(vm.max()), 1e-12)),
    }
    for p in percentiles:
        metrics[f"sf_p{str(p).replace('.', 'p')}"] = percentile_sf(vm, yield_stress, p)
    return metrics


def calibrate_percentile_to_reference(
    von_mises: np.ndarray,
    yield_stress: float,
    reference_sf: float,
    search_min: float = 90.0,
    search_max: float = 99.9,
    steps: int = 1000,
) -> Dict[str, float]:
    vm = np.asarray(von_mises).ravel()
    target_vm = float(yield_stress / max(reference_sf, 1e-12))
    percentiles = np.linspace(search_min, search_max, steps)
    values = np.percentile(vm, percentiles)
    idx = int(np.argmin(np.abs(values - target_vm)))
    best_p = float(percentiles[idx])
    best_vm = float(values[idx])
    return {
        "reference_sf": float(reference_sf),
        "reference_vm_target": target_vm,
        "best_percentile": best_p,
        "best_vm": best_vm,
        "best_sf": float(yield_stress / max(best_vm, 1e-12)),
        "abs_vm_error": float(abs(best_vm - target_vm)),
    }
