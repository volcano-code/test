"""Strict ACCS verification from the cached Hard450 FEM/PINO pairs.

This script matches the paper method:
  - random split into calibration/test sets;
  - increasing isotonic regression from surrogate log-SF to FEM log-SF;
  - one-sided conformal correction with the finite-sample rank
    k = ceil((n + 1) (1 - alpha));
  - certify only if the lower bound exceeds log10(tau).

It intentionally uses only certification/pino_vs_fem_hard450.json so it can be
rerun without the large raw FEM tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SOURCE = ROOT / "certification" / "pino_vs_fem_hard450.json"
OUT = ROOT / "results" / "accs_verify_strict.json"


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    residuals = np.sort(np.asarray(residuals, dtype=float))
    n = len(residuals)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(residuals[k - 1])


def run_once(x: np.ndarray, y: np.ndarray, alpha: float, seed: int, tau: float) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    half = len(idx) // 2
    cal_idx = idx[:half]
    test_idx = idx[half:]

    cal_perm = rng.permutation(cal_idx)
    iso_idx = cal_perm[: len(cal_perm) // 2]
    conf_idx = cal_perm[len(cal_perm) // 2 :]

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(x[iso_idx], y[iso_idx])

    residual = iso.predict(x[conf_idx]) - y[conf_idx]
    q = conformal_quantile(residual, alpha)
    lower = iso.predict(x[test_idx]) - q

    threshold = float(np.log10(tau))
    accept = lower >= threshold
    truth_safe = y[test_idx] >= threshold

    false_pass = int(np.logical_and(accept, ~truth_safe).sum())
    certified = int(accept.sum())
    true_safe = int(truth_safe.sum())
    true_safe_certified = int(np.logical_and(accept, truth_safe).sum())

    return {
        "seed": int(seed),
        "alpha": float(alpha),
        "q": q,
        "n_test": int(len(test_idx)),
        "n_true_safe": true_safe,
        "n_certified": certified,
        "n_false_pass": false_pass,
        "safe_certification_rate": float(true_safe_certified / max(true_safe, 1)),
        "acceptance_rate": float(certified / max(len(test_idx), 1)),
    }


def main() -> None:
    data = json.loads(SOURCE.read_text())
    fem = np.asarray(data["fem_sf"], dtype=float)
    pino = np.asarray(data["pino_sf"], dtype=float)
    mask = (fem > 0) & (pino > 0) & np.isfinite(fem) & np.isfinite(pino)
    fem = fem[mask]
    pino = pino[mask]

    x = np.log10(pino)
    y = np.log10(fem)
    tau = 1.3

    results = {}
    for alpha in (0.05, 0.10, 0.20):
        rows = [run_once(x, y, alpha, seed, tau) for seed in range(50)]
        results[f"alpha_{alpha:.2f}"] = {
            "runs": rows,
            "false_pass_mean": float(np.mean([r["n_false_pass"] for r in rows])),
            "false_pass_max": int(max(r["n_false_pass"] for r in rows)),
            "certified_mean": float(np.mean([r["n_certified"] for r in rows])),
            "safe_certification_rate_mean": float(
                np.mean([r["safe_certification_rate"] for r in rows])
            ),
            "acceptance_rate_mean": float(np.mean([r["acceptance_rate"] for r in rows])),
        }

    payload = {
        "schema_version": "accs_verify_strict_v1",
        "source": str(SOURCE),
        "n": int(len(y)),
        "tau": tau,
        "n_true_unsafe": int((y < np.log10(tau)).sum()),
        "protocol": "50 random half-test splits; inner half calibration split for isotonic/conformal",
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"Hard450 usable n={payload['n']}, unsafe={payload['n_true_unsafe']}")
    for key, rec in results.items():
        print(
            f"{key}: FP mean={rec['false_pass_mean']:.2f}, "
            f"FP max={rec['false_pass_max']}, "
            f"certified mean={rec['certified_mean']:.1f}, "
            f"safe-cert-rate={100 * rec['safe_certification_rate_mean']:.1f}%"
        )
    print(f"SAVED -> {OUT}")


if __name__ == "__main__":
    main()
