"""Calibrate detector thresholds on 38 OOD samples.

For every (modelnet40, thingi10k) verified sample:
  1. sf_true = FEM-verified safety factor from results/stage_a/.../*.json
  2. sf_pred = CalibratedPINO prediction
  3. Compare decision vs truth under different (k, log_err_std) settings.

Outputs:
  - Per-sample table
  - Confusion matrix at various thresholds
  - Recommended (k, std) trade-off (UNCERTAIN% vs FAIL recall)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from structural.detection import StrengthDetector, DetectorConfig

ROOT = Path(__file__).resolve().parents[2]
SF_TARGET = 15.0


def load_ood_truth():
    m = json.load(open(ROOT / "results/stage_a/modelnet40/fem_verification_results.json"))
    t = json.load(open(ROOT / "results/stage_a/thingi10k/fem_verification_full.json"))
    rows = []
    for r in m["results"]:
        rows.append(("modelnet40", r["model_id"], float(r["sf_fem"])))
    for r in t["results"]:
        rows.append(("thingi10k", r["model_id"], float(r["sf_fem"])))
    return rows


def main():
    det = StrengthDetector()
    truth = load_ood_truth()
    print(f"\nCalibrating on {len(truth)} OOD samples, sf_target={SF_TARGET}\n")

    rows = []
    for dataset, mid, sf_true in truth:
        vpath = ROOT / f"test_datasets/{dataset}/{mid}/voxels.npy"
        if not vpath.exists():
            continue
        # Use fem_fallback=False so we get raw PINO decisions
        r = det.detect_file(vpath, sf_target=SF_TARGET, fem_fallback=False)
        label_true = "PASS" if sf_true >= SF_TARGET else "FAIL"
        log_err = np.log10(max(r.sf_pred, 1e-6)) - np.log10(max(sf_true, 1e-6))
        rows.append({
            "dataset": dataset, "model_id": mid,
            "sf_true": sf_true, "sf_pred": r.sf_pred,
            "log_err": float(log_err),
            "label_true": label_true,
            "ood_score": r.ood_score,
        })
        print(f"  {dataset:<12}{mid:<12}  true={sf_true:8.2f}  pred={r.sf_pred:8.2f}  log_err={log_err:+.3f}  [{label_true}]")

    print(f"\n{'='*60}\nRAW RESIDUAL STATISTICS (log10 space)")
    errs = np.array([r["log_err"] for r in rows])
    bias = float(errs.mean())
    std = float(errs.std())
    print(f"  n={len(errs)}  mean(bias)={bias:+.3f}  "
          f"std={std:.3f}  MAE={np.abs(errs).mean():.3f}  max|.|={np.abs(errs).max():.3f}")
    print(f"  bias correction factor: SF_corrected = SF_pred / 10^{bias:.3f} "
          f"(={10**bias:.3f}x)")

    errs_bc = errs - bias
    std_bc = float(errs_bc.std())
    print(f"\n{'='*60}\nBIAS-CORRECTED RESIDUAL STATISTICS")
    print(f"  std_bc={std_bc:.3f}  MAE_bc={np.abs(errs_bc).mean():.3f}  "
          f"max|.|_bc={np.abs(errs_bc).max():.3f}")

    print(f"\n{'='*60}\nASYMMETRIC THRESHOLD SWEEP (bias-corrected, sf_target=15)")
    print(f"  rule: sf_pred_corrected = sf_pred / 10^bias")
    print(f"  PASS band uses k_pass*std, FAIL band uses k_fail*std  (asymmetric)")
    print(f"{'k_pass':>7}{'k_fail':>7}{'pass_thr':>11}{'fail_thr':>11}"
          f"{'TP':>5}{'TN':>5}{'FP':>5}{'FN':>5}{'UNC':>5}{'miss':>6}")

    best = None
    for k_pass in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for k_fail in [0.5, 1.0, 1.5, 2.0]:
            m_pass = k_pass * std_bc
            m_fail = k_fail * std_bc
            pass_thr = SF_TARGET * (10 ** m_pass)
            fail_thr = SF_TARGET / (10 ** m_fail)
            tp = tn = fp = fn = unc = fail_miss = 0
            for r in rows:
                sf_p_bc = r["sf_pred"] / (10 ** bias)
                sf_t, lab = r["sf_true"], r["label_true"]
                if sf_p_bc >= pass_thr:
                    if lab == "PASS":
                        tp += 1
                    else:
                        fn += 1
                        fail_miss += 1
                elif sf_p_bc <= fail_thr:
                    if lab == "FAIL":
                        tn += 1
                    else:
                        fp += 1
                else:
                    unc += 1
            marker = ""
            if fail_miss == 0:
                marker = "  *"
            print(f"{k_pass:>7.1f}{k_fail:>7.1f}{pass_thr:>11.2f}{fail_thr:>11.2f}"
                  f"{tp:>5}{tn:>5}{fp:>5}{fn:>5}{unc:>5}{fail_miss:>6}{marker}")
            # Hard constraint: 0 missed FAIL. Then maximize decided cases, min FP.
            if fail_miss == 0:
                decided = tp + tn
                score = decided - 0.5 * fp
                if best is None or score > best[0]:
                    best = (score, k_pass, k_fail, pass_thr, fail_thr, tp, tn, fp, unc)

    if best is not None:
        _, k_p, k_f, pth, fth, tp, tn, fp, unc = best
        print(f"\nRECOMMENDED (0 missed FAIL, max decided, min FP):")
        print(f"  k_pass={k_p}  k_fail={k_f}  bias={bias:+.3f}  std_bc={std_bc:.3f}")
        print(f"  pass_thr={pth:.2f}  fail_thr={fth:.2f}")
        print(f"  decided_PASS={tp}  decided_FAIL={tn}  false_repair={fp}  "
              f"UNCERTAIN={unc}/{len(rows)} ({100*unc/len(rows):.0f}%)")
        print(f"  FEM speedup: {len(rows)/(unc+1e-6):.1f}x vs pure-FEM  "
              f"(FEM saved on {len(rows)-unc} samples)")
    else:
        print("\nNO configuration achieves 0 missed FAIL — falling back to pure-FEM")

    out = {
        "n_samples": len(rows),
        "residual_raw": {
            "mean": bias, "std": std,
            "mae": float(np.abs(errs).mean()),
            "max_abs": float(np.abs(errs).max()),
        },
        "residual_bias_corrected": {
            "std": std_bc,
            "mae": float(np.abs(errs_bc).mean()),
            "max_abs": float(np.abs(errs_bc).max()),
        },
        "bias": bias,
        "recommended_k_pass": best[1] if best else None,
        "recommended_k_fail": best[2] if best else None,
        "pass_threshold_at_sf15": best[3] if best else None,
        "fail_threshold_at_sf15": best[4] if best else None,
        "rows": rows,
    }
    out_path = ROOT / "results/detector_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
