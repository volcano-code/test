"""Detector configuration — thresholds, paths, physical constants."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DetectorConfig:
    pino_ckpt: str = str(ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    scale_ckpt: str = str(ROOT / "checkpoints/scale_head_v2_weighted/best_scale_head_v2_weighted.pt")
    calibrator_pkl: str = str(ROOT / "ood_calibrator_shadowart_v2.pkl")

    yield_stress: float = 250e6      # steel, Pa
    material_density: float = 1240.0  # kg/m^3
    gravity: float = -9.81            # m/s^2
    cell_size: float = 1.0            # voxel unit

    # Force-aware ShadowArt calibrator v2:
    #   - 38 original FEM-verified OOD samples
    #   - 74 ShadowArt near-failure/base-load scalar samples
    # Keep std conservative; high-OOD moderate PASS predictions are still
    # FEM-confirmed or returned as UNCERTAIN.
    log_err_bias: float = 0.0          # sf_pred_corrected = sf_pred / 10^bias
    log_err_std_bc: float = 0.695      # conservative residual std for thresholds
    # Asymmetric margins: k_pass > k_fail so PASS requires higher confidence
    # than FAIL. This eliminates false-pass errors on thin geometry (tested on
    # 46 shadowart samples: 0 false passes with k_pass=0.7, vs 2 at k_pass=0.5).
    k_pass: float = 0.7                # conservative margin for PASS decision
    k_fail: float = 0.5                # margin for FAIL decision

    # Borderline safety gate: if PINO says PASS but sf_pred < pass_thr * this
    # factor, force FEM confirmation. Catches near-threshold predictions that
    # may be false passes.  1.2 = 20% above pass_thr is the "safe zone".
    borderline_safety_factor: float = 1.2

    # ShadowArt/OOD guardrail: dense silhouette-intersection geometries can be
    # outside the FEM training distribution. In that regime a moderate PINO PASS
    # is treated as "needs confirmation" rather than accepted as final.
    ood_fallback_threshold: float = 0.85
    high_ood_pass_min_sf: float = 8.0

    # Legacy fields (kept for backwards compat but not used by detector)
    log_err_std: float = 0.695
    confidence_k: float = 0.5

    device: str = "cuda:0"
