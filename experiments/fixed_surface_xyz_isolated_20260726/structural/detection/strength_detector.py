"""StrengthDetector — fast PINO-based PASS/FAIL/UNCERTAIN strength decision.

Decision rule (log10 space, with k-sigma margin):
    margin = k * log_err_std              # default k=2, std=0.257 -> 0.514
    pass_thr_log   = log10(sf_target) + margin
    fail_thr_log   = log10(sf_target) - margin
    sf_pass_thr    = sf_target * 10**margin  (≈ 3.27 * sf_target)
    sf_fail_thr    = sf_target / 10**margin  (≈ 0.31 * sf_target)

When sf_pred is outside [fail_thr, pass_thr] the PINO decision is accepted
directly. Inside the uncertain band a single voxel-FEM call confirms the
verdict.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from structural.models.load_model import load_surrogate
from structural.models.calibrated_pino import CalibratedPINO, _ood_score
from structural.analysis.fem_solver import voxel_fem_solve, quick_sf_check
from structural.export.export_closed_loop_safe_obj import (
    export_obj_candidate,
    revoxelize_obj_to_grid,
)
from .config import DetectorConfig
from .load_resolver import resolve as resolve_load


@dataclass
class DetectionResult:
    status: str                  # "PASS" | "FAIL" | "UNCERTAIN" | "FAIL_CONFIRMED" | "PASS_CONFIRMED"
    sf_target: float
    sf_pred_raw: float           # raw PINO calibrator output
    sf_pred: float               # bias-corrected, used for decision
    sf_confirmed: Optional[float] = None
    vm_max_pred: float = 0.0
    vm_max_confirmed: Optional[float] = None
    pass_threshold: float = 0.0
    fail_threshold: float = 0.0
    bias_log10: float = 0.0
    decision_path: str = "pino_only"   # pino_only | pino+fem | ood_calibrator
    ood_score: float = 0.0
    load_case: str = "standing"
    time_ms: float = 0.0
    next_action: str = "none"          # none | repair | manual_review

    def to_dict(self) -> dict:
        return asdict(self)


class StrengthDetector:
    def __init__(self, config: DetectorConfig | None = None):
        self.cfg = config or DetectorConfig()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        print(f"[StrengthDetector] loading PINO from {Path(self.cfg.pino_ckpt).name}")
        pino, _ = load_surrogate(self.cfg.pino_ckpt, device=self.device)
        pino.eval()
        for p in pino.parameters():
            p.requires_grad_(False)
        if self.cfg.scale_ckpt and Path(self.cfg.scale_ckpt).exists():
            print(f"[StrengthDetector] overlay scale_head_v2 from {Path(self.cfg.scale_ckpt).name}")
            ft = torch.load(self.cfg.scale_ckpt, map_location=self.device, weights_only=False)
            pino.scale_head_v2.load_state_dict(ft["scale_head_v2_state_dict"])
            if ft.get("scale_branch_state_dict"):
                for name, sd in ft["scale_branch_state_dict"].items():
                    getattr(pino, name).load_state_dict(sd)
            print(f"  epoch={ft.get('epoch')}  ood_err={ft.get('ood_err')}")
        self.pino = pino
        self.cal = CalibratedPINO(
            pino,
            calibrator_path=self.cfg.calibrator_pkl,
            yield_stress=self.cfg.yield_stress,
        )

    # ---------- thresholds ----------
    def _thresholds(self, sf_target: float) -> tuple[float, float]:
        m_pass = self.cfg.k_pass * self.cfg.log_err_std_bc
        m_fail = self.cfg.k_fail * self.cfg.log_err_std_bc
        pass_thr = sf_target * (10.0 ** m_pass)
        fail_thr = sf_target / (10.0 ** m_fail)
        return pass_thr, fail_thr

    # ---------- core detect ----------
    def detect(
        self,
        density: np.ndarray,
        force: np.ndarray,
        fixed_mask: np.ndarray,
        sf_target: float = 15.0,
        load_case: str = "standing",
        fem_fallback: bool = True,
    ) -> DetectionResult:
        t0 = time.time()
        density = np.asarray(density, dtype=np.float32)
        pass_thr, fail_thr = self._thresholds(sf_target)

        sf_pred_raw = self.cal.predict_safety_factor(density, force=force)
        sf_pred = sf_pred_raw / (10.0 ** self.cfg.log_err_bias)   # bias-corrected
        vm_max_pred = self.cfg.yield_stress / max(sf_pred, 1e-12)
        ood = _ood_score(density)

        result = DetectionResult(
            status="UNCERTAIN",
            sf_target=sf_target,
            sf_pred_raw=float(sf_pred_raw),
            sf_pred=float(sf_pred),
            vm_max_pred=float(vm_max_pred),
            pass_threshold=pass_thr,
            fail_threshold=fail_thr,
            bias_log10=float(self.cfg.log_err_bias),
            decision_path="pino_only",
            ood_score=float(ood),
            load_case=load_case,
        )

        # Borderline/OOD safety gates: PINO PASS predictions in risky regimes
        # are confirmed with FEM to prevent false passes. If FEM fallback is
        # disabled, return UNCERTAIN instead of silently accepting the PASS.
        safe_zone_thr = pass_thr * self.cfg.borderline_safety_factor
        needs_pass_confirmation = False
        confirmation_reasons = []
        if sf_pred >= pass_thr and sf_pred < safe_zone_thr:
            needs_pass_confirmation = True
            confirmation_reasons.append("borderline")
        if (
            sf_pred >= pass_thr
            and ood >= self.cfg.ood_fallback_threshold
            and sf_pred < self.cfg.high_ood_pass_min_sf
        ):
            needs_pass_confirmation = True
            confirmation_reasons.append("ood")
        confirmation_tag = "_".join(confirmation_reasons) or "pass"

        if sf_pred >= pass_thr and not needs_pass_confirmation:
            # Confident PASS — well above threshold and outside OOD review band.
            result.status = "PASS"
            result.next_action = "none"
        elif sf_pred >= pass_thr and needs_pass_confirmation and fem_fallback:
            # Risky PASS — above threshold but requiring FEM confirmation.
            fem_sf, fem_vm = self._run_fem(density, force, fixed_mask)
            result.sf_confirmed = fem_sf
            result.vm_max_confirmed = fem_vm
            result.decision_path = f"pino+fem_{confirmation_tag}"
            if fem_sf >= sf_target:
                result.status = "PASS_CONFIRMED"
                result.next_action = "none"
            else:
                result.status = "FAIL_CONFIRMED"
                result.next_action = "repair"
        elif sf_pred >= pass_thr:
            # Risky PASS but no FEM fallback. Keep the detector safe by refusing
            # to make an independent PASS claim for this sample.
            result.status = "UNCERTAIN"
            result.next_action = "manual_review"
            result.decision_path = f"pino_only_blocked_{confirmation_tag}"
        elif sf_pred <= fail_thr:
            result.status = "FAIL"
            result.next_action = "repair"
        else:
            # Uncertain band — either confirm with FEM or report UNCERTAIN
            if fem_fallback:
                fem_sf, fem_vm = self._run_fem(density, force, fixed_mask)
                result.sf_confirmed = fem_sf
                result.vm_max_confirmed = fem_vm
                result.decision_path = "pino+fem"
                if fem_sf >= sf_target:
                    result.status = "PASS_CONFIRMED"
                    result.next_action = "none"
                else:
                    result.status = "FAIL_CONFIRMED"
                    result.next_action = "repair"
            else:
                result.next_action = "manual_review"

        result.time_ms = (time.time() - t0) * 1000.0
        return result

    def detect_file(
        self,
        voxel_path: str | Path,
        sf_target: float = 15.0,
        load_case: str = "standing",
        fem_fallback: bool = True,
    ) -> DetectionResult:
        density, force, fixed_mask = resolve_load(
            voxel_path,
            load_case=load_case,
            material_density=self.cfg.material_density,
            gravity=self.cfg.gravity,
            cell_size=self.cfg.cell_size,
        )
        return self.detect(
            density, force, fixed_mask,
            sf_target=sf_target, load_case=load_case, fem_fallback=fem_fallback,
        )

    # ---------- FEM fallback ----------
    def _run_fem(self, density, force, fixed_mask) -> tuple[float, float]:
        res = voxel_fem_solve(
            density=density,
            force_field=force,
            fixed_mask=fixed_mask,
            cell_size=self.cfg.cell_size,
            verbose=False,
        )
        vm_field = np.asarray(res["von_mises"])
        solid = density > 0.3
        vm_solid = vm_field[solid] if solid.any() else vm_field
        vm_max = float(vm_solid.max()) if vm_solid.size else 0.0
        sf = float(self.cfg.yield_stress / max(vm_max, 1e-12))
        return sf, vm_max

    # ---------- closed-loop OBJ export with FEM re-verification ----------
    def _closed_loop_obj_export(
        self,
        density: np.ndarray,
        sf_target: float,
        output_dir: Path,
    ) -> dict:
        """Export repaired voxels → OBJ, revoxelize, FEM re-verify.

        Searches over (dilation, threshold, smooth) combinations.
        Returns the first candidate whose revoxelized FEM SF >= sf_target,
        preferring minimal geometry change.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        search_dir = output_dir / "obj_search"
        search_dir.mkdir(parents=True, exist_ok=True)

        from scipy.ndimage import binary_dilation as bd

        dilation_iters = [0, 1, 2]
        thresholds = [0.50, 0.45, 0.40]
        smooth_sigmas = [0.0, 0.5, 1.0]
        smooth_iterations_list = [0, 1, 2]

        all_candidates = []
        chosen = None

        for dil in dilation_iters:
            vol = density if dil == 0 else bd(density > 0.5, iterations=dil).astype(np.float32)
            safe_this_level = []

            for sigma in smooth_sigmas:
                for sm_it in smooth_iterations_list:
                    for thr in thresholds:
                        tag = f"dil{dil}_sig{sigma:.1f}_sm{sm_it}_thr{thr:.2f}"
                        cand_dir = search_dir / tag
                        obj_path = cand_dir / "candidate.obj"
                        cand_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            export_obj_candidate(
                                volume=vol, obj_path=obj_path,
                                threshold=thr, smooth_sigma=sigma,
                                smooth_iterations=sm_it,
                            )
                            revox = revoxelize_obj_to_grid(obj_path, density.shape)
                            fem = quick_sf_check(
                                revox.astype(np.float32),
                                yield_stress=self.cfg.yield_stress,
                                verbose=False,
                            )
                            sf_revox = float(fem["sf_min"])
                            vol_ratio = float(revox.sum() / max(density.sum(), 1))
                            cand = {
                                "tag": tag, "dilation": dil, "threshold": thr,
                                "smooth_sigma": sigma, "smooth_iterations": sm_it,
                                "sf_revox": sf_revox,
                                "vol_ratio": vol_ratio,
                                "safe": sf_revox >= sf_target,
                                "obj_path": str(obj_path),
                            }
                        except Exception as e:
                            cand = {"tag": tag, "safe": False, "error": str(e)}
                        all_candidates.append(cand)
                        if cand.get("safe"):
                            safe_this_level.append(cand)

            if safe_this_level:
                chosen = min(safe_this_level, key=lambda c: (c["dilation"], c["vol_ratio"]))
                break

        summary = {
            "n_searched": len(all_candidates),
            "n_safe": sum(1 for c in all_candidates if c.get("safe")),
            "chosen": chosen,
            "all": all_candidates,
        }
        with open(output_dir / "obj_export_summary.json", "w") as f:
            import json
            json.dump(summary, f, indent=2)

        if chosen is not None:
            final_obj = output_dir / "closed_loop_safe.obj"
            Path(chosen["obj_path"]).replace(final_obj)
            chosen["final_obj"] = str(final_obj)
            print(f"  [obj-export] SAFE: {chosen['tag']}  "
                  f"sf_revox={chosen['sf_revox']:.2f}  vol_ratio={chosen['vol_ratio']:.3f}")
        else:
            print(f"  [obj-export] FAILED: no candidate reached SF>={sf_target} "
                  f"(searched {len(all_candidates)} combinations)")

        return summary

    # ---------- end-to-end detect + repair + obj export ----------
    def detect_and_repair(
        self,
        voxel_path: str | Path,
        sf_target: float = 15.0,
        load_case: str = "standing",
        max_repair_iters: int = 20,
        export_obj: bool = True,
    ) -> dict:
        from structural.optimization.discrete_stage_c import discrete_stage_c

        density, force_np, fixed_mask = resolve_load(
            voxel_path,
            load_case=load_case,
            material_density=self.cfg.material_density,
            gravity=self.cfg.gravity,
            cell_size=self.cfg.cell_size,
        )
        force_t = torch.from_numpy(force_np).float().to(self.device)

        r1 = self.detect(density, force_np, fixed_mask, sf_target=sf_target,
                         load_case=load_case, fem_fallback=True)
        out = {"initial_detection": r1.to_dict()}

        if r1.next_action != "repair":
            out["repair"] = None
            out["obj_export"] = None
            out["final_detection"] = out["initial_detection"]
            return out

        print(f"\n[detect_and_repair] FAIL (sf_pred={r1.sf_pred:.2f}) → repair")
        repair_dir = Path("results/detect_and_repair") / Path(voxel_path).stem
        repair_dir.mkdir(parents=True, exist_ok=True)
        res, density_new = discrete_stage_c(
            pino=self.pino, density_np=density, force_tensor=force_t, device=self.device,
            sf_target=sf_target, max_iters=max_repair_iters, batch_size=64,
            fem_force=force_np, fem_fixed_mask=fixed_mask, fem_interval=5,
            calibrator=self.cal, output_dir=str(repair_dir),
        )
        out["repair"] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                         for k, v in res.items()}

        sf_final_fem = float(res["sf_final"])
        voxel_pass = sf_final_fem >= sf_target

        # Closed-loop OBJ export: export → revoxelize → FEM re-verify
        obj_result = None
        if export_obj and voxel_pass:
            print(f"\n[detect_and_repair] voxel SF={sf_final_fem:.2f} ≥ {sf_target} → obj export")
            obj_result = self._closed_loop_obj_export(
                density_new, sf_target, repair_dir / "obj_export",
            )
        out["obj_export"] = obj_result

        # Final status: both voxel AND obj must pass for full closure
        obj_pass = obj_result is not None and obj_result.get("chosen") is not None
        if voxel_pass and (obj_pass or not export_obj):
            final_status = "PASS_CONFIRMED"
            final_action = "none"
            decision = "stage_c_fem+obj_fem" if obj_pass else "stage_c_fem"
            sf_report = float(obj_result["chosen"]["sf_revox"]) if obj_pass else sf_final_fem
        elif voxel_pass and not obj_pass:
            final_status = "VOXEL_PASS_OBJ_FAIL"
            final_action = "manual_review"
            decision = "stage_c_fem+obj_fem"
            sf_report = sf_final_fem
        else:
            final_status = "FAIL_CONFIRMED"
            final_action = "manual_review"
            decision = "stage_c_fem"
            sf_report = sf_final_fem

        vm_final = self.cfg.yield_stress / max(sf_report, 1e-12)
        pass_thr, fail_thr = self._thresholds(sf_target)
        r2 = DetectionResult(
            status=final_status,
            sf_target=sf_target,
            sf_pred_raw=float("nan"),
            sf_pred=float("nan"),
            sf_confirmed=sf_report,
            vm_max_pred=0.0,
            vm_max_confirmed=float(vm_final),
            pass_threshold=pass_thr,
            fail_threshold=fail_thr,
            bias_log10=float(self.cfg.log_err_bias),
            decision_path=decision,
            ood_score=float(_ood_score(density_new)),
            load_case=load_case,
            time_ms=0.0,
            next_action=final_action,
        )
        out["final_detection"] = r2.to_dict()
        return out
