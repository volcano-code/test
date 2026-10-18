#!/usr/bin/env python3
"""Batch end-to-end candidate generation and selection-policy ablation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, generate_binary_structure, label
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generation"))

from datasets import load_data_from_list
from run_all_baselines import evaluate_metrics, load_pino
from structural.analysis.fem_solver import voxel_fem_solve
from structural.analysis.load_generator import LoadGenerator
from voxel_ops import (
    build_tfs_visual_hull,
    calibrate_tfs_projection,
    connectivity_preserving_carve,
    evaluate_tfs_volume,
    make_watertight_printable,
    minimize_material_printable,
    repair_tfs_projection_gaps,
    sdf_watertight_mesh,
)


YIELD_STRESS = 50e6
TAU = 1.3


def finite_sample_quantile(values: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(int(np.ceil((len(ordered) + 1) * (1 - alpha))) - 1, len(ordered) - 1)
    return float(ordered[max(rank, 0)])


class StoredIsotonic:
    def __init__(self, x_thresholds, y_thresholds):
        self.x_thresholds = np.asarray(x_thresholds, dtype=float)
        self.y_thresholds = np.asarray(y_thresholds, dtype=float)

    def predict(self, values):
        return np.interp(
            np.asarray(values, dtype=float),
            self.x_thresholds,
            self.y_thresholds,
            left=self.y_thresholds[0],
            right=self.y_thresholds[-1],
        )


def fit_acs(alpha: float):
    recalibrated = ROOT / "certification/acs_candidate_recalibrated_20260703.json"
    if recalibrated.is_file():
        artifact = json.loads(recalibrated.read_text())
        selected = artifact["selected"]
        if math.isclose(float(alpha), float(selected["alpha"]), rel_tol=0.0, abs_tol=1e-12):
            curve = artifact["isotonic"]
            model = StoredIsotonic(
                curve["x_thresholds_log10_pino_sf"],
                curve["y_thresholds_log10_fem_sf"],
            )
            return (
                model,
                float(selected["q_safe_effective_log10"]),
                float(selected["q_unsafe_effective_log10"]),
            )
    payload = json.loads((ROOT / "certification/pino_vs_fem_hard450.json").read_text())
    fem = np.asarray(payload["fem_sf"], dtype=float)
    pino = np.asarray(payload["pino_sf"], dtype=float)
    valid = np.isfinite(fem) & np.isfinite(pino) & (fem > 0) & (pino > 0)
    x, y = np.log10(pino[valid]), np.log10(fem[valid])
    rng = np.random.default_rng(20260624)
    idx = rng.permutation(len(y))
    iso_idx, conf_idx = np.array_split(idx, 2)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[iso_idx], y[iso_idx])
    pred = iso.predict(x[conf_idx])
    q_safe = finite_sample_quantile(pred - y[conf_idx], alpha)
    q_unsafe = finite_sample_quantile(y[conf_idx] - pred, alpha)
    return iso, q_safe, q_unsafe


def n6(volume: np.ndarray) -> int:
    _, count = label(volume.astype(bool), structure=generate_binary_structure(3, 1))
    return int(count)


def keep_largest(volume: np.ndarray) -> np.ndarray:
    """去除游离尘点, 只保留最大 6-连通分量。

    打印导出(export_marching_cubes)本就执行 keep_largest, 故清尘后的体素 = 真正
    被打印的几何。在候选生成阶段统一清尘, 可让 component_count_6 / watertight / IoU /
    省材 等指标反映实际打印件, 避免 1-2 个体素的尘点把"清尘后即单体"的高保真件误判为
    不可打印。真正多块碎裂(最大块占比远<1)的候选清尘后 IoU 会显著下降, 仍会被 0.95 门槛拒。
    """
    vol = volume.astype(bool)
    lab, count = label(vol, structure=generate_binary_structure(3, 1))
    if count <= 1:
        return vol
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def fem_sf(volume: np.ndarray, load_multiplier: float) -> dict:
    load = LoadGenerator().standing_load(volume.astype(np.float32))
    t0 = time.time()
    fem = voxel_fem_solve(volume.astype(np.float32), load["force"] * load_multiplier, load["fixed_mask"], verbose=False)
    vm = fem["von_mises"][volume.astype(bool)]
    vm_max = float(np.nanmax(vm)) if vm.size else float("nan")
    sf = float(YIELD_STRESS / max(vm_max, 1e-12)) if np.isfinite(vm_max) else None
    return {"sf_fem": sf, "safe": bool(sf is not None and sf >= TAU), "elapsed_s": float(time.time() - t0)}


def make_candidates(masks, resolution: int, fix_axis: bool,
                    print_mm: float | None = None, nozzle_mm: float = 0.4,
                    wall_safety: float = 2.0) -> tuple[dict[str, np.ndarray], dict]:
    """print_mm 给定时启用 scale-aware 最小壁厚 floor:
    floor_vox = ceil(wall_safety * nozzle_mm * R / print_mm)/2 层膨胀,
    使 carve(稀端) 与 watertight(密端) 都膨胀到物理可打印壁厚 -> 同尺度下整族可打印,
    null-space 厚实(多个不同密度的可打印成员)。print_mm=None 时退回固定 t=1,2 档(旧行为)。"""
    cal = calibrate_tfs_projection(masks, search_res=min(96, resolution), fix_axis=fix_axis)
    hull, _ = build_tfs_visual_hull(masks, resolution, calibration=cal)
    gap, _ = repair_tfs_projection_gaps(masks, resolution, calibration=cal)
    carve = connectivity_preserving_carve(gap)
    min_print, _ = minimize_material_printable(gap, bridge_radius=1)
    watertight, _ = make_watertight_printable(min_print, gap)
    _s6 = generate_binary_structure(3, 1)
    hb = hull.astype(bool)
    if print_mm is not None:
        # 物理最小壁厚 -> 膨胀层数(直径 floor_vox 体素 => 半径 (floor_vox-1)/2 层)
        floor_vox = int(np.ceil(wall_safety * nozzle_mm * resolution / print_mm))
        d_floor = max(int(np.ceil((floor_vox - 1) / 2.0)), 1)   # carve 稀端膨胀到 floor
        candidates = {
            "visual_hull": hull, "gap_filled_hull": gap,
            "carve_no_bridge": carve,                       # null-space 稀疏端点(不一定可打印)
            "min_printable_bridge_r1": min_print,
            "watertight_export": watertight,                # 密端
        }
        # scale-aware 多档密度族: 从 floor(最稀可打印)逐层加密 -> 覆盖一段连续省材区间,
        # 全部壁厚 >= 物理 floor(可打印), 不同膨胀层 = null-space 中不同密度的可打印成员。
        cb = carve.astype(bool)
        for i in range(6):                                  # d_floor .. d_floor+5
            candidates[f"carve_pf{i+1}"] = binary_dilation(cb, _s6, iterations=d_floor + i) & hb
    else:
        # 旧行为: 固定 t=2,3 膨胀(分辨率无关)
        carve_t2 = binary_dilation(carve.astype(bool), _s6, iterations=1) & hb
        carve_t3 = binary_dilation(carve.astype(bool), _s6, iterations=2) & hb
        candidates = {
            "visual_hull": hull, "gap_filled_hull": gap,
            "carve_no_bridge": carve, "carve_thick2": carve_t2, "carve_thick3": carve_t3,
            "min_printable_bridge_r1": min_print, "watertight_export": watertight,
        }
    # 清尘: 候选与实际打印件(export 会 keep_largest)对齐, 使可打印性判定基于真正打印的几何
    return {k: keep_largest(v) for k, v in candidates.items() if v.any()}, {"calibration": cal, "hull": hull}


def _selection_copy(row: dict, *, status: str, verified: bool, fallback: bool) -> dict:
    result = dict(row)
    result["_selection_status"] = status
    result["_selection_verified"] = bool(verified)
    result["_selection_fallback"] = bool(fallback)
    result["selection_status"] = status
    result["selection_verified"] = bool(verified)
    result["selection_fallback"] = bool(fallback)
    return result


def _best_available(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda r: (
            float(r.get("mean_iou", 0.0)),
            int(r.get("component_count_6", 999999)) == 1,
            bool(r.get("watertight", False)),
            float(r.get("material_saved_vs_hull", -1.0)),
            bool(r.get("fem_safe", False)),
        ),
    )


def _best_printable_or_available(rows: list[dict], printable: list[dict], status: str) -> dict | None:
    pool = printable if printable else rows
    chosen = _best_available(pool)
    if chosen is None:
        return None
    return _selection_copy(chosen, status=status, verified=False, fallback=True)


def select(rows: list[dict], policy: str) -> tuple[dict | None, int]:
    printable = [r for r in rows if r["mean_iou"] >= 0.95 and r["component_count_6"] == 1 and r["watertight"]]
    ordered = sorted(printable, key=lambda r: r["occupied_voxels"])
    if not ordered:
        return _best_printable_or_available(rows, printable, "fallback_no_printable_candidate"), 0
    if policy == "lightest_printable":
        return _selection_copy(ordered[0], status="verified_fabrication_only", verified=True, fallback=False), 0
    if policy == "raw_pino":
        safe = [r for r in ordered if r["raw_pino_safe"]]
        if safe:
            return _selection_copy(safe[0], status="verified_raw_pino", verified=True, fallback=False), 0
        return _best_printable_or_available(rows, printable, "fallback_no_raw_pino_safe"), 0
    if policy == "acs_safe":
        safe = [r for r in ordered if r["acs_certified_safe"]]
        if safe:
            return _selection_copy(safe[0], status="verified_acs_safe", verified=True, fallback=False), 0
        return _best_printable_or_available(rows, printable, "fallback_no_acs_safe"), 0
    if policy == "dual_acs_fem_fallback":
        fallbacks = 0
        for row in ordered:
            if row["acs_certified_safe"]:
                return _selection_copy(row, status="verified_acs_safe", verified=True, fallback=False), fallbacks
            if row["acs_certified_unsafe"]:
                continue
            fallbacks += 1
            if row["fem_safe"]:
                return _selection_copy(row, status="verified_fem_fallback", verified=True, fallback=False), fallbacks
        return _best_printable_or_available(rows, printable, "fallback_no_structurally_verified_candidate"), fallbacks
    if policy == "fem_oracle":
        safe = [r for r in ordered if r["fem_safe"]]
        if safe:
            return _selection_copy(safe[0], status="verified_fem_oracle", verified=True, fallback=False), len(ordered)
        return _best_printable_or_available(rows, printable, "fallback_no_fem_safe_candidate"), len(ordered)
    raise ValueError(policy)


# =============================================================================
# THE FINAL SELECTION STRATEGY  (ShadowForge, 唯一权威; 取代 select() 与 select_policy)
# 优先级(按重要性):
#   ① 硬约束(最重要, 必须全满足): 清尘后单体(C6=1) ∧ 水密 ∧ IoU≥tau_iou ∧ 安全 ∧ 可制造壁厚。
#        - 安全 = dual-ACS+FEM: acs_certified_safe(自主) 或 (非acs_unsafe ∧ fem_safe)(兜底);
#                 acs_certified_unsafe 直接判不安全(0假阳, 不送FEM)。
#        - 可制造壁厚 = 腐蚀存活率 erosion_survival ≥ q_floor: 排除 1 体素骨架(难看且难打印),
#                 数据上 watertight≥0.21 / 骨架≤0.03, 门槛 0.15 干净分隔。
#   ② 均衡目标: 在'质量达标(实心)'的可行集里 最大省材。即不为了多省材去选骨架,
#        而是在实心候选中取最省者(通常 = watertight_export, 省材~40-60% 且实心好看)。
#   ③ 并列优先 fixed(更保守, 从不抬高上报数), 但有两道护栏, 缺一不优先 fixed:
#        - 省材接近: |saving - best_saving| < eps_v;
#        - IoU 不被牺牲: fixed 的 IoU ≥ 并列带内最高 IoU - eps_iou(IoU 是最重要轴,
#          差距大就选高 IoU 的, 不为 fixed 偏好掉保真)。
#        最后以 SF 大者收尾。
#   IoU 门槛 tau_iou=0.90, 与论文一致(交付件实际 IoU 普遍更高~0.97; 0.90 仅为可行下限)。
# 对齐论文 Alg/优化:  min|V|  s.t.  C6=1, W=1, 壁厚, [L(V)≥logτ].
# =============================================================================
def select_final(rows, tau_iou=0.90, q_floor=0.15, eps_v=0.02, eps_iou=0.02, eps_sf=0.1,
                 save_lo=0.40, save_hi=0.85):
    """rows: free 与 fixed 候选混合, 每条须含 branch∈{free,fixed} 及
    mean_iou, component_count_6, watertight, material_saved_vs_hull, erosion_survival,
    acs_certified_safe, acs_certified_unsafe, fem_safe, sf_fem 字段。
    返回 (chosen|None, info) ; info 含 n_fem_used / decided_by。"""
    printable = [r for r in rows
                 if r["mean_iou"] >= tau_iou and r["component_count_6"] == 1 and r["watertight"]
                 and r.get("erosion_survival", 1.0) >= q_floor          # 质量/壁厚硬门槛: 排除骨架
                 and save_lo <= r["material_saved_vs_hull"] <= save_hi]  # 省材 40-85% 区间(质量↔省材均衡, 防过薄拉丝)
    # 安全判定(dual-ACS + FEM 兜底); 统计实际用到的 FEM 次数
    safe, n_fem = [], 0
    for r in printable:
        if r["acs_certified_safe"]:
            r = {**r, "_decided_by": "ACS自主"}; safe.append(r); continue
        if r["acs_certified_unsafe"]:
            continue  # 0假阳: 明确不安全, 不送FEM, 直接弃
        n_fem += 1     # 模糊带: 送 FEM 兜底
        if r["fem_safe"]:
            safe.append({**r, "_decided_by": "FEM兜底"})
    if not safe:
        return None, {"n_fem_used": n_fem, "status": "DEFER"}
    # 目标: 最大省材
    best = max(safe, key=lambda r: r["material_saved_vs_hull"])
    # eps_v 省材并列带
    near = [r for r in safe
            if abs(r["material_saved_vs_hull"] - best["material_saved_vs_hull"]) < eps_v]
    # IoU 护栏: 不为 fixed/SF 偏好牺牲明显保真
    iou_cap = max(r["mean_iou"] for r in near)
    near = [r for r in near if r["mean_iou"] >= iou_cap - eps_iou]
    # 优先 fixed(已满足省材接近 ∧ IoU 不掉)→ SF 大者收尾
    fixed_near = [r for r in near if r.get("branch") == "fixed"]
    pool = fixed_near if fixed_near else near
    chosen = max(pool, key=lambda r: r["sf_fem"])
    return chosen, {"n_fem_used": n_fem, "decided_by": chosen["_decided_by"], "branch": chosen.get("branch")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_triples", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--branch", choices=["free", "fixed"], default="free")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--load_multiplier", type=float, default=1.0)
    parser.add_argument("--pino_checkpoint", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/candidate_selection_batch_20260624.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pino = load_pino(str(args.pino_checkpoint), device)
    iso, q_safe, q_unsafe = fit_acs(args.alpha)
    tau_log = float(np.log10(TAU))

    sil_dir = ROOT / "data/silhouettes/bench3"
    names = sorted(p.stem for p in sil_dir.glob("*.png"))
    rng = random.Random(args.seed)
    triples = [rng.sample(names, 3) for _ in range(args.n_triples)]

    cases = []
    policies = ["lightest_printable", "raw_pino", "acs_safe", "dual_acs_fem_fallback", "fem_oracle"]
    for idx, triple in enumerate(triples):
        paths = [sil_dir / f"{name}.png" for name in triple]
        silhouettes, _ = load_data_from_list([str(p) for p in paths], True, (args.resolution, args.resolution), torch.device("cpu"), 3, False)
        masks = [np.asarray(x).squeeze().astype(np.float32) for x in silhouettes]
        candidates, aux = make_candidates(masks, args.resolution, args.branch == "fixed")
        hull = aux["hull"]
        cal = aux["calibration"]
        rows = []
        for cname, volume in candidates.items():
            metrics = evaluate_tfs_volume(volume.astype(np.float32), masks, cal, image_size=args.resolution)
            try:
                mesh = sdf_watertight_mesh(volume, smooth_sigma=0.6)
                watertight = bool(mesh.is_watertight)
            except Exception:
                watertight = False
            load = LoadGenerator().standing_load(volume.astype(np.float32))
            pino_metrics = evaluate_metrics(volume.astype(np.float32), hull.astype(np.float32), load["force"] * args.load_multiplier, pino, device)
            sf_pino = float(pino_metrics["sf_min_pino"])
            pred = float(iso.predict([np.log10(max(sf_pino, 1e-12))])[0])
            fem = fem_sf(volume, args.load_multiplier)
            rows.append({
                "candidate": cname,
                "occupied_voxels": int(volume.sum()),
                "material_saved_vs_hull": float(1.0 - int(volume.sum()) / max(int(hull.sum()), 1)),
                "mean_iou": metrics["mean_iou"],
                "component_count_6": n6(volume),
                "watertight": watertight,
                "sf_pino": sf_pino,
                "raw_pino_safe": bool(sf_pino >= TAU),
                "acs_lower_sf": float(10 ** (pred - q_safe)),
                "acs_upper_sf": float(10 ** (pred + q_unsafe)),
                "acs_certified_safe": bool(pred - q_safe >= tau_log),
                "acs_certified_unsafe": bool(pred + q_unsafe < tau_log),
                "sf_fem": fem["sf_fem"],
                "fem_safe": fem["safe"],
                "fem_elapsed_s": fem["elapsed_s"],
            })
        policy_rows = {}
        for policy in policies:
            chosen, fallbacks = select(rows, policy)
            policy_rows[policy] = {
                "selected": chosen["candidate"] if chosen else None,
                "selected_voxels": chosen["occupied_voxels"] if chosen else None,
                "selected_safe": bool(chosen and chosen["fem_safe"]),
                "selected_saving": chosen["material_saved_vs_hull"] if chosen else None,
                "fem_fallbacks": fallbacks,
            }
        cases.append({"case_id": idx, "triple": triple, "n_candidates": len(rows), "candidates": rows, "policies": policy_rows})
        print(f"[{idx+1}/{len(triples)}] {triple} candidates={len(rows)}")

    summary = {}
    for policy in policies:
        selected = [c["policies"][policy] for c in cases if c["policies"][policy]["selected"] is not None]
        summary[policy] = {
            "n_selected": len(selected),
            "selection_rate": float(len(selected) / max(len(cases), 1)),
            "safe_rate": float(np.mean([r["selected_safe"] for r in selected])) if selected else None,
            "mean_saving": float(np.mean([r["selected_saving"] for r in selected])) if selected else None,
            "mean_fem_fallbacks": float(np.mean([r["fem_fallbacks"] for r in selected])) if selected else None,
        }

    result = {
        "schema_version": "candidate_selection_batch_v1",
        "n_cases": len(cases),
        "branch": args.branch,
        "resolution": args.resolution,
        "alpha": args.alpha,
        "threshold_sf": TAU,
        "mean_candidates_per_case": float(np.mean([c["n_candidates"] for c in cases])) if cases else 0.0,
        "policy_summary": summary,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
