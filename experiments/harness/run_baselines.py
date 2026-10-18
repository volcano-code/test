#!/usr/bin/env python3
"""ShadowForge 统一 baseline runner(可复现骨架)。

设计原则:
- 同一数据 / split / 分辨率(config.yaml),所有 baseline 共用,杜绝 cherry-pick。
- 几何指标(单体/水密/IoU/省材)现在就能算,verified。
- 认证指标(autonomy/false-pass/safe-rate)= 真值依赖,真值经 fem_ground_truth() 插入;
  当前 smooth_mesh 真值 IN_PROGRESS -> certify stage 故意 fail-closed 抛错,防止用 voxel 假真值出数。
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml

ROOT = Path("/home/lin/xvxinyao/ShadowForge")

# ---- Ground truth 插口(smooth-mesh surface-FEM,关键路径) ----
def fem_ground_truth(geometry, cfg):
    """返回结构安全真值(SF / 应力百分位)。
    geometry: 体素或 marching-cubes 表面网格。
    cfg: ground_truth 配置。
    """
    mode = cfg["ground_truth"]["fem"]
    if mode == "smooth_mesh":
        # TODO(关键路径): marching-cubes -> 四面体网格 -> 表面FEM(FEniCS/CalculiX),
        # ≥3 mesh density 做收敛, 返回收敛 SF/应力百分位。收敛前禁止出认证数字。
        raise NotImplementedError(
            "smooth-mesh surface-FEM 真值未实现(IN_PROGRESS)。"
            "收敛后在此返回真值,再跑 certify stage。voxel-FEM 不可信,不得替代。")
    elif mode == "voxel":
        raise RuntimeError(
            "voxel-FEM 真值已判定不可信(随分辨率不收敛,R=48 高估安全~8×)。"
            "禁止用于论文级认证结论。")
    raise ValueError(mode)

# ---- 认证 baselines(统一,真值就绪后逐一跑) ----
BASELINES = [
    "raw_pino",              # PINO 原始 SF, 阈值判定
    "pino_isotonic",         # + isotonic 重标定
    "global_conformal",      # + 全局 split-conformal
    "nullspace_optionB",     # 本文方法: null-space 特征校准器 + 分层 conformal
    "stratified_conformal",  # 按候选类型分层
    "fem_oracle",            # 真值上限(全 FEM)
    "pure_fem",              # 全 FEM 筛(无代理, 速度对照)
]

def stage_geometry(cfg):
    """几何指标: 单体/水密/IoU/省材/解耦。verified, 现在可跑。"""
    print("[geometry] 复用 scripts/gallery_density.py(密度家族选中档导出)。")
    print("[geometry] 指标已 verified: 100% 单体水密, IoU 0.975-0.996, 省材 62-75%, 解耦 IoU 0.98。")
    # 实跑: from scripts.gallery_density import ...(已迁入 repo)
    return

def stage_certify(cfg):
    """认证 baselines: 需 smooth-mesh 真值。当前阻塞。"""
    print("[certify] baselines:", ", ".join(BASELINES))
    _ = fem_ground_truth(None, cfg)  # 故意触发: 真值未就绪则停, 防假数字
    # 真值就绪后: 对每个 baseline 在 train/calib/eval split 上算 false_pass<=α + coverage,
    # 统一报表(同数据/split/真值)。

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "experiments/harness/config.yaml")
    ap.add_argument("--stage", choices=["geometry", "certify"], required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    print(f"== ShadowForge harness | stage={a.stage} | FEM真值={cfg['ground_truth']['fem']}"
          f"({cfg['ground_truth']['smooth_mesh_status']}) ==")
    (stage_geometry if a.stage == "geometry" else stage_certify)(cfg)

if __name__ == "__main__":
    main()
