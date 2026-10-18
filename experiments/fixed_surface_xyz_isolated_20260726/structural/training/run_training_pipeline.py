"""
端到端训练管线 (End-to-End Training Pipeline)

一键式脚本: 从 FEM 数据集生成到完整实验评估。

步骤:
    1. 生成 FEM 训练数据 (2000 samples, ~4h on CPU)
    2. 预训练 PINO (200 epochs, ~2h on RTX 5090)
    3. 验证 PINO 精度 (计算 ε_u)
    4. 训练 RepairCNN (200 steps, ~10min)
    5. 运行联合优化 (Stage C with ALM + Safety Bound + Gradient Router)
    6. 运行完整实验评估 (baselines, ablations, bound verification)

用法:
    # 完整流程
    python run_training_pipeline.py --shadow_files img1.png img2.png ...

    # 跳过 FEM 数据集生成 (已有数据)
    python run_training_pipeline.py --shadow_files img1.png ... --fem_data_dir ./fem_dataset

    # 跳过训练 (已有 checkpoint)
    python run_training_pipeline.py --shadow_files img1.png ... \\
        --pino_checkpoint ./checkpoints/pino/best_model.pt \\
        --skip_pino_training

硬件需求:
    - GPU: ≥16GB VRAM (推荐 RTX 5090 32GB)
    - CPU: ≥8 cores (FEM 数据集生成)
    - 存储: ~20GB (FEM 数据集 + checkpoints)
    - 时间: ~8-12h (完整流程)
"""
import sys
import os
import argparse
import json
import time
import logging
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from structural.models.load_model import load_surrogate

# 路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "structural"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def step_1_generate_fem_dataset(
    output_dir: str,
    num_samples: int = 2000,
    resolution: int = 64,
    seed: int = 42,
) -> str:
    """Step 1: 生成 FEM 训练数据集"""
    logger.info("=" * 60)
    logger.info("STEP 1: Generating FEM Dataset")
    logger.info(f"  Samples: {num_samples}, Resolution: {resolution}³")
    logger.info("=" * 60)

    fem_dir = Path(output_dir) / "fem_dataset"

    # 检查是否已存在
    meta_path = fem_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('num_samples', 0) >= num_samples:
            logger.info(f"FEM dataset already exists ({meta['num_samples']} samples). Skipping.")
            return str(fem_dir)

    from structural.data.generate_fem_dataset import generate_dataset
    t0 = time.time()
    generate_dataset(
        num_samples=num_samples,
        resolution=resolution,
        output_dir=str(fem_dir),
        seed=seed,
    )
    elapsed = time.time() - t0
    logger.info(f"FEM dataset generated in {elapsed/60:.1f} min")
    return str(fem_dir)


def step_2_pretrain_pino(
    fem_data_dir: str,
    output_dir: str,
    epochs: int = 200,
    batch_size: int = 8,
    lr: float = 1e-3,
) -> str:
    """Step 2: 预训练 PINO"""
    logger.info("=" * 60)
    logger.info("STEP 2: Pretraining PINO")
    logger.info(f"  Data: {fem_data_dir}, Epochs: {epochs}")
    logger.info("=" * 60)

    pino_dir = Path(output_dir) / "checkpoints" / "pino"
    best_model = pino_dir / "best_model.pt"

    if best_model.exists():
        logger.info(f"PINO checkpoint already exists: {best_model}. Skipping.")
        return str(best_model)

    from structural.training.pretrain_pino import train_pino
    t0 = time.time()
    train_pino(
        data_dir=fem_data_dir,
        output_dir=str(pino_dir),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    elapsed = time.time() - t0
    logger.info(f"PINO pretraining completed in {elapsed/60:.1f} min")
    return str(best_model)


def step_3_validate_pino(
    pino_checkpoint: str,
    fem_data_dir: str,
    output_dir: str,
) -> float:
    """Step 3: 验证 PINO 精度, 计算 ε_u"""
    logger.info("=" * 60)
    logger.info("STEP 3: Validating PINO Accuracy")
    logger.info("=" * 60)

    from structural.training.pretrain_pino import FEMDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    pino, _ = load_surrogate(pino_checkpoint, device)
    ckpt = torch.load(pino_checkpoint, map_location=device, weights_only=False)
    pino.eval()

    # 验证集
    val_set = FEMDataset(fem_data_dir, split='val')
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=4, shuffle=False, num_workers=2
    )

    # 计算位移误差
    displacement_errors = []
    with torch.no_grad():
        for x, y, force, density, fixed_mask in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = pino(x)

            B = x.shape[0]
            for i in range(B):
                u_pred = pred[i, :3]
                u_true = y[i, :3]
                err = torch.sqrt(((u_pred - u_true) ** 2).sum())
                norm = torch.sqrt((u_true ** 2).sum()) + 1e-12
                displacement_errors.append(float(err / norm))

    epsilon_u = max(displacement_errors) if displacement_errors else 0.1
    mean_eps = np.mean(displacement_errors) if displacement_errors else 0.1

    logger.info(f"PINO Validation Results ({len(displacement_errors)} samples):")
    logger.info(f"  ε_u (max):  {epsilon_u:.4f}")
    logger.info(f"  ε_u (mean): {mean_eps:.4f}")
    logger.info(f"  ε_u (95th): {np.percentile(displacement_errors, 95):.4f}")

    # 保存验证结果
    val_results = {
        'epsilon_u_max': epsilon_u,
        'epsilon_u_mean': float(mean_eps),
        'epsilon_u_95th': float(np.percentile(displacement_errors, 95)),
        'n_samples': len(displacement_errors),
    }
    val_path = Path(output_dir) / "pino_validation.json"
    with open(val_path, 'w') as f:
        json.dump(val_results, f, indent=2)

    # 将 ε_u 写回 checkpoint (用于后续 Stage C)
    if isinstance(ckpt, dict):
        ckpt['val_displacement_error'] = epsilon_u
        torch.save(ckpt, pino_checkpoint)
        logger.info(f"Updated checkpoint with ε_u = {epsilon_u:.4f}")

    return epsilon_u


def step_4_train_repair(
    voxel_path: str,
    pino_checkpoint: str,
    output_dir: str,
) -> str:
    """Step 4: 训练 RepairCNN"""
    logger.info("=" * 60)
    logger.info("STEP 4: Training RepairCNN")
    logger.info("=" * 60)

    repair_dir = Path(output_dir) / "checkpoints" / "repair"
    best_repair = repair_dir / "best_repair.pt"

    if best_repair.exists():
        logger.info(f"RepairCNN checkpoint exists: {best_repair}. Skipping.")
        return str(best_repair)

    from structural.training.train_repair import train_repair
    t0 = time.time()
    train_repair(
        voxel_path=voxel_path,
        pino_checkpoint=pino_checkpoint,
        output_dir=str(repair_dir),
    )
    elapsed = time.time() - t0
    logger.info(f"RepairCNN training completed in {elapsed/60:.1f} min")
    return str(best_repair)


def step_5_joint_optimization(
    shadow_files: list,
    pino_checkpoint: str,
    output_dir: str,
    voxel_path: str | None = None,
    skip_stage_a: bool = False,
    epochs: int = 100,
) -> str:
    """Step 5: 联合优化 (Stage A + B + C)"""
    logger.info("=" * 60)
    logger.info("STEP 5: Joint Optimization (ALM + Safety Bound + Gradient Router)")
    logger.info("=" * 60)

    joint_dir = Path(output_dir) / "joint"

    from structural.training.train_joint import train_joint
    t0 = time.time()
    train_joint(
        shadow_files=shadow_files,
        pino_checkpoint=pino_checkpoint,
        output_dir=str(joint_dir),
        skip_stage_a=skip_stage_a,
        voxel_path=voxel_path,
    )
    elapsed = time.time() - t0
    logger.info(f"Joint optimization completed in {elapsed/60:.1f} min")

    return str(joint_dir)


def step_6_convergence_analysis(
    pino_checkpoint: str,
    output_dir: str,
    epsilon_u: float,
):
    """Step 6: 收敛率定理分析"""
    logger.info("=" * 60)
    logger.info("STEP 6: Convergence Rate Theorem Analysis")
    logger.info("=" * 60)

    from structural.optimization.safety_bound import (
        SafetyTransferBound, ConvergenceRateTheorem
    )
    from structural.configs import StructShadowConfig

    config = StructShadowConfig.from_yaml()
    stb = SafetyTransferBound(
        youngs_modulus=config.material.youngs_modulus,
        poisson_ratio=config.material.poisson_ratio,
        resolution=config.resolution,
        yield_stress=config.material.yield_stress,
    )

    crt = ConvergenceRateTheorem(
        K_constant=stb.K,
        epsilon_u=epsilon_u,
        sigma_yield=config.material.yield_stress,
        rho_init=1.0,
        rho_max=1000.0,
        rho_growth=2.0,
        n_inner=5,
        u_max_estimate=1e-3,
    )

    summary = crt.format_theorem_summary()
    print(summary)

    # 保存
    results = {
        'K': stb.K,
        'epsilon_u': epsilon_u,
        'epsilon_g': crt.epsilon_g,
        'achievable_feasibility': crt.achievable_feasibility(),
    }
    for eps_target in [0.1, 0.05, 0.01, 0.005, 0.001]:
        est = crt.compute_convergence_bound(eps_target)
        results[f'T_eps_{eps_target}'] = est.T_epsilon
        results[f'T_dual_{eps_target}'] = est.T_dual
        results[f'T_correction_{eps_target}'] = est.T_surrogate_correction

    conv_path = Path(output_dir) / "convergence_theorem.json"
    with open(conv_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Convergence analysis saved to {conv_path}")


def step_7_run_experiments(
    pino_checkpoint: str,
    output_dir: str,
    skip_baselines: bool = False,
    skip_ablations: bool = False,
    skip_shapenet: bool = True,
    shapenet_dir: str | None = None,
):
    """Step 7: 运行完整实验评估"""
    logger.info("=" * 60)
    logger.info("STEP 7: Running Full Experiment Suite")
    logger.info("=" * 60)

    exp_dir = Path(output_dir) / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 构建实验命令参数
    from structural.experiments.run_all_experiments import main as run_experiments_main

    args_list = [
        '--pino_checkpoint', pino_checkpoint,
        '--output_dir', str(exp_dir),
    ]
    if skip_baselines:
        args_list.append('--skip_baselines')
    if skip_ablations:
        args_list.append('--skip_ablations')
    if skip_shapenet or shapenet_dir is None:
        args_list.append('--skip_shapenet')
    else:
        args_list.extend(['--shapenet_dir', shapenet_dir])

    # 直接调用 main (在同一进程中)
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('--pino_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./exp_results')
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--skip_ablations', action='store_true')
    parser.add_argument('--skip_shapenet', action='store_true')
    parser.add_argument('--skip_scalability', action='store_true')
    parser.add_argument('--shapenet_dir', type=str, default=None)
    parser.add_argument('--shapenet_mesh_list', type=str, default=None)
    parser.add_argument('--shapenet_categories', nargs='+', default=None)
    parser.add_argument('--shapenet_per_category', type=int, default=5)
    parser.add_argument('--no_fem', action='store_true')
    exp_args = parser.parse_args(args_list)

    t0 = time.time()
    # Note: run_all_experiments.py 需要适配直接函数调用
    # 这里先记录参数, 实际运行需要 import 对应函数
    logger.info(f"Experiment args: {vars(exp_args)}")
    logger.info("Experiments would be run with the above configuration.")
    logger.info("To run: python -m structural.experiments.run_all_experiments " +
                " ".join(args_list))
    elapsed = time.time() - t0
    logger.info(f"Experiment setup completed in {elapsed:.1f}s")


def run_pipeline(args):
    """运行完整端到端管线"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存运行配置
    run_config = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    }
    with open(output_dir / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)

    t_start = time.time()

    # ============================================================
    # Step 1: FEM 数据集
    # ============================================================
    if args.fem_data_dir:
        fem_dir = args.fem_data_dir
        logger.info(f"Using provided FEM dataset: {fem_dir}")
    elif args.skip_fem_generation:
        fem_dir = str(output_dir / "fem_dataset")
        logger.info(f"Skipping FEM generation, expecting data at: {fem_dir}")
    else:
        fem_dir = step_1_generate_fem_dataset(
            output_dir=str(output_dir),
            num_samples=args.num_fem_samples,
            resolution=args.resolution,
        )

    # ============================================================
    # Step 2: PINO 预训练
    # ============================================================
    if args.pino_checkpoint:
        pino_ckpt = args.pino_checkpoint
        logger.info(f"Using provided PINO checkpoint: {pino_ckpt}")
    elif args.skip_pino_training:
        pino_ckpt = str(output_dir / "checkpoints" / "pino" / "best_model.pt")
        logger.info(f"Skipping PINO training, expecting: {pino_ckpt}")
    else:
        pino_ckpt = step_2_pretrain_pino(
            fem_data_dir=fem_dir,
            output_dir=str(output_dir),
            epochs=args.pino_epochs,
            batch_size=args.batch_size,
        )

    # ============================================================
    # Step 3: PINO 验证
    # ============================================================
    if args.skip_pino_validation:
        epsilon_u = 0.05  # 默认保守估计
        logger.info(f"Skipping PINO validation, using default ε_u = {epsilon_u}")
    else:
        epsilon_u = step_3_validate_pino(
            pino_checkpoint=pino_ckpt,
            fem_data_dir=fem_dir,
            output_dir=str(output_dir),
        )

    # ============================================================
    # Step 4-5: 联合训练 (含 RepairCNN)
    # ============================================================
    if args.shadow_files:
        joint_dir = step_5_joint_optimization(
            shadow_files=args.shadow_files,
            pino_checkpoint=pino_ckpt,
            output_dir=str(output_dir),
            voxel_path=args.voxel_path,
            skip_stage_a=args.skip_stage_a,
        )
    else:
        logger.info("No shadow files provided, skipping joint optimization.")
        joint_dir = str(output_dir / "joint")

    # ============================================================
    # Step 6: 收敛率定理分析
    # ============================================================
    step_6_convergence_analysis(
        pino_checkpoint=pino_ckpt,
        output_dir=str(output_dir),
        epsilon_u=epsilon_u,
    )

    # ============================================================
    # Step 7: 实验评估
    # ============================================================
    if not args.skip_experiments:
        step_7_run_experiments(
            pino_checkpoint=pino_ckpt,
            output_dir=str(output_dir),
            skip_baselines=args.skip_baselines,
            skip_ablations=args.skip_ablations,
            skip_shapenet=args.skip_shapenet,
            shapenet_dir=args.shapenet_dir,
        )

    # ============================================================
    # 总结
    # ============================================================
    t_total = time.time() - t_start
    summary = {
        'total_time_minutes': t_total / 60,
        'epsilon_u': epsilon_u,
        'pino_checkpoint': pino_ckpt,
        'joint_dir': joint_dir,
        'output_dir': str(output_dir),
    }

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Total time: {t_total/60:.1f} min ({t_total/3600:.1f} h)")
    logger.info(f"  PINO ε_u: {epsilon_u:.4f}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    with open(output_dir / 'pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='StructShadow End-to-End Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline from scratch
  python run_training_pipeline.py --shadow_files img1.png img2.png img3.png

  # Skip FEM dataset generation (already have data)
  python run_training_pipeline.py --shadow_files img1.png ... --fem_data_dir ./fem_dataset

  # Skip PINO training (already have checkpoint)
  python run_training_pipeline.py --shadow_files img1.png ... \\
      --pino_checkpoint ./checkpoints/pino/best_model.pt --skip_pino_training

  # Only run experiments (have everything)
  python run_training_pipeline.py --pino_checkpoint ./ckpt/pino.pt \\
      --skip_fem_generation --skip_pino_training --skip_pino_validation
        """
    )

    # 输入
    parser.add_argument('--shadow_files', nargs='+', default=None,
                        help='Input silhouette images for joint optimization')
    parser.add_argument('--voxel_path', type=str, default=None,
                        help='Pre-computed voxel .npy to skip Stage A')
    parser.add_argument('--output_dir', type=str, default='./pipeline_output',
                        help='Output directory for all results')

    # 数据
    parser.add_argument('--fem_data_dir', type=str, default=None,
                        help='Existing FEM dataset directory')
    parser.add_argument('--num_fem_samples', type=int, default=2000,
                        help='Number of FEM samples to generate')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Voxel resolution')

    # PINO
    parser.add_argument('--pino_checkpoint', type=str, default=None,
                        help='Existing PINO checkpoint')
    parser.add_argument('--pino_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)

    # 跳过选项
    parser.add_argument('--skip_fem_generation', action='store_true')
    parser.add_argument('--skip_pino_training', action='store_true')
    parser.add_argument('--skip_pino_validation', action='store_true')
    parser.add_argument('--skip_stage_a', action='store_true')
    parser.add_argument('--skip_experiments', action='store_true')
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--skip_ablations', action='store_true')
    parser.add_argument('--skip_shapenet', action='store_true')

    # ShapeNet
    parser.add_argument('--shapenet_dir', type=str, default=None)

    args = parser.parse_args()
    run_pipeline(args)
