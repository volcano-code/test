"""
端到端实验脚本

完整流程:
1. (可选) 生成FEM数据集
2. (可选) 预训练PINO
3. Stage A: ShadowArt形状重建
4. Stage B: 结构修复
5. Stage C: 联合精调
6. 导出网格 + 可打印性检查

用法:
    # 完整流程
    python run_experiment.py --shadow_files img1.png img2.png --name my_experiment

    # 跳过数据集生成和PINO训练（使用已有checkpoint）
    python run_experiment.py --shadow_files img1.png img2.png \
        --pino_checkpoint path/to/pino.pt --skip_data_gen --skip_pino_train

    # 跳过Stage A（使用已有体素）
    python run_experiment.py --shadow_files img1.png img2.png \
        --pino_checkpoint path/to/pino.pt --voxel_path path/to/voxels.npy \
        --skip_data_gen --skip_pino_train --skip_stage_a
"""
import sys
import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mesh_combined2"))
sys.path.insert(0, str(PROJECT_ROOT / "structural"))


def run_experiment(args):
    """运行完整实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实验目录
    exp_name = args.name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path(args.output_base) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 保存实验配置
    exp_config = vars(args)
    exp_config['device'] = str(device)
    exp_config['start_time'] = datetime.now().isoformat()
    with open(exp_dir / 'experiment_config.json', 'w') as f:
        json.dump(exp_config, f, indent=2)

    from structural.configs import StructShadowConfig
    config = StructShadowConfig.from_yaml(args.config)

    print("=" * 80)
    print(f"StructShadow Experiment: {exp_name}")
    print(f"  Device: {device}")
    print(f"  Resolution: {config.resolution}³")
    print(f"  Shadow files: {args.shadow_files}")
    print("=" * 80)

    t_start = time.time()

    # ============================================================
    # Step 1: FEM数据集生成
    # ============================================================
    data_dir = str(exp_dir / 'fem_dataset')
    if not args.skip_data_gen:
        print("\n[Step 1/6] Generating FEM dataset...")
        from structural.data.generate_fem_dataset import generate_dataset
        generate_dataset(
            num_samples=args.num_fem_samples,
            resolution=config.resolution,
            output_dir=data_dir,
            seed=args.seed,
        )
    else:
        if args.data_dir:
            data_dir = args.data_dir
        print(f"\n[Step 1/6] Skipped. Using dataset: {data_dir}")

    # ============================================================
    # Step 2: PINO预训练
    # ============================================================
    pino_checkpoint = args.pino_checkpoint
    if not args.skip_pino_train:
        print("\n[Step 2/6] Pretraining PINO...")
        from structural.training.pretrain_pino import train_pino
        pino_dir = str(exp_dir / 'checkpoints' / 'pino')
        train_pino(
            data_dir=data_dir,
            output_dir=pino_dir,
            epochs=args.pino_epochs,
            batch_size=args.batch_size,
            lr=args.pino_lr,
            config=config,
        )
        pino_checkpoint = str(Path(pino_dir) / 'best_model.pt')
    else:
        print(f"\n[Step 2/6] Skipped. Using PINO: {pino_checkpoint}")

    if pino_checkpoint is None:
        raise ValueError("No PINO checkpoint available. Run PINO training or provide --pino_checkpoint")

    # ============================================================
    # Step 3-5: Joint Training (Stage A + B + C)
    # ============================================================
    print("\n[Step 3-5/6] Running joint training pipeline...")
    from structural.training.train_joint import train_joint
    train_joint(
        shadow_files=args.shadow_files,
        pino_checkpoint=pino_checkpoint,
        output_dir=str(exp_dir / 'training'),
        config_path=args.config,
        skip_stage_a=args.skip_stage_a,
        voxel_path=args.voxel_path,
    )

    # ============================================================
    # Step 6: 导出 + 可打印性检查
    # ============================================================
    print("\n[Step 6/6] Exporting and checking printability...")
    from structural.export.voxel_to_mesh import voxel_to_stl, voxel_to_obj
    from structural.export.printability_check import check_printability

    export_dir = exp_dir / 'export'
    export_dir.mkdir(exist_ok=True)

    # 找到最终体素
    final_voxel_candidates = [
        exp_dir / 'training' / 'stage_c' / 'final_voxels.npy',
        exp_dir / 'training' / 'stage_b' / 'repaired_voxels.npy',
    ]
    final_voxel_path = None
    for candidate in final_voxel_candidates:
        if candidate.exists():
            final_voxel_path = candidate
            break

    if final_voxel_path is None:
        print("WARNING: No final voxel file found, skipping export")
    else:
        density = np.load(str(final_voxel_path))
        if density.ndim > 3:
            density = density.squeeze()

        # 导出STL
        voxel_to_stl(
            density,
            str(export_dir / 'model.stl'),
            voxel_size=1.0,
            smooth_iterations=3,
        )

        # 导出OBJ
        voxel_to_obj(
            density,
            str(export_dir / 'model.obj'),
            voxel_size=1.0,
            smooth_iterations=3,
        )

        # 可打印性检查
        report = check_printability(density, voxel_size=1.0)
        print(report.summary())

        with open(export_dir / 'printability_report.json', 'w') as f:
            json.dump({
                'is_printable': report.is_printable,
                'issues': report.issues,
                'warnings': report.warnings,
                'min_wall_thickness': report.min_wall_thickness,
                'max_overhang_ratio': report.max_overhang_ratio,
                'num_components': report.num_components,
                'total_volume': report.total_volume,
                'bounding_box': report.bounding_box,
            }, f, indent=2)

    # 总结
    elapsed = time.time() - t_start
    summary = {
        'experiment': exp_name,
        'elapsed_seconds': elapsed,
        'device': str(device),
        'status': 'completed',
        'end_time': datetime.now().isoformat(),
    }
    with open(exp_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Experiment complete: {exp_name}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Results: {exp_dir}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StructShadow End-to-End Experiment')

    # 必需参数
    parser.add_argument('--shadow_files', nargs='+', required=True, help='Input silhouette images')

    # 实验设置
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--output_base', type=str, default='./experiments', help='Base output directory')
    parser.add_argument('--config', type=str, default=None, help='Config YAML path')
    parser.add_argument('--seed', type=int, default=42)

    # 跳过选项
    parser.add_argument('--skip_data_gen', action='store_true', help='Skip FEM dataset generation')
    parser.add_argument('--skip_pino_train', action='store_true', help='Skip PINO pretraining')
    parser.add_argument('--skip_stage_a', action='store_true', help='Skip ShadowArt shape formation')

    # 预计算路径
    parser.add_argument('--data_dir', type=str, default=None, help='Pre-existing FEM dataset')
    parser.add_argument('--pino_checkpoint', type=str, default=None, help='Pre-trained PINO checkpoint')
    parser.add_argument('--voxel_path', type=str, default=None, help='Pre-computed voxels for skipping Stage A')

    # 训练参数
    parser.add_argument('--num_fem_samples', type=int, default=2000)
    parser.add_argument('--pino_epochs', type=int, default=200)
    parser.add_argument('--pino_lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    run_experiment(args)
