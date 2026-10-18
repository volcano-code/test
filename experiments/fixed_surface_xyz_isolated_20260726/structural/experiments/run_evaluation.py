"""
评估脚本 CLI — 批量运行所有指标并输出结果

用法:
    # 计算 PINO ε_u (训练完成后首先运行)
    python -m structural.experiments.run_evaluation epsilon_u \
        --checkpoint checkpoints/pino/best_model.pt \
        --data_dir fem_dataset

    # 评估单个方法的输出
    python -m structural.experiments.run_evaluation single \
        --density_path results/ours/model_001.npy \
        --target_path results/target/model_001.npy \
        --method_name "Ours"

    # 批量评估并生成对比表格
    python -m structural.experiments.run_evaluation batch \
        --results_dir results/ \
        --output results/comparison_table.tex
"""
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from structural.experiments.evaluate_metrics import (
    compute_3d_iou,
    compute_chamfer_distance,
    compute_silhouette_iou,
    compute_pino_epsilon_u,
    evaluate_single,
    aggregate_results,
    format_results_table,
    format_results_markdown,
    MethodResult,
)


def cmd_epsilon_u(args):
    """计算 PINO 位移/应力相对误差 ε_u"""
    from structural.models.pino import PINO3d

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载模型
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})

    model = PINO3d(
        in_channels=cfg.get('in_channels', 4),
        out_channels=cfg.get('out_channels', 9),
        modes=cfg.get('modes', 16),
        width=cfg.get('width', 64),
        n_layers=cfg.get('n_layers', 6),
        padding=cfg.get('padding', 8),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?')}")

    # 计算 ε_u
    results = compute_pino_epsilon_u(
        model, args.data_dir, device,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 50)
    print("PINO Accuracy Report (ε_u)")
    print("=" * 50)
    print(f"  Samples evaluated:    {results['n_samples']}")
    print(f"  ε_u (median):         {results['epsilon_u_median']:.4f}  ({results['epsilon_u_median']*100:.2f}%)")
    print(f"  ε_u (mean ± std):     {results['epsilon_u_mean']:.4f} ± {results['epsilon_u_std']:.4f}")
    print(f"  ε_u (95th percentile):{results['epsilon_u_p95']:.4f}  ({results['epsilon_u_p95']*100:.2f}%)")
    print(f"  ε_u (max):            {results['epsilon_u_max']:.4f}  ({results['epsilon_u_max']*100:.2f}%)")
    print(f"  ε_σ (median):         {results['epsilon_s_median']:.4f}  ({results['epsilon_s_median']*100:.2f}%)")
    print(f"  ε_σ (mean):           {results['epsilon_s_mean']:.4f}")
    print(f"  ε_σ (95th percentile):{results['epsilon_s_p95']:.4f}")

    # Safety Transfer Bound 可用性判断
    eps_u = results['epsilon_u_p95']
    if eps_u < 0.05:
        print(f"\n  [PASS] ε_u < 5%, Safety Transfer Bound 有效")
    elif eps_u < 0.15:
        print(f"\n  [WARN] ε_u = {eps_u*100:.1f}%, Bound 保守但可用")
    else:
        print(f"\n  [FAIL] ε_u = {eps_u*100:.1f}%, Bound 过于保守, 需改进 PINO")

    # 保存结果
    output_path = Path(args.checkpoint).parent / 'epsilon_u_report.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def cmd_single(args):
    """评估单个模型"""
    density = np.load(args.density_path)
    target = np.load(args.target_path)

    print(f"Density shape: {density.shape}, Target shape: {target.shape}")

    iou = compute_3d_iou(density, target)
    cd = compute_chamfer_distance(density, target)
    v_ratio = (density > 0.5).sum() / max((target > 0.5).sum(), 1)

    print(f"\n--- {args.method_name} ---")
    print(f"  3D IoU:           {iou:.4f}")
    print(f"  Chamfer Distance: {cd:.4f}")
    print(f"  Volume Ratio:     {v_ratio:.4f}")

    # 可打印性
    from structural.export.printability_check import check_printability
    report = check_printability(density)
    print(f"  Printable:        {report.is_printable}")
    print(f"  Min Wall Thick.:  {report.min_wall_thickness:.2f} mm")
    print(f"  Overhang Ratio:   {report.max_overhang_ratio:.1%}")
    print(f"  Components:       {report.num_components}")


def cmd_batch(args):
    """批量评估所有方法并生成表格"""
    results_dir = Path(args.results_dir)
    all_summaries = []

    # 每个方法一个子目录, 内含 model_xxx.npy 文件
    target_dir = results_dir / 'target'
    if not target_dir.exists():
        print(f"Error: target directory not found at {target_dir}")
        return

    method_dirs = [d for d in results_dir.iterdir()
                   if d.is_dir() and d.name != 'target']

    for method_dir in sorted(method_dirs):
        method_name = method_dir.name
        method_results = []

        for pred_file in sorted(method_dir.glob("*.npy")):
            target_file = target_dir / pred_file.name
            if not target_file.exists():
                continue

            density = np.load(pred_file)
            target = np.load(target_file)
            model_id = pred_file.stem

            result = MethodResult(model_id=model_id, method_name=method_name)
            result.iou_3d = compute_3d_iou(density, target)
            result.chamfer_distance = compute_chamfer_distance(density, target)
            result.volume_ratio = float((density > 0.5).sum()) / max(float((target > 0.5).sum()), 1.0)

            from structural.export.printability_check import check_printability
            pr = check_printability(density)
            result.is_printable = pr.is_printable
            result.overhang_ratio = pr.max_overhang_ratio
            result.wall_thickness_min = pr.min_wall_thickness
            result.n_components = pr.num_components

            method_results.append(result)

        if method_results:
            summary = aggregate_results(method_results, method_name)
            all_summaries.append(summary)
            print(f"  {method_name}: {summary.n_models} models, "
                  f"IoU={summary.iou_mean:.3f}, CD={summary.chamfer_mean:.2f}")

    if not all_summaries:
        print("No results found.")
        return

    # 输出表格
    print("\n" + format_results_markdown(all_summaries))

    if args.output:
        latex = format_results_table(all_summaries)
        with open(args.output, 'w') as f:
            f.write(latex)
        print(f"\nLaTeX table saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description='StructShadow Evaluation CLI')
    subparsers = parser.add_subparsers(dest='command')

    # epsilon_u
    p_eps = subparsers.add_parser('epsilon_u', help='Compute PINO epsilon_u')
    p_eps.add_argument('--checkpoint', type=str, required=True)
    p_eps.add_argument('--data_dir', type=str, required=True)
    p_eps.add_argument('--max_samples', type=int, default=300)

    # single
    p_single = subparsers.add_parser('single', help='Evaluate single model')
    p_single.add_argument('--density_path', type=str, required=True)
    p_single.add_argument('--target_path', type=str, required=True)
    p_single.add_argument('--method_name', type=str, default='Unknown')

    # batch
    p_batch = subparsers.add_parser('batch', help='Batch evaluate all methods')
    p_batch.add_argument('--results_dir', type=str, required=True)
    p_batch.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    if args.command == 'epsilon_u':
        cmd_epsilon_u(args)
    elif args.command == 'single':
        cmd_single(args)
    elif args.command == 'batch':
        cmd_batch(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
