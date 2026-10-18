"""
完整实验编排脚本 — 论文所有实验的一键运行

实验矩阵:
  Table 1a: 主实验 (7 baselines + Ours) × 50 程序化模型 × 6 metrics
  Table 1b: ShapeNet 泛化实验 (7 baselines + Ours) × 25 真实模型 × 6 metrics
  Table 2:  Safety Transfer Bound 验证 × 50 models
  Table 3:  消融实验 (5 ablation variants + Ours) × 50 models × 6 metrics
  Table 4:  ε_u 敏感性分析
  Table 5:  PINO 精度验证
  Table 6:  收敛对比
  Figure 1-7: 收敛曲线, ALM演化, 路由分布, 帕累托, SF散点, ε_u曲线

输出结构:
  {output_dir}/
    data/              ← 程序化测试数据集
    data_shapenet/     ← ShapeNet 测试数据集
    results/           ← 逐模型结果
    tables/            ← LaTeX 表格 (table1a, table1b, table2-6)
    figures/           ← 图表 (JSON + PDF)

用法:
    # 完整实验
    python run_all_experiments.py \
        --pino_checkpoint path/to/pino.pt \
        --output_dir ./paper_experiments \
        --n_models 50 \
        --shapenet_dir /path/to/ShapeNetCore.v2

    # 快速测试
    python run_all_experiments.py \
        --pino_checkpoint path/to/pino.pt \
        --output_dir ./test_run \
        --n_models 3 --no_fem --n_steps 50
"""

from __future__ import annotations

import sys
import argparse
import json
import time
import logging
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "structural"))

from structural.configs import StructShadowConfig
from structural.optimization.safety_bound import SafetyTransferBound
from structural.experiments.prepare_data import prepare_experiment_dataset
from structural.experiments.prepare_shapenet import prepare_shapenet_dataset
from structural.experiments.baselines import BASELINES, APPENDIX_BASELINES
from structural.experiments.ablation import ABLATIONS
from structural.experiments.evaluate_metrics import (
    evaluate_single, aggregate_results, format_results_table,
    format_results_markdown, MethodResult, ExperimentSummary,
    extract_load_cases, load_target_silhouettes,
)
from structural.experiments.bound_verification import (
    run_bound_verification_dataset,
    run_epsilon_sensitivity,
    format_bound_table_latex,
    format_epsilon_table_latex,
)
from structural.experiments.convergence_analysis import (
    run_convergence_analysis,
    compute_routing_statistics,
    format_convergence_table_latex,
)
from structural.experiments.pino_validation import (
    validate_pino_dataset,
    validate_pino_by_density_bin,
    format_pino_table_latex,
)
from structural.experiments.scalability import (
    run_scalability_experiment,
    format_scalability_table_latex,
    format_scalability_markdown,
)

logger = logging.getLogger(__name__)


def load_pino_for_eval(checkpoint_path: str, config: StructShadowConfig, device: torch.device):
    """加载 PINO 用于实验评估"""
    from structural.models.pino import PINO3d

    pino = PINO3d(
        in_channels=config.pino.in_channels,
        out_channels=config.pino.out_channels,
        modes=config.pino.modes,
        width=config.pino.width,
        n_layers=config.pino.n_layers,
        padding=config.pino.padding,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        pino.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict):
        pino.load_state_dict(ckpt)
    else:
        pino.load_state_dict(ckpt)

    pino.eval()
    for p in pino.parameters():
        p.requires_grad = False

    epsilon_u = None
    if isinstance(ckpt, dict) and 'val_displacement_error' in ckpt:
        epsilon_u = ckpt['val_displacement_error']

    return pino, epsilon_u


def run_structshadow_full(
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    n_steps: int = 200,
    lr: float = 1e-3,
    **kwargs,
) -> np.ndarray:
    """完整 StructShadow 方法 (ALM + Router + RepairCNN + Decoupling)

    作为 "Ours" 的实验入口。
    等价于 ablation._run_optimization_loop 所有开关全开。
    """
    from structural.experiments.ablation import _run_optimization_loop
    from structural.optimization.alm_optimizer import ALMConfig

    if device is None:
        device = next(pino_model.parameters()).device

    alm_config = ALMConfig(
        safety_factor_target=1.5,
        volume_budget_ratio=1.3,
        overhang_limit=0.15,
    )

    return _run_optimization_loop(
        density_original, force, pino_model, yield_stress, device,
        n_steps=n_steps, lr=lr,
        use_alm=True,
        use_router=True,
        use_repair=True,
        use_decoupling=True,
        use_conservative_calibration=True,
        stress_inflation_factor=config.surrogate.stress_inflation_factor,
        alm_config=alm_config,
    )


def run_method_on_dataset(
    method_name: str,
    method_fn,
    data_dir: str,
    pino_model: torch.nn.Module,
    config: StructShadowConfig,
    device: torch.device,
    max_models: int | None = None,
    n_steps: int = 200,
    run_fem: bool = False,
) -> list:
    """在数据集上运行一个方法, 收集所有模型的评估结果

    Args:
        method_name: 方法名称
        method_fn: 方法函数 (density, force, fixed_mask, ...) -> density
        data_dir: 数据集目录
        pino_model: PINO 模型
        config: 项目配置
        device: GPU
        max_models: 最多评估的模型数
        n_steps: 优化步数
        run_fem: 是否用 FEM 验证

    Returns:
        list of MethodResult
    """
    data_path = Path(data_dir)
    meta_file = data_path / "dataset_meta.json"

    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        model_ids = [m['model_id'] for m in meta['models']]
    else:
        model_dirs = sorted(data_path.glob("model_*"))
        model_ids = [d.name for d in model_dirs]

    if max_models is not None:
        model_ids = model_ids[:max_models]

    results = []
    yield_stress = config.material.yield_stress

    for i, model_id in enumerate(model_ids):
        model_dir = data_path / model_id

        voxel_file = model_dir / "voxels.npy"
        loads_file = model_dir / "loads.npz"

        if not voxel_file.exists() or not loads_file.exists():
            logger.warning(f"Skipping {model_id}: missing files")
            continue

        density_orig = np.load(voxel_file).astype(np.float32)
        loads = np.load(loads_file)
        load_cases = extract_load_cases(loads, config.evaluation.multi_load_aggregation)
        if not load_cases:
            logger.warning(f"Skipping {model_id}: no load cases")
            continue
        force = load_cases[0]['force']
        fixed_mask = load_cases[0]['fixed_mask']
        target_silhouettes = load_target_silhouettes(model_dir / "silhouettes")

        t0 = time.time()
        try:
            density_result = method_fn(
                density_original=density_orig,
                force=force,
                fixed_mask=fixed_mask,
                pino_model=pino_model,
                yield_stress=yield_stress,
                device=device,
                n_steps=n_steps,
            )
        except Exception as e:
            logger.warning(f"{method_name} failed on {model_id}: {e}")
            continue
        elapsed = time.time() - t0

        # 评估
        result = evaluate_single(
            density=density_result,
            density_original=density_orig,
            force=force,
            fixed_mask=fixed_mask,
            model_id=model_id,
            method_name=method_name,
            pino_model=pino_model,
            yield_stress=yield_stress,
            sf_target=1.5,
            volume_budget=1.3,
            run_fem=run_fem,
            device=device,
            target_silhouettes=target_silhouettes,
            load_cases=load_cases,
            stress_inflation_factor=config.surrogate.stress_inflation_factor,
            evaluation_config=config.evaluation,
        )
        result.elapsed_seconds = elapsed
        results.append(result)

        if (i + 1) % 10 == 0:
            sf_report = result.sf_min_fem if result.sf_min_fem > 0 else result.sf_min_surrogate
            print(f"  [{method_name}] {i+1}/{len(model_ids)}: "
                  f"MV-IoU={result.mv_iou:.3f}, SF={sf_report:.2f}, "
                  f"t={elapsed:.1f}s")

    return results


def run_all(args):
    """运行全部实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_dir = output_dir / "pilot"
    verified_dir = output_dir / "verified"
    paper_final_dir = output_dir / "paper_final"
    for p in [pilot_dir, verified_dir, paper_final_dir]:
        p.mkdir(exist_ok=True)

    config = StructShadowConfig.from_yaml(args.config)
    yield_stress = config.material.yield_stress

    print("=" * 80)
    print("StructShadow Full Experiment Suite")
    print(f"  Device: {device}")
    print(f"  Resolution: {config.resolution}³")
    print(f"  Models: {args.n_models}")
    print(f"  Steps per method: {args.n_steps}")
    print(f"  FEM verification: {not args.no_fem}")
    print(f"  Output: {output_dir}")
    print("=" * 80)

    t_total_start = time.time()

    # ================================================================
    # Phase 1: 数据准备
    # ================================================================
    data_dir = str(output_dir / "data")
    if args.data_dir:
        data_dir = args.data_dir
        print(f"\n[Phase 1] Using existing data: {data_dir}")
    else:
        print(f"\n[Phase 1] Generating test dataset ({args.n_models} models)...")
        prepare_experiment_dataset(
            output_dir=data_dir,
            n_models=args.n_models,
            resolution=config.resolution,
            seed=args.seed,
            run_fem=not args.no_fem,
        )

    # ================================================================
    # Phase 2: 加载 PINO
    # ================================================================
    print(f"\n[Phase 2] Loading PINO from {args.pino_checkpoint}...")
    pino_model, pino_epsilon_u = load_pino_for_eval(
        args.pino_checkpoint, config, device
    )
    if pino_epsilon_u is not None:
        print(f"  PINO ε_u = {pino_epsilon_u:.4f}")
    else:
        pino_epsilon_u = 0.10
        print(f"  PINO ε_u not in checkpoint, using conservative estimate: {pino_epsilon_u}")

    # ================================================================
    # Phase 3: 主实验 (Table 1) — 5 Baselines + Ours
    # ================================================================
    results_dir = verified_dir / "results"
    results_dir.mkdir(exist_ok=True)

    all_summaries = []
    all_method_results = {}

    # 5 基线方法
    if not args.skip_baselines:
        print(f"\n[Phase 3] Running baseline experiments (Table 1)...")
        for method_name, method_fn in BASELINES.items():
            print(f"\n  --- {method_name} ---")
            results = run_method_on_dataset(
                method_name=method_name,
                method_fn=method_fn,
                data_dir=data_dir,
                pino_model=pino_model,
                config=config,
                device=device,
                max_models=args.n_models,
                n_steps=args.n_steps,
                run_fem=not args.no_fem,
            )
            summary = aggregate_results(results, method_name)
            all_summaries.append(summary)
            all_method_results[method_name] = results

            print(f"  {method_name}: IoU={summary.iou_mean:.3f}±{summary.iou_std:.3f}, "
                  f"SF={summary.sf_min_mean:.2f}±{summary.sf_min_std:.2f}, "
                  f"CSR={summary.constraint_satisfaction_rate*100:.1f}%")
    else:
        print(f"\n[Phase 3] Skipping baselines (--skip_baselines)")

    # Ours (完整方法) — 总是运行
    print(f"\n  --- Ours (StructShadow) ---")
    ours_results = run_method_on_dataset(
        method_name="Ours",
        method_fn=run_structshadow_full,
        data_dir=data_dir,
        pino_model=pino_model,
        config=config,
        device=device,
        max_models=args.n_models,
        n_steps=args.n_steps,
        run_fem=not args.no_fem,
    )
    ours_summary = aggregate_results(ours_results, "Ours")
    all_summaries.append(ours_summary)
    all_method_results["Ours"] = ours_results

    print(f"  Ours: IoU={ours_summary.iou_mean:.3f}±{ours_summary.iou_std:.3f}, "
          f"SF={ours_summary.sf_min_mean:.2f}±{ours_summary.sf_min_std:.2f}, "
          f"CSR={ours_summary.constraint_satisfaction_rate*100:.1f}%")

    # 生成 Table 1
    tables_dir = paper_final_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    table1_latex = format_results_table(all_summaries)
    with open(tables_dir / "table1_main.tex", 'w') as f:
        f.write(table1_latex)

    table1_md = format_results_markdown(all_summaries)
    with open(tables_dir / "table1_main.md", 'w') as f:
        f.write(table1_md)
    print(f"\n[Table 1a] Main results (procedural):\n{table1_md}")

    # ================================================================
    # Phase 3b: ShapeNet 泛化实验 (Table 1b)
    # ================================================================
    shapenet_data_dir = None
    table1b_md = ""

    has_shapenet = (
        not args.skip_shapenet
        and (args.shapenet_dir is not None or args.shapenet_mesh_list is not None)
    )

    if has_shapenet:
        print(f"\n[Phase 3b] ShapeNet generalization experiment (Table 1b)...")
        shapenet_data_dir = str(verified_dir / "data_shapenet")

        # 准备 ShapeNet 数据
        cats = [c.strip() for c in args.shapenet_categories.split(',')]
        prepare_shapenet_dataset(
            output_dir=shapenet_data_dir,
            shapenet_dir=args.shapenet_dir,
            mesh_list=args.shapenet_mesh_list,
            categories=cats,
            per_category=args.shapenet_per_category,
            resolution=config.resolution,
            seed=args.seed,
            run_fem=not args.no_fem,
        )

        # 在 ShapeNet 数据集上运行所有方法 (baselines + ours)
        shapenet_summaries = []
        shapenet_method_results = {}

        methods_to_run = {}
        if not args.skip_baselines:
            methods_to_run.update(BASELINES)
        methods_to_run['Ours'] = run_structshadow_full

        for method_name, method_fn in methods_to_run.items():
            print(f"\n  --- [ShapeNet] {method_name} ---")
            sn_results = run_method_on_dataset(
                method_name=method_name,
                method_fn=method_fn,
                data_dir=shapenet_data_dir,
                pino_model=pino_model,
                config=config,
                device=device,
                max_models=None,  # 运行所有 ShapeNet 模型
                n_steps=args.n_steps,
                run_fem=not args.no_fem,
            )
            sn_summary = aggregate_results(sn_results, method_name)
            shapenet_summaries.append(sn_summary)
            shapenet_method_results[method_name] = sn_results

            print(f"  [SN] {method_name}: IoU={sn_summary.iou_mean:.3f}, "
                  f"SF={sn_summary.sf_min_mean:.2f}, "
                  f"CSR={sn_summary.constraint_satisfaction_rate*100:.1f}%")

        # 生成 Table 1b
        table1b_latex = format_results_table(shapenet_summaries)
        table1b_latex = table1b_latex.replace(
            "Quantitative comparison of all methods.",
            "Generalization to ShapeNet real-world models."
        ).replace("tab:main_results", "tab:shapenet_results")

        with open(tables_dir / "table1b_shapenet.tex", 'w') as f:
            f.write(table1b_latex)

        table1b_md = format_results_markdown(shapenet_summaries)
        with open(tables_dir / "table1b_shapenet.md", 'w') as f:
            f.write(table1b_md)
        print(f"\n[Table 1b] ShapeNet results:\n{table1b_md}")

        # 保存 ShapeNet 逐模型结果
        for method_name, results in shapenet_method_results.items():
            method_data = []
            for r in results:
                method_data.append({
                    'model_id': r.model_id,
                    'iou_3d': r.iou_3d,
                    'mv_iou': r.mv_iou,
                    'mv_ssim': r.mv_ssim,
                    'sf_min_pino': r.sf_min_pino,
                    'sf_min_surrogate': r.sf_min_surrogate,
                    'sf_min_fem': r.sf_min_fem,
                    'fem_safe': r.fem_safe,
                    'volume_ratio': r.volume_ratio,
                    'pino_sf_error': r.pino_sf_error,
                    'constraint_satisfied': r.constraint_satisfied,
                    'is_printable': r.is_printable,
                    'elapsed_seconds': r.elapsed_seconds,
                })
            safe_name = method_name.replace(' ', '_').replace('/', '_')
            with open(results_dir / f"results_shapenet_{safe_name}.json", 'w') as f:
                json.dump(method_data, f, indent=2)

        # 合并到 figure data
        for method_name, results in shapenet_method_results.items():
            sn_key = f"{method_name} (ShapeNet)"
            all_method_results[sn_key] = results
    else:
        if not args.skip_shapenet:
            print(f"\n[Phase 3b] Skipped (no --shapenet_dir or --shapenet_mesh_list)")
        else:
            print(f"\n[Phase 3b] Skipped (--skip_shapenet)")

    # ================================================================
    # Phase 4: 消融实验 (Table 3)
    # ================================================================
    ablation_summaries = [ours_summary]  # Ours 作为 full method
    table3_md = ""

    if not args.skip_ablations:
        print(f"\n[Phase 4] Running ablation experiments (Table 3)...")

        for abl_name, abl_fn in ABLATIONS.items():
            print(f"\n  --- {abl_name} ---")
            abl_results = run_method_on_dataset(
                method_name=abl_name,
                method_fn=abl_fn,
                data_dir=data_dir,
                pino_model=pino_model,
                config=config,
                device=device,
                max_models=args.n_models,
                n_steps=args.n_steps,
                run_fem=not args.no_fem,
            )
            abl_summary = aggregate_results(abl_results, abl_name)
            ablation_summaries.append(abl_summary)
            all_method_results[abl_name] = abl_results

            print(f"  {abl_name}: IoU={abl_summary.iou_mean:.3f}, "
                  f"SF={abl_summary.sf_min_mean:.2f}, "
                  f"CSR={abl_summary.constraint_satisfaction_rate*100:.1f}%")

        # 生成 Table 3
        table3_latex = format_results_table(ablation_summaries)
        table3_latex = table3_latex.replace(
            "Quantitative comparison of all methods.",
            "Ablation study results."
        ).replace("tab:main_results", "tab:ablation")

        with open(tables_dir / "table3_ablation.tex", 'w') as f:
            f.write(table3_latex)

        table3_md = format_results_markdown(ablation_summaries)
        with open(tables_dir / "table3_ablation.md", 'w') as f:
            f.write(table3_md)
        print(f"\n[Table 3] Ablation results:\n{table3_md}")
    else:
        print(f"\n[Phase 4] Skipping ablations (--skip_ablations)")

    # ================================================================
    # Phase 5: Safety Transfer Bound 验证 (Table 2)
    # ================================================================
    print(f"\n[Phase 5] Safety Transfer Bound verification (Table 2)...")
    stb = SafetyTransferBound(
        youngs_modulus=config.material.youngs_modulus,
        poisson_ratio=config.material.poisson_ratio,
        resolution=config.resolution,
        yield_stress=yield_stress,
    )

    bound_stats = {}
    if not args.no_fem:
        bound_stats = run_bound_verification_dataset(
            stb=stb,
            pino_model=pino_model,
            data_dir=data_dir,
            device=device,
            yield_stress=yield_stress,
            max_models=args.n_models,
        )

        table2_latex = format_bound_table_latex(bound_stats)
        with open(tables_dir / "table2_bound.tex", 'w') as f:
            f.write(table2_latex)

        with open(results_dir / "bound_verification.json", 'w') as f:
            serializable = {k: v for k, v in bound_stats.items() if k != 'per_model'}
            serializable['n_valid'] = bound_stats.get('n_valid', 0)
            json.dump(serializable, f, indent=2)
    else:
        print("  Skipped (--no_fem)")

    # ================================================================
    # Phase 6: ε_u 敏感性分析 (Table 4)
    # ================================================================
    print(f"\n[Phase 6] ε_u sensitivity analysis (Table 4)...")

    # 用第一个测试模型做敏感性分析
    data_path = Path(data_dir)
    first_model = sorted(data_path.glob("model_*"))[0] if list(data_path.glob("model_*")) else None

    eps_results = []
    if first_model is not None:
        density_sample = np.load(first_model / "voxels.npy")
        loads_sample = np.load(first_model / "loads.npz")
        force_sample = loads_sample['force_standing']

        eps_results = run_epsilon_sensitivity(
            stb=stb,
            pino_model=pino_model,
            density=density_sample,
            force=force_sample,
            device=device,
        )

        table4_latex = format_epsilon_table_latex(eps_results)
        with open(tables_dir / "table4_epsilon.tex", 'w') as f:
            f.write(table4_latex)

    # ================================================================
    # Phase 7: PINO 精度验证 (Table 5)
    # ================================================================
    pino_stats = {}
    if not args.no_fem:
        print(f"\n[Phase 7] PINO accuracy validation (Table 5)...")
        pino_stats = validate_pino_dataset(
            pino_model=pino_model,
            data_dir=data_dir,
            device=device,
            yield_stress=yield_stress,
            max_models=args.n_models,
        )

        table5_latex = format_pino_table_latex(pino_stats)
        with open(tables_dir / "table5_pino.tex", 'w') as f:
            f.write(table5_latex)

        # 按密度分箱
        binned = validate_pino_by_density_bin(
            pino_model, data_dir, device, yield_stress,
            n_bins=5, max_models=args.n_models,
        )
        with open(figures_dir / "fig7_pino_by_density.json", 'w') as f:
            json.dump(binned, f, indent=2)

        with open(results_dir / "pino_validation.json", 'w') as f:
            serializable = {k: v for k, v in pino_stats.items() if k != 'per_model'}
            json.dump(serializable, f, indent=2)
    else:
        print(f"\n[Phase 7] Skipping PINO validation (--no_fem)")

    # ================================================================
    # Phase 8: 收敛分析 (Figure 2-4)
    # ================================================================
    print(f"\n[Phase 8] Convergence analysis...")
    data_path = Path(data_dir)
    first_model = sorted(data_path.glob("model_*"))[0] if list(data_path.glob("model_*")) else None

    if first_model is not None:
        density_conv = np.load(first_model / "voxels.npy").astype(np.float32)
        loads_conv = np.load(first_model / "loads.npz")
        force_conv = loads_conv['force_standing']

        conv_data = run_convergence_analysis(
            density_original=density_conv,
            force=force_conv,
            pino_model=pino_model,
            yield_stress=yield_stress,
            device=device,
            n_steps=min(args.n_steps, 300),
            lr=1e-3,
        )

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        with open(figures_dir / "convergence_data.json", 'w') as f:
            json.dump(conv_data, f, indent=2)

        # 路由统计
        routing_stats = compute_routing_statistics(conv_data)
        with open(figures_dir / "routing_stats.json", 'w') as f:
            json.dump(routing_stats, f, indent=2)

        # 收敛对比表 (Table 6)
        table6_latex = format_convergence_table_latex(
            conv_data.get('alm', {}),
            conv_data.get('weighted_sum', {}),
        )
        with open(tables_dir / "table6_convergence.tex", 'w') as f:
            f.write(table6_latex)

        print(f"  Routing distribution: {routing_stats.get('fractions', {})}")

    # ================================================================
    # Phase 9: 分辨率可扩展性实验 (Table 7, Figure 8)
    # ================================================================
    if not args.skip_scalability:
        print(f"\n[Phase 9] Resolution scalability experiment (Table 7)...")
        data_path = Path(data_dir)
        model_dirs = sorted(data_path.glob("model_*"))[:min(5, args.n_models)]

        models_data = []
        for md in model_dirs:
            d = np.load(md / "voxels.npy").astype(np.float32)
            l = np.load(md / "loads.npz")
            models_data.append((d, l['force_standing'], l['fixed_mask_standing']))

        if models_data:
            pino_ckpts = {}
            # 默认只有 64³ 的 checkpoint
            if args.pino_checkpoint:
                pino_ckpts[64] = args.pino_checkpoint

            scalability_results = run_scalability_experiment(
                density_original_64=models_data[0][0],
                force_64=models_data[0][1],
                fixed_mask_64=models_data[0][2],
                pino_checkpoints=pino_ckpts,
                device=device,
                yield_stress=yield_stress,
                resolutions=[32, 64, 128],
                n_models=min(5, len(models_data)),
                models_data=models_data,
            )

            table7_latex = format_scalability_table_latex(scalability_results)
            with open(tables_dir / "table7_scalability.tex", 'w') as f:
                f.write(table7_latex)

            table7_md = format_scalability_markdown(scalability_results)
            with open(tables_dir / "table7_scalability.md", 'w') as f:
                f.write(table7_md)
            print(f"\n[Table 7] Scalability:\n{table7_md}")

            # 保存 figure 数据
            figures_dir = paper_final_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            serializable = []
            for r in scalability_results:
                serializable.append({k: v for k, v in r.items() if k != 'per_model'})
            with open(figures_dir / "fig8_scalability.json", 'w') as f:
                json.dump(serializable, f, indent=2)
    else:
        print(f"\n[Phase 9] Skipping scalability (--skip_scalability)")

    # ================================================================
    # Phase 10: 生成图表数据
    # ================================================================
    print(f"\n[Phase 10] Generating figure data...")
    figures_dir = paper_final_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Figure 1 data: 各方法的 IoU vs SF 帕累托前沿
    pareto_data = {}
    for method_name, results in all_method_results.items():
        pareto_data[method_name] = {
            'ious': [r.mv_iou for r in results],
            'sfs': [r.sf_min_fem if r.sf_min_fem > 0 else r.sf_min_surrogate for r in results],
        }
    with open(figures_dir / "fig1_pareto.json", 'w') as f:
        json.dump(pareto_data, f, indent=2)

    # Figure 2 data: SF_PINO vs SF_FEM 散点 (仅 FEM 验证过的)
    if not args.no_fem:
        scatter_data = []
        for results in all_method_results.values():
            for r in results:
                if r.sf_min_fem > 0:
                    scatter_data.append({
                        'method': r.method_name,
                        'sf_pino': r.sf_min_surrogate,
                        'sf_fem': r.sf_min_fem,
                    })
        with open(figures_dir / "fig2_sf_scatter.json", 'w') as f:
            json.dump(scatter_data, f, indent=2)

    # Figure 3 data: ε_u 敏感性曲线
    if eps_results:
        with open(figures_dir / "fig3_epsilon.json", 'w') as f:
            json.dump(eps_results, f, indent=2)

    # ================================================================
    # Phase 11: 保存完整实验结果
    # ================================================================
    print(f"\n[Phase 11] Saving all results...")

    # 保存每个方法的逐模型结果
    for method_name, results in all_method_results.items():
        method_data = []
        for r in results:
            method_data.append({
                'model_id': r.model_id,
                'iou_3d': r.iou_3d,
                'mv_iou': r.mv_iou,
                'mv_bce': r.mv_bce,
                'mv_ssim': r.mv_ssim,
                'sf_min_pino': r.sf_min_pino,
                'sf_min_surrogate': r.sf_min_surrogate,
                'sf_min_fem': r.sf_min_fem,
                'fem_safe': r.fem_safe,
                'all_constraints_satisfied': r.all_constraints_satisfied,
                'displacement_max_surrogate': r.displacement_max_surrogate,
                'displacement_max_fem': r.displacement_max_fem,
                'volume_ratio': r.volume_ratio,
                'pino_sf_error': r.pino_sf_error,
                'constraint_satisfied': r.constraint_satisfied,
                'is_printable': r.is_printable,
                'overhang_ratio': r.overhang_ratio,
                'failure_reason': r.failure_reason,
                'per_load': r.per_load,
                'elapsed_seconds': r.elapsed_seconds,
            })
        safe_name = method_name.replace(' ', '_').replace('/', '_')
        with open(results_dir / f"results_{safe_name}.json", 'w') as f:
            json.dump(method_data, f, indent=2)

    # 总摘要
    elapsed_total = time.time() - t_total_start
    experiment_summary = {
        'n_models': args.n_models,
        'n_steps': args.n_steps,
        'resolution': config.resolution,
        'run_fem': not args.no_fem,
        'layout': {
            'pilot_dir': str(pilot_dir),
            'verified_dir': str(verified_dir),
            'paper_final_dir': str(paper_final_dir),
        },
        'total_time_seconds': elapsed_total,
        'total_time_minutes': elapsed_total / 60,
        'methods_evaluated': list(all_method_results.keys()),
        'main_results': {
            s.method_name: {
                'mv_iou': f"{s.iou_mean:.3f}±{s.iou_std:.3f}",
                'fem_sf': f"{(s.sf_fem_mean if s.sf_fem_mean > 0 else s.sf_min_mean):.2f}±{(s.sf_fem_std if s.sf_fem_mean > 0 else s.sf_min_std):.2f}",
                'fem_safe_rate': f"{s.fem_safe_rate*100:.1f}%",
                'vol_ratio': f"{s.volume_ratio_mean:.2f}±{s.volume_ratio_std:.2f}",
            }
            for s in all_summaries
        },
    }
    if has_shapenet and shapenet_summaries:
        experiment_summary['shapenet_results'] = {
            s.method_name: {
                'iou': f"{s.iou_mean:.3f}±{s.iou_std:.3f}",
                'sf_min': f"{s.sf_min_mean:.2f}±{s.sf_min_std:.2f}",
                'csr': f"{s.constraint_satisfaction_rate*100:.1f}%",
            }
            for s in shapenet_summaries
        }

    if bound_stats:
        experiment_summary['bound_verification'] = {
            'validity_rate': bound_stats.get('bound_validity_rate', 0),
            'tightness': bound_stats.get('tightness_mean', 0),
            'eps_u_mean': bound_stats.get('eps_u_mean', 0),
        }

    with open(paper_final_dir / "experiment_summary.json", 'w') as f:
        json.dump(experiment_summary, f, indent=2)

    # 保存一份轻量 pilot 摘要，便于小规模 smoke / pilot 跟踪
    with open(pilot_dir / "pilot_summary.json", 'w') as f:
        json.dump({
            'n_models': args.n_models,
            'methods': list(all_method_results.keys()),
            'main_results': experiment_summary['main_results'],
        }, f, indent=2)

    # ================================================================
    # 最终输出
    # ================================================================
    print(f"\n{'=' * 80}")
    print(f"All experiments complete!")
    print(f"  Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print(f"  Models evaluated: {args.n_models}")
    print(f"  Methods compared: {len(all_method_results)}")
    print(f"{'=' * 80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  tables/  — LaTeX tables (table1-4)")
    print(f"  figures/ — Figure data (JSON)")
    print(f"  results/ — Per-model results")
    print(f"\n[Table 1a] Main comparison (procedural):")
    print(table1_md)
    if table1b_md:
        print(f"\n[Table 1b] ShapeNet generalization:")
        print(table1b_md)
    if table3_md:
        print(f"\n[Table 3] Ablation:")
        print(table3_md)

    # ================================================================
    # Phase 12: 生成 matplotlib 图表
    # ================================================================
    try:
        from structural.experiments.plot_results import generate_all_figures
        print(f"\n[Phase 12] Generating matplotlib figures...")
        generate_all_figures(str(output_dir))
    except ImportError:
        print(f"\n[Phase 12] Skipped (matplotlib not available)")
    except Exception as e:
        print(f"\n[Phase 12] Figure generation failed: {e}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    parser = argparse.ArgumentParser(
        description='StructShadow Full Paper Experiment Suite'
    )

    # 必需
    parser.add_argument('--pino_checkpoint', type=str, required=True,
                        help='Path to pre-trained PINO checkpoint')

    # 实验设置
    parser.add_argument('--output_dir', type=str, default='./paper_experiments',
                        help='Output directory for all results')
    parser.add_argument('--config', type=str, default=None,
                        help='Config YAML path')
    parser.add_argument('--n_models', type=int, default=50,
                        help='Number of test models')
    parser.add_argument('--n_steps', type=int, default=200,
                        help='Optimization steps per method')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Voxel resolution')
    parser.add_argument('--seed', type=int, default=42)

    # 数据
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Pre-existing test dataset directory')

    # ShapeNet
    parser.add_argument('--shapenet_dir', type=str, default=None,
                        help='Path to ShapeNetCore.v2 directory')
    parser.add_argument('--shapenet_mesh_list', type=str, default=None,
                        help='Text file with mesh paths (alternative to shapenet_dir)')
    parser.add_argument('--shapenet_categories', type=str,
                        default='chair,table,lamp,airplane,vessel',
                        help='Comma-separated ShapeNet categories')
    parser.add_argument('--shapenet_per_category', type=int, default=5,
                        help='Models per ShapeNet category')

    # 开关
    parser.add_argument('--no_fem', action='store_true',
                        help='Skip FEM verification (faster, no SGLDBench needed)')
    parser.add_argument('--skip_baselines', action='store_true',
                        help='Skip baseline experiments')
    parser.add_argument('--skip_ablations', action='store_true',
                        help='Skip ablation experiments')
    parser.add_argument('--skip_shapenet', action='store_true',
                        help='Skip ShapeNet evaluation')
    parser.add_argument('--skip_scalability', action='store_true',
                        help='Skip resolution scalability experiment')

    args = parser.parse_args()
    run_all(args)
