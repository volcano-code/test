"""
GPU 空闲后一键运行所有 baseline + StructShadow 主方法

运行顺序 (GPU 可用时):
  1. WeightedSum       (无 ALM 消融)
  2. NTOPo             (神经拓扑优化)
  3. SIMP              (经典拓扑优化，用 PINO 加速)
  4. Stava             (启发式后处理)
  5. StructShadow      (完整方法 Stage B + C)

每个方法对 test_datasets 中的所有模型运行，并计算:
  IoU_3D, SF_min_PINO, SF_min_FEM (可选), Vol_Ratio, CSR, Time

用法:
    # 检查 GPU 是否空闲后运行
    python run_all_baselines.py \
        --pino_checkpoint ./checkpoints/pino/best_model.pt \
        --dataset_dirs ./test_datasets/modelnet40 ./test_datasets/thingi10k \
        --methods weighted_sum ntopo simp stava \
        --n_steps 200 \
        --output_dir ./results/baselines

    # 仅跑 SIMP (GPU 空闲后)
    python run_all_baselines.py --methods simp --n_steps 200
"""
import sys
import json
import time
import argparse
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from structural.analysis.stress_eval import get_yield_stress_for_model
from structural.models.load_model import forward_surrogate_physical, load_surrogate

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "structural"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

YIELD_STRESS = 50e6
ALL_METHODS = ['shadowart_only', 'stava', 'weighted_sum', 'simp', 'ntopo']
BASELINE_SCHEMA_VERSION = "baseline_results_v2"


def load_pino(checkpoint_path: str, device: torch.device):
    model, _ = load_surrogate(checkpoint_path, device)
    logger.info(
        "PINO loaded: is_v3=%s has_scale_head=%s",
        getattr(model, '_is_v3', False),
        getattr(model, '_has_scale_head', False),
    )
    return model


def build_run_manifest(args, device: torch.device) -> dict:
    return {
        "output_schema_version": BASELINE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pino_checkpoint": str(args.pino_checkpoint),
        "device": str(device),
        "dataset_dirs": [str(x) for x in args.dataset_dirs],
        "methods": list(args.methods),
        "n_steps": int(args.n_steps),
        "yield_stress_physical": float(YIELD_STRESS),
        "constraint_protocol": {
            "sf_threshold": 1.5,
            "vol_ratio_max": 1.3,
            "iou_metric": "iou_3d",
            "structural_metric": "sf_min_pino",
        },
        "notes": [
            "Rows in this file are surrogate-backed unless sf_fem is explicitly attached elsewhere.",
            "Paper-facing ranking should prefer FEM-backed outputs when available.",
        ],
    }


def failure_reason_from_metrics(metrics: dict) -> str:
    if not metrics:
        return "method_failed"
    if not metrics.get('constraint_satisfied', False):
        if float(metrics.get('sf_min_pino', 0.0)) < 1.5:
            return "sf_pino_below_threshold"
        if float(metrics.get('vol_ratio', 999.0)) > 1.3:
            return "volume_budget_exceeded"
        return "constraint_unsatisfied"
    return ""


def dataset_name_from_meta(meta: dict, data_path: Path) -> str:
    return str(meta.get('source', data_path.name))


def shard_dir_for(output_dir: Path, source: str, method: str) -> Path:
    return output_dir / "shards" / source / method


def shard_path_for(output_dir: Path, source: str, method: str, model_id: str) -> Path:
    return shard_dir_for(output_dir, source, method) / f"{model_id}.json"


def load_existing_dataset_results(output_dir: Path, source: str, methods: list[str]) -> dict:
    loaded = {m: [] for m in methods}
    for method in methods:
        method_dir = shard_dir_for(output_dir, source, method)
        if not method_dir.exists():
            continue
        for shard in sorted(method_dir.glob("*.json")):
            try:
                loaded[method].append(json.loads(shard.read_text()))
            except Exception as exc:
                logger.warning("Failed to read shard %s: %s", shard, exc)
    return loaded


def save_shard(output_dir: Path, source: str, method: str, model_id: str, payload: dict):
    shard_path = shard_path_for(output_dir, source, method, model_id)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text(json.dumps(payload, indent=2))


def save_dataset_results(output_dir: Path, source: str, results: dict):
    out_file = output_dir / f"results_{source}.json"
    out_file.write_text(json.dumps(results, indent=2))
    summary_rows = {}
    for method, items in results.items():
        valid = [r for r in items if isinstance(r, dict) and 'sf_min_pino' in r and 'iou_3d' in r]
        if not valid:
            summary_rows[method] = {
                "n_total": len(items),
                "n_valid": 0,
                "constraint_success_pct": 0.0,
            }
            continue
        summary_rows[method] = {
            "n_total": len(items),
            "n_valid": len(valid),
            "sf_mean": float(np.mean([r['sf_min_pino'] for r in valid])),
            "iou_mean": float(np.mean([r['iou_3d'] for r in valid])),
            "vol_mean": float(np.mean([r['vol_ratio'] for r in valid])),
            "constraint_success_pct": float(np.mean([bool(r['constraint_satisfied']) for r in valid]) * 100.0),
            "elapsed_mean_s": float(np.mean([r['elapsed_s'] for r in valid if r.get('elapsed_s') is not None])) if any(r.get('elapsed_s') is not None for r in valid) else None,
        }
    (output_dir / f"summary_{source}.json").write_text(json.dumps(summary_rows, indent=2))
    logger.info("Saved dataset shard summary: %s", out_file)


def save_combined_results(output_dir: Path, methods: list[str], dataset_dirs: list[str]):
    combined = {m: [] for m in methods}
    for dataset_dir in dataset_dirs:
        data_path = Path(dataset_dir)
        meta_file = data_path / "dataset_meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            continue
        source = dataset_name_from_meta(meta, data_path)
        dataset_file = output_dir / f"results_{source}.json"
        if not dataset_file.exists():
            continue
        try:
            data = json.loads(dataset_file.read_text())
        except Exception:
            continue
        for method in methods:
            combined[method].extend(data.get(method, []))
    (output_dir / "results_combined.json").write_text(json.dumps(combined, indent=2))
    summary_rows = {}
    for method, items in combined.items():
        valid = [r for r in items if isinstance(r, dict) and 'sf_min_pino' in r and 'iou_3d' in r]
        summary_rows[method] = {
            "n_total": len(items),
            "n_valid": len(valid),
            "constraint_success_pct": float(np.mean([bool(r.get('constraint_satisfied', False)) for r in valid]) * 100.0) if valid else 0.0,
            "paper_eligible_n": int(sum(1 for r in items if bool(r.get('paper_eligible', r.get('constraint_satisfied', False))))),
            "fem_backed_n": int(sum(1 for r in items if bool(r.get('fem_backed', False)))),
        }
    (output_dir / "results_summary.json").write_text(json.dumps(summary_rows, indent=2))
    return combined


def evaluate_metrics(density_opt: np.ndarray, density_orig: np.ndarray,
                     force: np.ndarray, pino_model, device: torch.device) -> dict:
    """计算 IoU_3D, SF_min_PINO, Vol_Ratio。"""
    yield_eval = get_yield_stress_for_model(pino_model, YIELD_STRESS)
    with torch.no_grad():
        rho = torch.from_numpy(density_opt).float().unsqueeze(0).unsqueeze(0).to(device)
        f = torch.from_numpy(force).float().unsqueeze(0).to(device)
        pino_fields = forward_surrogate_physical(pino_model, rho, f)
        vm = pino_fields['von_mises']
        solid = (rho > 0.5).float()
        vm_max = float((vm * solid).max())

    sf_pino = yield_eval / (vm_max + 1e-12)
    vol_ratio = float((density_opt > 0.5).sum()) / max(float((density_orig > 0.5).sum()), 1)

    pred_s = (density_opt > 0.5)
    orig_s = (density_orig > 0.5)
    iou_3d = float((pred_s & orig_s).sum()) / max(float((pred_s | orig_s).sum()), 1)

    return {
        'sf_min_pino': sf_pino,
        'vm_max': vm_max,
        'yield_stress_eval': yield_eval,
        'vol_ratio': vol_ratio,
        'iou_3d': iou_3d,
        'constraint_satisfied': sf_pino >= 1.5 and vol_ratio <= 1.3,
        'metric_source': 'sf_min_pino',
        'fem_backed': False,
    }


def run_method(method: str, density_orig: np.ndarray, force: np.ndarray,
               fixed_mask: np.ndarray, pino_model, device: torch.device,
               n_steps: int) -> tuple[np.ndarray, float]:
    """运行指定方法，返回 (optimized_density, elapsed_seconds)"""
    from structural.experiments.baselines import (
        run_baseline_shadowart_only,
        run_baseline_stava_postprocess,
        run_baseline_weighted_sum,
        run_baseline_simp,
        run_baseline_ntopo,
    )
    t0 = time.time()
    kwargs = dict(
        density_original=density_orig,
        force=force,
        fixed_mask=fixed_mask,
        pino_model=pino_model,
        yield_stress=YIELD_STRESS,
        device=device,
        n_steps=n_steps,
    )
    if method == 'shadowart_only':
        density_out = run_baseline_shadowart_only(density_orig)
    elif method == 'stava':
        density_out = run_baseline_stava_postprocess(**kwargs)
    elif method == 'weighted_sum':
        density_out = run_baseline_weighted_sum(**kwargs)
    elif method == 'simp':
        density_out = run_baseline_simp(**kwargs)
    elif method == 'ntopo':
        density_out = run_baseline_ntopo(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    return density_out, time.time() - t0


def run_all_on_dataset(dataset_dir: str, pino_model, device: torch.device,
                       methods: list, n_steps: int, output_dir: Path, resume: bool = True) -> dict:
    data_path = Path(dataset_dir)
    meta_file = data_path / "dataset_meta.json"
    if not meta_file.exists():
        logger.error(f"No dataset_meta.json in {dataset_dir}")
        return {}

    with open(meta_file) as f:
        meta = json.load(f)

    source = dataset_name_from_meta(meta, data_path)
    n_models = len(meta['models'])
    print(f"\n{'='*65}")
    print(f"Dataset: {source} ({n_models} models)")
    print(f"Methods: {methods}")
    print(f"{'='*65}")

    all_results = load_existing_dataset_results(output_dir, source, methods) if resume else {m: [] for m in methods}
    done_keys = {
        (method, str(item.get('model_id')))
        for method, items in all_results.items()
        for item in items
        if isinstance(item, dict)
    }

    for i, m_info in enumerate(meta['models']):
        model_id = m_info['model_id']
        voxel_file = data_path / model_id / "voxels.npy"
        loads_file = data_path / model_id / "loads.npz"
        if not voxel_file.exists():
            continue

        density_orig = np.load(voxel_file).astype(np.float32)
        if loads_file.exists():
            loads = np.load(loads_file)
            force = loads['force_standing'].astype(np.float32)
            fixed_mask = loads['fixed_mask_standing'].astype(np.float32)
        else:
            from structural.analysis.load_generator import LoadGenerator
            ld = LoadGenerator().standing_load(density_orig)
            force = ld['force'].astype(np.float32)
            fixed_mask = ld['fixed_mask'].astype(np.float32)

        print(f"\n  [{i+1:3d}/{n_models}] {model_id} ({m_info.get('category','?')})")

        for method in methods:
            if resume and (method, model_id) in done_keys:
                print(f"    {method:15s}: resume-skip")
                continue
            try:
                density_opt, elapsed = run_method(
                    method, density_orig, force, fixed_mask,
                    pino_model, device, n_steps
                )
                metrics = evaluate_metrics(density_opt, density_orig, force, pino_model, device)
                metrics.update({
                    'model_id': model_id,
                    'category': m_info.get('category', 'unknown'),
                    'source': source,
                    'elapsed_s': elapsed,
                    'method': method,
                    'n_steps': int(n_steps),
                    'pino_checkpoint': getattr(pino_model, '_checkpoint_path', None),
                    'output_schema_version': BASELINE_SCHEMA_VERSION,
                })
                metrics['failure_reason'] = failure_reason_from_metrics(metrics)
                all_results[method].append(metrics)
                done_keys.add((method, model_id))
                save_shard(output_dir, source, method, model_id, metrics)
                save_dataset_results(output_dir, source, all_results)
                status = "✓" if metrics['constraint_satisfied'] else "✗"
                print(f"    {method:15s}: SF={metrics['sf_min_pino']:5.2f} "
                      f"IoU={metrics['iou_3d']:.3f} Vol={metrics['vol_ratio']:.2f} "
                      f"{status} ({elapsed:.0f}s)")
            except Exception as e:
                logger.error(f"    {method} FAILED: {e}")
                all_results[method].append({
                    'model_id': model_id,
                    'category': m_info.get('category', 'unknown'),
                    'source': source,
                    'method': method,
                    'elapsed_s': None,
                    'n_steps': int(n_steps),
                    'pino_checkpoint': getattr(pino_model, '_checkpoint_path', None),
                    'constraint_satisfied': False,
                    'fem_backed': False,
                    'metric_source': 'sf_min_pino',
                    'failure_reason': f"exception:{type(e).__name__}",
                    'error': str(e),
                    'output_schema_version': BASELINE_SCHEMA_VERSION,
                })
                done_keys.add((method, model_id))
                save_shard(output_dir, source, method, model_id, all_results[method][-1])
                save_dataset_results(output_dir, source, all_results)

    # 保存结果
    save_dataset_results(output_dir, source, all_results)

    return all_results


def print_summary(all_results: dict):
    """打印 LaTeX 风格的汇总表"""
    print(f"\n{'='*75}")
    print(f"{'Method':<18} {'SF_min':>8} {'±':>5} {'IoU':>7} {'±':>5} "
          f"{'Vol':>6} {'CSR%':>6} {'Time':>7}")
    print(f"{'-'*75}")
    for method, results in all_results.items():
        if not results:
            continue
        valid = [r for r in results if 'sf_min_pino' in r and 'iou_3d' in r and 'vol_ratio' in r]
        failed = len(results) - len(valid)
        if not valid:
            print(f"  {method:<16} {'---':>8} {'---':>5} {'---':>7} {'---':>5} {'---':>6} {'0.0':>6} {'---':>7}  fail={failed}")
            continue
        sf = [r['sf_min_pino'] for r in valid]
        iou = [r['iou_3d'] for r in valid]
        vol = [r['vol_ratio'] for r in valid]
        csr = np.mean([r['constraint_satisfied'] for r in valid]) * 100
        t = [r['elapsed_s'] for r in valid if r.get('elapsed_s') is not None]
        print(f"  {method:<16} {np.mean(sf):8.3f} {np.std(sf):5.3f} "
              f"{np.mean(iou):7.3f} {np.std(iou):5.3f} "
              f"{np.mean(vol):6.2f} {csr:6.1f} {np.mean(t) if t else float('nan'):7.1f}s  fail={failed}")
    print(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pino_checkpoint', default='./checkpoints/pino/best_model.pt')
    parser.add_argument('--dataset_dirs', nargs='+',
                        default=['./test_datasets/modelnet40', './test_datasets/thingi10k'])
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS,
                        choices=ALL_METHODS)
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--output_dir', default='./results/baselines')
    parser.add_argument('--device', default=None)
    parser.add_argument('--resume', type=int, default=1,
                        help='Resume from per-sample shards if available (1/0).')
    args = parser.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    # GPU 内存检查
    if device.type == 'cuda':
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        if free_mem < 4.0:
            print(f"⚠ GPU only {free_mem:.1f} GB free. Consider --device cpu or wait for PINO training.")
            if free_mem < 2.0:
                print("  Falling back to CPU automatically.")
                device = torch.device('cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_manifest.json", 'w') as f:
        json.dump(build_run_manifest(args, device), f, indent=2)

    pino_model = load_pino(args.pino_checkpoint, device)
    pino_model._checkpoint_path = str(args.pino_checkpoint)

    combined = {m: [] for m in args.methods}
    for dataset_dir in args.dataset_dirs:
        results = run_all_on_dataset(
            dataset_dir, pino_model, device,
            args.methods, args.n_steps, output_dir, resume=bool(args.resume)
        )
        for m, rs in results.items():
            combined[m].extend(rs)

    # 合并结果
    combined = save_combined_results(output_dir, args.methods, args.dataset_dirs)

    print_summary(combined)
    print(f"\nAll results saved to {output_dir}/")


if __name__ == '__main__':
    main()
