"""
统一评价指标模块

8个维度评估每种方法的输出:
1. IoU — 外形与目标剪影的3D交并比
2. SF_min — FEM验证的最小安全系数
3. 约束满足率 — N个模型中 SF ≥ τ 的比例
4. 体积增加比 — V_opt / V_original
5. FEM验证误差 — |SF_PINO - SF_FEM| / SF_FEM
6. 计算时间 — 端到端秒数
7. Chamfer Distance — 表面质量
8. ε_u (PINO位移相对误差) — Safety Transfer Bound 关键参数
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class MethodResult:
    """单个方法在单个模型上的完整评估结果"""
    model_id: str
    method_name: str

    # 体素
    density: np.ndarray | None = None            # [N, N, N]
    density_original: np.ndarray | None = None    # [N, N, N]

    # 指标
    iou_3d: float = 0.0                  # 3D IoU vs 原始体素
    mv_iou: float = 0.0                  # 多视图轮廓 IoU
    mv_bce: float = 0.0                  # 多视图轮廓 BCE
    mv_ssim: float = 0.0                 # 多视图轮廓 SSIM
    sf_min_pino: float = 0.0             # 兼容字段: 等于保守 surrogate SF
    sf_min_surrogate: float = 0.0        # 保守 surrogate 最小安全系数
    sf_min_fem: float = 0.0              # FEM 验证最小安全系数
    sf_lower_bound: float = 0.0          # 经验保守筛查下界
    volume_ratio: float = 1.0            # V_opt / V_original
    pino_sf_error: float = 0.0           # |SF_surrogate - SF_FEM| / SF_FEM
    max_von_mises_pino: float = 0.0      # 兼容字段: 等于 surrogate VM
    max_von_mises_surrogate: float = 0.0 # 保守 surrogate 最大 VM 应力
    max_von_mises_fem: float = 0.0       # FEM 最大 VM 应力
    displacement_max_surrogate: float = 0.0
    displacement_max_fem: float = 0.0
    overhang_ratio: float = 0.0          # 悬垂比例
    wall_thickness_min: float = 0.0      # 最小壁厚 (mm)
    n_components: int = 1                # 连通分量数
    is_printable: bool = True            # 可打印性
    fem_safe: bool = False               # 最终是否 FEM 安全
    all_constraints_satisfied: bool = False
    converged: bool = True
    used_fem_final: bool = False
    failure_reason: str = ""
    per_load: List[Dict[str, float]] = field(default_factory=list)
    chamfer_distance: float = 0.0       # Chamfer Distance
    constraint_satisfied: bool = False   # 所有约束是否满足
    elapsed_seconds: float = 0.0         # 端到端用时


@dataclass
class ExperimentSummary:
    """整个实验的聚合统计"""
    method_name: str
    n_models: int = 0

    # 聚合指标 (mean ± std)
    iou_mean: float = 0.0
    iou_std: float = 0.0
    mv_bce_mean: float = 0.0
    mv_ssim_mean: float = 0.0
    sf_min_mean: float = 0.0
    sf_min_std: float = 0.0
    sf_fem_mean: float = 0.0
    sf_fem_std: float = 0.0
    displacement_mean_mm: float = 0.0
    constraint_satisfaction_rate: float = 0.0
    fem_safe_rate: float = 0.0
    convergence_rate: float = 0.0
    volume_ratio_mean: float = 0.0
    volume_ratio_std: float = 0.0
    pino_sf_error_mean: float = 0.0
    pino_sf_error_std: float = 0.0
    chamfer_mean: float = 0.0
    chamfer_std: float = 0.0
    time_mean: float = 0.0
    time_std: float = 0.0
    printability_rate: float = 0.0


def compute_3d_iou(
    density_pred: np.ndarray,
    density_target: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """3D 体素交并比

    Args:
        density_pred: [N, N, N] 预测密度
        density_target: [N, N, N] 目标密度
        threshold: 二值化阈值

    Returns:
        IoU ∈ [0, 1]
    """
    pred = (density_pred > threshold).astype(bool)
    target = (density_target > threshold).astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_silhouette_iou(
    density: np.ndarray,
    target_silhouettes: list,
    projection_axes: list | None = None,
) -> float:
    """多视图剪影 IoU (MV-IoU)

    将体素投影到多个视角与目标剪影对比。
    使用简单的正交投影 (max projection along axis)。

    Args:
        density: [N, N, N] 体素场
        target_silhouettes: list of [H, W] 目标剪影 (0-1)
        projection_axes: 投影轴列表, 默认 [0, 1, 2] 对应 front/side/top

    Returns:
        平均 IoU
    """
    if projection_axes is None:
        projection_axes = [0, 1, 2]

    solid = (density > 0.5).astype(np.float32)
    ious = []

    for i, axis in enumerate(projection_axes):
        if i >= len(target_silhouettes):
            break
        # 正交投影: 沿轴取 max
        proj = solid.max(axis=axis)

        target = target_silhouettes[i]
        # 如果尺寸不匹配, resize
        if proj.shape != target.shape:
            from scipy.ndimage import zoom
            scale = (target.shape[0] / proj.shape[0], target.shape[1] / proj.shape[1])
            proj = zoom(proj, scale, order=0)

        pred_bin = (proj > 0.5).astype(bool)
        target_bin = (target > 0.5).astype(bool)

        inter = np.logical_and(pred_bin, target_bin).sum()
        union = np.logical_or(pred_bin, target_bin).sum()
        if union > 0:
            ious.append(inter / union)

    return float(np.mean(ious)) if ious else 0.0


def load_target_silhouettes(silhouette_dir: str | Path | None) -> list:
    """从目录加载多视图目标剪影."""
    if silhouette_dir is None:
        return []

    sil_dir = Path(silhouette_dir)
    if not sil_dir.exists():
        return []

    try:
        from PIL import Image
    except ImportError:
        return []

    silhouettes = []
    for path in sorted(sil_dir.glob("view_*.png")):
        img = Image.open(path).convert("L")
        silhouettes.append((np.array(img, dtype=np.float32) / 255.0))
    return silhouettes


def compute_multi_view_metrics(
    density: np.ndarray,
    target_silhouettes: list,
    projection_axes: list | None = None,
) -> Dict[str, float]:
    """多视图轮廓指标."""
    if not target_silhouettes:
        return {'mv_iou': 0.0, 'mv_bce': 1.0, 'mv_ssim': 0.0}

    if projection_axes is None:
        projection_axes = list(range(min(3, len(target_silhouettes))))
        if len(target_silhouettes) >= 6:
            projection_axes = [0, 1, 2, 0, 1, 2]

    solid = (density > 0.5).astype(np.float32)
    ious = []
    bces = []
    ssims = []

    try:
        from scipy.ndimage import zoom
    except ImportError:
        zoom = None

    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        structural_similarity = None

    for i, axis in enumerate(projection_axes):
        if i >= len(target_silhouettes):
            break
        proj = solid.max(axis=axis).astype(np.float32)
        target = target_silhouettes[i].astype(np.float32)
        if proj.shape != target.shape and zoom is not None:
            scale = (target.shape[0] / proj.shape[0], target.shape[1] / proj.shape[1])
            proj = zoom(proj, scale, order=0)

        proj = np.clip(proj, 1e-5, 1.0 - 1e-5)
        target = np.clip(target, 0.0, 1.0)
        pred_bin = proj > 0.5
        target_bin = target > 0.5
        inter = np.logical_and(pred_bin, target_bin).sum()
        union = np.logical_or(pred_bin, target_bin).sum()
        ious.append(float(inter / union) if union > 0 else 1.0)
        bces.append(float(-(target * np.log(proj) + (1.0 - target) * np.log(1.0 - proj)).mean()))
        if structural_similarity is not None:
            ssims.append(float(structural_similarity(proj, target, data_range=1.0)))
        else:
            ssims.append(float(1.0 - np.abs(proj - target).mean()))

    return {
        'mv_iou': float(np.mean(ious)) if ious else 0.0,
        'mv_bce': float(np.mean(bces)) if bces else 1.0,
        'mv_ssim': float(np.mean(ssims)) if ssims else 0.0,
    }


def extract_load_cases(loads_npz, aggregation: str = "worst_case") -> list[Dict[str, np.ndarray]]:
    """从 loads.npz 提取统一 load case 列表."""
    keys = set(loads_npz.files)
    force_keys = sorted(k for k in keys if k.startswith("force_"))
    cases = []
    for fk in force_keys:
        suffix = fk[len("force_"):]
        fixed_key = f"fixed_mask_{suffix}"
        if fixed_key in keys:
            cases.append({
                'name': suffix,
                'force': loads_npz[fk],
                'fixed_mask': loads_npz[fixed_key],
            })

    if not cases and 'force_standing' in keys and 'fixed_mask_standing' in keys:
        cases.append({
            'name': 'standing',
            'force': loads_npz['force_standing'],
            'fixed_mask': loads_npz['fixed_mask_standing'],
        })

    if not cases and 'force' in keys and 'fixed_mask' in keys:
        cases.append({
            'name': 'default',
            'force': loads_npz['force'],
            'fixed_mask': loads_npz['fixed_mask'],
        })

    if aggregation == "first_only" and cases:
        return [cases[0]]
    return cases


def compute_chamfer_distance(
    density_pred: np.ndarray,
    density_target: np.ndarray,
    threshold: float = 0.5,
    n_points: int = 10000,
) -> float:
    """Chamfer Distance — 表面点云之间的对称距离

    从两个体素场各提取表面点云，计算双向最近邻距离均值。

    Args:
        density_pred: [N, N, N] 预测密度
        density_target: [N, N, N] 目标密度
        threshold: 二值化阈值
        n_points: 采样点数上限

    Returns:
        CD (越小越好, 0 = 完全重合)
    """
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree

    def _extract_surface_points(vol: np.ndarray) -> np.ndarray:
        solid = vol > threshold
        eroded = binary_erosion(solid)
        surface = solid & ~eroded
        pts = np.argwhere(surface).astype(np.float64)
        if len(pts) > n_points:
            idx = np.random.choice(len(pts), n_points, replace=False)
            pts = pts[idx]
        return pts

    pts_pred = _extract_surface_points(density_pred)
    pts_target = _extract_surface_points(density_target)

    if len(pts_pred) == 0 or len(pts_target) == 0:
        return float('inf')

    tree_pred = cKDTree(pts_pred)
    tree_target = cKDTree(pts_target)

    d_pred_to_target, _ = tree_target.query(pts_pred)
    d_target_to_pred, _ = tree_pred.query(pts_target)

    cd = float(d_pred_to_target.mean() + d_target_to_pred.mean()) / 2.0
    return cd


@torch.no_grad()
def compute_pino_epsilon_u(
    pino_model: torch.nn.Module,
    data_dir: str,
    device: torch.device | None = None,
    max_samples: int = 100,
) -> Dict[str, float]:
    """计算 PINO 位移相对误差 ε_u — Safety Transfer Bound 的关键参数

    ε_u = median( ||u_pred - u_fem|| / ||u_fem|| )

    使用 per-sample 反归一化: 预测值和真值在归一化空间比较,
    相对误差与 scale 无关。

    Args:
        pino_model: 预训练 PINO 模型
        data_dir: FEM 数据集目录
        device: 计算设备
        max_samples: 最多评估样本数

    Returns:
        dict with epsilon_u_median, epsilon_u_mean, epsilon_u_p95, etc.
    """
    from pathlib import Path

    if device is None:
        device = next(pino_model.parameters()).device

    pino_model.eval()
    data_path = Path(data_dir)
    files = sorted(data_path.glob("sample_*.npz"))[:max_samples]

    relative_errors = []
    stress_relative_errors = []

    for f in files:
        data = np.load(f)
        density = torch.from_numpy(data['density']).float().unsqueeze(0)     # [1, N, N, N]
        force = torch.from_numpy(data['force']).float()                       # [3, N, N, N]
        displacement = torch.from_numpy(data['displacement']).float()         # [3, N, N, N]
        stress = torch.from_numpy(data['stress']).float()                     # [6, N, N, N]

        # PINO 预测 (scale-aware if available)
        from structural.models.load_model import forward_surrogate
        x_density = density.unsqueeze(0).to(device)
        x_force = force.unsqueeze(0).to(device)
        _, pred, log_vm_max_pred = forward_surrogate(pino_model, x_density, x_force)
        pred_disp_norm = pred[0, :3].cpu()
        pred_stress_norm = pred[0, 3:].cpu()

        disp_norm = torch.sign(displacement) * torch.log10(torch.abs(displacement) * 1e6 + 1.0)
        solid_mask = (density[0] > 0.1)
        if solid_mask.any():
            vm_gt = compute_von_mises_torch(stress.unsqueeze(0))[0, 0]
            vm_max = float(vm_gt[solid_mask].max().clamp(min=1e-6).item())
        else:
            vm_max = 1.0
        stress_norm = stress / vm_max

        # 相对误差 (在归一化空间算 = 在物理空间算, 因为 scale 约掉)
        mask_3 = solid_mask.unsqueeze(0).expand(3, -1, -1, -1)
        mask_6 = solid_mask.unsqueeze(0).expand(6, -1, -1, -1)

        disp_err = (pred_disp_norm - disp_norm)[mask_3]
        disp_true = disp_norm[mask_3]
        disp_norm_val = torch.sqrt((disp_true ** 2).sum())
        if disp_norm_val > 1e-12:
            eps_u = float(torch.sqrt((disp_err ** 2).sum()) / disp_norm_val)
            relative_errors.append(eps_u)

        stress_err = (pred_stress_norm - stress_norm)[mask_6]
        stress_true = stress_norm[mask_6]
        stress_norm_val = torch.sqrt((stress_true ** 2).sum())
        if stress_norm_val > 1e-12:
            eps_s = float(torch.sqrt((stress_err ** 2).sum()) / stress_norm_val)
            stress_relative_errors.append(eps_s)

    if not relative_errors:
        return {'epsilon_u_median': float('inf'), 'epsilon_u_mean': float('inf')}

    rel = np.array(relative_errors)
    srel = np.array(stress_relative_errors) if stress_relative_errors else np.array([0.0])

    return {
        'epsilon_u_median': float(np.median(rel)),
        'epsilon_u_mean': float(np.mean(rel)),
        'epsilon_u_std': float(np.std(rel)),
        'epsilon_u_p95': float(np.percentile(rel, 95)),
        'epsilon_u_max': float(np.max(rel)),
        'epsilon_s_median': float(np.median(srel)),
        'epsilon_s_mean': float(np.mean(srel)),
        'epsilon_s_p95': float(np.percentile(srel, 95)),
        'n_samples': len(relative_errors),
    }


@torch.no_grad()
def evaluate_structural_pino(
    density: np.ndarray,
    force: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    stress_inflation_factor: float = 1.0,
) -> Dict[str, float]:
    """用 PINO 评估结构指标

    Args:
        density: [N, N, N] 密度场
        force: [3, N, N, N] 力场
        pino_model: 预训练 PINO
        yield_stress: 屈服强度

    Returns:
        dict with sf_min, max_vm, displacement_max
    """
    from structural.models.load_model import forward_surrogate_physical

    if device is None:
        device = next(pino_model.parameters()).device

    rho = torch.from_numpy(density).float().unsqueeze(0).unsqueeze(0).to(device)
    f = torch.from_numpy(force).float().unsqueeze(0).to(device)
    pino_fields = forward_surrogate_physical(pino_model, rho, f)
    pino_out = pino_fields['output']
    vm = pino_fields['von_mises']
    solid_mask = (rho > 0.1).float()
    vm_masked = vm * solid_mask

    vm_max_raw = float(vm_masked.max())
    vm_max = vm_max_raw * float(stress_inflation_factor)
    sf_min = yield_stress / (vm_max + 1e-12)

    u = pino_out[:, :3]
    u_max = float((torch.norm(u, dim=1, keepdim=True) * solid_mask).max())

    return {
        'sf_min': sf_min,
        'max_vm': vm_max,
        'max_vm_raw': vm_max_raw,
        'displacement_max': u_max,
    }


def evaluate_structural_fem(
    density: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    yield_stress: float = 50.0e6,
    youngs_modulus: float = 3.5e9,
    poisson_ratio: float = 0.36,
) -> Dict[str, float]:
    """用真实 FEM 评估结构指标 (SGLDBench)

    Args:
        density: [N, N, N]
        force: [3, N, N, N]
        fixed_mask: [1, N, N, N]
        yield_stress, youngs_modulus, poisson_ratio: 材料参数

    Returns:
        dict with sf_min, max_vm, displacement, stress fields
    """
    try:
        from structural.data.generate_fem_dataset import density_to_fem_fields
        from sgldbench.config import MaterialProperties, SolverSettings

        material = MaterialProperties(
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
            cell_size=1.0,
        )
        solver = SolverSettings(tolerance=1e-5, max_iterations=500)

        result = density_to_fem_fields(
            density.astype(bool),
            force, fixed_mask,
            material, solver,
        )

        if result is None:
            raise RuntimeError('SGLDBench density_to_fem_fields returned None')

        vm = result['von_mises']
        solid = density > 0.5
        vm_solid = vm[solid]

        if len(vm_solid) == 0:
            return {
                'sf_min': 0.0,
                'max_vm': 0.0,
                'fem_success': False,
                'error': 'no solid voxels after thresholding',
            }

        vm_max = float(vm_solid.max())
        sf_min = yield_stress / (vm_max + 1e-12)

        displacement = result.get('displacement')
        if displacement is not None:
            disp_mag = np.linalg.norm(displacement, axis=0)
            disp_max = float(disp_mag[solid].max()) if solid.any() else 0.0
        else:
            disp_max = 0.0

        return {
            'sf_min': sf_min,
            'max_vm': vm_max,
            'displacement_max': disp_max,
            'displacement': result['displacement'],
            'stress': result['stress'],
            'von_mises': result['von_mises'],
            'fem_success': True,
        }

    except Exception as primary_error:
        logger.warning("SGLDBench FEM failed (%s); trying project voxel-FEM", primary_error)
        try:
            from structural.analysis.fem_solver import voxel_fem_solve

            fallback = voxel_fem_solve(
                density=np.asarray(density, dtype=np.float32),
                force_field=np.asarray(force, dtype=np.float32),
                fixed_mask=np.asarray(fixed_mask, dtype=np.float32),
                E=youngs_modulus,
                nu=poisson_ratio,
                verbose=False,
            )
            solid = np.asarray(density) > 0.5
            vm = fallback['von_mises'][solid]
            vm = vm[np.isfinite(vm)]
            if fallback.get('n_dofs', 0) <= 0 or vm.size == 0:
                raise RuntimeError('voxel-FEM produced no constrained finite solution')
            vm_max = float(vm.max())
            displacement = fallback['displacement']
            disp_mag = np.linalg.norm(displacement, axis=0)
            return {
                'sf_min': float(yield_stress / max(vm_max, 1e-12)),
                'max_vm': vm_max,
                'displacement_max': float(disp_mag[solid].max()),
                'displacement': displacement,
                'stress': fallback['stress'],
                'von_mises': fallback['von_mises'],
                'fem_success': True,
                'fem_backend': 'project_voxel_fem_fallback',
                'primary_error': str(primary_error),
            }
        except Exception as fallback_error:
            logger.warning("FEM evaluation failed: %s", fallback_error)
            return {
                'sf_min': 0.0,
                'max_vm': float('inf'),
                'fem_success': False,
                'error': f'{primary_error}; fallback: {fallback_error}',
            }


def evaluate_printability(
    density: np.ndarray,
    voxel_size: float = 1.0,
) -> Dict[str, float]:
    """可打印性评估"""
    from structural.export.printability_check import check_printability
    report = check_printability(density, voxel_size=voxel_size)
    return {
        'is_printable': report.is_printable,
        'wall_thickness_min': report.min_wall_thickness,
        'overhang_ratio': report.max_overhang_ratio,
        'n_components': report.num_components,
        'total_volume': report.total_volume,
    }


def evaluate_single(
    density: np.ndarray,
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    model_id: str,
    method_name: str,
    pino_model: torch.nn.Module | None = None,
    yield_stress: float = 50.0e6,
    sf_target: float = 1.5,
    volume_budget: float = 1.3,
    run_fem: bool = False,
    device: torch.device | None = None,
    target_silhouettes: list | None = None,
    load_cases: list | None = None,
    stress_inflation_factor: float = 1.0,
    evaluation_config: object | None = None,
) -> MethodResult:
    """对单个模型的单个方法输出做完整评估

    Args:
        density: [N, N, N] 方法输出的密度
        density_original: [N, N, N] 目标/原始形状
        force: [3, N, N, N] 测试载荷
        fixed_mask: [1, N, N, N]
        model_id: 模型标识
        method_name: 方法名称
        pino_model: 预训练PINO (用于快速评估)
        yield_stress: 屈服强度
        sf_target: 安全系数目标
        volume_budget: 体积预算倍率
        run_fem: 是否用真实FEM验证
        target_silhouettes: 目标剪影列表 (可选, 用于MV-IoU)

    Returns:
        MethodResult
    """
    result = MethodResult(model_id=model_id, method_name=method_name)

    displacement_limit = getattr(evaluation_config, 'displacement_limit', 5e-3)
    volume_budget = getattr(evaluation_config, 'volume_budget_ratio', volume_budget)
    overhang_limit = getattr(evaluation_config, 'overhang_limit', 0.15)
    aggregation = getattr(evaluation_config, 'multi_load_aggregation', 'worst_case')

    # 3D IoU
    result.iou_3d = compute_3d_iou(density, density_original)
    result.density = density
    result.density_original = density_original

    # 轮廓指标
    if target_silhouettes is not None:
        mv_metrics = compute_multi_view_metrics(density, target_silhouettes)
        result.mv_iou = mv_metrics['mv_iou']
        result.mv_bce = mv_metrics['mv_bce']
        result.mv_ssim = mv_metrics['mv_ssim']
        result.iou_3d = result.mv_iou

    # Chamfer Distance
    result.chamfer_distance = compute_chamfer_distance(density, density_original)

    # 体积比
    v_orig = (density_original > 0.5).sum()
    v_new = (density > 0.5).sum()
    result.volume_ratio = float(v_new) / max(float(v_orig), 1.0)

    if load_cases is None:
        load_cases = [{'name': 'primary', 'force': force, 'fixed_mask': fixed_mask}]

    per_load = []
    for case in load_cases:
        case_result = {'name': case.get('name', 'load')}
        if pino_model is not None:
            pino_eval = evaluate_structural_pino(
                density, case['force'], pino_model, yield_stress, device,
                stress_inflation_factor=stress_inflation_factor,
            )
            case_result.update({
                'sf_surrogate': pino_eval['sf_min'],
                'vm_surrogate': pino_eval['max_vm'],
                'vm_surrogate_raw': pino_eval['max_vm_raw'],
                'u_surrogate': pino_eval['displacement_max'],
            })
        if run_fem:
            fem_eval = evaluate_structural_fem(
                density, case['force'], case['fixed_mask'], yield_stress
            )
            case_result.update({
                'fem_success': fem_eval.get('fem_success', False),
                'sf_fem': fem_eval.get('sf_min', 0.0),
                'vm_fem': fem_eval.get('max_vm', float('inf')),
                'u_fem': fem_eval.get('displacement_max', 0.0),
            })
        per_load.append(case_result)

    result.per_load = per_load

    if per_load:
        surrogate_sfs = [x.get('sf_surrogate', 0.0) for x in per_load if 'sf_surrogate' in x]
        surrogate_vms = [x.get('vm_surrogate', float('inf')) for x in per_load if 'vm_surrogate' in x]
        surrogate_us = [x.get('u_surrogate', 0.0) for x in per_load if 'u_surrogate' in x]
        if surrogate_sfs:
            result.sf_min_surrogate = min(surrogate_sfs)
            result.sf_min_pino = result.sf_min_surrogate
        if surrogate_vms:
            result.max_von_mises_surrogate = max(surrogate_vms)
            result.max_von_mises_pino = result.max_von_mises_surrogate
        if surrogate_us:
            result.displacement_max_surrogate = max(surrogate_us)

        fem_success_cases = [x for x in per_load if x.get('fem_success', False)]
        if fem_success_cases:
            result.used_fem_final = True
            result.sf_min_fem = min(x['sf_fem'] for x in fem_success_cases)
            result.max_von_mises_fem = max(x['vm_fem'] for x in fem_success_cases)
            result.displacement_max_fem = max(x['u_fem'] for x in fem_success_cases)
            if result.sf_min_fem > 0.0 and result.sf_min_surrogate > 0.0:
                result.pino_sf_error = abs(result.sf_min_surrogate - result.sf_min_fem) / result.sf_min_fem

    # 可打印性
    print_eval = evaluate_printability(density)
    result.is_printable = print_eval['is_printable']
    result.wall_thickness_min = print_eval['wall_thickness_min']
    result.overhang_ratio = print_eval['overhang_ratio']
    result.n_components = print_eval['n_components']

    # 约束是否全部满足
    sf_metric = result.sf_min_fem if (run_fem and result.sf_min_fem > 0.0 and aggregation == "worst_case") else result.sf_min_surrogate
    u_metric = result.displacement_max_fem if (run_fem and result.displacement_max_fem > 0.0 and aggregation == "worst_case") else result.displacement_max_surrogate
    sf_ok = sf_metric >= sf_target
    vol_ok = result.volume_ratio <= volume_budget
    disp_ok = u_metric <= displacement_limit if u_metric > 0.0 else True
    print_ok = result.is_printable and result.overhang_ratio <= overhang_limit
    result.fem_safe = result.sf_min_fem >= sf_target if result.used_fem_final else False
    result.all_constraints_satisfied = sf_ok and vol_ok and disp_ok and print_ok
    result.constraint_satisfied = result.all_constraints_satisfied
    if not sf_ok:
        result.failure_reason = "unsafe"
    elif not disp_ok:
        result.failure_reason = "displacement_exceeded"
    elif not vol_ok:
        result.failure_reason = "volume_exceeded"
    elif not print_ok:
        result.failure_reason = "printability_failed"
    else:
        result.failure_reason = ""

    return result


def aggregate_results(results: List[MethodResult], method_name: str) -> ExperimentSummary:
    """聚合多个模型的结果为统计摘要"""
    n = len(results)
    if n == 0:
        return ExperimentSummary(method_name=method_name)

    ious = [r.iou_3d for r in results]
    sfs = [r.sf_min_surrogate for r in results]
    fem_sfs = [r.sf_min_fem for r in results if r.sf_min_fem > 0]
    mv_bces = [r.mv_bce for r in results]
    mv_ssims = [r.mv_ssim for r in results]
    disps = [1e3 * (r.displacement_max_fem if r.displacement_max_fem > 0 else r.displacement_max_surrogate) for r in results]
    vols = [r.volume_ratio for r in results]
    errs = [r.pino_sf_error for r in results if r.pino_sf_error > 0]
    cds = [r.chamfer_distance for r in results if r.chamfer_distance < float('inf')]
    times = [r.elapsed_seconds for r in results]
    cs = [r.constraint_satisfied for r in results]
    ps = [r.is_printable for r in results]
    fem_safe = [r.fem_safe for r in results]
    converged = [r.converged for r in results]

    return ExperimentSummary(
        method_name=method_name,
        n_models=n,
        iou_mean=float(np.mean(ious)),
        iou_std=float(np.std(ious)),
        mv_bce_mean=float(np.mean(mv_bces)) if mv_bces else 0.0,
        mv_ssim_mean=float(np.mean(mv_ssims)) if mv_ssims else 0.0,
        sf_min_mean=float(np.mean(sfs)),
        sf_min_std=float(np.std(sfs)),
        sf_fem_mean=float(np.mean(fem_sfs)) if fem_sfs else 0.0,
        sf_fem_std=float(np.std(fem_sfs)) if fem_sfs else 0.0,
        displacement_mean_mm=float(np.mean(disps)) if disps else 0.0,
        constraint_satisfaction_rate=float(np.mean(cs)),
        fem_safe_rate=float(np.mean(fem_safe)) if fem_safe else 0.0,
        convergence_rate=float(np.mean(converged)) if converged else 0.0,
        volume_ratio_mean=float(np.mean(vols)),
        volume_ratio_std=float(np.std(vols)),
        pino_sf_error_mean=float(np.mean(errs)) if errs else 0.0,
        pino_sf_error_std=float(np.std(errs)) if errs else 0.0,
        chamfer_mean=float(np.mean(cds)) if cds else 0.0,
        chamfer_std=float(np.std(cds)) if cds else 0.0,
        time_mean=float(np.mean(times)),
        time_std=float(np.std(times)),
        printability_rate=float(np.mean(ps)),
    )


def format_results_table(summaries: List[ExperimentSummary]) -> str:
    """格式化为 LaTeX 表格"""
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Quantitative comparison of all methods.}\n"
        "\\label{tab:main_results}\n"
        "\\begin{tabular}{l|ccccccc}\n"
        "\\toprule\n"
        "Method & MV-IoU$\\uparrow$ & SSIM$\\uparrow$ & FEM SF$\\uparrow$ & FEM-safe(\\%)$\\uparrow$ "
        "& Disp.(mm)$\\downarrow$ & Vol. Ratio$\\downarrow$ & Time(s)$\\downarrow$ \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for s in summaries:
        row = (
            f"{s.method_name} & "
            f"{s.iou_mean:.3f}$\\pm${s.iou_std:.3f} & "
            f"{s.mv_ssim_mean:.3f} & "
            f"{(s.sf_fem_mean if s.sf_fem_mean > 0 else s.sf_min_mean):.2f}$\\pm${(s.sf_fem_std if s.sf_fem_mean > 0 else s.sf_min_std):.2f} & "
            f"{s.fem_safe_rate * 100:.1f} & "
            f"{s.displacement_mean_mm:.2f} & "
            f"{s.volume_ratio_mean:.2f}$\\pm${s.volume_ratio_std:.2f} & "
            f"{s.time_mean:.1f} \\\\"
        )
        rows.append(row)

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )

    return header + "\n".join(rows) + "\n" + footer


def format_results_markdown(summaries: List[ExperimentSummary]) -> str:
    """格式化为 Markdown 表格 (方便调试)"""
    lines = [
        "| Method | MV-IoU↑ | MV-SSIM↑ | FEM-SF↑ | FEM-safe(%)↑ | Disp(mm)↓ | Vol.Ratio↓ | Time(s)↓ |",
        "|--------|---------|----------|---------|---------------|-----------|------------|----------|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.method_name} | "
            f"{s.iou_mean:.3f}±{s.iou_std:.3f} | "
            f"{s.mv_ssim_mean:.3f} | "
            f"{(s.sf_fem_mean if s.sf_fem_mean > 0 else s.sf_min_mean):.2f}±{(s.sf_fem_std if s.sf_fem_mean > 0 else s.sf_min_std):.2f} | "
            f"{s.fem_safe_rate * 100:.1f} | "
            f"{s.displacement_mean_mm:.2f} | "
            f"{s.volume_ratio_mean:.2f}±{s.volume_ratio_std:.2f} | "
            f"{s.time_mean:.1f} |"
        )
    return "\n".join(lines)
