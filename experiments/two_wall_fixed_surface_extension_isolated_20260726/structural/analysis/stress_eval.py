"""
应力评估工具 — PyTorch可微分版本

用于PINO输出的后处理:
- Von Mises应力计算 (可微分)
- 安全系数计算
- 主应力提取
"""
import torch
import numpy as np
import logging


logger = logging.getLogger(__name__)
_MISSING_SCALE_WARNED = False


def log_to_stress(log_stress: torch.Tensor, max_abs_log: float = 8.0) -> torch.Tensor:
    """Convert PINO v3 log-scale stress back to physical stress.

    v3 output format: sign(σ) · log10(|σ| + 1)
    Inverse: sign(x) · (10^|x| - 1)
    """
    log_stress = torch.clamp(log_stress, min=-max_abs_log, max=max_abs_log)
    return torch.sign(log_stress) * (10.0 ** torch.abs(log_stress) - 1.0)


def log_to_displacement(log_disp: torch.Tensor) -> torch.Tensor:
    """Convert PINO log-scale displacement back to physical displacement (meters).

    Training format: sign(u) · log10(|u| · 1e6 + 1)
    Inverse:         sign(x) · (10^|x| - 1) / 1e6
    """
    log_disp = torch.clamp(log_disp, min=-8.0, max=8.0)
    return torch.sign(log_disp) * (10.0 ** torch.abs(log_disp) - 1.0) / 1e6


def pino_stress_to_physical(
    pino_out_stress: torch.Tensor,
    is_v3: bool,
    stress_scale: float = 1.0,
    log_vm_max_pred: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Convert raw PINO stress output to usable stress tensor.

    For per-sample normalized models (v3/v12+):
        Output is σ/vm_max (normalized, peak VM ≈ 1.0). If log_vm_max_pred is
        provided, recover physical stress as:
            σ_phys = (σ / vm_max) * 10^(log_vm_max_pred)
        If scale is missing, falls back to normalized stress and emits a warning.

    For older non-normalized models:
        Returns stress * stress_scale (physical Pa).
    """
    if is_v3:
        if log_vm_max_pred is not None:
            if not torch.is_tensor(log_vm_max_pred):
                log_vm_max_pred = torch.as_tensor(
                    log_vm_max_pred,
                    device=pino_out_stress.device,
                    dtype=pino_out_stress.dtype,
                )
            log_vm_max_pred = log_vm_max_pred.to(
                device=pino_out_stress.device, dtype=pino_out_stress.dtype
            )
            if log_vm_max_pred.ndim == 1:
                log_vm_max_pred = log_vm_max_pred[:, None]
            while log_vm_max_pred.ndim < pino_out_stress.ndim:
                log_vm_max_pred = log_vm_max_pred.unsqueeze(-1)
            vm_scale = torch.pow(10.0, log_vm_max_pred)
            return pino_out_stress * vm_scale

        global _MISSING_SCALE_WARNED
        if not _MISSING_SCALE_WARNED:
            logger.warning(
                "pino_stress_to_physical called for a scale-aware normalized model "
                "without log_vm_max_pred; falling back to normalized stress."
            )
            _MISSING_SCALE_WARNED = True
        return pino_out_stress
    else:
        return pino_out_stress * stress_scale


def pino_displacement_to_physical(pino_out_disp: torch.Tensor, is_v3: bool,
                                  disp_scale: float = 1.0) -> torch.Tensor:
    """Convert raw PINO displacement output to physical displacement (meters).

    For v3 models: output is sign(u)·log10(|u|·1e6+1), needs log-inverse.
    For older models: output * disp_scale.
    """
    if is_v3:
        return log_to_displacement(pino_out_disp)
    else:
        return pino_out_disp * disp_scale


def get_yield_stress_for_model(pino_model, physical_yield_stress: float = 50e6) -> float:
    """Get the effective yield stress threshold for constraint evaluation.

    Per-sample normalized models output stress/vm_max (peak ≈ 1.0).
    Scale-aware normalized models recover physical stress and should keep the
    physical material threshold.
    """
    if getattr(pino_model, '_per_sample_normalized', False):
        if getattr(pino_model, '_has_scale_head', False):
            return physical_yield_stress
        return 1.0
    return physical_yield_stress


def compute_von_mises_torch(stress: torch.Tensor) -> torch.Tensor:
    """可微分的von Mises应力计算

    Args:
        stress: [B, 6, N, N, N] — [σxx, σyy, σzz, σyz, σzx, σxy]
    Returns:
        [B, 1, N, N, N] — von Mises应力
    """
    sxx = stress[:, 0]
    syy = stress[:, 1]
    szz = stress[:, 2]
    syz = stress[:, 3]
    szx = stress[:, 4]
    sxy = stress[:, 5]

    vm = torch.sqrt(
        0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)
        + 3.0 * (sxy**2 + syz**2 + szx**2)
        + 1e-12  # 数值稳定
    )
    return vm.unsqueeze(1)


def compute_safety_factor(
    von_mises: torch.Tensor,
    yield_stress: float = 50.0e6,
) -> torch.Tensor:
    """安全系数场: SF = σ_yield / σ_vm

    Args:
        von_mises: [B, 1, N, N, N]
        yield_stress: 屈服强度
    Returns:
        [B, 1, N, N, N] — 安全系数 (clip to [0, 100])
    """
    return torch.clamp(yield_stress / (von_mises + 1e-12), 0, 100)


def compute_principal_stresses_torch(stress: torch.Tensor) -> tuple:
    """计算主应力 (可微分)

    Args:
        stress: [B, 6, N, N, N] — Voigt记号
    Returns:
        eigenvalues: [B, 3, N, N, N] — 从大到小排列
        eigenvectors: [B, 3, 3, N, N, N] — 对应的主方向
    """
    B, _, Nx, Ny, Nz = stress.shape

    # 重排为 [B, N*N*N, 3, 3] 对称矩阵
    sxx = stress[:, 0].reshape(B, -1)
    syy = stress[:, 1].reshape(B, -1)
    szz = stress[:, 2].reshape(B, -1)
    syz = stress[:, 3].reshape(B, -1)
    szx = stress[:, 4].reshape(B, -1)
    sxy = stress[:, 5].reshape(B, -1)

    M = stress.shape[2] * stress.shape[3] * stress.shape[4]
    mat = torch.zeros(B, M, 3, 3, device=stress.device, dtype=stress.dtype)
    mat[:, :, 0, 0] = sxx
    mat[:, :, 1, 1] = syy
    mat[:, :, 2, 2] = szz
    mat[:, :, 0, 1] = sxy
    mat[:, :, 1, 0] = sxy
    mat[:, :, 1, 2] = syz
    mat[:, :, 2, 1] = syz
    mat[:, :, 0, 2] = szx
    mat[:, :, 2, 0] = szx

    # 特征值分解
    eigenvalues, eigenvectors = torch.linalg.eigh(mat)  # [B, M, 3], [B, M, 3, 3]

    # 降序排列
    eigenvalues = eigenvalues.flip(-1)
    eigenvectors = eigenvectors.flip(-1)

    eigenvalues = eigenvalues.reshape(B, Nx, Ny, Nz, 3).permute(0, 4, 1, 2, 3)
    eigenvectors = eigenvectors.reshape(B, Nx, Ny, Nz, 3, 3).permute(0, 4, 5, 1, 2, 3)

    return eigenvalues, eigenvectors


def structural_loss(
    von_mises: torch.Tensor,
    density: torch.Tensor,
    yield_stress: float = 50.0e6,
) -> torch.Tensor:
    """结构安全损失: 惩罚超过屈服应力的区域

    L = mean( ReLU(σ_vm/σ_yield - 1)² × ρ )
    """
    ratio = von_mises / yield_stress
    excess = torch.relu(ratio - 1.0)
    mask = (density > 0.1).float()
    return (excess ** 2 * mask).sum() / (mask.sum() + 1e-8)


def volume_loss(
    density: torch.Tensor,
    original_density: torch.Tensor,
    budget_ratio: float = 1.3,
) -> torch.Tensor:
    """体积预算损失

    L = ReLU(V_new/V_budget - 1)²
    """
    v_original = original_density.sum()
    v_new = density.sum()
    v_budget = v_original * budget_ratio
    return torch.relu(v_new / v_budget - 1.0) ** 2


def shape_preservation_loss(
    density_new: torch.Tensor,
    density_original: torch.Tensor,
) -> torch.Tensor:
    """外形保持损失: 只约束表面体素

    使用3D边界检测: 原始形状的表面体素不应被修改
    """
    # 简单实现: 通过max_pool检测表面
    orig_binary = (density_original > 0.5).float()

    # 膨胀 - 原始 = 外表面
    dilated = torch.nn.functional.max_pool3d(
        orig_binary, kernel_size=3, stride=1, padding=1
    )
    surface = dilated - orig_binary  # 外表面邻域
    # 内表面
    eroded = -torch.nn.functional.max_pool3d(
        -orig_binary, kernel_size=3, stride=1, padding=1
    )
    inner_surface = orig_binary - eroded

    surface_mask = (surface + inner_surface).clamp(0, 1)

    return ((density_new - density_original) ** 2 * surface_mask).mean()


def smoothness_loss(field: torch.Tensor) -> torch.Tensor:
    """修复场平滑正则化: ||∇Δρ||²"""
    dx = field[:, :, 1:, :, :] - field[:, :, :-1, :, :]
    dy = field[:, :, :, 1:, :] - field[:, :, :, :-1, :]
    dz = field[:, :, :, :, 1:] - field[:, :, :, :, :-1]
    return (dx ** 2).mean() + (dy ** 2).mean() + (dz ** 2).mean()


def compute_protected_shell_mask(
    density: torch.Tensor,
    shell_width: int = 2,
) -> torch.Tensor:
    """Compute a protected boundary shell from the original solid volume.

    The shell is preserved during repair so that silhouette and local appearance
    stay as unchanged as possible. Only the interior core is allowed to absorb
    additive/subtractive edits.
    """
    solid = (density > 0.5).float()
    eroded = solid
    for _ in range(max(shell_width, 1)):
        eroded = -torch.nn.functional.max_pool3d(
            -eroded, kernel_size=3, stride=1, padding=1
        )
    return (solid - eroded).clamp(0, 1)


def material_redistribution_loss(
    density_new: torch.Tensor,
    density_original: torch.Tensor,
    interior_mask: torch.Tensor,
    max_net_increase_ratio: float = 0.02,
) -> torch.Tensor:
    """Penalize net material growth to encourage redistribution over addition.

    The repair is allowed to add a tiny amount of net material, but most changes
    should come from moving capacity from low-stress interior regions to critical
    load paths.
    """
    added = torch.relu(density_new - density_original) * interior_mask
    removed = torch.relu(density_original - density_new) * interior_mask
    interior_mass = (density_original * interior_mask).sum().clamp(min=1e-8)
    allowed_net = interior_mass * max_net_increase_ratio
    net_add = added.sum() - removed.sum()
    return torch.relu(net_add - allowed_net) ** 2 / (allowed_net ** 2 + 1e-8)


def repair_locality_loss(
    density_new: torch.Tensor,
    density_original: torch.Tensor,
    high_priority_mask: torch.Tensor,
    low_priority_mask: torch.Tensor,
    interior_mask: torch.Tensor,
) -> torch.Tensor:
    """Encourage add/remove operations to stay in semantically appropriate regions.

    - additions should happen in high-stress / vulnerable interior regions
    - removals should happen in low-stress / redundant interior regions
    """
    added = torch.relu(density_new - density_original) * interior_mask
    removed = torch.relu(density_original - density_new) * interior_mask

    add_total = added.sum().clamp(min=1e-8)
    remove_total = removed.sum().clamp(min=1e-8)

    add_outside = (added * (1.0 - high_priority_mask)).sum() / add_total
    remove_outside = (removed * (1.0 - low_priority_mask)).sum() / remove_total
    return add_outside + remove_outside


def overhang_penalty(density: torch.Tensor, max_angle_deg: float = 45.0) -> torch.Tensor:
    """悬垂惩罚: 检测超过max_angle的无支撑结构

    简化实现: 对于每个实体体素，检查其正下方是否有支撑
    """
    # density: [B, 1, N, N, N], Y轴(dim=3)为竖直方向
    solid = (density > 0.5).float()

    # 下方体素 (Y轴正方向为向上)
    support_below = torch.zeros_like(solid)
    support_below[:, :, :, 1:, :] = solid[:, :, :, :-1, :]

    # 对角支撑 (45度以内)
    support_diag = torch.nn.functional.max_pool3d(
        support_below, kernel_size=(3, 1, 3), stride=1, padding=(1, 0, 1)
    )

    # 无支撑且非底层的实体体素
    unsupported = solid * (1 - support_diag)
    # 底层不惩罚
    unsupported[:, :, :, 0, :] = 0

    return unsupported.mean()
