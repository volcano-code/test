"""
联合训练管线 (Stage A + B + C)

三阶段训练策略:
  Stage A: 形状形成 (ShadowArt原有流程)
  Stage B: 结构修复 (冻结形状，优化RepairCNN)
  Stage C: 约束感知联合优化 (ALM硬约束 + 自适应梯度路由 + 安全保证)

方法论贡献 (Stage C):
  1. ALM (增广拉格朗日法): 将安全系数/体积/可打印性作为硬约束
  2. Safety Transfer Bound: PINO近似误差→结构安全保证
  3. Adaptive Gradient Router: 约束感知的内外解耦

用法:
    python train_joint.py --shadow_files img1.png img2.png ... --pino_checkpoint path/to/pino.pt
"""
import sys
import os
import argparse
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mesh_combined2"))
sys.path.insert(0, str(PROJECT_ROOT / "generation"))
sys.path.insert(0, str(PROJECT_ROOT / "structural"))

from structural.models.load_model import load_surrogate, build_pino_input, forward_surrogate_physical
from structural.models.repair_cnn import RepairCNN
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.vulnerability import VulnerabilityDetector
from structural.analysis.stress_eval import (
    compute_von_mises_torch, pino_stress_to_physical,
    pino_displacement_to_physical, get_yield_stress_for_model,
    structural_loss, volume_loss,
    shape_preservation_loss, smoothness_loss, overhang_penalty,
)
from structural.configs import StructShadowConfig
from structural.optimization.alm_optimizer import ALMOptimizer, ALMConfig
from structural.optimization.safety_bound import SafetyTransferBound, AdaptiveSafetyBound
from structural.optimization.gradient_router import AdaptiveGradientRouter

import logging
logger = logging.getLogger(__name__)


def _compute_projection_iou(density: np.ndarray, density_original: np.ndarray, threshold: float = 0.5) -> float:
    """Project voxel fields along 3 axes and average the IoU."""
    solid_new = density > threshold
    solid_orig = density_original > threshold
    ious: list[float] = []
    for axis in range(3):
        proj_new = solid_new.any(axis=axis)
        proj_orig = solid_orig.any(axis=axis)
        union = np.logical_or(proj_new, proj_orig).sum()
        if union == 0:
            ious.append(1.0)
            continue
        inter = np.logical_and(proj_new, proj_orig).sum()
        ious.append(float(inter / union))
    return float(np.mean(ious)) if ious else 0.0



def load_pino(checkpoint_path: str, config: StructShadowConfig, device: torch.device):
    """加载预训练 PINO/UFNO surrogate"""
    pino, ckpt = load_surrogate(checkpoint_path, device)

    for p in pino.parameters():
        p.requires_grad = False

    # 提取 PINO 验证误差 (如果checkpoint中记录了)
    pino_epsilon_u = ckpt.get('val_displacement_error', None)
    if pino_epsilon_u is not None:
        logger.info(f"PINO ε_u from checkpoint: {pino_epsilon_u:.4f}")

    return pino, pino_epsilon_u


def run_stage_a(
    shadow_files: list,
    output_dir: str,
    voxel_res: int = 64,
    num_views: int = 6,
    Niter: int = 500,
    lr: float = 1e-3,
    device: torch.device = None,
    **kwargs,
) -> str:
    """Stage A: ShadowArt形状形成

    直接调用mesh_combined2/train.py的训练流程
    返回训练好的体素路径
    """
    print("=" * 80)
    print("STAGE A: Shape Formation (ShadowArt)")
    print("=" * 80)

    from train import train as shadowart_train

    output_path = Path(output_dir) / "stage_a"
    output_path.mkdir(parents=True, exist_ok=True)

    shadowart_train(
        device=device,
        shadow_files=shadow_files,
        output_dir=str(output_path),
        Niter=Niter,
        lr=lr,
        exp_sample_id="structshadow",
        num_views=num_views,
        mirror_mode=False,
        TSF_mode=(num_views == 6),
        TSF4v_mode=False,
        zdist=1.7,
        camera_mode="perp",
        img_size=128,
        debug_mode=False,
        voxel_res=voxel_res,
        use_visual_hull=True,
        post_export_bundle=False,
    )

    voxel_files = list(output_path.glob("*.npy"))
    if voxel_files:
        return str(voxel_files[-1])

    return str(output_path / "voxels.npy")


def run_stage_b(
    voxel_path: str,
    pino,
    config: StructShadowConfig,
    output_dir: str,
    device: torch.device,
) -> RepairCNN:
    """Stage B: 结构修复优化"""
    print("\n" + "=" * 80)
    print("STAGE B: Structural Repair Optimization")
    print("=" * 80)

    output_path = Path(output_dir) / "stage_b"
    output_path.mkdir(parents=True, exist_ok=True)

    voxel_data = np.load(voxel_path)
    if voxel_data.ndim == 4:
        voxel_data = voxel_data[0]

    N = voxel_data.shape[0]
    density_original = torch.from_numpy(voxel_data).float().unsqueeze(0).unsqueeze(0).to(device)

    repair_cnn = RepairCNN(
        in_channels=3,
        channels=config.repair.channels,
        delta_max=config.repair.delta_max,
    ).to(device)

    optimizer = torch.optim.Adam(repair_cnn.parameters(), lr=config.repair.lr)
    rc = config.repair

    load_gen = LoadGenerator(material_density=config.material.density)
    rng = np.random.default_rng(42)

    load_cases = []
    for _ in range(8):
        ld = load_gen.generate_random_load(voxel_data, rng)
        load_cases.append(load_gen.to_torch(ld, device))

    def compute_interior_mask(density_tensor):
        solid = (density_tensor > 0.5).float()
        return -torch.nn.functional.max_pool3d(-solid, kernel_size=5, stride=1, padding=2)

    effective_yield = get_yield_stress_for_model(pino, config.material.yield_stress)
    vuln_detector = VulnerabilityDetector(yield_stress=effective_yield)

    history = []
    best_loss = float('inf')

    for step in tqdm(range(1, rc.steps + 1), desc="Stage B"):
        repair_cnn.train()

        load = load_cases[step % len(load_cases)]
        force = load['force'].unsqueeze(0)

        with torch.no_grad():
            pino_fields = forward_surrogate_physical(pino, density_original, force)
        vm = pino_fields['von_mises']
        vm_norm = vm / (vm.max() + 1e-8)

        if step == 1 or step % 10 == 0:
            vuln_result = vuln_detector.detect(
                density_original[0, 0].detach().cpu().numpy(),
                vm[0, 0].detach().cpu().numpy()
            )
            vuln_map = torch.from_numpy(
                vuln_result['vulnerability_map']
            ).float().unsqueeze(0).unsqueeze(0).to(device)
            vuln_map = vuln_map / (vuln_map.max() + 1e-8)

        repair_input = torch.cat([density_original, vm_norm, vuln_map], dim=1)
        repair_fields = repair_cnn(repair_input)
        interior_mask = compute_interior_mask(density_original)
        density_repaired = repair_cnn.apply_repair(density_original, repair_fields, interior_mask)

        pino_fields_new = forward_surrogate_physical(pino, density_repaired, force)
        vm_new = pino_fields_new['von_mises']

        total_loss = (
            rc.lambda_structural * structural_loss(vm_new, density_repaired, effective_yield)
            + rc.lambda_volume * volume_loss(density_repaired, density_original, rc.volume_budget)
            + rc.lambda_shape * shape_preservation_loss(density_repaired, density_original)
            + rc.lambda_smooth * smoothness_loss(repair_fields['combined'])
            + rc.lambda_printable * overhang_penalty(density_repaired)
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(repair_cnn.parameters(), 1.0)
        optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save(repair_cnn.state_dict(), output_path / 'best_repair.pt')

        history.append({'step': step, 'loss': total_loss.item()})

    with open(output_path / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 最终评估
    repair_cnn.load_state_dict(torch.load(output_path / 'best_repair.pt', weights_only=True))
    with torch.no_grad():
        eval_load = load_cases[0]
        eval_force = eval_load['force'].unsqueeze(0)
        pino_eval = forward_surrogate_physical(pino, density_original, eval_force)
        vm_eval = pino_eval['von_mises']
        vm_eval_norm = vm_eval / (vm_eval.max() + 1e-8)
        vuln_eval = vuln_map
        final_input = torch.cat([density_original, vm_eval_norm, vuln_eval], dim=1)
        final_fields = repair_cnn(final_input)
        final_interior = compute_interior_mask(density_original)
        final_density = repair_cnn.apply_repair(density_original, final_fields, final_interior)
        np.save(output_path / 'repaired_voxels.npy', final_density.cpu().numpy())

    with torch.no_grad():
        pino_final = forward_surrogate_physical(pino, final_density, eval_force)
        vm_final = pino_final['von_mises']
        max_vm_before = vm_eval.max().item()
        max_vm_after = vm_final.max().item()
    print(f"Stage B complete. Best loss: {best_loss:.4e}")
    print(f"  Max VM stress: {max_vm_before:.2e} → {max_vm_after:.2e}")

    return repair_cnn


def run_stage_c(
    shadow_files: list,
    voxel_path: str,
    pino,
    repair_cnn: RepairCNN,
    config: StructShadowConfig,
    output_dir: str,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-4,
    pino_epsilon_u: float | None = None,
):
    """Stage C: 保守 surrogate + FEM 校验的约束联合优化."""
    print("\n" + "=" * 80)
    print("STAGE C: Constrained Joint Optimization (ALM + conservative surrogate + FEM probes)")
    print("=" * 80)

    output_path = Path(output_dir) / "stage_c"
    output_path.mkdir(parents=True, exist_ok=True)

    from utils import get_voxel_renderer, create_cameras_TFS_mode, create_cameras
    from datasets import load_data_from_list
    from losses import differentiable_iou_loss, silh_loss, MS_SSIM

    N = config.resolution
    num_views = len(shadow_files)
    volume_extent_world = 1.7
    img_size = 128

    # --- 相机与渲染器 ---
    # Use ring cameras to match the number of silhouette files directly.
    # TFS mode may produce fewer cameras than views; ring mode is 1:1.
    cameras, Rs, Ts = create_cameras(num_views, device, 1.7, "perp", False)

    renderer = get_voxel_renderer(device, cameras, img_size, volume_extent_world, N)

    silhs, silhs_tensors = load_data_from_list(
        shadow_files, False, (img_size, img_size), device, num_views, False
    )
    silhs_tensors = [s.to(device) for s in silhs_tensors]
    actual_num_views = len(silhs_tensors)

    from pytorch3d.renderer.cameras import FoVOrthographicCameras
    def render_view(n, vol_model):
        """Render a single view, matching train.py's per-view rendering."""
        R = Rs[n:n + 1].to(device=device, dtype=torch.float32).contiguous()
        T = Ts[n:n + 1].to(device=device, dtype=torch.float32).contiguous()
        bc = FoVOrthographicCameras(R=R, T=T, device=device)
        rendered_images, _ = vol_model(bc).split([3, 1], dim=-1)
        return rendered_images[0][..., 0]

    # --- 模型 ---
    from structural.models.volume_model import StructuralVolumeModel
    model = StructuralVolumeModel(
        renderer=renderer,
        volume_size=N,
        voxel_size=volume_extent_world / N,
        repair_cnn=repair_cnn,
        pino_model=pino,
    ).to(device)

    # 加载 Stage A 体素 (logit 变换在 temperature 设置后执行)
    voxel_data = np.load(voxel_path)
    if voxel_data.ndim == 4:
        voxel_data = voxel_data[0]

    # --- 载荷 ---
    load_gen = LoadGenerator(material_density=config.material.density)
    rng = np.random.default_rng(42)
    load_cases = []
    for _ in range(4):
        ld = load_gen.generate_random_load(voxel_data, rng)
        load_cases.append(load_gen.to_torch(ld, device))

    ms_ssim_fn = MS_SSIM(device=device)

    # --- Temperature 退火调度 ---
    # 优化前期 temperature 低 → sigmoid 导数大 → 梯度能流通
    # 后期逐步升高 → 逼近二值化
    temp_init = 1.0
    temp_final = 15.0
    temp_anneal_epochs = int(epochs * 0.7)  # 前70%做退火
    model.temperature = temp_init

    # 初始化 log_densities — 用中等 logit 范围
    # 压缩到 ±3 范围: sigmoid(±3) ≈ 0.95/0.05, 导数 ≈ 0.05 → 梯度可流通
    # 同时保持足够的密度对比度让 PINO 产生有意义的输出
    with torch.no_grad():
        density_tensor = torch.from_numpy(voxel_data).float().to(device)
        density_clamped = torch.clamp(density_tensor, 0.05, 0.95)
        logit = torch.log(density_clamped / (1 - density_clamped))
        model.log_densities.data[0] = logit / model.temperature
    density_original = model.density.detach().unsqueeze(0)

    # --- 优化器 (在 temperature 和 logit 初始化之后创建) ---
    optimizer = torch.optim.Adam([
        {'params': [model.log_densities], 'lr': lr * 10},
        {'params': repair_cnn.parameters(), 'lr': lr * 0.5},
    ])

    # --- 动态 displacement limit ---
    # 基于初始结构的位移设定合理上限，而非硬编码 5mm
    with torch.no_grad():
        _init_force = load_cases[0]['force'].unsqueeze(0)
        _init_fields = forward_surrogate_physical(pino, density_original, _init_force)
        _init_u = pino_displacement_to_physical(_init_fields['output'][:, :3], getattr(pino, '_is_v3', False))
        _init_u_max = float(torch.norm(_init_u, dim=1).max())
        adaptive_disp_limit = max(_init_u_max * 1.5, 5e-3)  # 允许 50% 恶化，下限 5mm
        print(f"  [Stage C] Initial u_max={_init_u_max:.4e}, adaptive displacement_limit={adaptive_disp_limit:.4e}")

    # ================================================================
    # 贡献1: 初始化 ALM 优化器
    # ================================================================
    alm_config = ALMConfig(
        safety_factor_target=1.5,
        volume_budget_ratio=config.repair.volume_budget,
        overhang_limit=0.15,
        displacement_limit=adaptive_disp_limit,
        rho_init=1.0,
        rho_max=1000.0,
        rho_growth=2.0,
        mu_max=1000.0,
        constraint_tol=0.01,
        dual_update_interval=5,
        lambda_appearance=10.0,
        lambda_shape_preserve=5.0,
        lambda_smoothness=1.0,
    )
    alm = ALMOptimizer(alm_config)

    # ================================================================
    # 保守 surrogate 设置
    # ================================================================
    effective_yield_stb = get_yield_stress_for_model(pino, config.material.yield_stress)
    stb = SafetyTransferBound(
        youngs_modulus=config.material.youngs_modulus,
        poisson_ratio=config.material.poisson_ratio,
        resolution=config.resolution,
        yield_stress=effective_yield_stb,
    )
    # 使用 PINO checkpoint 中的验证误差; 若无, 使用保守估计
    epsilon_u = pino_epsilon_u if pino_epsilon_u is not None else 0.10
    logger.info(f"Using PINO ε_u = {epsilon_u:.4f} for safety bound")
    surrogate_cfg = getattr(config, 'surrogate', None)
    stress_inflation = getattr(surrogate_cfg, 'stress_inflation_factor', 1.15)
    fem_probe_interval = getattr(surrogate_cfg, 'fem_probe_interval', 25)
    rollback_on_fem_fail = getattr(surrogate_cfg, 'rollback_on_fem_fail', True)
    acceptance_metric = getattr(surrogate_cfg, 'acceptance_metric', 'best_fem_feasible')

    # 自适应安全界: 在优化过程中动态估计局部 ε_u
    adaptive_stb = AdaptiveSafetyBound(
        base_bound=stb,
        n_probes=8,
        delta=0.05,
        update_interval=50,
        ema_alpha=0.3,
    )

    # ================================================================
    # 调度器: 保留为启发式更新调度，不再作为理论卖点
    # ================================================================
    grad_router = AdaptiveGradientRouter(
        sf_target=alm_config.safety_factor_target,
        sil_loss_threshold=0.5,
        structure_priority_sf_ratio=0.8,
        shape_priority_sil_ratio=1.5,
        interior_boost=3.0,
        surface_boost=3.0,
        repair_boost=2.0,
        warmup_steps=20,
    )

    # --- 训练循环 ---
    history = []
    best_loss = float('inf')
    safety_bound_log = []
    safe_checkpoint = None
    best_surrogate_feasible = None
    best_appearance_feasible = None
    best_fem_feasible = None

    for epoch in tqdm(range(1, epochs + 1), desc="Stage C (ALM)"):
        model.train()
        repair_cnn.train()

        # ============ Temperature 退火 ============
        if epoch <= temp_anneal_epochs:
            t = (epoch - 1) / max(temp_anneal_epochs - 1, 1)
            model.temperature = temp_init + (temp_final - temp_init) * t
        else:
            model.temperature = temp_final

        # ============ 前向: 渲染剪影 (per-view, matching train.py) ============
        l_sil = torch.tensor(0.0, device=device)
        for v_idx in range(actual_num_views):
            pred_sil = render_view(v_idx, model)
            gt_sil = silhs_tensors[v_idx]
            l_sil = l_sil + differentiable_iou_loss(pred_sil, gt_sil)
            l_sil = l_sil + silh_loss(pred_sil, gt_sil)
            l_sil = l_sil + ms_ssim_fn(
                pred_sil.unsqueeze(0).unsqueeze(0),
                gt_sil.unsqueeze(0).unsqueeze(0)
            )
        l_sil = l_sil / actual_num_views

        # ============ 前向: 结构分析 ============
        load = load_cases[epoch % len(load_cases)]
        force = load['force'].unsqueeze(0)
        density_current = model.density.unsqueeze(0)  # [1, 1, N, N, N]

        # 直接传连续密度给 PINO — 梯度直通
        # PINO 虽在二值密度上训练，但对连续输入也产生合理梯度
        pino_fields = forward_surrogate_physical(pino, density_current, force)
        pino_out = pino_fields['output']
        is_v3 = getattr(pino, '_is_v3', False)
        u_pred = pino_displacement_to_physical(pino_out[:, :3], is_v3)  # physical meters
        vm_raw = pino_fields['von_mises']
        vm = vm_raw * stress_inflation

        # ============ 贡献1: ALM 约束计算 (含 g4 位移约束) ============
        effective_yield_c = get_yield_stress_for_model(pino, config.material.yield_stress)
        violations = alm.compute_constraint_violations(
            von_mises=vm,
            density=density_current,
            density_original=density_original,
            yield_stress=effective_yield_c,
            displacement=u_pred,
        )

        # 正则项 (非约束, 直接加入目标函数)
        l_shape = shape_preservation_loss(density_current, density_original)
        l_smooth = smoothness_loss(density_current)

        # ALM 总损失
        alm_result = alm.compute_alm_loss(
            appearance_loss=l_sil,
            violations=violations,
            shape_preserve_loss=l_shape,
            smoothness_loss_val=l_smooth,
        )
        total_loss = alm_result['total']

        # ============ 反向传播 ============
        optimizer.zero_grad()
        total_loss.backward()

        # ============ 贡献3: 自适应梯度路由 ============
        # 计算当前安全系数 (用于路由决策)
        with torch.no_grad():
            solid_mask = (density_current > 0.1).float()
            vm_max_current = float((vm * solid_mask).max())
            sf_current = effective_yield_c / (vm_max_current + 1e-12)
            # 最大位移 (固体区域内)
            u_magnitude = torch.norm(u_pred, dim=1, keepdim=True)
            u_max_current = float((u_magnitude * solid_mask).max())
            # 体积比
            vol_ratio = float(density_current.sum() / density_original.sum().clamp(min=1.0))
            n_violated = sum(
                1 for v in violations.values()
                if float(v.detach()) > alm_config.constraint_tol
            )
            surrogate_feasible = (
                sf_current >= alm_config.safety_factor_target
                and u_max_current <= alm_config.displacement_limit
                and vol_ratio <= alm_config.volume_budget_ratio
                and float(violations['overhang'].detach()) <= alm_config.constraint_tol
            )

        routing_decision = grad_router.route(
            model=model,
            repair_cnn=repair_cnn,
            sf_current=sf_current,
            sil_loss=float(l_sil.detach()),
            n_constraints_violated=n_violated,
            step=epoch,
            volume_ratio=vol_ratio,
            u_max=u_max_current,
        )

        # 梯度裁剪 — 分开处理密度和修复网络
        # PINO 梯度量级 ~1e6，需要对应的 clip 值
        if model.log_densities.grad is not None:
            ld_norm = model.log_densities.grad.norm()
            if ld_norm > 0:
                # 归一化到单位球后乘以合理步长
                model.log_densities.grad.div_(ld_norm).mul_(1.0)
        torch.nn.utils.clip_grad_norm_(repair_cnn.parameters(), 1.0)

        optimizer.step()

        # ============ 贡献1: ALM 对偶更新 ============
        alm.update_violations(violations)
        alm.step(epoch)

        # ============ 保守筛查: 周期性 safety screening ============
        if epoch % fem_probe_interval == 0 or epoch == 1:
            with torch.no_grad():
                # 传统全局界
                bound_result = stb.compute_bound(
                    pino_model=pino,
                    density=density_current.detach(),
                    force=force,
                    epsilon_u=epsilon_u,
                )

                # 自适应界: 更新局部 ε_u 估计
                adaptive_stb.update(epoch, pino, density_current.detach(), force)
                adaptive_result = adaptive_stb.compute_adaptive_bound(
                    pino, density_current.detach(), force, epsilon_u,
                )

                safety_bound_log.append({
                    'epoch': epoch,
                    'sf_pino': bound_result.sf_pino,
                    'sf_lower_bound_global': bound_result.sf_lower_bound,
                    'sf_lower_bound_adaptive': adaptive_result.sf_lower_bound,
                    'epsilon_u_global': epsilon_u,
                    'epsilon_u_local': adaptive_result.epsilon_u_local,
                    'confidence': adaptive_result.confidence,
                    'vm_max_pino': bound_result.vm_max_pino,
                })
                print(f"\n  [Safety Screen] epoch {epoch}: "
                      f"SF_PINO={bound_result.sf_pino:.3f}, "
                      f"SF_lower(global)≥{bound_result.sf_lower_bound:.3f}, "
                      f"SF_lower(adaptive)≥{adaptive_result.sf_lower_bound:.3f}, "
                      f"ε_local={adaptive_result.epsilon_u_local:.4f}")

                if surrogate_feasible:
                    best_surrogate_feasible = {
                        'epoch': epoch,
                        'load_case_index': int(epoch % len(load_cases)),
                        'sil_loss': float(l_sil.detach()),
                        'log_densities': model.log_densities.detach().cpu().clone(),
                        'repair_cnn': repair_cnn.state_dict(),
                    }
                    if best_appearance_feasible is None or float(l_sil.detach()) < best_appearance_feasible['sil_loss']:
                        best_appearance_feasible = {
                            'epoch': epoch,
                            'load_case_index': int(epoch % len(load_cases)),
                            'sil_loss': float(l_sil.detach()),
                            'log_densities': model.log_densities.detach().cpu().clone(),
                            'repair_cnn': repair_cnn.state_dict(),
                        }

                if surrogate_feasible:
                    try:
                        from structural.experiments.evaluate_metrics import evaluate_structural_fem
                        fem_eval = evaluate_structural_fem(
                            density_current[0, 0].detach().cpu().numpy(),
                            force[0].detach().cpu().numpy(),
                            load['fixed_mask'].detach().cpu().numpy(),
                            yield_stress=config.material.yield_stress,
                            youngs_modulus=config.material.youngs_modulus,
                            poisson_ratio=config.material.poisson_ratio,
                        )
                    except Exception as exc:
                        fem_eval = {'fem_success': False, 'error': str(exc)}

                    fem_safe = bool(fem_eval.get('fem_success', False) and fem_eval.get('sf_min', 0.0) >= alm_config.safety_factor_target)
                    safety_bound_log[-1]['fem_success'] = bool(fem_eval.get('fem_success', False))
                    safety_bound_log[-1]['sf_fem'] = float(fem_eval.get('sf_min', 0.0))
                    safety_bound_log[-1]['u_fem'] = float(fem_eval.get('displacement_max', 0.0))
                    safety_bound_log[-1]['fem_safe'] = fem_safe
                    if fem_safe:
                        safe_checkpoint = {
                            'epoch': epoch,
                            'load_case_index': int(epoch % len(load_cases)),
                            'sil_loss': float(l_sil.detach()),
                            'log_densities': model.log_densities.detach().cpu().clone(),
                            'repair_cnn': repair_cnn.state_dict(),
                        }
                        if best_fem_feasible is None or float(l_sil.detach()) < best_fem_feasible['sil_loss']:
                            best_fem_feasible = dict(safe_checkpoint)
                    elif rollback_on_fem_fail and safe_checkpoint is not None:
                        model.log_densities.data.copy_(safe_checkpoint['log_densities'].to(device))
                        repair_cnn.load_state_dict(safe_checkpoint['repair_cnn'])
                        stress_inflation = min(stress_inflation * 1.05, 1.75)

        # ============ 日志 ============
        record = {
            'epoch': epoch,
            'total': total_loss.item(),
            'appearance': float(alm_result['appearance']),
            'penalty': float(alm_result['penalty']),
            'rho': alm_result['rho'],
            'sf_surrogate': sf_current,
            'routing_mode': routing_decision.mode.value,
            'scheduler_mode': routing_decision.mode.value,
            'surrogate_feasible': surrogate_feasible,
            'stress_inflation_factor': stress_inflation,
        }
        record['u_max'] = u_max_current
        record['vol_ratio'] = vol_ratio
        # 约束详情
        for name in ['safety', 'volume', 'overhang', 'displacement']:
            record[f'g_{name}'] = float(alm_result.get(f'g_{name}', 0))
            record[f'mu_{name}'] = float(alm_result.get(f'mu_{name}', 0))
        history.append(record)

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save({
                'log_densities': model.log_densities.data.cpu(),
                'repair_cnn': repair_cnn.state_dict(),
                'alm_state': alm.state_dict(),
                'epoch': epoch,
            }, output_path / 'best_joint.pt')

        if epoch % 10 == 0:
            alm_summary = alm.get_summary()
            print(f"  Epoch {epoch}: total={total_loss.item():.4e}, "
                  f"sil={float(l_sil):.4e}, SF_sur={sf_current:.2f}, "
                  f"u_max={u_max_current*1e3:.1f}mm, "
                  f"ρ={alm_summary['rho']:.1f}, "
                  f"mode={routing_decision.mode.value}, "
                  f"feasible={alm.is_feasible()}")

    # ============ 最终保存 ============
    selected_state = None
    selected_label = 'unverified_final'
    if acceptance_metric == 'best_fem_feasible' and best_fem_feasible is not None:
        selected_state = best_fem_feasible
    elif acceptance_metric == 'best_surrogate_feasible' and best_surrogate_feasible is not None:
        selected_state = best_surrogate_feasible
    elif best_fem_feasible is not None:
        selected_state = best_fem_feasible
        selected_label = 'best_fem_feasible'
    elif best_surrogate_feasible is not None:
        selected_state = best_surrogate_feasible
        selected_label = 'best_surrogate_feasible'
    elif best_appearance_feasible is not None:
        selected_state = best_appearance_feasible
        selected_label = 'best_appearance_feasible'

    selected_label_before_final_fem = selected_label
    if selected_state is not None:
        model.log_densities.data.copy_(selected_state['log_densities'].to(device))
        repair_cnn.load_state_dict(selected_state['repair_cnn'])

    with torch.no_grad():
        final_density = model.density.cpu().numpy()
        np.save(output_path / 'final_voxels.npy', final_density)

    final_density_np = np.squeeze(final_density)
    if final_density_np.ndim != 3:
        raise ValueError(f"Expected final density [N,N,N], got {final_density.shape}")
    density_original_np = density_original[0, 0].detach().cpu().numpy()
    vol_ratio_final = float(
        (final_density_np > 0.5).sum()
        / max(float((density_original_np > 0.5).sum()), 1.0)
    )
    iou_final = _compute_projection_iou(final_density_np, density_original_np)

    selected_load_idx = 0
    if selected_state is not None:
        selected_load_idx = int(selected_state.get('load_case_index', 0)) % len(load_cases)
    final_force_np = load_cases[selected_load_idx]['force'].detach().cpu().numpy()
    final_fixed_mask_np = load_cases[selected_load_idx]['fixed_mask'].detach().cpu().numpy()
    try:
        from structural.experiments.evaluate_metrics import evaluate_structural_fem
        final_fem_eval = evaluate_structural_fem(
            final_density_np,
            final_force_np,
            final_fixed_mask_np,
            yield_stress=config.material.yield_stress,
            youngs_modulus=config.material.youngs_modulus,
            poisson_ratio=config.material.poisson_ratio,
        )
    except Exception as exc:
        logger.warning("Final FEM evaluation failed during Stage C export: %s", exc)
        final_fem_eval = {'fem_success': False, 'error': str(exc)}

    fem_safe_final = bool(
        final_fem_eval.get('fem_success', False)
        and float(final_fem_eval.get('sf_min', 0.0)) >= alm_config.safety_factor_target
    )
    if fem_safe_final and selected_label != 'best_fem_feasible':
        # Promote surrogate-feasible selections that pass the final FEM check.
        selected_label = 'best_fem_feasible'

    final_fem_payload = {
        'schema_version': 'stage_c_final_fem_v1',
        'selected_state_before_final_fem': selected_label_before_final_fem,
        'selected_state': selected_label,
        'selected_load_case_index': selected_load_idx,
        'sf_fem_final': float(final_fem_eval.get('sf_min', 0.0)) if final_fem_eval.get('fem_success', False) else None,
        'fem_safe_final': fem_safe_final if final_fem_eval.get('fem_success', False) else None,
        'vol_ratio_final': vol_ratio_final,
        'iou_final': iou_final,
        'vm_max_fem_final': float(final_fem_eval.get('max_vm', 0.0)) if final_fem_eval.get('fem_success', False) else None,
        'u_fem_final': float(final_fem_eval.get('displacement_max', 0.0)) if final_fem_eval.get('fem_success', False) else None,
        'fem_success': bool(final_fem_eval.get('fem_success', False)),
        'error': final_fem_eval.get('error'),
        'summary': {
            'sf_fem_final': float(final_fem_eval.get('sf_min', 0.0)) if final_fem_eval.get('fem_success', False) else None,
            'fem_safe_final': fem_safe_final if final_fem_eval.get('fem_success', False) else None,
            'vol_ratio_final': vol_ratio_final,
            'iou_final': iou_final,
            'sf_fem': float(final_fem_eval.get('sf_min', 0.0)) if final_fem_eval.get('fem_success', False) else None,
            'safe': fem_safe_final if final_fem_eval.get('fem_success', False) else None,
        },
    }
    with open(output_path / 'final_fem_eval.json', 'w') as f:
        json.dump(final_fem_payload, f, indent=2)

    # 最终 Safety Bound 评估
    with torch.no_grad():
        final_d = model.density.unsqueeze(0).to(device)
        final_force = load_cases[selected_load_idx]['force'].unsqueeze(0)
        final_bound = stb.compute_bound(pino, final_d, final_force, epsilon_u)
        safety_bound_log.append({
            'epoch': 'final',
            'sf_pino': final_bound.sf_pino,
            'sf_lower_bound': final_bound.sf_lower_bound,
            'vm_max_pino': final_bound.vm_max_pino,
            'selected_state': selected_label,
            'sf_fem': final_fem_payload['sf_fem_final'],
            'fem_safe': final_fem_payload['fem_safe_final'],
            'vol_ratio_final': vol_ratio_final,
            'iou_final': iou_final,
        })

    # 保存所有日志
    with open(output_path / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    with open(output_path / 'safety_bound_log.json', 'w') as f:
        json.dump(safety_bound_log, f, indent=2)

    with open(output_path / 'alm_state.json', 'w') as f:
        json.dump(alm.get_summary(), f, indent=2)

    routing_stats = grad_router.get_mode_statistics()
    with open(output_path / 'routing_stats.json', 'w') as f:
        json.dump(routing_stats, f, indent=2)
    with open(output_path / 'scheduler_stats.json', 'w') as f:
        json.dump(routing_stats, f, indent=2)

    # 保存 Lyapunov 函数轨迹 (用于论文 Figure)
    lyapunov_summary = grad_router.get_lyapunov_summary()
    with open(output_path / 'lyapunov_summary.json', 'w') as f:
        json.dump(lyapunov_summary, f, indent=2)

    lyapunov_history = grad_router.get_lyapunov_history()
    with open(output_path / 'lyapunov_history.json', 'w') as f:
        json.dump(lyapunov_history, f, indent=2)

    # 保存自适应安全界历史
    adaptive_probe_history = adaptive_stb.get_probe_history()
    with open(output_path / 'adaptive_bound_history.json', 'w') as f:
        json.dump(adaptive_probe_history, f, indent=2)

    print(f"\nStage C complete. Best loss: {best_loss:.4e}")
    print(f"  Final SF (PINO): {final_bound.sf_pino:.3f}")
    print(f"  Final SF lower bound: {final_bound.sf_lower_bound:.3f}")
    if final_fem_payload['sf_fem_final'] is not None:
        print(f"  Final SF (FEM): {final_fem_payload['sf_fem_final']:.3f}")
    print(f"  Selected state: {selected_label}")
    print(f"  ALM feasible: {alm.is_feasible()}")
    print(f"  Routing stats: {routing_stats}")
    print(f"  Lyapunov: V_init={lyapunov_summary.get('V_initial', 0):.4f} → "
          f"V_final={lyapunov_summary.get('V_final', 0):.4f}, "
          f"monotone_rate={lyapunov_summary.get('monotone_decrease_rate', 0):.2%}")


def train_joint(
    shadow_files: list,
    pino_checkpoint: str,
    output_dir: str = './experiments/joint',
    config_path: str | None = None,
    skip_stage_a: bool = False,
    voxel_path: str | None = None,
):
    """完整三阶段训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = StructShadowConfig.from_yaml(config_path)

    print("=" * 80)
    print("StructShadow Joint Training Pipeline")
    print(f"  Resolution: {config.resolution}³")
    print(f"  Device: {device}")
    print(f"  Views: {len(shadow_files)}")
    print("=" * 80)

    # 加载PINO
    pino_result = load_pino(pino_checkpoint, config, device)
    if isinstance(pino_result, tuple):
        pino, pino_epsilon_u = pino_result
    else:
        pino, pino_epsilon_u = pino_result, None
    print(f"PINO loaded from {pino_checkpoint}")

    # Stage A
    if skip_stage_a and voxel_path:
        print(f"Skipping Stage A, using provided voxels: {voxel_path}")
    else:
        voxel_path = run_stage_a(
            shadow_files=shadow_files,
            output_dir=output_dir,
            voxel_res=config.resolution,
            device=device,
        )

    # 验证 Stage A
    if not Path(voxel_path).exists():
        raise FileNotFoundError(f"Stage A voxel not found: {voxel_path}")
    voxel_check = np.load(voxel_path)
    if voxel_check.ndim == 4:
        voxel_check = voxel_check[0]
    solid_ratio = (voxel_check > 0.5).mean()
    print(f"\n[Validation] Stage A: solid_ratio={solid_ratio:.3f}, shape={voxel_check.shape}")
    if solid_ratio < 0.01:
        logger.warning("Stage A produced nearly empty voxels!")

    # Stage B
    repair_cnn = run_stage_b(
        voxel_path=voxel_path,
        pino=pino,
        config=config,
        output_dir=output_dir,
        device=device,
    )

    # 验证 Stage B
    stage_b_path = Path(output_dir) / "stage_b" / "repaired_voxels.npy"
    if stage_b_path.exists():
        repaired = np.load(str(stage_b_path)).squeeze()
        rep_ratio = (repaired > 0.5).mean()
        vol_change = rep_ratio / max(solid_ratio, 1e-8)
        print(f"[Validation] Stage B: solid_ratio={rep_ratio:.3f}, "
              f"volume_change={vol_change:.2f}x")
    else:
        logger.warning("Stage B did not produce repaired_voxels.npy")

    # Stage C (with ALM + Safety Bound + Gradient Router)
    run_stage_c(
        shadow_files=shadow_files,
        voxel_path=voxel_path,
        pino=pino,
        repair_cnn=repair_cnn,
        config=config,
        output_dir=output_dir,
        device=device,
        pino_epsilon_u=pino_epsilon_u,
    )

    # 验证 Stage C
    stage_c_path = Path(output_dir) / "stage_c" / "final_voxels.npy"
    if stage_c_path.exists():
        final = np.load(str(stage_c_path)).squeeze()
        final_ratio = (final > 0.5).mean()
        print(f"[Validation] Stage C: solid_ratio={final_ratio:.3f}")
    else:
        logger.warning("Stage C did not produce final_voxels.npy")

    print("\n" + "=" * 80)
    print("All stages complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StructShadow Joint Training')
    parser.add_argument('--shadow_files', nargs='+', required=True, help='Input silhouette images')
    parser.add_argument('--pino_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./experiments/joint')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--skip_stage_a', action='store_true')
    parser.add_argument('--voxel_path', type=str, default=None, help='Pre-computed voxel .npy for skipping Stage A')
    args = parser.parse_args()

    train_joint(
        shadow_files=args.shadow_files,
        pino_checkpoint=args.pino_checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        skip_stage_a=args.skip_stage_a,
        voxel_path=args.voxel_path,
    )
