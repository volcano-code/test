"""
修复优化训练 (Stage B)

冻结体素密度，只优化RepairCNN:
- 输入: 已训练好的ShadowArt形状 + 预训练PINO
- 输出: 训练好的RepairCNN权重

损失:
    L = λ_str·L_structural + λ_vol·L_volume + λ_shape·L_shape
      + λ_smooth·L_smooth + λ_print·L_printable
"""
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from structural.models.load_model import load_surrogate, forward_surrogate_physical
from structural.models.repair_cnn import RepairCNN
from structural.analysis.stress_eval import (
    compute_von_mises_torch, pino_stress_to_physical,
    get_yield_stress_for_model,
    structural_loss, volume_loss,
    shape_preservation_loss, smoothness_loss, overhang_penalty,
    compute_protected_shell_mask, material_redistribution_loss,
    repair_locality_loss,
)
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.vulnerability import VulnerabilityDetector
from structural.infill.infill_guidance import InfillGuidanceGenerator
from structural.configs import StructShadowConfig

import logging
logger = logging.getLogger(__name__)


def train_repair(
    voxel_path: str,
    pino_checkpoint: str,
    output_dir: str = './checkpoints/repair',
    config: StructShadowConfig | None = None,
):
    """修复网络训练

    Args:
        voxel_path: ShadowArt训练好的体素 .npy 文件
        pino_checkpoint: 预训练PINO权重路径
        output_dir: 输出目录
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = StructShadowConfig.from_yaml()

    # 加载体素
    voxel_data = np.load(voxel_path)
    if voxel_data.ndim == 4:
        voxel_data = voxel_data[0]  # Remove batch dim if present

    N = voxel_data.shape[0]
    print(f"Loaded voxels: {voxel_data.shape}, solid ratio: {(voxel_data > 0.5).mean():.3f}")

    density_original = torch.from_numpy(voxel_data).float().unsqueeze(0).unsqueeze(0).to(device)
    # [1, 1, N, N, N]

    # 加载PINO/UFNO (自动检测架构)
    pino, ckpt = load_surrogate(pino_checkpoint, device)
    for p in pino.parameters():
        p.requires_grad = False

    # 修复CNN
    repair_cnn = RepairCNN(
        in_channels=3,
        channels=config.repair.channels,
        delta_max=config.repair.delta_max,
    ).to(device)

    optimizer = torch.optim.Adam(repair_cnn.parameters(), lr=config.repair.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.repair.steps
    )

    # 载荷生成
    load_gen = LoadGenerator(
        material_density=config.material.density,
    )

    # 预生成多种载荷
    density_np = voxel_data
    load_cases = []
    rng = np.random.default_rng(42)
    for _ in range(8):
        load_dict = load_gen.generate_random_load(density_np, rng)
        load_cases.append(load_gen.to_torch(load_dict, device))

    # 生成填充先验 (用第一个载荷工况)
    print("Generating infill guidance...")
    with torch.no_grad():
        first_load = load_cases[0]
        pino_fields = forward_surrogate_physical(pino, density_original, first_load['force'].unsqueeze(0))
        vm_init = pino_fields['von_mises']

    infill_gen = InfillGuidanceGenerator()
    infill_guidance = infill_gen.generate(
        density_np,
        vm_init[0, 0].cpu().numpy(),
    )
    infill_guidance_t = torch.from_numpy(infill_guidance).float().unsqueeze(0).unsqueeze(0).to(device)

    # 脆弱性检测器
    effective_yield = get_yield_stress_for_model(pino, config.material.yield_stress)
    vuln_detector = VulnerabilityDetector(yield_stress=effective_yield)
    print(f"Infill guidance: mean={infill_guidance.mean():.4f}, max={infill_guidance.max():.4f}")

    # 训练循环
    rc = config.repair
    # Protect a thin boundary shell to preserve the outer contour / silhouette.
    protected_shell = compute_protected_shell_mask(
        density_original, shell_width=rc.shell_width
    )
    interior_mask = ((density_original > 0.5).float() - protected_shell).clamp(0, 1)
    history = []
    best_loss = float('inf')

    print(f"\nStarting repair optimization ({rc.steps} steps)...")
    pbar = tqdm(range(1, rc.steps + 1))

    for step in pbar:
        repair_cnn.train()

        # 随机选择载荷工况
        load_idx = step % len(load_cases)
        load = load_cases[load_idx]
        force = load['force'].unsqueeze(0)       # [1, 3, N, N, N]
        fixed_mask = load['fixed_mask'].unsqueeze(0)  # [1, 1, N, N, N]

        # 第一步: 用原始密度做初始分析，获取修复输入
        with torch.no_grad():
            pino_fields = forward_surrogate_physical(pino, density_original, force)
        vm = pino_fields['von_mises']
        vm_norm = vm / (vm.max() + 1e-8)

        # 使用 VulnerabilityDetector 生成脆弱图（每10步更新一次以节省计算）
        if step == 1 or step % 10 == 0:
            vm_np = vm[0, 0].detach().cpu().numpy()
            density_np_current = density_original[0, 0].detach().cpu().numpy()
            vuln_result = vuln_detector.detect(density_np_current, vm_np)
            vuln_map = torch.from_numpy(
                vuln_result['vulnerability_map']
            ).float().unsqueeze(0).unsqueeze(0).to(device)
            # 归一化到 [0, 1]
            vuln_map = vuln_map / (vuln_map.max() + 1e-8)

        # 修复: CNN 输出修复场
        repair_input = torch.cat([density_original, vm_norm, vuln_map], dim=1)
        repair_fields = repair_cnn(repair_input)

        density_repaired = repair_cnn.apply_repair(density_original, repair_fields, interior_mask)

        high_priority = torch.maximum(vm_norm, vuln_map)
        high_priority = torch.maximum(high_priority, infill_guidance_t)
        high_priority_mask = ((high_priority > rc.high_stress_threshold).float() * interior_mask).clamp(0, 1)
        low_priority_mask = (
            (vm_norm < rc.low_stress_threshold).float()
            * (1.0 - high_priority_mask)
            * interior_mask
        ).clamp(0, 1)

        # 核心修复: 用修复后密度跑 PINO，梯度反传到 RepairCNN
        # 这确保修复网络能"看到"自己修复的效果
        pino_fields_new = forward_surrogate_physical(pino, density_repaired, force)
        vm_new = pino_fields_new['von_mises']

        # 损失
        l_structural = structural_loss(vm_new, density_repaired, effective_yield)
        l_volume = volume_loss(density_repaired, density_original, rc.volume_budget)
        l_shape = shape_preservation_loss(density_repaired, density_original)
        l_smooth = smoothness_loss(repair_fields['combined'])
        l_printable = overhang_penalty(density_repaired)
        l_redistribution = material_redistribution_loss(
            density_repaired,
            density_original,
            interior_mask,
            max_net_increase_ratio=rc.max_net_material_ratio,
        )
        l_locality = repair_locality_loss(
            density_repaired,
            density_original,
            high_priority_mask,
            low_priority_mask,
            interior_mask,
        )

        # 填充先验监督 (指数衰减: 前期强引导 → 后期自由优化)
        infill_weight = 5.0 * (0.97 ** step)
        l_infill_prior = ((repair_fields['infill'] - infill_guidance_t * interior_mask) ** 2).mean()

        total_loss = (
            rc.lambda_structural * l_structural
            + rc.lambda_volume * l_volume
            + rc.lambda_shape * l_shape
            + rc.lambda_smooth * l_smooth
            + rc.lambda_printable * l_printable
            + rc.lambda_redistribution * l_redistribution
            + rc.lambda_locality * l_locality
            + infill_weight * l_infill_prior
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(repair_cnn.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # 记录
        record = {
            'step': step,
            'total': total_loss.item(),
            'structural': l_structural.item(),
            'volume': l_volume.item(),
            'shape': l_shape.item(),
            'smooth': l_smooth.item(),
            'printable': l_printable.item(),
            'redistribution': l_redistribution.item(),
            'locality': l_locality.item(),
            'infill_prior': l_infill_prior.item(),
            'max_vm': vm_new.max().item(),
        }
        history.append(record)

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save({
                'step': step,
                'model_state_dict': repair_cnn.state_dict(),
                'loss': best_loss,
            }, output_path / 'best_repair.pt')

        pbar.set_postfix(
            loss=f"{total_loss.item():.4e}",
            struct=f"{l_structural.item():.4e}",
            redist=f"{l_redistribution.item():.3e}",
            maxvm=f"{vm_new.max().item():.2e}",
        )

    # 保存最终模型和历史
    torch.save(repair_cnn.state_dict(), output_path / 'final_repair.pt')

    import json
    with open(output_path / 'repair_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 保存修复后的体素 (用所有载荷工况评估后取最优)
    with torch.no_grad():
        # 用第一个载荷做最终评估
        eval_load = load_cases[0]
        eval_force = eval_load['force'].unsqueeze(0)
        pino_eval = forward_surrogate_physical(pino, density_original, eval_force)
        vm_eval = pino_eval['von_mises']
        vm_eval_norm = vm_eval / (vm_eval.max() + 1e-8)

        vuln_eval = vuln_detector.detect(
            density_original[0, 0].cpu().numpy(),
            vm_eval[0, 0].cpu().numpy()
        )
        vuln_eval_map = torch.from_numpy(
            vuln_eval['vulnerability_map']
        ).float().unsqueeze(0).unsqueeze(0).to(device)
        vuln_eval_map = vuln_eval_map / (vuln_eval_map.max() + 1e-8)

        final_input = torch.cat([density_original, vm_eval_norm, vuln_eval_map], dim=1)
        final_fields = repair_cnn(final_input)
        final_shell = compute_protected_shell_mask(density_original, shell_width=rc.shell_width)
        final_interior = ((density_original > 0.5).float() - final_shell).clamp(0, 1)
        final_density = repair_cnn.apply_repair(density_original, final_fields, final_interior)
        np.save(output_path / 'repaired_voxels.npy', final_density.cpu().numpy())

        # 最终应力评估
        pino_final = forward_surrogate_physical(pino, final_density, eval_force)
        vm_final = pino_final['von_mises']

    print(f"\nRepair training complete. Best loss: {best_loss:.4e}")
    print(f"Max VM stress: initial={vm_eval.max().item():.2e} → final={vm_final.max().item():.2e}")

    return repair_cnn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train repair CNN')
    parser.add_argument('--voxel_path', type=str, required=True, help='Path to ShadowArt voxel .npy')
    parser.add_argument('--pino_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints/repair')
    args = parser.parse_args()

    train_repair(
        voxel_path=args.voxel_path,
        pino_checkpoint=args.pino_checkpoint,
        output_dir=args.output_dir,
    )
