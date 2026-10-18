"""
PINO预训练脚本 v2 — 改进版 (Stage 0)

相比v1的主要改进:
1. Von Mises辅助损失: 直接监督Von Mises应力的相对误差
2. 通道加权数据损失: 应力通道权重3x, 位移通道权重1x
3. Log-cosh损失: 对离群值更鲁棒, 梯度更稳定
4. 数据增强: 随机坐标轴翻转 (利用物理对称性)
5. 更长训练: 500 epochs, warmup + cosine annealing
6. 混合精度训练: 加速训练, 允许更大batch或更多epochs
7. 梯度累积: 有效batch_size = batch_size * accum_steps

用法:
    python pretrain_pino_v2.py --data_dir ./fem_dataset --epochs 500
"""
import sys
import argparse
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as TF
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from structural.models.pino import PINO3d, PINOLoss
from structural.configs import StructShadowConfig


# ─────────────────────────────────────────────────────────────────────────────
# Von Mises computation utility
# ─────────────────────────────────────────────────────────────────────────────

def compute_von_mises(stress: torch.Tensor) -> torch.Tensor:
    """Compute Von Mises stress from 6-channel Voigt stress tensor.

    Args:
        stress: [B, 6, N, N, N] — [σxx, σyy, σzz, σyz, σzx, σxy]
    Returns:
        [B, 1, N, N, N] — Von Mises stress (scalar field)
    """
    sxx = stress[:, 0:1]
    syy = stress[:, 1:2]
    szz = stress[:, 2:3]
    syz = stress[:, 3:4]
    szx = stress[:, 4:5]
    sxy = stress[:, 5:6]

    # σ_vm = sqrt(0.5 * [(σxx-σyy)² + (σyy-σzz)² + (σzz-σxx)²] + 3*(σxy² + σyz² + σzx²))
    vm_sq = (0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
             + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))
    # Clamp to avoid sqrt(0) gradient issues
    return torch.sqrt(vm_sq.clamp(min=1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# Improved Loss Function
# ─────────────────────────────────────────────────────────────────────────────

class PINOLossV2(nn.Module):
    """Improved PINO loss with Von Mises auxiliary loss and channel weighting.

    Components:
    1. Channel-weighted log-cosh data loss (stress channels weighted 3x)
    2. Von Mises relative L1 loss (directly targets the metric we care about)
    3. Von Mises peak loss (penalizes underestimation of max VM stress per sample)
    4. BC loss (Dirichlet)
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_vm: float = 2.0,
        lambda_vm_peak: float = 0.5,
        lambda_bc: float = 1.0,
        disp_weight: float = 1.0,
        stress_weight: float = 3.0,
        stress_scale: float = 3146304.25,
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_vm = lambda_vm
        self.lambda_vm_peak = lambda_vm_peak
        self.lambda_bc = lambda_bc
        self.stress_scale = stress_scale

        # Channel weights: [ux, uy, uz, sxx, syy, szz, syz, szx, sxy]
        weights = torch.tensor(
            [disp_weight] * 3 + [stress_weight] * 6,
            dtype=torch.float32
        )
        # Normalize so mean weight = 1
        weights = weights / weights.mean()
        # Shape for broadcasting: [1, 9, 1, 1, 1]
        self.register_buffer('channel_weights', weights.view(1, 9, 1, 1, 1))

    def log_cosh_loss(self, pred: torch.Tensor, target: torch.Tensor,
                      weights: torch.Tensor | None = None,
                      mask: torch.Tensor | None = None) -> torch.Tensor:
        """Log-cosh loss: smooth approximation to L1, differentiable at 0.

        log(cosh(x)) ≈ 0.5*x² for small x, |x| - log(2) for large x.
        More robust to outliers than MSE while still smooth.
        """
        diff = pred - target
        if weights is not None:
            diff = diff * weights.sqrt()  # Apply sqrt so that squared effect matches weighting
        # Numerically stable: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
        loss = torch.abs(diff) + TF.softplus(-2.0 * torch.abs(diff)) - math.log(2.0)
        if mask is not None:
            loss = loss * mask
        return loss.mean()

    def von_mises_loss(
        self,
        pred_stress: torch.Tensor,
        target_stress: torch.Tensor,
        density: torch.Tensor,
        target_vm: torch.Tensor | None = None,
    ) -> dict:
        """Von Mises focused losses.

        Args:
            pred_stress: [B, 6, N, N, N] predicted stress (normalized)
            target_stress: [B, 6, N, N, N] target stress (normalized)
            density: [B, 1, N, N, N]
            target_vm: [B, 1, N, N, N] pre-computed GT Von Mises (in physical units, optional)
        Returns:
            dict with 'vm_relative' and 'vm_peak' losses
        """
        mask = (density > 0.1).float()

        pred_vm = compute_von_mises(pred_stress)      # normalized space
        target_vm_norm = compute_von_mises(target_stress)  # normalized space

        # Relative L1 loss on Von Mises: |pred_vm - target_vm| / (target_vm + eps)
        # This directly optimizes relative error, which is our evaluation metric
        eps = 1e-6
        rel_error = torch.abs(pred_vm - target_vm_norm) / (target_vm_norm + eps)
        vm_relative = (rel_error * mask).sum() / mask.sum().clamp(min=1)

        # Peak VM loss: penalize error in max VM stress per sample
        # This targets the safety factor computation directly
        B = pred_stress.shape[0]
        vm_peak_loss = torch.tensor(0.0, device=pred_stress.device)
        for i in range(B):
            m = mask[i] > 0.5
            if m.any():
                pred_max = pred_vm[i][m].max()
                target_max = target_vm_norm[i][m].max()
                # Asymmetric: penalize underestimation more (safety-critical)
                error = pred_max - target_max
                # Underestimation penalty: 2x weight
                if error < 0:
                    vm_peak_loss = vm_peak_loss + 2.0 * error ** 2
                else:
                    vm_peak_loss = vm_peak_loss + error ** 2
        vm_peak_loss = vm_peak_loss / max(B, 1)

        return {
            'vm_relative': vm_relative,
            'vm_peak': vm_peak_loss,
        }

    def bc_loss(
        self,
        displacement: torch.Tensor,
        fixed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Dirichlet BC: displacement = 0 on fixed nodes."""
        return ((displacement * fixed_mask) ** 2).mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        force: torch.Tensor,
        density: torch.Tensor,
        fixed_mask: torch.Tensor,
        target_vm: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            pred:       [B, 9, N, N, N] — PINO output
            target:     [B, 9, N, N, N] — FEM ground truth (normalized)
            force:      [B, 3, N, N, N] — force field (normalized)
            density:    [B, 1, N, N, N] — density field
            fixed_mask: [B, 1, N, N, N] — fixed DOF mask
            target_vm:  [B, 1, N, N, N] — GT Von Mises stress (optional, physical units)
        """
        pred_disp = pred[:, :3]
        pred_stress = pred[:, 3:]
        target_stress = target[:, 3:]

        mask = (density > 0.1).float()
        # Expand mask to 9 channels
        mask_9 = mask.expand_as(pred)

        # 1. Channel-weighted log-cosh data loss
        data_loss = self.log_cosh_loss(pred, target, self.channel_weights, mask_9)

        # 2. Von Mises losses
        vm_losses = self.von_mises_loss(pred_stress, target_stress, density, target_vm)

        # 3. BC loss
        boundary_loss = self.bc_loss(pred_disp, fixed_mask)

        total = (self.lambda_data * data_loss
                 + self.lambda_vm * vm_losses['vm_relative']
                 + self.lambda_vm_peak * vm_losses['vm_peak']
                 + self.lambda_bc * boundary_loss)

        return {
            'total': total,
            'data': data_loss,
            'vm_relative': vm_losses['vm_relative'],
            'vm_peak': vm_losses['vm_peak'],
            'bc': boundary_loss,
            # Keep these keys for compatibility with validation logging
            'pde': torch.tensor(0.0),
            'equilibrium': torch.tensor(0.0),
            'constitutive': torch.tensor(0.0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset with data augmentation
# ─────────────────────────────────────────────────────────────────────────────

class FEMDatasetV2(Dataset):
    """FEM dataset with data augmentation (random axis flips).

    Physics-consistent augmentation:
    - Flipping along axis k negates the k-th component of displacement and force,
      and negates shear stress components involving that axis.

    Stress Voigt ordering: [σxx, σyy, σzz, σyz, σzx, σxy]
    - Flip X: negate ux, Fx, σzx, σxy  (indices involving x)
    - Flip Y: negate uy, Fy, σyz, σxy  (indices involving y)
    - Flip Z: negate uz, Fz, σyz, σzx  (indices involving z)
    """

    FORCE_SCALE = 12164.43
    DISP_SCALE = 0.00997422
    STRESS_SCALE = 3146304.25

    def __init__(self, data_dir: str, split: str = 'train', train_ratio: float = 0.85,
                 augment: bool = False):
        self.data_dir = Path(data_dir)
        files = sorted(self.data_dir.glob("sample_*.npz"))

        n_train = int(len(files) * train_ratio)
        if split == 'train':
            self.files = files[:n_train]
        else:
            self.files = files[n_train:]

        self.augment = augment and (split == 'train')
        print(f"FEMDatasetV2 [{split}]: {len(self.files)} samples, augment={self.augment}")

    def __len__(self):
        return len(self.files)

    def _apply_flip(self, density, force, displacement, stress, von_mises, fixed_mask, axis):
        """Apply physics-consistent flip along spatial axis.

        Args:
            axis: 0 (X), 1 (Y), or 2 (Z) — spatial dimension to flip
            All tensors have spatial dims at indices [1,2,3] (after channel dim)
        """
        spatial_dim = axis + 1  # +1 because dim 0 is channel

        density = density.flip(spatial_dim)
        fixed_mask = fixed_mask.flip(spatial_dim)
        von_mises = von_mises.flip(spatial_dim)

        force = force.flip(spatial_dim)
        displacement = displacement.flip(spatial_dim)
        stress = stress.flip(spatial_dim)

        # Negate the axis-aligned component of force and displacement
        force[axis] = -force[axis]
        displacement[axis] = -displacement[axis]

        # Negate shear stress components involving the flipped axis
        # Voigt: [σxx(0), σyy(1), σzz(2), σyz(3), σzx(4), σxy(5)]
        if axis == 0:  # flip X: negate σzx(4), σxy(5)
            stress[4] = -stress[4]
            stress[5] = -stress[5]
        elif axis == 1:  # flip Y: negate σyz(3), σxy(5)
            stress[3] = -stress[3]
            stress[5] = -stress[5]
        elif axis == 2:  # flip Z: negate σyz(3), σzx(4)
            stress[3] = -stress[3]
            stress[4] = -stress[4]

        return density, force, displacement, stress, von_mises, fixed_mask

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        density = torch.from_numpy(data['density']).float().unsqueeze(0)       # [1, N, N, N]
        force = torch.from_numpy(data['force']).float()                         # [3, N, N, N]
        fixed_mask = torch.from_numpy(data['fixed_mask']).float()               # [1, N, N, N]
        displacement = torch.from_numpy(data['displacement']).float()           # [3, N, N, N]
        stress = torch.from_numpy(data['stress']).float()                       # [6, N, N, N]

        # Von Mises ground truth (in physical units)
        if 'von_mises' in data:
            von_mises = torch.from_numpy(data['von_mises']).float().unsqueeze(0)  # [1, N, N, N]
        else:
            von_mises = torch.zeros(1, *density.shape[1:])

        # Data augmentation: random axis flips
        if self.augment:
            for axis in range(3):
                if torch.rand(1).item() > 0.5:
                    density, force, displacement, stress, von_mises, fixed_mask = \
                        self._apply_flip(density, force, displacement, stress,
                                         von_mises, fixed_mask, axis)

        # Global normalization
        force_norm = force / self.FORCE_SCALE
        displacement_norm = displacement / self.DISP_SCALE
        stress_norm = stress / self.STRESS_SCALE

        # Input: [rho, Fx_norm, Fy_norm, Fz_norm]
        x = torch.cat([density, force_norm], dim=0)                             # [4, N, N, N]
        # Target: [ux_norm, uy_norm, uz_norm, sxx_norm, ...]
        y = torch.cat([displacement_norm, stress_norm], dim=0)                  # [9, N, N, N]

        # Normalize Von Mises with same stress scale
        vm_norm = von_mises / self.STRESS_SCALE                                 # [1, N, N, N]

        return x, y, force_norm, density, fixed_mask, vm_norm


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                accum_steps=1):
    model.train()
    total_loss = {
        'total': 0, 'data': 0, 'vm_relative': 0, 'vm_peak': 0, 'bc': 0,
        'pde': 0, 'equilibrium': 0, 'constitutive': 0,
    }
    n_batches = 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        x, y, force, density, fixed_mask, vm_gt = [b.to(device) for b in batch]

        with autocast(device_type='cuda', enabled=scaler.is_enabled()):
            pred = model(x)
            losses = criterion(pred, y, force, density, fixed_mask, vm_gt)
            loss = losses['total'] / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        for k in total_loss:
            if k in losses:
                val = losses[k]
                total_loss[k] += val.item() if torch.is_tensor(val) else val
        n_batches += 1

        pbar.set_postfix(
            loss=f"{losses['total'].item():.4e}",
            data=f"{losses['data'].item():.4e}",
            vm=f"{losses['vm_relative'].item():.4e}",
        )

    return {k: v / max(n_batches, 1) for k, v in total_loss.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = {
        'total': 0, 'data': 0, 'vm_relative': 0, 'vm_peak': 0, 'bc': 0,
        'pde': 0, 'equilibrium': 0, 'constitutive': 0,
    }
    n_batches = 0

    # Also track actual Von Mises relative errors for monitoring
    all_vm_rel_errors = []

    for batch in loader:
        x, y, force, density, fixed_mask, vm_gt = [b.to(device) for b in batch]

        with autocast(device_type='cuda', enabled=True):
            pred = model(x)

        # Compute loss in fp32
        losses = criterion(pred.float(), y, force, density, fixed_mask, vm_gt)

        for k in total_loss:
            if k in losses:
                val = losses[k]
                total_loss[k] += val.item() if torch.is_tensor(val) else val
        n_batches += 1

        # Compute actual VM relative max error (the metric we care about)
        pred_stress = pred[:, 3:].float()
        target_stress = y[:, 3:]
        mask = (density > 0.1).float()
        pred_vm = compute_von_mises(pred_stress)
        target_vm = compute_von_mises(target_stress)

        B = pred.shape[0]
        for i in range(B):
            m = mask[i, 0] > 0.5
            if m.any():
                pred_max = pred_vm[i, 0][m].max().item()
                target_max = target_vm[i, 0][m].max().item()
                if target_max > 1e-8:
                    rel_err = abs(pred_max - target_max) / target_max
                    all_vm_rel_errors.append(rel_err)

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_loss.items()}

    # Add VM max relative error stats
    if all_vm_rel_errors:
        vm_arr = np.array(all_vm_rel_errors)
        avg_losses['vm_max_rel_mean'] = float(vm_arr.mean())
        avg_losses['vm_max_rel_median'] = float(np.median(vm_arr))
        avg_losses['vm_max_rel_p90'] = float(np.percentile(vm_arr, 90))

    return avg_losses


def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_pino_v2(
    data_dir: str,
    output_dir: str = './checkpoints/pino_v2',
    epochs: int = 500,
    batch_size: int = 4,
    lr: float = 5e-4,
    warmup_epochs: int = 20,
    accum_steps: int = 2,
    resume: bool = False,
    resume_from_v1: str | None = None,
    no_early_stop: bool = False,
    config: StructShadowConfig | None = None,
    use_amp: bool = True,
    # Loss weights
    lambda_data: float = 1.0,
    lambda_vm: float = 2.0,
    lambda_vm_peak: float = 0.5,
    lambda_bc: float = 1.0,
):
    """PINO v2 pretraining with Von Mises focused loss."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Settings: epochs={epochs}, batch_size={batch_size}, lr={lr}, "
          f"accum_steps={accum_steps}, effective_batch={batch_size*accum_steps}")
    print(f"Loss weights: data={lambda_data}, vm={lambda_vm}, "
          f"vm_peak={lambda_vm_peak}, bc={lambda_bc}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = StructShadowConfig.from_yaml()

    # Data
    train_set = FEMDatasetV2(data_dir, split='train', augment=True)
    val_set = FEMDatasetV2(data_dir, split='val', augment=False)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model (same architecture as v1, loaded from config)
    model = PINO3d(
        in_channels=config.pino.in_channels,
        out_channels=config.pino.out_channels,
        modes=config.pino.modes,
        width=config.pino.width,
        n_layers=config.pino.n_layers,
        padding=config.pino.padding,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PINO parameters: {n_params:,}")

    # Optionally initialize from v1 checkpoint (fine-tune)
    if resume_from_v1:
        v1_ckpt = torch.load(resume_from_v1, map_location=device, weights_only=False)
        model.load_state_dict(v1_ckpt['model_state_dict'])
        print(f"  Loaded v1 weights from {resume_from_v1}")

    # Loss
    criterion = PINOLossV2(
        lambda_data=lambda_data,
        lambda_vm=lambda_vm,
        lambda_vm_peak=lambda_vm_peak,
        lambda_bc=lambda_bc,
        stress_scale=FEMDatasetV2.STRESS_SCALE,
    ).to(device)

    # Optimizer: AdamW with lower LR for fine-tuning stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_epochs, epochs)

    # Mixed precision
    scaler = GradScaler(enabled=(use_amp and device.type == 'cuda'))

    # Resume
    start_epoch = 1
    history = []
    best_val_loss = float('inf')
    best_vm_metric = float('inf')
    no_improve_count = 0

    if resume:
        ckpt_path = output_path / 'best_model.pt'
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            best_val_loss = ckpt.get('val_loss', float('inf'))
            best_vm_metric = ckpt.get('vm_max_rel_mean', float('inf'))
            start_epoch = ckpt.get('epoch', 0) + 1
            for _ in range(start_epoch - 1):
                scheduler.step()
            hist_path = output_path / 'training_history.json'
            if hist_path.exists():
                with open(hist_path) as f:
                    history = json.load(f)
            print(f"  Resumed from epoch {start_epoch - 1}, "
                  f"best_val_loss={best_val_loss:.4e}, best_vm={best_vm_metric:.4f}")
        else:
            print(f"  No checkpoint found at {ckpt_path}, training from scratch")

    patience = 50  # Longer patience for longer training

    for epoch in range(start_epoch, epochs + 1):
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, accum_steps=accum_steps
        )
        val_losses = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr_current = optimizer.param_groups[0]['lr']

        vm_mean = val_losses.get('vm_max_rel_mean', float('inf'))
        vm_median = val_losses.get('vm_max_rel_median', float('inf'))
        vm_p90 = val_losses.get('vm_max_rel_p90', float('inf'))

        history.append({
            'epoch': epoch,
            'lr': lr_current,
            'train': train_losses,
            'val': val_losses,
        })

        print(f"  Train: {train_losses['total']:.4e} | Val: {val_losses['total']:.4e} | "
              f"VM rel: mean={vm_mean:.4f} median={vm_median:.4f} p90={vm_p90:.4f} | "
              f"LR: {lr_current:.2e}")

        # Save best — use VM mean relative error as primary metric
        # Fall back to total loss if VM metric not available
        is_best = False
        if vm_mean < best_vm_metric:
            best_vm_metric = vm_mean
            is_best = True
        elif vm_mean == float('inf') and val_losses['total'] < best_val_loss:
            is_best = True

        if is_best:
            best_val_loss = val_losses['total']
            no_improve_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'vm_max_rel_mean': vm_mean,
                'vm_max_rel_median': vm_median,
                'vm_max_rel_p90': vm_p90,
                'normalization_mode': 'global',
                'normalization': {
                    'force_scale': FEMDatasetV2.FORCE_SCALE,
                    'disp_scale': FEMDatasetV2.DISP_SCALE,
                    'stress_scale': FEMDatasetV2.STRESS_SCALE,
                },
                'config': {
                    'modes': config.pino.modes,
                    'width': config.pino.width,
                    'n_layers': config.pino.n_layers,
                    'in_channels': config.pino.in_channels,
                    'out_channels': config.pino.out_channels,
                    'padding': config.pino.padding,
                },
                'loss_config': {
                    'lambda_data': lambda_data,
                    'lambda_vm': lambda_vm,
                    'lambda_vm_peak': lambda_vm_peak,
                    'lambda_bc': lambda_bc,
                },
                'version': 'v2',
            }, output_path / 'best_model.pt')
            print(f"  -> New best model (VM rel mean={vm_mean:.4f}, val_loss={best_val_loss:.4e})")
        else:
            no_improve_count += 1

        # Periodic save
        if epoch % 50 == 0:
            torch.save(model.state_dict(), output_path / f'epoch_{epoch}.pt')

        # Early stopping
        if not no_early_stop and no_improve_count >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Save history
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nPINO v2 pretraining complete.")
    print(f"  Best val loss: {best_val_loss:.4e}")
    print(f"  Best VM max rel error (mean): {best_vm_metric:.4f}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain PINO v2 with Von Mises loss')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to FEM dataset directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/pino_v2',
                        help='Output checkpoint directory')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Total training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU (default: 4)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Peak learning rate (default: 5e-4)')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='LR warmup epochs (default: 20)')
    parser.add_argument('--accum_steps', type=int, default=2,
                        help='Gradient accumulation steps (default: 2, effective batch=8)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from v2 checkpoint')
    parser.add_argument('--resume_from_v1', type=str, default=None,
                        help='Initialize from v1 checkpoint path (fine-tune)')
    parser.add_argument('--no_early_stop', action='store_true',
                        help='Disable early stopping')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision training')
    # Loss weights
    parser.add_argument('--lambda_data', type=float, default=1.0)
    parser.add_argument('--lambda_vm', type=float, default=2.0)
    parser.add_argument('--lambda_vm_peak', type=float, default=0.5)
    parser.add_argument('--lambda_bc', type=float, default=1.0)

    args = parser.parse_args()

    train_pino_v2(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        accum_steps=args.accum_steps,
        resume=args.resume,
        resume_from_v1=args.resume_from_v1,
        no_early_stop=args.no_early_stop,
        use_amp=not args.no_amp,
        lambda_data=args.lambda_data,
        lambda_vm=args.lambda_vm,
        lambda_vm_peak=args.lambda_vm_peak,
        lambda_bc=args.lambda_bc,
    )
