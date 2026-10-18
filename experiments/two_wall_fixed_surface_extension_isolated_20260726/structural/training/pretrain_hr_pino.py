"""
HR-PINO 训练脚本 — 基于 PINO v3 的高分辨率 256³ 训练

训练策略:
1. Transfer learning: 从 PINO v3 checkpoint 加载 F-FNO core 权重
2. 差异学习率: core (pretrained) 1e-5, encoder/decoder (random init) 1e-4
3. 渐进式分辨率训练 (可选):
   Phase 1: 64³ 原始数据直接训练 (warm-up encoder/decoder)
   Phase 2: 128³ 上采样数据
   Phase 3: 256³ 原生数据 (需要 256³ FEM data)
4. Gradient checkpointing 默认开启以控制显存
5. 复用 v3 的 log-scale stress + asymmetric safety loss

用法:
    # Phase 1: 在 64³ 数据上 warm-up
    python pretrain_hr_pino.py \
        --data_dirs fem_dataset,fem_dataset_v2 \
        --pino_checkpoint checkpoints/pino_v3/best_model.pt \
        --output_dir checkpoints/hr_pino \
        --resolution 64 --epochs 100 --batch_size 4

    # Phase 2: 在 128³ 数据上过渡
    python pretrain_hr_pino.py \
        --data_dirs fem_dataset,fem_dataset_v2 \
        --resume_from checkpoints/hr_pino/best_model.pt \
        --output_dir checkpoints/hr_pino \
        --resolution 128 --epochs 100 --batch_size 2

    # Phase 3: 在 256³ 原生数据上精调
    python pretrain_hr_pino.py \
        --data_dirs fem_dataset_256 \
        --resume_from checkpoints/hr_pino/best_model.pt \
        --output_dir checkpoints/hr_pino_256 \
        --resolution 256 --epochs 200 --batch_size 1
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from structural.models.hr_pino import HRPINO3d, count_parameters
from structural.training.pretrain_pino_v3 import (
    PINOLossV3, stress_to_log, log_to_stress, compute_von_mises, collate_v3,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: Multi-resolution FEM data (upsample to target resolution)
# ─────────────────────────────────────────────────────────────────────────────

class FEMDatasetHR(Dataset):
    """FEM dataset that upsamples 64³ data to target resolution.

    For 64³→256³: trilinear interpolation on density/force/displacement/stress.
    For native 256³ data: loaded directly.
    """

    FORCE_SCALE = 12164.43
    DISP_SCALE = 0.00997422

    def __init__(self, data_dirs, target_resolution=256, split='train',
                 train_ratio=0.85, augment=False):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        all_files = []
        for d in data_dirs:
            files = sorted(Path(d).glob("sample_*.npz"))
            all_files.extend(files)

        all_files.sort(key=lambda f: f.name)
        n_train = int(len(all_files) * train_ratio)
        if split == 'train':
            self.files = all_files[:n_train]
        else:
            self.files = all_files[n_train:]

        self.target_res = target_resolution
        self.augment = augment and (split == 'train')

        print(f"FEMDatasetHR [{split}]: {len(self.files)} samples from "
              f"{len(data_dirs)} dirs, target_res={target_resolution}³, "
              f"augment={self.augment}")

    def __len__(self):
        return len(self.files)

    def _upsample_field(self, field: torch.Tensor, target_size: int) -> torch.Tensor:
        """Upsample a [C, N, N, N] field to [C, target, target, target]."""
        if field.shape[-1] == target_size:
            return field
        return TF.interpolate(
            field.unsqueeze(0), size=(target_size,) * 3,
            mode='trilinear', align_corners=False
        ).squeeze(0)

    def _apply_flip(self, density, force, displacement, stress, von_mises,
                     fixed_mask, axis):
        """Physics-consistent flip (same as v3)."""
        spatial_dim = axis + 1
        density = density.flip(spatial_dim)
        fixed_mask = fixed_mask.flip(spatial_dim)
        von_mises = von_mises.flip(spatial_dim)
        force = force.flip(spatial_dim)
        displacement = displacement.flip(spatial_dim)
        stress = stress.flip(spatial_dim)

        force[axis] = -force[axis]
        displacement[axis] = -displacement[axis]

        if axis == 0:
            stress[4] = -stress[4]
            stress[5] = -stress[5]
        elif axis == 1:
            stress[3] = -stress[3]
            stress[5] = -stress[5]
        elif axis == 2:
            stress[3] = -stress[3]
            stress[4] = -stress[4]

        return density, force, displacement, stress, von_mises, fixed_mask

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        density = torch.from_numpy(data['density']).float().unsqueeze(0)
        force = torch.from_numpy(data['force']).float()
        fixed_mask = torch.from_numpy(data['fixed_mask']).float()
        displacement = torch.from_numpy(data['displacement']).float()
        stress = torch.from_numpy(data['stress']).float()

        if 'von_mises' in data:
            von_mises = torch.from_numpy(data['von_mises']).float().unsqueeze(0)
        else:
            s = stress.unsqueeze(0)
            von_mises = compute_von_mises(s)[0]

        # Augmentation (before upsampling for speed)
        if self.augment:
            for axis in range(3):
                if torch.rand(1).item() > 0.5:
                    density, force, displacement, stress, von_mises, fixed_mask = \
                        self._apply_flip(density, force, displacement, stress,
                                         von_mises, fixed_mask, axis)

        # Upsample to target resolution if needed
        N = density.shape[-1]
        if N != self.target_res:
            density = self._upsample_field(density, self.target_res)
            force = self._upsample_field(force, self.target_res)
            fixed_mask = self._upsample_field(fixed_mask, self.target_res)
            displacement = self._upsample_field(displacement, self.target_res)
            stress = self._upsample_field(stress, self.target_res)
            von_mises = self._upsample_field(von_mises, self.target_res)

        # Normalize
        force_norm = force / self.FORCE_SCALE
        disp_norm = displacement / self.DISP_SCALE
        stress_log = stress_to_log(stress)

        x = torch.cat([density, force_norm], dim=0)

        return {
            'x': x,
            'disp_norm': disp_norm,
            'stress_log': stress_log,
            'von_mises': von_mises,
            'density': density,
            'fixed_mask': fixed_mask,
            'solid_fraction': float((density > 0.5).float().mean()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                accum_steps=1):
    model.train()
    totals = {'total': 0, 'disp': 0, 'stress_log': 0,
              'vm_asym': 0, 'vm_peak': 0, 'bc': 0}
    n_batches = 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        x = batch['x'].to(device)
        disp_norm = batch['disp_norm'].to(device)
        stress_log = batch['stress_log'].to(device)
        vm = batch['von_mises'].to(device)
        density = batch['density'].to(device)
        fixed_mask = batch['fixed_mask'].to(device)

        # No AMP for HR-PINO (FFT + high-res = problematic with float16)
        pred = model(x)
        losses = criterion(pred, disp_norm, stress_log, vm, density, fixed_mask)
        loss = losses['total'] / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        for k in totals:
            if k in losses:
                totals[k] += losses[k].item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{losses['total'].item():.4e}",
            slog=f"{losses['stress_log'].item():.4e}",
            vasym=f"{losses['vm_asym'].item():.4e}",
        )

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    totals = {'total': 0, 'disp': 0, 'stress_log': 0,
              'vm_asym': 0, 'vm_peak': 0, 'bc': 0}
    n_batches = 0

    vm_rel_errors = []
    sf_ratios = []

    for batch in loader:
        x = batch['x'].to(device)
        disp_norm = batch['disp_norm'].to(device)
        stress_log = batch['stress_log'].to(device)
        vm = batch['von_mises'].to(device)
        density = batch['density'].to(device)
        fixed_mask = batch['fixed_mask'].to(device)

        pred = model(x)
        losses = criterion(pred.float(), disp_norm, stress_log, vm, density, fixed_mask)

        for k in totals:
            if k in losses:
                totals[k] += losses[k].item()
        n_batches += 1

        # Compute actual metrics
        pred_stress_log = pred[:, 3:].float()
        pred_stress_phys = log_to_stress(pred_stress_log)
        pred_vm = compute_von_mises(pred_stress_phys)
        mask = (density > 0.1).float()

        B = pred.shape[0]
        for i in range(B):
            m = mask[i, 0] > 0.5
            if m.any():
                pred_max = pred_vm[i, 0][m].max().item()
                target_max = vm[i, 0][m].max().item()
                if target_max > 1e-6:
                    rel = abs(pred_max - target_max) / target_max
                    vm_rel_errors.append(rel)

                    sf_pred = 50e6 / max(pred_max, 1e-10)
                    sf_true = 50e6 / target_max
                    sf_ratios.append(sf_pred / sf_true)

    avg = {k: v / max(n_batches, 1) for k, v in totals.items()}

    if vm_rel_errors:
        arr = np.array(vm_rel_errors)
        avg['vm_max_rel_mean'] = float(arr.mean())
        avg['vm_max_rel_median'] = float(np.median(arr))
        avg['vm_max_rel_p90'] = float(np.percentile(arr, 90))

    if sf_ratios:
        sr = np.array(sf_ratios)
        avg['sf_ratio_mean'] = float(sr.mean())
        avg['sf_ratio_median'] = float(np.median(sr))
        avg['conservative_pct'] = float((sr <= 1.0).mean() * 100)

    return avg


def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs,
                                min_lr_ratio=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train_hr_pino(
    data_dirs: list[str],
    output_dir: str = './checkpoints/hr_pino',
    pino_checkpoint: str | None = None,
    resume_from: str | None = None,
    resolution: int = 64,
    epochs: int = 200,
    batch_size: int = 2,
    lr_core: float = 1e-5,
    lr_encdec: float = 1e-4,
    warmup_epochs: int = 10,
    accum_steps: int = 4,
    **loss_kwargs,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Resolution: {resolution}³")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    train_set = FEMDatasetHR(data_dirs, target_resolution=resolution,
                              split='train', augment=True)
    val_set = FEMDatasetHR(data_dirs, target_resolution=resolution,
                            split='val', augment=False)

    # Weighted sampler: oversample thin shells
    sample_weights = []
    for i in range(len(train_set)):
        d = np.load(train_set.files[i])
        sf = (d['density'] > 0.5).mean()
        if sf < 0.05:
            sample_weights.append(3.0)
        elif sf < 0.10:
            sample_weights.append(2.0)
        else:
            sample_weights.append(1.0)

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_set), replacement=True
    )

    # Reduce num_workers for high-res to save RAM
    n_workers = 4 if resolution <= 128 else 2

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler,
        num_workers=n_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_v3,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=True,
        collate_fn=collate_v3,
    )

    # ── Model ──
    if resume_from:
        # Resume HR-PINO training
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model_cfg = ckpt.get('model_config', {})
        model = HRPINO3d(
            use_gradient_checkpoint=True,
            **{k: v for k, v in model_cfg.items() if k != 'use_gradient_checkpoint'},
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Resumed HR-PINO from {resume_from}")
    elif pino_checkpoint:
        # Transfer from PINO v3
        model = HRPINO3d.from_pino_checkpoint(
            pino_checkpoint, device=str(device),
            use_gradient_checkpoint=True,
        )
    else:
        # Train from scratch
        model = HRPINO3d(use_gradient_checkpoint=True).to(device)
        print("HR-PINO: training from scratch (no pretrained core)")

    param_counts = count_parameters(model)
    print(f"HR-PINO parameters: {param_counts['total']:,}")
    for k, v in param_counts.items():
        if k != 'total':
            print(f"  {k}: {v:,}")

    # ── Loss ──
    criterion = PINOLossV3(
        disp_scale=FEMDatasetHR.DISP_SCALE,
        **loss_kwargs,
    ).to(device)

    # ── Optimizer with differential LR ──
    param_groups = model.param_groups(lr_core=lr_core, lr_encdec=lr_encdec)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_epochs, epochs)

    # No AMP — cuFFT half-precision requires power-of-2 sizes
    scaler = GradScaler(enabled=False)

    # ── Training state ──
    start_epoch = 1
    history = []
    best_vm_metric = float('inf')
    best_conservative = 0.0
    no_improve = 0
    patience = 40

    if resume_from and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_vm_metric = ckpt.get('vm_max_rel_mean', float('inf'))
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch - 1}")

    # ── Training loop ──
    print(f"\nStarting training: epochs={epochs}, batch={batch_size}, "
          f"accum={accum_steps}, lr_core={lr_core}, lr_encdec={lr_encdec}")
    print(f"Effective batch size: {batch_size * accum_steps}")

    for epoch in range(start_epoch, epochs + 1):
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, accum_steps
        )
        val_losses = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr_cur_core = optimizer.param_groups[0]['lr']
        lr_cur_ed = optimizer.param_groups[1]['lr']
        vm_mean = val_losses.get('vm_max_rel_mean', float('inf'))
        vm_med = val_losses.get('vm_max_rel_median', float('inf'))
        sf_ratio = val_losses.get('sf_ratio_mean', float('nan'))
        cons_pct = val_losses.get('conservative_pct', 0)

        history.append({
            'epoch': epoch,
            'lr_core': lr_cur_core, 'lr_encdec': lr_cur_ed,
            'train': train_losses, 'val': val_losses,
        })

        print(f"  Train: {train_losses['total']:.4e} | Val: {val_losses['total']:.4e}")
        print(f"  VM rel: mean={vm_mean:.4f} med={vm_med:.4f} | "
              f"SF ratio: {sf_ratio:.3f} | Conservative: {cons_pct:.1f}% | "
              f"LR: core={lr_cur_core:.2e} ed={lr_cur_ed:.2e}")

        # Save best
        is_best = vm_mean < best_vm_metric
        if is_best:
            best_vm_metric = vm_mean
            best_conservative = cons_pct
            no_improve = 0

            model_config = {
                'in_channels': model.in_channels,
                'out_channels': model.out_channels,
                'fno_modes': model.fno_blocks[0].spectral_conv.modes
                    if hasattr(model.fno_blocks[0], 'spectral_conv') else 16,
                'fno_width': model.bridge_in[0].out_channels,
                'fno_layers': len(model.fno_blocks),
                'fno_padding': model.fno_padding,
            }

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model_config,
                'vm_max_rel_mean': vm_mean,
                'sf_ratio_mean': sf_ratio,
                'conservative_pct': cons_pct,
                'val_loss': val_losses['total'],
                'resolution': resolution,
                'normalization': {
                    'force_scale': FEMDatasetHR.FORCE_SCALE,
                    'disp_scale': FEMDatasetHR.DISP_SCALE,
                    'stress_mode': 'log10',
                },
                'version': 'hr_pino_v1',
            }, output_path / 'best_model.pt')
            print(f"  -> Best (VM={vm_mean:.4f}, conservative={cons_pct:.1f}%)")
        else:
            no_improve += 1

        if epoch % 50 == 0:
            torch.save(model.state_dict(), output_path / f'epoch_{epoch}.pt')

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save history
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nHR-PINO training complete.")
    print(f"  Best VM max rel error: {best_vm_metric:.4f}")
    print(f"  Conservative predictions: {best_conservative:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HR-PINO: High-Resolution PINO Training')
    parser.add_argument('--data_dirs', type=str, required=True,
                        help='Comma-separated FEM dataset directories')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/hr_pino')
    parser.add_argument('--pino_checkpoint', type=str, default=None,
                        help='PINO v3 checkpoint for transfer learning (first time)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='HR-PINO checkpoint to resume training')
    parser.add_argument('--resolution', type=int, default=64,
                        choices=[64, 128, 256],
                        help='Target resolution for training')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr_core', type=float, default=1e-5)
    parser.add_argument('--lr_encdec', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--accum_steps', type=int, default=4)
    # Loss weights
    parser.add_argument('--lambda_disp', type=float, default=1.0)
    parser.add_argument('--lambda_stress_log', type=float, default=3.0)
    parser.add_argument('--lambda_vm_asym', type=float, default=2.0)
    parser.add_argument('--lambda_vm_peak', type=float, default=1.0)
    parser.add_argument('--lambda_bc', type=float, default=1.0)
    parser.add_argument('--underestimate_weight', type=float, default=2.0)
    parser.add_argument('--overestimate_weight', type=float, default=0.5)

    args = parser.parse_args()

    loss_kwargs = {
        'lambda_disp': args.lambda_disp,
        'lambda_stress_log': args.lambda_stress_log,
        'lambda_vm_asym': args.lambda_vm_asym,
        'lambda_vm_peak': args.lambda_vm_peak,
        'lambda_bc': args.lambda_bc,
        'underestimate_weight': args.underestimate_weight,
        'overestimate_weight': args.overestimate_weight,
    }

    train_hr_pino(
        data_dirs=args.data_dirs.split(','),
        output_dir=args.output_dir,
        pino_checkpoint=args.pino_checkpoint,
        resume_from=args.resume_from,
        resolution=args.resolution,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_core=args.lr_core,
        lr_encdec=args.lr_encdec,
        warmup_epochs=args.warmup_epochs,
        accum_steps=args.accum_steps,
        **loss_kwargs,
    )
