"""
PINO预训练脚本 (Stage 0)

使用FEM数据集训练F-FNO:
- 数据损失: MSE(pred, FEM真值)
- PDE损失: 平衡方程残差
- BC损失: 固定面位移约束

用法:
    python pretrain_pino.py --data_dir ./fem_dataset --epochs 200
"""
import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from structural.models.pino import PINO3d, PINOLoss
from structural.configs import StructShadowConfig


class FEMDataset(Dataset):
    """FEM数据集加载器 (含归一化)

    归一化策略 (global):
    - density: 已经是 [0, 1]，不变
    - force: 除以全局 force_scale → [-1, 1]
    - displacement: 除以全局 disp_scale → 大部分 [-1, 1]
    - stress: 除以全局 stress_scale → 大部分 [-1, 1]

    全局归一化因子从训练集 p99 计算，存入 checkpoint 以确保推理一致。
    """

    # 全局归一化因子 (训练集 p99)
    FORCE_SCALE = 12164.43
    DISP_SCALE = 0.00997422
    STRESS_SCALE = 3146304.25

    def __init__(self, data_dir: str, split: str = 'train', train_ratio: float = 0.85):
        self.data_dir = Path(data_dir)
        files = sorted(self.data_dir.glob("sample_*.npz"))

        n_train = int(len(files) * train_ratio)
        if split == 'train':
            self.files = files[:n_train]
        else:
            self.files = files[n_train:]

        self.normalization_mode = 'global'
        print(f"FEMDataset [{split}]: {len(self.files)} samples, norm={self.normalization_mode}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        density = torch.from_numpy(data['density']).float().unsqueeze(0)    # [1, N, N, N]
        force = torch.from_numpy(data['force']).float()                      # [3, N, N, N]
        fixed_mask = torch.from_numpy(data['fixed_mask']).float()            # [1, N, N, N]
        displacement = torch.from_numpy(data['displacement']).float()        # [3, N, N, N]
        stress = torch.from_numpy(data['stress']).float()                    # [6, N, N, N]

        # 全局归一化
        force_norm = force / self.FORCE_SCALE
        displacement_norm = displacement / self.DISP_SCALE
        stress_norm = stress / self.STRESS_SCALE

        # 输入: [rho, Fx_norm, Fy_norm, Fz_norm]
        x = torch.cat([density, force_norm], dim=0)                          # [4, N, N, N]
        # 目标: [ux_norm, uy_norm, uz_norm, sxx_norm, ...]
        y = torch.cat([displacement_norm, stress_norm], dim=0)               # [9, N, N, N]

        return x, y, force_norm, density, fixed_mask


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = {'total': 0, 'data': 0, 'pde': 0, 'equilibrium': 0, 'constitutive': 0, 'bc': 0}
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        x, y, force, density, fixed_mask = batch[0], batch[1], batch[2], batch[3], batch[4]
        x = x.to(device)
        y = y.to(device)
        force = force.to(device)
        density = density.to(device)
        fixed_mask = fixed_mask.to(device)

        pred = model(x)
        losses = criterion(pred, y, force, density, fixed_mask)

        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in total_loss:
            total_loss[k] += losses[k].item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{losses['total'].item():.4e}",
            data=f"{losses['data'].item():.4e}",
            pde=f"{losses['pde'].item():.4e}",
        )

    return {k: v / max(n_batches, 1) for k, v in total_loss.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = {'total': 0, 'data': 0, 'pde': 0, 'equilibrium': 0, 'constitutive': 0, 'bc': 0}
    n_batches = 0

    for batch in loader:
        x, y, force, density, fixed_mask = batch[0], batch[1], batch[2], batch[3], batch[4]
        x = x.to(device)
        y = y.to(device)
        force = force.to(device)
        density = density.to(device)
        fixed_mask = fixed_mask.to(device)

        pred = model(x)
        losses = criterion(pred, y, force, density, fixed_mask)

        for k in total_loss:
            total_loss[k] += losses[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_loss.items()}


def train_pino(
    data_dir: str,
    output_dir: str = './checkpoints/pino',
    epochs: int = 200,
    batch_size: int = 8,
    lr: float = 1e-3,
    resume: bool = False,
    no_early_stop: bool = False,
    config: StructShadowConfig | None = None,
):
    """PINO预训练主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = StructShadowConfig.from_yaml()

    # 数据
    train_set = FEMDataset(data_dir, split='train')
    val_set = FEMDataset(data_dir, split='val')

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # 模型
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

    # 损失和优化器
    # PDE/本构损失暂时禁用 — 涉及物理量级在归一化空间不匹配
    # 使用纯监督学习 (data + bc), 全局归一化确保推理时 scale 一致
    criterion = PINOLoss(
        lambda_data=1.0,
        lambda_pde=0.0,
        lambda_bc=1.0,
        lambda_constitutive=0.0,
        resolution=config.resolution,
        youngs_modulus=config.material.youngs_modulus,
        poisson_ratio=config.material.poisson_ratio,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 断点续训
    start_epoch = 1
    history = []
    best_val_loss = float('inf')
    no_improve_count = 0

    if resume:
        ckpt_path = output_path / 'best_model.pt'
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            best_val_loss = ckpt.get('val_loss', float('inf'))
            start_epoch = ckpt.get('epoch', 0) + 1
            # 快进 scheduler 到对应 epoch
            for _ in range(start_epoch - 1):
                scheduler.step()
            # 加载已有 history
            hist_path = output_path / 'training_history.json'
            if hist_path.exists():
                with open(hist_path) as f:
                    history = json.load(f)
            print(f"  Resumed from epoch {start_epoch - 1}, best_val_loss={best_val_loss:.4e}")
        else:
            print(f"  No checkpoint found at {ckpt_path}, training from scratch")

    patience = 30  # early stopping patience

    for epoch in range(start_epoch, epochs + 1):
        # Per-sample normalization 下跳过 PDE warmup (PDE loss 已禁用)

        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_losses = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr_current = optimizer.param_groups[0]['lr']
        history.append({
            'epoch': epoch,
            'lr': lr_current,
            'train': train_losses,
            'val': val_losses,
        })

        print(f"  Train: {train_losses['total']:.4e} | Val: {val_losses['total']:.4e} "
              f"(eq={val_losses.get('equilibrium', 0):.2e}, "
              f"const={val_losses.get('constitutive', 0):.2e}) | LR: {lr_current:.2e}")

        # 保存最优
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            no_improve_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'normalization_mode': 'global',
                'normalization': {
                    'force_scale': FEMDataset.FORCE_SCALE,
                    'disp_scale': FEMDataset.DISP_SCALE,
                    'stress_scale': FEMDataset.STRESS_SCALE,
                },
                'config': {
                    'modes': config.pino.modes,
                    'width': config.pino.width,
                    'n_layers': config.pino.n_layers,
                    'in_channels': config.pino.in_channels,
                    'out_channels': config.pino.out_channels,
                    'padding': config.pino.padding,
                },
            }, output_path / 'best_model.pt')
            print(f"  -> New best model saved (val_loss={best_val_loss:.4e})")
        else:
            no_improve_count += 1

        # 定期保存
        if epoch % 50 == 0:
            torch.save(model.state_dict(), output_path / f'epoch_{epoch}.pt')

        # Early stopping (可通过 --no_early_stop 禁用)
        if not no_early_stop and no_improve_count >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # 保存历史
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nPINO pretraining complete. Best val loss: {best_val_loss:.4e}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain PINO on FEM dataset')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints/pino')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', action='store_true', help='Resume from best_model.pt')
    parser.add_argument('--no_early_stop', action='store_true', help='Disable early stopping')
    args = parser.parse_args()

    train_pino(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        no_early_stop=args.no_early_stop,
    )
