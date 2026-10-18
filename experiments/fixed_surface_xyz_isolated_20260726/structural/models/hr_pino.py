"""
HR-PINO: Hierarchical Resolution Physics-Informed Neural Operator

支持 256³ 原生输入，通过 U-Net 式 encoder-decoder 包裹 F-FNO core：
  Encoder: 256³ → 128³ → 64³  (strided 3D conv)
  Core:    F-FNO @ 64³         (复用 PINO3d 已训练权重)
  Decoder: 64³ → 128³ → 256³  (trilinear upsample + conv + skip)

输入: [ρ, Fx, Fy, Fz] → B × 4 × 256 × 256 × 256
输出: [ux, uy, uz, σxx, σyy, σzz, σyz, σzx, σxy] → B × 9 × 256 × 256 × 256

显存优化:
  - Encoder/Decoder 使用窄通道 (32ch)
  - Core 仅在 64³ 运行 F-FNO (64ch, 16 modes)
  - 梯度检查点 (gradient checkpointing) 可选
  - 预计 ~8GB @ batch=1, 256³
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .pino import SpectralConv3d_Factorized, FNOBlock3d, PINO3d


class EncoderBlock(nn.Module):
    """下采样块: Conv3d(stride=2) + InstanceNorm + GELU + Conv3d(1x1)"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """上采样块: Trilinear upsample + 1x1 skip compress + Conv3d"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # Compress skip to reduce memory at high resolutions
        self.skip_compress = nn.Conv3d(skip_ch, min(skip_ch, 8), 1)
        compressed_skip = min(skip_ch, 8)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch + compressed_skip, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        skip = self.skip_compress(skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HRPINO3d(nn.Module):
    """Hierarchical Resolution PINO for native 256³ inference.

    Architecture:
        256³ ──[Enc1]──> 128³ ──[Enc2]──> 64³
                  │ skip1          │ skip2
                  v                v
        256³ <──[Dec1]── 128³ <──[Dec2]── 64³ <── F-FNO Core
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 9,
        # Encoder/Decoder widths
        enc_channels: tuple = (32, 48),
        dec_channels: tuple = (48, 32),
        # F-FNO core config (matches PINO3d)
        fno_modes: int = 16,
        fno_width: int = 64,
        fno_layers: int = 6,
        fno_padding: int = 8,
        # Options
        use_gradient_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_gc = use_gradient_checkpoint

        # ── Encoder ──
        # Input: [ρ, Fx, Fy, Fz] = 4ch @ 256³
        self.enc1 = EncoderBlock(in_channels, enc_channels[0])   # 256→128, 4→32
        self.enc2 = EncoderBlock(enc_channels[0], enc_channels[1])  # 128→64, 32→48

        # ── Bridge: enc_channels[-1] → fno_width (adapt to F-FNO input) ──
        # F-FNO core expects width channels (not in_channels+3)
        # We handle grid coords and lifting here instead of in PINO3d
        self.bridge_in = nn.Sequential(
            nn.Conv3d(enc_channels[-1] + 3, fno_width, 1),  # +3 for grid coords
            nn.GELU(),
        )

        # ── F-FNO Core @ 64³ ──
        self.fno_blocks = nn.ModuleList([
            FNOBlock3d(fno_width, fno_modes) for _ in range(fno_layers)
        ])
        self.fno_padding = fno_padding

        # Bridge out: fno_width → dec_channels[0] for decoder
        self.bridge_out = nn.Sequential(
            nn.Conv3d(fno_width, enc_channels[-1], 1),
            nn.GELU(),
        )

        # ── Decoder ──
        # Dec2: 64³→128³, skip from enc1 (enc_channels[0])
        self.dec2 = DecoderBlock(enc_channels[-1], enc_channels[0], dec_channels[0])
        # Dec1: 128³→256³, skip from input (in_channels)
        self.dec1 = DecoderBlock(dec_channels[0], in_channels, dec_channels[1])

        # ── Output head ──
        self.output_head = nn.Sequential(
            nn.Conv3d(dec_channels[-1], dec_channels[-1] * 2, 1),
            nn.GELU(),
            nn.Conv3d(dec_channels[-1] * 2, out_channels, 1),
        )

        # Grid cache
        self._grid_cache = {}

    def _get_grid(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Generate normalized 3D grid coordinates [1, 3, Nx, Ny, Nz]."""
        key = (*shape, str(device))
        if key not in self._grid_cache:
            Nx, Ny, Nz = shape
            gx = torch.linspace(0, 1, Nx, device=device)
            gy = torch.linspace(0, 1, Ny, device=device)
            gz = torch.linspace(0, 1, Nz, device=device)
            grid_x, grid_y, grid_z = torch.meshgrid(gx, gy, gz, indexing='ij')
            grid = torch.stack([grid_x, grid_y, grid_z], dim=0).unsqueeze(0)
            self._grid_cache[key] = grid
        return self._grid_cache[key]

    def _run_fno_core(self, x: torch.Tensor) -> torch.Tensor:
        """Run F-FNO blocks with optional padding (same as PINO3d)."""
        p = self.fno_padding
        if p > 0:
            x = F.pad(x, [p, p, p, p, p, p])

        for block in self.fno_blocks:
            if self.use_gc and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        if p > 0:
            x = x[:, :, p:-p, p:-p, p:-p]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4, H, W, D] — supports any resolution (256³, 128³, 64³, ...)
               Encoder adapts via strided conv; core always runs at H/4 × W/4 × D/4
        Returns:
            [B, 9, H, W, D]
        """
        B = x.shape[0]

        # ── Encoder ──
        skip_input = x                           # [B, 4, 256, 256, 256]
        if self.use_gc and self.training:
            e1 = grad_checkpoint(self.enc1, x, use_reentrant=False)
        else:
            e1 = self.enc1(x)                     # [B, 32, 128, 128, 128]
        skip1 = e1
        if self.use_gc and self.training:
            e2 = grad_checkpoint(self.enc2, e1, use_reentrant=False)
        else:
            e2 = self.enc2(e1)                    # [B, 48, 64, 64, 64]

        # ── Bridge in: add grid coords at 64³ resolution ──
        _, _, Nx, Ny, Nz = e2.shape
        grid = self._get_grid((Nx, Ny, Nz), e2.device).expand(B, -1, -1, -1, -1)
        e2_with_grid = torch.cat([e2, grid], dim=1)  # [B, 48+3, 64, 64, 64]
        core_in = self.bridge_in(e2_with_grid)        # [B, fno_width, 64, 64, 64]

        # ── F-FNO Core @ 64³ ──
        core_out = self._run_fno_core(core_in)        # [B, fno_width, 64, 64, 64]

        # ── Bridge out ──
        d = self.bridge_out(core_out)                 # [B, 48, 64, 64, 64]

        # ── Decoder with skip connections ──
        if self.use_gc and self.training:
            d = grad_checkpoint(self.dec2, d, skip1, use_reentrant=False)
            d = grad_checkpoint(self.dec1, d, skip_input, use_reentrant=False)
        else:
            d = self.dec2(d, skip1)                   # [B, 48, 128, 128, 128]
            d = self.dec1(d, skip_input)              # [B, 32, 256, 256, 256]

        # ── Output ──
        out = self.output_head(d)                     # [B, 9, 256, 256, 256]
        return out

    def predict_displacement_and_stress(self, x: torch.Tensor):
        """Split output into displacement and stress for physics loss."""
        out = self.forward(x)
        displacement = out[:, :3]
        stress = out[:, 3:]
        return displacement, stress

    @classmethod
    def from_pino_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = 'cpu',
        use_gradient_checkpoint: bool = True,
        **kwargs,
    ) -> 'HRPINO3d':
        """Create HR-PINO and load F-FNO core weights from a trained PINO3d checkpoint.

        This allows transfer learning: the F-FNO core (the expensive part) is
        initialized from a trained 64³ PINO, while encoder/decoder are randomly
        initialized and fine-tuned.

        Args:
            checkpoint_path: Path to PINO3d (v1/v3) checkpoint .pt file
            device: Target device
            use_gradient_checkpoint: Enable gradient checkpointing for training
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract model config from checkpoint
        if 'model_config' in ckpt:
            cfg = ckpt['model_config']
        else:
            cfg = {
                'in_channels': 4, 'out_channels': 9,
                'modes': 16, 'width': 64, 'n_layers': 6, 'padding': 8,
            }

        # Create HR-PINO with matching F-FNO config
        model = cls(
            in_channels=cfg.get('in_channels', 4),
            out_channels=cfg.get('out_channels', 9),
            fno_modes=cfg.get('modes', 16),
            fno_width=cfg.get('width', 64),
            fno_layers=cfg.get('n_layers', 6),
            fno_padding=cfg.get('padding', 8),
            use_gradient_checkpoint=use_gradient_checkpoint,
            **kwargs,
        )

        # Load F-FNO core weights from PINO3d checkpoint
        state = ckpt.get('model_state_dict', ckpt)

        # Map PINO3d.blocks.{i}.* → HRPINO3d.fno_blocks.{i}.*
        loaded_keys = []
        for key, value in state.items():
            if key.startswith('blocks.'):
                new_key = key.replace('blocks.', 'fno_blocks.', 1)
                if new_key in model.state_dict():
                    model.state_dict()[new_key].copy_(value)
                    loaded_keys.append(key)

        # Use load_state_dict with strict=False for partial loading
        fno_state = {}
        for key, value in state.items():
            if key.startswith('blocks.'):
                new_key = key.replace('blocks.', 'fno_blocks.', 1)
                fno_state[new_key] = value

        missing, unexpected = model.load_state_dict(fno_state, strict=False)
        n_loaded = len(fno_state)
        n_missing = len([k for k in missing if 'fno_blocks' in k])

        print(f"HR-PINO: loaded {n_loaded} F-FNO core params from checkpoint")
        print(f"  FNO blocks missing: {n_missing}, Encoder/Decoder: random init")
        print(f"  Config: modes={cfg.get('modes')}, width={cfg.get('width')}, "
              f"layers={cfg.get('n_layers')}, padding={cfg.get('padding')}")

        return model.to(device)

    def param_groups(self, lr_core: float = 1e-5, lr_encdec: float = 1e-4):
        """Separate param groups for differential learning rates.

        Core (pretrained F-FNO): lower LR to preserve learned features
        Encoder/Decoder (random init): higher LR for fast convergence
        """
        core_params = []
        encdec_params = []

        for name, param in self.named_parameters():
            if 'fno_blocks' in name or 'bridge_in' in name:
                core_params.append(param)
            else:
                encdec_params.append(param)

        return [
            {'params': core_params, 'lr': lr_core, 'name': 'fno_core'},
            {'params': encdec_params, 'lr': lr_encdec, 'name': 'enc_dec'},
        ]


def count_parameters(model: nn.Module) -> dict:
    """Count trainable parameters by component."""
    counts = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        component = name.split('.')[0]
        counts[component] = counts.get(component, 0) + param.numel()

    total = sum(counts.values())
    return {**counts, 'total': total}
