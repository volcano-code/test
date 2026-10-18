"""
PINO3d — Physics-Informed Neural Operator (Factorized FNO) for 3D Structural Analysis

输入: [ρ, Fx, Fy, Fz] → 4通道 64³
输出: [ux, uy, uz, σxx, σyy, σzz, σyz, σzx, σxy] → 9通道 64³

架构: F-FNO (Factorized Fourier Neural Operator)
- 使用分离式频谱卷积: 分别沿 x/y/z 维度做FFT
- 比完整3D FFT节省显存，适合64³输入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class SpectralConv3d_Factorized(nn.Module):
    """分离式3D频谱卷积 — F-FNO核心层

    分别沿x/y/z三个维度做1D FFT + 频域线性变换 + IFFT，
    而非完整的3D FFT，显著降低计算量和显存占用。
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        # 三个维度各一组频域权重
        self.weights_x = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))
        self.weights_y = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))
        self.weights_z = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def _spectral_mul_1d(self, x_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bimd,iom->bomd", x_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Sx, Sy, Sz = x.shape
        m = self.modes
        out = torch.zeros(B, self.out_channels, Sx, Sy, Sz, device=x.device, dtype=x.dtype)

        # X维度
        x_ft = torch.fft.rfft(x, dim=2)
        x_ft_trunc = x_ft[:, :, :m, :, :]
        x_ft_flat = x_ft_trunc.reshape(B, C, m, Sy * Sz)
        out_ft_flat = self._spectral_mul_1d(x_ft_flat, self.weights_x)
        out_ft_x = torch.zeros(B, self.out_channels, x_ft.shape[2], Sy, Sz,
                                device=x.device, dtype=x_ft.dtype)
        out_ft_x[:, :, :m, :, :] = out_ft_flat.reshape(B, self.out_channels, m, Sy, Sz)
        out += torch.fft.irfft(out_ft_x, n=Sx, dim=2)

        # Y维度
        x_ft = torch.fft.rfft(x, dim=3)
        Sy_ft = x_ft.shape[3]
        x_ft_trunc = x_ft[:, :, :, :m, :]
        x_ft_perm = x_ft_trunc.permute(0, 1, 3, 2, 4)
        x_ft_flat = x_ft_perm.reshape(B, C, m, Sx * Sz)
        out_ft_flat = self._spectral_mul_1d(x_ft_flat, self.weights_y)
        out_ft_y = torch.zeros(B, self.out_channels, Sx, Sy_ft, Sz,
                                device=x.device, dtype=x_ft.dtype)
        out_ft_y[:, :, :, :m, :] = out_ft_flat.reshape(
            B, self.out_channels, m, Sx, Sz).permute(0, 1, 3, 2, 4)
        out += torch.fft.irfft(out_ft_y, n=Sy, dim=3)

        # Z维度
        x_ft = torch.fft.rfft(x, dim=4)
        Sz_ft = x_ft.shape[4]
        x_ft_trunc = x_ft[:, :, :, :, :m]
        x_ft_perm = x_ft_trunc.permute(0, 1, 4, 2, 3)
        x_ft_flat = x_ft_perm.reshape(B, C, m, Sx * Sy)
        out_ft_flat = self._spectral_mul_1d(x_ft_flat, self.weights_z)
        out_ft_z = torch.zeros(B, self.out_channels, Sx, Sy, Sz_ft,
                                device=x.device, dtype=x_ft.dtype)
        out_ft_z[:, :, :, :, :m] = out_ft_flat.reshape(
            B, self.out_channels, m, Sx, Sy).permute(0, 1, 3, 4, 2)
        out += torch.fft.irfft(out_ft_z, n=Sz, dim=4)

        return out


class FNOBlock3d(nn.Module):
    """FNO块: 频谱卷积 + 1x1x1卷积(skip) + 归一化 + 激活"""

    def __init__(self, channels: int, modes: int, use_checkpoint: bool = False):
        super().__init__()
        self.spectral_conv = SpectralConv3d_Factorized(channels, channels, modes)
        self.conv_skip = nn.Conv3d(channels, channels, 1)
        self.norm = nn.InstanceNorm3d(channels)
        self.use_checkpoint = use_checkpoint

    def _forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral_conv(x) + self.conv_skip(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_fn, x, use_reentrant=False)
        return self._forward_fn(x)


class PINO3d(nn.Module):
    """Physics-Informed Neural Operator for 3D elasticity (flat FNO)"""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 9,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
        padding: int = 8,
    ):
        super().__init__()
        self._has_scale_head = False
        self._scale_branch_prefixes = ()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.padding = padding

        self.lifting = nn.Conv3d(in_channels + 3, width, 1)
        self.blocks = nn.ModuleList([
            FNOBlock3d(width, modes) for _ in range(n_layers)
        ])
        self.projection = nn.Sequential(
            nn.Conv3d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv3d(width * 2, out_channels, 1),
        )
        self._grid_cache = {}

    def _get_grid(self, shape: tuple, device: torch.device) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Nx, Ny, Nz = x.shape
        grid = self._get_grid((Nx, Ny, Nz), x.device).expand(B, -1, -1, -1, -1)
        x = torch.cat([x, grid], dim=1)
        x = self.lifting(x)
        p = self.padding
        if p > 0:
            x = F.pad(x, [p, p, p, p, p, p])
        for block in self.blocks:
            x = block(x)
        if p > 0:
            x = x[:, :, p:-p, p:-p, p:-p]
        out = self.projection(x)
        return out

    def predict_displacement_and_stress(self, x: torch.Tensor):
        out = self.forward(x)
        return out[:, :3], out[:, 3:]


class UFNO3d(nn.Module):
    """U-shaped Fourier Neural Operator for 3D elasticity (v7b proven architecture)

    Architecture:
        64³ → Lift → Enc1(64³) ──skip──→ Dec1(64³) → DualHead → Output (9ch field)
                       ↓                    ↑
                     Enc2(32³) ──skip──→ Dec2(32³)
                       ↓                    ↑
                     Bottleneck(16³) ───────┘
                          ↓
                     ScaleHead → log10(vm_max) scalar  [NEW]

    The ScaleHead predicts the per-sample von Mises maximum in log-space,
    recovering the absolute stress magnitude lost by per-sample normalization.

    + Gradient checkpointing on enc2/bottleneck/dec2 to save ~9GB VRAM
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 9,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
        padding: int = 0,
    ):
        super().__init__()
        self._has_scale_head = True
        self._scale_branch_prefixes = ('scale_head',)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.padding = padding

        w = width
        n_enc = max(n_layers // 2, 1)
        n_bot = 2
        n_dec = max(n_layers // 2, 1)

        # Lifting
        self.lifting = nn.Conv3d(in_channels + 3, w, 1)

        # Encoder level 1: full resolution (64³)
        self.enc1 = nn.ModuleList([FNOBlock3d(w, modes) for _ in range(n_enc)])

        # Downsample 64³ → 32³
        self.down1 = nn.Conv3d(w, w * 2, kernel_size=2, stride=2)
        w2 = w * 2
        modes2 = max(modes // 2, 4)

        # Encoder level 2 (gradient checkpointed)
        self.enc2 = nn.ModuleList([
            FNOBlock3d(w2, modes2, use_checkpoint=True) for _ in range(n_enc)])

        # Downsample 32³ → 16³
        self.down2 = nn.Conv3d(w2, w2 * 2, kernel_size=2, stride=2)
        w4 = w2 * 2
        modes4 = max(modes // 4, 2)

        # Bottleneck (gradient checkpointed)
        self.bottleneck = nn.ModuleList([
            FNOBlock3d(w4, modes4, use_checkpoint=True) for _ in range(n_bot)])

        # Decoder level 2 (gradient checkpointed)
        self.up2 = nn.ConvTranspose3d(w4, w2, kernel_size=2, stride=2)
        self.dec2_proj = nn.Conv3d(w2 * 2, w2, 1)
        self.dec2 = nn.ModuleList([
            FNOBlock3d(w2, modes2, use_checkpoint=True) for _ in range(n_dec)])

        # Decoder level 1
        self.up1 = nn.ConvTranspose3d(w2, w, kernel_size=2, stride=2)
        self.dec1_proj = nn.Conv3d(w * 2, w, 1)
        self.dec1 = nn.ModuleList([FNOBlock3d(w, modes) for _ in range(n_dec)])

        # Dual-Head projection (v7b original design)
        # Displacement head (smooth, low-frequency dominated)
        self.disp_head = nn.Sequential(
            nn.Conv3d(w, w, 1),
            nn.GELU(),
            nn.Conv3d(w, 3, 1),
        )
        # Stress head (sharp, high-frequency details matter)
        self.stress_head = nn.Sequential(
            nn.Conv3d(w, w * 2, 1),
            nn.GELU(),
            nn.Conv3d(w * 2, w, 1),
            nn.GELU(),
            nn.Conv3d(w, 6, 1),
        )

        # Scale head: predicts log10(vm_max) from bottleneck features
        # Recovers absolute stress magnitude lost by per-sample normalization.
        # Input: bottleneck feature [B, w4, 16, 16, 16] → scalar [B, 1]
        self.scale_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),          # [B, w4, 1, 1, 1]
            nn.Flatten(),                      # [B, w4]
            nn.Linear(w4, w4 // 2),
            nn.GELU(),
            nn.Linear(w4 // 2, 1),            # [B, 1] → log10(vm_max)
        )

        self._grid_cache = {}

    def _get_grid(self, shape: tuple, device: torch.device) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Nx, Ny, Nz = x.shape

        grid = self._get_grid((Nx, Ny, Nz), x.device).expand(B, -1, -1, -1, -1)
        x = torch.cat([x, grid], dim=1)

        x = self.lifting(x)

        p = self.padding
        if p > 0:
            x = F.pad(x, [p, p, p, p, p, p])

        # Encoder level 1
        h1 = x
        for block in self.enc1:
            h1 = block(h1)
        skip1 = h1

        # Downsample → level 2
        h2 = self.down1(h1)
        for block in self.enc2:
            h2 = block(h2)
        skip2 = h2

        # Downsample → bottleneck
        h = self.down2(h2)
        for block in self.bottleneck:
            h = block(h)

        # Scale head: extract log10(vm_max) from bottleneck
        self._bottleneck_feat = h  # cache for scale_head access

        # Upsample → level 2 + skip
        h = self.up2(h)
        if h.shape != skip2.shape:
            h = F.interpolate(h, size=skip2.shape[2:], mode='trilinear', align_corners=False)
        h = torch.cat([h, skip2], dim=1)
        h = self.dec2_proj(h)
        for block in self.dec2:
            h = block(h)

        # Upsample → level 1 + skip
        h = self.up1(h)
        if h.shape != skip1.shape:
            h = F.interpolate(h, size=skip1.shape[2:], mode='trilinear', align_corners=False)
        h = torch.cat([h, skip1], dim=1)
        h = self.dec1_proj(h)
        for block in self.dec1:
            h = block(h)

        # Remove padding
        if p > 0:
            h = h[:, :, p:-p, p:-p, p:-p]

        # Dual-Head output
        disp = self.disp_head(h)      # [B, 3, N, N, N]
        stress = self.stress_head(h)   # [B, 6, N, N, N]

        return torch.cat([disp, stress], dim=1)  # [B, 9, N, N, N]

    def predict_scale(self) -> torch.Tensor:
        """Return log10(vm_max) prediction from cached bottleneck features.

        Must be called AFTER forward(). Returns [B, 1].
        """
        if not hasattr(self, '_bottleneck_feat') or self._bottleneck_feat is None:
            raise RuntimeError("predict_scale() called before forward()")
        feat = self._bottleneck_feat.detach() if getattr(self, '_detach_scale_head_input', False) else self._bottleneck_feat
        return self.scale_head(feat)

    def forward_with_scale(self, x: torch.Tensor):
        """Forward pass returning both field output and scale prediction.

        Returns:
            fields: [B, 9, N, N, N] — normalized displacement + stress
            log_vm_max: [B, 1] — predicted log10(vm_max) for de-normalization
        """
        fields = self.forward(x)
        feat = self._bottleneck_feat.detach() if getattr(self, '_detach_scale_head_input', False) else self._bottleneck_feat
        log_vm_max = self.scale_head(feat)
        return fields, log_vm_max

    def predict_displacement_and_stress(self, x: torch.Tensor):
        out = self.forward(x)
        return out[:, :3], out[:, 3:]


class UFNODualBranch3d(nn.Module):
    """Dual-branch scale-aware UFNO.

    Field prediction and absolute-scale prediction use separate trunks to reduce
    gradient interference between normalized field supervision and vm_max scale
    supervision. The field branch keeps the proven UFNO topology; the scale
    branch is a lighter U-shaped encoder that only predicts log10(vm_max).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 9,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
        padding: int = 0,
    ):
        super().__init__()
        self._has_scale_head = True
        self._scale_branch_prefixes = ('scale_lifting', 'scale_down', 'scale_enc',
                                       'scale_bottleneck', 'scale_head')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.padding = padding

        w = width
        n_enc = max(n_layers // 2, 1)
        n_bot = 2
        n_dec = max(n_layers // 2, 1)

        # Field branch: identical to UFNO3d so old good behavior is preserved.
        self.lifting = nn.Conv3d(in_channels + 3, w, 1)
        self.enc1 = nn.ModuleList([FNOBlock3d(w, modes) for _ in range(n_enc)])
        self.down1 = nn.Conv3d(w, w * 2, kernel_size=2, stride=2)
        w2 = w * 2
        modes2 = max(modes // 2, 4)
        self.enc2 = nn.ModuleList([
            FNOBlock3d(w2, modes2, use_checkpoint=True) for _ in range(n_enc)])
        self.down2 = nn.Conv3d(w2, w2 * 2, kernel_size=2, stride=2)
        w4 = w2 * 2
        modes4 = max(modes // 4, 2)
        self.bottleneck = nn.ModuleList([
            FNOBlock3d(w4, modes4, use_checkpoint=True) for _ in range(n_bot)])
        self.up2 = nn.ConvTranspose3d(w4, w2, kernel_size=2, stride=2)
        self.dec2_proj = nn.Conv3d(w2 * 2, w2, 1)
        self.dec2 = nn.ModuleList([
            FNOBlock3d(w2, modes2, use_checkpoint=True) for _ in range(n_dec)])
        self.up1 = nn.ConvTranspose3d(w2, w, kernel_size=2, stride=2)
        self.dec1_proj = nn.Conv3d(w * 2, w, 1)
        self.dec1 = nn.ModuleList([FNOBlock3d(w, modes) for _ in range(n_dec)])
        self.disp_head = nn.Sequential(
            nn.Conv3d(w, w, 1),
            nn.GELU(),
            nn.Conv3d(w, 3, 1),
        )
        self.stress_head = nn.Sequential(
            nn.Conv3d(w, w * 2, 1),
            nn.GELU(),
            nn.Conv3d(w * 2, w, 1),
            nn.GELU(),
            nn.Conv3d(w, 6, 1),
        )

        # Scale branch: separate encoder, lighter than the field branch.
        scale_w = max(w // 2, 16)
        scale_w2 = scale_w * 2
        scale_w4 = scale_w2 * 2
        scale_modes2 = max(modes // 2, 4)
        scale_modes4 = max(modes // 4, 2)
        self.scale_lifting = nn.Conv3d(in_channels + 3, scale_w, 1)
        self.scale_down1 = nn.Conv3d(scale_w, scale_w2, kernel_size=2, stride=2)
        self.scale_enc2 = nn.ModuleList([
            FNOBlock3d(scale_w2, scale_modes2, use_checkpoint=True) for _ in range(max(1, n_enc - 1))
        ])
        self.scale_down2 = nn.Conv3d(scale_w2, scale_w4, kernel_size=2, stride=2)
        self.scale_bottleneck = nn.ModuleList([
            FNOBlock3d(scale_w4, scale_modes4, use_checkpoint=True) for _ in range(max(1, n_bot))
        ])
        self.scale_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(scale_w4, scale_w2),
            nn.GELU(),
            nn.Linear(scale_w2, 1),
        )

        # Enhanced scale head v2: multi-scale pooling + structural features.
        # Activated by calling enable_enhanced_scale_head().
        n_struct_feat = 6  # vol_frac, surface_vol_ratio, min_cross_{x,y,z}, connectivity
        n_pool_4 = scale_w4 * 64  # 4x4x4 pool → 64 spatial cells
        self._enhanced_scale_head_dim = scale_w4 + n_pool_4 + n_struct_feat
        self.scale_head_v2_pool4 = nn.AdaptiveAvgPool3d(4)
        self.scale_head_v2 = nn.Sequential(
            nn.Linear(scale_w4 + n_pool_4 + n_struct_feat, scale_w2 * 2),
            nn.GELU(),
            nn.Linear(scale_w2 * 2, scale_w2),
            nn.GELU(),
            nn.Linear(scale_w2, 1),
        )
        self._use_enhanced_scale = False

        self._grid_cache = {}

    def _get_grid(self, shape: tuple, device: torch.device) -> torch.Tensor:
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

    def _field_forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.lifting(x)
        p = self.padding
        if p > 0:
            h1 = F.pad(h1, [p, p, p, p, p, p])

        for block in self.enc1:
            h1 = block(h1)
        skip1 = h1

        h2 = self.down1(h1)
        for block in self.enc2:
            h2 = block(h2)
        skip2 = h2

        h = self.down2(h2)
        for block in self.bottleneck:
            h = block(h)

        h = self.up2(h)
        if h.shape != skip2.shape:
            h = F.interpolate(h, size=skip2.shape[2:], mode='trilinear', align_corners=False)
        h = torch.cat([h, skip2], dim=1)
        h = self.dec2_proj(h)
        for block in self.dec2:
            h = block(h)

        h = self.up1(h)
        if h.shape != skip1.shape:
            h = F.interpolate(h, size=skip1.shape[2:], mode='trilinear', align_corners=False)
        h = torch.cat([h, skip1], dim=1)
        h = self.dec1_proj(h)
        for block in self.dec1:
            h = block(h)

        if p > 0:
            h = h[:, :, p:-p, p:-p, p:-p]

        disp = self.disp_head(h)
        stress = self.stress_head(h)
        return torch.cat([disp, stress], dim=1)

    def enable_enhanced_scale_head(self):
        """Switch to enhanced scale head with structural features + multi-scale pooling."""
        self._use_enhanced_scale = True

    def disable_enhanced_scale_head(self):
        """Revert to original scale head."""
        self._use_enhanced_scale = False

    @staticmethod
    def _compute_structural_features(density_ch: torch.Tensor) -> torch.Tensor:
        """Compute structural geometry features from density channel.

        Args:
            density_ch: [B, 1, N, N, N] density channel (first channel of input)
        Returns:
            [B, 6] structural features
        """
        B = density_ch.shape[0]
        feats = []
        for b in range(B):
            d = density_ch[b, 0]  # [N, N, N]
            solid = (d > 0.3).float()
            N = d.shape[0]
            vol_frac = solid.mean()

            # Surface-to-volume ratio (approximate via gradient magnitude)
            # Solid voxels adjacent to void
            padded = F.pad(solid.unsqueeze(0).unsqueeze(0), [1]*6, mode='constant', value=0)
            # 6-connected neighbors
            surface_count = (
                (solid - padded[0, 0, 2:, 1:-1, 1:-1]).abs() +
                (solid - padded[0, 0, :-2, 1:-1, 1:-1]).abs() +
                (solid - padded[0, 0, 1:-1, 2:, 1:-1]).abs() +
                (solid - padded[0, 0, 1:-1, :-2, 1:-1]).abs() +
                (solid - padded[0, 0, 1:-1, 1:-1, 2:]).abs() +
                (solid - padded[0, 0, 1:-1, 1:-1, :-2]).abs()
            )
            n_surface = (surface_count > 0.5).float().sum()
            n_solid = solid.sum().clamp(min=1)
            sv_ratio = n_surface / n_solid

            # Min cross-section area along each axis (normalized by N²)
            cross_x = solid.sum(dim=(1, 2))  # [N]
            cross_y = solid.sum(dim=(0, 2))  # [N]
            cross_z = solid.sum(dim=(0, 1))  # [N]
            # Only consider slices that have some solid
            min_cx = cross_x[cross_x > 0].min() / (N * N) if (cross_x > 0).any() else torch.tensor(0.0, device=d.device)
            min_cy = cross_y[cross_y > 0].min() / (N * N) if (cross_y > 0).any() else torch.tensor(0.0, device=d.device)
            min_cz = cross_z[cross_z > 0].min() / (N * N) if (cross_z > 0).any() else torch.tensor(0.0, device=d.device)

            # Connectivity: ratio of solid slices to total slices with solid
            n_slices_x = (cross_x > 0).float().sum() / N
            feats.append(torch.stack([vol_frac, sv_ratio, min_cx, min_cy, min_cz, n_slices_x]))

        return torch.stack(feats, dim=0)  # [B, 6]

    def _scale_forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.scale_lifting(x)
        h = self.scale_down1(h)
        for block in self.scale_enc2:
            h = block(h)
        h = self.scale_down2(h)
        for block in self.scale_bottleneck:
            h = block(h)
        self._scale_feat = h

        if self._use_enhanced_scale:
            # Multi-scale pooling
            pool1 = F.adaptive_avg_pool3d(h, 1).flatten(1)  # [B, C]
            pool4 = self.scale_head_v2_pool4(h).flatten(1)  # [B, C*64]
            # Structural features from density (first channel of input, before grid)
            density_ch = x[:, :1, :, :, :]  # density is first channel
            struct_feat = self._compute_structural_features(density_ch)
            combined = torch.cat([pool1, pool4, struct_feat], dim=1)
            return self.scale_head_v2(combined)
        else:
            return self.scale_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, Nx, Ny, Nz = x.shape
        grid = self._get_grid((Nx, Ny, Nz), x.device).expand(B, -1, -1, -1, -1)
        xg = torch.cat([x, grid], dim=1)
        fields = self._field_forward(xg)
        # keep scale branch cache available for predict_scale()
        _ = self._scale_forward(xg)
        return fields

    def predict_scale(self) -> torch.Tensor:
        if not hasattr(self, '_scale_feat') or self._scale_feat is None:
            raise RuntimeError("predict_scale() called before forward()")
        return self.scale_head(self._scale_feat)

    def forward_with_scale(self, x: torch.Tensor):
        B, _, Nx, Ny, Nz = x.shape
        grid = self._get_grid((Nx, Ny, Nz), x.device).expand(B, -1, -1, -1, -1)
        xg = torch.cat([x, grid], dim=1)
        fields = self._field_forward(xg)
        log_vm_max = self._scale_forward(xg)
        return fields, log_vm_max

    def predict_displacement_and_stress(self, x: torch.Tensor):
        out = self.forward(x)
        return out[:, :3], out[:, 3:]


class PINOLoss(nn.Module):
    """PINO损失函数: 数据损失 + PDE残差(平衡方程+本构关系) + 边界条件"""

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_pde: float = 0.1,
        lambda_bc: float = 1.0,
        lambda_constitutive: float = 0.1,
        resolution: int = 64,
        youngs_modulus: float = 3.5e9,
        poisson_ratio: float = 0.36,
        simp_penalty: float = 3.0,
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_constitutive = lambda_constitutive
        self.h = 1.0 / resolution
        self.E = youngs_modulus
        self.nu = poisson_ratio
        self.simp_penalty = simp_penalty

        lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        mu = self.E / (2 * (1 + self.nu))
        C = torch.zeros(6, 6)
        C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
        C[3, 3] = C[4, 4] = C[5, 5] = mu
        self.register_buffer('C_matrix', C)

    def _grad_3d(self, f, dim):
        spatial_dim = dim + 2
        N = f.shape[spatial_dim]
        h = self.h
        f_plus = torch.roll(f, -1, spatial_dim)
        f_minus = torch.roll(f, 1, spatial_dim)
        grad = (f_plus - f_minus) / (2 * h)
        idx_f = [slice(None)] * 5
        idx_f[spatial_dim] = 0
        idx_f1 = [slice(None)] * 5
        idx_f1[spatial_dim] = 1
        grad[tuple(idx_f)] = (f[tuple(idx_f1)] - f[tuple(idx_f)]) / h
        idx_b = [slice(None)] * 5
        idx_b[spatial_dim] = N - 1
        idx_b1 = [slice(None)] * 5
        idx_b1[spatial_dim] = N - 2
        grad[tuple(idx_b)] = (f[tuple(idx_b)] - f[tuple(idx_b1)]) / h
        return grad

    def equilibrium_residual(self, stress, force, density):
        sxx, syy, szz = stress[:, 0:1], stress[:, 1:2], stress[:, 2:3]
        syz, szx, sxy = stress[:, 3:4], stress[:, 4:5], stress[:, 5:6]
        res_x = self._grad_3d(sxx, 0) + self._grad_3d(sxy, 1) + self._grad_3d(szx, 2) + force[:, 0:1]
        res_y = self._grad_3d(sxy, 0) + self._grad_3d(syy, 1) + self._grad_3d(syz, 2) + force[:, 1:2]
        res_z = self._grad_3d(szx, 0) + self._grad_3d(syz, 1) + self._grad_3d(szz, 2) + force[:, 2:3]
        residual = torch.cat([res_x, res_y, res_z], dim=1)
        mask = (density > 0.1).float()
        char_scale = (force ** 2).mean().clamp(min=1e-8)
        return ((residual * mask) ** 2).mean() / char_scale

    def constitutive_residual(self, stress, displacement, density):
        ux, uy, uz = displacement[:, 0:1], displacement[:, 1:2], displacement[:, 2:3]
        exx = self._grad_3d(ux, 0)
        eyy = self._grad_3d(uy, 1)
        ezz = self._grad_3d(uz, 2)
        eyz = self._grad_3d(uy, 2) + self._grad_3d(uz, 1)
        ezx = self._grad_3d(uz, 0) + self._grad_3d(ux, 2)
        exy = self._grad_3d(ux, 1) + self._grad_3d(uy, 0)
        strain = torch.cat([exx, eyy, ezz, eyz, ezx, exy], dim=1)
        B, _, Nx, Ny, Nz = stress.shape
        strain_flat = strain.reshape(B, 6, -1)
        sigma_from_u = torch.einsum('ij,bjm->bim', self.C_matrix, strain_flat)
        sigma_from_u = sigma_from_u.reshape(B, 6, Nx, Ny, Nz)
        sigma_from_u = sigma_from_u * (density ** self.simp_penalty)
        residual = stress - sigma_from_u
        mask = (density > 0.1).float()
        char_scale = (stress ** 2).mean().clamp(min=1e-8)
        return ((residual * mask) ** 2).mean() / char_scale

    def forward(self, pred, target, density, fixed_mask, force=None, epoch=0):
        pred_disp = pred[:, :3]
        pred_stress = pred[:, 3:]
        target_disp = target[:, :3]
        target_stress = target[:, 3:]
        mask = (density > 0.1).float()
        l_data = F.mse_loss(pred * mask, target * mask)
        l_bc = ((pred_disp * fixed_mask) ** 2).mean()
        l_pde = torch.tensor(0.0, device=pred.device)
        l_const = torch.tensor(0.0, device=pred.device)
        if force is not None and self.lambda_pde > 0:
            l_pde = self.equilibrium_residual(pred_stress, force, density)
            l_const = self.constitutive_residual(pred_stress, pred_disp, density)
        total = (self.lambda_data * l_data
                 + self.lambda_bc * l_bc
                 + self.lambda_pde * l_pde
                 + self.lambda_constitutive * l_const)
        return {
            'total': total, 'data': l_data, 'bc': l_bc,
            'pde': l_pde, 'constitutive': l_const,
        }
