"""Baseline neural operator architectures for comparison.

Implements:
  - DeepONet3d: Branch-trunk architecture adapted for 3D voxel stress prediction
  - GeoFNO3d: Geometry-aware FNO with coordinate deformation (Li et al., 2022)

Both models accept the same input format as PINO/UFNO:
  x: [B, C_in, N, N, N]  (C_in = 5: density, force_xyz, sdf)
and output:
  out: [B, 9, N, N, N]  (3 displacement + 6 stress components)

Scale head: both include a scale_head for log10(vm_max) prediction,
compatible with the UFNO-dual training pipeline.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# DeepONet3d — Branch-Trunk Architecture for 3D Structural Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class DeepONet3d(nn.Module):
    """DeepONet adapted for 3D voxel-to-voxel stress prediction.

    Branch network encodes the input function (density + force + SDF) via a
    3D-CNN encoder.  Trunk network encodes spatial coordinates via an MLP.
    The output at each query point is the inner product of branch and trunk
    representations, producing per-voxel displacement + stress.

    Reference: Lu et al., "Learning nonlinear operators via DeepONet", 2021.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 9,
        branch_width: int = 64,
        trunk_width: int = 128,
        n_basis: int = 64,
        resolution: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_basis = n_basis
        self.resolution = resolution

        # Branch: 3D CNN encoder
        self.branch = nn.Sequential(
            nn.Conv3d(in_channels, branch_width, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(branch_width, branch_width, 3, stride=2, padding=1),  # 64→32
            nn.GELU(),
            nn.Conv3d(branch_width, branch_width * 2, 3, stride=2, padding=1),  # 32→16
            nn.GELU(),
            nn.Conv3d(branch_width * 2, branch_width * 4, 3, stride=2, padding=1),  # 16→8
            nn.GELU(),
            nn.AdaptiveAvgPool3d(4),  # → 4³
            nn.Flatten(),
            nn.Linear(branch_width * 4 * 64, branch_width * 4),
            nn.GELU(),
            nn.Linear(branch_width * 4, n_basis * out_channels),
        )

        # Trunk: MLP on coordinates
        self.trunk = nn.Sequential(
            nn.Linear(3, trunk_width),
            nn.GELU(),
            nn.Linear(trunk_width, trunk_width),
            nn.GELU(),
            nn.Linear(trunk_width, trunk_width),
            nn.GELU(),
            nn.Linear(trunk_width, n_basis),
        )

        # Bias for each output channel
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Scale head for log10(vm_max)
        self.scale_head = nn.Sequential(
            nn.Linear(n_basis * out_channels, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self._has_scale_head = True

        # Initialize weights with small values to prevent exploding outputs
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_coords(self, N: int, device: torch.device) -> torch.Tensor:
        """Build [N,N,N,3] coordinate grid on the given device."""
        lin = torch.linspace(0, 1, N, device=device)
        gz, gy, gx = torch.meshgrid(lin, lin, lin, indexing='ij')
        return torch.stack([gx, gy, gz], dim=-1)

    def _forward_core(self, x: torch.Tensor):
        B, C, N, _, _ = x.shape
        device = x.device

        # Branch: [B, n_basis * out_channels]
        branch_out = self.branch(x)

        # Trunk: [N³, n_basis] — always on same device as x
        coords = self._get_coords(N, device)
        trunk_out = self.trunk(coords.reshape(-1, 3))  # [N³, n_basis]

        # Reshape branch: [B, out_channels, n_basis]
        branch_reshape = branch_out.view(B, self.out_channels, self.n_basis)

        # Output: [B, out_channels, N³] via einsum
        out = torch.einsum('bop,sp->bos', branch_reshape, trunk_out)
        out = out + self.bias.view(1, -1, 1)
        out = out.view(B, self.out_channels, N, N, N)

        log_vm = self.scale_head(branch_out)
        return out, log_vm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self._forward_core(x)
        return out

    def forward_with_scale(self, x: torch.Tensor):
        return self._forward_core(x)


# ═══════════════════════════════════════════════════════════════════════════════
# GeoFNO3d — Geometry-Aware Fourier Neural Operator
# ═══════════════════════════════════════════════════════════════════════════════

class SpectralConv3d(nn.Module):
    """Standard 3D spectral convolution for FNO."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, modes,
                                dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Sx, Sy, Sz = x.shape
        m = self.modes
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(B, self.out_channels, Sx, Sy, Sz // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m, :m, :m] = torch.einsum(
            'bixyz,ioxyz->boxyz', x_ft[:, :, :m, :m, :m], self.weights
        )
        return torch.fft.irfftn(out_ft, s=[Sx, Sy, Sz])


class GeoFNO3d(nn.Module):
    """Geometry-aware FNO (Geo-FNO) for 3D structural analysis.

    Key idea: learn a coordinate deformation from physical space to a latent
    uniform space where spectral convolutions are more effective.  A small
    deformation network predicts per-voxel displacements that warp the
    coordinate grid; the main FNO operates on the deformed representation.

    Reference: Li et al., "Fourier Neural Operator with Learned
    Deformations for PDEs on General Geometries", JMLR 2023.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 9,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width

        # Deformation network: predict per-voxel coordinate offsets
        self.deform_net = nn.Sequential(
            nn.Conv3d(in_channels, width, 1),
            nn.GELU(),
            nn.Conv3d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(width, 3, 1),
            nn.Tanh(),  # bounded offsets
        )
        self.deform_scale = nn.Parameter(torch.tensor(0.1))  # learnable offset magnitude

        # Lifting: input + 3 coord channels → width
        self.lifting = nn.Conv3d(in_channels + 3, width, 1)

        # FNO layers
        self.spectral_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv3d(width, width, modes))
            self.conv_layers.append(nn.Conv3d(width, width, 1))

        # Projection head
        self.proj = nn.Sequential(
            nn.Conv3d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv3d(width * 2, out_channels, 1),
        )

        # Scale head for log10(vm_max)
        self.scale_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self._has_scale_head = True

    def _get_base_coords(self, N: int, device: torch.device) -> torch.Tensor:
        """Return [1, 3, N, N, N] coordinate grid in [0, 1]."""
        lin = torch.linspace(0, 1, N, device=device)
        gz, gy, gx = torch.meshgrid(lin, lin, lin, indexing='ij')
        return torch.stack([gx, gy, gz], dim=0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self._forward_impl(x)
        return out

    def forward_with_scale(self, x: torch.Tensor):
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor):
        B, C, N, _, _ = x.shape

        # Deformation: learn coordinate offsets
        deform = self.deform_net(x) * self.deform_scale  # [B, 3, N, N, N]
        base_coords = self._get_base_coords(N, x.device).expand(B, -1, -1, -1, -1)
        deformed_coords = base_coords + deform

        # Lifting with deformed coordinates
        h = self.lifting(torch.cat([x, deformed_coords], dim=1))

        # FNO layers
        for sp_layer, conv_layer in zip(self.spectral_layers, self.conv_layers):
            h1 = sp_layer(h)
            h2 = conv_layer(h)
            h = h + F.gelu(h1 + h2)

        # Scale prediction from last hidden state
        log_vm = self.scale_head(h)

        # Output projection
        out = self.proj(h)
        return out, log_vm
