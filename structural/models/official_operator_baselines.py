"""Adapters for official GINO and Transolver implementations.

The adapters only reconcile the official model APIs with P-CRAFT's dense C48
tensor contract.  They do not reimplement either operator.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[2]
_TRANSOLVER_ROOT = _ROOT / "external_baselines" / "Transolver" / "PDE-Solving-StandardBenchmark"


def _coordinate_grid(resolution: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, resolution, device=device, dtype=dtype)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    return grid


class _DenseOperatorAdapter(nn.Module):
    """Shared dense-output and scalar-scale contract used by the trainer."""

    _scale_branch_prefixes = ("scale_head",)

    def __init__(self, in_channels: int, out_channels: int, scale_width: int):
        super().__init__()
        pooled_channels = 2 * (in_channels + out_channels)
        self.scale_head = nn.Sequential(
            nn.Linear(pooled_channels, scale_width),
            nn.GELU(),
            nn.Linear(scale_width, 1),
        )
        self._detach_scale_head_input = False

    def _predict_scale(self, x: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        features = torch.cat((x, field), dim=1)
        pooled = torch.cat(
            (
                features.mean(dim=(2, 3, 4)),
                features.std(dim=(2, 3, 4), unbiased=False),
            ),
            dim=1,
        )
        if self._detach_scale_head_input:
            pooled = pooled.detach()
        return self.scale_head(pooled)

    def forward_with_scale(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        field = self.forward(x)
        return field, self._predict_scale(x, field)


class OfficialGINO3d(_DenseOperatorAdapter):
    """Official neuraloperator GINO with dense-grid input/output adaptation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 9,
        modes: int = 8,
        width: int = 48,
        n_layers: int = 4,
        padding: int = 0,
        resolution: int = 48,
        latent_resolution: int = 12,
    ):
        del padding
        super().__init__(in_channels, out_channels, scale_width=max(32, width))
        try:
            from neuralop.models import GINO
        except ImportError as exc:
            raise ImportError(
                "Official GINO requires the pinned neuraloperator checkout. "
                "Run `python -m pip install -e external_baselines/neuraloperator`."
            ) from exc

        self.resolution = int(resolution)
        self.latent_resolution = min(int(latent_resolution), self.resolution)
        latent_spacing = 1.0 / max(self.latent_resolution - 1, 1)
        radius = min(0.5, 1.35 * latent_spacing)
        n_modes = tuple(
            min(int(modes), max(2, self.latent_resolution // 2)) for _ in range(3)
        )
        self.operator = GINO(
            in_channels=in_channels,
            out_channels=out_channels,
            gno_coord_dim=3,
            in_gno_radius=radius,
            out_gno_radius=radius,
            # For the official linear integral transform, kernel and input
            # feature widths must agree before the pointwise lifting layer.
            fno_in_channels=in_channels,
            fno_n_modes=n_modes,
            fno_hidden_channels=width,
            fno_n_layers=n_layers,
            gno_embed_channels=max(16, width // 2),
            gno_use_open3d=False,
            gno_use_torch_scatter=False,
        )
        self.register_buffer("_dense_grid", torch.empty(0), persistent=False)
        self.register_buffer("_latent_grid", torch.empty(0), persistent=False)

    def _grids(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._dense_grid.numel() == 0
            or self._dense_grid.device != x.device
            or self._dense_grid.dtype != x.dtype
        ):
            self._dense_grid = _coordinate_grid(self.resolution, x.device, x.dtype)
            self._latent_grid = _coordinate_grid(
                self.latent_resolution, x.device, x.dtype
            )
        return self._dense_grid, self._latent_grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-3:] != (self.resolution,) * 3:
            raise ValueError(
                f"GINO adapter expects {self.resolution}^3 input, got {tuple(x.shape[-3:])}"
            )
        dense_grid, latent_grid = self._grids(x)
        batch = x.shape[0]
        point_features = x.permute(0, 2, 3, 4, 1).reshape(
            batch, self.resolution**3, x.shape[1]
        )
        out = self.operator(
            input_geom=dense_grid.reshape(1, -1, 3),
            latent_queries=latent_grid.unsqueeze(0),
            output_queries=latent_grid.reshape(1, -1, 3),
            x=point_features,
        )
        latent_field = out.reshape(
            batch,
            self.latent_resolution,
            self.latent_resolution,
            self.latent_resolution,
            -1,
        ).permute(0, 4, 1, 2, 3).contiguous()
        return F.interpolate(
            latent_field,
            size=(self.resolution,) * 3,
            mode="trilinear",
            align_corners=True,
        )


def _load_official_transolver():
    module_path = _TRANSOLVER_ROOT / "model" / "Transolver_Structured_Mesh_3D.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Official Transolver checkout is missing: {module_path}"
        )
    root_text = str(_TRANSOLVER_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location(
        "pcraft_official_transolver_3d", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official Transolver module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


class OfficialTransolver3d(_DenseOperatorAdapter):
    """Official Transolver structured-mesh 3D model with C48 adaptation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 9,
        modes: int = 16,
        width: int = 128,
        n_layers: int = 5,
        padding: int = 0,
        resolution: int = 48,
        slice_num: int = 32,
    ):
        del modes, padding
        super().__init__(in_channels, out_channels, scale_width=max(32, width // 2))
        Model = _load_official_transolver()
        n_head = 8 if width % 8 == 0 else 4
        self.resolution = int(resolution)
        self.operator = Model(
            space_dim=3,
            n_layers=n_layers,
            n_hidden=width,
            n_head=n_head,
            mlp_ratio=1,
            fun_dim=in_channels,
            out_dim=out_channels,
            slice_num=slice_num,
            unified_pos=False,
            H=self.resolution,
            W=self.resolution,
            D=self.resolution,
        )
        self.register_buffer("_grid", torch.empty(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-3:] != (self.resolution,) * 3:
            raise ValueError(
                "Transolver adapter expects "
                f"{self.resolution}^3 input, got {tuple(x.shape[-3:])}"
            )
        if self._grid.numel() == 0 or self._grid.device != x.device or self._grid.dtype != x.dtype:
            self._grid = _coordinate_grid(self.resolution, x.device, x.dtype).reshape(
                1, self.resolution**3, 3
            )
        batch = x.shape[0]
        coords = self._grid.expand(batch, -1, -1)
        features = x.permute(0, 2, 3, 4, 1).reshape(
            batch, self.resolution**3, x.shape[1]
        )
        out = self.operator(coords, features)
        return out.reshape(
            batch, self.resolution, self.resolution, self.resolution, -1
        ).permute(0, 4, 1, 2, 3).contiguous()
