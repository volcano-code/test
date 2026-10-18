from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import torch


_HERE = Path(__file__).resolve().parent
_HELPER = _HERE / "render_bridge_py310.py"
_PYTHON310 = Path(
    os.environ.get(
        "PCRAFT_PYTHON310",
        "/home/lin/anaconda3/envs/shadowart-revisited/bin/python",
    )
)
_CACHE_DIR = Path(
    os.environ.get(
        "PCRAFT_SADEKAR_BRIDGE_CACHE",
        str(_HERE / "cache"),
    )
)
_MEMORY_CACHE: dict[str, np.ndarray] = {}


class SadekarTFSProtocol:
    """Proxy the legacy renderer to the Python 3.10 PyTorch3D environment."""

    def __init__(
        self,
        device: torch.device | str = "cuda:0",
        image_size: int = 256,
        zdist: float = 2.5,
        volume_extent_world: float = 1.7,
        n_pts_per_ray: int = 150,
        **_: object,
    ) -> None:
        self.image_size = int(image_size)
        self.zdist = float(zdist)
        self.volume_extent = float(volume_extent_world)
        self.n_pts_per_ray = int(n_pts_per_ray)
        self.device = str(device)

    def _cache_key(self, densities: torch.Tensor) -> str:
        array = (
            densities.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
        )
        digest = hashlib.sha256()
        digest.update(array.tobytes())
        digest.update(
            (
                f"|{self.image_size}|{self.zdist:.9g}|"
                f"{self.volume_extent:.9g}|{self.n_pts_per_ray}"
            ).encode("ascii")
        )
        return digest.hexdigest()

    def _render_all(self, densities: torch.Tensor) -> np.ndarray:
        key = self._cache_key(densities)
        if key in _MEMORY_CACHE:
            return _MEMORY_CACHE[key]

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        density_path = _CACHE_DIR / f"{key}.npy"
        render_path = _CACHE_DIR / f"{key}.npz"

        if not render_path.exists():
            density = (
                densities.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
            )
            np.save(density_path, density)
            subprocess.run(
                [
                    str(_PYTHON310),
                    str(_HELPER),
                    "--density",
                    str(density_path),
                    "--output",
                    str(render_path),
                    "--image-size",
                    str(self.image_size),
                    "--zdist",
                    str(self.zdist),
                    "--volume-extent",
                    str(self.volume_extent),
                    "--n-pts-per-ray",
                    str(self.n_pts_per_ray),
                    "--device",
                    self.device,
                ],
                check=True,
            )

        with np.load(render_path) as payload:
            views = payload["views"].astype(np.float32, copy=True)
        _MEMORY_CACHE[key] = views
        return views

    def render(self, densities: torch.Tensor, view_index: int) -> torch.Tensor:
        views = self._render_all(densities)
        return torch.from_numpy(views[int(view_index)]).to(densities.device)

    @staticmethod
    def binary_mask(rendered: torch.Tensor) -> torch.Tensor:
        maximum = rendered.max()
        if float(maximum) <= 0.0:
            return torch.zeros_like(rendered, dtype=torch.bool)
        return rendered >= (0.8 * maximum)
