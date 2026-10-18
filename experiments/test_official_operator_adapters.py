#!/usr/bin/env python3
"""Fast shape/gradient checks for the official operator adapters."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structural.models.official_operator_baselines import (
    OfficialGINO3d,
    OfficialTransolver3d,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        choices=("gino_official", "transolver_official", "all"),
        default="all",
    )
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    classes = {
        "gino_official": OfficialGINO3d,
        "transolver_official": OfficialTransolver3d,
    }
    selected = classes if args.arch == "all" else {args.arch: classes[args.arch]}

    for name, cls in selected.items():
        kwargs = dict(
            in_channels=6,
            out_channels=9,
            width=16 if name == "gino_official" else 32,
            n_layers=2,
            resolution=args.resolution,
        )
        if name == "gino_official":
            kwargs.update(modes=3, latent_resolution=min(6, args.resolution))
        model = cls(**kwargs).to(device)
        x = torch.randn(1, 6, *(args.resolution,) * 3, device=device)
        field, scale = model.forward_with_scale(x)
        assert field.shape == (1, 9, *(args.resolution,) * 3), field.shape
        assert scale.shape == (1, 1), scale.shape
        (field.square().mean() + scale.square().mean()).backward()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            {
                "arch": name,
                "field_shape": tuple(field.shape),
                "scale_shape": tuple(scale.shape),
                "trainable_parameters": trainable,
                "device": str(device),
            }
        )


if __name__ == "__main__":
    main()
