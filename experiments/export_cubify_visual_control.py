#!/usr/bin/env python3
"""Export a binary occupancy volume with PyTorch3D cubify.

This is a *visualization control*: it does not repair, smooth, fill, or
re-score geometry.  It lets a method's native binary voxel grid be displayed
with exactly the same cube-surface operator used by the Sadekar voxel baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from pytorch3d.io import save_obj
from pytorch3d.ops import cubify


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help=".npy occupancy volume")
    p.add_argument("output", type=Path, help="output OBJ")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    vol = np.squeeze(np.load(args.input))
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3-D occupancy volume, got {vol.shape}")
    binary = torch.from_numpy((vol > args.threshold).astype(np.float32))[None]
    if not bool(binary.any()):
        raise ValueError("No occupied voxels after thresholding")

    # cubify is deliberately the only geometry operation in this script.
    mesh = cubify(binary, thresh=0.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_obj(str(args.output), mesh.verts_packed().cpu(), mesh.faces_packed().cpu())
    np.save(args.output.with_suffix(".binary_voxels.npy"), binary[0].numpy().astype(np.uint8))
    print(f"cubify: {args.input} -> {args.output}")
    print(f"grid={tuple(vol.shape)}, occupied={int(binary.sum())}, threshold={args.threshold}")


if __name__ == "__main__":
    main()
