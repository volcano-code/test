from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch


REAL_PROTOCOL = Path(
    "/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/"
    "XUXINYAO/shadow-project2/mesh/voxel/sadekar_protocol.py"
)


def load_real_protocol():
    spec = importlib.util.spec_from_file_location(
        "_pcraft_real_sadekar_protocol",
        REAL_PROTOCOL,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Sadekar protocol from {REAL_PROTOCOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SadekarTFSProtocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--zdist", type=float, required=True)
    parser.add_argument("--volume-extent", type=float, required=True)
    parser.add_argument("--n-pts-per-ray", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_class = load_real_protocol()
    device = torch.device(args.device)
    protocol = protocol_class(
        image_size=args.image_size,
        zdist=args.zdist,
        volume_extent_world=args.volume_extent,
        n_pts_per_ray=args.n_pts_per_ray,
        device=device,
    )
    density = torch.from_numpy(np.load(args.density)).to(device=device)

    rendered_views = []
    with torch.no_grad():
        for view_index in range(6):
            rendered = protocol.render(density, view_index)
            rendered_views.append(
                rendered.detach().to(device="cpu", dtype=torch.float32).numpy()
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, views=np.stack(rendered_views, axis=0))


if __name__ == "__main__":
    main()
