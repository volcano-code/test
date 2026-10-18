#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
SOURCE="$ROOT/paper_aaai_submission/symbol_silhouettes_20260721"
INPUTS="$ROOT/data/physical_print_aaai_maple_legacy256_20260722"
OUT=${OUT:-$ROOT/results/physical_print_aaai_maple_dark_knight_legacy_20260722}

mkdir -p "$INPUTS"
"$PY" - "$SOURCE" "$INPUTS" <<'PY'
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from PIL import ImageDraw
from scipy.ndimage import distance_transform_edt, label


def connect_components(mask: Image.Image, width: int = 2) -> Image.Image:
    """Add minimal print-target bridges between disconnected foreground parts."""
    array = np.asarray(mask, dtype=np.uint8) > 0
    while True:
        labels, count = label(array)
        if count <= 1:
            return Image.fromarray(array.astype(np.uint8) * 255, mode="L")
        sizes = np.bincount(labels.ravel())[1:]
        anchor = labels == (int(np.argmax(sizes)) + 1)
        distance, nearest = distance_transform_edt(~anchor, return_indices=True)
        best = None
        for component in range(1, count + 1):
            part = labels == component
            if np.any(part & anchor):
                continue
            points = np.argwhere(part)
            index = int(np.argmin(distance[part]))
            y1, x1 = points[index]
            y0, x0 = nearest[:, y1, x1]
            candidate = (float(distance[y1, x1]), int(x0), int(y0), int(x1), int(y1))
            if best is None or candidate < best:
                best = candidate
        if best is None:
            return Image.fromarray(array.astype(np.uint8) * 255, mode="L")
        _, x0, y0, x1, y1 = best
        bridged = Image.fromarray(array.astype(np.uint8) * 255, mode="L")
        ImageDraw.Draw(bridged).line((x0, y0, x1, y1), fill=255, width=width)
        array = np.asarray(bridged, dtype=np.uint8) > 0

source, output = map(Path, sys.argv[1:])
for src_name, dst_name in (
    ("aaai_triangle_silhouette.png", "aaai_triangle_256.png"),
    ("maple_leaf_silhouette.png", "maple_leaf_256.png"),
):
    image = Image.open(source / src_name).convert("L")
    mask = image.point(lambda value: 255 if value >= 128 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError(f"empty silhouette: {src_name}")
    crop = mask.crop(bbox)
    target_h = 220
    target_w = max(1, round(crop.width * target_h / crop.height))
    target_w = min(target_w, 220)
    crop = crop.resize((target_w, target_h), Image.Resampling.NEAREST)
    canvas = Image.new("L", (256, 256), 0)
    canvas.paste(crop, ((256 - target_w) // 2, 18))
    canvas = connect_components(canvas)
    canvas.save(output / dst_name)
PY

cd "$ROOT"

"$PY" shadowforge_pipeline/unified_run.py \
  --silhouettes "$INPUTS/aaai_triangle_256.png" "$INPUTS/maple_leaf_256.png" \
  --branch soft-geometry \
  --output "$OUT" \
  --tsf 0 --mr 0 \
  --soft-resolution 256 \
  --soft-control-res 0 \
  --soft-proposal-mode regularized \
  --soft-iterations 2000 \
  --soft-lr 0.007 \
  --soft-save-every 100 \
  --soft-seed 42 \
  --soft-connectivity-start-iou 0.60 \
  --soft-fidelity-gate 0.85 \
  --soft-structure-warmup-steps 300 \
  --soft-structure-ramp-steps 300 \
  --soft-seed-thresholds 0.15,0.25,0.35,0.45,0.55 \
  --soft-disable-losses volume \
  --core-resolution 48 \
  --delivery-shell-voxels 2 \
  --print-mm 70 \
  --min-iou 0.975 \
  --min-view-iou 0.95 \
  --min-saving 0.20 \
  --max-saving 0.65 \
  --min-erosion-survival 0.18 \
  --max-unsupported-ratio 0.10 \
  --max-support-islands 1000000000 \
  --min-bed-area-mm2 10 \
  --selection-iou-tolerance 0.01 \
  --selection-policy fdm \
  --screen fem_only \
  --structural-resolution 48 \
  --sf-threshold 1.5 \
  --load-multiplier 1.0 \
  --display-fixture \
  --fixture-margin-mm 8 \
  --fixture-base-thickness-mm 4 \
  --fixture-clearance-mm 0.8 \
  --fixture-retainer-height-mm 1.5 \
  --save-candidate-evidence \
  --device cuda:0
