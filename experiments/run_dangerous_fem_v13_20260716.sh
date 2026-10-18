#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
BASE="$ROOT/data/taskgrouped_strict_pino_v11_broad_boundary_20260716/index.json"
PRIOR="$ROOT/data/taskgrouped_strict_pino_v12_near_boundary_unsafe_20260716/index.json"
ADDON="$ROOT/data/taskgrouped_strict_pino_v13_dangerous_20260716"
UNION="$ROOT/data/taskgrouped_strict_pino_v13_union_20260716/index.json"

cd "$ROOT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

"$PY" experiments/build_dangerous_fem_v13_20260716.py \
  --source-index "$BASE" \
  --prior-augmentation "$PRIOR" \
  --output "$ADDON" \
  --source-sf-low 0.90 \
  --source-sf-high 3.50 \
  --target-bins 0.35:0.75:300,0.75:0.95:300,0.95:1.15:450,1.15:1.30:450 \
  --max-per-task 16 \
  --seeds-per-task 12 \
  --max-evaluations 8000

"$PY" experiments/merge_pino_training_augmentation_20260716.py \
  --base-index "$BASE" \
  --augmentation-index "$ADDON/index.json" \
  --output "$UNION"

"$PY" - "$UNION" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
for split in ("train", "calibration", "sealed_test"):
    rows = [row for row in data["rows"] if row["split"] == split]
    print(split, "rows=", len(rows), "tasks=", len({row["task"] for row in rows}),
          "classes=", Counter("safe" if float(row["sf"]) >= 1.3 else "unsafe" for row in rows))
PY
