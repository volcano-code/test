#!/usr/bin/env bash
# Evaluate the frozen fixed3/mr=1 material trajectories with Raw PINO ranking
# and exhaustive voxel FEM. This script intentionally reuses completed soft
# artifacts and therefore performs no differentiable-proposal retraining.
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
BENCH=${BENCH:-$ROOT/results/fixed3_mr1_connected10_pending_20260722/benchmark_manifest.json}
FROZEN=${FROZEN:-$ROOT/results/fixed3_mr1_connected10_pcraft_20260722}
OUT=${OUT:-$ROOT/results/formal_material_trajectory_fixed3_20260722}
CKPT=${CKPT:-$ROOT/checkpoints/strict_physics_residual_pino_v16_shadowart_scale_20260718/best_model.pt}
POLICY=${POLICY:-$ROOT/results/pino_ranked_fem_taskdisjoint_v16_20260720/frozen_policy.json}

mkdir -p "$OUT"

# Keep this experiment queued behind the two GPU-facing jobs that were already
# active when it was requested. The waiting service itself consumes no GPU.
for unit in \
  pcraft-fig4-blade-heroes-fixed3-20260722.service \
  pcraft-cv-pino-closed-loop-20260722.service; do
  while systemctl --user is-active --quiet "$unit"; do
    printf '[queue] waiting for %s\n' "$unit"
    sleep 60
  done
done

mapfile -t CASES < <("$PY" - "$BENCH" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1]))["cases"]:
    print(row["case_id"])
PY
)

run_case() {
  local case_id=$1
  local mode=$2
  local screen=$3
  local artifact="$FROZEN/$case_id/seed_17/soft_training"
  local dest="$OUT/$case_id/$mode"

  if [[ -f "$dest/MANIFEST.json" ]]; then
    printf '[reuse] %s %s\n' "$case_id" "$mode"
    return
  fi
  if [[ ! -f "$artifact/metrics.json" ]]; then
    printf '[skip] missing frozen artifact: %s\n' "$artifact" >&2
    return
  fi

  mapfile -t views < <("$PY" - "$BENCH" "$case_id" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
row = next(item for item in data["cases"] if item["case_id"] == sys.argv[2])
print("\n".join(row["target_paths_xyz"]))
PY
)
  mkdir -p "$dest"
  args=(
    --silhouettes "${views[@]}"
    --output "$dest" --branch soft-geometry --tsf 1 --mr 1
    --soft-resolution 256 --soft-control-res 0 --soft-proposal-mode proposal
    --soft-artifact-dir "$artifact" --core-resolution 48
    --soft-seed-thresholds 0.05,0.10,0.15,0.20,0.25,0.35,0.45,0.55,0.65
    --bridge-radius 2 --bridge-passes 48 --delivery-shell-voxels 2
    --min-iou 0.95 --min-view-iou 0.90 --min-saving 0.000001 --max-saving 0.85
    --selection-iou-tolerance 0.01 --selection-policy material
    --structural-resolution 48 --sf-threshold 1.3 --screen "$screen"
    --device cuda:0
  )
  if [[ "$screen" == pino_ranked_fem ]]; then
    args+=(--pino-checkpoint "$CKPT" --pino-ranked-policy "$POLICY"
           --pino-top-k 3 --pino-top-k-fallback expand)
  fi
  "$PY" "$ROOT/shadowforge_pipeline/unified_run.py" "${args[@]}" \
    > "$dest/RUN_OUTPUT.txt" 2>&1
}

for case_id in "${CASES[@]}"; do
  run_case "$case_id" pino_ranked_fem pino_ranked_fem
  run_case "$case_id" exhaustive_fem fem_only
done

"$PY" - "$OUT" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
rows = []
for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    pino_path = case_dir / "pino_ranked_fem/MANIFEST.json"
    fem_path = case_dir / "exhaustive_fem/MANIFEST.json"
    if not (pino_path.is_file() and fem_path.is_file()):
        continue
    pino = json.loads(pino_path.read_text())
    fem = json.loads(fem_path.read_text())
    pino_by_name = {row["candidate"]: row for row in pino["candidates"]}
    for fem_row in fem["candidates"]:
        name = fem_row["candidate"]
        pino_row = pino_by_name.get(name, {})
        rows.append({
            "case_id": case_dir.name,
            "candidate": name,
            "mean_iou": fem_row.get("mean_iou"),
            "min_iou": fem_row.get("min_iou"),
            "mean_dice": fem_row.get("mean_dice"),
            "saving": fem_row.get("saving"),
            "single_body": fem_row.get("components") == 1,
            "watertight": bool(fem_row.get("watertight")),
            "sf_fem": (fem_row.get("structural_screen") or {}).get("sf_fem"),
            "fem_safe": bool(fem_row.get("structural_pass")),
            "sf_pino": (pino_row.get("pino_prediction") or {}).get("sf_pino"),
            "pino_monotone_sf": pino_row.get("pino_monotone_sf"),
            "pino_rank": pino_row.get("pino_rank"),
            "fem_checked_by_ranked_policy": bool(pino_row.get("fem_checked", False)),
        })

summary = {
    "schema": "pcraft_formal_material_trajectory_fixed3_v1",
    "protocol": "10 fixed-three-view tasks, mr=1, seed=17, frozen 256^3 soft fields",
    "cases": len({row["case_id"] for row in rows}),
    "candidate_points": len(rows),
    "mean_iou": float(np.mean([r["mean_iou"] for r in rows])) if rows else None,
    "mean_dice": float(np.mean([r["mean_dice"] for r in rows])) if rows else None,
    "mean_saving": float(np.mean([r["saving"] for r in rows])) if rows else None,
    "single_body_rate": float(np.mean([r["single_body"] for r in rows])) if rows else None,
    "watertight_rate": float(np.mean([r["watertight"] for r in rows])) if rows else None,
    "pino_fem_pairs": sum(r["sf_pino"] is not None and r["sf_fem"] is not None for r in rows),
}
(root / "trajectory_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
PY

