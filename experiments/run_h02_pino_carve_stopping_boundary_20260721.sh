#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
OUT=${OUT:-$ROOT/results/h02_sword_fox_pino_carve_stopping_boundary_20260721}
CKPT=${CKPT:-$ROOT/checkpoints/strict_physics_residual_pino_v16_shadowart_scale_20260718/best_model.pt}
SOURCE=${SOURCE:-$ROOT/results/physical_print_soft_geometry_20260714/H02_sword_fox}

L="$SOURCE/inputs/left_wall_x_target.png"
R="$SOURCE/inputs/right_wall_y_target.png"
ARTIFACT="$SOURCE/soft_training"

common=(
  --silhouettes "$L" "$R"
  --branch soft-geometry
  --tsf 0 --mr 1
  --soft-resolution 256
  --soft-control-res 0
  --core-resolution 48
  --soft-proposal-mode proposal
  --soft-disable-losses volume
  --soft-artifact-dir "$ARTIFACT"
  --soft-seed-thresholds 0.15,0.25,0.35,0.45,0.55
  --delivery-shell-voxels 2
  --min-iou 0.975
  --min-view-iou 0.95
  --min-saving 0.0
  --max-saving 0.85
  --selection-iou-tolerance 0.01
  --selection-policy material
  --structural-resolution 48
  --sf-threshold 1.3
  --print-mm 70
  --device cuda:0
  --save-candidate-evidence
)

run_case() {
  local name=$1
  shift
  local dest="$OUT/$name"
  if [[ -f "$dest/MANIFEST.json" ]]; then
    echo "skip completed $name"
    return
  fi
  "$PY" "$ROOT/shadowforge_pipeline/unified_run.py" \
    "${common[@]}" --output "$dest" "$@"
}

mkdir -p "$OUT"

run_case pino_stop_no_rollback \
  --screen pino_rollback_fem --pino-checkpoint "$CKPT" \
  --pino-stop-margin 0.10 --pino-max-rollbacks 0

run_case pino_stop_fem_rollback \
  --screen pino_rollback_fem --pino-checkpoint "$CKPT" \
  --pino-stop-margin 0.10 --pino-max-rollbacks -1

run_case exhaustive_fem --screen fem_only
run_case material_only --screen skip

"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/MANIFEST.json")):
    data = json.loads(path.read_text())
    selected = data.get("selected") or {}
    policy = data.get("pino_structural_policy") or {}
    rows.append({
        "policy": path.parent.name,
        "status": data.get("status", "SUCCESS"),
        "candidate": selected.get("candidate"),
        "mean_iou": selected.get("mean_iou"),
        "min_iou": selected.get("min_iou"),
        "saving": selected.get("saving"),
        "fabrication_pass": selected.get("fabrication_pass"),
        "structural_pass": selected.get("structural_pass"),
        "sf_fem": (selected.get("structural_screen") or {}).get("sf_fem"),
        "candidate_count": policy.get("candidate_count"),
        "fem_calls": policy.get("fem_calls"),
        "fem_call_reduction_rate": policy.get("fem_call_reduction_rate"),
        "trajectories": policy.get("trajectories"),
    })
(root / "boundary_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY
