#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
OUT=${OUT:-$ROOT/results/dark_knight_pino_carve_stopping_pilot_20260721}
CKPT=${CKPT:-$ROOT/checkpoints/strict_physics_residual_pino_v16_shadowart_scale_20260718/best_model.pt}
ARTIFACT=${ARTIFACT:-$ROOT/results/dark_knight_fdm_7cm_fdmrank_20260713/soft_training}
INPUTS=${INPUTS:-$ROOT/results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709/dark_knight_left__magic_scholar_right/inputs}

L="$INPUTS/left_wall_x_view1_target.png"
R="$INPUTS/right_wall_y_view2_target.png"

common=(
  --silhouettes "$L" "$R"
  --branch soft-geometry
  --tsf 0 --mr 0
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

# The two stopping policies are the primary experiment.  max-rollbacks=0
# isolates the predicted stopping boundary; -1 enables conservative FEM rollback.
run_case pino_stop_no_rollback \
  --screen pino_rollback_fem --pino-checkpoint "$CKPT" \
  --pino-stop-margin 0.10 --pino-max-rollbacks 0

run_case pino_stop_fem_rollback \
  --screen pino_rollback_fem --pino-checkpoint "$CKPT" \
  --pino-stop-margin 0.10 --pino-max-rollbacks -1

# Oracle and geometry-only controls use exactly the same frozen soft field and
# deterministic carving trajectory.
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
(root / "pilot_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY
