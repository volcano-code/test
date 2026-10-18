#!/usr/bin/env bash
# Controlled self-weight boundary trajectory for the material-selection figure.
# The protocol fixes the three silhouettes, frozen 256^3 soft field, material,
# camera, and carving schedule. Only the physical display scale is set to
# 210 mm so that the same trajectory spans the SF=1.3 structural boundary.
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
CASE_ID=${CASE_ID:-connected_06}
BENCH=${BENCH:-$ROOT/results/fixed3_mr1_connected10_pending_20260722/benchmark_manifest.json}
FROZEN=${FROZEN:-$ROOT/results/fixed3_mr1_connected10_pcraft_20260722}
OUT=${OUT:-$ROOT/results/connected06_boundary_material_trajectory_20260723}
CKPT=${CKPT:-$ROOT/checkpoints/strict_physics_residual_pino_v16_shadowart_scale_20260718/best_model.pt}
PRINT_MM=${PRINT_MM:-210}
CARVE_CHECKPOINT_VOXELS=${CARVE_CHECKPOINT_VOXELS:-0}
MAX_CARVE_CHECKPOINTS=${MAX_CARVE_CHECKPOINTS:-256}
STOP_AT_FIRST_GATE_FAILURE=${STOP_AT_FIRST_GATE_FAILURE:-0}

ARTIFACT="$FROZEN/$CASE_ID/seed_17/soft_training"
mkdir -p "$OUT"

mapfile -t VIEWS < <("$PY" - "$BENCH" "$CASE_ID" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
case = next(row for row in data["cases"] if row["case_id"] == sys.argv[2])
print("\n".join(case["target_paths_xyz"]))
PY
)

if [[ ! -f "$ARTIFACT/metrics.json" ]]; then
  echo "Missing frozen soft artifact: $ARTIFACT/metrics.json" >&2
  exit 2
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Missing PINO checkpoint: $CKPT" >&2
  exit 2
fi

COMMON=(
  --silhouettes "${VIEWS[@]}"
  --output ""
  --branch soft-geometry
  --tsf 1 --mr 1
  --soft-resolution 256 --soft-control-res 0
  --soft-proposal-mode proposal --soft-artifact-dir "$ARTIFACT"
  --core-resolution 48
  --soft-seed-thresholds 0.55
  --bridge-radius 2 --bridge-passes 48
  --delivery-shell-voxels 2
  --min-iou 0.95 --min-view-iou 0.90
  --min-saving 0.000001 --max-saving 0.85
  --allow-best-effort-fidelity-fallback
  --selection-iou-tolerance 0.01 --selection-policy material
  --structural-resolution 48 --sf-threshold 1.3
  --print-mm "$PRINT_MM"
  --device cuda:0 --save-candidate-evidence
)
if (( CARVE_CHECKPOINT_VOXELS > 0 )); then
  COMMON+=(
    --carve-checkpoint-voxels "$CARVE_CHECKPOINT_VOXELS"
    --max-carve-checkpoints "$MAX_CARVE_CHECKPOINTS"
  )
fi

run_case() {
  local name=$1
  shift
  local dest="$OUT/$name"
  # The shell opens the redirected log before unified_run.py can create its output.
  mkdir -p "$dest"
  if [[ -f "$dest/MANIFEST.json" ]]; then
    echo "[reuse] $name"
    return
  fi
  local args=("${COMMON[@]}")
  for i in "${!args[@]}"; do
    [[ "${args[$i]}" == "--output" ]] && args[$((i + 1))]="$dest"
  done
  "$PY" "$ROOT/shadowforge_pipeline/unified_run.py" "${args[@]}" "$@" \
    > "$dest/RUN_OUTPUT.txt" 2>&1
}

# Ground truth for the single, fixed trajectory. In gate-driven mode the first
# failed state is retained as boundary evidence and all lighter states are
# omitted; the immediately preceding state is the rollback target.
FEM_ARGS=(--screen fem_only)
if (( STOP_AT_FIRST_GATE_FAILURE == 1 )); then
  FEM_ARGS+=(--stop-at-first-gate-failure)
fi
run_case exhaustive_fem "${FEM_ARGS[@]}"

# Deployment policy: PINO stops at the predicted boundary; FEM verifies and
# rolls back to heavier candidates only when needed.
run_case pino_rollback_fem \
  --screen pino_rollback_fem --pino-checkpoint "$CKPT" \
  --pino-stop-margin 0.05 --pino-max-rollbacks -1

"$PY" - "$OUT" "$CASE_ID" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
case_id = sys.argv[2]
fem = json.loads((out / "exhaustive_fem/MANIFEST.json").read_text())
pino = json.loads((out / "pino_rollback_fem/MANIFEST.json").read_text())
pino_by_name = {row["candidate"]: row for row in pino["candidates"]}
rows = []
for step, row in enumerate(fem["candidates"]):
    p = pino_by_name.get(row["candidate"], {})
    screen = row.get("structural_screen") or {}
    rows.append({
        "case_id": case_id,
        "step": step,
        "candidate": row["candidate"],
        "saving": row.get("saving"),
        "mean_iou": row.get("mean_iou"),
        "min_iou": row.get("min_iou"),
        "mean_dice": row.get("mean_dice"),
        "fabrication_pass": row.get("fabrication_pass"),
        "fabrication_pass_without_fidelity": row.get("fabrication_pass_without_fidelity"),
        "structural_screen_admissible": row.get("structural_screen_admissible"),
        "best_effort_fidelity": row.get("best_effort_fidelity", False),
        "sf_fem": screen.get("sf_fem"),
        "fem_elapsed_s": screen.get("elapsed_s"),
        "fem_safe": row.get("structural_pass"),
        "sf_pino": (p.get("pino_prediction") or {}).get("sf_pino"),
        "pino_elapsed_s": (p.get("pino_prediction") or {}).get("elapsed_s"),
        "sf_pino_monotone": p.get("pino_monotone_sf"),
        "pino_predicted_safe": p.get("pino_predicted_safe"),
        "fem_checked_by_policy": p.get("fem_checked", False),
    })

payload = {
    "schema": "pcraft_controlled_boundary_trajectory_v1",
    "protocol": {
        "case_id": case_id,
        "views": 3,
        "mr": 1,
        "soft_resolution": 256,
        "single_seed_threshold": 0.55,
        "carve_checkpoint_voxels": int(__import__("os").environ.get("CARVE_CHECKPOINT_VOXELS", "0")),
        "stop_at_first_gate_failure": bool(int(__import__("os").environ.get("STOP_AT_FIRST_GATE_FAILURE", "0"))),
        "print_mm": float(__import__("os").environ.get("PRINT_MM", "210")),
        "sf_threshold": 1.3,
        "purpose": "same-trajectory PINO stopping and FEM rollback audit",
    },
    "pino_policy": pino.get("pino_structural_policy"),
    "rows": rows,
}
(out / "boundary_trajectory_rows.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
