#!/usr/bin/env bash
# Freeze one soft field per task, then compare structural policies on exactly
# the same repaired-and-carved candidate trajectory.
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
OUT=${OUT:-$ROOT/results/pino_carve_stopping_20260720}
CKPT=${CKPT:-$ROOT/checkpoints/strict_physics_residual_pino_v16_shadowart_scale_20260718/best_model.pt}
POLICY=${POLICY:-$ROOT/results/pino_ranked_fem_taskdisjoint_v16_20260720/frozen_policy.json}
CASE_LIMIT=${CASE_LIMIT:-0}

CASES=(
  "H01_future_mech|anime" "H02_sword_fox|anime" "H03_street_mecha|anime"
  "H04_mage_warrior|anime" "H05_archer_heroine|anime" "H06_knight_mage|anime"
  "L01_parrot_snail|simple" "L02_owl_turtle|simple" "L03_frog_beetle|simple"
  "L04_armadillo_bunny|simple" "L05_puma_teddy|simple" "L06_lobster_duck|simple"
)

run_pipeline() {
  local task="$1" family="$2" mode="$3" artifact="$4"; shift 4
  local inputs="$ROOT/results/fem_two_wall_print_candidate_search_20260714/$family/$task/inputs"
  local dest="$OUT/$task/$mode"
  [[ -f "$dest/MANIFEST.json" ]] && { echo "skip completed $task $mode"; return; }
  args=(
    --silhouettes "$inputs/left_wall_x_target.png" "$inputs/right_wall_y_target.png"
    --branch soft-geometry --output "$dest" --tsf 1 --mr 1
    --soft-resolution 256 --soft-control-res 0 --core-resolution 48
    --soft-proposal-mode proposal --soft-disable-losses volume
    --soft-iterations 2000 --soft-lr 0.007 --soft-save-every 100 --soft-seed 42
    --soft-export-final-step
    --soft-seed-thresholds 0.15,0.25,0.35,0.45,0.55
    --delivery-shell-voxels 2 --min-iou 0.975 --min-view-iou 0.95
    --min-saving 0.0 --max-saving 0.85 --selection-iou-tolerance 0.01
    --selection-policy material --structural-resolution 48 --sf-threshold 1.3
    --print-mm 70 --device cuda:0
  )
  [[ -n "$artifact" ]] && args+=(--soft-artifact-dir "$artifact")
  "$PY" "$ROOT/shadowforge_pipeline/unified_run.py" "${args[@]}" "$@"
}

mkdir -p "$OUT"
case_index=0
for case in "${CASES[@]}"; do
  (( CASE_LIMIT > 0 && case_index >= CASE_LIMIT )) && break
  ((case_index += 1))
  IFS='|' read -r task family <<< "$case"
  run_pipeline "$task" "$family" material_only "" --screen skip --save-candidate-evidence
  artifact="$OUT/$task/material_only/soft_training"
  run_pipeline "$task" "$family" exhaustive_fem "$artifact" --screen fem_only
  run_pipeline "$task" "$family" pino_top3 "$artifact" \
    --screen pino_ranked_fem --pino-checkpoint "$CKPT" \
    --pino-ranked-policy "$POLICY" --pino-top-k 3 --pino-top-k-fallback expand
  for margin in 0.00 0.10 0.20; do
    tag=${margin/./p}
    run_pipeline "$task" "$family" "pino_rollback_m$tag" "$artifact" \
      --screen pino_rollback_fem --pino-checkpoint "$CKPT" \
      --pino-stop-margin "$margin" --pino-max-rollbacks -1
  done
done

"$PY" - "$OUT" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
rows = []
for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    full_path = task_dir / "exhaustive_fem/MANIFEST.json"
    if not full_path.is_file():
        continue
    full = json.loads(full_path.read_text())
    truth = {
        row["candidate"]: bool(row.get("structural_pass", False))
        for row in full["candidates"]
    }
    oracle = full.get("selected", {}).get("candidate")
    for manifest_path in sorted(task_dir.glob("*/MANIFEST.json")):
        data = json.loads(manifest_path.read_text())
        selected = data.get("selected") or {}
        candidate = selected.get("candidate")
        policy = data.get("pino_structural_policy") or {}
        valid = sum(bool(row.get("fabrication_pass")) for row in data.get("candidates", []))
        if manifest_path.parent.name == "exhaustive_fem":
            fem_calls = sum(row.get("structural_screen", {}).get("decided_by") == "FEM" for row in data["candidates"])
        else:
            fem_calls = int(policy.get("fem_calls", 0))
        rows.append({
            "task": task_dir.name,
            "policy": manifest_path.parent.name,
            "candidate": candidate,
            "mean_iou": selected.get("mean_iou"),
            "saving": selected.get("saving"),
            "fem_safe": truth.get(candidate) if candidate else False,
            "exact_oracle_match": candidate == oracle,
            "fem_calls": fem_calls,
            "valid_candidates": valid,
            "fem_budget": fem_calls / max(valid, 1),
        })

groups = defaultdict(list)
for row in rows:
    groups[row["policy"]].append(row)
summary = []
for policy, items in sorted(groups.items()):
    summary.append({
        "policy": policy,
        "tasks": len(items),
        "mean_iou": float(np.mean([x["mean_iou"] for x in items if x["mean_iou"] is not None])),
        "mean_saving": float(np.mean([x["saving"] for x in items if x["saving"] is not None])),
        "fem_safe_selection_rate": float(np.mean([x["fem_safe"] for x in items])),
        "exact_exhaustive_fem_agreement": float(np.mean([x["exact_oracle_match"] for x in items])),
        "mean_fem_budget": float(np.mean([x["fem_budget"] for x in items])),
    })
(root / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(root / "summary.json")
PY
