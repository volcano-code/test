#!/usr/bin/env bash
# Audit whether bridge repair changes delivery outcomes on existing frozen
# two-view soft fields. This is a screening audit, not the final ablation.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
OUT="$ROOT/results/table2b_bridge_trigger_audit_v2_20260727"

CASES=(
  "future_mech|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H01_future_mech/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H01_future_mech/inputs/right_wall_y_target.png|$ROOT/results/soft_novel_loss_ablations_v3_20260715/future_mech/full/seed_17"
  "sword_fox|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H02_sword_fox/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H02_sword_fox/inputs/right_wall_y_target.png|$ROOT/results/soft_novel_loss_ablations_v3_20260715/sword_fox/full/seed_17"
  "parrot_snail|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L01_parrot_snail/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L01_parrot_snail/inputs/right_wall_y_target.png|$ROOT/results/soft_novel_loss_ablations_v3_20260715/parrot_snail/full/seed_17"
  "owl_turtle|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L02_owl_turtle/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L02_owl_turtle/inputs/right_wall_y_target.png|$ROOT/results/soft_novel_loss_ablations_v3_20260715/owl_turtle/full/seed_17"
  "dark_knight|$ROOT/results/dark_knight_protocol_exact_rerun_20260726/inputs/left_wall_x_target.png|$ROOT/results/dark_knight_protocol_exact_rerun_20260726/inputs/right_wall_y_target.png|$ROOT/results/dark_knight_protocol_exact_rerun_20260726/soft_training"
  "aaai_maple|$ROOT/results/aaai_maple_printsafe_display_100mm_r1_20260722/inputs/left_wall_x_target.png|$ROOT/results/aaai_maple_printsafe_display_100mm_r1_20260722/inputs/right_wall_y_target.png|$ROOT/results/aaai_maple_printsafe_display_100mm_r1_20260722/soft_training"
)

run_variant() {
  local task="$1" left="$2" right="$3" frozen="$4" variant="$5"
  local flags=()
  if [[ "$variant" == "w_o_bridge_repair" ]]; then
    flags+=(--disable-bridge)
  fi
  local output="$OUT/$task/$variant"
  if [[ -f "$output/MANIFEST.json" ]]; then
    echo "skip completed $task/$variant"
    return
  fi
  echo "===== task=$task variant=$variant ====="
  "$PY" "$ROOT/shadowforge_pipeline/unified_run.py" \
    --silhouettes "$left" "$right" \
    --branch soft-geometry --output "$output" \
    --tsf 0 --mr 0 \
    --soft-resolution 256 --soft-control-res 0 \
    --soft-artifact-dir "$frozen" \
    --soft-seed-thresholds 0.15 \
    --bridge-radius 1 --bridge-passes 8 \
    --core-resolution 48 --delivery-shell-voxels 2 \
    --delivery-shortlist-size 1 \
    --min-iou 0.0 --min-view-iou 0.0 \
    --min-saving 0.0 --max-saving 1.0 --min-erosion-survival 0.0 \
    --selection-policy material --selection-iou-tolerance 0.0 \
    --screen skip --save-candidate-evidence --device cpu \
    "${flags[@]}"
}

mkdir -p "$OUT"
for item in "${CASES[@]}"; do
  IFS='|' read -r task left right frozen <<< "$item"
  run_variant "$task" "$left" "$right" "$frozen" full
  run_variant "$task" "$left" "$right" "$frozen" w_o_bridge_repair
done

"$PY" - "$OUT" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    pair = {}
    for variant in ("full", "w_o_bridge_repair"):
        path = task_dir / variant / "MANIFEST.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        selected = data.get("selected")
        if selected is None:
            candidates = data.get("candidates") or []
            if not candidates:
                continue
            selected = max(
                candidates,
                key=lambda row: (float(row.get("mean_iou", 0.0)),
                                 float(row.get("saving", 0.0))),
            )
        pair[variant] = {
            "mean_iou": float(selected.get("mean_iou", 0.0)),
            "saving": float(selected.get("saving", 0.0)),
            "components": int(selected.get("components", 0)),
            "watertight": bool(selected.get("watertight", False)),
            "candidate": selected.get("candidate"),
        }
    if len(pair) != 2:
        continue
    full = pair["full"]
    no_bridge = pair["w_o_bridge_repair"]
    changed = (
        not math.isclose(full["mean_iou"], no_bridge["mean_iou"], abs_tol=1e-9)
        or not math.isclose(full["saving"], no_bridge["saving"], abs_tol=1e-9)
        or full["components"] != no_bridge["components"]
        or full["watertight"] != no_bridge["watertight"]
        or full["candidate"] != no_bridge["candidate"]
    )
    rows.append({
        "task": task_dir.name,
        "bridge_sensitive": changed,
        "full": full,
        "w_o_bridge_repair": no_bridge,
    })

summary = {
    "purpose": "screen frozen proposals for a bridge-sensitive Table 2b cohort",
    "n_complete": len(rows),
    "n_bridge_sensitive": sum(row["bridge_sensitive"] for row in rows),
    "rows": rows,
}
(root / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
