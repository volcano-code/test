#!/usr/bin/env bash
# Soft-stage-only ablation. No geometry cleanup, repair, carving, shell, or
# mesh QA is invoked. Each row reports the raw thresholded occupancy proposal's
# projection IoU and Dice under the same two-wall protocol.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
VOXEL=/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
OUT="$ROOT/results/soft_stage_loss_ablations_20260715"

# Four two-wall tasks, split across original-character and simple silhouettes.
CASES=(
  "H01_future_mech|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H01_future_mech/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H01_future_mech/inputs/right_wall_y_target.png"
  "H02_sword_fox|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H02_sword_fox/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H02_sword_fox/inputs/right_wall_y_target.png"
  "L01_parrot_snail|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L01_parrot_snail/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L01_parrot_snail/inputs/right_wall_y_target.png"
  "L02_owl_turtle|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L02_owl_turtle/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L02_owl_turtle/inputs/right_wall_y_target.png"
)

run_variant() {
  local task="$1" left="$2" right="$3" name="$4" disabled="$5"
  local dir="$OUT/$task/$name"
  mkdir -p "$dir"
  echo "===== task=$task variant=$name disabled=${disabled:-<none>} ====="
  "$PY" "$VOXEL/train.py" cuda:0 "$dir" 2000 0.007 1 \
    -sdlist "$left" "$right" \
    -mr 1 -tsf 0 -tsf4 0 -cam ortho -imsz 256 \
    --res 256 -eres 256 --control-res 0 --save-every 100 \
    --proposal-mode regularized \
    --connectivity-start-iou 0.60 --fidelity-gate 0.85 \
    --structure-warmup-steps 300 --structure-ramp-steps 300 \
    --no-cleanup --export-final-step \
    --disable-losses volume ${disabled}
}

# L2/L1 are the projection-baseline terms. These are the additional soft losses
# implemented by ShadowForge and active in the deployed configuration.
VARIANTS=(
  "full|"
  "w_o_Ldice|dice"
  "w_o_Lssim|ssim"
  "w_o_Lfp|false_positive"
  "w_o_Ledge|edge"
  "w_o_Lsmooth|smooth"
  "w_o_Llap|laplacian"
  "w_o_Lconn|connectivity"
  "w_o_Lspur|spurious"
  "w_o_Lwall|wall"
  "w_o_Lspec|spectral"
  "w_o_Ldev|deviation"
)

mkdir -p "$OUT"
for case in "${CASES[@]}"; do
  IFS='|' read -r task left right <<< "$case"
  for variant in "${VARIANTS[@]}"; do
    IFS='|' read -r name disabled <<< "$variant"
    run_variant "$task" "$left" "$right" "$name" "$disabled"
  done
done

"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for metrics in sorted(root.glob("*/*/metrics.json")):
    data = json.loads(metrics.read_text())
    rows.append((metrics.parents[1].name, metrics.parent.name,
                 data.get("mean_iou"), data.get("mean_dice")))

out = root / "soft_stage_iou_dice.tsv"
with out.open("w", encoding="utf-8") as f:
    f.write("task\tvariant\tmean_iou\tmean_dice\n")
    for task, variant, iou, dice in rows:
        f.write(f"{task}\t{variant}\t{iou:.6f}\t{dice:.6f}\n")
print(out)
PY
