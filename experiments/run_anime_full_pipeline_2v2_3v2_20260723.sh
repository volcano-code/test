#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
OUT=${OUT:-$ROOT/results/anime_full_pipeline_2v2_3v2_20260723}
CKPT=${CKPT:-$ROOT/checkpoints/strict_physics_residual_pino_v16_shadowart_scale_20260718/best_model.pt}
POLICY=${POLICY:-$ROOT/results/pino_ranked_fem_taskdisjoint_v16_20260720/frozen_policy.json}

run_case() {
  local name=$1
  local tsf=$2
  shift 2
  local views=()
  while [[ $# -gt 0 && $1 != -- ]]; do
    views+=("$1")
    shift
  done
  shift

  local dest="$OUT/$name"
  if [[ -f "$dest/MANIFEST.json" ]]; then
    printf '[skip] completed %s\n' "$name"
    return
  fi

  mkdir -p "$dest"
  printf '[start] %s (%s views, tsf=%s) %s\n' \
    "$name" "${#views[@]}" "$tsf" "$(date --iso-8601=seconds)"

  "$PY" "$ROOT/shadowforge_pipeline/unified_run.py" \
    --silhouettes "${views[@]}" \
    --output "$dest" \
    --branch soft-geometry \
    --tsf "$tsf" \
    --mr 0 \
    --soft-resolution 256 \
    --soft-control-res 0 \
    --soft-proposal-mode proposal \
    --soft-disable-losses volume \
    --soft-iterations 2000 \
    --soft-lr 0.007 \
    --soft-save-every 100 \
    --soft-seed 42 \
    --soft-seed-thresholds 0.05,0.10,0.15,0.20,0.25,0.35,0.45,0.55,0.65 \
    --bridge-radius 2 \
    --bridge-passes 48 \
    --core-resolution 48 \
    --delivery-shell-voxels 2 \
    --min-iou 0.95 \
    --min-view-iou 0.90 \
    --min-saving 0.000001 \
    --max-saving 0.85 \
    --min-erosion-survival 0.08 \
    --selection-iou-tolerance 0.01 \
    --selection-policy material \
    --print-mm 70 \
    --screen pino_ranked_fem \
    --structural-resolution 48 \
    --sf-threshold 1.3 \
    --pino-checkpoint "$CKPT" \
    --pino-ranked-policy "$POLICY" \
    --pino-top-k 3 \
    --pino-top-k-fallback expand \
    --save-candidate-evidence \
    --device cuda:0 \
    > "$dest/RUN_OUTPUT.txt" 2>&1

  printf '[done] %s %s\n' "$name" "$(date --iso-8601=seconds)"
}

mkdir -p "$OUT"

TWO_ROOT="$ROOT/results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709"
THREE_ROOT="$ROOT/results/latest30_anime_silhouette_screen_20260708/normalized_white_on_black"

run_case two_view_fox_miko_violinist 0 \
  "$TWO_ROOT/fox_miko_left__violinist_right/inputs/left_wall_x_view1_target.png" \
  "$TWO_ROOT/fox_miko_left__violinist_right/inputs/right_wall_y_view2_target.png" \
  --

run_case two_view_future_mecha_skater 0 \
  "$TWO_ROOT/future_mecha_left__skater_right/inputs/left_wall_x_view1_target.png" \
  "$TWO_ROOT/future_mecha_left__skater_right/inputs/right_wall_y_view2_target.png" \
  --

run_case three_view_blackrabbit_fox_prince 1 \
  "$THREE_ROOT/20_黑色兔子剪影.png" \
  "$THREE_ROOT/05_狐狸披风小孩剪影.png" \
  "$THREE_ROOT/22_可爱王子矢量剪影.png" \
  --

run_case three_view_catgirl_rabbit_mushroom 1 \
  "$THREE_ROOT/10_黑色猫耳披风人物轮廓.png" \
  "$THREE_ROOT/08_兔耳小人_图标.png" \
  "$THREE_ROOT/01_蘑菇帽子的小精灵.png" \
  --

"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    manifest = case_dir / "MANIFEST.json"
    if not manifest.is_file():
        rows.append({"case": case_dir.name, "status": "INCOMPLETE"})
        continue
    data = json.loads(manifest.read_text())
    selected = data.get("selected") or {}
    rows.append({
        "case": case_dir.name,
        "status": data.get("status"),
        "candidate": selected.get("candidate"),
        "mean_iou": selected.get("mean_iou"),
        "mean_dice": selected.get("mean_dice"),
        "saving": selected.get("saving"),
        "components": selected.get("components"),
        "watertight": selected.get("watertight"),
        "structural_pass": selected.get("structural_pass"),
        "display_stl": data.get("display_stl"),
        "print_stl": data.get("print_stl"),
    })
(root / "SUMMARY.json").write_text(json.dumps({"cases": rows}, indent=2))
print(json.dumps({"summary": str(root / "SUMMARY.json"), "cases": rows}, indent=2))
PY
