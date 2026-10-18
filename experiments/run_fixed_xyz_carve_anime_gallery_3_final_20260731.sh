#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
BLENDER=${BLENDER:-$(command -v blender || true)}
SOURCE='/home/lin/下载/original_anime_silhouettes_30(1) (2)/original_anime_silhouettes_30'
OUTPUT_ROOT="$ROOT/results/fixed_xyz_carve_anime_gallery_3_final_20260731"
LOG_ROOT="$ROOT/logs/fixed_xyz_carve_anime_gallery_3_final_20260731"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

run_case() {
  local name=$1
  local first=$2
  local second=$3
  local third=$4
  local output="$OUTPUT_ROOT/$name"
  local log="$LOG_ROOT/$name.log"

  mkdir -p "$output"
  "$PY" -u "$ROOT/shadowforge_pipeline/run.py" \
    --pipeline-branch fixed_xyz_carve \
    --silhouettes "$SOURCE/$first" "$SOURCE/$second" "$SOURCE/$third" \
    --output "$output" \
    --fixed-xyz-resolution 256 \
    --fixed-xyz-candidate carve_pf6 \
    --fixed-xyz-min-view-iou 0.950 \
    --min-iou 0.970 \
    --min-saving 0.30 \
    --max-saving 0.60 \
    --selection-saving-target 0.58 \
    --min-erosion-survival 0.15 \
    --print-mm 100 \
    --nozzle-mm 0.4 \
    --wall-safety 4.0 \
    --screen skip \
    >"$log" 2>&1

  if [[ -n "$BLENDER" ]]; then
    "$BLENDER" -b \
      --python "$ROOT/experiments/render_inspection_stl.py" \
      -- "$output/selected_display.stl" "$output/selected_render.png" \
      >>"$log" 2>&1
  fi
}

# Keep at most two 256^3 jobs resident at once.
run_case campus_motion \
  '03_学生女孩剪影.png' \
  '20_滑板动作中的年轻人.png' \
  '24_滑板空中动作.png' &
pid_campus=$!

run_case urban_allies \
  '21_英雄与伙伴的勇敢姿态.png' \
  '23_街头风猫耳女孩剪影.png' \
  '10_黑色轮廓中的快乐少年.png' &
pid_urban=$!

wait "$pid_campus"
wait "$pid_urban"

run_case future_stealth \
  '28_未来战士的自信身影.png' \
  '04_忍者动态潜行姿势.png' \
  '15_猫耳连帽少女剪影.png'

"$PY" "$ROOT/experiments/assemble_fixed_xyz_carve_anime_gallery_3_20260731.py" \
  --output-root "$OUTPUT_ROOT"
