#!/usr/bin/env bash
set -euo pipefail

# CPU-only candidate search: leave cuda:0 exclusively to the active PINO job.
ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
ANIME=/home/lin/xvxinyao/shadow/anime_shadow_processed_black_bg_light_simplified
SIMPLE=/home/lin/xvxinyao/shadow/normal_shadow
OUT="$ROOT/results/fem_two_wall_print_candidate_search_20260714"

mkdir -p "$OUT"

run_case() {
  local group="$1"
  local name="$2"
  local left="$3"
  local right="$4"

  "$PY" "$ROOT/shadowforge_pipeline/unified_run.py" \
    --silhouettes "$left" "$right" \
    --branch geometry \
    --output "$OUT/$group/$name" \
    --tsf 0 --mr 0 \
    --print-mm 70 \
    --min-iou 0.975 --min-view-iou 0.95 \
    --min-saving 0.20 --max-saving 0.80 \
    --selection-iou-tolerance 0.01 \
    --selection-policy fdm \
    --screen fem_only --structural-resolution 48 \
    --max-unsupported-ratio 1.0 --max-support-islands 1000000 --min-bed-area-mm2 0 \
    --display-fixture --save-candidate-evidence \
    --device cpu
}

# Original-character silhouettes: retain recognizable forms but avoid the prior
# dark-knight pair whose Bambu preview showed inaccessible internal support.
run_case anime H01_future_mech \
  "$ANIME/A005_005_未来战士的剪影.png" \
  "$ANIME/A014_014_未来战士的机械轮廓.png"
run_case anime H02_sword_fox \
  "$ANIME/A003_003_剑士剪影与风之衣饰.png" \
  "$ANIME/A013_013_狐女神秘身影.png"
run_case anime H03_street_mecha \
  "$ANIME/A010_010_未来街头英雄轮廓.png" \
  "$ANIME/A024_022_未来机械战士的身影.png"
run_case anime H04_mage_warrior \
  "$ANIME/A026_024_魔法师的黑白剪影.png" \
  "$ANIME/A051_11_勇敢战士的动态轮廓.png"
run_case anime H05_archer_heroine \
  "$ANIME/A009_009_神圣的弓箭手剪影.png" \
  "$ANIME/A034_030_自信少女与棒球棒.png"
run_case anime H06_knight_mage \
  "$ANIME/A027_025_黑暗骑士的孤独身影.png" \
  "$ANIME/A058_14_魔法师的召唤姿势.png"

# Compact, thick reference silhouettes for the low-difficulty control group.
run_case simple L01_parrot_snail \
  "$SIMPLE/birds_parrot.png" "$SIMPLE/sea_snail.png"
run_case simple L02_owl_turtle \
  "$SIMPLE/birds_owl.png" "$SIMPLE/reptiles_turtle.png"
run_case simple L03_frog_beetle \
  "$SIMPLE/reptiles_frog.png" "$SIMPLE/bugs_beetle.png"
run_case simple L04_armadillo_bunny \
  "$SIMPLE/armadillo_0.png" "$SIMPLE/bunny_0.png"
run_case simple L05_puma_teddy \
  "$SIMPLE/puma.png" "$SIMPLE/teddy2.png"
run_case simple L06_lobster_duck \
  "$SIMPLE/x_lobster.png" "$SIMPLE/duck.png"
