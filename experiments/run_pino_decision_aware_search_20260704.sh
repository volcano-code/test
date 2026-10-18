#!/usr/bin/env bash
set -euo pipefail

cd /home/lin/xvxinyao/ShadowForge

run_one() {
  local tag="$1"
  shift
  PYTHONUNBUFFERED=1 /home/lin/anaconda3/bin/python -u experiments/train_decision_aware_pino.py \
    --out "results/decision_aware_pino_${tag}_20260704.json" \
    --output-dir "checkpoints/decision_aware_pino_${tag}_20260704" \
    "$@" \
    > "results/decision_aware_pino_${tag}_20260704.log" 2>&1
}

run_one e8_cal300_w36_miss14 --epochs 8 --external-calibration 300 --false-safe-weight 36 --safe-miss-weight 1.4 --hardset-unsafe-weight 2.0 --threshold-margin 0.00
run_one e10_cal300_w44_miss18 --epochs 10 --external-calibration 300 --false-safe-weight 44 --safe-miss-weight 1.8 --hardset-unsafe-weight 2.5 --threshold-margin 0.00
run_one e8_cal300_w36_miss22_margin005 --epochs 8 --external-calibration 300 --false-safe-weight 36 --safe-miss-weight 2.2 --hardset-unsafe-weight 2.0 --threshold-margin 0.005

/home/lin/anaconda3/bin/python - <<'PY'
import json
from pathlib import Path

root = Path('/home/lin/xvxinyao/ShadowForge/results')
rows = []
for path in sorted(root.glob('decision_aware_pino_*_20260704.json')):
    if path.name == 'decision_aware_pino_search_summary_20260704.json':
        continue
    try:
        data = json.loads(path.read_text())
        sel = data.get('selected', {})
        rows.append({
            'file': str(path),
            'checkpoint': data.get('checkpoint'),
            'joint_false_pass': sel.get('joint_false_pass'),
            'joint_auto_safe_true': sel.get('joint_auto_safe_true'),
            'candidate_auto_safe_rate': sel.get('candidate_heldout', {}).get('auto_safe_rate'),
            'external_auto_safe_rate': sel.get('external_heldout', {}).get('auto_safe_rate'),
            'hardset_auto_safe_rate': sel.get('hardset_test_near_boundary', {}).get('auto_safe_rate'),
            'promotion_eligible': data.get('promotion_eligible_for_full_fem_replacement_claim'),
        })
    except Exception as exc:
        rows.append({'file': str(path), 'error': str(exc)})
rows.sort(key=lambda r: (
    r.get('joint_false_pass') is not None,
    -(r.get('joint_false_pass') or 10**9),
    r.get('candidate_auto_safe_rate') or -1,
    r.get('external_auto_safe_rate') or -1,
), reverse=True)
(root / 'decision_aware_pino_search_summary_20260704.json').write_text(json.dumps({'schema': 'decision_aware_pino_search_summary_v1', 'rows': rows}, indent=2), encoding='utf-8')
print(json.dumps(rows[:5], indent=2))
PY
