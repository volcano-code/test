#!/usr/bin/env python3
"""Merge broad v6 and boundary-focused v7 FEM rows without task leakage.

Rows describing the same task/case/candidate are taken from v7; v6 supplies
the remaining broad-domain rows. Existing split ownership is required to agree
between both source indices. Paths are made absolute so the union index can
live in a separate directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def key(row: dict) -> tuple:
    return (
        row.get('split'), row.get('task'), row.get('branch'), row.get('case_id'),
        row.get('candidate'), row.get('parent_candidate'),
        row.get('structural_representation'),
    )


def load(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text())
    rows = payload.get('rows', [])
    if not rows:
        raise RuntimeError(f'No rows in {path}')
    return payload, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--broad-index', type=Path, required=True)
    ap.add_argument('--boundary-index', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    broad_payload, broad = load(args.broad_index)
    boundary_payload, boundary = load(args.boundary_index)
    owner: dict[str, str] = {}
    for row in broad + boundary:
        task, split = row['task'], row['split']
        if task in owner and owner[task] != split:
            raise RuntimeError(f'Task leakage for {task}: {owner[task]} vs {split}')
        owner[task] = split

    merged: dict[tuple, dict] = {}
    source_counts = {'broad_v6': 0, 'boundary_v7': 0}
    for source, index, rows in (
        ('broad_v6', args.broad_index, broad),
        ('boundary_v7', args.boundary_index, boundary),
    ):
        for original in rows:
            row = dict(original)
            path = Path(row['path'])
            if not path.is_absolute():
                path = (index.parent / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            row['path'] = str(path)
            row['union_source'] = source
            merged[key(row)] = row  # boundary_v7 deliberately replaces v6

    output_rows = sorted(merged.values(), key=lambda r: (r['split'], r['task'], str(r.get('case_id')), r['candidate']))
    for row in output_rows:
        source_counts[row['union_source']] += 1
    split_counts = {
        split: {
            'rows': sum(r['split'] == split for r in output_rows),
            'tasks': len({r['task'] for r in output_rows if r['split'] == split}),
            'safe': sum(r['split'] == split and bool(r['fem_safe']) for r in output_rows),
            'unsafe': sum(r['split'] == split and not bool(r['fem_safe']) for r in output_rows),
        }
        for split in ('train', 'calibration', 'sealed_test')
    }
    payload = {
        'schema': 'taskgrouped_strict_pino_broad_boundary_union_v11',
        'split_unit': 'complete silhouette task',
        'sources': {
            'broad_v6': str(args.broad_index.resolve()),
            'boundary_v7': str(args.boundary_index.resolve()),
        },
        'deduplication': 'boundary v7 replaces matching v6 task/case/candidate rows',
        'source_counts': source_counts,
        'counts': split_counts,
        'rows': output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'source_counts': source_counts, 'counts': split_counts}, indent=2))


if __name__ == '__main__':
    main()
