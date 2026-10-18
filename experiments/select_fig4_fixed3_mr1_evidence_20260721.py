#!/usr/bin/env python3
"""Rank and package representative Figure 4 evidence after full evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy_if_exists(source: Path, target: Path) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--sadekar", type=Path, required=True)
    parser.add_argument("--shadowforge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=4)
    args = parser.parse_args()

    benchmark = json.loads(args.manifest.read_text())
    difficulty = {row["case_id"]: row["input_only_selection"]["difficulty_score"] for row in benchmark["cases"]}
    targets = {row["case_id"]: row["target_paths_xyz"] for row in benchmark["cases"]}
    table = json.loads(args.table.read_text())
    by_case: dict[str, dict[str, dict]] = {}
    for row in table["per_case"]:
        by_case.setdefault(row["case_id"], {})[row["method"]] = row

    ranked = []
    for case_id, methods in by_case.items():
        if set(methods) != {"Mitra-style", "Sadekar voxel", "Sadekar mesh", "P-CRAFT"}:
            continue
        ours = methods["P-CRAFT"]
        baseline_best = max(methods[name]["mean_iou"] for name in ("Mitra-style", "Sadekar voxel", "Sadekar mesh"))
        deliverability_gap = sum(
            int(ours["single_body"] and ours["watertight"] and not (methods[name]["single_body"] and methods[name]["watertight"]))
            for name in ("Mitra-style", "Sadekar voxel", "Sadekar mesh")
        )
        score = 2.0 * (ours["mean_iou"] - baseline_best) + 0.20 * deliverability_gap + 0.10 * difficulty[case_id]
        ranked.append({"case_id": case_id, "figure_score": score, "input_difficulty": difficulty[case_id],
                       "iou_advantage_over_best_baseline": ours["mean_iou"] - baseline_best,
                       "deliverability_gap_count": deliverability_gap, "methods": methods})
    ranked.sort(key=lambda row: (-row["figure_score"], row["case_id"]))
    chosen = ranked[: args.top]

    sad = json.loads(args.sadekar.read_text())
    sad_runs = sad["rows"]
    for rank, item in enumerate(chosen, 1):
        case_id = item["case_id"]
        folder = args.output / f"{rank:02d}_{case_id}"
        for view, path in enumerate(targets[case_id]):
            copy_if_exists(Path(path), folder / f"target_xyz_{view}.png")
        # Use seed 43 when available, otherwise the first completed author run.
        for method, kind in (("sadekar_voxel", "voxel"), ("sadekar_mesh", "mesh")):
            candidates = [row for row in sad_runs if row["case_id"] == case_id and row["method"] == f"Sadekar {kind}"]
            run = next((row for row in candidates if row["seed"] == 43), candidates[0] if candidates else None)
            if run:
                artifact = Path(run["artifact"])
                pattern = "sample_0view_{}_silh.png" if kind == "voxel" else "sample_0_pred_view_{}.png"
                for view in range(6):
                    copy_if_exists(artifact.parent / pattern.format(view), folder / f"{method}_view_{view}.png")
        sf_runs = [row for row in json.loads((args.shadowforge / "SUMMARY.json").read_text()) if row["case_id"] == case_id]
        sf = next((row for row in sf_runs if row["seed"] == 43), sf_runs[0] if sf_runs else None)
        if sf:
            run_dir = Path(sf["manifest"]).parent
            for view in range(6):
                copy_if_exists(run_dir / "projections" / f"projection_view_{view}.png", folder / f"pcraft_view_{view}.png")
            copy_if_exists(run_dir / "selected_display_fixed.stl", folder / "pcraft_selected_display_fixed.stl")
        (folder / "metrics.json").write_text(json.dumps(item, indent=2) + "\n")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fig4_candidate_ranking.json").write_text(json.dumps({"ranked": ranked, "chosen": chosen}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "chosen": [row["case_id"] for row in chosen]}, indent=2))


if __name__ == "__main__":
    main()
