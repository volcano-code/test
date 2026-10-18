#!/usr/bin/env python3
"""Compute true mean Dice for Table 2 selected candidates.

The saved 100-case candidate pool records mean IoU but not Dice.  This script
reconstructs the same 48^3 candidates from the saved triples and branch setting,
then computes Dice from rendered binary projections for each selected policy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]

from candidate_selection_batch import make_candidates
from datasets import load_data_from_list
from voxel_ops import compute_dice_np, project_tfs_volume, resize_binary_mask

OUT = ROOT / "results" / "paper_table_drafts_20260708"
RES = 48


def load_cases() -> list[dict]:
    cases = []
    for branch in ("free", "fixed"):
        path = ROOT / "results" / f"candidate_selection_batch_{branch}50_20260624.json"
        data = json.loads(path.read_text())
        for case in data["cases"]:
            case = dict(case)
            case["branch"] = branch
            cases.append(case)
    return cases


def by_name(case: dict, name: str) -> dict | None:
    return next((c for c in case["candidates"] if c["candidate"] == name), None)


def candidate_ok(c: dict, *, fem=False) -> bool:
    if float(c.get("mean_iou", -1)) < 0.95:
        return False
    if int(c.get("component_count_6", 999999)) != 1:
        return False
    if not bool(c.get("watertight", False)):
        return False
    s = float(c.get("material_saved_vs_hull", -999))
    if not (0.40 <= s <= 0.85):
        return False
    if fem and not bool(c.get("fem_safe", False)):
        return False
    return True


def selected_name(case: dict, policy: str) -> str | None:
    if policy == "visual_hull":
        return "visual_hull"
    if policy == "terminal_export":
        cand = by_name(case, "watertight_export")
        return "watertight_export" if cand is not None and candidate_ok(cand, fem=True) else None
    mapped = {
        "lightest_printable": "lightest_printable",
        "acs_safe": "acs_safe",
        "final_selector": "dual_acs_fem_fallback",
        "verified_selector": "fem_oracle",
    }[policy]
    return case.get("policies", {}).get(mapped, {}).get("selected")


def mean_dice_for_volume(volume: np.ndarray, masks: list[np.ndarray], calibration: dict) -> float:
    preds = project_tfs_volume(volume.astype(bool), calibration)
    dices = []
    for pred, gt in zip(preds[:3], masks[:3]):
        dices.append(compute_dice_np(resize_binary_mask(pred, RES), resize_binary_mask(gt, RES)))
    return float(np.mean(dices))


def main() -> None:
    sil_dir = ROOT / "data/silhouettes/bench3"
    policies = [
        ("Visual hull reference", "visual_hull"),
        ("Terminal export only", "terminal_export"),
        ("Lightest fabrication-valid", "lightest_printable"),
        ("ACS-only selector", "acs_safe"),
        ("Final selector", "final_selector"),
        ("FEM-oracle upper bound", "verified_selector"),
    ]
    dice_by_policy = {label: [] for label, _ in policies}
    per_case = []
    for case in load_cases():
        paths = [sil_dir / f"{name}.png" for name in case["triple"]]
        silhouettes, _ = load_data_from_list(
            [str(p) for p in paths], True, (RES, RES), torch.device("cpu"), 3, False
        )
        masks = [np.asarray(x).squeeze().astype(np.float32) for x in silhouettes]
        candidates, aux = make_candidates(masks, RES, case["branch"] == "fixed")
        row = {"branch": case["branch"], "case_id": case["case_id"], "triple": case["triple"], "dice": {}}
        for label, key in policies:
            cname = selected_name(case, key)
            if cname is None or cname not in candidates:
                continue
            val = mean_dice_for_volume(candidates[cname], masks, aux["calibration"])
            dice_by_policy[label].append(val)
            row["dice"][label] = val
        per_case.append(row)

    summary = {
        label: {
            "selected": len(vals),
            "mean_dice": float(np.mean(vals)) if vals else None,
        }
        for label, vals in dice_by_policy.items()
    }
    out = OUT / "table2_selected_dice_100case_20260708.json"
    out.write_text(json.dumps({
        "schema": "table2_selected_dice_100case_v1",
        "source": "recomputed 48^3 candidates from saved triples and branch settings",
        "summary": summary,
        "per_case": per_case,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
