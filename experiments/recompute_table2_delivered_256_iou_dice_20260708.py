#!/usr/bin/env python3
"""Recompute Table 2 projection metrics after 48-core -> 256-shell delivery."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, generate_binary_structure

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]

from candidate_selection_batch import make_candidates, keep_largest, n6  # noqa: E402
from datasets import load_data_from_list  # noqa: E402
from voxel_ops import build_tfs_visual_hull, calibrate_tfs_projection, evaluate_tfs_volume  # noqa: E402

SRC = ROOT / "results/candidate_selection_100case_iou_dice_20260708/candidate_selection_100case_iou_dice.json"
OUT_DIR = ROOT / "results/table2_delivered_256_iou_dice_20260708"
OUT_DIR.mkdir(parents=True, exist_ok=True)
S6 = generate_binary_structure(3, 1)

POLICY_LABELS = {
    "lightest_printable": "Lightest fabrication-valid",
    "raw_pino": "PINO-only selector",
    "acs_safe": "ACS-only selector",
    "dual_acs_fem_fallback": "ShadowForge verified selector",
    "fem_oracle": "FEM-oracle upper bound",
}


def load_masks(names: list[str], resolution: int) -> list[np.ndarray]:
    paths = [ROOT / "data/silhouettes/bench3" / f"{name}.png" for name in names]
    tensors, _ = load_data_from_list(
        [str(path) for path in paths],
        True,
        (resolution, resolution),
        torch.device("cpu"),
        3,
        False,
    )
    return [np.asarray(x).squeeze().astype(np.float32) for x in tensors]


def resize_nearest_48_to_256(core: np.ndarray) -> np.ndarray:
    # 48 does not divide 256. Index mapping avoids scipy.zoom's occasional
    # off-by-one output while preserving nearest-neighbor semantics.
    idx = np.floor(np.arange(256, dtype=np.float32) * (core.shape[0] / 256.0)).astype(np.int64)
    idx = np.clip(idx, 0, core.shape[0] - 1)
    return core.astype(bool)[np.ix_(idx, idx, idx)]


def deliver_256(core48: np.ndarray, hull256: np.ndarray, shell_t: int) -> np.ndarray:
    core_up = resize_nearest_48_to_256(core48)
    clipped = core_up & hull256
    shell = hull256 & ~binary_erosion(hull256, structure=S6, iterations=int(shell_t))
    return keep_largest(clipped | shell)


def summarize(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    source = json.loads(SRC.read_text())
    rows: list[dict] = []
    per_case: list[dict] = []
    for idx, case in enumerate(source["cases"], start=1):
        branch = case["branch"]
        fixed = branch == "fixed"
        triple = case["triple"]
        masks48 = load_masks(triple, 48)
        cands48, _ = make_candidates(masks48, 48, fixed)
        masks256 = load_masks(triple, 256)
        cal256 = calibrate_tfs_projection(masks256, search_res=96, fix_axis=fixed)
        hull256, _ = build_tfs_visual_hull(masks256, 256, calibration=cal256)
        hull256 = hull256.astype(bool)
        gt6_256 = masks256 + [np.fliplr(x) for x in masks256]

        case_record = {"case_id": case["case_id"], "branch": branch, "triple": triple, "policies": {}}
        for policy, label in POLICY_LABELS.items():
            selected = case["policies"][policy]["selected"]
            core = cands48.get(selected)
            if core is None:
                case_record["policies"][policy] = {"status": "missing_candidate", "selected": selected}
                continue
            delivered = deliver_256(core, hull256, shell_t=2)
            ev = evaluate_tfs_volume(delivered.astype(np.float32), gt6_256, cal256, image_size=256)
            rec = {
                "policy": policy,
                "label": label,
                "case_id": case["case_id"],
                "branch": branch,
                "triple": "+".join(triple),
                "selected": selected,
                "core_iou_48": float(case["policies"][policy]["selected_mean_iou"]),
                "core_dice_48": float(case["policies"][policy]["selected_mean_dice"]),
                "delivered_iou_256": float(ev["mean_iou"]),
                "delivered_dice_256": float(ev["mean_dice"]),
                "delivered_min_iou_256": float(ev["min_iou"]),
                "delivered_components_6": int(n6(delivered)),
                "material_saved_vs_h256": float(1.0 - int(delivered.sum()) / max(int(hull256.sum()), 1)),
                "fem_safe": bool(case["policies"][policy]["selected_fem_safe"]),
                "fallback": bool(case["policies"][policy]["selected_fallback"]),
                "fem_fallbacks": int(case["policies"][policy]["fem_fallbacks"]),
            }
            rows.append(rec)
            case_record["policies"][policy] = rec
        per_case.append(case_record)
        print(f"[{idx:03d}/{len(source['cases'])}] {branch} {'+'.join(triple)}", flush=True)

    summary = []
    for policy, label in POLICY_LABELS.items():
        subset = [row for row in rows if row["policy"] == policy]
        summary.append({
            "policy": policy,
            "label": label,
            "cases": f"{len(subset)}/{len(source['cases'])}",
            "delivered_iou_256": summarize([row["delivered_iou_256"] for row in subset]),
            "delivered_dice_256": summarize([row["delivered_dice_256"] for row in subset]),
            "core_iou_48": summarize([row["core_iou_48"] for row in subset]),
            "core_dice_48": summarize([row["core_dice_48"] for row in subset]),
            "validated_reduction": summarize([row["material_saved_vs_h256"] for row in subset]),
            "fem_safe_rate": summarize([1.0 if row["fem_safe"] else 0.0 for row in subset]),
            "fallback_rate": summarize([1.0 if row["fallback"] else 0.0 for row in subset]),
            "mean_fem_fallbacks": summarize([row["fem_fallbacks"] for row in subset]),
        })

    result = {
        "schema": "table2_delivered_256_iou_dice_v1",
        "source": str(SRC),
        "rule": "V256 = keep_largest((C48 nearest-neighbor upsampled to 256 intersect H256) union (H256 minus erode(H256, shell_t=2)))",
        "n_cases": len(source["cases"]),
        "n_policy_rows": len(rows),
        "summary": summary,
        "per_case": per_case,
    }
    (OUT_DIR / "table2_delivered_256_iou_dice.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (OUT_DIR / "table2_delivered_256_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label", "cases", "delivered_iou_256", "delivered_dice_256",
                "validated_reduction", "fem_safe_rate", "fallback_rate", "mean_fem_fallbacks",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in summary:
            writer.writerow({
                "label": row["label"],
                "cases": row["cases"],
                "delivered_iou_256": f"{row['delivered_iou_256']:.3f}",
                "delivered_dice_256": f"{row['delivered_dice_256']:.3f}",
                "validated_reduction": f"{100.0 * row['validated_reduction']:.1f}%",
                "fem_safe_rate": f"{100.0 * row['fem_safe_rate']:.1f}%",
                "fallback_rate": f"{100.0 * row['fallback_rate']:.1f}%",
                "mean_fem_fallbacks": f"{row['mean_fem_fallbacks']:.2f}",
            })
    print(json.dumps({"out": str(OUT_DIR), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
