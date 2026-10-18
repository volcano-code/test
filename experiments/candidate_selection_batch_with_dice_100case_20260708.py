#!/usr/bin/env python3
"""Fresh 100-case selector audit with IoU and Dice recorded at generation time."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generation"))
sys.path.insert(0, str(ROOT / "experiments"))

from candidate_selection_batch import (  # noqa: E402
    TAU,
    fem_sf,
    fit_acs,
    make_candidates,
    n6,
    select,
)
from datasets import load_data_from_list  # noqa: E402
from run_all_baselines import evaluate_metrics, load_pino  # noqa: E402
from structural.analysis.load_generator import LoadGenerator  # noqa: E402
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh  # noqa: E402


POLICIES = [
    "lightest_printable",
    "raw_pino",
    "acs_safe",
    "dual_acs_fem_fallback",
    "fem_oracle",
]


def candidate_by_name(rows: list[dict], name: str | None) -> dict | None:
    if name is None:
        return None
    return next((r for r in rows if r["candidate"] == name), None)


def fmt_policy_summary(cases: list[dict]) -> dict:
    summary = {}
    for policy in POLICIES:
        selected = [c["policies"][policy] for c in cases if c["policies"][policy]["selected"] is not None]
        summary[policy] = {
            "n_selected": len(selected),
            "fem_safe_rate": float(np.mean([r["selected_fem_safe"] for r in selected])) if selected else None,
            "fallback_rate": float(np.mean([r["selected_fallback"] for r in selected])) if selected else None,
            "mean_iou": float(np.mean([r["selected_mean_iou"] for r in selected])) if selected else None,
            "mean_dice": float(np.mean([r["selected_mean_dice"] for r in selected])) if selected else None,
            "mean_saving": float(np.mean([r["selected_saving"] for r in selected])) if selected else None,
            "mean_fem_fallbacks": float(np.mean([r["fem_fallbacks"] for r in selected])) if selected else None,
        }
    return summary


def run_branch(args: argparse.Namespace, branch: str, triples: list[list[str]], pino, device, iso, q_safe, q_unsafe) -> dict:
    tau_log = float(np.log10(TAU))
    sil_dir = ROOT / "data/silhouettes/bench3"
    cases = []
    for idx, triple in enumerate(triples):
        paths = [sil_dir / f"{name}.png" for name in triple]
        silhouettes, _ = load_data_from_list(
            [str(p) for p in paths],
            True,
            (args.resolution, args.resolution),
            torch.device("cpu"),
            3,
            False,
        )
        masks = [np.asarray(x).squeeze().astype(np.float32) for x in silhouettes]
        candidates, aux = make_candidates(
            masks,
            args.resolution,
            branch == "fixed",
            print_mm=args.print_mm,
            nozzle_mm=args.nozzle_mm,
            wall_safety=args.wall_safety,
        )
        hull = aux["hull"]
        cal = aux["calibration"]
        rows = []
        for cname, volume in candidates.items():
            metrics = evaluate_tfs_volume(volume.astype(np.float32), masks, cal, image_size=args.resolution)
            try:
                mesh = sdf_watertight_mesh(volume, smooth_sigma=0.6)
                watertight = bool(mesh.is_watertight)
            except Exception:
                watertight = False
            load = LoadGenerator().standing_load(volume.astype(np.float32))
            pino_metrics = evaluate_metrics(
                volume.astype(np.float32),
                hull.astype(np.float32),
                load["force"] * args.load_multiplier,
                pino,
                device,
            )
            sf_pino = float(pino_metrics["sf_min_pino"])
            pred = float(iso.predict([np.log10(max(sf_pino, 1e-12))])[0])
            fem = fem_sf(volume, args.load_multiplier)
            rows.append({
                "candidate": cname,
                "occupied_voxels": int(volume.sum()),
                "material_saved_vs_hull": float(1.0 - int(volume.sum()) / max(int(hull.sum()), 1)),
                "mean_iou": float(metrics["mean_iou"]),
                "mean_dice": float(metrics["mean_dice"]),
                "view_ious": [float(x) for x in metrics.get("view_ious", [])],
                "view_dices": [float(x) for x in metrics.get("view_dices", [])],
                "component_count_6": n6(volume),
                "watertight": watertight,
                "sf_pino": sf_pino,
                "raw_pino_safe": bool(sf_pino >= TAU),
                "acs_lower_sf": float(10 ** (pred - q_safe)),
                "acs_upper_sf": float(10 ** (pred + q_unsafe)),
                "acs_certified_safe": bool(pred - q_safe >= tau_log),
                "acs_certified_unsafe": bool(pred + q_unsafe < tau_log),
                "sf_fem": fem["sf_fem"],
                "fem_safe": fem["safe"],
                "fem_elapsed_s": fem["elapsed_s"],
            })
        policy_rows = {}
        for policy in POLICIES:
            chosen, fallbacks = select(rows, policy)
            policy_rows[policy] = {
                "selected": chosen["candidate"] if chosen else None,
                "selected_voxels": chosen["occupied_voxels"] if chosen else None,
                "selected_fem_safe": bool(chosen and chosen["fem_safe"]),
                "selected_saving": chosen["material_saved_vs_hull"] if chosen else None,
                "selected_mean_iou": chosen["mean_iou"] if chosen else None,
                "selected_mean_dice": chosen["mean_dice"] if chosen else None,
                "selection_status": chosen.get("selection_status") if chosen else None,
                "selected_fallback": bool(chosen and chosen.get("selection_fallback", False)),
                "fem_fallbacks": fallbacks,
            }
        cases.append({
            "case_id": idx,
            "branch": branch,
            "triple": triple,
            "n_candidates": len(rows),
            "candidates": rows,
            "policies": policy_rows,
        })
        print(f"[{branch} {idx + 1}/{len(triples)}] {triple} candidates={len(rows)}", flush=True)
    return {
        "branch": branch,
        "n_cases": len(cases),
        "policy_summary": fmt_policy_summary(cases),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_total", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--load_multiplier", type=float, default=1.0)
    parser.add_argument("--print_mm", type=float, default=None)
    parser.add_argument("--nozzle_mm", type=float, default=0.4)
    parser.add_argument("--wall_safety", type=float, default=2.0)
    parser.add_argument("--pino_checkpoint", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model.pt")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results" / "candidate_selection_100case_iou_dice_20260708")
    args = parser.parse_args()

    if args.n_total % 2 != 0:
        raise ValueError("--n_total must be even so free/fixed branches use the same number of cases")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pino = load_pino(str(args.pino_checkpoint), device)
    iso, q_safe, q_unsafe = fit_acs(args.alpha)

    sil_dir = ROOT / "data/silhouettes/bench3"
    names = sorted(p.stem for p in sil_dir.glob("*.png"))
    rng = random.Random(args.seed)
    triples = [rng.sample(names, 3) for _ in range(args.n_total // 2)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    branch_results = [
        run_branch(args, "free", triples, pino, device, iso, q_safe, q_unsafe),
        run_branch(args, "fixed", triples, pino, device, iso, q_safe, q_unsafe),
    ]
    cases = [case for result in branch_results for case in result["cases"]]
    result = {
        "schema_version": "candidate_selection_batch_iou_dice_v2",
        "n_cases": len(cases),
        "n_candidates": int(sum(c["n_candidates"] for c in cases)),
        "branches": ["free", "fixed"],
        "resolution": args.resolution,
        "seed": args.seed,
        "alpha": args.alpha,
        "threshold_sf": TAU,
        "print_mm": args.print_mm,
        "nozzle_mm": args.nozzle_mm,
        "wall_safety": args.wall_safety,
        "policy_summary": fmt_policy_summary(cases),
        "branch_results": branch_results,
        "cases": cases,
    }
    out = args.output_dir / "candidate_selection_100case_iou_dice.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    tsv = args.output_dir / "policy_summary.tsv"
    with tsv.open("w", encoding="utf-8") as f:
        f.write("Policy\tCases\tMean IoU\tMean Dice\tValidated reduction\tFEM-safe\tFallback\tMean FEM fallbacks\n")
        for policy, row in result["policy_summary"].items():
            if row["mean_iou"] is None:
                f.write(f"{policy}\t0/{result['n_cases']}\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\n")
            else:
                f.write(
                    f"{policy}\t{row['n_selected']}/{result['n_cases']}\t"
                    f"{row['mean_iou']:.3f}"
                    f"\t{row['mean_dice']:.3f}\t{100 * row['mean_saving']:.1f}%\t"
                    f"{100 * row['fem_safe_rate']:.1f}%\t{100 * row['fallback_rate']:.1f}%\t"
                    f"{row['mean_fem_fallbacks']:.2f}\n"
                )
    print(json.dumps({"output": str(out), "summary_tsv": str(tsv), "policy_summary": result["policy_summary"]}, indent=2))


if __name__ == "__main__":
    main()
