#!/usr/bin/env python3
"""Build Table 1, per-case supplement, and Figure 3 from controlled results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure

from evaluate_controlled_benchmark_20260710 import mesh_to_volume, sadekar_path, transformed


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = ROOT / "results/controlled_benchmark_geometry_only_20260710"
PAPER = ROOT / "paper_aaai_submission"
METHODS = ["Mitra-style carving", "Sadekar voxel-based", "Sadekar mesh-based", "ShadowForge"]


def pct(value):
    return "N/A" if value is None else f"{100*value:.1f}\\%"


def table1(aggregate: list[dict]) -> None:
    by = {row["method"]: row for row in aggregate}
    lines = [
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Method & Mean IoU & Worst IoU & Dice & Saving & Single & Watertight \\",
        r"\midrule",
    ]
    for method in METHODS:
        row = by[method]
        stochastic = method.startswith("Sadekar")
        iou = f"{row['mean_iou']:.3f} $\\pm$ {row['std_iou']:.3f}" if stochastic else f"{row['mean_iou']:.3f}"
        dice = f"{row['mean_dice']:.3f} $\\pm$ {row['std_dice']:.3f}" if stochastic else f"{row['mean_dice']:.3f}"
        vals = [iou, f"{row['worst_view_iou']:.3f}", dice, pct(row["mean_saving"]),
                pct(row["single_body_rate"]), pct(row["watertight_rate"])]
        label = method
        if method == "ShadowForge":
            label = r"\textbf{ShadowForge}"
            vals = [r"\textbf{" + value + "}" for value in vals]
        lines.append(label + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (SRC / "table1.tex").write_text("\n".join(lines) + "\n")


def median_seed(rows: list[dict]) -> int:
    rows = sorted(rows, key=lambda row: row["mean_iou"])
    return int(rows[len(rows) // 2]["seed"])


def method_dir(method: str) -> str:
    return method.lower().replace(" ", "_").replace("-", "_")


def image_path(case_id: str, method: str, view: int, seed: int | None) -> Path:
    if method == "Target silhouettes":
        return SRC / "evaluated_shadows" / case_id / "targets" / f"view{view}.png"
    suffix = "deterministic" if seed is None else f"seed_{seed}"
    return SRC / "evaluated_shadows" / case_id / method_dir(method) / f"{suffix}_view{view}.png"


def evaluated_volume(case_id: str, method: str, seed: int | None, transforms: dict) -> np.ndarray:
    if method in {"Mitra-style carving", "ShadowForge"}:
        name = method.lower().replace(" ", "_") + "_voxels256.npy"
        return np.load(SRC / "evaluated_shadows" / case_id / name).astype(bool)
    kind = "voxel" if "voxel" in method else "mesh"
    path = sadekar_path(kind, case_id, int(seed))
    if kind == "voxel":
        volume = np.squeeze(np.load(path)) > 0.05
    else:
        volume, _ = mesh_to_volume(path)
    transform = transforms[kind]
    return transformed(volume, tuple(transform["perm"]), tuple(transform["flips"]))


def draw_volume(ax, volume: np.ndarray) -> None:
    step = 2
    small = volume[::step, ::step, ::step]
    surface = small & ~binary_erosion(small, structure=generate_binary_structure(3, 1))
    points = np.argwhere(surface)
    if len(points) > 9000:
        points = points[::max(1, len(points) // 9000)]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.55, c="#2F7F72", alpha=0.92, depthshade=True)
    n = small.shape[0]
    ax.set(xlim=(0, n), ylim=(0, n), zlim=(0, n))
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=-55); ax.set_axis_off()


def figure3(manifest: dict, rows: list[dict]) -> None:
    cases = [case for case in manifest["cases"] if case["split"] == "test"]
    display_methods = ["Target silhouettes", *METHODS]
    transforms = json.loads((SRC / "coordinate_normalization.json").read_text())
    fig = plt.figure(figsize=(15.8, 7.2), dpi=220, facecolor="white")
    outer = fig.add_gridspec(len(display_methods), len(cases), left=0.105, right=0.995,
                             bottom=0.06, top=0.91, wspace=0.13, hspace=0.30)
    for col, case in enumerate(cases):
        case_id = case["case_id"]
        title = " / ".join(name.replace("x_", "").replace("birds_", "").replace("reptiles_", "").replace("sea_", "").replace("bugs_", "") for name in case["target_names"])
        fig.text(0.105 + (col + 0.5) * (0.89 / len(cases)), 0.952, f"({chr(97+col)}) {title}",
                 ha="center", va="center", fontsize=10.5, weight="bold")
        for row_idx, method in enumerate(display_methods):
            inner = outer[row_idx, col].subgridspec(1, 4, width_ratios=(1.15, 1, 1, 1), wspace=0.035)
            seed = None
            metric_rows = [r for r in rows if r["case_id"] == case_id and r["method"] == method]
            if method.startswith("Sadekar"):
                seed = median_seed(metric_rows)
                metric = next(r for r in metric_rows if r["seed"] == seed)
            elif method != "Target silhouettes":
                metric = metric_rows[0]
            if method == "Target silhouettes":
                ax_model = fig.add_subplot(inner[0, 0]); ax_model.axis("off")
                ax_model.text(0.5, 0.5, "fixed\nviews", ha="center", va="center", fontsize=7.2, color="#5B6672")
            else:
                ax_model = fig.add_subplot(inner[0, 0], projection="3d")
                draw_volume(ax_model, evaluated_volume(case_id, method, seed, transforms))
            for view in range(3):
                ax = fig.add_subplot(inner[0, view + 1])
                ax.imshow(Image.open(image_path(case_id, method, view, seed)).convert("L"), cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.55); spine.set_edgecolor("#858D96")
            if method != "Target silhouettes":
                x = 0.105 + (col + 0.5) * (0.89 / len(cases))
                y = 0.91 - (row_idx + 0.93) * (0.85 / len(display_methods))
                fig.text(x, y, f"IoU {metric['mean_iou']:.3f}  DS {metric['mean_dice']:.3f}",
                         ha="center", va="top", fontsize=7.2, color="#303840")
    for row_idx, method in enumerate(display_methods):
        y = 0.91 - (row_idx + 0.46) * (0.85 / len(display_methods))
        fig.text(0.092, y, method, ha="right", va="center", fontsize=9.2,
                 weight="bold" if method in {"Target silhouettes", "ShadowForge"} else "normal")
    fig.savefig(PAPER / "fig3_main_comparison_methods.png", bbox_inches="tight", facecolor="white")
    fig.savefig(PAPER / "fig3_main_comparison_methods.pdf", bbox_inches="tight", facecolor="white")


def supplement(rows: list[dict], manifest: dict) -> None:
    cases = [case for case in manifest["cases"] if case["split"] == "test"]
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Case & Method & IoU & Worst & Dice & Saving & Seed \\", r"\midrule"]
    for case in cases:
        case_rows = [r for r in rows if r["case_id"] == case["case_id"]]
        for method in METHODS:
            vals = [r for r in case_rows if r["method"] == method]
            if method.startswith("Sadekar"):
                row = next(r for r in vals if r["seed"] == median_seed(vals))
            else:
                row = vals[0]
            lines.append("{} & {} & {:.3f} & {:.3f} & {:.3f} & {} & {} \\\\".format(
                case["case_id"].replace("_", r"\_"), method, row["mean_iou"], row["worst_iou"],
                row["mean_dice"], pct(row.get("saving")), row.get("seed") or "--"))
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (SRC / "table_supp_per_case.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    manifest = json.loads((SRC / "benchmark_manifest.json").read_text())
    rows = json.loads((SRC / "per_case_metrics.json").read_text())
    aggregate = json.loads((SRC / "prior_comparison_results.json").read_text())["aggregate"]
    table1(aggregate); supplement(rows, manifest); figure3(manifest, rows)
    print(SRC / "table1.tex")
    print(PAPER / "fig3_main_comparison_methods.png")


if __name__ == "__main__":
    main()
