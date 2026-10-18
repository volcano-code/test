"""Rebuild Fig. 3 and Fig. 5 from controlled artifacts, with real Blender panels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
PAPER = ROOT / "paper_aaai_submission"
SRC = ROOT / "results/controlled_benchmark_geometry_only_20260710"
POOL = ROOT / "results/figure45_candidate_pool_case5A_ant_peacock_frog_20260709"
CASES = [
    "case1A_owl_bat_turtle",
    "case2A_parrot_dragon_snail",
    "case3A_spider_scorpion_beetle",
    "case6B_ladybug_frog_peacock",
]
METHODS = ["Target silhouettes", "Mitra-style carving", "Sadekar mesh-based", "Sadekar voxel-based", "ShadowForge"]


def method_dir(method: str) -> str:
    return method.lower().replace(" ", "_").replace("-", "_")


def image_path(case_id: str, method: str, view: int, seed: int | None = None) -> Path:
    base = SRC / "evaluated_shadows" / case_id
    if method == "Target silhouettes":
        return base / "targets" / f"view{view}.png"
    suffix = "deterministic" if seed is None else f"seed_{seed}"
    return base / method_dir(method) / f"{suffix}_view{view}.png"


def median_row(rows: list[dict], case_id: str, method: str) -> dict:
    selected = [row for row in rows if row["case_id"] == case_id and row["method"] == method]
    if method.startswith("Sadekar"):
        selected.sort(key=lambda row: row["mean_iou"])
        return selected[len(selected) // 2]
    return selected[0]


def clean_name(case_id: str) -> str:
    return " / ".join(case_id.replace("case1A_", "").replace("case2A_", "").replace("case3A_", "").replace("case6B_", "").split("_"))


def add_shadow(ax, path: Path):
    ax.imshow(Image.open(path).convert("L"), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#8a939d"); sp.set_linewidth(0.55)


def make_fig3():
    rows = json.loads((SRC / "per_case_metrics.json").read_text())
    fig = plt.figure(figsize=(15.8, 9.1), dpi=220, facecolor="white")
    grid = fig.add_gridspec(6, 4, left=0.12, right=0.99, bottom=0.055, top=0.92, hspace=0.36, wspace=0.18,
                            height_ratios=[1.65, 1, 1, 1, 1, 1])
    for col, case_id in enumerate(CASES):
        fig.text(0.12 + (col + .5) * (.87 / 4), .958, f"({chr(97 + col)}) {clean_name(case_id)}",
                 ha="center", va="center", fontsize=10.5, weight="bold")
        top = fig.add_subplot(grid[0, col])
        top.imshow(Image.open(PAPER / "fig3_blender_top_20260710" / f"{case_id}.png").convert("RGB"))
        top.set_title("ShadowForge 3D sculpture", fontsize=7.8, pad=3)
        top.set_xticks([]); top.set_yticks([])
        for sp in top.spines.values(): sp.set_visible(False)
        sf = json.loads((SRC / "shadowforge_authoritative_fixed_shell2" / case_id / "MANIFEST.json").read_text())
        delivery = sf["delivery"]
        top.text(.5, -.12, f"IoU {delivery['mean_iou']:.3f}   DS {delivery['mean_dice']:.3f}",
                 transform=top.transAxes, ha="center", va="top", fontsize=7.2, color="#303840")
        for ri, method in enumerate(METHODS):
            inner = grid[ri + 1, col].subgridspec(1, 3, wspace=0.04)
            seed = None if not method.startswith("Sadekar") else int(median_row(rows, case_id, method)["seed"])
            for view in range(3):
                add_shadow(fig.add_subplot(inner[0, view]), image_path(case_id, method, view, seed))
            if method != "Target silhouettes":
                row = median_row(rows, case_id, method)
                box = grid[ri + 1, col].get_position(fig)
                fig.text((box.x0 + box.x1) / 2, box.y0 - .020,
                         f"IoU {row['mean_iou']:.3f}   DS {row['mean_dice']:.3f}",
                         ha="center", va="top", fontsize=7.1, color="#303840")
    for ri, method in enumerate(METHODS):
        box = grid[ri + 1, 0].get_position(fig)
        fig.text(.108, (box.y0 + box.y1) / 2, method, ha="right", va="center", fontsize=9.4,
                 weight="bold" if method in {"Target silhouettes", "ShadowForge"} else "normal")
    for ext in ("png", "pdf"):
        fig.savefig(PAPER / f"fig3_main_comparison_methods.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def orient(mask: np.ndarray, target: np.ndarray) -> np.ndarray:
    candidates = [mask, mask.T, np.flipud(mask), np.fliplr(mask), np.rot90(mask), np.rot90(mask, 2), np.rot90(mask, 3)]
    def score(x):
        if x.shape != target.shape:
            x = np.array(Image.fromarray(x.astype(np.uint8) * 255).resize(target.shape[::-1], Image.Resampling.NEAREST)) > 0
        return np.logical_and(x, target).sum() / max(np.logical_or(x, target).sum(), 1)
    return max(candidates, key=score)


def make_fig5():
    manifest = json.loads((POOL / "MANIFEST.json").read_text())
    rows = manifest["candidates"]
    by_name = {r["candidate"]: r for r in rows}
    variants = [
        ("Visual hull", "visual_hull", "reference", "wo_nullspace_carving"),
        ("w/o connectivity guard", "carve_no_bridge", "fragmented", "wo_connectivity_guard"),
        ("w/o watertight repair", "min_printable_bridge_r1", "non-manifold", "wo_watertight_repair"),
        ("Repaired candidate", "watertight_export", "fabrication-valid", "full"),
    ]
    targets = []
    for p in manifest["inputs"]:
        img = np.array(Image.open(p["path"]).convert("L"))
        targets.append(img < 128 if np.median(np.r_[img[:4,:4].ravel(), img[-4:,-4:].ravel()]) > 127 else img > 127)
    fig = plt.figure(figsize=(15.8, 5.8), dpi=220, facecolor="white")
    fig.text(.035, .935, "(a) Geometry-stage evidence", fontsize=13, fontweight="bold")
    fig.text(.675, .935, "(b) Candidate selection over the same pool", fontsize=13, fontweight="bold")
    left = fig.add_gridspec(3, 4, left=.055, right=.63, bottom=.12, top=.84, hspace=.22, wspace=.12,
                            height_ratios=[2.2, .85, .45])
    hull_n = float(by_name["visual_hull"]["occupied_voxels"])
    for ci, (title, key, note, render_name) in enumerate(variants):
        r = by_name[key]
        vol = np.load(ROOT / r["artifact"]).astype(bool)
        ax = fig.add_subplot(left[0, ci])
        model = Image.open(PAPER / "fig_case5a_blender_models" / f"{render_name}.png").convert("RGBA")
        alpha = model.getchannel("A")
        bbox = alpha.getbbox()
        if bbox:
            pad = int(.04 * max(model.size))
            model = model.crop((max(0,bbox[0]-pad), max(0,bbox[1]-pad), min(model.width,bbox[2]+pad), min(model.height,bbox[3]+pad)))
        white = Image.new("RGBA", model.size, "white")
        model = Image.alpha_composite(white, model).convert("RGB")
        ax.imshow(model); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8.2, pad=2)
        for sp in ax.spines.values(): sp.set_visible(False)
        strip = left[1, ci].subgridspec(1,3,wspace=.04)
        for vi, axis in enumerate((2,0,1)):
            p = vol.max(axis=axis).T
            p = orient(p, targets[vi])
            a = fig.add_subplot(strip[0,vi]); a.imshow(p, cmap="gray_r", interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_color("#8a939d"); sp.set_linewidth(.45)
        box = left[2,ci].get_position(fig)
        save = 100*(1 - vol.sum()/hull_n)
        txt = f"saving {save:.1f}%\n{note}"
        fig.text((box.x0+box.x1)/2,(box.y0+box.y1)/2,txt,ha="center",va="center",fontsize=7.2,
                 weight="bold" if key=="watertight_export" else "normal")
    fig.text(.044,.49,"Re-rendered\nshadows",ha="right",va="center",fontsize=7.4)
    # Selection plot uses stored candidate metadata from this exact pool.
    ax = fig.add_axes([.70,.22,.26,.60])
    threshold=.15
    ax.axhspan(threshold,1.05,color="#eef8ef",zorder=0); ax.axhspan(0,threshold,color="#fdecec",zorder=0)
    for r in rows:
        x=100*r["material_saved_vs_hull"]; y=float(r.get("erosion_survival") or 0)
        accepted=bool(r.get("manufacturing_pass"))
        if r["candidate"]=="watertight_export":
            ax.scatter(x,y,s=260,marker="*",c="#168a5a",edgecolors="black",linewidths=.8,zorder=5,label="selected")
        elif accepted:
            ax.scatter(x,y,s=46,c="#26a643",edgecolors="black",linewidths=.35,zorder=4,label="fabrication-valid")
        else:
            ax.scatter(x,y,s=40,c="#9c9c9c",edgecolors="white",linewidths=.25,zorder=3,label="rejected")
    ax.axhline(threshold,color="0.35",ls="--",lw=.8); ax.text(98,threshold+.02,"wall-survival threshold",ha="right",fontsize=7)
    ax.set(xlabel="material reduction (%)",ylabel="wall survival",xlim=(-4,100),ylim=(0,1.05))
    ax.grid(color="white",lw=.55)
    handles, labels = ax.get_legend_handles_labels(); dedup={}
    for h,l in zip(handles,labels): dedup.setdefault(l,h)
    ax.legend(dedup.values(),dedup.keys(),frameon=False,loc="upper left",fontsize=7)
    ax.annotate("selected after\nfabrication checks",xy=(100*by_name['watertight_export']['material_saved_vs_hull'],by_name['watertight_export']['erosion_survival']),xytext=(42,.82),arrowprops={"arrowstyle":"->","lw":.8},fontsize=7.5)
    for ext in ("png", "pdf"):
        fig.savefig(PAPER / f"fig5_geometry_failure_modes.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_fig3()
    make_fig5()
