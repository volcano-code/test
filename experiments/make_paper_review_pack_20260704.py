#!/usr/bin/env python3
"""Build review-only paper assets from local evidence.

This script does not edit the paper. It creates:
  - normalized anime silhouettes and a contact sheet;
  - a fast triplet screen for the 30 anime silhouettes;
  - draft figures/tables for paper review.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
for p in [ROOT, ROOT / "generation", ROOT / "experiments"]:
    sys.path.insert(0, str(p))

from candidate_selection_batch import make_candidates, n6  # noqa: E402
from datasets import load_data_from_list  # noqa: E402
from voxel_ops import evaluate_tfs_volume  # noqa: E402

SRC_ANIME = Path("/home/lin/下载/original_anime_silhouettes_30")
OUT = ROOT / "results/paper_review_pack_20260704"
ANIME_OUT = ROOT / "data/silhouettes/original_anime_30_20260704"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def binarize_and_fit(path: Path, size: int = 512, margin: int = 34) -> tuple[np.ndarray, dict]:
    im = Image.open(path).convert("L")
    arr = np.asarray(im)
    fg = arr < 128
    ys, xs = np.where(fg)
    if len(xs) == 0:
        raise ValueError(f"empty silhouette: {path}")
    crop = fg[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = crop.shape
    scale = min((size - 2 * margin) / max(w, 1), (size - 2 * margin) / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = Image.fromarray((crop.astype(np.uint8) * 255), "L").resize((nw, nh), Image.Resampling.NEAREST)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    out = np.asarray(canvas) > 127
    info = {
        "source": str(path),
        "source_size": list(im.size),
        "foreground_ratio_source": round(float(fg.mean()), 4),
        "foreground_ratio_512": round(float(out.mean()), 4),
        "bbox_source": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    }
    return out, info


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(path)


def make_contact_sheet(items: list[dict], out: Path, cols: int = 6) -> None:
    thumb, label_h, pad = 150, 28, 14
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * (thumb + pad) + pad, rows * (thumb + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, rec in enumerate(items):
        r, c = divmod(idx, cols)
        x = pad + c * (thumb + pad)
        y = pad + r * (thumb + label_h + pad)
        im = Image.open(rec["path"]).convert("L").resize((thumb, thumb), Image.Resampling.NEAREST)
        sheet.paste(Image.merge("RGB", (im, im, im)), (x, y))
        draw.text((x, y + thumb + 4), rec["id"], fill=(20, 20, 20), font=font(14, True))
    sheet.save(out)


def write_table_png(rows: list[list[str]], out: Path, title: str) -> None:
    col_w = [max(120, min(310, max(len(str(r[i])) for r in rows) * 8 + 24)) for i in range(len(rows[0]))]
    row_h = 34
    w = sum(col_w) + 2
    h = row_h * (len(rows) + 1) + 52
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((10, 10), title, fill=(0, 0, 0), font=font(18, True))
    y = 44
    for ri, row in enumerate(rows):
        x = 1
        fill = (238, 242, 247) if ri == 0 else ((250, 250, 250) if ri % 2 else (255, 255, 255))
        d.rectangle([1, y, w - 2, y + row_h], fill=fill, outline=(205, 211, 220))
        for ci, text in enumerate(row):
            d.text((x + 8, y + 8), str(text), fill=(0, 0, 0), font=font(12, ri == 0))
            x += col_w[ci]
            d.line([x, y, x, y + row_h], fill=(220, 225, 232))
        y += row_h
    im.save(out)


def preprocess_anime() -> list[dict]:
    ANIME_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(sorted(SRC_ANIME.glob("*.png")), 1):
        mask, info = binarize_and_fit(p)
        sid = f"anime30_{i:02d}"
        dst = ANIME_OUT / f"{sid}.png"
        save_mask(mask, dst)
        rows.append({"id": sid, "file": p.name, "path": str(dst), **info})
    (ANIME_OUT / "MANIFEST.json").write_text(json.dumps({"n": len(rows), "items": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    make_contact_sheet(rows, OUT / "anime30_contact_sheet.png")
    return rows


def representative_triplets(n: int) -> list[tuple[int, int, int]]:
    """Small deterministic screen set spanning the 30 silhouettes.

    Full C(30,3) screening is intentionally left for a background job; this set
    gives a fast go/no-go signal for whether the library is worth promoting to
    256^3 construction.
    """
    triples = []
    for a in range(0, min(n, 10)):
        triples.append((a, (a + 10) % n, (a + 20) % n))
    for a in range(0, min(n, 10), 2):
        triples.append((a, (a + 7) % n, (a + 14) % n))
        triples.append((a + 1, (a + 13) % n, (a + 23) % n))
    for t in [(0, 3, 11), (1, 8, 16), (2, 12, 21), (4, 17, 29), (5, 19, 27),
              (6, 14, 24), (7, 18, 26), (9, 20, 28), (10, 15, 22), (13, 23, 29)]:
        if max(t) < n:
            triples.append(t)
    # Preserve order while removing duplicates.
    out = []
    seen = set()
    for t in triples:
        tt = tuple(sorted(t))
        if tt not in seen:
            seen.add(tt)
            out.append(tt)
    return out


def quick_screen_anime(rows: list[dict], res: int = 48, max_records: int = 50) -> list[dict]:
    paths = [r["path"] for r in rows]
    records = []
    for ids in representative_triplets(len(paths)):
        names = [rows[i]["id"] for i in ids]
        try:
            sils, _ = load_data_from_list([paths[i] for i in ids], True, (res, res), torch.device("cpu"), 3, False)
            masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils]
            candidates, aux = make_candidates(masks, res, True, print_mm=100.0, nozzle_mm=0.4, wall_safety=2.0)
            gt6 = masks + [np.fliplr(m) for m in masks]
            hull = aux["hull"].astype(bool)
            best = None
            for cname, vol0 in candidates.items():
                vol = vol0.astype(bool)
                if n6(vol) != 1:
                    continue
                ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=256)
                saving = float(1.0 - int(vol.sum()) / max(int(hull.sum()), 1))
                rec = {
                    "silhouettes": names,
                    "candidate": cname,
                    "mean_iou": round(float(ev["mean_iou"]), 4),
                    "min_iou": round(float(min(ev["view_ious"][:3])), 4),
                    "saving": round(saving, 4),
                    "single": True,
                    "screen_res": res,
                }
                if best is None or (rec["mean_iou"], 0.4 <= saving <= 0.85, rec["saving"]) > (best["mean_iou"], 0.4 <= best["saving"] <= 0.85, best["saving"]):
                    best = rec
            if best:
                best["quick_pass"] = bool(best["mean_iou"] >= 0.975 and best["min_iou"] >= 0.95 and 0.40 <= best["saving"] <= 0.85)
                records.append(best)
        except Exception as e:
            records.append({"silhouettes": names, "error": str(e), "quick_pass": False, "screen_res": res})
    records.sort(key=lambda r: (r.get("quick_pass", False), r.get("mean_iou", -1), r.get("saving", -1)), reverse=True)
    (OUT / "anime30_triplet_screen_64.json").write_text(json.dumps({"schema": "anime30_triplet_screen_v1", "n": len(records), "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(OUT / "anime30_triplet_screen_top.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["silhouettes", "candidate", "mean_iou", "min_iou", "saving", "single", "quick_pass", "screen_res"])
        wr.writeheader()
        for r in records[:max_records]:
            rr = dict(r)
            rr["silhouettes"] = " ".join(rr.get("silhouettes", []))
            wr.writerow({k: rr.get(k, "") for k in wr.fieldnames})
    return records


def load_json(path: str) -> dict:
    return json.load(open(ROOT / path, encoding="utf-8"))


def draft_tables() -> dict:
    nsa = load_json("results/nsa_style_shadowart_refs_12_10cm_256/MANIFEST.json")
    models = nsa["models"]
    ious = [m["iou"] for m in models]
    savings = [m["saving"] for m in models]

    cap = [
        ["Method", "View flexibility", "Material", "Print QA", "Structural screen"],
        ["Mitra", "fixed/protocol", "not reported", "not reported", "no"],
        ["Sadekar voxel", "fixed/protocol", "not reported", "fragment risk", "no"],
        ["NSA", "light/screen optimized", "volume reported", "fabrication shown", "no FEM/PINO"],
        ["Ours", "free+fixed candidates", "manifest-backed", "single+watertight+erosion", "PINO+ACS/FEM labels"],
    ]
    write_table_png(cap, OUT / "table1_capability_comparison_draft.png", "Draft Table 1: capability comparison")

    main = [
        ["Evidence", "Cases", "IoU", "Dice", "Printable / Material"],
        ["Mitra prior", "prior", "0.813", "0.897", "not reported"],
        ["Sadekar prior", "prior", "0.906", "0.951", "not reported; fragments after binarization"],
        ["Ours 10cm fixed/ref", str(len(models)), f"{np.mean(ious):.3f}", f"{np.mean([2*x/(1+x) for x in ious]):.3f}", f"12/12 QA; {min(savings)*100:.1f}-{max(savings)*100:.1f}% saving"],
    ]
    write_table_png(main, OUT / "table2_main_quantitative_draft.png", "Draft Table 2: main quantitative comparison")

    qa = [["Example", "Branch", "IoU", "Saving", "Single", "Watertight"]]
    for m in models:
        qa.append([m["example"], m["selected_branch"], f"{m['iou']:.4f}", f"{m['saving']*100:.1f}%", str(m["single"]), str(m["watertight"])])
    write_table_png(qa, OUT / "table3_material_print_qa_draft.png", "Draft Table 3: material and print QA")

    pino = load_json("results/decision_aware_pino_conservative_20260704.json")
    sel = pino["selected"]
    pr = [
        ["Split", "safe/total", "false pass", "auto safe rate"],
        ["candidate heldout", f"{sel['candidate_heldout']['safe']}/{sel['candidate_heldout']['n']}", str(sel["candidate_heldout"]["auto_false_pass"]), f"{sel['candidate_heldout']['auto_safe_rate']*100:.1f}%"],
        ["external heldout", f"{sel['external_heldout']['safe']}/{sel['external_heldout']['n']}", str(sel["external_heldout"]["auto_false_pass"]), f"{sel['external_heldout']['auto_safe_rate']*100:.1f}%"],
        ["near-boundary hardset", f"{sel['hardset_test_near_boundary']['safe']}/{sel['hardset_test_near_boundary']['n']}", str(sel["hardset_test_near_boundary"]["auto_false_pass"]), f"{sel['hardset_test_near_boundary']['auto_safe_rate']*100:.1f}%"],
        ["joint", "-", str(sel["joint_false_pass"]), f"{sel['joint_auto_safe_true']} accepted"],
    ]
    write_table_png(pr, OUT / "table4_pino_status_draft.png", "Draft Table 4: PINO+ACS status")
    return {"nsa_models": models}


def copy_panel_image(src: Path, size: int = 120) -> Image.Image:
    if src.exists():
        im = Image.open(src).convert("L").resize((size, size), Image.Resampling.NEAREST)
        return Image.merge("RGB", (im, im, im))
    return Image.new("RGB", (size, size), (240, 240, 240))


def find_result_images(model: dict) -> tuple[list[Path], list[Path]]:
    d = Path(model["voxels"]).parent
    inputs = sorted((d / "inputs").glob("*.png"))
    sils = sorted((d / "silhouettes").glob("*.png"))
    return inputs, sils


def draft_gallery(models: list[dict]) -> None:
    rows = models[:6]
    cell, pad, label_h = 116, 12, 42
    cols = 8
    w = pad + cols * (cell + pad)
    h = pad + len(rows) * (cell + label_h + pad) + 36
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    headers = ["model", "in0", "in1", "in2", "proj0", "proj1", "proj2", "metrics"]
    for c, hd in enumerate(headers):
        d.text((pad + c * (cell + pad), 8), hd, fill=(0, 0, 0), font=font(13, True))
    for r, m in enumerate(rows):
        y = 36 + r * (cell + label_h + pad)
        x0 = pad
        d.text((x0, y + 4), m["example"][:16], fill=(0, 0, 0), font=font(11, True))
        d.text((x0, y + 24), f"{m['selected_branch']}/{m['candidate']}", fill=(40, 40, 40), font=font(10))
        inputs, sils = find_result_images(m)
        for i in range(3):
            im.paste(copy_panel_image(inputs[i] if i < len(inputs) else Path(""), cell), (pad + (i + 1) * (cell + pad), y))
            im.paste(copy_panel_image(sils[i] if i < len(sils) else Path(""), cell), (pad + (i + 4) * (cell + pad), y))
        mx = pad + 7 * (cell + pad)
        d.text((mx, y + 16), f"IoU {m['iou']:.4f}", fill=(0, 0, 0), font=font(12, True))
        d.text((mx, y + 40), f"save {m['saving']*100:.1f}%", fill=(0, 0, 0), font=font(12))
        d.text((mx, y + 64), f"QA {m['single']}/{m['watertight']}", fill=(0, 0, 0), font=font(12))
    im.save(OUT / "fig_physical_gallery_draft.png")


def draft_aaai_panel() -> None:
    man = load_json("results/curated_symbolic_fixed_10cm_256/MANIFEST.json")
    items = [r for r in man["examples"] if r["example"].startswith("fig1_")]
    cell, pad = 132, 12
    w = pad + 5 * (cell + pad)
    h = pad + len(items) * (cell + 48 + pad) + 24
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    headers = ["example", "input0", "input1", "input2", "status"]
    for c, hd in enumerate(headers):
        d.text((pad + c * (cell + pad), 8), hd, fill=(0, 0, 0), font=font(13, True))
    for r, item in enumerate(items):
        y = 36 + r * (cell + 48 + pad)
        d.text((pad, y + 8), item["example"], fill=(0, 0, 0), font=font(12, True))
        d.text((pad, y + 30), item["selected"]["candidate"], fill=(40, 40, 40), font=font(10))
        base = Path(item["selected"].get("voxels", "")).parent if item["accepted"] else ROOT / "results/curated_symbolic_fixed_10cm_256/rejected" / item["example"]
        inputs = sorted((base / "inputs").glob("*.png"))
        for i in range(3):
            im.paste(copy_panel_image(inputs[i] if i < len(inputs) else Path(""), cell), (pad + (i + 1) * (cell + pad), y))
        sx = pad + 4 * (cell + pad)
        d.text((sx, y + 12), "ACCEPT" if item["accepted"] else "REJECT", fill=(0, 110, 0) if item["accepted"] else (160, 0, 0), font=font(14, True))
        d.text((sx, y + 40), f"IoU {item['selected']['iou']}", fill=(0, 0, 0), font=font(12))
        d.text((sx, y + 62), f"save {item['selected']['saving']*100:.1f}%", fill=(0, 0, 0), font=font(12))
    im.save(OUT / "fig_aaai_10cm_status_draft.png")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = preprocess_anime()
    screen = quick_screen_anime(rows)
    data = draft_tables()
    draft_gallery(data["nsa_models"])
    draft_aaai_panel()
    summary = {
        "anime30": {
            "n": len(rows),
            "foreground_ratio_source_range": [min(r["foreground_ratio_source"] for r in rows), max(r["foreground_ratio_source"] for r in rows)],
            "quick_screen_res": 64,
            "triplets_screened": len(screen),
            "quick_pass_count": sum(1 for r in screen if r.get("quick_pass")),
            "top5": screen[:5],
        },
        "outputs": sorted(str(p.relative_to(ROOT)) for p in OUT.glob("*")),
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
