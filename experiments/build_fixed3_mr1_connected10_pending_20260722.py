#!/usr/bin/env python3
"""Build a pending fixed-three-view benchmark from strictly connected masks.

The script uses input evidence only. It never reads outputs from Mitra, Sadekar,
or P-CRAFT. Every selected mask is one 4-connected foreground component, and
every three-view visual hull is one 6-connected volume. The resulting manifest
is marked pending so it cannot be mistaken for an approved benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
POOLS = {
    "anime": Path("/home/lin/xvxinyao/shadow/anime_shadow_processed_black_bg_light_simplified"),
    "normal": Path("/home/lin/xvxinyao/shadow/normal_shadow"),
    "bench3": ROOT / "data/silhouettes/bench3",
    "complex": ROOT / "data/silhouettes/complex_v2_filled",
}
S4_2D = ndimage.generate_binary_structure(2, 1)
S8_2D = ndimage.generate_binary_structure(2, 2)
S6_3D = ndimage.generate_binary_structure(3, 1)


@lru_cache(maxsize=1024)
def _load_mask_cached(path_text: str, resolution: int) -> np.ndarray:
    path = Path(path_text)
    image = Image.open(path).convert("L").resize((resolution, resolution), Image.Resampling.NEAREST)
    mask = np.asarray(image) >= 128
    if float(mask.mean()) > 0.5:
        mask = ~mask
    return mask.astype(bool)


def load_mask(path: Path, resolution: int) -> np.ndarray:
    return _load_mask_cached(str(path.resolve()), resolution)


def component_count(mask: np.ndarray, structure: np.ndarray) -> int:
    return int(ndimage.label(mask, structure=structure)[1])


def mask_metrics(path: Path) -> dict | None:
    mask = load_mask(path, 256)
    if not mask.any():
        return None
    components4 = component_count(mask, S4_2D)
    components8 = component_count(mask, S8_2D)
    mirrored4 = component_count(np.fliplr(mask), S4_2D)
    fraction = float(mask.mean())
    if components4 != 1 or components8 != 1 or mirrored4 != 1:
        return None
    if not 0.045 <= fraction <= 0.38:
        return None
    eroded = ndimage.binary_erosion(mask, structure=S8_2D)
    boundary = mask & ~eroded
    distance = ndimage.distance_transform_edt(mask)
    filled = ndimage.binary_fill_holes(mask)
    holes = component_count(filled & ~mask, S8_2D)
    area = max(int(mask.sum()), 1)
    return {
        "path": str(path.resolve()),
        "foreground_fraction": fraction,
        "components_4": components4,
        "components_8": components8,
        "mirrored_components_4": mirrored4,
        "holes_8": holes,
        "boundary_over_sqrt_area": float(boundary.sum() / np.sqrt(area)),
        "thin_fraction_r2": float(np.logical_and(mask, distance <= 2.0).sum() / area),
    }


def normalize_complexity(rows: list[dict]) -> None:
    names = ("boundary_over_sqrt_area", "thin_fraction_r2", "holes_8")
    normalized = {}
    for name in names:
        values = np.asarray([float(row[name]) for row in rows])
        lo, hi = np.percentile(values, [10, 90])
        normalized[name] = np.clip((values - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)
    for index, row in enumerate(rows):
        row["complexity"] = float(
            0.55 * normalized["boundary_over_sqrt_area"][index]
            + 0.35 * normalized["thin_fraction_r2"][index]
            + 0.10 * normalized["holes_8"][index]
        )


def visual_hull(masks: list[np.ndarray]) -> np.ndarray:
    x, y, z = masks
    return (
        np.broadcast_to(x[None, :, :], (x.shape[0],) * 3)
        & np.broadcast_to(y[:, None, :], (y.shape[0],) * 3)
        & np.broadcast_to(z[:, :, None], (z.shape[0],) * 3)
    )


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum() / max(union, 1))


def triple_audit(rows: tuple[dict, dict, dict], resolution: int) -> dict | None:
    masks = [load_mask(Path(row["path"]), resolution) for row in rows]
    hull = visual_hull(masks)
    if not hull.any():
        return None
    components = component_count(hull, S6_3D)
    if components != 1:
        return None
    projections = [hull.any(axis=axis) for axis in range(3)]
    view_ious = [iou(pred, target) for pred, target in zip(projections, masks, strict=True)]
    if min(view_ious) < 0.72:
        return None
    return {
        "resolution": resolution,
        "occupied_voxels": int(hull.sum()),
        "occupancy_fraction": float(hull.mean()),
        "components_6": components,
        "view_ious": view_ious,
        "mean_iou": float(np.mean(view_ious)),
        "min_iou": float(np.min(view_ious)),
    }


def collect_rows() -> list[dict]:
    rows = []
    seen_masks: set[str] = set()
    for source, folder in POOLS.items():
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.png")):
            if "contact" in path.name.lower():
                continue
            result = mask_metrics(path)
            if result is not None:
                digest = hashlib.sha1(load_mask(path, 256).tobytes()).hexdigest()
                if digest in seen_masks:
                    continue
                seen_masks.add(digest)
                result["source"] = source
                result["mask_sha1"] = digest
                rows.append(result)
    normalize_complexity(rows)
    return sorted(rows, key=lambda row: (-row["complexity"], row["path"]))


def choose_triples(rows: list[dict], count: int, seed: int) -> list[tuple[tuple[dict, dict, dict], dict]]:
    rng = random.Random(seed)
    pool = rows[:80]
    candidates = []
    attempts = 30000
    for _ in range(attempts):
        triple = tuple(rng.sample(pool, 3))
        screen = triple_audit(triple, 64)
        if screen is None:
            continue
        score = (
            0.65 * float(np.mean([row["complexity"] for row in triple]))
            + 0.35 * screen["min_iou"]
        )
        candidates.append((score, triple, screen))
    candidates.sort(key=lambda item: -item[0])

    selected = []
    used: set[str] = set()
    signatures: set[tuple[str, ...]] = set()
    for _, triple, screen in candidates:
        paths = tuple(row["path"] for row in triple)
        signature = tuple(sorted(paths))
        if signature in signatures or any(path in used for path in paths):
            continue
        final = triple_audit(triple, 256)
        if final is None:
            continue
        selected.append((triple, final))
        signatures.add(signature)
        used.update(paths)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Only found {len(selected)} disjoint connected triples; requested {count}")
    return selected


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def make_contact_sheet(cases: list[dict], output: Path) -> None:
    tile = 168
    label_width = 145
    header = 42
    canvas = Image.new("RGB", (label_width + 3 * tile, header + len(cases) * tile), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for col, axis in enumerate(("X", "Y", "Z")):
        draw.text((label_width + col * tile + 72, 14), f"View {axis}", fill="black", font=font)
    for row_index, case in enumerate(cases):
        y = header + row_index * tile
        draw.text((12, y + 54), case["case_id"], fill="black", font=font)
        draw.text((12, y + 74), "1 comp / hull 1", fill=(0, 110, 90), font=font)
        for col, path in enumerate(case["target_paths_xyz"]):
            mask = Image.open(path).convert("L").resize((tile - 12, tile - 12), Image.Resampling.NEAREST)
            panel = Image.new("RGB", (tile, tile), "black")
            panel.paste(mask.convert("RGB"), (6, 6))
            canvas.paste(panel, (label_width + col * tile, y))
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/fixed3_mr1_connected10_pending_20260722",
    )
    args = parser.parse_args()
    rows = collect_rows()
    selected = choose_triples(rows, args.tasks, args.seed)
    inputs = args.output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, (triple, hull_audit) in enumerate(selected, 1):
        case_id = f"connected_{index:02d}"
        saved_paths = []
        view_audits = []
        for axis, row in zip(("x", "y", "z"), triple, strict=True):
            destination = inputs / f"{case_id}_view_{axis}.png"
            mask = load_mask(Path(row["path"]), 256)
            save_mask(mask, destination)
            saved_paths.append(str(destination.resolve()))
            view_audits.append({
                "axis": axis,
                "source_path": row["path"],
                "canonical_path": str(destination.resolve()),
                **{key: value for key, value in row.items() if key not in {"path"}},
                "canonical_components_4": component_count(mask, S4_2D),
                "mirrored_components_4": component_count(np.fliplr(mask), S4_2D),
            })
        cases.append({
            "case_id": case_id,
            "stratum": "strictly-connected-input-challenge",
            "target_paths_xyz": saved_paths,
            "sadekar_tfs_targets": [saved_paths[1], saved_paths[0], saved_paths[2]],
            "input_connectivity_audit": view_audits,
            "visual_hull_audit": hull_audit,
        })

    payload = {
        "schema": "fixed3_mr1_connected10_pending_v1",
        "status": "PENDING_USER_APPROVAL",
        "purpose": "Strictly connected input benchmark proposed for Table 1 and Figure 4",
        "selection_policy": (
            "Input-only selection: exactly one 4-connected foreground component per original and mirrored mask; "
            "exactly one 6-connected three-view visual hull; no method output used."
        ),
        "protocol": {
            "views": 3,
            "declared_rendered_views": 6,
            "axes": ["x", "y", "z"],
            "camera": "fixed orthographic TFS",
            "tsf": 1,
            "mr": 1,
            "zdist": 1.7,
            "resolution": 256,
            "seeds": [17, 29, 43],
            "evaluation": "shared canonical masks and prescribed mirrors; no registration",
        },
        "candidate_pool": {"audited_masks": len(rows)},
        "cases": cases,
    }
    manifest = args.output / "benchmark_manifest.pending.json"
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    make_contact_sheet(cases, args.output / "connected10_contact_sheet.png")
    print(json.dumps({
        "output": str(args.output),
        "manifest": str(manifest),
        "status": payload["status"],
        "tasks": len(cases),
        "audited_connected_masks": len(rows),
        "minimum_hull_iou": min(case["visual_hull_audit"]["min_iou"] for case in cases),
        "all_input_components_4": sorted({
            view["canonical_components_4"] for case in cases for view in case["input_connectivity_audit"]
        }),
        "all_mirrored_components_4": sorted({
            view["mirrored_components_4"] for case in cases for view in case["input_connectivity_audit"]
        }),
        "all_hull_components_6": sorted({case["visual_hull_audit"]["components_6"] for case in cases}),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
