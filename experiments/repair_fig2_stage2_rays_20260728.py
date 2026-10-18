#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SOURCE = Path(
    "/home/lin/.codex/generated_images/"
    "019f119a-77c1-7752-9195-6d595e5d6c63/"
    "call_WCCMXQMTYQvAYqKxS3zXlXTS.png"
)
LEFT_TARGET = ROOT / (
    "results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709/"
    "dark_knight_left__magic_scholar_right/inputs/left_wall_x_view1_target.png"
)
RIGHT_TARGET = ROOT / (
    "results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709/"
    "dark_knight_left__magic_scholar_right/inputs/right_wall_y_view2_target.png"
)
OUTPUT = ROOT / (
    "paper_aaai_submission/fig2_pipeline_candidates_20260728/"
    "pcraft_pipeline_v18_candidate.png"
)


def panel_image(path: Path, size: int = 512, inset: float = 0.10) -> np.ndarray:
    target = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if target is None:
        raise FileNotFoundError(path)

    _, target = cv2.threshold(target, 127, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(target > 0)
    crop = target[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    canvas = np.zeros((size, size), dtype=np.uint8)
    available = int(size * (1.0 - 2.0 * inset))
    scale = min(available / crop.shape[1], available / crop.shape[0])
    resized = cv2.resize(
        crop,
        (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    y0 = (size - resized.shape[0]) // 2
    x0 = (size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def place_panel(
    image: np.ndarray,
    panel: np.ndarray,
    outer: np.ndarray,
    inner: np.ndarray,
) -> None:
    cv2.fillConvexPoly(image, outer, (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.polylines(image, [outer], True, (145, 152, 156), 2, cv2.LINE_AA)

    h, w = panel.shape[:2]
    source = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    transform = cv2.getPerspectiveTransform(source, inner.astype(np.float32))
    warped = cv2.warpPerspective(
        panel,
        transform,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, inner, 255, lineType=cv2.LINE_AA)
    image[mask > 0] = warped[mask > 0]
    cv2.polylines(image, [inner], True, (55, 62, 66), 2, cv2.LINE_AA)


def main() -> None:
    image = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(SOURCE)

    # Rebuild only the folded-panel region. All surrounding pipeline content is untouched.
    cv2.rectangle(image, (235, 176), (600, 640), (255, 255, 255), thickness=-1)

    left_outer = np.array([[250, 194], [419, 286], [419, 548], [250, 632]], np.int32)
    left_inner = np.array([[262, 210], [408, 293], [408, 537], [262, 612]], np.int32)
    right_outer = np.array([[421, 286], [590, 194], [590, 632], [421, 548]], np.int32)
    right_inner = np.array([[431, 293], [578, 210], [578, 612], [431, 537]], np.int32)

    place_panel(image, panel_image(LEFT_TARGET), left_outer, left_inner)
    place_panel(image, panel_image(RIGHT_TARGET), right_outer, right_inner)

    orange = (0, 112, 235)
    blue = (210, 78, 15)

    # Orange: left silhouette -> right, parallel to the right-panel bottom edge.
    for start in [(342, 342), (344, 367), (346, 392), (348, 417)]:
        end = (437, start[1] + 48)
        cv2.arrowedLine(image, start, end, orange, 3, cv2.LINE_AA, tipLength=0.11)

    # Blue: right silhouette -> left, parallel to the left-panel bottom edge.
    for start in [(501, 342), (499, 367), (497, 392), (495, 417)]:
        end = (406, start[1] + 48)
        cv2.arrowedLine(image, start, end, blue, 3, cv2.LINE_AA, tipLength=0.11)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), image):
        raise RuntimeError(f"Could not write {OUTPUT}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
