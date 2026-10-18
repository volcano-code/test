#!/usr/bin/env python3
"""Composite the original photographed model into the best cleaned background."""

from pathlib import Path

import cv2
import numpy as np


ORIGINAL = Path(
    "/home/lin/下载/"
    "_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=1048800200845345766"
    "&skey=@crypt_711ac12b_2f68271e309b281999963c560aef59e3"
    "&mmweb_appid=wx_webfilehelper.jpeg"
)
BACKGROUND = Path(
    "/home/lin/xvxinyao/ShadowForge/paper_aaai_submission/"
    "physical_experiment_figures_20260724/"
    "aaai_maple_clean_white_box_preview_v7.png"
)
OUTPUT = Path(
    "/home/lin/xvxinyao/ShadowForge/paper_aaai_submission/"
    "physical_experiment_figures_20260724/"
    "aaai_maple_clean_box_original_model_locked_v9.png"
)


def polygon_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
    return mask


def main() -> None:
    original = cv2.imread(str(ORIGINAL), cv2.IMREAD_COLOR)
    background = cv2.imread(str(BACKGROUND), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(ORIGINAL)
    if background is None:
        raise FileNotFoundError(BACKGROUND)
    if original.shape != background.shape:
        background = cv2.resize(
            background,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )

    height, width = original.shape[:2]
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    # The physical object occupies this fixed polygon in the original photograph.
    # Color segmentation inside it avoids copying tape or the cardboard surface.
    roi = polygon_mask(
        (height, width),
        [(530, 488), (785, 470), (866, 552), (850, 704), (700, 800), (535, 690)],
    )
    blue_body = cv2.inRange(
        hsv,
        np.array([82, 42, 8], dtype=np.uint8),
        np.array([148, 255, 215], dtype=np.uint8),
    )
    dark_detail = cv2.inRange(
        hsv,
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([179, 255, 78], dtype=np.uint8),
    )
    # Dark pixels are accepted only when they are immediately attached to the
    # blue printed body. This prevents transparent tape, support residue, and
    # dirty contact marks below the object from entering the model cutout.
    blue_anchor = cv2.dilate(
        blue_body, np.ones((5, 5), dtype=np.uint8), iterations=1
    )
    attached_dark = cv2.bitwise_and(dark_detail, blue_anchor)
    lower_chroma = cv2.inRange(
        hsv,
        np.array([72, 32, 0], dtype=np.uint8),
        np.array([155, 255, 255], dtype=np.uint8),
    )
    lower_region = np.zeros((height, width), dtype=np.uint8)
    lower_region[680:, :] = 255
    attached_dark = cv2.bitwise_and(
        attached_dark,
        cv2.bitwise_or(cv2.bitwise_not(lower_region), lower_chroma),
    )
    candidate = cv2.bitwise_and(
        cv2.bitwise_or(blue_body, attached_dark), roi
    )
    damage_exclusion = polygon_mask(
        (height, width),
        [(790, 535), (885, 535), (904, 624), (863, 677), (792, 651)],
    )
    candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(damage_exclusion))

    # Retain connected components that overlap the central body seed. This keeps
    # thin printed fins while rejecting unrelated dark marks in the ROI.
    seed = np.zeros((height, width), dtype=np.uint8)
    seed[520:735, 590:815] = 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8), connectivity=8
    )
    keep = np.zeros((height, width), dtype=np.uint8)
    for label in range(1, count):
        component = labels == label
        if np.any(component & (seed > 0)) and stats[label, cv2.CC_STAT_AREA] >= 4:
            keep[component] = 255

    keep = cv2.morphologyEx(
        keep, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )
    core = cv2.erode(keep, np.ones((3, 3), dtype=np.uint8), iterations=1)
    edge = cv2.subtract(
        cv2.dilate(keep, np.ones((3, 3), dtype=np.uint8), iterations=1), core
    )

    # Hard-copy the interior and blend only the one-pixel contour. The cleaned
    # background already contains a plausible contact shadow, so no tape or
    # cardboard pixels from the original photograph are copied.
    alpha = np.zeros((height, width), dtype=np.float32)
    alpha[core > 0] = 1.0
    alpha = np.maximum(
        alpha,
        0.92 * (cv2.GaussianBlur(edge, (0, 0), 0.7).astype(np.float32) / 255.0),
    )
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    result = np.clip(
        background.astype(np.float32) * (1.0 - alpha)
        + original.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)

    # Re-assert exact source pixels throughout the protected model core.
    result[core > 0] = original[core > 0]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), result):
        raise RuntimeError(f"Could not write {OUTPUT}")

    outside = cv2.bitwise_not(
        cv2.dilate(
            keep,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
    )
    print(
        {
            "output": str(OUTPUT),
            "model_core_pixels": int(np.count_nonzero(core)),
            "model_core_max_channel_diff": int(
                cv2.absdiff(original, result)[core > 0].max()
            ),
            "background_outside_composite_max_channel_diff": int(
                cv2.absdiff(background, result)[outside > 0].max()
            ),
        }
    )


if __name__ == "__main__":
    main()
