#!/usr/bin/env python3
"""Conservatively retouch a physical-experiment photo while locking evidence."""

from pathlib import Path

import cv2
import numpy as np


SOURCE = Path(
    "/home/lin/下载/"
    "_cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=1048800200845345766"
    "&skey=@crypt_711ac12b_2f68271e309b281999963c560aef59e3"
    "&mmweb_appid=wx_webfilehelper.jpeg"
)
OUTPUT = Path(
    "/home/lin/xvxinyao/ShadowForge/paper_aaai_submission/"
    "physical_experiment_figures_20260724/"
    "aaai_maple_real_photo_background_retouch_v5.png"
)


def main() -> None:
    image = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(SOURCE)

    original = image.copy()
    height, width = image.shape[:2]
    if (width, height) != (1706, 1279):
        raise ValueError(f"Unexpected image size: {(width, height)}")

    # Mild background-only cleanup: suppress color cast and small sensor noise.
    smooth = cv2.bilateralFilter(image, d=5, sigmaColor=16, sigmaSpace=16)
    cleaned = cv2.addWeighted(image, 0.82, smooth, 0.18, 0)

    # Remove only isolated wall blemishes visible in the source.
    blemish_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(blemish_mask, (391, 183), 5, 255, -1)
    cv2.circle(blemish_mask, (1678, 451), 4, 255, -1)
    cleaned = cv2.inpaint(cleaned, blemish_mask, 5, cv2.INPAINT_TELEA)

    # Detect only the brown damaged cardboard patch beside the model. Replace
    # its pixels with intact white box texture sampled farther to the right.
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    patch_mask = np.zeros((height, width), dtype=np.uint8)
    roi = np.zeros_like(patch_mask)
    roi[545:650, 785:880] = 255
    brown = cv2.inRange(hsv, np.array([0, 22, 20]), np.array([40, 255, 235]))
    patch_mask = cv2.bitwise_and(brown, roi)
    patch_mask = cv2.morphologyEx(
        patch_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )
    patch_mask = cv2.dilate(
        patch_mask, np.ones((5, 5), np.uint8), iterations=2
    )
    patch_bounds = np.zeros_like(patch_mask)
    patch_polygon = np.array(
        [[790, 550], [860, 552], [880, 600], [860, 645], [805, 638], [785, 595]],
        dtype=np.int32,
    )
    cv2.fillPoly(patch_bounds, [patch_polygon], 255)
    patch_mask = cv2.bitwise_and(patch_mask, patch_bounds)

    intact_texture = np.roll(original, shift=-190, axis=1)
    alpha = cv2.GaussianBlur(patch_mask, (0, 0), 2.2).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    cleaned = np.clip(
        cleaned.astype(np.float32) * (1.0 - alpha)
        + intact_texture.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)

    # Lock both illuminated shadow regions exactly to the source photograph.
    evidence_mask = np.zeros((height, width), dtype=np.uint8)
    evidence_mask[125:575, 235:720] = 255
    evidence_mask[120:650, 920:1585] = 255

    # Lock the actual blue printed model with a color mask. Blue model pixels
    # are restored even where the repair mask feathers across its boundary.
    model_roi = np.zeros_like(evidence_mask)
    model_roi[475:815, 555:955] = 255
    blue = cv2.inRange(hsv, np.array([80, 24, 15]), np.array([145, 255, 190]))
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 62]))
    model_mask = cv2.bitwise_and(cv2.bitwise_or(blue, dark), model_roi)
    model_mask = cv2.morphologyEx(
        model_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    model_mask = cv2.dilate(model_mask, np.ones((3, 3), np.uint8), iterations=1)
    model_mask = cv2.bitwise_and(model_mask, cv2.bitwise_not(patch_mask))
    evidence_mask = cv2.bitwise_or(evidence_mask, model_mask)

    cleaned[evidence_mask > 0] = original[evidence_mask > 0]
    cleaned[blue > 0] = original[blue > 0]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), cleaned):
        raise RuntimeError(f"Failed to write {OUTPUT}")

    protected_diff = cv2.absdiff(original, cleaned)[evidence_mask > 0]
    max_diff = int(protected_diff.max()) if protected_diff.size else 0
    changed = int(np.count_nonzero(np.any(original != cleaned, axis=2)))
    print(
        {
            "output": str(OUTPUT),
            "protected_max_channel_diff": max_diff,
            "changed_background_pixels": changed,
            "repaired_patch_pixels": int(np.count_nonzero(patch_mask)),
        }
    )


if __name__ == "__main__":
    main()
