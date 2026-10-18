#!/usr/bin/env python3
"""Create a clean platform-only crop while preserving model and shadows."""

from pathlib import Path

import cv2
import numpy as np


SOURCE = Path(
    "/home/lin/xvxinyao/ShadowForge/paper_aaai_submission/"
    "physical_experiment_figures_20260724/"
    "aaai_maple_real_photo_background_retouch_v5.png"
)
OUTPUT = Path(
    "/home/lin/xvxinyao/ShadowForge/paper_aaai_submission/"
    "physical_experiment_figures_20260724/"
    "aaai_maple_real_photo_smooth_platform_crop_v6.png"
)


def main() -> None:
    original = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(SOURCE)

    height, width = original.shape[:2]
    if (width, height) != (1706, 1279):
        raise ValueError(f"Unexpected image size: {(width, height)}")

    output = original.copy()

    # Perspective outline of the visible upper platform.
    platform = np.array(
        [[286, 500], [808, 308], [1297, 770], [650, 1195]],
        dtype=np.int32,
    )
    platform_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(platform_mask, platform, 255)

    # Build a restrained warm-gray planar surface. The slight x/y variation
    # preserves depth while removing tape, corrugation, and cardboard damage.
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = np.clip((xx - 286.0) / (1297.0 - 286.0), 0.0, 1.0)
    ny = np.clip((yy - 308.0) / (1195.0 - 308.0), 0.0, 1.0)
    luminance = 204.0 - 24.0 * ny + 5.0 * nx
    plane = np.stack(
        [luminance - 2.0, luminance, luminance + 1.5],
        axis=2,
    )
    plane = np.clip(plane, 0, 255).astype(np.uint8)

    # Feather only the platform boundary by a few pixels.
    alpha = cv2.GaussianBlur(platform_mask, (0, 0), 1.3).astype(np.float32)
    alpha = (alpha / 255.0)[..., None]
    output = np.clip(
        output.astype(np.float32) * (1.0 - alpha)
        + plane.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)

    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    # Restore the printed model exactly. The mask covers its dark/blue body
    # and a narrow anti-aliased boundary without retaining the old platform.
    model_roi = np.zeros((height, width), dtype=np.uint8)
    model_roi[470:820, 535:970] = 255
    blue = cv2.inRange(hsv, np.array([75, 18, 7]), np.array([150, 255, 210]))
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 82]))
    model_mask = cv2.bitwise_and(cv2.bitwise_or(blue, dark), model_roi)
    model_mask = cv2.morphologyEx(
        model_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    model_mask = cv2.dilate(model_mask, np.ones((3, 3), np.uint8), iterations=1)
    # Restore the model after the platform replacement. All pixels outside
    # the platform polygon, including both walls and shadows, were never edited.
    output[model_mask > 0] = original[model_mask > 0]

    # Hide the cardboard body below the upper platform. The operation starts
    # below both measured shadows and is feathered into the white page.
    lower = np.zeros((height, width), dtype=np.uint8)
    lower[500:, :] = 255
    lower = cv2.bitwise_and(lower, cv2.bitwise_not(platform_mask))
    lower_alpha = cv2.GaussianBlur(lower, (0, 0), 2.0).astype(np.float32)
    lower_alpha = (lower_alpha / 255.0)[..., None]
    white = np.full_like(output, 255)
    output = np.clip(
        output.astype(np.float32) * (1.0 - lower_alpha)
        + white.astype(np.float32) * lower_alpha,
        0,
        255,
    ).astype(np.uint8)
    output[model_mask > 0] = original[model_mask > 0]

    # Tight landscape crop: the two shadows remain complete, while the lower
    # box body, labels, floor sheet, and surrounding clutter are excluded.
    x0, y0, x1, y1 = 180, 55, 1425, 925
    cropped = output[y0:y1, x0:x1]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), cropped):
        raise RuntimeError(f"Failed to write {OUTPUT}")

    # Audit source preservation before cropping.
    shadow_mask = np.zeros((height, width), dtype=np.uint8)
    shadow_mask[125:465, 235:720] = 255
    shadow_mask[120:465, 920:1585] = 255
    shadow_mask = cv2.bitwise_and(shadow_mask, cv2.bitwise_not(platform_mask))
    shadow_diff = cv2.absdiff(original, output)[shadow_mask > 0]
    model_diff = cv2.absdiff(original, output)[model_mask > 0]
    print(
        {
            "output": str(OUTPUT),
            "crop_size": [int(cropped.shape[1]), int(cropped.shape[0])],
            "shadow_max_channel_diff": int(shadow_diff.max()),
            "model_max_channel_diff": int(model_diff.max()),
        }
    )


if __name__ == "__main__":
    main()
