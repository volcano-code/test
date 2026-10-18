#!/usr/bin/env python3
"""Clean the cardboard support while preserving experimental evidence exactly."""

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
    "aaai_maple_real_photo_locked_box_cleanup_v8.png"
)


def polygon_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
    return mask


def fitted_plane(
    image: np.ndarray,
    mask: np.ndarray,
    samples: list[tuple[int, int]],
    target_offset: float,
) -> np.ndarray:
    """Fit a planar luminance field from selected clean cardboard locations."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    rows = []
    values = []
    for x, y in samples:
        rows.append([x, y, 1.0])
        values.append(lab[y, x, 0])
    coeff, *_ = np.linalg.lstsq(
        np.asarray(rows, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
        rcond=None,
    )
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    luminance = coeff[0] * xx + coeff[1] * yy + coeff[2] + target_offset
    luminance = np.clip(luminance, 180, 246)

    # Retain broad photographic illumination but remove tape, labels, stains,
    # scratches, and corrugated high-frequency clutter.
    broad = cv2.GaussianBlur(lab[..., 0], (0, 0), 34)
    broad_centered = broad - float(np.median(broad[mask > 0]))
    luminance = np.clip(luminance + 0.18 * broad_centered, 180, 246)

    clean_lab = np.zeros_like(lab)
    clean_lab[..., 0] = luminance
    clean_lab[..., 1] = 128.0
    clean_lab[..., 2] = 128.0

    # Very restrained paper grain avoids a featureless computer-generated plane.
    rng = np.random.default_rng(20260724)
    grain = rng.normal(0.0, 0.75, size=luminance.shape).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 0.55)
    clean_lab[..., 0] = np.clip(clean_lab[..., 0] + grain, 0, 255)
    return cv2.cvtColor(clean_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def blend_region(
    destination: np.ndarray,
    source: np.ndarray,
    mask: np.ndarray,
    feather: float,
) -> np.ndarray:
    alpha = cv2.GaussianBlur(mask, (0, 0), feather).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    return np.clip(
        destination.astype(np.float32) * (1.0 - alpha)
        + source.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def main() -> None:
    original = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(SOURCE)
    height, width = original.shape[:2]
    if (width, height) != (1706, 1279):
        raise ValueError(f"Unexpected source dimensions: {(width, height)}")

    # The masks stop slightly inside every physical edge so that the real box
    # silhouette, seams, perspective, and depth cues remain untouched.
    top_mask = polygon_mask(
        (height, width),
        [(296, 505), (806, 318), (1288, 770), (656, 1180)],
    )
    left_face_mask = polygon_mask(
        (height, width),
        [(300, 528), (650, 1192), (716, 1278), (526, 1278), (315, 754)],
    )
    front_face_mask = polygon_mask(
        (height, width),
        [(666, 1195), (1290, 787), (1272, 1014), (1128, 1278), (738, 1278)],
    )
    box_mask = cv2.bitwise_or(
        top_mask, cv2.bitwise_or(left_face_mask, front_face_mask)
    )

    top_texture = fitted_plane(
        original,
        top_mask,
        [(360, 535), (690, 380), (1115, 672), (730, 1040), (1180, 790)],
        target_offset=7.0,
    )
    left_texture = fitted_plane(
        original,
        left_face_mask,
        [(340, 650), (455, 925), (585, 1130), (625, 1210)],
        target_offset=10.0,
    )
    front_texture = fitted_plane(
        original,
        front_face_mask,
        [(1190, 875), (1090, 1060), (910, 1190), (790, 1240)],
        target_offset=12.0,
    )

    cleaned = original.copy()
    cleaned = blend_region(cleaned, top_texture, top_mask, feather=2.4)
    cleaned = blend_region(cleaned, left_texture, left_face_mask, feather=2.0)
    cleaned = blend_region(cleaned, front_texture, front_face_mask, feather=2.0)

    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    # Pixel-lock the physical model and a narrow contact region around it.
    model_roi = np.zeros((height, width), dtype=np.uint8)
    model_roi[470:800, 545:825] = 255
    blue = cv2.inRange(hsv, np.array([80, 20, 12]), np.array([150, 255, 205]))
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 72]))
    model_mask = cv2.bitwise_and(cv2.bitwise_or(blue, dark), model_roi)
    model_mask = cv2.morphologyEx(
        model_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )
    model_mask = cv2.dilate(model_mask, np.ones((5, 5), np.uint8), iterations=2)
    damage_exclusion = polygon_mask(
        (height, width),
        [(794, 548), (867, 550), (885, 610), (860, 650), (802, 638)],
    )
    model_mask = cv2.bitwise_and(model_mask, cv2.bitwise_not(damage_exclusion))

    # Restore a narrow, softly feathered contact zone so the untouched model
    # retains its real grounding shadow instead of appearing composited.
    contact_mask = cv2.dilate(
        model_mask, np.ones((11, 11), np.uint8), iterations=2
    )
    contact_mask = cv2.bitwise_and(
        contact_mask, cv2.bitwise_not(damage_exclusion)
    )
    contact_alpha = (
        cv2.GaussianBlur(contact_mask, (0, 0), 3.0).astype(np.float32) / 255.0
    )[..., None]
    cleaned = np.clip(
        cleaned.astype(np.float32) * (1.0 - contact_alpha)
        + original.astype(np.float32) * contact_alpha,
        0,
        255,
    ).astype(np.uint8)

    # Lock both measured wall projections and their illumination halos.
    shadow_mask = np.zeros((height, width), dtype=np.uint8)
    shadow_mask[115:570, 220:730] = 255
    shadow_mask[105:655, 900:1600] = 255
    shadow_mask = cv2.bitwise_and(shadow_mask, cv2.bitwise_not(box_mask))
    evidence_mask = cv2.bitwise_or(model_mask, shadow_mask)
    cleaned[evidence_mask > 0] = original[evidence_mask > 0]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), cleaned):
        raise RuntimeError(f"Could not write {OUTPUT}")

    protected_diff = cv2.absdiff(original, cleaned)[evidence_mask > 0]
    changed_mask = np.any(original != cleaned, axis=2)
    outside_box = cv2.bitwise_not(
        cv2.dilate(box_mask, np.ones((11, 11), np.uint8), iterations=1)
    )
    outside_diff = cv2.absdiff(original, cleaned)[outside_box > 0]
    print(
        {
            "output": str(OUTPUT),
            "model_and_shadow_max_channel_diff": int(protected_diff.max()),
            "outside_box_max_channel_diff": int(outside_diff.max()),
            "changed_box_pixels": int(np.count_nonzero(changed_mask)),
        }
    )


if __name__ == "__main__":
    main()
