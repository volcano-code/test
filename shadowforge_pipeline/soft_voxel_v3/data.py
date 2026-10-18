"""Target loading for the fixed three-view P-CRAFT soft experiment.

This module freezes the target convention used by the validated v2 protocol:
Otsu binarization, nearest-neighbour resize, union foreground bounding box,
and the historical independent-axis 70% centering transform.  Mirrored views
are produced by flipping each raw input and then normalizing it independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch


def normalize_target(
    image: np.ndarray,
    size: int,
    preserve_aspect: bool = False,
) -> np.ndarray:
    """Binarize and center one target using the frozen target convention."""

    if image.ndim != 2:
        raise ValueError("target image must be grayscale")
    if size <= 0:
        raise ValueError("size must be positive")
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.resize(binary, (size, size), interpolation=cv2.INTER_NEAREST)
    foreground = cv2.findNonZero(binary)
    if foreground is None:
        return binary
    x, y, width, height = cv2.boundingRect(foreground)
    if preserve_aspect:
        scale = min(0.7 * size / max(width, 1), 0.7 * size / max(height, 1))
        matrix = np.float32(
            [
                [scale, 0, size / 2 - scale * (x + width / 2)],
                [0, scale, size / 2 - scale * (y + height / 2)],
            ]
        )
    else:
        scale_x = 0.7 * size / max(width, 1)
        scale_y = 0.7 * size / max(height, 1)
        matrix = np.float32(
            [
                [scale_x, 0, size / 2 * (1 - scale_x) + (size / 2 - (x + width / 2))],
                [0, scale_y, size / 2 * (1 - scale_y) + (size / 2 - (y + height / 2))],
            ]
        )
    return cv2.warpAffine(binary, matrix, (size, size), flags=cv2.INTER_NEAREST)


def _read_grayscale(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_targets(
    paths: Sequence[str | Path],
    mirror: bool,
    size: int,
    device: torch.device,
    preserve_aspect: bool = False,
) -> torch.Tensor:
    """Load normalized base targets followed by independently normalized flips."""

    images = [
        normalize_target(_read_grayscale(path), size, preserve_aspect)
        for path in paths
    ]
    if mirror:
        images.extend(
            normalize_target(cv2.flip(_read_grayscale(path), 1), size, preserve_aspect)
            for path in paths
        )
    array = np.stack(images).astype(np.float32) / 255.0
    return torch.from_numpy(array).to(device)
