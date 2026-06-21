# -*- coding: utf-8 -*-
"""Dataset, ROI, and image transform helpers."""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Optional, Tuple

from PIL import Image


RoiBox = Tuple[float, float, float, float]


def pig_id_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3:
        return parts[-2]
    return parts[-1] if parts else "unknown_pig"


def load_roi_config(path: Optional[str]) -> Dict[str, RoiBox]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    cfg: Dict[str, RoiBox] = {}
    for key, value in raw_cfg.items():
        if isinstance(value, (list, tuple)) and len(value) == 4:
            cfg[key] = tuple(_clamp01(v) for v in value)  # type: ignore[assignment]
    return cfg


def crop_by_roi(img: Image.Image, roi: RoiBox) -> Image.Image:
    width, height = img.size
    x0, y0, x1, y1 = roi
    left = max(0, min(int(x0 * width), width - 1))
    top = max(0, min(int(y0 * height), height - 1))
    right = max(left + 1, min(int(x1 * width), width))
    bottom = max(top + 1, min(int(y1 * height), height))
    return img.crop((left, top, right, bottom))


def center_roi_box(img: Image.Image, keep_ratio: float) -> RoiBox:
    keep_ratio = max(0.1, min(float(keep_ratio), 1.0))
    width, height = img.size
    new_width = int(width * keep_ratio)
    new_height = int(height * keep_ratio)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    return left / width, top / height, (left + new_width) / width, (top + new_height) / height


def jitter_roi_box(roi: RoiBox, amount: float) -> RoiBox:
    if amount <= 0:
        return roi

    x0, y0, x1, y1 = map(float, roi)
    width = max(1e-6, x1 - x0)
    height = max(1e-6, y1 - y0)
    tx = random.uniform(-amount, amount) * width
    ty = random.uniform(-amount, amount) * height
    sx = random.uniform(1.0 - amount, 1.0 + amount)
    sy = random.uniform(1.0 - amount, 1.0 + amount)
    cx = (x0 + x1) * 0.5 + tx
    cy = (y0 + y1) * 0.5 + ty
    new_width = width * sx
    new_height = height * sy
    nx0 = _clamp01(cx - new_width * 0.5)
    ny0 = _clamp01(cy - new_height * 0.5)
    nx1 = _clamp01(cx + new_width * 0.5)
    ny1 = _clamp01(cy + new_height * 0.5)
    if nx1 - nx0 < 0.02:
        nx1 = _clamp01(nx0 + 0.02)
    if ny1 - ny0 < 0.02:
        ny1 = _clamp01(ny0 + 0.02)
    return nx0, ny0, nx1, ny1


class Letterbox:
    def __init__(self, out_size: int, pad_color=(114, 114, 114), scale_jitter: float = 0.0):
        self.out_size = int(out_size)
        self.pad_color = pad_color
        self.scale_jitter = float(scale_jitter)

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        scale = min(self.out_size / width, self.out_size / height)
        if self.scale_jitter > 0:
            low = max(0.5, 1.0 - self.scale_jitter)
            scale = min(scale * random.uniform(low, 1.0), 1.0)

        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        img = img.resize((new_width, new_height), Image.BILINEAR)

        canvas = Image.new("RGB", (self.out_size, self.out_size), self.pad_color)
        left = (self.out_size - new_width) // 2
        top = (self.out_size - new_height) // 2
        canvas.paste(img, (left, top))
        return canvas


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
