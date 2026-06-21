# -*- coding: utf-8 -*-
"""Smoke tests for shared utility functions."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch.nn as nn
from PIL import Image

CODE_DIR = pathlib.Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from utils import (  # noqa: E402
    Letterbox,
    binary_auc,
    confusion_counts,
    count_parameters,
    crop_by_roi,
    model_size_mb,
    pig_id_from_path,
    safe_div,
)


def main() -> None:
    assert pig_id_from_path("Dataset/full/exposed/pig01/a.jpg") == "pig01"
    assert safe_div(4, 2) == 2.0
    assert safe_div(1, 0) == 0.0

    img = Image.new("RGB", (100, 50), (255, 0, 0))
    cropped = crop_by_roi(img, (0.1, 0.2, 0.9, 0.8))
    assert cropped.size == (80, 30)

    boxed = Letterbox(64)(img)
    assert boxed.size == (64, 64)

    cm = confusion_counts(np.array([0, 1, 1]), np.array([0, 0, 1]), 2)
    assert cm.tolist() == [[1, 0], [1, 1]]
    assert binary_auc(np.array([0, 1]), np.array([0.1, 0.9])) == 1.0

    model = nn.Linear(4, 2)
    assert count_parameters(model) == 10
    assert model_size_mb(model) > 0
    print("[utils] shared utilities")


if __name__ == "__main__":
    main()
