# -*- coding: utf-8 -*-
"""Model size and parameter-count helpers."""

from __future__ import annotations

import os
import tempfile

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return int(sum(p.numel() for p in params))


def model_size_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        path = tmp.name
    try:
        torch.save(model.state_dict(), path)
        return os.path.getsize(path) / (1024 * 1024)
    finally:
        if os.path.exists(path):
            os.remove(path)
