# -*- coding: utf-8 -*-
"""Smoke tests for the organized SqueezeNet-family models.

These tests do not train. They only verify that every model family used by the
experiments can be constructed and can run a forward pass.
"""

from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from models.factory import build_model  # noqa: E402


def _forward(model):
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (2, 2)


def test_baseline_squeezenet_forward():
    model = build_model(num_classes=2, variant="baseline", pool_type="gap", pretrained=False)
    _forward(model)


def test_fusion_only_forward():
    model = build_model(
        num_classes=2,
        variant="fusion_only",
        msfu_mode="fusion_only",
        pool_type="gap",
        pretrained=False,
        qk_dim=16,
    )
    _forward(model)


def test_msfunet_full_dual_forward():
    model = build_model(
        num_classes=2,
        variant="msfunet",
        msfu_mode="full",
        pool_type="guided",
        pretrained=False,
        qk_dim=16,
    )
    _forward(model)


def test_msfunet_deep_only_forward():
    model = build_model(
        num_classes=2,
        variant="msfunet",
        msfu_mode="full",
        bqs_mode="deep",
        pool_type="guided",
        pretrained=False,
        qk_dim=16,
    )
    _forward(model)


def test_layer_ablation_disable_mid_forward():
    model = build_model(
        num_classes=2,
        variant="msfunet",
        msfu_mode="full",
        tap_idx_z=5,
        tap_idx_y=-1,
        pool_type="guided",
        pretrained=False,
        qk_dim=16,
    )
    _forward(model)


def test_msanet_forward():
    model = build_model(num_classes=2, variant="msanet35", pretrained=False)
    _forward(model)
