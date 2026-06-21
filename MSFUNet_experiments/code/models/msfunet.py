# -*- coding: utf-8 -*-
"""MSFUNet proposed model.

This file contains only the proposed model path:
- Fusion Only
- Full MSFUNet
- Deep-only / Shallow-only / Dual-score BQS ablations
- Pool3 / Pool5 / Fire9 feature-level ablations
"""

from ._squeezenet_core import (
    MSFUConfig,
    SQUEEZENET_TAPS,
    MultiScaleFeatureUnit,
    SqueezeNetConfig,
    SqueezeNetMSFUNet,
    SqueezeNetWithMSFU,
)


def build_msfunet(num_classes: int, **kwargs):
    """Build the proposed MSFUNet variant with the shared factory."""
    kwargs.setdefault("variant", "msfunet")
    from .factory import build_model

    return build_model(num_classes=num_classes, **kwargs)


__all__ = [
    "MSFUConfig",
    "SQUEEZENET_TAPS",
    "MultiScaleFeatureUnit",
    "SqueezeNetConfig",
    "SqueezeNetMSFUNet",
    "SqueezeNetWithMSFU",
    "build_msfunet",
]
