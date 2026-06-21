# -*- coding: utf-8 -*-
"""Comparison models.

MSANet is included only as a comparison baseline. The proposed model is
MSFUNet, defined in `msfunet.py`.
"""

from ._squeezenet_core import (
    MSAAddition,
    SqueezeNetBaseline,
    SqueezeNetWithAttention,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SqueezeNetSimple(nn.Module):
    """Original no-attention SqueezeNet comparison baseline."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        self.squeezenet = models.squeezenet1_1(weights=weights)
        self.squeezenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeezenet.classifier(self.squeezenet.features(x))
        return torch.flatten(x, 1)


class SqueezeNetWithDropout(nn.Module):
    """SqueezeNet baseline with global average pooling and configurable dropout."""

    def __init__(self, num_classes: int, dropout_rate: float = 0.5, pretrained: bool = True):
        super().__init__()
        weights = models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        self.squeezenet = models.squeezenet1_1(weights=weights)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeezenet.features(x).mean(dim=(2, 3))
        return self.fc(self.dropout(x))


class SqueezeNetWithBatchNorm(nn.Module):
    """SqueezeNet baseline with pooled-feature batch normalization."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        self.squeezenet = models.squeezenet1_1(weights=weights)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeezenet.features(x).mean(dim=(2, 3))
        return self.fc(self.dropout(F.relu(self.bn(x))))

# These exports are used for baseline/backbone comparisons, not as the proposed method.
__all__ = [
    "MSAAddition",
    "SqueezeNetBaseline",
    "SqueezeNetSimple",
    "SqueezeNetWithAttention",
    "SqueezeNetWithDropout",
    "SqueezeNetWithBatchNorm",
]
