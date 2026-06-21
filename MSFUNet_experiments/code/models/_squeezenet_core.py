# -*- coding: utf-8 -*-
"""Unified SqueezeNet-family models for the MSFUNet experiments.

This module contains the shared implementation for:

- SqueezeNet baseline
- MSANet comparison model
- Fusion Only
- MSFUNet Full
- MSFUNet ablation variants

Design pattern used here:

- Config objects describe experiment choices.
- Small modules implement one responsibility each.
- ModelFactory builds named variants for experiments and scripts.

The public compatibility function is `build_model(num_classes, **kwargs)`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


SQUEEZENET_TAPS: Dict[str, int] = {
    "conv1": 0,
    "relu1": 1,
    "pool1": 2,
    "fire2": 3,
    "fire3": 4,
    "pool3": 5,
    "fire4": 6,
    "fire5": 7,
    "pool5": 8,
    "fire6": 9,
    "fire7": 10,
    "fire8": 11,
    "fire9": 12,
}


@dataclass(frozen=True)
class MSFUConfig:
    """Configuration for the multi-scale feature unit and BQS weighting."""

    mode: str = "full"                 # "full" or "fusion_only"
    score_mode: str = "dual"           # "dual", "deep", "shallow"
    lambda_mode: str = "learnable"     # "learnable" or "fixed"
    lambda_fixed: float = 0.5
    topk_ratio: float = 0.0
    background_scale: float = 0.0
    init_gamma: float = 0.05
    use_style_norm: bool = True
    style_p: float = 0.0
    style_alpha: float = 0.0
    softk_tau: float = 0.5
    softk_alpha: float = 1.0
    use_coord_score: bool = True
    use_local_refine: bool = True
    qk_dim: int = 128
    use_se_in_fusion: bool = True


@dataclass(frozen=True)
class SqueezeNetConfig:
    """Configuration for the backbone, feature taps, and classifier head."""

    variant: str = "msfunet"            # "baseline", "fusion_only", "msfunet", "msanet35", "msanet53"
    pool_type: str = "guided"           # "guided", "gap", "attn", "gem"
    pretrained: bool = True
    tap_deep: int = SQUEEZENET_TAPS["fire9"]
    tap_shallow: int = SQUEEZENET_TAPS["pool3"]
    tap_mid: int = SQUEEZENET_TAPS["pool5"]
    msfu: MSFUConfig = MSFUConfig()


def _squeezenet11(pretrained: bool) -> nn.Module:
    """Create SqueezeNet-1.1 while supporting old and new torchvision APIs."""

    if not pretrained:
        return tv_models.squeezenet1_1(weights=None)
    try:
        weights = tv_models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        return tv_models.squeezenet1_1(weights=weights)
    except Exception:
        return tv_models.squeezenet1_1(pretrained=True)


def _normalize_choice(value: str, allowed: Iterable[str], default: str) -> str:
    value = (value or default).lower()
    return value if value in allowed else default


def make_coord_maps(h: int, w: int, device, dtype=None) -> torch.Tensor:
    """Create x/y/r coordinate maps used by coordinate-aware scoring."""

    ys = torch.linspace(0, 1, steps=h, device=device).view(h, 1).repeat(1, w)
    xs = torch.linspace(0, 1, steps=w, device=device).view(1, w).repeat(h, 1)
    rc = torch.sqrt((xs - 0.5) ** 2 + (ys - 0.5) ** 2)
    coord = torch.stack([xs, ys, rc], dim=0)
    return coord.to(dtype=dtype) if dtype is not None else coord


class StyleNorm(nn.Module):
    """Instance-normalization block used for optional style suppression."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


def maybe_style_blend(
    x: torch.Tensor,
    norm: Optional[nn.Module],
    probability: float,
    alpha: float,
) -> torch.Tensor:
    if norm is None or probability <= 0.0 or alpha <= 0.0:
        return x
    if torch.rand((), device=x.device) < probability:
        return (1.0 - alpha) * x + alpha * norm(x)
    return x


class SEBlock(nn.Module):
    """Squeeze-and-excitation block used after multi-scale fusion."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)


class GeM(nn.Module):
    """Generalized mean pooling head option."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=self.eps).pow(self.p)
        return x.mean(dim=(-1, -2)).pow(1.0 / self.p)


class FeatureTapper:
    """Capture selected SqueezeNet feature maps through forward hooks."""

    def __init__(self, features: nn.Module, tap_deep: int, tap_shallow: int, tap_mid: int):
        self.features = features
        self.indices = {"deep": int(tap_deep), "shallow": int(tap_shallow), "mid": int(tap_mid)}
        self.outputs: Dict[str, Optional[torch.Tensor]] = {"deep": None, "shallow": None, "mid": None}
        self._handles = []

    def reset(self) -> None:
        for key in self.outputs:
            self.outputs[key] = None

    def attach(self) -> None:
        self.detach()
        for name, index in self.indices.items():
            if index < 0:
                continue
            self._handles.append(self.features[index].register_forward_hook(self._make_hook(name)))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _make_hook(self, name: str):
        def hook(_, __, output):
            self.outputs[name] = output

        return hook


class PoolingHead(nn.Module):
    """Classifier head with GAP, GeM, attention, or guided pooling."""

    def __init__(self, channels: int, num_classes: int, pool_type: str):
        super().__init__()
        self.pool_type = _normalize_choice(pool_type, ("gap", "attn", "gem", "guided"), "gap")
        self.attn = nn.Conv2d(channels, 1, kernel_size=1) if self.pool_type == "attn" else None
        self.gem = GeM() if self.pool_type == "gem" else None
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, feat: torch.Tensor, guidance: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pool_type == "guided":
            if guidance is None:
                pooled = feat.mean(dim=(2, 3))
            else:
                numerator = (feat * guidance).sum(dim=(2, 3))
                denominator = guidance.sum(dim=(2, 3)).clamp_min(1e-6)
                pooled = numerator / denominator
        elif self.pool_type == "gem":
            pooled = self.gem(feat)
        elif self.pool_type == "attn":
            weights = self.attn(feat).flatten(1)
            weights = torch.softmax(weights, dim=1).view(feat.size(0), 1, feat.size(2), feat.size(3))
            pooled = (feat * weights).sum(dim=(2, 3))
        else:
            pooled = feat.mean(dim=(2, 3))
        return self.fc(pooled)


class MultiScaleFeatureUnit(nn.Module):
    """MSFUNet fusion block with optional BQS score guidance."""

    def __init__(
        self,
        config: MSFUConfig,
        in_ch_deep: int = 512,
        in_ch_mid: int = 256,
        in_ch_shallow: int = 128,
        proj_ratio: float = 0.5,
    ):
        super().__init__()
        self.config = config
        self.in_ch_deep = int(in_ch_deep)
        self.in_ch_mid = int(in_ch_mid)
        self.in_ch_shallow = int(in_ch_shallow)
        self.mid_ch = max(8, int(self.in_ch_deep * float(proj_ratio)))

        mode = _normalize_choice(config.mode, ("full", "fusion_only", "now"), "full")
        self.mode = "fusion_only" if mode == "now" else mode
        self.score_mode = _normalize_choice(config.score_mode, ("dual", "deep", "shallow", "shal"), "dual")
        if self.score_mode == "shal":
            self.score_mode = "shallow"
        self.lambda_mode = _normalize_choice(config.lambda_mode, ("learnable", "fixed"), "learnable")

        self.topk_ratio = float(max(0.0, min(1.0, config.topk_ratio)))
        self.background_scale = float(config.background_scale)
        self.softk_tau = float(max(1e-4, config.softk_tau))
        self.softk_alpha = float(max(0.0, min(1.0, config.softk_alpha)))
        self.qk_dim = int(config.qk_dim)

        self.style_deep = StyleNorm(self.in_ch_deep) if config.use_style_norm else None
        self.style_mid = StyleNorm(self.in_ch_mid) if config.use_style_norm else None
        self.style_shallow = StyleNorm(self.in_ch_shallow) if config.use_style_norm else None

        self.proj_deep = nn.Conv2d(self.in_ch_deep, self.in_ch_deep, kernel_size=1)
        self.proj_mid = nn.Conv2d(self.in_ch_mid, self.mid_ch, kernel_size=1)
        self.proj_shallow = nn.Conv2d(self.in_ch_shallow, self.mid_ch, kernel_size=1)

        fuse_layers = [
            nn.Conv2d(self.in_ch_deep + self.mid_ch * 2, self.in_ch_deep, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_ch_deep),
            nn.ReLU(inplace=True),
        ]
        if config.use_se_in_fusion:
            fuse_layers.append(SEBlock(self.in_ch_deep))
        self.fuse = nn.Sequential(*fuse_layers)

        self.gamma = nn.Parameter(torch.tensor(float(config.init_gamma), dtype=torch.float32))

        if self.mode == "full":
            self.q_deep = nn.Conv2d(self.in_ch_deep, self.qk_dim, 1, bias=False)
            self.k_shallow = nn.Conv2d(self.mid_ch * 2, self.qk_dim, 1, bias=False)
            self.q_shallow = nn.Conv2d(self.mid_ch * 2, self.qk_dim, 1, bias=False)
            self.k_deep = nn.Conv2d(self.in_ch_deep, self.qk_dim, 1, bias=False)
            self.coord_to_qd = nn.Conv2d(3, self.qk_dim, 1, bias=False) if config.use_coord_score else None
            self.coord_to_qs = nn.Conv2d(3, self.qk_dim, 1, bias=False) if config.use_coord_score else None
            self.coord_to_kd = nn.Conv2d(3, self.qk_dim, 1, bias=False) if config.use_coord_score else None
            self.coord_to_ks = nn.Conv2d(3, self.qk_dim, 1, bias=False) if config.use_coord_score else None
            self.local_refine = (
                nn.Conv2d(self.in_ch_deep, self.in_ch_deep, kernel_size=3, padding=1, groups=self.in_ch_deep)
                if config.use_local_refine
                else None
            )
            if self.score_mode == "dual" and self.lambda_mode == "learnable":
                self.lambda_raw = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
            else:
                self.lambda_raw = None
        else:
            self.q_deep = self.k_shallow = self.q_shallow = self.k_deep = None
            self.coord_to_qd = self.coord_to_qs = self.coord_to_kd = self.coord_to_ks = None
            self.local_refine = None
            self.lambda_raw = None

        self.adapt_mid: Optional[nn.Conv2d] = None
        self.adapt_shallow: Optional[nn.Conv2d] = None

    def _adapt_channels(self, x: torch.Tensor, expected: int, attr_name: str) -> torch.Tensor:
        if x.shape[1] == expected:
            return x
        layer = getattr(self, attr_name)
        needs_new = (
            layer is None
            or layer.in_channels != x.shape[1]
            or layer.out_channels != expected
            or next(layer.parameters()).device != x.device
            or layer.weight.dtype != x.dtype
        )
        if needs_new:
            layer = nn.Conv2d(x.shape[1], expected, kernel_size=1).to(device=x.device, dtype=x.dtype)
            setattr(self, attr_name, layer)
        return layer(x)

    def _soft_topk_weights(self, score: torch.Tensor, k: int) -> torch.Tensor:
        batch, _, height, width = score.shape
        flat = score.view(batch, -1)
        if k <= 0 or k >= height * width:
            return torch.ones_like(score)
        threshold = torch.topk(flat, k=k, dim=1).values[:, -1].view(batch, 1, 1, 1)
        soft = torch.sigmoid((score - threshold) / self.softk_tau)
        indices = torch.topk(flat, k=k, dim=1).indices
        hard_flat = torch.zeros_like(flat)
        hard_flat.scatter_(1, indices, 1.0)
        hard = hard_flat.view(batch, 1, height, width)
        return self.softk_alpha * soft + (1.0 - self.softk_alpha) * hard

    def _score_to_weight(self, score: torch.Tensor) -> torch.Tensor:
        mean = score.mean(dim=(2, 3), keepdim=True)
        std = score.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return torch.sigmoid(((score - mean) / std) / self.softk_tau)

    def forward(
        self,
        deep: torch.Tensor,
        shallow: torch.Tensor,
        mid: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, _, height, width = deep.shape

        mid = F.interpolate(mid, size=(height, width), mode="bilinear", align_corners=False)
        shallow = F.interpolate(shallow, size=(height, width), mode="bilinear", align_corners=False)
        mid = self._adapt_channels(mid, self.in_ch_mid, "adapt_mid")
        shallow = self._adapt_channels(shallow, self.in_ch_shallow, "adapt_shallow")

        cfg = self.config
        deep_n = maybe_style_blend(deep, self.style_deep, cfg.style_p, cfg.style_alpha)
        mid_n = maybe_style_blend(mid, self.style_mid, cfg.style_p, cfg.style_alpha)
        shallow_n = maybe_style_blend(shallow, self.style_shallow, cfg.style_p, cfg.style_alpha)

        p_deep = self.proj_deep(deep_n)
        p_mid = self.proj_mid(mid_n)
        p_shallow = self.proj_shallow(shallow_n)
        fused = self.fuse(torch.cat([p_deep, p_mid, p_shallow], dim=1))

        if self.mode == "fusion_only":
            return deep + self.gamma * fused, None

        shallow_context = torch.cat([p_mid, p_shallow], dim=1)
        q_deep = self.q_deep(p_deep)
        k_shallow = self.k_shallow(shallow_context)
        q_shallow = self.q_shallow(shallow_context)
        k_deep = self.k_deep(p_deep)

        if cfg.use_coord_score:
            coord = make_coord_maps(height, width, deep.device, p_deep.dtype).unsqueeze(0).repeat(batch, 1, 1, 1)
            q_deep = q_deep + self.coord_to_qd(coord)
            q_shallow = q_shallow + self.coord_to_qs(coord)
            k_deep = k_deep + self.coord_to_kd(coord)
            k_shallow = k_shallow + self.coord_to_ks(coord)

        scale = float(self.qk_dim) ** 0.5
        score_deep = (q_deep * k_shallow).sum(dim=1, keepdim=True) / scale
        score_shallow = (q_shallow * k_deep).sum(dim=1, keepdim=True) / scale

        if self.score_mode == "deep":
            score = score_deep
        elif self.score_mode == "shallow":
            score = score_shallow
        else:
            if self.lambda_mode == "fixed":
                lam = torch.tensor(cfg.lambda_fixed, device=deep.device, dtype=score_deep.dtype).clamp(0.0, 1.0)
            else:
                lam = torch.sigmoid(self.lambda_raw).to(dtype=score_deep.dtype)
            score = lam * score_deep + (1.0 - lam) * score_shallow

        weight = self._score_to_weight(score)
        if self.topk_ratio > 0.0:
            k = max(1, int(self.topk_ratio * height * width))
            weight = 0.5 * (weight + self._soft_topk_weights(score, k))

        if self.local_refine is not None:
            refined = self.local_refine(fused)
            fused = weight * refined + (1.0 - weight) * fused
        residual = weight * fused + self.background_scale * (1.0 - weight) * fused
        return deep + self.gamma * residual, weight


class SqueezeNetBaseline(nn.Module):
    """SqueezeNet baseline without MSFU or BQS."""

    def __init__(self, num_classes: int, pool_type: str = "gap", pretrained: bool = True):
        super().__init__()
        self.backbone = _squeezenet11(pretrained)
        self.head = PoolingHead(512, num_classes, pool_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.features(x))


class SqueezeNetMSFUNet(nn.Module):
    """SqueezeNet backbone with MSFU fusion and optional guided pooling."""

    def __init__(self, num_classes: int, config: SqueezeNetConfig):
        super().__init__()
        self.config = config
        self.backbone = _squeezenet11(config.pretrained)
        self.tapper = FeatureTapper(self.backbone.features, config.tap_deep, config.tap_shallow, config.tap_mid)
        pool_type = config.pool_type
        if config.variant == "fusion_only" and pool_type == "guided":
            pool_type = "gap"
        self.msfu = MultiScaleFeatureUnit(config.msfu)
        self.head = PoolingHead(512, num_classes, pool_type)

    def _zeros_like_deep(self, deep: torch.Tensor, channels: int) -> torch.Tensor:
        batch, _, height, width = deep.shape
        return torch.zeros((batch, channels, height, width), device=deep.device, dtype=deep.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.tapper.reset()
        self.tapper.attach()
        try:
            _ = self.backbone.features(x)
        finally:
            self.tapper.detach()

        deep = self.tapper.outputs["deep"]
        if deep is None:
            raise RuntimeError(f"Deep feature tap failed at index {self.config.tap_deep}.")

        shallow = self.tapper.outputs["shallow"]
        mid = self.tapper.outputs["mid"]
        if self.config.tap_shallow < 0 or shallow is None:
            shallow = self._zeros_like_deep(deep, 128)
        if self.config.tap_mid < 0 or mid is None:
            mid = self._zeros_like_deep(deep, 256)

        fused, weight = self.msfu(deep=deep, shallow=shallow, mid=mid)
        return self.head(fused, guidance=weight)


class MSAAddition(nn.Module):
    """MSANet pixel-wise matrix-addition comparison model.

    `order="pool3_pool5"` corresponds to the zy file.
    `order="pool5_pool3"` corresponds to the yz file.
    """

    def __init__(self, num_classes: int, order: str = "pool3_pool5", pretrained: bool = True):
        super().__init__()
        self.order = _normalize_choice(order, ("pool3_pool5", "pool5_pool3"), "pool3_pool5")
        self.backbone = _squeezenet11(pretrained)
        self.query = nn.Conv2d(512, 64, kernel_size=1)
        self.key = nn.Conv2d(512, 64, kernel_size=1)
        if self.order == "pool3_pool5":
            self.low_proj = nn.Conv2d(128, 169, kernel_size=1)
            self.mid_proj = nn.Conv2d(256, 169, kernel_size=1)
        else:
            self.low_proj = nn.Conv2d(256, 169, kernel_size=1)
            self.mid_proj = nn.Conv2d(128, 169, kernel_size=1)
        self.channel_proj = nn.Conv2d(169, 512, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features
        deep = features(x)
        before_pool3 = features[:5](x)
        pool3 = features[5](before_pool3)
        before_pool5 = features[:8](x)
        pool5 = features[8](before_pool5)
        batch, _, height, width = deep.shape
        n = height * width

        attention = torch.bmm(
            self.query(deep).view(batch, -1, n).permute(0, 2, 1),
            self.key(deep).view(batch, -1, n),
        )
        attention = torch.softmax(attention, dim=-1)

        first, second = (pool3, pool5) if self.order == "pool3_pool5" else (pool5, pool3)
        if first.shape[-2:] != (height, width):
            first = F.interpolate(first, size=(height, width), mode="bilinear", align_corners=False)
        if second.shape[-2:] != (height, width):
            second = F.interpolate(second, size=(height, width), mode="bilinear", align_corners=False)
        first_map = self.low_proj(first).view(batch, -1, n)
        second_map = self.mid_proj(second).view(batch, -1, n)
        out = attention + first_map + second_map
        out = self.channel_proj(out.view(batch, 169, height, width))
        out = self.gamma * out + deep
        pooled = out.mean(dim=(2, 3))
        return self.fc(pooled)


class ModelFactory:
    """Build named variants used by the six experiment cases."""

    @staticmethod
    def build(num_classes: int, config: SqueezeNetConfig) -> nn.Module:
        variant = _normalize_choice(
            config.variant,
            ("baseline", "fusion_only", "msfunet", "msanet35", "msanet53"),
            "msfunet",
        )
        if variant == "baseline":
            return SqueezeNetBaseline(num_classes, pool_type=config.pool_type, pretrained=config.pretrained)
        if variant == "msanet35":
            return MSAAddition(num_classes, order="pool3_pool5", pretrained=config.pretrained)
        if variant == "msanet53":
            return MSAAddition(num_classes, order="pool5_pool3", pretrained=config.pretrained)
        msfu_config = config.msfu
        if variant == "fusion_only":
            msfu_config = replace(msfu_config, mode="fusion_only")
        else:
            msfu_config = replace(msfu_config, mode="full")
        return SqueezeNetMSFUNet(num_classes, replace(config, variant=variant, msfu=msfu_config))


def build_model(
    num_classes: int,
    exp_variant: Optional[str] = None,
    variant: Optional[str] = None,
    msfu_mode: str = "full",
    pool_type: str = "guided",
    pretrained: bool = True,
    tap_idx_x: int = SQUEEZENET_TAPS["fire9"],
    tap_idx_z: int = SQUEEZENET_TAPS["pool3"],
    tap_idx_y: int = SQUEEZENET_TAPS["pool5"],
    topk_ratio: float = 0.0,
    style_p: float = 0.0,
    style_alpha: float = 0.0,
    use_style_norm: bool = True,
    msfu_bg_scale: float = 0.0,
    msfu_init_gamma: float = 0.05,
    softk_tau: float = 0.5,
    softk_alpha: float = 1.0,
    use_coord_score: bool = True,
    use_local_refine: bool = True,
    qk_dim: int = 128,
    bqs_mode: str = "dual",
    lambda_mode: str = "learnable",
    lambda_fixed: float = 0.5,
    use_se_in_fusion: bool = True,
) -> nn.Module:
    """Compatibility entry point used by trainer scripts and smoke tests."""

    chosen_variant = variant or exp_variant or "msfunet"
    chosen_variant = "msfunet" if chosen_variant == "msfu" else chosen_variant
    if chosen_variant == "baseline":
        pool_type = "gap" if pool_type == "guided" else pool_type
    if msfu_mode == "now":
        chosen_variant = "fusion_only"

    msfu = MSFUConfig(
        mode="fusion_only" if msfu_mode in ("now", "fusion_only") else "full",
        score_mode=bqs_mode,
        lambda_mode=lambda_mode,
        lambda_fixed=lambda_fixed,
        topk_ratio=topk_ratio,
        background_scale=msfu_bg_scale,
        init_gamma=msfu_init_gamma,
        use_style_norm=use_style_norm,
        style_p=style_p,
        style_alpha=style_alpha,
        softk_tau=softk_tau,
        softk_alpha=softk_alpha,
        use_coord_score=use_coord_score,
        use_local_refine=use_local_refine,
        qk_dim=qk_dim,
        use_se_in_fusion=use_se_in_fusion,
    )
    config = SqueezeNetConfig(
        variant=chosen_variant,
        pool_type=pool_type,
        pretrained=pretrained,
        tap_deep=tap_idx_x,
        tap_shallow=tap_idx_z,
        tap_mid=tap_idx_y,
        msfu=msfu,
    )
    return ModelFactory.build(num_classes, config)


class SqueezeNetSimple(SqueezeNetBaseline):
    """Backward-compatible name for the SqueezeNet baseline."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes=num_classes, pool_type="gap", pretrained=pretrained)


class SqueezeNetWithMSFU(SqueezeNetMSFUNet):
    """Backward-compatible wrapper for older trainer arguments."""

    def __init__(
        self,
        num_classes: int,
        tap_idx_z: int = SQUEEZENET_TAPS["pool3"],
        tap_idx_y: int = SQUEEZENET_TAPS["pool5"],
        tap_idx_x: int = SQUEEZENET_TAPS["fire9"],
        pool_type: str = "guided",
        pretrained: bool = True,
        **kwargs,
    ):
        config = SqueezeNetConfig(
            variant="msfunet",
            pool_type=pool_type,
            pretrained=pretrained,
            tap_deep=tap_idx_x,
            tap_shallow=tap_idx_z,
            tap_mid=tap_idx_y,
            msfu=MSFUConfig(
                mode=kwargs.get("msfu_mode", "full"),
                score_mode=kwargs.get("bqs_mode", "dual"),
                lambda_mode=kwargs.get("lambda_mode", "learnable"),
                lambda_fixed=kwargs.get("lambda_fixed", 0.5),
                topk_ratio=kwargs.get("topk_ratio", 0.0),
                background_scale=kwargs.get("msfu_bg_scale", kwargs.get("background_scale", 0.0)),
                init_gamma=kwargs.get("msfu_init_gamma", kwargs.get("init_gamma", 0.05)),
                use_style_norm=kwargs.get("use_style_norm", True),
                style_p=kwargs.get("style_p", 0.0),
                style_alpha=kwargs.get("style_alpha", 0.0),
                softk_tau=kwargs.get("softk_tau", 0.5),
                softk_alpha=kwargs.get("softk_alpha", 1.0),
                use_coord_score=kwargs.get("use_coord_score", True),
                use_local_refine=kwargs.get("use_local_refine", True),
                qk_dim=kwargs.get("qk_dim", 128),
                use_se_in_fusion=kwargs.get("use_se_in_fusion", True),
            ),
        )
        super().__init__(num_classes=num_classes, config=config)


class SqueezeNetWithAttention(MSAAddition):
    """Backward-compatible name for the MSANet comparison model."""

    def __init__(self, num_classes: int, order: str = "pool3_pool5", pretrained: bool = True):
        super().__init__(num_classes=num_classes, order=order, pretrained=pretrained)
