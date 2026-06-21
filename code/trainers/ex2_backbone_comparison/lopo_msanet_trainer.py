# Organized filename: lopo_msanet_trainer.py
# Purpose: LOPO trainer for MSANet in the backbone comparison.
# Original source: train_lopo_msanet.py

import pathlib as _pathlib
import sys as _sys
_PROJECT_ROOT = _pathlib.Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

# -*- coding: utf-8 -*-
"""
LOPO for MSANet (SqueezeNet) — 合併 val_tune/val_full 為單一 val_loader 版 (含資料診斷 data_diag)
重點特色：
- ROI (per-pig JSON / center-fallback / jitter)
- Letterbox（等比 + 補邊，無失真）for train/val/test
- A/B/C/color_robust 增強、MixUp/CutMix/LabelSmoothing、Weighted Sampler
- Temp scaling、(可選) TTA(hflip)
- EarlyStopping/選模一律使用「val 的 Macro@thr」（每個 epoch 掃門檻）
- 門檻：在 val 上掃門檻，夾到 [thr_min, thr_max] 後帶到 test（並與 EM/翻轉一致）
- (可選) Prior correction via EM（Saerens et al., 2002）
  * 以 --em_scope 控制：off / val / test / both（預設 test）
  * --em_alpha_val / --em_alpha_test control EM correction strength (0–1)
- （方向保護1）val AUC < 0.5 →（已停用）
- （方向保護2）Test 端無監督自動翻轉（mean(p) 靠近 0.5 準則），門檻同步 1-t（預設關）
- （方向保護3）Test AUC < 0.5 →（已停用）
- (可選) SAM optimizer
- --data_diag writes per-fold prior, appearance, near-duplicate, and AUC diagnostics
- --enable_thr_quantile_map maps the validation threshold quantile to the test distribution
- --enable_temp_scaling selects save-best temperature scaling when temperature mode is off
- （健檢強化）DataLoader 產生 0 個 batch 立即拋錯；train loop 0 batch 警告
- （洩漏防護）嚴格斷言：test 豬絕不出現在 train/val；各 split 兩兩互斥

Optional D1/D2/D3 diagnostics:
- --align_color_to_train / --align_sample：推論端色彩對齊（val/test 同步使用 eval 統計）
- --adabn：Test 前執行 AdaBN（僅刷新 BN running mean/var，不改權重）

Confusion-matrix outputs:
- 每豬（該 fold 的 test set）輸出：raw 與 row-normalized 混淆矩陣 CSV
- 全豬（所有 folds 合併）輸出：raw 與 row-normalized 混淆矩陣 CSV
"""
import os, json, time, argparse, random, csv
from collections import Counter, defaultdict
from typing import Tuple, Optional, List, Set

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel

from PIL import Image
from torchvision import transforms, datasets
import torch.nn.functional as F
from transformers.optimization import get_cosine_schedule_with_warmup

# Optional diagnostic dependencies.
_HAS_SK = False
_HAS_IH = False
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _HAS_SK = True
except Exception:
    pass

try:
    import imagehash
    _HAS_IH = True
except Exception:
    pass

# Confusion-matrix support from scikit-learn.
_CM_AVAILABLE = False
if _HAS_SK:
    try:
        from sklearn.metrics import confusion_matrix
        _CM_AVAILABLE = True
    except Exception:
        _CM_AVAILABLE = False

# Comparison models from the organized, self-contained model package.
from MSFUNet_experiments.code.models._squeezenet_core import (
    SqueezeNetWithAttention,
    SqueezeNetWithMSFU,
)

class MSA_Addition_Pool35(SqueezeNetWithAttention):
    def __init__(self, num_classes):
        super().__init__(num_classes=num_classes, order="pool3_pool5")

class MSA_Addition_Pool53(SqueezeNetWithAttention):
    def __init__(self, num_classes):
        super().__init__(num_classes=num_classes, order="pool5_pool3")

# ===== Focal Loss =====
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, logits, target):
        if target.dtype.is_floating_point():
            log_prob = F.log_softmax(logits, dim=1)
            prob = torch.softmax(logits, dim=1)
            pt = (prob * target).sum(dim=1).clamp_min(1e-8)
            loss = -((1 - pt).pow(self.gamma) * (target * log_prob).sum(dim=1))
        else:
            ce = F.cross_entropy(logits, target, reduction="none", weight=None)
            pt = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
            loss = (1 - pt).pow(self.gamma) * ce
            if self.alpha is not None:
                if isinstance(self.alpha, (list, tuple)):
                    a = torch.tensor(self.alpha, dtype=logits.dtype, device=logits.device)
                    alpha_t = a.gather(0, target)
                elif torch.is_tensor(self.alpha):
                    alpha_t = self.alpha.gather(0, target)
                else:
                    alpha_t = torch.tensor(float(self.alpha), dtype=logits.dtype, device=logits.device)
                loss = alpha_t * loss
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss

# ===== Prior shift correction via EM =====
@torch.no_grad()
def prior_correction_em(model, loader, device, max_iter=20, eps=1e-5):
    model.eval()
    ps = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        z = model(x)
        p = z.softmax(1)
        ps.append(p.detach())
    P = torch.cat(ps, dim=0)  # [N, C]
    pi = (P.mean(dim=0) / P.mean(dim=0).sum()).clamp_min(1e-6)
    for _ in range(max_iter):
        W = (P * pi)
        W = W / (W.sum(dim=1, keepdim=True) + 1e-12)
        pi_new = (W.mean(dim=0) / W.mean(dim=0).sum()).clamp_min(1e-6)
        if (pi_new - pi).abs().max().item() < eps:
            pi = pi_new; break
        pi = pi_new
    bias = torch.log(pi + 1e-12)
    bias = bias - bias.mean()
    return bias.to(device)  # [C]

# ==================== 固定隨機種子 ====================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(os.environ.get("MSFUNET_DETERMINISTIC", "0") == "1", warn_only=True)

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id); random.seed(seed + worker_id)

# ==================== 基本工具 & ROI ====================
def pig_id_of(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3:
        return parts[-2]
    return parts[-1] if parts else "unknown_pig"

def load_roi_cfg(path: Optional[str]):
    if not path: return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)  # {pig_id: [x0,y0,x1,y1]} (0~1)
    ok = {}
    for k, v in cfg.items():
        if isinstance(v, (list, tuple)) and len(v) == 4:
            ok[k] = [float(max(0.0, min(1.0, x))) for x in v]
    return ok

def crop_by_roi(img: Image.Image, roi: Tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = roi
    left   = max(0, min(int(x0 * w), w - 1))
    top    = max(0, min(int(y0 * h), h - 1))
    right  = max(left + 1, min(int(x1 * w), w))
    bottom = max(top + 1,  min(int(y1 * h), h))
    return img.crop((left, top, right, bottom))

def center_roi_box(img: Image.Image, keep_ratio: float):
    keep_ratio = max(0.1, min(keep_ratio, 1.0))
    w, h = img.size
    nw, nh = int(w * keep_ratio), int(h * keep_ratio)
    left = (w - nw) // 2; top  = (h - nh) // 2
    return left / w, top / h, (left + nw) / w, (top + nh) / h

def _clamp01(x): return max(0.0, min(1.0, float(x)))

def jitter_roi_box(roi, j: float):
    if not j or j <= 0: return roi
    x0, y0, x1, y1 = map(float, roi)
    w, h = max(1e-6, x1-x0), max(1e-6, y1-y0)
    tx = random.uniform(-j, j) * w
    ty = random.uniform(-j, j) * h
    sx = random.uniform(1.0 - j, 1.0 + j)
    sy = random.uniform(1.0 - j, 1.0 + j)
    cx = (x0+x1)*0.5 + tx; cy = (y0+y1)*0.5 + ty
    nw, nh = w*sx, h*sy
    nx0, ny0 = _clamp01(cx - nw*0.5), _clamp01(cy - nh*0.5)
    nx1, ny1 = _clamp01(cx + nw*0.5), _clamp01(cy + nh*0.5)
    if nx1-nx0 < 0.02: nx1 = _clamp01(nx0 + 0.02)
    if ny1-ny0 < 0.02: ny1 = _clamp01(ny0 + 0.02)
    return (nx0, ny0, nx1, ny1)

# ==================== Letterbox ====================
class Letterbox:
    def __init__(self, out_size: int, pad_color=(114,114,114), scale_jitter=0.0):
        self.out = out_size
        self.pad_color = pad_color
        self.scale_jitter = float(scale_jitter)
    def __call__(self, img: Image.Image):
        w, h = img.size
        s = min(self.out / w, self.out / h)
        if self.scale_jitter > 0:
            lo = max(0.5, 1.0 - self.scale_jitter)
            s = s * random.uniform(lo, 1.0)
            s = min(s, 1.0)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.out, self.out), self.pad_color)
        left = (self.out - nw) // 2; top  = (self.out - nh) // 2
        canvas.paste(img, (left, top))
        return canvas

# ==================== 自訂增強 ====================
class ChannelDrop:
    def __init__(self, p=0.12): self.p = p
    def __call__(self, img: Image.Image):
        if random.random() < self.p:
            r, g, b = img.split()
            zero = Image.new("L", img.size, 0)
            choice = random.choice([0,1,2])
            ch = [r,g,b]; ch[choice] = zero
            img = Image.merge("RGB", ch)
        return img

# ==================== D1: 色彩對齊用 Transform ====================
class ColorAlignToTrain:
    """
    Insert BEFORE Normalize(mean,std).
    x is Tensor[C,H,W] in [0,1].
    """
    def __init__(self, mu_tr: torch.Tensor, std_tr: torch.Tensor,
                 mu_eval: torch.Tensor, std_eval: torch.Tensor):
        self.mu_tr   = mu_tr.view(-1,1,1)
        self.std_tr  = std_tr.view(-1,1,1).clamp_min(1e-6)
        self.mu_eval = mu_eval.view(-1,1,1)
        self.std_eval= std_eval.view(-1,1,1).clamp_min(1e-6)
    def __call__(self, x: torch.Tensor):
        return (x - self.mu_eval)/self.std_eval * self.std_tr + self.mu_tr

# ==================== Transforms ====================
def build_transforms_train(aug_preset: str, img_size: int):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    aug_list = [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.RandomAffine(
            degrees=10, translate=(0.05,0.05), scale=(0.95,1.05), shear=0.0
        )], p=0.5),
        transforms.RandomApply([transforms.RandomAutocontrast()], p=0.15),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    ]
    if aug_preset in ("color_robust","pig_color"):
        aug_list = [ChannelDrop(0.12)] + aug_list
    if aug_preset in ("color_robust","pig_color","C","heavy","default"):
        aug_list.insert(1, transforms.RandomVerticalFlip(0.3))
    aug_list.insert(2, transforms.RandomApply([transforms.ColorJitter(
        brightness=0.3, contrast=0.3,
        saturation=(0.0, 0.6 if aug_preset in ("color_robust","pig_color") else 0.8),
        hue=0.02
    )], p=0.7))
    aug_list.append(transforms.RandomGrayscale(p=0.25 if aug_preset in ("color_robust","pig_color") else 0.15))
    tf_train = transforms.Compose([
        *aug_list,
        Letterbox(img_size, pad_color=(114,114,114), scale_jitter=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.35),
    ])
    return tf_train

def build_transforms_eval(img_size: int,
                          aligner: Optional[ColorAlignToTrain]=None):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    ops = [
        Letterbox(img_size, pad_color=(114,114,114), scale_jitter=0.0),
        transforms.ToTensor(),
    ]
    if aligner is not None:
        ops.append(aligner)
    ops.append(transforms.Normalize(mean, std))
    return transforms.Compose(ops)

# ==================== Dataset（先 ROI 再 transform） ====================
class PigImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, roi_cfg=None,
                 default_center=None, roi_jitter: float = 0.0):
        super().__init__(root=root, transform=transform)
        self.roi_cfg = roi_cfg or {}
        self.default_center = default_center
        self.roi_jitter = float(roi_jitter)
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path).convert("RGB")
        pid = pig_id_of(path)
        roi = None
        if pid in self.roi_cfg:
            roi = self.roi_cfg[pid]
        elif self.default_center is not None:
            roi = center_roi_box(img, self.default_center)
        if roi is not None:
            if self.roi_jitter > 0.0:
                roi = jitter_roi_box(roi, self.roi_jitter)
            img = crop_by_roi(img, roi)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

# ==================== Early Stopping ====================
class EarlyStopper:
    def __init__(self, patience=7, min_delta=0.0, mode="max"):
        self.patience = patience; self.min_delta = min_delta; self.mode = mode
        self.best = -float("inf"); self.count = 0; self.best_epoch = 0
    def step(self, value, epoch):
        improved = (value > (self.best + self.min_delta))
        if improved: self.best = value; self.count = 0; self.best_epoch = epoch
        else: self.count += 1
        return improved, (self.count >= self.patience)

# ==================== 模型建立（原始 MSA） ====================
def build_msa_model(num_classes: int, model_type: str):
    if model_type == "MSA_Addition_Pool35":
        model = MSA_Addition_Pool35(num_classes)
    elif model_type == "MSA_Addition_Pool53":
        model = MSA_Addition_Pool53(num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model

# ==================== Mixup / CutMix / Soft CE ====================
def one_hot(labels, num_classes, smoothing=0.0):
    with torch.no_grad():
        y = torch.empty(size=(labels.size(0), num_classes), device=labels.device)
        if num_classes > 1:
            y.fill_(smoothing / (num_classes - 1))
            y = y.scatter(1, labels.unsqueeze(1), 1.0 - smoothing)
        else:
            y.fill_(1.0)
    return y

def soft_cross_entropy(logits, target_prob):
    log_prob = F.log_softmax(logits, dim=1)
    return -(target_prob * log_prob).sum(dim=1).mean()

def apply_mixup_cutmix(x, y, num_classes, mixup_alpha=0.1, cutmix_alpha=0.0):
    lam = 1.0
    y1 = one_hot(y, num_classes, smoothing=0.0)
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return x, y1, lam
    use_cutmix = (random.random() < 0.5) and (cutmix_alpha > 0)
    if use_cutmix:
        beta = torch.distributions.Beta(cutmix_alpha, cutmix_alpha)
        lam = float(beta.sample())
        B, C, H, W = x.size()
        index = torch.randperm(B, device=x.device)
        cx, cy = random.randint(0, W - 1), random.randint(0, H - 1)
        w = int(W * (1 - lam) ** 0.5); h = int(H * (1 - lam) ** 0.5)
        x0, y0 = max(0, cx - w // 2), max(0, cy - h // 2)
        x1, y1b = min(W, cx + w // 2), min(H, cy + h // 2)
        x[:, :, y0:y1b, x0:x1] = x[index, :, y0:y1b, x0:x1]
        lam = 1 - ((x1 - x0) * (y1b - y0) / (W * H))
        y2 = one_hot(y[index], num_classes, smoothing=0.0)
        y_soft = lam * y1 + (1 - lam) * y2
        return x, y_soft, lam
    else:
        beta = torch.distributions.Beta(mixup_alpha, mixup_alpha)
        lam = float(beta.sample())
        index = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[index, :]
        y2 = one_hot(y[index], num_classes, smoothing=0.0)
        y_soft = lam * y1 + (1 - lam) * y2
        return x, y_soft, lam

# ==================== Temperature scaling ====================
class _Temp(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))
    def forward(self, z): return z / self.log_t.exp()

@torch.no_grad()
def _gather_logits_targets(model, loader, device):
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        z = model(x); zs.append(z); ys.append(y)
    return torch.cat(zs), torch.cat(ys)

def fit_temperature(model, val_loader, device):
    z, y = _gather_logits_targets(model, val_loader, device)
    if z.numel() == 0: return None
    t = _Temp().to(device)
    nll = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([t.log_t], lr=0.1, max_iter=50)
    def _closure():
        opt.zero_grad(set_to_none=True)
        loss = nll(t(z), y); loss.backward(); return loss
    try:
        opt.step(_closure)
    except Exception:
        opt = torch.optim.Adam([t.log_t], lr=1e-2)
        for _ in range(200):
            opt.zero_grad(set_to_none=True)
            loss = nll(t(z), y); loss.backward(); opt.step()
    return t

# ==================== Threshold quantile mapping ====================
def quantile_map_threshold(t_star_val: float, val_probs: np.ndarray, test_probs: np.ndarray):
    q = float((val_probs <= t_star_val).mean())
    q = min(max(q, 1e-4), 1 - 1e-4)
    t_test = float(np.quantile(test_probs, q))
    return t_test, q

# ==================== WeightedRandomSampler ====================
def build_weighted_sampler(dataset, indices, alpha=1.0, beta=0.7, gamma=0.5):
    labels, pigs, pigcls = [], [], []
    for i in indices:
        path, y = dataset.samples[i]; pid = pig_id_of(path)
        labels.append(y); pigs.append(pid); pigcls.append((pid, y))
    cls_counter = Counter(labels); pig_counter = Counter(pigs); pigcls_counter = Counter(pigcls)
    weights = []
    for i in indices:
        path, y = dataset.samples[i]; pid = pig_id_of(path)
        wc = 1.0 / max(1, cls_counter[y])
        wp = 1.0 / max(1, pig_counter[pid])
        wg = 1.0 / max(1, pigcls_counter[(pid, y)])
        weights.append((wc**alpha) * (wp**beta) * (wg**gamma))
    w = torch.as_tensor(weights, dtype=torch.double)
    if not torch.isfinite(w).all() or float(w.sum()) == 0.0:
        w = torch.ones_like(w, dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(indices), replacement=True)

# ==================== SAM Optimizer（安全版） ====================
class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, "Invalid rho"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.rho = float(rho)
        self.adaptive = bool(adaptive)
    def _grad_norm(self) -> float:
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                g = (torch.abs(p) * p.grad) if self.adaptive else p.grad
                norms.append(torch.norm(g, p=2))
        if not norms: return 0.0
        stacked = torch.stack(norms)
        return float(torch.norm(stacked, p=2).item())
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        scale = self._grad_norm() + 1e-12
        for group in self.param_groups:
            rho_over_scale = float(group["rho"]) / scale
            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) * p.grad) if self.adaptive else p.grad
                e_w = e_w * rho_over_scale
                self.state[p]["e_w"] = e_w
                p.add_(e_w)
        if zero_grad: self.zero_grad(set_to_none=True)
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                e_w = self.state[p].get("e_w", None)
                if e_w is not None: p.sub_(e_w)
        self.base_optimizer.step()
        if zero_grad: self.zero_grad(set_to_none=True)
    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError("Use first_step and second_step for SAM.")

# ==================== Param groups ====================
def build_param_groups_msanet(model, backbone_lr, head_lr, msfu_lr, weight_decay):
    pg_backbone, pg_head, pg_msfu = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        lname = name.lower()
        if any(k in lname for k in ["msfu", "style_norm", "stylenorm", "style", "affine", "lambda_raw", "q_deep", "k_shal", "q_shal", "k_deep"]):
            pg_msfu.append(p)
        elif any(k in lname for k in ["classifier", "fc", "head"]):
            pg_head.append(p)
        else:
            pg_backbone.append(p)
    param_groups = []
    if pg_backbone: param_groups.append({"params": pg_backbone, "lr": backbone_lr, "weight_decay": weight_decay})
    if pg_head:     param_groups.append({"params": pg_head,     "lr": head_lr,     "weight_decay": weight_decay})
    if pg_msfu:     param_groups.append({"params": pg_msfu,     "lr": msfu_lr,     "weight_decay": weight_decay})
    total = sum(p.numel() for g in param_groups for p in g["params"])
    for i, g in enumerate(param_groups):
        n_params = sum(p.numel() for p in g["params"])
        print(f"[ParamGroup {i}] lr={g['lr']:.2e}, wd={g['weight_decay']}, n_params={n_params}")
    print(f"[Param check] total trainable params = {total}")
    return param_groups

# ==================== 評估 & 門檻工具 ====================
@torch.no_grad()
def evaluate(model, loader, criterion_hard, device, class_names,
             tta_hflip=False, temp_model: Optional[_Temp]=None, threshold: Optional[float]=None):
    model.eval()
    net = model.module if isinstance(model, AveragedModel) else model
    loss_sum, correct, total = 0.0, 0, 0
    per_cls_c = [0]*len(class_names); per_cls_t = [0]*len(class_names)
    n_batches = 0
    for x, y in loader:
        n_batches += 1
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = net(x)
        if tta_hflip:
            x2 = torch.flip(x, dims=[-1]); logits = 0.5 * (logits + net(x2))
        if temp_model is not None:
            logits = temp_model(logits)
        loss = criterion_hard(logits, y)
        loss_sum += float(loss.item())
        if threshold is not None and logits.shape[1] == 2:
            prob1 = logits.softmax(1)[:,1]; pred = (prob1 >= threshold).long()
        else:
            pred = logits.argmax(1)
        correct += (pred == y).sum().item(); total += y.size(0)
        for yy, pp in zip(y, pred):
            per_cls_t[yy.item()] += 1
            if yy == pp: per_cls_c[yy.item()] += 1
    if n_batches == 0:
        print("[DBG] WARNING: evaluate loader yielded 0 batches!")
    acc = 100.0 * correct / max(1, total)
    cls_acc = {class_names[i]: (100.0*per_cls_c[i]/per_cls_t[i] if per_cls_t[i] else 0.0)
               for i in range(len(class_names))}
    macro = sum(cls_acc.values()) / max(1, len(cls_acc))
    return (loss_sum / max(1, n_batches)), acc, cls_acc, macro

@torch.no_grad()
def _collect_probs_binary(model, loader, device, temp_model=None):
    model.eval()
    net = model.module if isinstance(model, AveragedModel) else model
    ps, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = net(x)
        if temp_model is not None: logits = temp_model(logits)
        p1 = logits.softmax(1)[:,1].detach().cpu().numpy()
        ps.append(p1); ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)

def tune_threshold_macro_acc(p1: np.ndarray, y: np.ndarray, lo=0.1, hi=0.9, n=41):
    ts = np.linspace(lo, hi, n)
    best_t, best_macro = 0.5, -1.0
    for t in ts:
        pred = (p1 >= t).astype(np.int32)
        a0 = (pred[y==0] == 0).mean() if (y==0).any() else 0.0
        a1 = (pred[y==1] == 1).mean() if (y==1).any() else 0.0
        macro = 0.5*(a0+a1)
        if macro > best_macro: best_macro, best_t = macro, t
    return float(best_t), float(best_macro*100.0)

# ==================== Dataset diagnostics ====================
def _pig_class_table(dataset, indices):
    tb = defaultdict(Counter)
    for i in indices:
        path, y = dataset.samples[i]
        pid = pig_id_of(path)
        tb[pid][y] += 1
    return tb

def _fmt_prior(tb, class_names):
    lines = []
    for pid in sorted(tb.keys()):
        tot = sum(tb[pid].values())
        parts = [f"{class_names[c]}:{tb[pid][c]}/{tot}={tb[pid][c]/tot:.2f}" for c in range(len(class_names))]
        lines.append(f"{pid} | " + " , ".join(parts))
    return "\n".join(lines)

def _safe_auc(p, y):
    if not _HAS_SK: return float('nan')
    try: return roc_auc_score(y, p)
    except Exception: return float('nan')

def _avg_precision(p, y):
    if not _HAS_SK: return float('nan')
    try: return average_precision_score(y, p)
    except Exception: return float('nan')

def _binary_report(y_true, y_pred, y_prob):
    """Return the standard binary metric set used by the backbone comparison."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    safe = lambda numerator, denominator: float(numerator / denominator) if denominator else 0.0
    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)
    return {
        "acc": 100.0 * safe(tp + tn, len(y_true)),
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * safe(2.0 * precision * recall, precision + recall),
        "specificity": 100.0 * safe(tn, tn + fp),
        "auc": _safe_auc(np.asarray(y_prob), y_true),
    }

@torch.no_grad()
def _measure_latency(model, device, img_size, warmup=30, iters=200):
    """Measure batch-one latency and throughput on the active device."""
    model.eval()
    sample = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(max(0, warmup)):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(max(1, iters)):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0 / max(1, iters)
    return latency_ms, 1000.0 / max(latency_ms, 1e-12)

def _paths(dataset, indices):
    return [dataset.samples[i][0] for i in indices]

def _phash(path, size=8):
    if not _HAS_IH: return None
    try: return imagehash.phash(Image.open(path).convert("RGB"), hash_size=size)
    except Exception: return None

def _near_dup_count(train_paths, test_paths, max_pairs=5000):
    if not _HAS_IH: return None
    rnd = random.Random(1234)
    tr_sub = rnd.sample(train_paths, min(len(train_paths), max_pairs))
    te_sub = rnd.sample(test_paths, min(len(test_paths), max_pairs//2))
    train_hash = {}
    for p in tr_sub:
        h = _phash(p)
        if h is not None:
            train_hash.setdefault(str(h), 0)
            train_hash[str(h)] += 1
    hit = 0
    for p in te_sub:
        h = _phash(p)
        if h is None: continue
        hs = str(h)
        if hs in train_hash:
            hit += 1
    return hit

def _img_stats(paths, take=500):
    vals = []
    rnd = random.Random(2024)
    samp = rnd.sample(paths, min(take, len(paths)))
    for p in samp:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        arr = np.asarray(im)/255.0
        gray = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
        vals.append([gray.mean(), gray.std(), arr[...,0].mean(), arr[...,1].mean(), arr[...,2].mean()])
    if not vals: return None
    m = np.mean(vals, 0); s = np.std(vals, 0)
    return m, s

def _roi_cover_ratio(paths, roi_cfg, default_center, take=200):
    area = []
    rnd = random.Random(99)
    samp = rnd.sample(paths, min(take, len(paths)))
    for p in samp:
        try:
            im  = Image.open(p).convert("RGB")
        except Exception:
            continue
        pid = pig_id_of(p)
        if pid in roi_cfg: x0,y0,x1,y1 = roi_cfg[pid]
        elif default_center is not None: x0,y0,x1,y1 = center_roi_box(im, default_center)
        else: continue
        area.append( (x1-x0) * (y1-y0) )
    if not area: return None
    return float(np.mean(area)), float(np.std(area))

# ==================== 方向翻轉保護 ====================
class FlipLogits(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        return -self.base(x)

@torch.no_grad()
def maybe_flip_logits_on_low_auc(model_or_wrap, val_loader, device, temp_model=None, auc_lo=0.5):
    return model_or_wrap, False

@torch.no_grad()
def auto_flip_by_test_mean(model_or_wrap, test_loader, device, temp_model=None, target_mean=0.5, margin=0.05):
    if margin is None or margin <= 0:
        return model_or_wrap, False, float("nan"), float("nan")
    net = model_or_wrap
    ps = []
    for x, _ in test_loader:
        x = x.to(device, non_blocking=True)
        z = net(x)
        if temp_model is not None:
            z = temp_model(z)
        p1 = z.softmax(1)[:, 1].detach().cpu().numpy()
        ps.append(p1)
    p = np.concatenate(ps)
    m0 = float(p.mean())
    m1 = float((1.0 - p).mean())
    d0 = abs(m0 - target_mean)
    d1 = abs(m1 - target_mean)
    if (d0 - d1) > margin:
        print(f"[AutoFlip-Test] mean(p)={m0:.3f} → mean(1-p)={m1:.3f} 更接近 {target_mean}（Δ={d0-d1:.3f}）→ Test 使用翻轉。")
        return FlipLogits(model_or_wrap), True, m0, m1
    return model_or_wrap, False, m0, m1

@torch.no_grad()
def maybe_flip_by_test_auc(model_or_wrap, test_loader, device, temp_model=None, tuned_thr=None):
    if not _HAS_SK:
        return model_or_wrap, tuned_thr, False, float('nan')
    net = model_or_wrap
    ps, ys = [], []
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        z = net(x)
        if temp_model is not None:
            z = temp_model(z)
        p1 = z.softmax(1)[:, 1].detach().cpu().numpy()
        ps.append(p1); ys.append(y.numpy())
    p = np.concatenate(ps); y = np.concatenate(ys)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float('nan')
    return model_or_wrap, tuned_thr, False, auc

# ==================== Train / Eval（單輪） ====================
def train_epoch(model, loader, num_classes, criterion_hard, device,
                mixup_alpha, cutmix_alpha, label_smoothing,
                optimizer, scheduler=None, ema: Optional[AveragedModel]=None,
                clip_grad: float=0.0, scheduler_is_plateau: bool=False,
                conf_penalty: float=0.0, logit_l2: float=0.0, lambda_reg: float=0.0,
                use_sam: bool=False):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    def forward_and_loss(batch_x, batch_y):
        x, y = batch_x, batch_y
        x_mix, y_soft, _ = apply_mixup_cutmix(x, y, num_classes, mixup_alpha, cutmix_alpha)
        logits = model(x_mix)
        if mixup_alpha > 0 or cutmix_alpha > 0 or label_smoothing > 0:
            if label_smoothing > 0 and (mixup_alpha <= 0 and cutmix_alpha <= 0):
                y_soft = one_hot(y, num_classes, smoothing=label_smoothing)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            loss = criterion_hard(logits, y)
        if conf_penalty and conf_penalty > 0.0:
            log_probs = torch.log_softmax(logits.float(), dim=1)
            probs     = torch.softmax(logits.float(), dim=1)
            cp = (probs * log_probs).sum(dim=1).mean()
            loss = loss + conf_penalty * cp
        if logit_l2 and logit_l2 > 0.0:
            loss = loss + logit_l2 * logits.float().pow(2).mean()
        if lambda_reg and lambda_reg > 0.0 and hasattr(model, "msfu"):
            try:
                lam = torch.sigmoid(model.msfu.lambda_raw)
                loss = loss + lambda_reg * (lam - 0.5).pow(2)
            except Exception:
                pass
        return loss, logits

    n_batches = 0
    for x, y in loader:
        n_batches += 1
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_sam:
            loss1, logits1 = forward_and_loss(x, y)
            loss1.backward()
            if clip_grad and clip_grad > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.first_step(zero_grad=True)

            loss2, logits2 = forward_and_loss(x, y)
            loss2.backward()
            if clip_grad and clip_grad > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.second_step(zero_grad=True)

            if scheduler is not None and (not scheduler_is_plateau):
                try:
                    scheduler.step()
                except Exception:
                    pass

            loss = loss2.detach()
            logits = logits2.detach()
        else:
            loss, logits = forward_and_loss(x, y)
            loss.backward()
            if clip_grad and clip_grad > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

            if scheduler is not None and (not scheduler_is_plateau):
                try:
                    scheduler.step()
                except Exception:
                    pass

        if ema is not None:
            ema.update_parameters(model)

        loss_sum += float(loss.item())
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    if n_batches == 0:
        print("[DBG] WARNING: train_loader yielded 0 batches!")
    return loss_sum / max(1, n_batches), 100.0 * correct / max(1, total)

# ==================== 包裝器（共用） ====================
class _WithBiasScaled(nn.Module):
    def __init__(self, m, bias, alpha=1.0):
        super().__init__(); self.m=m; self.bias=bias; self.alpha=float(alpha)
    def forward(self, x):
        z = self.m(x)
        return z + (self.alpha * self.bias if self.bias is not None else 0)

# ==================== Statistical estimation and D1 construction ====================
def _estimate_tensor_stats(image_paths: List[str], roi_cfg, default_center, img_size: int, take: int=1200):
    """
    估計在 Letterbox+ToTensor 後（未 Normalize）的 RGB 每通道 μ/σ。
    僅做輕量取樣加速。
    """
    rnd = random.Random(2025)
    samp = rnd.sample(image_paths, min(take, len(image_paths)))
    lb = Letterbox(img_size, pad_color=(114,114,114), scale_jitter=0.0)
    means = []; sqs = []; n_pix = 0
    for p in samp:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        pid = pig_id_of(p)
        if pid in roi_cfg:
            im = crop_by_roi(im, roi_cfg[pid])
        elif default_center is not None:
            im = crop_by_roi(im, center_roi_box(im, default_center))
        im = lb(im)
        x = transforms.functional.to_tensor(im)  # [C,H,W], [0,1]
        C,H,W = x.shape
        means.append(x.view(C, -1).mean(dim=1))
        sqs.append((x**2).view(C, -1).mean(dim=1))
        n_pix += H*W
    if not means:
        # fallback 到 ImageNet 統計（不致崩）
        mu = torch.tensor([0.485,0.456,0.406])
        sd = torch.tensor([0.229,0.224,0.225])
        return mu, sd
    mu = torch.stack(means,0).mean(0)
    ex2= torch.stack(sqs,0).mean(0)
    var= (ex2 - mu**2).clamp_min(1e-9)
    sd = var.sqrt()
    return mu, sd

# ==================== Leakage-safe split helpers ====================
def _pigs_of_indices(dataset: PigImageFolder, idxs: List[int]) -> Set[str]:
    return set(pig_id_of(dataset.samples[i][0]) for i in idxs)

def _assert_disjoint_sets(a: Set[str], b: Set[str], msg: str):
    inter = a & b
    assert len(inter) == 0, f"[SplitError] {msg}：集合相交 = {sorted(list(inter))}"

def _assert_no_leak(train_idx, val_idx, test_idx, ds_eval_base, test_pig: str):
    pigs_train = _pigs_of_indices(ds_eval_base, train_idx)
    pigs_val   = _pigs_of_indices(ds_eval_base, val_idx)
    pigs_test  = _pigs_of_indices(ds_eval_base, test_idx)
    _assert_disjoint_sets(pigs_train, pigs_val,   "train 與 val 不應有交集")
    _assert_disjoint_sets(pigs_train, pigs_test,  "train 與 test 不應有交集")
    _assert_disjoint_sets(pigs_val,   pigs_test,  "val 與 test 不應有交集")
    assert test_pig in pigs_test, f"[SplitError] test_pig='{test_pig}' 並未出現在 test split"
    assert test_pig not in pigs_train, f"[Leakage] test_pig='{test_pig}' 出現在 train！"
    assert test_pig not in pigs_val,   f"[Leakage] test_pig='{test_pig}' 出現在 val！"

# ==================== D3: AdaBN ====================
@torch.no_grad()
def adabn_calibrate(model: nn.Module, loader: DataLoader):
    was_training = model.training
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for x, _ in loader:
        _ = model(x.to(next(model.parameters()).device, non_blocking=True))
    if not was_training:
        model.eval()

# ==================== Confusion-matrix utilities ====================
def _save_confusion_matrices(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              num_classes: int,
                              out_raw_csv: str,
                              out_row_norm_csv: str,
                              verbose_title: Optional[str] = None):
    """
    存 raw 與 row-normalized 混淆矩陣到 CSV，並在 console 印出。
    """
    labels = list(range(num_classes))
    if _CM_AVAILABLE:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    else:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_row_norm = cm / np.maximum(row_sums, 1)

    os.makedirs(os.path.dirname(out_raw_csv), exist_ok=True)
    np.savetxt(out_raw_csv, cm, fmt="%d", delimiter=",")
    np.savetxt(out_row_norm_csv, cm_row_norm, fmt="%.6f", delimiter=",")

    title = f"===== Confusion Matrix — {verbose_title} =====" if verbose_title else "===== Confusion Matrix ====="
    print("\n" + title)
    print(cm)
    print(f"[SAVE] {out_raw_csv}")
    print("===== Row-normalized =====")
    print(cm_row_norm)
    print(f"[SAVE] {out_row_norm_csv}")

# ==================== 主流程：LOPO ====================
def run_lopo(args):
    if args.img_size != 224:
        print(f"[WARN] 建議 SqueezeNet/MSA 用 --img_size 224；目前 = {args.img_size}。")
    set_seed(args.seed)
    roi_cfg = load_roi_cfg(args.roi_cfg)

    # 用於取樣與 split 的 base（不套 transform，避免重複 decode）
    default_center = None if args.roi_fallback == "none" else args.roi_center
    if args.no_roi: default_center = None
    base_eval = PigImageFolder(root=args.path_d, transform=None, roi_cfg=roi_cfg,
                               default_center=default_center, roi_jitter=0.0)
    class_names = base_eval.classes
    num_classes = len(class_names)

    # 以檔案實際路徑收集 pig id
    all_pigs = sorted(set(pig_id_of(p) for p, _ in base_eval.samples))
    print(f"偵測到豬數：{len(all_pigs)} → {all_pigs}")

    # Train transform 先建好
    tf_train = build_transforms_train(args.aug, args.img_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.path_r, exist_ok=True); os.makedirs(args.path_m, exist_ok=True)

    # 便利開關：若指定 --enable_temp_scaling 且 temp_mode=off，改成 savebest
    if args.enable_temp_scaling and args.temp_mode == "off":
        print("[TempScaling] --enable_temp_scaling 啟用 → temp_mode 改為 'savebest'")
        args.temp_mode = "savebest"

    pigs_to_run = all_pigs if not args.pigs else [p for p in args.pigs.split(",") if p in all_pigs]
    if not pigs_to_run:
        raise ValueError("沒有可用的豬可跑；請檢查 --pigs 或資料夾結構。")

    fold_metrics = []

    # Aggregate predictions across held-out pigs.
    ALL_Y_TRUE: List[int] = []
    ALL_Y_PRED: List[int] = []
    ALL_NUM_CLASSES = None

    for fold_idx, test_pig in enumerate(pigs_to_run, start=1):
        # --- 明確建立三個 index 集合 ---
        test_idx, trainval_idx = [], []
        for i, (path, _) in enumerate(base_eval.samples):
            (test_idx if pig_id_of(path) == test_pig else trainval_idx).append(i)

        # --- 依 pig 區分 val 與 train，確保 pig-level 互斥 ---
        trainval_pigs = sorted(set(pig_id_of(base_eval.samples[i][0]) for i in trainval_idx))
        rng = random.Random(args.seed + fold_idx)
        rng.shuffle(trainval_pigs)

        # val 豬的數量：至少 2；若總豬數不足 3，退而求其次確保不為 0
        n_val = max(2, int(len(trainval_pigs) * args.vr))
        n_val = min(n_val, max(1, len(trainval_pigs) - 1))  # 不能把 train 用光
        val_pigs = set(trainval_pigs[:n_val])
        train_pigs = set(trainval_pigs[n_val:])

        # 依據 pig set 拉回索引
        val_idx   = [i for i in trainval_idx if pig_id_of(base_eval.samples[i][0]) in val_pigs]
        train_idx = [i for i in trainval_idx if pig_id_of(base_eval.samples[i][0]) in train_pigs]

        # ===== 洩漏嚴格斷言（關鍵修正）=====
        _assert_no_leak(train_idx, val_idx, test_idx, base_eval, test_pig)

        # ===== 先驗分佈表 =====
        if args.data_diag:
            tb_train = _pig_class_table(base_eval, train_idx)
            tb_val   = _pig_class_table(base_eval, val_idx)
            tb_test  = _pig_class_table(base_eval, test_idx)
            print("[PRIOR] train per-pig:\n" + _fmt_prior(tb_train, class_names))
            print("[PRIOR] val   per-pig:\n" + _fmt_prior(tb_val, class_names))
            print("[PRIOR] test  per-pig:\n" + _fmt_prior(tb_test, class_names))

        # 構建 Dataset（train/eval 各自一份）
        ds_train_base = PigImageFolder(root=args.path_d, transform=tf_train, roi_cfg=roi_cfg,
                                       default_center=default_center, roi_jitter=args.roi_jitter)

        # ====== D1: 色彩對齊（估計統計 + 建立 eval transform）======
        aligner = None
        if args.align_color_to_train:
            tr_paths = _paths(base_eval, train_idx)
            ev_paths = _paths(base_eval, sorted(set(val_idx) | set(test_idx)))
            mu_tr, sd_tr = _estimate_tensor_stats(tr_paths, roi_cfg, default_center, args.img_size, take=args.align_sample)
            mu_ev, sd_ev = _estimate_tensor_stats(ev_paths, roi_cfg, default_center, args.img_size, take=args.align_sample)
            print(f"[ColorAlign] train μ={mu_tr.tolist()} σ={sd_tr.tolist()}")
            print(f"[ColorAlign] eval  μ={mu_ev.tolist()} σ={sd_ev.tolist()}")
            aligner = ColorAlignToTrain(mu_tr, sd_tr, mu_ev, sd_ev)

        tf_eval = build_transforms_eval(args.img_size, aligner=aligner)
        ds_eval_base  = PigImageFolder(root=args.path_d, transform=tf_eval,  roi_cfg=roi_cfg,
                                       default_center=default_center, roi_jitter=0.0)

        # DataLoaders
        use_simple_shuffle = (args.alpha == 0.0 and args.beta == 0.0 and args.gamma == 0.0)
        if use_simple_shuffle:
            train_loader = DataLoader(Subset(ds_train_base, train_idx), batch_size=args.b, shuffle=True,
                                      num_workers=args.nw, pin_memory=True, drop_last=True,
                                      worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))
        else:
            sampler = build_weighted_sampler(ds_train_base, train_idx,
                                             alpha=args.alpha, beta=args.beta, gamma=args.gamma)
            train_loader = DataLoader(Subset(ds_train_base, train_idx), batch_size=args.b, sampler=sampler,
                                      num_workers=args.nw, pin_memory=True, drop_last=True,
                                      worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))

        if len(train_loader) == 0:
            raise RuntimeError(f"[Data] train_loader has 0 batches! len(train_idx)={len(train_idx)}, "
                               f"batch_size={args.b}, drop_last=True。建議先以 --alpha 0 --beta 0 --gamma 0 排除 sampler 影響，或暫時將 drop_last=False。")

        val_loader  = DataLoader(Subset(ds_eval_base,  val_idx), batch_size=args.b, shuffle=False,
                                 num_workers=args.nw, pin_memory=True, drop_last=False,
                                 worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))
        test_loader = DataLoader(Subset(ds_eval_base,  test_idx), batch_size=args.b, shuffle=False,
                                 num_workers=args.nw, pin_memory=True, drop_last=False,
                                 worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))

        # 類別加權 CE（以 train 分佈估計）
        y_train = [ds_train_base.samples[i][1] for i in train_idx]
        cls_freq = Counter(y_train)
        ce_w = torch.tensor(
            [max(1.0, sum(cls_freq.values())/(len(cls_freq)*max(1, cls_freq.get(c, 0)))) for c in range(num_classes) ],
            dtype=torch.float, device=device
        )

        # 訓練/驗證 criterion
        use_cw = getattr(args, "use_class_weight", False)
        if args.loss_type == "focal":
            criterion_hard = FocalLoss(gamma=args.focal_gamma, alpha=None, reduction="mean")
            criterion_val  = nn.CrossEntropyLoss()
        else:
            criterion_hard = nn.CrossEntropyLoss(weight=ce_w if use_cw else None)
            criterion_val  = nn.CrossEntropyLoss()

        # 建模
        if args.use_msfu:
            print("[Route] Using SqueezeNet + MSFU (dual-score + guided pooling)")
            model = SqueezeNetWithMSFU(
                num_classes=num_classes,
                tap_idx_z=args.tap_idx_z, tap_idx_y=args.tap_idx_y,
                topk_ratio=args.topk_ratio,
                style_p=args.style_p, style_alpha=args.style_alpha,
                use_style_norm=(not args.no_style_norm),
                msfu_bg_scale=args.msfu_bg_scale, msfu_init_gamma=args.msfu_init_gamma,
                softk_tau=args.softk_tau, softk_alpha=args.softk_alpha,
                use_coord_score=(not args.no_coord_score),
                use_local_refine=(not args.no_local_refine),
                pool_type=args.pool_type
            ).to(device)
            model.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
                _ = model(dummy)
            model.train()
        else:
            print(f"[Route] Using original MSANet: {args.model_type}")
            model = build_msa_model(num_classes, args.model_type).to(device)

        # ====== 第二階段：從上一階段權重初始化（可選） ======
        if args.init_from_dir:
            cand_paths = []
            if args.init_pick in ("best", "final"):
                cand_paths.append(os.path.join(args.init_from_dir, f"{args.init_pick}_{test_pig}.pth"))
            else:
                cand_paths.extend([
                    os.path.join(args.init_from_dir, f"best_{test_pig}.pth"),
                    os.path.join(args.init_from_dir, f"final_{test_pig}.pth"),
                ])
            loaded = False
            for wpath in cand_paths:
                if os.path.exists(wpath):
                    try:
                        state = torch.load(wpath, map_location=device, weights_only=True)
                    except TypeError:
                        state = torch.load(wpath, map_location=device)
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    missing, unexpected = model.load_state_dict(state, strict=args.init_strict)
                    print(f"[Init] Loaded '{wpath}' (missing={len(missing)}, unexpected={len(unexpected)})")
                    if len(missing) > 0:
                        print(f"[Init] Missing keys example: {missing[:5]}")
                    if len(unexpected) > 0:
                        print(f"[Init] Unexpected keys example: {unexpected[:5]}")
                    loaded = True
                    break
            if not loaded:
                print(f"[Init][WARN] No init weight found for test pig='{test_pig}' in dir: {args.init_from_dir}")

        # Optimizer & Scheduler（分組 LR）
        param_groups = build_param_groups_msanet(
            model, backbone_lr=args.backbone_lr, head_lr=args.head_lr,
            msfu_lr=(args.msfu_lr if args.use_msfu else args.head_lr), weight_decay=args.wd
        )

        if args.use_sam:
            optimizer = SAM(param_groups, base_optimizer=optim.AdamW, rho=args.sam_rho,
                            adaptive=False, lr=args.head_lr, weight_decay=args.wd)
            scheduler = None
        else:
            optimizer = optim.AdamW(param_groups, weight_decay=args.wd)
            steps_per_epoch = max(1, len(train_loader))
            total_steps = steps_per_epoch * args.e
            if args.lr_sched == "cosine":
                warmup_steps = int(max(0, args.warmup_ratio) * total_steps)
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
                )
            else:
                scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3, cooldown=1)

        ema = None
        if args.ema_decay and args.ema_decay > 0.0:
            ema = AveragedModel(model, avg_fn=lambda avg_p, p, n: args.ema_decay * avg_p + (1.0 - args.ema_decay) * p)

        # Logs/ckpt
        log_path_txt = os.path.join(args.path_r, f"lopo_{test_pig}.txt")
        log_path_csv = os.path.join(args.path_r, f"lopo_{test_pig}.csv")
        best_path    = os.path.join(args.path_m, f"best_{test_pig}.pth")
        best_temp_path = os.path.join(args.path_m, f"best_{test_pig}_temp.pt")
        final_path   = os.path.join(args.path_m, f"final_{test_pig}.pth")

        print(f"\n===== LOPO Fold {fold_idx}/{len(pigs_to_run)} | Test pig = {test_pig} =====")
        print(f"Val pigs: {sorted(list(val_pigs))}")
        print(f"Train pigs: {sorted(list(train_pigs))}")
        print(f"Sizes — train:{len(train_idx)} | val:{len(val_idx)} | test:{len(test_idx)}")
        if args.roi_cfg or (default_center is not None):
            print(f"ROI: cfg={'yes' if args.roi_cfg else 'no'}, fallback={args.roi_fallback}, center={default_center}, jitter={args.roi_jitter}")

        with open(log_path_txt, "w", encoding="utf-8") as f:
            f.write(f"Fold test pig: {test_pig}\nModel route: {'MSFU' if args.use_msfu else args.model_type}\n")
            f.write(f"Train pigs: {sorted(list(train_pigs))}\n")
            f.write(f"Val pigs: {sorted(list(val_pigs))}\n")
            f.write(f"Sizes train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}\n\n")
            if args.data_diag:
                f.write("[PRIOR] train per-pig:\n" + _fmt_prior(tb_train, class_names) + "\n")
                f.write("[PRIOR] val   per-pig:\n" + _fmt_prior(tb_val, class_names) + "\n")
                f.write("[PRIOR] test  per-pig:\n" + _fmt_prior(tb_test, class_names) + "\n\n")

        with open(log_path_csv, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["epoch",
                             "train_loss","train_acc",
                             "val_tune_loss","val_tune_acc","val_tune_macro",
                             "val_full_loss","val_full_acc","val_full_macro",
                             "lr","is_best"])

        stopper = EarlyStopper(patience=args.es_patience, min_delta=0.0, mode="max")
        best_score = -float("inf"); best_epoch = 0; best_temp_state = None

        for ep in range(1, args.e+1):
            # γ 暖身
            if args.use_msfu and args.gamma_warmup_epochs > 0 and hasattr(model, "msfu"):
                warm_E = max(1, args.gamma_warmup_epochs)
                ratio = min(1.0, ep / warm_E)
                with torch.no_grad():
                    model.msfu.gamma.data.fill_(args.msfu_init_gamma * ratio)

            # scheduler 傳遞（cosine: 每 batch；plateau: epoch 後）
            use_plateau = (not args.use_sam) and isinstance(scheduler, ReduceLROnPlateau)
            sched_for_train = None if use_plateau else (scheduler if (not args.use_sam) else None)

            tr_loss, tr_acc = train_epoch(
                model, train_loader, num_classes, criterion_hard, device,
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, label_smoothing=args.ls,
                optimizer=optimizer, scheduler=sched_for_train,
                ema=ema, clip_grad=args.clip_grad, scheduler_is_plateau=use_plateau,
                conf_penalty=args.conf_penalty, logit_l2=args.logit_l2, lambda_reg=args.lambda_reg,
                use_sam=args.use_sam
            )

            eval_model = ema if ema is not None else model

            # 溫度
            temp_for_eval = None
            if args.temp_mode == "epoch":
                temp_for_eval = fit_temperature(eval_model, val_loader, device)
            elif args.temp_mode == "savebest" and best_score > -float("inf"):
                if best_temp_state is not None:
                    temp_for_eval = _Temp().to(device); temp_for_eval.load_state_dict(best_temp_state)

            # ===== 驗證：依 em_scope 決定是否對 val 套 EM（可縮放 α）=====
            eval_for_val = eval_model
            if args.prior_correction == "em" and args.em_scope in ("val","both"):
                em_bias_val = prior_correction_em(eval_model, val_loader, device)
                eval_for_val = _WithBiasScaled(eval_model, em_bias_val, alpha=args.em_alpha_val).to(device)

            # 方向保護（val AUC 翻轉）【封印：no-op】
            guard_model = eval_for_val
            guard_temp  = temp_for_eval
            guard_model, _ = maybe_flip_logits_on_low_auc(
                guard_model, val_loader, device, temp_model=guard_temp, auc_lo=0.5
            )

            # ---- (1) Argmax Macro（觀察用）----
            va_loss, va_acc, _, va_macro_argmax = evaluate(
                guard_model, val_loader, criterion_val, device, class_names,
                tta_hflip=args.tta_hflip, temp_model=guard_temp
            )

            # ---- (2) 調閾 Macro（掃門檻，作為選模/early stop/scheduler 指標）----
            p1_val, y_val = _collect_probs_binary(guard_model, val_loader, device, temp_model=guard_temp)
            t_best, macro_tuned = tune_threshold_macro_acc(
                p1_val, y_val, lo=args.thr_min, hi=args.thr_max, n=args.thr_steps
            )

            # scheduler（若是 plateau，根據 val 指標調整；macro 越大越好 → 用 1 - macro）
            if use_plateau:
                if args.select_by == "macro":
                    scheduler.step(1.0 - macro_tuned/100.0)
                else:
                    scheduler.step(va_loss)

            # 顯示 LR
            lr_now = optimizer.base_optimizer.param_groups[0]["lr"] if args.use_sam else optimizer.param_groups[0]["lr"]
            score_now = (macro_tuned if args.select_by == "macro" else (-va_loss))

            # savebest 的溫度在「改善」時更新
            if (score_now > best_score) and args.temp_mode == "savebest":
                temp_best = fit_temperature(guard_model, val_loader, device)
                if temp_best is not None:
                    best_temp_state = temp_best.state_dict()
                    torch.save(best_temp_state, best_temp_path)

            improved, should_stop = stopper.step(score_now, ep)
            is_best = False
            if improved:
                best_score = score_now; best_epoch = ep; is_best = True
                sd = model.state_dict()
                if isinstance(ema, AveragedModel):
                    sd_ema = ema.module.state_dict()
                    sd.update({k: v for k, v in sd_ema.items() if k in sd and v.shape == sd[k].shape})
                torch.save(sd, best_path)
                if args.temp_mode == "savebest" and best_temp_state is not None:
                    torch.save(best_temp_state, best_temp_path)

            line = (f"[{ep}/{args.e}] Train L:{tr_loss:.6f} A:{tr_acc:.2f}% | "
                    f"Val L:{va_loss:.4f} A:{va_acc:.2f}% "
                    f"Macro(argmax):{va_macro_argmax:.2f}% | Macro@thr:{macro_tuned:.2f}% (t={t_best:.3f}) "
                    f"{'(best)' if is_best else ''} | LR:{lr_now:.6g}")
            print(line)

            with open(log_path_txt, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            with open(log_path_csv, "a", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow([ep,
                                 f"{tr_loss:.6f}", f"{tr_acc:.2f}",
                                 f"{va_loss:.6f}", f"{va_acc:.2f}", f"{macro_tuned:.2f}",
                                 f"{va_loss:.6f}", f"{va_acc:.2f}", f"{macro_tuned:.2f}",
                                 f"{lr_now:.8f}", int(is_best)])

            if should_stop:
                print(f"[EarlyStopping] patience={args.es_patience} 於 epoch {ep} 觸發。")
                break

        sd_final = model.state_dict()
        if isinstance(ema, AveragedModel):
            sd_ema = ema.module.state_dict()
            sd_final.update({k: v for k, v in sd_ema.items() if k in sd_final and v.shape == sd_final[k].shape})
        torch.save(sd_final, final_path)
        print(f"Best model → {best_path} | Final → {final_path} | best_epoch={best_epoch}")

        # 測試：載入 best（+ best 溫度）
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state, strict=False)

        best_temp_for_test = None
        if args.temp_mode in ("epoch","savebest") and os.path.exists(best_temp_path):
            best_temp_for_test = _Temp().to(device)
            best_temp_for_test.load_state_dict(torch.load(best_temp_path, map_location=device))

        # ===== EM prior 修正（測試集：可縮放 α）=====
        eval_for_test = model
        if args.prior_correction == "em" and args.em_scope in ("test","both"):
            em_bias_test = prior_correction_em(model, test_loader, device)
            eval_for_test = _WithBiasScaled(model, em_bias_test, alpha=args.em_alpha_test).to(device)

        # （val AUC 檢查）【封印：不翻轉】
        if args.prior_correction == "em" and args.em_scope in ("val","both","test"):
            em_bias_val_check = prior_correction_em(model, val_loader, device)
            guard_wrap_for_check = _WithBiasScaled(model, em_bias_val_check, alpha=args.em_alpha_val).to(device)
            _ = maybe_flip_logits_on_low_auc(guard_wrap_for_check, val_loader, device, temp_model=best_temp_for_test, auc_lo=0.5)

        # ===== 門檻掃描（val；與 test 端處理一致）=====
        tuned_thr = None
        tuned_macro = None
        p1_val_for_map = None
        if (args.tune_threshold or args.calib_thresh == "macro") and num_classes == 2:
            if args.prior_correction == "em" and args.em_scope in ("test","both","val"):
                em_bias_val_for_tune = prior_correction_em(model, val_loader, device)
                class _WrapForProbVal(nn.Module):
                    def __init__(self, base, bias, alpha, temp):
                        super().__init__(); self.base=base; self.bias=bias; self.alpha=alpha; self.temp=temp
                    def forward(self, x):
                        z = self.base(x); z = z + (self.alpha * self.bias if self.bias is not None else 0)
                        return self.temp(z) if self.temp is not None else z
                wrap_prob = _WrapForProbVal(model, em_bias_val_for_tune, args.em_alpha_val, best_temp_for_test).to(device)
                p1_val, y_val = _collect_probs_binary(wrap_prob, val_loader, device, temp_model=None)
            else:
                p1_val, y_val = _collect_probs_binary(model, val_loader, device, temp_model=best_temp_for_test)
            t_raw, t_macro = tune_threshold_macro_acc(p1_val, y_val, lo=args.thr_min, hi=args.thr_max, n=args.thr_steps)
            tuned_thr = float(min(args.thr_max, max(args.thr_min, t_raw)))
            tuned_macro = t_macro
            p1_val_for_map = p1_val
            print(f"[Threshold Tuning] best threshold on val = {t_raw:.3f} (clamped to {tuned_thr:.3f}); macro={t_macro:.2f}%")

        # =====（原：Test AUC 主要保險）→【封印：僅回傳 AUC，不翻轉】=====
        if num_classes == 2:
            eval_for_test, tuned_thr, flipped_auc, auc_val = maybe_flip_by_test_auc(
                eval_for_test, test_loader, device, temp_model=best_temp_for_test, tuned_thr=tuned_thr
            )

        # Map the validation threshold quantile to the test distribution.
        if num_classes == 2 and args.enable_thr_quantile_map and tuned_thr is not None:
            p1_test, _ = _collect_probs_binary(eval_for_test, test_loader, device, temp_model=best_temp_for_test)
            if p1_val_for_map is None:
                if args.prior_correction == "em" and args.em_scope in ("test","both","val"):
                    em_bias_val_for_tune = prior_correction_em(model, val_loader, device)
                    class _WrapForProbVal2(nn.Module):
                        def __init__(self, base, bias, alpha, temp):
                            super().__init__(); self.base=base; self.bias=bias; self.alpha=alpha; self.temp=temp
                        def forward(self, x):
                            z = self.base(x); z = z + (self.alpha * self.bias if self.bias is not None else 0)
                            return self.temp(z) if self.temp is not None else z
                    wrap_prob2 = _WrapForProbVal2(model, em_bias_val_for_tune, args.em_alpha_val, best_temp_for_test).to(device)
                    p1_val_for_map, _ = _collect_probs_binary(wrap_prob2, val_loader, device, temp_model=None)
                else:
                    p1_val_for_map, _ = _collect_probs_binary(model, val_loader, device, temp_model=best_temp_for_test)

            mapped_thr, q_val = quantile_map_threshold(tuned_thr, p1_val_for_map, p1_test)
            mapped_thr_clamped = float(min(args.thr_max, max(args.thr_min, mapped_thr)))
            print(f"[QuantileMap] q=F_val(t*)≈{q_val:.4f} → t*_test≈{mapped_thr:.3f} (clamped {mapped_thr_clamped:.3f})")
            tuned_thr = mapped_thr_clamped

        # ===== （D3）AdaBN：僅刷新 BN 統計（optional）=====
        if args.adabn:
            print("[AdaBN] calibrating BN stats on test loader...")
            adabn_calibrate(eval_for_test, test_loader)

        # ===== 最終測試 =====
        te_loss, te_acc, te_cls, te_macro = evaluate(eval_for_test, test_loader, nn.CrossEntropyLoss(), device, class_names,
                                                     tta_hflip=args.tta_hflip, temp_model=best_temp_for_test,
                                                     threshold=tuned_thr)
        va_loss_end, va_acc_end, _, va_macro_end = evaluate(eval_for_test, val_loader, nn.CrossEntropyLoss(), device, class_names,
                                                            tta_hflip=args.tta_hflip, temp_model=best_temp_for_test,
                                                            threshold=tuned_thr)
        print(f"[TEST @ best] L:{te_loss:.4f} A:{te_acc:.2f}% Macro:{te_macro:.2f}% | "
              f"[val @ best] L:{va_loss_end:.4f} A:{va_acc_end:.2f}% Macro:{va_macro_end:.2f}%")
        with open(log_path_txt, "a", encoding="utf-8") as f:
            cls_txt = ", ".join([f"{k}: {v:.2f}%" for k, v in te_cls.items()])
            f.write(f"[TEST @ best] L:{te_loss:.4f} A:{te_acc:.2f}% Macro:{te_macro:.2f}% | "
                    f"[val @ best] L:{va_loss_end:.4f} A:{va_acc_end:.2f}% Macro:{va_macro_end:.2f}%\n")
            if tuned_thr is not None:
                f.write(f"[Threshold Tuning / QMap] final_thr={tuned_thr:.3f} (val_macro={tuned_macro if tuned_macro is not None else float('nan'):.2f}%)\n")
            f.write("Each Test Class Acc: " + cls_txt + "\n")

        # Collect fold predictions using the same evaluation configuration.
        @torch.no_grad()
        def _collect_preds(model_like, loader, device, temp_model=None, threshold=None, tta=False):
            model_like.eval()
            ys, preds, probs = [], [], []
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                z = model_like(x)
                if tta:
                    z = 0.5 * (z + model_like(torch.flip(x, dims=[-1])))
                if temp_model is not None:
                    z = temp_model(z)
                if z.shape[1] == 2:
                    p1 = z.softmax(1)[:, 1]
                    probs.append(p1.detach().cpu().numpy())
                if z.shape[1] == 2 and threshold is not None:
                    pred = (p1 >= threshold).long()
                else:
                    pred = z.argmax(1)
                ys.append(y.numpy())
                preds.append(pred.detach().cpu().numpy())
            probabilities = np.concatenate(probs) if probs else np.full(sum(len(y) for y in ys), np.nan)
            return np.concatenate(ys), np.concatenate(preds), probabilities

        # Record the evaluation configuration associated with the matrix.
        print(f"[CM CONFIG] thr={tuned_thr if tuned_thr is not None else 'argmax'} | "
              f"temp={'ON' if best_temp_for_test is not None else 'OFF'} | "
              f"em_scope={args.em_scope}(α_val={args.em_alpha_val}, α_test={args.em_alpha_test}) | "
              f"tta={'ON' if args.tta_hflip else 'OFF'} | "
              f"adabn={'ON' if args.adabn else 'OFF'}")

        y_te_true, y_te_pred, y_te_prob = _collect_preds(
            eval_for_test, test_loader, device, temp_model=best_temp_for_test, threshold=tuned_thr, tta=args.tta_hflip
        )

        # Normalize prediction types and guard against invalid class indices.
        y_te_true = y_te_true.astype(np.int32)
        y_te_pred = y_te_pred.astype(np.int32)
        y_te_true = np.clip(y_te_true, 0, num_classes-1)
        y_te_pred = np.clip(y_te_pred, 0, num_classes-1)

        # Write per-pig confusion matrices.
        per_pig_raw_csv = os.path.join(args.path_r, f"confusion_matrix_{test_pig}.csv")
        per_pig_row_csv = os.path.join(args.path_r, f"confusion_matrix_{test_pig}_row_normalized.csv")
        _save_confusion_matrices(
            y_te_true, y_te_pred, num_classes, per_pig_raw_csv, per_pig_row_csv,
            verbose_title=f"{test_pig} (TEST set, raw & row-norm)"
        )

        # Accumulate predictions for the merged confusion matrix.
        ALL_Y_TRUE.extend(y_te_true.tolist())
        ALL_Y_PRED.extend(y_te_pred.tolist())
        if ALL_NUM_CLASSES is None:
            ALL_NUM_CLASSES = num_classes

        # ===== 資料診斷 =====
        if args.data_diag:
            if num_classes == 2 and _HAS_SK:
                def _prob_collector(net, loader, temp):
                    net.eval()
                    ps, ys = [], []
                    with torch.no_grad():
                        for x, y in loader:
                            x = x.to(device)
                            z = net(x)
                            if temp is not None: z = temp(z)
                            p1 = z.softmax(1)[:,1].cpu().numpy()
                            ps.append(p1); ys.append(y.numpy())
                    return np.concatenate(ps), np.concatenate(ys)
                p_val, y_val = _prob_collector(eval_for_test, val_loader, best_temp_for_test)
                p_te,  y_te  = _prob_collector(eval_for_test, test_loader, best_temp_for_test)
                print(f"[AUC] val={_safe_auc(p_val,y_val):.3f} | test={_safe_auc(p_te,y_te):.3f} | "
                      f"PR(val)={_avg_precision(p_val,y_val):.3f} | PR(test)={_avg_precision(p_te,y_te):.3f}")
                with open(log_path_txt, "a", encoding="utf-8") as f:
                    f.write(f"[AUC] val={_safe_auc(p_val,y_val):.3f} | test={_safe_auc(p_te,y_te):.3f} | "
                            f"PR(val)={_avg_precision(p_val,y_val):.3f} | PR(test)={_avg_precision(p_te,y_te):.3f}\n")
            else:
                print("[AUC] 多類分類或無 sklearn，略過 AUC/PR-AUC 檢查。")

            te_paths = _paths(base_eval, test_idx)
            tr_paths = _paths(base_eval, train_idx)
            dup_ct = _near_dup_count(tr_paths, te_paths)
            if dup_ct is None:
                print("[NEAR-DUP] 未安裝 imagehash，略過近重複檢查。")
            else:
                print(f"[NEAR-DUP] test 與 train 近似（同 pHash）的影像數（抽樣）≈ {dup_ct}")
                with open(log_path_txt, "a", encoding="utf-8") as f:
                    f.write(f"[NEAR-DUP] approx hits={dup_ct} (sampled)\n")

            mtr = _img_stats(tr_paths)
            mte = _img_stats(te_paths)
            print("[IMG STATS] train mean±std:", mtr, "\n[IMG STATS] test  mean±std:", mte)
            with open(log_path_txt, "a", encoding="utf-8") as f:
                f.write(f"[IMG STATS] train mean±std: {mtr}\n[IMG STATS] test  mean±std: {mte}\n")

            rtr = _roi_cover_ratio(tr_paths, roi_cfg, default_center)
            rte = _roi_cover_ratio(te_paths, roi_cfg, default_center)
            print("[ROI AREA] train:", rtr, "\n[ROI AREA] test :", rte)
            with open(log_path_txt, "a", encoding="utf-8") as f:
                f.write(f"[ROI AREA] train: {rtr}\n[ROI AREA] test : {rte}\n")

        report = _binary_report(y_te_true, y_te_pred, y_te_prob)
        report.update({
            "pig": test_pig,
            "thr": float(tuned_thr) if tuned_thr is not None else 0.5,
            "params": sum(p.numel() for p in model.parameters()),
        })
        report["latency_ms"], report["fps"] = _measure_latency(
            model, device, args.img_size, args.lat_warmup, args.lat_iters
        )
        fold_metrics.append(report)

        # Refit（可選）
        if args.refit:
            refit_indices = sorted(set(train_idx) | set(val_idx))
            refit_loader = DataLoader(Subset(ds_train_base, refit_indices), batch_size=args.b, shuffle=True,
                                      num_workers=args.nw, pin_memory=True, drop_last=True,
                                      worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))
            state = torch.load(best_path, map_location=device)
            model.load_state_dict(state, strict=False)
            opt_refit = optim.AdamW(model.parameters(), lr=args.head_lr*args.refit_lr_factor, weight_decay=args.wd)
            steps_per_epoch2 = max(1, len(refit_loader))
            total_steps2 = steps_per_epoch2 * max(1, args.refit_epochs)
            sched_refit = get_cosine_schedule_with_warmup(opt_refit, int(0.1*total_steps2), total_steps2)
            for ep2 in range(1, args.refit_epochs+1):
                r_loss, r_acc = train_epoch(
                    model, refit_loader, num_classes, criterion_hard, device,
                    mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, label_smoothing=args.ls,
                    optimizer=opt_refit, scheduler=sched_refit, ema=None, clip_grad=args.clip_grad,
                    scheduler_is_plateau=False, conf_penalty=args.conf_penalty, logit_l2=args.logit_l2,
                    lambda_reg=args.lambda_reg, use_sam=False
                )
                print(f"[Refit {ep2}/{args.refit_epochs}] L:{r_loss:.4f} A:{r_acc:.2f}%")
            te2_loss, te2_acc, _, te2_macro = evaluate(eval_for_test, test_loader, nn.CrossEntropyLoss(), device, class_names,
                                                       tta_hflip=args.tta_hflip, temp_model=best_temp_for_test,
                                                       threshold=tuned_thr)
            print(f"[TEST @ refit(train+val)] L:{te2_loss:.4f} A:{te2_acc:.2f}% Macro:{te2_macro:.2f}%")
            with open(log_path_txt, "a", encoding="utf-8") as f:
                f.write(f"[TEST @ refit(train+val)] L:{te2_loss:.4f} A:{te2_acc:.2f}% Macro:{te2_macro:.2f}%\n")

    print("\n===== LOPO summary (Test Acc @ best) =====")
    for row in fold_metrics:
        print(f"{row['pig']}: Acc={row['acc']:.2f}% F1={row['f1']:.2f}% AUC={row['auc']:.4f}")
    if fold_metrics:
        avg = sum(row["acc"] for row in fold_metrics) / len(fold_metrics)
        print(f"Average: {avg:.2f}%")
        summary_path = os.path.join(args.path_r, "lopo_summary.csv")
        columns = ["pig", "acc", "precision", "recall", "f1", "specificity", "auc", "thr", "params", "latency_ms", "fps"]
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in fold_metrics:
                writer.writerow({key: row[key] for key in columns})
            writer.writerow({
                "pig": "AVERAGE",
                **{key: float(np.nanmean([row[key] for row in fold_metrics])) for key in columns[1:]},
            })
        print(f"[SAVE] {summary_path}")

    # Merged confusion matrix across all held-out pigs.
    if len(ALL_Y_TRUE) == 0:
        print("[CONFMAT][WARN] ALL_Y_TRUE 為空，無法產生全豬混淆矩陣。")
    else:
        y_true_all = np.asarray(ALL_Y_TRUE, dtype=np.int32)
        y_pred_all = np.asarray(ALL_Y_PRED, dtype=np.int32)
        K = ALL_NUM_CLASSES if ALL_NUM_CLASSES is not None else (int(max(y_true_all.max(), y_pred_all.max())) + 1)
        y_true_all = np.clip(y_true_all, 0, K-1)
        y_pred_all = np.clip(y_pred_all, 0, K-1)

        out_cm_csv = os.path.join(args.path_r, "confusion_matrix_all_pigs.csv")
        out_cm_norm_csv = os.path.join(args.path_r, "confusion_matrix_all_pigs_row_normalized.csv")
        _save_confusion_matrices(
            y_true_all, y_pred_all, K, out_cm_csv, out_cm_norm_csv, verbose_title="ALL PIGS (ALL folds merged)"
        )

# ==================== Command-line entry point ====================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        "LOPO for MSANet — ROI + Letterbox + Aug + MixUp/CutMix + Weighted Sampler + EMA + Temp + Threshold + (opt)SAM + (opt)EM + (opt)data_diag + (opt)QuantileMap + (opt)ColorAlign + (opt)AdaBN + (ADD)ConfMat(per-pig & all-pigs)"
    )

    # Paths and model selection.
    ap.add_argument("--path_d", type=str, required=True)
    ap.add_argument("--path_r", type=str, default="./Result/lopo_msa")
    ap.add_argument("--path_m", type=str, default="./Model/lopo_msa")
    ap.add_argument("--pigs", type=str, default="")

    # Select the MSA or MSFUNet implementation.
    ap.add_argument("--use_msfu", action="store_true")
    ap.add_argument("--model_type", type=str, default="MSA_Addition_Pool35",
                    choices=["MSA_Addition_Pool35","MSA_Addition_Pool53"])

    # MSFU 參數
    ap.add_argument("--topk_ratio", type=float, default=0.0)
    ap.add_argument("--style_p", type=float, default=0.0)
    ap.add_argument("--style_alpha", type=float, default=0.0)
    ap.add_argument("--msfu_init_gamma", type=float, default=0.05)
    ap.add_argument("--msfu_bg_scale", type=float, default=0.0)
    ap.add_argument("--tap_idx_z", type=int, default=5)
    ap.add_argument("--tap_idx_y", type=int, default=8)
    ap.add_argument("--no_style_norm", action="store_true")
    ap.add_argument("--pool_type", type=str, default="guided", choices=["guided","gap","attn","gem"])
    ap.add_argument("--softk_tau", type=float, default=0.5)
    ap.add_argument("--softk_alpha", type=float, default=1.0)
    ap.add_argument("--no_coord_score", action="store_true")
    ap.add_argument("--no_local_refine", action="store_true")

    # ROI
    ap.add_argument("--roi_cfg", type=str, default="")
    ap.add_argument("--roi_center", type=float, default=0.90)
    ap.add_argument("--roi_fallback", type=str, default="center", choices=["center","none"])
    ap.add_argument("--roi_jitter", type=float, default=0.02)
    ap.add_argument("--no_roi", action="store_true")

    # split/訓練
    ap.add_argument("--vr", type=float, default=0.25)
    ap.add_argument("--val_cap_per_class_per_pig", type=int, default=350)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-e", type=int, default=55)
    ap.add_argument("-b", type=int, default=16)
    ap.add_argument("--nw", type=int, default=4)
    ap.add_argument("--lat_warmup", type=int, default=30)
    ap.add_argument("--lat_iters", type=int, default=200)

    # 影像與增強
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--aug", type=str, default="color_robust",
                    choices=["A","B","C","light","pig_robust","heavy","default","color_robust"])

    # Optimizer parameter groups.
    ap.add_argument("--backbone_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=5e-5)
    ap.add_argument("--msfu_lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--es_patience", type=int, default=14)

    # Scheduler / EMA / clip grad
    ap.add_argument("--lr_sched", type=str, default="cosine", choices=["cosine","plateau"])
    ap.add_argument("--warmup_ratio", type=float, default=0.20)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    # sampler / mix
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta",  type=float, default=0.7)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--mixup", type=float, default=0.05)
    ap.add_argument("--cutmix",type=float, default=0.0)
    ap.add_argument("--ls",    type=float, default=0.03)

    # Temp / report / refit
    ap.add_argument("--temp_mode", type=str, default="savebest", choices=["off","epoch","savebest"])
    ap.add_argument("--tta_hflip", action="store_true")
    ap.add_argument("--report_every", type=int, default=5)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--refit_epochs", type=int, default=3)
    ap.add_argument("--refit_lr_factor", type=float, default=0.7)

    # CE class weight
    ap.add_argument("--use_class_weight", action="store_true")

    # 模型選擇與門檻調整
    ap.add_argument("--select_by", type=str, default="macro", choices=["loss","macro"])
    ap.add_argument("--tune_threshold", action="store_true")
    ap.add_argument("--calib_thresh", type=str, default="macro", choices=["off","macro"])

    # 門檻掃描區間 (夾值)
    ap.add_argument("--thr_min", type=float, default=0.25)
    ap.add_argument("--thr_max", type=float, default=0.75)
    ap.add_argument("--thr_steps", type=int, default=41)

    # γ 暖身 & 校準強化
    ap.add_argument("--gamma_warmup_epochs", type=int, default=5)
    ap.add_argument("--lambda_reg", type=float, default=0.0)
    ap.add_argument("--conf_penalty", type=float, default=0.0)
    ap.add_argument("--logit_l2", type=float, default=0.0)

    # SAM
    ap.add_argument("--use_sam", action="store_true")
    ap.add_argument("--sam_rho", type=float, default=0.05)

    # 初始化（載入前一階段權重）
    ap.add_argument("--init_from_dir", type=str, default="")
    ap.add_argument("--init_pick", type=str, default="best",
                    choices=["best", "final", "any"])
    ap.add_argument("--init_strict", action="store_true")

    # 損失與先驗修正
    ap.add_argument("--loss_type", type=str, default="ce", choices=["ce","focal"])
    ap.add_argument("--prior_correction", type=str, default="off", choices=["off","em"])
    ap.add_argument("--focal_gamma", type=float, default=2.0)

    # EM application scope and correction strength.
    ap.add_argument("--em_scope", type=str, default="test",
                    choices=["off", "val", "test", "both"],
                    help="EM prior correction 套用範圍：off/val/test/both（預設 test）")
    ap.add_argument("--em_alpha_val", type=float, default=1.0,
                    help="驗證階段 EM 偏置縮放（0~1）")
    ap.add_argument("--em_alpha_test", type=float, default=1.0,
                    help="測試階段 EM 偏置縮放（0~1）")

    # Dataset diagnostics.
    ap.add_argument("--data_diag", action="store_true", default=True)

    # Unsupervised test flip margin; non-positive values disable the option.
    ap.add_argument("--auto_flip_margin", type=float, default=0.0)

    # Convenience flag for save-best temperature scaling.
    ap.add_argument("--enable_temp_scaling", action="store_true")

    # Validation-to-test threshold quantile mapping.
    ap.add_argument("--enable_thr_quantile_map", action="store_true")

    # D1 and D3 diagnostic options.
    ap.add_argument("--align_color_to_train", action="store_true",
                    help="啟用推論端色彩對齊 (val/test 使用 eval 統計對齊到 train 統計)")
    ap.add_argument("--align_sample", type=int, default=1200,
                    help="估計 μ/σ 時的取樣上限張數（越大越穩但越慢）")
    ap.add_argument("--adabn", action="store_true",
                    help="在 test 前使用無標籤資料刷新 BN running stats")

    args = ap.parse_args()

    if args.calib_thresh == "macro":
        args.tune_threshold = True

    torch.backends.cudnn.benchmark = os.environ.get("MSFUNET_DETERMINISTIC", "0") != "1"
    torch.backends.cuda.matmul.allow_tf32 = os.environ.get("MSFUNET_DETERMINISTIC", "0") != "1"
    os.makedirs(args.path_r, exist_ok=True); os.makedirs(args.path_m, exist_ok=True)
    t0 = time.time()
    run_lopo(args)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
