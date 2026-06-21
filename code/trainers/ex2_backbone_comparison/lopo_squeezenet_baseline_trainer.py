# Organized filename: lopo_squeezenet_baseline_trainer.py
# Purpose: LOPO trainer for the SqueezeNet baseline in the backbone comparison.
# Original source: train_without_msa_v2.py

import pathlib as _pathlib
import sys as _sys
_PROJECT_ROOT = _pathlib.Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

# -*- coding: utf-8 -*-
"""
train_lopo_squeezenet.py

Leak-safe LOPO training for SqueezeNet (no MSA) with ROI, class×pig balancing, small tuning-val (val_tune),
full validation (val_full), EarlyStopping, one-time Test eval, and optional refit using train ∪ val_full.

Each held-out pig produces the complete test metric set:
1) acc 2) precision 3) recall 4) f1 5) specificity 6) auc 7) thr 8) params 9) latency_ms / fps
The aggregate table is written to ``{path_r}/lopo_test_metrics.csv``.

Stability add-ons:
- Cosine LR with warmup (batch-wise stepping) or ReduceLROnPlateau
- EMA for eval (set --ema_decay 0 to disable)
- Gradient clipping
- Sampler stability: alpha=beta=gamma=0 → plain shuffle
- Temperature scaling on validation ( --temp_mode off|epoch|savebest )
- Color-robust aug + optional TTA(hflip)

Macro-accuracy oriented add-ons (this version):
- --select_by {loss,macro}: choose/best/early-stop by macro acc (val_tune)
- --tune_threshold (binary): sweep threshold on val_full to maximize macro acc and use it for test
- evaluate() returns (loss, overall_acc, per_cls_acc_dict, macro_acc) and supports threshold (binary)
"""

import os, json, time, argparse, random, csv
from collections import Counter, defaultdict
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel

from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import functional as TvF
import torch.nn.functional as F

# Scikit-learn metrics used by the aggregate report.
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# SqueezeNet comparison variants.
from MSFUNet_experiments.code.models.comparison_models import (
    SqueezeNetSimple, SqueezeNetWithDropout, SqueezeNetWithBatchNorm
)

# ==================== 隨機種子 ====================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(os.environ.get("MSFUNET_DETERMINISTIC", "0") == "1", warn_only=True)

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

# ==================== 基本工具 ====================
def pig_id_of(path: str) -> str:
    # 預期: .../<class>/<pig_id>/<img>.jpg
    parts = os.path.normpath(path).split(os.sep)
    return parts[-2] if len(parts) >= 3 else "unknown_pig"

def load_roi_cfg(path: Optional[str]):
    if not path: return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)  # {pig_id: [x0,y0,x1,y1]} 相對座標
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

def center_roi_box(img: Image.Image, keep_ratio: float) -> Tuple[float, float, float, float]:
    keep_ratio = max(0.1, min(keep_ratio, 1.0))
    w, h = img.size
    nw, nh = int(w * keep_ratio), int(h * keep_ratio)
    left = (w - nw) // 2
    top  = (h - nh) // 2
    return left / w, top / h, (left + nw) / w, (top + nh) / h

# ---------- ROI jitter ----------
def _clamp01(x): return max(0.0, min(1.0, float(x)))

def jitter_roi_box(roi, j: float):
    if not j or j <= 0:
        return roi
    x0, y0, x1, y1 = map(float, roi)
    w, h = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
    tx = random.uniform(-j, j) * w
    ty = random.uniform(-j, j) * h
    sx = random.uniform(1.0 - j, 1.0 + j)
    sy = random.uniform(1.0 - j, 1.0 + j)
    cx = (x0 + x1) * 0.5 + tx
    cy = (y0 + y1) * 0.5 + ty
    nw, nh = w * sx, h * sy
    nx0, ny0 = _clamp01(cx - nw * 0.5), _clamp01(cy - nh * 0.5)
    nx1, ny1 = _clamp01(cx + nw * 0.5), _clamp01(cy + nh * 0.5)
    if nx1 - nx0 < 0.02: nx1 = _clamp01(nx0 + 0.02)
    if ny1 - ny0 < 0.02: ny1 = _clamp01(ny0 + 0.02)
    return (nx0, ny0, nx1, ny1)

# ==================== StateDict Robust I/O ====================
def _unwrap_state_dict(obj):
    """
    支援各種 checkpoint 包法：
    - state_dict 本身就是 dict(參數)
    - {'state_dict': ...}
    - {'model_state_dict': ...}
    - {'ema_state_dict': ...}
    """
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint is not a dict. Got: {type(obj)}")

    # 如果已經看起來是參數（裡面 value 多為 Tensor），直接回傳
    if len(obj) > 0 and all(isinstance(v, torch.Tensor) for v in obj.values() if v is not None):
        return obj

    for k in ["state_dict", "model_state_dict", "net", "model", "ema_state_dict", "ema", "weights"]:
        if k in obj and isinstance(obj[k], dict):
            return obj[k]

    # 找不到就硬回傳（交給後面 clean 再判斷）
    return obj

def clean_state_dict(sd: dict) -> dict:
    """
    會處理：
    - EMA AveragedModel 的 n_averaged
    - DataParallel/DistributedDataParallel 的 module.
    - 其他常見包法前綴
    """
    if not isinstance(sd, dict):
        raise TypeError(f"state_dict must be a dict, got {type(sd)}")

    new_sd = {}
    for k, v in sd.items():
        if k == "n_averaged":
            continue

        nk = k
        # 依序剝掉常見前綴（可能有多層）
        for prefix in ["module.", "model.", "net."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]

        new_sd[nk] = v
    return new_sd

def save_clean_state(model_or_ema, path: str):
    # AveragedModel.state_dict() 會包含 n_averaged + module.xxx，所以一定要 clean
    sd = model_or_ema.state_dict()
    sd = clean_state_dict(sd)
    torch.save(sd, path)

def load_clean_state_into(model, ckpt_path: str, device):
    try:
        raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        raw = torch.load(ckpt_path, map_location=device)

    sd = _unwrap_state_dict(raw)
    sd = clean_state_dict(sd)

    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        missing, unexpected = [], []
        try:
            r = model.load_state_dict(sd, strict=False)
            missing = list(getattr(r, "missing_keys", []))
            unexpected = list(getattr(r, "unexpected_keys", []))
        except Exception:
            pass
        msg = (
            f"\n[load_clean_state_into] FAILED strict=True\n"
            f"ckpt={ckpt_path}\n"
            f"missing_keys(sample)={missing[:20]} (total={len(missing)})\n"
            f"unexpected_keys(sample)={unexpected[:20]} (total={len(unexpected)})\n"
        )
        raise RuntimeError(msg) from e

# ---------- Transforms ----------
class ChannelDrop:
    """隨機把某個通道清零，逼模型少依賴顏色。"""
    def __init__(self, p=0.12): self.p = p
    def __call__(self, img: Image.Image):
        if random.random() < self.p:
            r, g, b = img.split()
            zero = Image.new("L", img.size, 0)
            idx = random.choice([0, 1, 2])
            ch = [r, g, b]; ch[idx] = zero
            img = Image.merge("RGB", ch)
        return img

def build_transforms(aug_preset: str, img_size: int):
    base_resize = max(img_size + 32, int(img_size * 1.12))
    mean, std = [0.5]*3, [0.5]*3  # 歸一化到 [-1,1]

    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if aug_preset in ("color_robust", "pig_color"):
        tf_train = transforms.Compose([
            transforms.Resize((base_resize, base_resize)),
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([transforms.RandomAffine(10, translate=(0.05,0.05),
                                                            scale=(0.95,1.05))], p=0.4),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=(0.0, 0.6), hue=0.02
            )], p=0.7),
            transforms.RandomGrayscale(p=0.30),
            transforms.RandomApply([transforms.Lambda(TvF.equalize)], p=0.15),
            transforms.RandomApply([transforms.RandomAutocontrast()], p=0.15),
            ChannelDrop(p=0.12),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.20),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.35),
        ])
    elif aug_preset in ("B", "pig_robust"):
        tf_train = transforms.Compose([
            transforms.Resize((base_resize, base_resize)),
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([transforms.RandomAffine(10, translate=(0.05, 0.05),
                                                            scale=(0.95, 1.05))], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.6),
            transforms.RandomGrayscale(p=0.35),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.35),
        ])
    elif aug_preset in ("C", "heavy"):
        tf_train = transforms.Compose([
            transforms.Resize((base_resize, base_resize)),
            transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0), ratio=(0.7, 1.4)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)], p=0.7),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.5),
        ])
    elif aug_preset in ("G", "pig_gray"):
        tf_train = transforms.Compose([
            transforms.Resize((base_resize, base_resize)),
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(10, translate=(0.05,0.05), scale=(0.95,1.05)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.3),
        ])
    elif aug_preset == "light":
        tf_train = transforms.Compose([
            transforms.Resize((base_resize, base_resize)),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        tf_train = transforms.Compose([
            transforms.Resize((base_resize, base_resize)),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25),
        ])
    return tf_train, tf_eval

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

# ==================== Mixup / CutMix / Soft CE ====================
def one_hot(labels, num_classes, smoothing=0.0):
    with torch.no_grad():
        y = torch.empty(size=(labels.size(0), num_classes), device=labels.device)
        y.fill_(smoothing / (num_classes - 1) if num_classes > 1 else 0.0)
        y.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing if num_classes > 1 else 1.0)
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
        self.log_t = nn.Parameter(torch.zeros(1))  # 初始 T=1
    def forward(self, z): return z / self.log_t.exp()

@torch.no_grad()
def _gather_logits_targets(model, loader, device):
    model.eval()
    net = model.module if isinstance(model, AveragedModel) else model
    zs, ys = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        z = net(x)  # logits
        zs.append(z); ys.append(y)
    return torch.cat(zs), torch.cat(ys)

def fit_temperature(model, val_loader, device):
    z, y = _gather_logits_targets(model, val_loader, device)
    if z.numel() == 0:
        return None
    t = _Temp().to(device)
    nll = nn.CrossEntropyLoss()
    try:
        opt = torch.optim.LBFGS([t.log_t], lr=0.1, max_iter=50)
        def _closure():
            opt.zero_grad(set_to_none=True)
            loss = nll(t(z), y); loss.backward(); return loss
        opt.step(_closure)
    except Exception:
        opt = torch.optim.Adam([t.log_t], lr=1e-2)
        for _ in range(200):
            opt.zero_grad(set_to_none=True)
            loss = nll(t(z), y); loss.backward(); opt.step()
    return t

# ==================== WeightedRandomSampler ====================
def build_weighted_sampler(dataset, indices, alpha=1.0, beta=0.7, gamma=0.5):
    labels, pigs, pigcls = [], [], []
    for i in indices:
        path, y = dataset.samples[i]
        pid = pig_id_of(path)
        labels.append(y); pigs.append(pid); pigcls.append((pid, y))
    cls_counter = Counter(labels)
    pig_counter = Counter(pigs)
    pigcls_counter = Counter(pigcls)
    weights = []
    for i in indices:
        path, y = dataset.samples[i]
        pid = pig_id_of(path)
        wc = 1.0 / max(1, cls_counter[y])
        wp = 1.0 / max(1, pig_counter[pid])
        wg = 1.0 / max(1, pigcls_counter[(pid, y)])
        weights.append((wc**alpha) * (wp**beta) * (wg**gamma))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(indices), replacement=True)

# ==================== Train / Eval（含 macro 與 threshold） ====================
def train_epoch(model, loader, num_classes, criterion_hard, device, scaler,
                mixup_alpha, cutmix_alpha, label_smoothing, optimizer,
                scheduler=None, ema: Optional[AveragedModel]=None, clip_grad: float=0.0):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            x_mix, y_soft, _ = apply_mixup_cutmix(x, y, num_classes, mixup_alpha, cutmix_alpha)
            logits = model(x_mix)
            if mixup_alpha > 0 or cutmix_alpha > 0 or label_smoothing > 0:
                if label_smoothing > 0 and (mixup_alpha <= 0 and cutmix_alpha <= 0):
                    y_soft = one_hot(y, num_classes, smoothing=label_smoothing)
                loss = soft_cross_entropy(logits, y_soft)
            else:
                loss = criterion_hard(logits, y)
        if device.type == "cuda":
            scaler.scale(loss).backward()
            if clip_grad and clip_grad > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if clip_grad and clip_grad > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update_parameters(model)
        loss_sum += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / max(1, len(loader)), 100.0 * correct / max(1, total)

@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names,
             temp_model: Optional[_Temp]=None, tta_hflip: bool=False, threshold: Optional[float]=None):
    """回傳: (loss, overall_acc, per_cls_acc_dict, macro_acc)；threshold 僅二分類生效。"""
    model.eval()
    net = model.module if isinstance(model, AveragedModel) else model
    loss_sum, correct, total = 0.0, 0, 0
    per_cls_c = [0]*len(class_names); per_cls_t = [0]*len(class_names)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = net(x)
            if tta_hflip:
                x2 = torch.flip(x, dims=[-1])
                logits = 0.5 * (logits + net(x2))
            if temp_model is not None:
                logits = temp_model(logits)
            loss = criterion(logits, y)
        loss_sum += loss.item()

        if (threshold is not None) and (logits.shape[1] == 2):
            prob1 = logits.softmax(1)[:,1]
            pred = (prob1 >= threshold).long()
        else:
            pred = logits.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)
        for yy, pp in zip(y, pred):
            per_cls_t[yy.item()] += 1
            if yy == pp: per_cls_c[yy.item()] += 1
    acc = 100.0 * correct / max(1, total)
    cls_acc = {class_names[i]: (100.0*per_cls_c[i]/per_cls_t[i] if per_cls_t[i] else 0.0)
               for i in range(len(class_names))}
    macro = sum(cls_acc.values()) / max(1, len(cls_acc))
    return loss_sum / max(1, len(loader)), acc, cls_acc, macro

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

def tune_threshold_macro_acc(p1: np.ndarray, y: np.ndarray,
                             thr_min: float = 0.25, thr_max: float = 0.75, thr_steps: int = 41):
    thr_min = float(thr_min); thr_max = float(thr_max)
    thr_steps = int(thr_steps)

    thr_min = max(0.0, min(1.0, thr_min))
    thr_max = max(0.0, min(1.0, thr_max))
    if thr_max < thr_min:
        thr_min, thr_max = thr_max, thr_min
    thr_steps = max(3, thr_steps)

    ts = np.linspace(thr_min, thr_max, thr_steps)

    best_t, best_macro = 0.5, -1.0
    for t in ts:
        pred = (p1 >= t).astype(np.int32)
        a0 = (pred[y == 0] == 0).mean() if (y == 0).any() else 0.0
        a1 = (pred[y == 1] == 1).mean() if (y == 1).any() else 0.0
        macro = 0.5 * (a0 + a1)
        if macro > best_macro:
            best_macro, best_t = macro, t

    return float(best_t)

# ==================== Binary classification metrics ====================
@torch.no_grad()
def collect_binary_metrics(model, loader, device, temp_model=None, threshold=0.5, tta_hflip: bool=False):
    """
    回傳 (acc, precision, recall, f1, specificity, auc)
    - acc/precision/recall/f1/specificity: 用 threshold 轉成 hard pred
    - auc: 用機率 y_prob 計算
    """
    model.eval()
    net = model.module if isinstance(model, AveragedModel) else model

    y_true, y_pred, y_prob = [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = net(x)
        if tta_hflip:
            x2 = torch.flip(x, dims=[-1])
            logits = 0.5 * (logits + net(x2))
        if temp_model is not None:
            logits = temp_model(logits)

        prob = logits.softmax(1)[:, 1].detach().cpu().numpy()
        pred = (prob >= float(threshold)).astype(int)

        y_true.extend(y.numpy())
        y_pred.extend(pred.tolist())
        y_prob.extend(prob.tolist())

    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        spec = 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    return float(acc), float(prec), float(rec), float(f1), float(spec), float(auc)

# ==================== Latency and throughput measurement ====================
@torch.no_grad()
def measure_latency_fps(model, device, img_size: int, batch_size: int = 1,
                        warmup: int = 30, iters: int = 200, use_amp: bool = True):
    """
    用隨機張量測 latency(ms) 與 fps。
    - latency: 每 batch 平均推論時間 / batch_size
    - fps: batch_size / (avg_time_sec_per_batch)
    """
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # warmup
    for _ in range(max(0, warmup)):
        if device.type == "cuda":
            with autocast(device_type="cuda", enabled=use_amp):
                _ = model(x)
        else:
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(max(1, iters)):
        if device.type == "cuda":
            with autocast(device_type="cuda", enabled=use_amp):
                _ = model(x)
        else:
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_sec_per_batch = (t1 - t0) / max(1, iters)
    fps = batch_size / avg_sec_per_batch
    lat_ms = (avg_sec_per_batch * 1000.0) / batch_size  # per-image latency
    return float(lat_ms), float(fps)

# ==================== Early Stopping（支援 macro/ loss） ====================
class EarlyStopper:
    def __init__(self, patience=7, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # "min" for loss, "max" for score
        self.best = float("inf") if mode=="min" else -float("inf")
        self.count = 0
        self.best_epoch = 0
    def step(self, value, epoch):
        improved = (value < (self.best - self.min_delta)) if self.mode=="min" else (value > (self.best + self.min_delta))
        if improved:
            self.best = value
            self.count = 0
            self.best_epoch = epoch
        else:
            self.count += 1
        return improved, (self.count >= self.patience)

# ==================== Val split ====================
def build_val_splits(base_dataset, candidate_indices, val_pigs_set, cap_per_class_per_pig: int):
    train_idx, val_pool = [], []
    for i in candidate_indices:
        path, y = base_dataset.samples[i]
        pig = pig_id_of(path)
        if pig in val_pigs_set:
            val_pool.append((i, pig, y))
        else:
            train_idx.append(i)
    val_full_idx = [i for i, _, _ in val_pool]
    by_pig_cls = defaultdict(list)
    for i, pig, y in val_pool:
        by_pig_cls[(pig, y)].append(i)
    val_tune_idx = []
    for (pig, y), idxs in by_pig_cls.items():
        rnd = random.Random(12345 + hash(pig) + int(y))
        rnd.shuffle(idxs)
        if cap_per_class_per_pig and cap_per_class_per_pig > 0:
            val_tune_idx.extend(idxs[:cap_per_class_per_pig])
        else:
            val_tune_idx.extend(idxs)
    def pigs_of(idxs):
        return set(pig_id_of(base_dataset.samples[i][0]) for i in idxs)
    assert len(pigs_of(train_idx) & pigs_of(val_full_idx)) == 0, "train/val 有相同豬 → 洩漏"
    return train_idx, val_tune_idx, val_full_idx

# ==================== 模型 ====================
def setup_squeezenet(num_classes: int, model_type: str, dropout_rate: float, freeze_option=0):
    if model_type == 'SqueezeNetSimple':
        model = SqueezeNetSimple(num_classes)
    elif model_type == 'SqueezeNetWithDropout':
        model = SqueezeNetWithDropout(num_classes, dropout_rate=dropout_rate)
    else:
        model = SqueezeNetWithBatchNorm(num_classes)

    if freeze_option == 1:
        # 只訓練 classifier，凍結 backbone
        for n, p in model.named_parameters():
            if 'classifier' not in n:
                p.requires_grad = False
    return model

def build_param_groups_squeezenet(model, backbone_lr, head_lr, weight_decay, freeze_first_k=0):
    back_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if ("features" in n) or n.startswith("squeezenet.features"):
            back_params.append(p)
        else:
            head_params.append(p)
    if not back_params:
        return [{"params": head_params, "lr": head_lr, "weight_decay": weight_decay}]
    return [
        {"params": back_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr,      "weight_decay": weight_decay},
    ]

# ==================== 主流程：LOPO ====================
def run_lopo(args):
    set_seed(args.seed)
    roi_cfg = load_roi_cfg(args.roi_cfg)
    tf_train, tf_eval = build_transforms(args.aug, args.img_size)

    default_center = None if args.roi_fallback == "none" else args.roi_center

    base_eval = PigImageFolder(root=args.path_d, transform=None, roi_cfg=roi_cfg,
                               default_center=default_center, roi_jitter=0.0)
    class_names = base_eval.classes
    num_classes = len(class_names)

    if num_classes != 2:
        raise ValueError(f"目前九指標輸出版本預設二分類。你現在 classes={class_names} (num_classes={num_classes})")

    all_pigs = sorted(set(pig_id_of(p) for p, _ in base_eval.samples))
    print(f"偵測到豬數：{len(all_pigs)} → {all_pigs}")

    ds_train_base = PigImageFolder(root=args.path_d, transform=tf_train, roi_cfg=roi_cfg,
                                   default_center=default_center,
                                   roi_jitter=(0.0 if args.no_roi else args.roi_jitter))
    ds_eval_base  = PigImageFolder(root=args.path_d, transform=tf_eval,  roi_cfg=roi_cfg,
                                   default_center=default_center, roi_jitter=0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.path_r, exist_ok=True); os.makedirs(args.path_m, exist_ok=True)

    pigs_to_run = all_pigs if not args.pigs else [p for p in args.pigs.split(",") if p in all_pigs]
    if not pigs_to_run:
        raise ValueError("沒有可用的豬可跑；請檢查 --pigs 或資料夾結構。")

    fold_metrics = []

    # 這個 CSV 會聚合所有 fold 的九指標（論文表格用）
    metrics_csv = os.path.join(args.path_r, "lopo_test_metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pig","thr","acc","precision","recall","f1","specificity","auc","params","latency_ms","fps"])

    for fold_idx, test_pig in enumerate(pigs_to_run, start=1):
        test_idx, trainval_idx = [], []
        for i, (path, _) in enumerate(ds_eval_base.samples):
            (test_idx if pig_id_of(path) == test_pig else trainval_idx).append(i)

        trainval_pigs = sorted(set(pig_id_of(ds_eval_base.samples[i][0]) for i in trainval_idx))
        rng = random.Random(args.seed + fold_idx)
        rng.shuffle(trainval_pigs)
        n_val = max(1, int(len(trainval_pigs) * args.vr))
        val_pigs = set(trainval_pigs[:n_val])

        train_idx, val_tune_idx, val_full_idx = build_val_splits(
            base_dataset=ds_eval_base,
            candidate_indices=trainval_idx,
            val_pigs_set=val_pigs,
            cap_per_class_per_pig=args.val_cap_per_class_per_pig
        )

        train_ds     = Subset(ds_train_base, train_idx)
        val_tune_ds  = Subset(ds_eval_base,  val_tune_idx)
        val_full_ds  = Subset(ds_eval_base,  val_full_idx)
        test_ds      = Subset(ds_eval_base,  test_idx)

        use_simple_shuffle = (args.alpha == 0.0 and args.beta == 0.0 and args.gamma == 0.0)
        if use_simple_shuffle:
            train_loader = DataLoader(train_ds, batch_size=args.b, shuffle=True,
                                      num_workers=args.nw, pin_memory=True, drop_last=True,
                                      worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))
        else:
            sampler = build_weighted_sampler(ds_train_base, train_idx,
                                             alpha=args.alpha, beta=args.beta, gamma=args.gamma)
            train_loader = DataLoader(train_ds, batch_size=args.b, sampler=sampler,
                                      num_workers=args.nw, pin_memory=True, drop_last=True,
                                      worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))

        val_tune_loader = DataLoader(val_tune_ds, batch_size=args.b, shuffle=False,
                                     num_workers=args.nw, pin_memory=True, drop_last=False,
                                     worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))
        val_full_loader = DataLoader(val_full_ds, batch_size=args.b, shuffle=False,
                                     num_workers=args.nw, pin_memory=True, drop_last=False,
                                     worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))
        test_loader     = DataLoader(test_ds, batch_size=args.b, shuffle=False,
                                     num_workers=args.nw, pin_memory=True, drop_last=False,
                                     worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))

        y_train = [ds_train_base.samples[i][1] for i in train_idx]
        cls_freq = Counter(y_train)
        ce_w = torch.tensor(
            [max(1.0, sum(cls_freq.values())/(len(cls_freq)*max(1, cls_freq.get(c, 0)))) for c in range(num_classes)],
            dtype=torch.float, device=device
        )
        criterion_hard = nn.CrossEntropyLoss(weight=(ce_w if args.use_class_weight else None))
        criterion_val  = nn.CrossEntropyLoss()

        model = setup_squeezenet(num_classes, args.model_type, args.dropout_rate, freeze_option=args.f).to(device)

        param_groups = build_param_groups_squeezenet(
            model,
            backbone_lr=args.backbone_lr if args.f == 0 else 0.0,
            head_lr=args.head_lr,
            weight_decay=args.wd,
            freeze_first_k=args.freeze_first_k if args.f == 0 else 0
        )
        optimizer = optim.AdamW(param_groups, weight_decay=args.wd)

        steps_per_epoch = max(1, len(train_loader))
        total_steps = steps_per_epoch * args.e
        if args.lr_sched == "cosine":
            warmup_steps = int(max(0, args.warmup_ratio) * total_steps)
            from transformers.optimization import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
            plateau_mode = False
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3, cooldown=1)
            plateau_mode = True

        try:
            scaler = GradScaler(device_type="cuda", enabled=(device.type == "cuda"))
        except TypeError:
            scaler = GradScaler(enabled=(device.type == "cuda"))

        ema = None
        if args.ema_decay and args.ema_decay > 0.0:
            ema = AveragedModel(
                model,
                avg_fn=lambda avg_p, p, n: args.ema_decay * avg_p + (1.0 - args.ema_decay) * p
            )

        log_path_txt = os.path.join(args.path_r, f"lopo_{test_pig}.txt")
        log_path_csv = os.path.join(args.path_r, f"lopo_{test_pig}.csv")
        best_path = os.path.join(args.path_m, f"best_{test_pig}.pth")
        best_temp_path = os.path.join(args.path_m, f"best_{test_pig}_temp.pt")
        final_path= os.path.join(args.path_m, f"final_{test_pig}.pth")

        print(f"\n===== LOPO Fold {fold_idx}/{len(pigs_to_run)} | Test pig = {test_pig} =====")
        print(f"Val pigs: {sorted(list(val_pigs))}")
        print(f"Sizes — train:{len(train_idx)} | val_tune:{len(val_tune_idx)} | val_full:{len(val_full_idx)} | test:{len(test_idx)}")
        if args.roi_cfg or (args.roi_fallback != "none"):
            print(f"ROI: cfg={'yes' if args.roi_cfg else 'no'}, fallback={args.roi_fallback}, center={None if args.roi_fallback=='none' else args.roi_center}")

        with open(log_path_txt, "w", encoding="utf-8") as ftxt:
            ftxt.write(f"Fold test pig: {test_pig}\n")
            ftxt.write(f"Val pigs: {sorted(list(val_pigs))}\n")
            ftxt.write(f"Sizes train/val_tune/val_full/test={len(train_idx)}/{len(val_tune_idx)}/{len(val_full_idx)}/{len(test_idx)}\n\n")

        with open(log_path_csv, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["epoch","train_loss","train_acc","val_tune_loss","val_tune_best",
                             "val_tune_acc","val_tune_macro","lr_backbone","lr_head"])

        select_by_macro = (args.select_by == "macro")
        stopper = EarlyStopper(patience=args.es_patience, min_delta=0.0, mode=("max" if select_by_macro else "min"))
        best_sel = -float("inf") if select_by_macro else float("inf")
        best_epoch = 0
        best_temp_state = None

        for ep in range(1, args.e+1):
            tr_loss, tr_acc = train_epoch(
                model, train_loader, num_classes, criterion_hard, device, scaler,
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, label_smoothing=args.ls,
                optimizer=optimizer,
                scheduler=(scheduler if not plateau_mode else None),
                ema=ema,
                clip_grad=args.clip_grad
            )

            eval_model = ema if ema is not None else model

            temp_for_eval = None
            if args.temp_mode == "epoch":
                temp_for_eval = fit_temperature(eval_model, val_tune_loader, device)
            elif args.temp_mode == "savebest" and best_temp_state is not None:
                temp_for_eval = _Temp().to(device)
                temp_for_eval.load_state_dict(best_temp_state)

            va_loss, va_acc, _, va_macro = evaluate(
                eval_model, val_tune_loader, criterion_val, device, class_names,
                temp_model=temp_for_eval, tta_hflip=args.tta_hflip
            )

            if plateau_mode:
                scheduler.step(va_loss if not select_by_macro else (1.0 - va_macro/100.0))

            sel_value = va_macro if select_by_macro else (-va_loss)

            if sel_value > best_sel and args.temp_mode == "savebest":
                temp_best = fit_temperature(eval_model, val_tune_loader, device)
                if temp_best is not None:
                    best_temp_state = temp_best.state_dict()
                    torch.save(best_temp_state, best_temp_path)
                    va_loss, va_acc, _, va_macro = evaluate(
                        eval_model, val_tune_loader, criterion_val, device, class_names,
                        temp_model=temp_best, tta_hflip=args.tta_hflip
                    )
                sel_value = va_macro if select_by_macro else (-va_loss)

            improved, should_stop = stopper.step(sel_value, ep)
            if improved:
                best_sel = sel_value
                best_epoch = ep
                save_clean_state((ema if ema is not None else model), best_path)
                if args.temp_mode == "savebest" and best_temp_state is not None:
                    torch.save(best_temp_state, best_temp_path)

            if len(optimizer.param_groups) == 1:
                lr_backbone = optimizer.param_groups[0]["lr"]
                lr_head     = optimizer.param_groups[0]["lr"]
            else:
                lr_backbone = optimizer.param_groups[0]["lr"]
                lr_head     = optimizer.param_groups[1]["lr"]

            note_best = "(best by macro)" if select_by_macro else f"(best:{(-best_sel):.4f})"
            line = (f"[{ep}/{args.e}] "
                    f"Train L:{tr_loss:.4f} A:{tr_acc:.2f}% | "
                    f"Val_tune L:{va_loss:.4f} {note_best} A:{va_acc:.2f}% Macro:{va_macro:.2f}% | "
                    f"LRb:{lr_backbone:.6g} LRh:{lr_head:.6g}")
            print(line)

            if (ep % 5 == 0) or (ep == args.e) or should_stop:
                temp_for_full = None
                if args.temp_mode == "epoch":
                    temp_for_full = fit_temperature(eval_model, val_tune_loader, device)
                elif args.temp_mode == "savebest" and os.path.exists(best_temp_path):
                    temp_for_full = _Temp().to(device)
                    try:
                        temp_for_full.load_state_dict(torch.load(best_temp_path, map_location=device, weights_only=True))
                    except TypeError:
                        temp_for_full.load_state_dict(torch.load(best_temp_path, map_location=device))

                vaF_loss, vaF_acc, _, vaF_macro = evaluate(
                    eval_model, val_full_loader, criterion_val, device, class_names,
                    temp_model=temp_for_full, tta_hflip=args.tta_hflip
                )
                with open(log_path_txt, "a", encoding="utf-8") as ftxt:
                    ftxt.write(line + f"\n[val_full] L:{vaF_loss:.4f} A:{vaF_acc:.2f}% Macro:{vaF_macro:.2f}%\n")

            with open(log_path_csv, "a", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow([ep, f"{tr_loss:.6f}", f"{tr_acc:.2f}",
                                 f"{va_loss:.6f}", f"{(-best_sel if not select_by_macro else va_macro):.6f}",
                                 f"{va_acc:.2f}", f"{va_macro:.2f}",
                                 f"{lr_backbone:.8f}", f"{lr_head:.8f}"])

            if should_stop:
                print(f"[EarlyStopping] patience={args.es_patience} 於 epoch {ep} 觸發。")
                break

        # 最終權重（原模型）
        torch.save(clean_state_dict(model.state_dict()), final_path)
        print(f"Best model (by {'macro' if select_by_macro else 'val_loss'}) → {best_path} | Final → {final_path} | best_epoch={best_epoch}")

        # ===== 測試（載入最佳權重 & 最佳溫度） =====
        load_clean_state_into(model, best_path, device)

        best_temp_for_test = None
        if args.temp_mode in ("epoch", "savebest") and os.path.exists(best_temp_path):
            best_temp_for_test = _Temp().to(device)
            try:
                best_temp_for_test.load_state_dict(torch.load(best_temp_path, map_location=device, weights_only=True))
            except TypeError:
                best_temp_for_test.load_state_dict(torch.load(best_temp_path, map_location=device))

        tuned_thr = None
        if args.tune_threshold and (num_classes == 2):
            p1_val, y_val = _collect_probs_binary(model, val_full_loader, device, temp_model=best_temp_for_test)
            tuned_thr = tune_threshold_macro_acc(
                p1_val, y_val,
                thr_min=args.thr_min, thr_max=args.thr_max, thr_steps=args.thr_steps
            )
            print(f"[Threshold Tuning] best threshold on val_full = {tuned_thr:.3f} "
                  f"(range={args.thr_min:.3f}~{args.thr_max:.3f}, steps={args.thr_steps})")

        thr_used = tuned_thr if tuned_thr is not None else 0.5

        # Preserve the standard loss, accuracy, and macro-accuracy output.
        te_loss, te_acc, te_cls, te_macro = evaluate(
            model, test_loader, nn.CrossEntropyLoss(), device, class_names,
            temp_model=best_temp_for_test, tta_hflip=args.tta_hflip, threshold=tuned_thr
        )
        print(f"[TEST @ best] L:{te_loss:.4f} A:{te_acc:.2f}% Macro:{te_macro:.2f}% (thr={thr_used:.3f})")

        # 九指標：acc/prec/rec/f1/spec/auc + thr/params/lat/fps
        acc, prec, rec, f1, spec, auc = collect_binary_metrics(
            model, test_loader, device,
            temp_model=best_temp_for_test,
            threshold=thr_used,
            tta_hflip=args.tta_hflip
        )

        params = sum(p.numel() for p in model.parameters())

        # latency/fps：用 dummy input 測（保持一致性）
        lat_ms, fps = measure_latency_fps(
            model, device, img_size=args.img_size,
            batch_size=args.lat_bs,
            warmup=args.lat_warmup,
            iters=args.lat_iters,
            use_amp=(device.type == "cuda")
        )

        print(
            f"[TEST 9-metrics] pig={test_pig} "
            f"acc={acc:.6f} prec={prec:.6f} rec={rec:.6f} f1={f1:.6f} "
            f"spec={spec:.6f} auc={auc:.6f} thr={thr_used:.4f} "
            f"params={params} lat_ms={lat_ms:.6f} fps={fps:.6f}"
        )

        with open(log_path_txt, "a", encoding="utf-8") as ftxt:
            cls_txt = ", ".join([f"{k}: {v:.2f}%" for k, v in te_cls.items()])
            ftxt.write(f"[TEST @ best] L:{te_loss:.4f} A:{te_acc:.2f}% Macro:{te_macro:.2f}% (thr={thr_used:.3f})\n")
            ftxt.write("Each Test Class Acc: " + cls_txt + "\n")
            ftxt.write(
                f"[TEST 9-metrics] "
                f"acc={acc:.6f}, precision={prec:.6f}, recall={rec:.6f}, f1={f1:.6f}, "
                f"specificity={spec:.6f}, auc={auc:.6f}, thr={thr_used:.4f}, "
                f"params={params}, lat_ms={lat_ms:.6f}, fps={fps:.6f}\n"
            )

        # Append the fold metrics to the aggregate CSV table.
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                test_pig,
                f"{thr_used:.4f}",
                f"{acc:.6f}",
                f"{prec:.6f}",
                f"{rec:.6f}",
                f"{f1:.6f}",
                f"{spec:.6f}",
                f"{auc:.6f}",
                params,
                f"{lat_ms:.6f}",
                f"{fps:.6f}",
            ])

        fold_metrics.append((test_pig, te_acc))

        # ===== 可選：Refit（train+val_full） =====
        if args.refit:
            refit_idx = train_idx + val_full_idx
            refit_ds  = Subset(ds_train_base, refit_idx)
            if use_simple_shuffle:
                refit_loader = DataLoader(refit_ds, batch_size=args.b, shuffle=True,
                                          num_workers=args.nw, pin_memory=True, drop_last=True,
                                          worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))
            else:
                refit_sampler = build_weighted_sampler(ds_train_base, refit_idx,
                                                       alpha=args.alpha, beta=args.beta, gamma=args.gamma)
                refit_loader  = DataLoader(refit_ds, batch_size=args.b, sampler=refit_sampler,
                                           num_workers=args.nw, pin_memory=True, drop_last=True,
                                           worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))

            model_refit = setup_squeezenet(num_classes, args.model_type, args.dropout_rate, freeze_option=args.f).to(device)
            param_groups = build_param_groups_squeezenet(model_refit,
                                                         backbone_lr=args.backbone_lr if args.f == 0 else 0.0,
                                                         head_lr=args.head_lr,
                                                         weight_decay=args.wd,
                                                         freeze_first_k=args.freeze_first_k if args.f == 0 else 0)
            opt_refit = optim.AdamW(param_groups, weight_decay=args.wd)
            try:
                scaler_refit = GradScaler(device_type="cuda", enabled=(device.type == "cuda"))
            except TypeError:
                scaler_refit = GradScaler(enabled=(device.type == "cuda"))

            steps_per_epoch2 = max(1, len(refit_loader))
            total_steps2 = steps_per_epoch2 * (args.refit_epochs if args.refit_epochs > 0 else max(1, best_epoch))
            from transformers.optimization import get_cosine_schedule_with_warmup
            sched_refit = get_cosine_schedule_with_warmup(opt_refit, int(0.1*total_steps2), total_steps2)

            refit_epochs = args.refit_epochs if args.refit_epochs > 0 else max(1, best_epoch)
            for ep2 in range(1, refit_epochs+1):
                train_epoch(model_refit, refit_loader, num_classes, criterion_hard, device,
                            scaler_refit, args.mixup, args.cutmix, args.ls, opt_refit,
                            scheduler=sched_refit, ema=None, clip_grad=args.clip_grad)

            te2_loss, te2_acc, _, te2_macro = evaluate(
                model_refit, test_loader, nn.CrossEntropyLoss(), device, class_names,
                temp_model=best_temp_for_test, tta_hflip=args.tta_hflip, threshold=tuned_thr
            )
            print(f"[TEST @ refit(train+val_full)] L:{te2_loss:.4f} A:{te2_acc:.2f}% Macro:{te2_macro:.2f}%")
            with open(log_path_txt, "a", encoding="utf-8") as ftxt:
                ftxt.write(f"[TEST @ refit(train+val_full)] L:{te2_loss:.4f} A:{te2_acc:.2f}% Macro:{te2_macro:.2f}%\n")

    print("\n===== LOPO summary (Test Acc @ best) =====")
    for pig, accv in fold_metrics:
        print(f"{pig}: {accv:.2f}%")
    if fold_metrics:
        avg = sum(a for _, a in fold_metrics) / len(fold_metrics)
        print(f"Average: {avg:.2f}%")
        print(f"Saved 9-metrics CSV → {metrics_csv}")

# ==================== Command-line entry point ====================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        "LOPO (SqueezeNet) — ROI + aug + Mixup/CutMix + leak-safe splits + EMA + TempScale + Macro-oriented selection"
    )
    ap.add_argument("--path_d", type=str, required=True, help="資料根目錄（.../exposed|not_exposed/<pig>/...）")
    ap.add_argument("--path_r", type=str, default="./Result/lopo", help="log/指標輸出資料夾")
    ap.add_argument("--path_m", type=str, default="./Model/lopo",  help="模型權重輸出資料夾")
    ap.add_argument("--pigs", type=str, default="", help="只跑特定豬 (pig01,pig03)，預設全跑")

    ap.add_argument("--roi_cfg", type=str, default="", help="每豬 ROI 配置 JSON，格式 {pig:[x0,y0,x1,y1]，相對座標0~1}")
    ap.add_argument("--roi_center", type=float, default=0.99, help="備援中心比例裁切（0~1；0.99≈幾乎全圖）")
    ap.add_argument("--roi_fallback", type=str, default="center", choices=["center", "none"],
                    help="當該豬沒有 ROI cfg 時：center=用 roi_center 中心裁切；none=全圖")
    ap.add_argument("--no_roi", action="store_true", help="（僅影響 roi_jitter=0）不關閉 cfg；保留參數以兼容舊指令")
    ap.add_argument("--roi_jitter", type=float, default=0.0, help="訓練時 ROI 平移/縮放隨機幅度（相對 ROI 寬高）")

    ap.add_argument("--vr", type=float, default=0.2, help="每折驗證豬比例（以豬為單位）")
    ap.add_argument("--val_cap_per_class_per_pig", type=int, default=300,
                    help="val_tune 每『豬×類別』最多取幾張（0=不設上限；val_full 全取）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-e",  type=int, default=30, help="epochs")
    ap.add_argument("-b",  type=int, default=8,  help="batch size")
    ap.add_argument("--nw", type=int, default=4,  help="num_workers")
    ap.add_argument("--img_size", type=int, default=224, help="輸入影像尺寸（建議 224）")

    ap.add_argument("--model_type", type=str,
                    choices=["SqueezeNetSimple", "SqueezeNetWithDropout", "SqueezeNetWithBatchNorm"],
                    default="SqueezeNetSimple")
    ap.add_argument("--dropout_rate", type=float, default=0.5, help="SqueezeNetWithDropout 的 dropout 率")

    ap.add_argument("--head_lr", type=float, default=1e-4, help="head/classifier 學習率")
    ap.add_argument("--backbone_lr", type=float, default=3e-5, help="backbone 學習率")
    ap.add_argument("--wd", type=float, default=5e-4, help="weight decay (L2)")
    ap.add_argument("-f",  type=int, default=0,  help="0=分組學習率, 1=只訓練 classifier（凍結 backbone）")
    ap.add_argument("--freeze_first_k", type=int, default=0, help="（SqueezeNet多為整體features；此參數保留為兼容）")
    ap.add_argument("--es_patience", type=int, default=10, help="EarlyStopping patience")

    ap.add_argument("--alpha", type=float, default=0.0, help="類別權重指數（WeightedSampler）")
    ap.add_argument("--beta",  type=float, default=0.0, help="豬權重指數（WeightedSampler）")
    ap.add_argument("--gamma", type=float, default=0.0, help="每『豬×類別』權重指數（WeightedSampler）")
    ap.add_argument("--mixup", type=float, default=0.2, help="mixup α（0=關閉）")
    ap.add_argument("--cutmix",type=float, default=0.0, help="cutmix α（0=關閉）")
    ap.add_argument("--ls",    type=float, default=0.0, help="label smoothing（0=關閉；與 mixup/cutmix 可併用）")
    ap.add_argument("--use_class_weight", action="store_true", help="啟用 CrossEntropy 類別權重（依 train 分佈）")

    ap.add_argument("--refit", action="store_true", help="訓練後用 train∪val_full 重訓，再測試")
    ap.add_argument("--refit_epochs", type=int, default=0, help="refit 輪數；0=使用 best_epoch")

    ap.add_argument("--aug", type=str, default="light",
                    choices=["default", "pig_robust", "heavy", "light", "B", "C",
                             "color_robust", "pig_gray", "G"],
                    help="B≈pig_robust（中等強度），C≈heavy，color_robust=降色依賴，pig_gray/G=全灰階。")

    ap.add_argument("--lr_sched", type=str, default="cosine", choices=["cosine","plateau"],
                    help="cosine=Cosine with warmup，plateau=ReduceLROnPlateau")
    ap.add_argument("--warmup_ratio", type=float, default=0.1, help="cosine 時 warmup 佔比（0~0.2 常見）")
    ap.add_argument("--ema_decay", type=float, default=0.999, help="EMA 衰減；設 0 以關閉")
    ap.add_argument("--clip_grad", type=float, default=1.0, help="梯度裁剪 max-norm；0=關閉")
    ap.add_argument("--temp_mode", type=str, default="epoch", choices=["off","epoch","savebest"],
                    help="off=不用溫度標定；epoch=每次驗證都標定；savebest=僅在刷新最佳時標定")

    ap.add_argument("--select_by", type=str, default="macro", choices=["loss","macro"],
                    help="loss=以 val_tune loss 選最優；macro=以 val_tune Macro Acc 選最優")
    ap.add_argument("--tune_threshold", action="store_true",
                    help="(二分類) 用 val_full 掃門檻，最大化 Macro Acc，並用該門檻跑 test/val_full")

    ap.add_argument("--tta_hflip", action="store_true",
                    help="在驗證/測試時做水平翻轉 TTA（logits 取平均）")
    ap.add_argument("--thr_min", type=float, default=0.25,
                    help="(binary threshold sweep) min threshold")
    ap.add_argument("--thr_max", type=float, default=0.75,
                    help="(binary threshold sweep) max threshold")
    ap.add_argument("--thr_steps", type=int, default=41,
                    help="(binary threshold sweep) number of thresholds")

    # Latency and throughput benchmark options.
    ap.add_argument("--lat_bs", type=int, default=1, help="latency 測試 batch size（建議 1）")
    ap.add_argument("--lat_warmup", type=int, default=30, help="latency 測試 warmup 次數")
    ap.add_argument("--lat_iters", type=int, default=200, help="latency 測試迭代次數")

    args = ap.parse_args()
    torch.backends.cudnn.benchmark = os.environ.get("MSFUNET_DETERMINISTIC", "0") != "1"
    torch.backends.cuda.matmul.allow_tf32 = os.environ.get("MSFUNET_DETERMINISTIC", "0") != "1"
    t0 = time.time()
    run_lopo(args)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
