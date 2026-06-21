# Organized filename: lopo_resnet18_trainer.py
# Purpose: LOPO trainer for ResNet-18 in the backbone comparison.
# Original source: resNet.py

# -*- coding: utf-8 -*-
"""
pureresnet18_lopo_train_v2.py

Leak-safe LOPO training for ResNet-18 using the standardized MSFUNet evaluation protocol:
- Pig-level LOPO: test pig holdout
- Train/Val pig-level exclusive split with leak assertions
- ROI per-pig JSON + center fallback + roi_jitter (train only)
- Letterbox (aspect-ratio pad) for train/val/test
- Train augmentation preset: color_robust (ChannelDrop, ColorJitter, RandomGray, Flip, Affine, Blur, Autocontrast, RandomErasing)
- Eval: Letterbox + Normalize, optional ColorAlignToTrain
- WeightedRandomSampler (alpha,beta,gamma); if alpha=beta=gamma=0 -> shuffle
- AdamW param-groups (backbone_lr / head_lr) + wd
- Cosine LR with warmup (step per batch) or Plateau
- EMA decay=0.999
- MixUp / CutMix / LabelSmoothing
- EarlyStopping on val_tune (loss) + val threshold sweep (macro@thr) for model selection & test threshold
- Temperature scaling (off|epoch|savebest)
- Optional TTA(hflip)
- Optional AdaBN before test (update BN stats using train loader)
- Outputs metrics:
  Accuracy, Precision, Recall, F1, Specificity, AUC, Params, Latency(ms), FPS
- Saves per-pig and all-pigs confusion matrices (raw + row-normalized CSV)

Folder structure (2 classes):
  path_d/
    exposed/
      pig01/
        xxx.jpg
    not_exposed/
      pig01/
        yyy.jpg

Run:
  python pureresnet18_lopo_train_v2.py --path_d Dataset/full --roi_cfg roi.json --img_size 224 -e 55 -b 16 --nw 4
"""

import os, json, time, argparse, random, csv, math
from collections import Counter, defaultdict
from typing import Tuple, Optional, List, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel

from PIL import Image, ImageOps
from torchvision import transforms, datasets
from torchvision.transforms import functional as TvF
from torchvision import models
from torchvision.models import ResNet18_Weights

from transformers.optimization import get_cosine_schedule_with_warmup


# =========================
# Utils: seed / worker
# =========================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(os.environ.get("MSFUNET_DETERMINISTIC", "0") == "1", warn_only=True)

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id); random.seed(seed + worker_id)


# =========================
# Path helpers / pig-id
# =========================
def pig_id_of(path: str) -> str:
    # expected: .../<class>/<pig_id>/<img>.jpg
    parts = os.path.normpath(path).split(os.sep)
    return parts[-2] if len(parts) >= 3 else "unknown_pig"


# =========================
# ROI
# =========================
def load_roi_cfg(path: Optional[str]):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)  # {pig_id:[x0,y0,x1,y1]} 0~1
    ok = {}
    for k, v in cfg.items():
        if isinstance(v, (list, tuple)) and len(v) == 4:
            ok[k] = [float(max(0.0, min(1.0, x))) for x in v]
    return ok

def crop_by_roi(img: Image.Image, roi: Tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = roi
    L = max(0, min(int(x0*w), w-1)); T = max(0, min(int(y0*h), h-1))
    R = max(L+1, min(int(x1*w), w));  B = max(T+1,  min(int(y1*h), h))
    return img.crop((L, T, R, B))

def center_roi_box(img: Image.Image, keep_ratio: float) -> Tuple[float, float, float, float]:
    keep_ratio = max(0.1, min(keep_ratio, 1.0))
    w, h = img.size
    nw, nh = int(w*keep_ratio), int(h*keep_ratio)
    L = (w-nw)//2; T = (h-nh)//2
    return L/w, T/h, (L+nw)/w, (T+nh)/h

def _clamp01(x): return max(0.0, min(1.0, float(x)))

def jitter_roi_box(roi, j: float):
    """ROI translate/scale jitter. j=0.08 => ±8% (relative to ROI w/h)."""
    if not j or j <= 0:
        return roi
    x0, y0, x1, y1 = map(float, roi)
    w, h = max(1e-6, x1-x0), max(1e-6, y1-y0)
    tx = random.uniform(-j, j) * w; ty = random.uniform(-j, j) * h
    sx = random.uniform(1.0-j, 1.0+j); sy = random.uniform(1.0-j, 1.0+j)
    cx = (x0+x1)*0.5 + tx; cy = (y0+y1)*0.5 + ty
    nw, nh = w*sx, h*sy
    nx0, ny0 = _clamp01(cx - nw*0.5), _clamp01(cy - nh*0.5)
    nx1, ny1 = _clamp01(cx + nw*0.5), _clamp01(cy + nh*0.5)
    if nx1-nx0 < 0.02: nx1 = _clamp01(nx0 + 0.02)
    if ny1-ny0 < 0.02: ny1 = _clamp01(ny0 + 0.02)
    return (nx0, ny0, nx1, ny1)


# =========================
# Letterbox (aspect ratio pad)
# =========================
class Letterbox:
    def __init__(self, out_size: int, fill=0):
        self.out = int(out_size)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            return img.resize((self.out, self.out))
        s = min(self.out / w, self.out / h)
        nw, nh = max(1, int(round(w*s))), max(1, int(round(h*s)))
        img_r = img.resize((nw, nh), Image.BILINEAR)
        pad_w = self.out - nw
        pad_h = self.out - nh
        left = pad_w // 2
        top  = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        return ImageOps.expand(img_r, border=(left, top, right, bottom), fill=self.fill)


# =========================
# Color robustness aug
# =========================
class ChannelDrop:
    """Randomly zero one channel to reduce color dependence."""
    def __init__(self, p=0.12):
        self.p = p
    def __call__(self, img: Image.Image):
        if random.random() < self.p:
            r, g, b = img.split()
            zero = Image.new("L", img.size, 0)
            ch = [r, g, b]
            ch[random.choice([0,1,2])] = zero
            img = Image.merge("RGB", ch)
        return img

class ColorAlignToTrain:
    """
    Optional: align eval image mean/std roughly to train stats (very light).
    This is intentionally conservative (you can disable).
    """
    def __init__(self, ref_mean=(0.5,0.5,0.5), ref_std=(0.25,0.25,0.25), p=1.0):
        self.ref_mean = ref_mean
        self.ref_std = ref_std
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        arr = np.asarray(img).astype(np.float32) / 255.0
        m = arr.mean(axis=(0,1), keepdims=True)
        s = arr.std(axis=(0,1), keepdims=True) + 1e-6
        ref_m = np.array(self.ref_mean, dtype=np.float32).reshape(1,1,3)
        ref_s = np.array(self.ref_std, dtype=np.float32).reshape(1,1,3)
        arr2 = (arr - m) / s * ref_s + ref_m
        arr2 = np.clip(arr2, 0.0, 1.0)
        return Image.fromarray((arr2*255).astype(np.uint8))

def build_transforms(aug_preset: str, img_size: int, use_color_align_eval: bool=False):
    mean, std = [0.5]*3, [0.5]*3  # normalize to [-1,1]
    base_resize = max(img_size + 32, int(img_size * 1.12))

    # eval: Letterbox + Normalize (+ optional color align)
    eval_ops = [Letterbox(img_size, fill=0)]
    if use_color_align_eval:
        eval_ops.append(ColorAlignToTrain(p=1.0))
    eval_ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    tf_eval = transforms.Compose(eval_ops)

    if aug_preset in ("color_robust", "pig_color"):
        tf_train = transforms.Compose([
            transforms.Resize((base_resize, base_resize)),
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([transforms.RandomAffine(10, translate=(0.05,0.05), scale=(0.95,1.05))], p=0.4),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=(0.0,0.6), hue=0.02)], p=0.7),
            transforms.RandomGrayscale(p=0.30),
            transforms.RandomApply([transforms.Lambda(TvF.equalize)], p=0.15),
            transforms.RandomApply([transforms.RandomAutocontrast()], p=0.15),
            ChannelDrop(p=0.12),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.20),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.12),
            Letterbox(img_size, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.35),
        ])
    elif aug_preset in ("B", "pig_robust"):
        tf_train = transforms.Compose([
            transforms.Resize((base_resize, base_resize)),
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([transforms.RandomAffine(10, translate=(0.05, 0.05), scale=(0.95, 1.05))], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.6),
            transforms.RandomGrayscale(p=0.35),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.15),
            Letterbox(img_size, fill=0),
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
            Letterbox(img_size, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.5),
        ])
    elif aug_preset in ("G", "pig_gray", "pig_grey"):
        tf_train = transforms.Compose([
            transforms.Resize((base_resize, base_resize)),
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(10, translate=(0.05,0.05), scale=(0.95,1.05)),
            transforms.Grayscale(num_output_channels=3),
            Letterbox(img_size, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.3),
        ])
    elif aug_preset == "light":
        tf_train = transforms.Compose([
            transforms.Resize((base_resize, base_resize)),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(0.5),
            Letterbox(img_size, fill=0),
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
            Letterbox(img_size, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25),
        ])

    return tf_train, tf_eval


# =========================
# Dataset (ROI first, then transform)
# =========================
class PigImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, roi_cfg=None, default_center=None, roi_jitter: float = 0.0):
        super().__init__(root=root, transform=transform)
        self.roi_cfg = roi_cfg or {}
        self.default_center = default_center
        self.roi_jitter = float(roi_jitter)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path).convert("RGB")

        pid = pig_id_of(path)
        roi = self.roi_cfg.get(pid, None)
        if roi is None and self.default_center is not None:
            roi = center_roi_box(img, self.default_center)

        if roi is not None:
            if self.roi_jitter > 0.0:
                roi = jitter_roi_box(roi, self.roi_jitter)
            img = crop_by_roi(img, roi)

        if self.transform is not None:
            img = self.transform(img)
        return img, target


# =========================
# ResNet-18
# =========================
def setup_model(num_classes: int, freeze_option=0):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_dim = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_dim, num_classes),
    )
    if freeze_option == 1:
        for n, p in model.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False
    return model

def _iter_resnet_blocks(model: nn.Module):
    # order: layer1..layer4, each is Sequential of BasicBlock
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name, None)
        if layer is None:
            continue
        for blk in layer:
            yield blk

def build_param_groups(model, backbone_lr, head_lr, weight_decay, freeze_first_k=0):
    # model is the underlying model (not DataParallel)
    if freeze_first_k > 0:
        k = 0
        for blk in _iter_resnet_blocks(model):
            if k >= freeze_first_k:
                break
            for p in blk.parameters():
                p.requires_grad = False
            k += 1

    back_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if n.startswith("fc.") else back_params).append(p)

    return [
        {"params": back_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


# =========================
# Mixup / Cutmix / Soft CE
# =========================
def one_hot(labels, num_classes, smoothing=0.0):
    with torch.no_grad():
        y = torch.empty((labels.size(0), num_classes), device=labels.device)
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
        cx, cy = random.randint(0, W-1), random.randint(0, H-1)
        w = int(W * (1 - lam) ** 0.5); h = int(H * (1 - lam) ** 0.5)
        x0, y0 = max(0, cx - w//2), max(0, cy - h//2)
        x1, y1b = min(W, cx + w//2), min(H, cy + h//2)
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


# =========================
# Temperature scaling
# =========================
class _Temp(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))  # T=1
    def forward(self, z): return z / self.log_t.exp()

def _unwrap_eval_net(model: nn.Module) -> nn.Module:
    """
    Always return a callable nn.Module that runs forward(x)->logits correctly.
    Supports DataParallel / AveragedModel / plain Module.
    """
    if isinstance(model, nn.DataParallel):
        return model  # DataParallel itself is callable
    if isinstance(model, AveragedModel):
        return model.module  # averaged underlying module
    return model

@torch.no_grad()
def _gather_logits_targets(model, loader, device, tta_hflip=False):
    net = _unwrap_eval_net(model)
    net.eval()
    zs, ys = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        z = net(x)
        if tta_hflip:
            z = 0.5 * (z + net(torch.flip(x, dims=[-1])))
        zs.append(z); ys.append(y)
    return torch.cat(zs), torch.cat(ys)

def fit_temperature(model, val_loader, device, tta_hflip=False):
    z, y = _gather_logits_targets(model, val_loader, device, tta_hflip=tta_hflip)
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


# =========================
# Sampler
# =========================
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

    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(indices),
        replacement=True
    )


# =========================
# Metrics
# =========================
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def safe_div(a, b): return float(a) / float(b) if b != 0 else 0.0

def binary_metrics_from_cm(cm: np.ndarray):
    # assume class 1 is positive
    tn = int(cm[0,0]); fp = int(cm[0,1]); fn = int(cm[1,0]); tp = int(cm[1,1])
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2*prec*rec, prec + rec) if (prec + rec) > 0 else 0.0
    spec = safe_div(tn, tn + fp)
    return acc, prec, rec, f1, spec

def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float64)
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        pos = y_true == 1
        neg = y_true == 0
        n_pos = int(pos.sum()); n_neg = int(neg.sum())
        if n_pos == 0 or n_neg == 0:
            return 0.0
        ranks = y_score.argsort().argsort().astype(np.float64) + 1.0
        sum_pos = float(ranks[pos].sum())
        auc = (sum_pos - n_pos*(n_pos+1)/2.0) / (n_pos*n_neg)
        return float(auc)

@torch.no_grad()
def eval_collect(model, loader, device, temp_model: Optional[_Temp]=None, tta_hflip=False):
    net = _unwrap_eval_net(model)
    net.eval()
    all_y = []
    all_pred = []
    all_prob1 = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device_type=("cuda" if device.type=="cuda" else "cpu"), enabled=True):
            logits = net(x)
            if tta_hflip:
                logits = 0.5 * (logits + net(torch.flip(x, dims=[-1])))
            if temp_model is not None:
                logits = temp_model(logits)
            prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        all_y.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        if prob.size(1) >= 2:
            all_prob1.append(prob[:, 1].detach().cpu().numpy())
    y_true = np.concatenate(all_y) if all_y else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.zeros((0,), dtype=np.int64)
    prob1 = np.concatenate(all_prob1) if all_prob1 else np.zeros((0,), dtype=np.float64)
    return y_true, y_pred, prob1

@torch.no_grad()
def evaluate_loss_acc(model, loader, criterion, device, temp_model: Optional[_Temp]=None, tta_hflip=False):
    net = _unwrap_eval_net(model)
    net.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device_type=("cuda" if device.type=="cuda" else "cpu"), enabled=True):
            logits = net(x)
            if tta_hflip:
                logits = 0.5 * (logits + net(torch.flip(x, dims=[-1])))
            if temp_model is not None:
                logits = temp_model(logits)
            loss = criterion(logits, y)
        loss_sum += float(loss.item())
        pred = logits.argmax(1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
    acc = 100.0 * correct / max(1, total)
    return loss_sum / max(1, len(loader)), acc

def threshold_sweep_binary(y_true: np.ndarray, prob1: np.ndarray, thr_min=0.25, thr_max=0.75, steps=41):
    best = {"thr": 0.5, "macro": -1.0, "acc": -1.0, "f1": -1.0, "spec": -1.0, "prec": -1.0, "rec": -1.0}
    if y_true.size == 0:
        return best
    for t in np.linspace(thr_min, thr_max, steps):
        y_pred = (prob1 >= t).astype(np.int64)
        cm = confusion_counts(y_true, y_pred, 2)
        acc, prec, rec, f1, spec = binary_metrics_from_cm(cm)
        macro = 0.5 * (rec + spec)  # macro@thr
        if macro > best["macro"]:
            best = {"thr": float(t), "macro": float(macro), "acc": float(acc),
                    "f1": float(f1), "spec": float(spec), "prec": float(prec), "rec": float(rec)}
    return best


# =========================
# Train loop
# =========================
def train_epoch(model, loader, num_classes, criterion_hard, device, scaler,
                mixup_alpha, cutmix_alpha, label_smoothing, optimizer,
                scheduler=None, ema: Optional[AveragedModel]=None, clip_grad: float=0.0):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=("cuda" if device.type=="cuda" else "cpu"), enabled=True):
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

        loss_sum += float(loss.item())
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(y.size(0))

    return loss_sum / max(1, len(loader)), 100.0 * correct / max(1, total)


# =========================
# EarlyStopping
# =========================
class EarlyStopper:
    def __init__(self, patience=14, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.count = 0

    def step(self, val_loss: float):
        improved = val_loss < (self.best - self.min_delta)
        if improved:
            self.best = float(val_loss)
            self.count = 0
        else:
            self.count += 1
        return improved, (self.count >= self.patience)


# =========================
# Split helper (train/val pigs)
# =========================
def build_val_splits(base_dataset, candidate_indices, val_pigs_set, cap_per_class_per_pig: int):
    train_idx, val_pool = [], []
    for i in candidate_indices:
        path, y = base_dataset.samples[i]
        pig = pig_id_of(path)
        (val_pool if pig in val_pigs_set else train_idx).append((i, pig, y) if pig in val_pigs_set else i)

    val_full_idx = [i for i, _, _ in val_pool]

    # val_tune: cap samples per (pig, class)
    by_pig_cls = defaultdict(list)
    for i, pig, y in val_pool:
        by_pig_cls[(pig, y)].append(i)

    val_tune_idx = []
    for (pig, y), idxs in by_pig_cls.items():
        rnd = random.Random(12345 + (hash(pig) & 0x7fffffff) + int(y))
        rnd.shuffle(idxs)
        val_tune_idx.extend(idxs if cap_per_class_per_pig <= 0 else idxs[:cap_per_class_per_pig])

    def pigs_of(idxs):
        return set(pig_id_of(base_dataset.samples[i][0]) for i in idxs)

    assert len(pigs_of(train_idx) & pigs_of(val_full_idx)) == 0, "train/val 有相同豬 → 洩漏"
    return train_idx, val_tune_idx, val_full_idx


# =========================
# AdaBN (optional) — meaningful for ResNet (BN exists)
# =========================
@torch.no_grad()
def adapt_bn(model, loader, device, max_batches=50):
    net = _unwrap_eval_net(model)
    net.train()
    bn_layers = [m for m in net.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    if len(bn_layers) == 0:
        return
    for m in bn_layers:
        m.running_mean.zero_()
        m.running_var.fill_(1)
    n = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = net(x)
        n += 1
        if n >= max_batches:
            break


# =========================
# Model stats: params & latency
# =========================
def count_params(model) -> int:
    net = _unwrap_eval_net(model)
    return sum(p.numel() for p in net.parameters())

@torch.no_grad()
def measure_latency(model, device, img_size=224, batch_size=1, iters=60, warmup=10):
    net = _unwrap_eval_net(model)
    net.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        _ = net(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = net(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    ms = (dt / iters) * 1000.0
    fps = (batch_size / (dt / iters)) if dt > 0 else 0.0
    return ms, fps


def save_confusion_csv(cm: np.ndarray, class_names: List[str], out_path: str, out_path_rownorm: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + class_names)
        for i, name in enumerate(class_names):
            w.writerow([name] + [int(x) for x in cm[i].tolist()])

    cmn = cm.astype(np.float64)
    row = cmn.sum(axis=1, keepdims=True) + 1e-12
    cmn = cmn / row
    with open(out_path_rownorm, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + class_names)
        for i, name in enumerate(class_names):
            w.writerow([name] + [f"{x:.6f}" for x in cmn[i].tolist()])


# =========================
# Save/Load helpers (FIX: EMA state_dict key mismatch)
# =========================
def _state_dict_for_save(model_or_ema: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return a state_dict that can always be loaded into the *base* ResNet (not AveragedModel wrapper).
    - If DataParallel: save its .module
    - If AveragedModel: save its .module
    - Else: save itself
    """
    if isinstance(model_or_ema, nn.DataParallel):
        return model_or_ema.module.state_dict()
    if isinstance(model_or_ema, AveragedModel):
        return model_or_ema.module.state_dict()
    return model_or_ema.state_dict()

def _load_state_dict_into_base(base_model: nn.Module, state: Dict[str, torch.Tensor], strict: bool = True):
    # base_model should be non-DataParallel, non-AveragedModel (plain ResNet)
    missing, unexpected = base_model.load_state_dict(state, strict=strict)
    if (len(missing) > 0 or len(unexpected) > 0) and strict:
        print("[WARN] strict load had mismatches. missing:", missing, " unexpected:", unexpected)


# =========================
# Main LOPO
# =========================
def run_lopo(args):
    set_seed(args.seed)
    roi_cfg = load_roi_cfg(args.roi_cfg)
    default_center = None if args.roi_fallback == "none" else args.roi_center
    tf_train, tf_eval = build_transforms(args.aug, args.img_size, use_color_align_eval=args.color_align_eval)

    base_eval = PigImageFolder(
        root=args.path_d, transform=None,
        roi_cfg=roi_cfg, default_center=default_center, roi_jitter=0.0
    )
    class_names = base_eval.classes
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("資料至少要兩類。")

    all_pigs = sorted(set(pig_id_of(p) for p, _ in base_eval.samples))
    print(f"偵測到豬數：{len(all_pigs)} → {all_pigs}")

    ds_train_base = PigImageFolder(
        root=args.path_d, transform=tf_train,
        roi_cfg=roi_cfg, default_center=default_center,
        roi_jitter=(0.0 if args.no_roi else args.roi_jitter)
    )
    ds_eval_base = PigImageFolder(
        root=args.path_d, transform=tf_eval,
        roi_cfg=roi_cfg, default_center=default_center, roi_jitter=0.0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = os.environ.get("MSFUNET_DETERMINISTIC", "0") != "1"
    torch.backends.cuda.matmul.allow_tf32 = os.environ.get("MSFUNET_DETERMINISTIC", "0") != "1"

    os.makedirs(args.path_r, exist_ok=True)
    os.makedirs(args.path_m, exist_ok=True)

    pigs_to_run = all_pigs if not args.pigs else [p for p in args.pigs.split(",") if p in all_pigs]
    if not pigs_to_run:
        raise ValueError("沒有可用的豬可跑；請檢查 --pigs 或資料夾結構。")

    all_fold_rows = []
    all_cm_sum = np.zeros((num_classes, num_classes), dtype=np.int64)

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

        def pigs_of(idxs):
            return set(pig_id_of(ds_eval_base.samples[i][0]) for i in idxs)
        assert len(pigs_of(train_idx) & pigs_of(val_full_idx)) == 0, "train/val 洩漏"
        assert len(pigs_of(train_idx) & pigs_of(test_idx)) == 0, "train/test 洩漏"
        assert len(pigs_of(val_full_idx) & pigs_of(test_idx)) == 0, "val/test 洩漏"

        train_ds = Subset(ds_train_base, train_idx)
        val_tune_ds = Subset(ds_eval_base, val_tune_idx)
        val_full_ds = Subset(ds_eval_base, val_full_idx)
        test_ds = Subset(ds_eval_base, test_idx)

        use_simple_shuffle = (args.alpha == 0.0 and args.beta == 0.0 and args.gamma == 0.0)
        persistent = (args.nw > 0)

        if use_simple_shuffle:
            train_loader = DataLoader(
                train_ds, batch_size=args.b, shuffle=True,
                num_workers=args.nw, pin_memory=True, drop_last=True,
                worker_init_fn=worker_init_fn, persistent_workers=persistent
            )
        else:
            sampler = build_weighted_sampler(ds_train_base, train_idx, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
            train_loader = DataLoader(
                train_ds, batch_size=args.b, sampler=sampler,
                num_workers=args.nw, pin_memory=True, drop_last=True,
                worker_init_fn=worker_init_fn, persistent_workers=persistent
            )

        val_tune_loader = DataLoader(val_tune_ds, batch_size=args.b, shuffle=False, num_workers=args.nw,
                                     pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn,
                                     persistent_workers=persistent)
        val_full_loader = DataLoader(val_full_ds, batch_size=args.b, shuffle=False, num_workers=args.nw,
                                     pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn,
                                     persistent_workers=persistent)
        test_loader = DataLoader(test_ds, batch_size=args.b, shuffle=False, num_workers=args.nw,
                                 pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn,
                                 persistent_workers=persistent)

        # loss weights for training CE
        y_train = [ds_train_base.samples[i][1] for i in train_idx]
        cls_freq = Counter(y_train)
        ce_w = torch.tensor(
            [max(1.0, sum(cls_freq.values()) / (len(cls_freq) * max(1, cls_freq.get(c, 0)))) for c in range(num_classes)],
            dtype=torch.float, device=device
        )
        criterion_hard = nn.CrossEntropyLoss(weight=ce_w)
        criterion_val = nn.CrossEntropyLoss()

        # model
        model = setup_model(num_classes, freeze_option=args.f).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # optimizer groups
        base_for_groups = model.module if isinstance(model, nn.DataParallel) else model
        param_groups = build_param_groups(
            base_for_groups,
            backbone_lr=(args.backbone_lr if args.f == 0 else 0.0),
            head_lr=args.head_lr,
            weight_decay=args.wd,
            freeze_first_k=(args.freeze_first_k if args.f == 0 else 0),
        )
        optimizer = optim.AdamW(param_groups, weight_decay=args.wd)

        # scheduler
        steps_per_epoch = max(1, len(train_loader))
        total_steps = steps_per_epoch * args.e
        if args.lr_sched == "cosine":
            warmup_steps = int(max(0, args.warmup_ratio) * total_steps)
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
            plateau_mode = False
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3, cooldown=1)
            plateau_mode = True

        # scaler
        try:
            scaler = GradScaler(device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=True)
        except TypeError:
            scaler = GradScaler(enabled=(device.type == "cuda"))

        # EMA
        ema = None
        if args.ema_decay and args.ema_decay > 0.0:
            base = model.module if isinstance(model, nn.DataParallel) else model
            ema = AveragedModel(
                base,
                avg_fn=lambda avg_p, p, n: args.ema_decay * avg_p + (1.0 - args.ema_decay) * p
            ).to(device)

        # logs & ckpts
        log_path_txt = os.path.join(args.path_r, f"lopo_{test_pig}.txt")
        log_path_csv = os.path.join(args.path_r, f"lopo_{test_pig}.csv")
        best_path = os.path.join(args.path_m, f"best_{test_pig}.pth")
        best_temp_path = os.path.join(args.path_m, f"best_{test_pig}_temp.pt")

        print(f"\n===== LOPO Fold {fold_idx}/{len(pigs_to_run)} | Test pig = {test_pig} =====")
        print(f"Val pigs: {sorted(list(val_pigs))}")
        print(f"Sizes — train:{len(train_idx)} | val_tune:{len(val_tune_idx)} | val_full:{len(val_full_idx)} | test:{len(test_idx)}")

        with open(log_path_txt, "w", encoding="utf-8") as ftxt:
            ftxt.write(f"Fold test pig: {test_pig}\n")
            ftxt.write(f"Val pigs: {sorted(list(val_pigs))}\n")
            ftxt.write(f"Sizes train/val_tune/val_full/test={len(train_idx)}/{len(val_tune_idx)}/{len(val_full_idx)}/{len(test_idx)}\n\n")

        with open(log_path_csv, "w", newline="", encoding="utf-8") as fcsv:
            csv.writer(fcsv).writerow([
                "epoch","train_loss","train_acc",
                "val_tune_loss","val_tune_acc",
                "val_full_loss","val_full_acc",
                "best_val_tune_loss","best_epoch",
                "thr_best_macro","thr_best",
                "lr_backbone","lr_head"
            ])

        stopper = EarlyStopper(patience=args.es_patience, min_delta=0.0)
        best_val = float("inf")
        best_epoch = 0
        best_temp_state = None
        best_thr = 0.5
        best_thr_macro = -1.0

        for ep in range(1, args.e + 1):
            tr_loss, tr_acc = train_epoch(
                model, train_loader, num_classes, criterion_hard, device, scaler,
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, label_smoothing=args.ls,
                optimizer=optimizer,
                scheduler=(scheduler if not plateau_mode else None),
                ema=ema, clip_grad=args.clip_grad
            )

            # choose model for validation
            eval_model = ema if ema is not None else model

            # temperature model for evaluation
            temp_for_eval = None
            if args.temp_mode == "epoch":
                temp_for_eval = fit_temperature(eval_model, val_tune_loader, device, tta_hflip=args.tta_hflip)
            elif args.temp_mode == "savebest" and best_temp_state is not None:
                temp_for_eval = _Temp().to(device)
                temp_for_eval.load_state_dict(best_temp_state)

            va_loss, va_acc = evaluate_loss_acc(
                eval_model, val_tune_loader, criterion_val, device,
                temp_model=temp_for_eval, tta_hflip=args.tta_hflip
            )

            if plateau_mode:
                scheduler.step(va_loss)

            # savebest temperature: only when val improves
            if args.temp_mode == "savebest" and va_loss < best_val:
                temp_best = fit_temperature(eval_model, val_tune_loader, device, tta_hflip=args.tta_hflip)
                if temp_best is not None:
                    best_temp_state = temp_best.state_dict()
                    torch.save(best_temp_state, best_temp_path)
                    va_loss, va_acc = evaluate_loss_acc(
                        eval_model, val_tune_loader, criterion_val, device,
                        temp_model=temp_best, tta_hflip=args.tta_hflip
                    )

            improved, should_stop = stopper.step(va_loss)
            if va_loss < best_val:
                best_val = va_loss
                best_epoch = ep
                torch.save(_state_dict_for_save(eval_model), best_path)

            vaF_loss, vaF_acc = (float("nan"), float("nan"))
            if (ep % args.val_full_every == 0) or (ep == args.e) or should_stop:
                temp_for_full = None
                if args.temp_mode == "epoch":
                    temp_for_full = fit_temperature(eval_model, val_tune_loader, device, tta_hflip=args.tta_hflip)
                elif args.temp_mode == "savebest" and best_temp_state is not None:
                    temp_for_full = _Temp().to(device)
                    temp_for_full.load_state_dict(best_temp_state)

                vaF_loss, vaF_acc = evaluate_loss_acc(
                    eval_model, val_full_loader, criterion_val, device,
                    temp_model=temp_for_full, tta_hflip=args.tta_hflip
                )

                if num_classes == 2:
                    y_true_v, _, prob1_v = eval_collect(eval_model, val_full_loader, device, temp_model=temp_for_full, tta_hflip=args.tta_hflip)
                    best_sweep = threshold_sweep_binary(y_true_v, prob1_v, args.thr_min, args.thr_max, args.thr_steps)
                    thr_macro, thr_val = best_sweep["macro"], best_sweep["thr"]
                    if thr_macro > best_thr_macro:
                        best_thr_macro = thr_macro
                        best_thr = thr_val

                with open(log_path_txt, "a", encoding="utf-8") as ftxt:
                    ftxt.write(
                        f"[{ep}/{args.e}] Train L:{tr_loss:.4f} A:{tr_acc:.2f}% | "
                        f"Val_tune L:{va_loss:.4f} A:{va_acc:.2f}% | "
                        f"Val_full L:{vaF_loss:.4f} A:{vaF_acc:.2f}% | "
                        f"thr_best_macro:{best_thr_macro:.4f} thr:{best_thr:.3f}\n"
                    )

            lr_backbone = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else 0.0
            lr_head = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]["lr"]

            line = (f"[{ep}/{args.e}] Train L:{tr_loss:.4f} A:{tr_acc:.2f}% | "
                    f"Val_tune L:{va_loss:.4f} (best:{best_val:.4f}@{best_epoch}) A:{va_acc:.2f}% | "
                    f"Val_full A:{vaF_acc if not math.isnan(vaF_acc) else -1:.2f}% | "
                    f"thr_best_macro:{best_thr_macro:.4f} thr:{best_thr:.3f} | "
                    f"LRb:{lr_backbone:.3g} LRh:{lr_head:.3g}")
            print(line)

            with open(log_path_csv, "a", newline="", encoding="utf-8") as fcsv:
                csv.writer(fcsv).writerow([
                    ep, f"{tr_loss:.6f}", f"{tr_acc:.2f}",
                    f"{va_loss:.6f}", f"{va_acc:.2f}",
                    f"{vaF_loss:.6f}" if not math.isnan(vaF_loss) else "",
                    f"{vaF_acc:.2f}" if not math.isnan(vaF_acc) else "",
                    f"{best_val:.6f}", best_epoch,
                    f"{best_thr_macro:.6f}", f"{best_thr:.6f}",
                    f"{lr_backbone:.10f}", f"{lr_head:.10f}"
                ])

            if should_stop:
                print(f"[EarlyStopping] patience={args.es_patience} 於 epoch {ep} 觸發。")
                break

        print(f"Best (val_tune loss) model → {best_path} | best_epoch={best_epoch}")

        # =========================
        # Load best + evaluate TEST
        # =========================
        state = torch.load(best_path, map_location=device)

        # load into base resnet (avoid DP/EMA key issues)
        base_resnet = model.module if isinstance(model, nn.DataParallel) else model
        _load_state_dict_into_base(base_resnet, state, strict=True)

        best_temp_for_test = None
        if args.temp_mode in ("epoch", "savebest") and best_temp_state is not None:
            best_temp_for_test = _Temp().to(device)
            best_temp_for_test.load_state_dict(best_temp_state)

        if args.adabn:
            adapt_bn(model, train_loader, device, max_batches=args.adabn_batches)

        params = count_params(model)
        lat_ms, fps = measure_latency(model, device, img_size=args.img_size, batch_size=1,
                                      iters=args.lat_iters, warmup=args.lat_warmup)

        y_true_te, _, prob1_te = eval_collect(model, test_loader, device, temp_model=best_temp_for_test, tta_hflip=args.tta_hflip)

        if num_classes == 2:
            y_pred_te = (prob1_te >= best_thr).astype(np.int64)
            cm_te = confusion_counts(y_true_te, y_pred_te, 2)
            acc, prec, rec, f1, spec = binary_metrics_from_cm(cm_te)
            auc = roc_auc_binary(y_true_te, prob1_te)
        else:
            y_true_te, y_pred_te, _ = eval_collect(model, test_loader, device, temp_model=best_temp_for_test, tta_hflip=args.tta_hflip)
            cm_te = confusion_counts(y_true_te, y_pred_te, num_classes)
            acc = safe_div(np.trace(cm_te), cm_te.sum())
            prec = rec = f1 = spec = auc = 0.0

        fold_row = {
            "pig": test_pig,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1, "spec": spec, "auc": auc,
            "thr": best_thr if num_classes == 2 else 0.0,
            "params": params, "lat_ms": lat_ms, "fps": fps
        }
        all_fold_rows.append(fold_row)
        all_cm_sum += cm_te

        print(
            f"[TEST] pig={test_pig}  "
            f"Acc={acc*100:.2f}%  P={prec*100:.2f}%  R={rec*100:.2f}%  F1={f1*100:.2f}%  "
            f"Spec={spec*100:.2f}%  AUC={auc:.4f}  thr={fold_row['thr']:.3f}  "
            f"Params={params/1e6:.3f}M  Lat={lat_ms:.2f}ms  FPS={fps:.1f}"
        )

        cm_path = os.path.join(args.path_r, f"cm_{test_pig}.csv")
        cmn_path = os.path.join(args.path_r, f"cm_{test_pig}_rownorm.csv")
        save_confusion_csv(cm_te, class_names, cm_path, cmn_path)

        with open(log_path_txt, "a", encoding="utf-8") as ftxt:
            ftxt.write(
                f"\n[TEST] Acc={acc*100:.2f}% Prec={prec*100:.2f}% Recall={rec*100:.2f}% "
                f"F1={f1*100:.2f}% Spec={spec*100:.2f}% AUC={auc:.4f} thr={fold_row['thr']:.3f}\n"
                f"Params={params}  Lat(ms)={lat_ms:.3f}  FPS={fps:.2f}\n"
            )

    # =========================
    # All pigs summary + CM
    # =========================
    print("\n===== LOPO summary (per pig) =====")
    for r in all_fold_rows:
        print(f"{r['pig']}: Acc={r['acc']*100:.2f}% F1={r['f1']*100:.2f}% AUC={r['auc']:.4f} thr={r['thr']:.3f}")

    if all_fold_rows:
        acc_avg = float(np.mean([r["acc"] for r in all_fold_rows]))
        f1_avg = float(np.mean([r["f1"] for r in all_fold_rows]))
        auc_avg = float(np.mean([r["auc"] for r in all_fold_rows]))
        spec_avg = float(np.mean([r["spec"] for r in all_fold_rows]))
        prec_avg = float(np.mean([r["prec"] for r in all_fold_rows]))
        rec_avg = float(np.mean([r["rec"] for r in all_fold_rows]))
        lat_avg = float(np.mean([r["lat_ms"] for r in all_fold_rows]))
        fps_avg = float(np.mean([r["fps"] for r in all_fold_rows]))
        params_avg = float(np.mean([r["params"] for r in all_fold_rows]))

        print("\n===== LOPO summary (avg) =====")
        print(
            f"Acc={acc_avg*100:.2f}%  Prec={prec_avg*100:.2f}%  Recall={rec_avg*100:.2f}%  "
            f"F1={f1_avg*100:.2f}%  Spec={spec_avg*100:.2f}%  AUC={auc_avg:.4f}  "
            f"Params={params_avg/1e6:.3f}M  Lat={lat_avg:.2f}ms  FPS={fps_avg:.1f}"
        )

        cm_path = os.path.join(args.path_r, "cm_all_pigs.csv")
        cmn_path = os.path.join(args.path_r, "cm_all_pigs_rownorm.csv")
        save_confusion_csv(all_cm_sum, class_names, cm_path, cmn_path)

        summary_path = os.path.join(args.path_r, "lopo_summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pig","acc","prec","rec","f1","spec","auc","thr","params","lat_ms","fps"])
            for r in all_fold_rows:
                w.writerow([
                    r["pig"],
                    f"{r['acc']:.6f}", f"{r['prec']:.6f}", f"{r['rec']:.6f}", f"{r['f1']:.6f}",
                    f"{r['spec']:.6f}", f"{r['auc']:.6f}", f"{r['thr']:.6f}",
                    r["params"], f"{r['lat_ms']:.6f}", f"{r['fps']:.6f}"
                ])


# =========================
# CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser("LOPO — ResNet-18 + ROI + Letterbox + robust aug + EarlyStop + EMA + Temp + thr-sweep + TTA")

    # data & output
    ap.add_argument("--path_d", type=str, required=True)
    ap.add_argument("--path_r", type=str, default="./Result/lopo_resnet18")
    ap.add_argument("--path_m", type=str, default="./Model/lopo_resnet18")
    ap.add_argument("--pigs", type=str, default="")

    # ROI
    ap.add_argument("--roi_cfg", type=str, default="")
    ap.add_argument("--roi_center", type=float, default=0.99)
    ap.add_argument("--roi_fallback", type=str, default="center", choices=["center", "none"])
    ap.add_argument("--no_roi", action="store_true")
    ap.add_argument("--roi_jitter", type=float, default=0.02)

    # split / train
    ap.add_argument("--vr", type=float, default=0.25)
    ap.add_argument("--val_cap_per_class_per_pig", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-e", type=int, default=55)
    ap.add_argument("-b", type=int, default=16)
    ap.add_argument("--nw", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=224)

    # optim
    ap.add_argument("--head_lr", type=float, default=5e-5)
    ap.add_argument("--backbone_lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("-f", type=int, default=0, help="0=finetune backbone+head, 1=freeze backbone train head only")
    ap.add_argument("--freeze_first_k", type=int, default=0, help="freeze first K residual blocks across layer1..4 (only when -f 0)")
    ap.add_argument("--es_patience", type=int, default=14)

    # sampler
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--gamma", type=float, default=0.5)

    # mixup/cutmix/ls
    ap.add_argument("--mixup", type=float, default=0.05)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--ls", type=float, default=0.03)

    # augmentation
    ap.add_argument("--aug", type=str, default="color_robust",
                    choices=["default", "pig_robust", "heavy", "light", "B", "C", "color_robust", "pig_gray", "G"])
    ap.add_argument("--color_align_eval", action="store_true")

    # LR / EMA / clip / temp / TTA
    ap.add_argument("--lr_sched", type=str, default="cosine", choices=["cosine", "plateau"])
    ap.add_argument("--warmup_ratio", type=float, default=0.20)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--temp_mode", type=str, default="savebest", choices=["off", "epoch", "savebest"])
    ap.add_argument("--tta_hflip", action="store_true")

    # val_full eval frequency + threshold sweep
    ap.add_argument("--val_full_every", type=int, default=5)
    ap.add_argument("--thr_min", type=float, default=0.25)
    ap.add_argument("--thr_max", type=float, default=0.75)
    ap.add_argument("--thr_steps", type=int, default=41)

    # AdaBN + latency
    ap.add_argument("--adabn", action="store_true")
    ap.add_argument("--adabn_batches", type=int, default=50)
    ap.add_argument("--lat_iters", type=int, default=60)
    ap.add_argument("--lat_warmup", type=int, default=10)

    args = ap.parse_args()
    t0 = time.time()
    run_lopo(args)
    print(f"\nTotal time: {time.time() - t0:.1f}s")
