# Organized filename: lopo_vit_tiny_trainer.py
# Purpose: LOPO trainer for ViT-Tiny in the backbone comparison.
# Original source: train_vit_small.py

# -*- coding: utf-8 -*-
"""
lopo_vit_timm_unified.py

Unified leak-safe LOPO protocol for timm ViT mixed-hook (extendable).
Dataset structure:
data_dir/
  exposed/<pig_id>/*.jpg
  not_exposed/<pig_id>/*.jpg

Key features aligned with the standardized MSFUNet LOPO protocol:
- Strict pig-level LOPO (no leakage assertions)
- ROI: per-pig JSON + center fallback + roi_jitter
- Letterbox (aspect-ratio padding) for train/val/test
- color_robust augmentation preset
- WeightedRandomSampler with alpha/beta/gamma (class × pig × pig-class)
- AdamW param groups (backbone_lr/head_lr)
- Cosine LR + warmup (step-wise)
- EMA
- MixUp (default 0.05), LabelSmoothing (default 0.03)
- EarlyStopping on best Macro-F1@thr from val_tune threshold sweep
- Temperature scaling (savebest)
- Optional TTA(hflip)
- Metrics: Acc/Prec/Rec/F1/Spec/AUC + Params + Latency(ms)/FPS
- Save per-pig and all-pigs confusion matrices (raw + row-normalized CSV)

Fixes:
- AMP/GradScaler compatibility: DO NOT pass device_type to GradScaler.
- autocast enabled only on CUDA.
"""

import os, json, time, argparse, random, csv
from collections import Counter, defaultdict
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# AMP compatibility:
# - Prefer torch.cuda.amp for broad compatibility.
# - If only torch.amp exists, fallback safely.
try:
    from torch.cuda.amp import GradScaler, autocast  # most compatible
except Exception:
    from torch.amp import GradScaler, autocast  # newer API

from PIL import Image, ImageOps
from torchvision import datasets, transforms
from torchvision.transforms import functional as TvF

import timm
from transformers.optimization import get_cosine_schedule_with_warmup


# ----------------- Repro -----------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(os.environ.get("MSFUNET_DETERMINISTIC", "0") == "1", warn_only=True)

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id); random.seed(seed + worker_id)


# ----------------- Pig ID / ROI -----------------
def pig_id_of(path: str) -> str:
    # expected: .../<class>/<pig_id>/<img>.jpg
    parts = os.path.normpath(path).split(os.sep)
    return parts[-2] if len(parts) >= 3 else "unknown_pig"

def load_roi_cfg(path: Optional[str]) -> Dict[str, List[float]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)  # {pig_id:[x0,y0,x1,y1]} in 0~1
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
    keep_ratio = max(0.1, min(float(keep_ratio), 1.0))
    w, h = img.size
    nw, nh = int(w*keep_ratio), int(h*keep_ratio)
    L = (w-nw)//2; T = (h-nh)//2
    return L/w, T/h, (L+nw)/w, (T+nh)/h

def _clamp01(x): return max(0.0, min(1.0, float(x)))

def jitter_roi_box(roi, j: float):
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


# ----------------- Letterbox -----------------
class Letterbox:
    """Aspect ratio padding to square."""
    def __init__(self, size: int, fill: int = 0):
        self.size = int(size)
        self.fill = int(fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            return img.resize((self.size, self.size))
        scale = min(self.size / w, self.size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img_r = img.resize((nw, nh), resample=Image.BILINEAR)
        pad_w = self.size - nw
        pad_h = self.size - nh
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        img_p = ImageOps.expand(img_r, border=(left, top, right, bottom), fill=self.fill)
        return img_p


# ----------------- Augs (color_robust preset) -----------------
class ChannelDrop:
    def __init__(self, p=0.12): self.p = float(p)
    def __call__(self, img: Image.Image):
        if random.random() < self.p:
            r, g, b = img.split()
            zero = Image.new("L", img.size, 0)
            ch = [r, g, b]
            ch[random.choice([0, 1, 2])] = zero
            img = Image.merge("RGB", ch)
        return img

def build_transforms_color_robust(img_size: int):
    mean, std = [0.5]*3, [0.5]*3  # [-1, 1]
    tf_train = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=(0.0,0.6), hue=0.02)], p=0.7),
        transforms.RandomGrayscale(p=0.30),
        transforms.RandomApply([transforms.Lambda(TvF.equalize)], p=0.15),
        transforms.RandomApply([transforms.RandomAutocontrast()], p=0.15),
        ChannelDrop(p=0.12),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.20),
        transforms.RandomApply([transforms.RandomAffine(10, translate=(0.05,0.05), scale=(0.95,1.05))], p=0.4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.35),
    ])
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # NOTE: Letterbox in Dataset for train/val/test consistency
    return tf_train, tf_eval


# ----------------- Dataset: ROI -> Letterbox -> transform -----------------
class PigImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, roi_cfg=None,
                 roi_fallback_center: Optional[float]=0.99,
                 roi_jitter: float = 0.0,
                 use_roi: bool = True,
                 letterbox_size: int = 224):
        super().__init__(root=root, transform=None)
        self.tf = transform
        self.roi_cfg = roi_cfg or {}
        self.roi_fallback_center = roi_fallback_center
        self.roi_jitter = float(roi_jitter)
        self.use_roi = bool(use_roi)
        self.letterbox = Letterbox(letterbox_size, fill=0)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path).convert("RGB")

        if self.use_roi:
            pid = pig_id_of(path)
            roi = self.roi_cfg.get(pid, None)
            if roi is None and self.roi_fallback_center is not None:
                roi = center_roi_box(img, self.roi_fallback_center)
            if roi is not None:
                if self.roi_jitter > 0:
                    roi = jitter_roi_box(roi, self.roi_jitter)
                img = crop_by_roi(img, roi)

        img = self.letterbox(img)

        if self.tf is not None:
            img = self.tf(img)
        return img, target


# ----------------- WeightedRandomSampler (class × pig × pig-class) -----------------
def build_weighted_sampler(dataset: PigImageFolder, indices: List[int],
                          alpha=0.5, beta=0.7, gamma=0.5) -> WeightedRandomSampler:
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
                                 num_samples=len(indices),
                                 replacement=True)


# ----------------- Mixup + Label smoothing -----------------
def one_hot(labels, num_classes, smoothing=0.0):
    with torch.no_grad():
        y = torch.empty((labels.size(0), num_classes), device=labels.device)
        y.fill_(smoothing / (num_classes - 1) if num_classes > 1 else 0.0)
        y.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing if num_classes > 1 else 1.0)
    return y

def soft_cross_entropy(logits, target_prob):
    log_prob = F.log_softmax(logits, dim=1)
    return -(target_prob * log_prob).sum(dim=1).mean()

def apply_mixup(x, y, num_classes, mixup_alpha=0.05, label_smoothing=0.03):
    if mixup_alpha <= 0:
        return x, one_hot(y, num_classes, smoothing=label_smoothing), 1.0
    beta_dist = torch.distributions.Beta(mixup_alpha, mixup_alpha)
    lam = float(beta_dist.sample())
    index = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[index, :]
    y1 = one_hot(y, num_classes, smoothing=label_smoothing)
    y2 = one_hot(y[index], num_classes, smoothing=label_smoothing)
    y_soft = lam * y1 + (1 - lam) * y2
    return x_mix, y_soft, lam


# ----------------- Temperature scaling -----------------
class _Temp(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))  # T=1
    def forward(self, z): return z / self.log_t.exp()

@torch.no_grad()
def _gather_logits_targets(model, loader, device):
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        z = model(x)
        zs.append(z); ys.append(y)
    if not zs:
        return None, None
    return torch.cat(zs), torch.cat(ys)

def fit_temperature(model, val_loader, device):
    z, y = _gather_logits_targets(model, val_loader, device)
    if z is None or z.numel() == 0:
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


# ----------------- EMA -----------------
class EMA:
    """Lightweight EMA for state_dict parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n].data)
        self.backup = {}


# ----------------- Model: timm ViT + multi-layer hook concat -----------------
class MixedViTModel(nn.Module):
    def __init__(self, timm_name: str, num_classes: int, layer_indices: List[int]):
        super().__init__()
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0, global_pool="")
        assert hasattr(self.backbone, "blocks"), "This timm model has no .blocks (not a ViT-like encoder)."
        self.layer_indices = sorted(set(int(i) for i in layer_indices))
        self.hidden_dim = int(getattr(self.backbone, "embed_dim", None) or self.backbone.num_features)

        self._features = []

        def hook_fn(module, inp, out):
            # out: (B, N, D)
            self._features.append(out)

        for idx in self.layer_indices:
            self.backbone.blocks[idx].register_forward_hook(hook_fn)

        in_dim = len(self.layer_indices) * self.hidden_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        self._features = []
        _ = self.backbone(x)
        pooled = [t.mean(dim=1) for t in self._features]
        feat = torch.cat(pooled, dim=1)
        return self.head(feat)


def build_param_groups(model: nn.Module, backbone_lr: float, head_lr: float, weight_decay: float):
    back_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            back_params.append(p)
        else:
            head_params.append(p)

    return [
        {"params": back_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


# ----------------- Metrics helpers -----------------
def _confusion_binary(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return tn, fp, fn, tp

def compute_metrics_binary(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(np.int64)
    tn, fp, fn, tp = _confusion_binary(y_true, y_pred)

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec  = tp / max(1, (tp + fn))
    f1   = (2 * prec * rec) / max(1e-12, (prec + rec))
    spec = tn / max(1, (tn + fp))

    # macro F1 for binary = average(F1_pos, F1_neg)
    prec_n = tn / max(1, (tn + fn))
    rec_n  = tn / max(1, (tn + fp))
    f1_n   = (2 * prec_n * rec_n) / max(1e-12, (prec_n + rec_n))
    macro_f1 = 0.5 * (f1 + f1_n)

    return {
        "acc": acc, "precision": prec, "recall": rec, "f1": f1,
        "specificity": spec, "macro_f1": macro_f1,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

def auc_safe(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return 0.0
        y_true = np.asarray(y_true).astype(np.int64)
        y_prob = np.asarray(y_prob).astype(np.float64)
        pos = y_prob[y_true == 1]
        neg = y_prob[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.0
        all_scores = np.concatenate([pos, neg])
        ranks = all_scores.argsort().argsort().astype(np.float64) + 1.0
        r_pos = ranks[:len(pos)].sum()
        auc = (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
        return float(auc)
    except Exception:
        return 0.0


@torch.no_grad()
def predict_probs(model: nn.Module, loader, device, temp: Optional[_Temp]=None, tta_hflip: bool=False):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.numpy()
        logits = model(x)
        if tta_hflip:
            logits2 = model(torch.flip(x, dims=[-1]))
            logits = 0.5 * (logits + logits2)
        if temp is not None:
            logits = temp(logits)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        ys.append(y); ps.append(prob)
    if not ys:
        return np.array([]), np.array([])
    return np.concatenate(ys), np.concatenate(ps)


def threshold_sweep_macro_f1(y_true, y_prob, thr_min=0.25, thr_max=0.75, steps=41):
    thrs = np.linspace(thr_min, thr_max, steps)
    best = {"thr": 0.5, "macro_f1": -1.0}
    for t in thrs:
        m = compute_metrics_binary(y_true, y_prob, float(t))
        if m["macro_f1"] > best["macro_f1"]:
            best = {"thr": float(t), "macro_f1": float(m["macro_f1"]), **m}
    return best


# ----------------- Latency / FPS -----------------
@torch.no_grad()
def measure_latency_fps(model: nn.Module, device, img_size: int, iters: int = 200, warmup: int = 30):
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    dt = (t1 - t0) / max(1, iters)
    latency_ms = 1000.0 * dt
    fps = 1.0 / max(1e-12, dt)
    return latency_ms, fps


# ----------------- Early stop on macro-F1 -----------------
class EarlyStopperMax:
    def __init__(self, patience=14, min_delta=1e-6):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = -1e18
        self.count = 0

    def step(self, score):
        improved = score > (self.best + self.min_delta)
        if improved:
            self.best = score
            self.count = 0
        else:
            self.count += 1
        return improved, (self.count >= self.patience)


# ----------------- Train epoch -----------------
def train_one_epoch(model, loader, device, optimizer, scheduler, scaler,
                    num_classes: int, mixup_alpha: float, label_smoothing: float,
                    clip_grad: float, ema: Optional[EMA]):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    use_amp = (device.type == "cuda")

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # IMPORTANT: autocast only enabled on CUDA
        with autocast(enabled=use_amp):
            x_mix, y_soft, _ = apply_mixup(
                x, y, num_classes,
                mixup_alpha=mixup_alpha,
                label_smoothing=label_smoothing
            )
            logits = model(x_mix)
            loss = soft_cross_entropy(logits, y_soft)

        if use_amp:
            scaler.scale(loss).backward()
            if clip_grad and clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        loss_sum += float(loss.item())
        pred = logits.argmax(1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return loss_sum / max(1, len(loader)), 100.0 * correct / max(1, total)


# ----------------- Confusion matrix save -----------------
def save_confusion_csv(path_csv: str, tn, fp, fn, tp, normalize: bool=False):
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=np.float64 if normalize else np.int64)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.maximum(row_sum, 1e-12))
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "pred_0", "pred_1"])
        w.writerow(["true_0",
                    f"{cm[0,0]:.6f}" if normalize else int(cm[0,0]),
                    f"{cm[0,1]:.6f}" if normalize else int(cm[0,1])])
        w.writerow(["true_1",
                    f"{cm[1,0]:.6f}" if normalize else int(cm[1,0]),
                    f"{cm[1,1]:.6f}" if normalize else int(cm[1,1])])


# ----------------- LOPO runner -----------------
def run_lopo(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = os.environ.get("MSFUNET_DETERMINISTIC", "0") != "1"
    torch.backends.cuda.matmul.allow_tf32 = os.environ.get("MSFUNET_DETERMINISTIC", "0") != "1"

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    roi_cfg = load_roi_cfg(args.roi_cfg)
    tf_train, tf_eval = build_transforms_color_robust(args.img_size)

    roi_fallback_center = None if args.roi_fallback == "none" else args.roi_center
    use_roi = (not args.no_roi)

    ds_train = PigImageFolder(
        root=args.data_dir,
        transform=tf_train,
        roi_cfg=roi_cfg,
        roi_fallback_center=roi_fallback_center,
        roi_jitter=args.roi_jitter,
        use_roi=use_roi,
        letterbox_size=args.img_size
    )
    ds_eval = PigImageFolder(
        root=args.data_dir,
        transform=tf_eval,
        roi_cfg=roi_cfg,
        roi_fallback_center=roi_fallback_center,
        roi_jitter=0.0,
        use_roi=use_roi,
        letterbox_size=args.img_size
    )

    class_names = ds_eval.classes
    assert len(class_names) == 2, f"Binary only. Got classes={class_names}"

    all_pigs = sorted(set(pig_id_of(p) for p, _ in ds_eval.samples))
    print(f"[INFO] pigs={len(all_pigs)} -> {all_pigs}")

    pigs_to_run = all_pigs if not args.pigs else [p for p in args.pigs.split(",") if p in all_pigs]
    if not pigs_to_run:
        raise ValueError("No pigs to run. Check --pigs or folder structure.")

    all_true, all_prob = [], []
    fold_rows = []

    for fold_idx, test_pig in enumerate(pigs_to_run, start=1):
        test_idx, trainval_idx = [], []
        for i, (path, _) in enumerate(ds_eval.samples):
            (test_idx if pig_id_of(path) == test_pig else trainval_idx).append(i)

        trainval_pigs = sorted(set(pig_id_of(ds_eval.samples[i][0]) for i in trainval_idx))
        rng = random.Random(args.seed + fold_idx)
        rng.shuffle(trainval_pigs)

        n_val = max(1, int(round(len(trainval_pigs) * args.vr)))
        val_pigs = set(trainval_pigs[:n_val])

        train_idx = [i for i in trainval_idx if pig_id_of(ds_eval.samples[i][0]) not in val_pigs]
        val_full_idx = [i for i in trainval_idx if pig_id_of(ds_eval.samples[i][0]) in val_pigs]

        by_pig_cls = defaultdict(list)
        for i in val_full_idx:
            path, y = ds_eval.samples[i]
            by_pig_cls[(pig_id_of(path), y)].append(i)
        val_tune_idx = []
        for (pid, y), idxs in by_pig_cls.items():
            rr = random.Random(12345 + hash(pid) + int(y))
            rr.shuffle(idxs)
            if args.val_cap_per_class_per_pig > 0:
                idxs = idxs[:args.val_cap_per_class_per_pig]
            val_tune_idx.extend(idxs)

        def pigs_of(idxs):
            return set(pig_id_of(ds_eval.samples[i][0]) for i in idxs)
        assert len(pigs_of(train_idx) & pigs_of(val_full_idx)) == 0, "Leakage: train and val share pigs!"
        assert len(pigs_of(train_idx) & pigs_of(test_idx)) == 0, "Leakage: train and test share pigs!"
        assert len(pigs_of(val_full_idx) & pigs_of(test_idx)) == 0, "Leakage: val and test share pigs!"

        train_ds = Subset(ds_train, train_idx)
        val_tune_ds = Subset(ds_eval, val_tune_idx)
        val_full_ds = Subset(ds_eval, val_full_idx)
        test_ds = Subset(ds_eval, test_idx)

        if args.alpha == 0 and args.beta == 0 and args.gamma == 0:
            train_loader = DataLoader(
                train_ds, batch_size=args.batch, shuffle=True,
                num_workers=args.nw, pin_memory=True, drop_last=True,
                worker_init_fn=worker_init_fn, persistent_workers=(args.nw > 0)
            )
        else:
            sampler = build_weighted_sampler(ds_train, train_idx, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch, sampler=sampler,
                num_workers=args.nw, pin_memory=True, drop_last=True,
                worker_init_fn=worker_init_fn, persistent_workers=(args.nw > 0)
            )

        val_tune_loader = DataLoader(
            val_tune_ds, batch_size=args.batch, shuffle=False,
            num_workers=args.nw, pin_memory=True, drop_last=False,
            worker_init_fn=worker_init_fn, persistent_workers=(args.nw > 0)
        )
        val_full_loader = DataLoader(
            val_full_ds, batch_size=args.batch, shuffle=False,
            num_workers=args.nw, pin_memory=True, drop_last=False,
            worker_init_fn=worker_init_fn, persistent_workers=(args.nw > 0)
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch, shuffle=False,
            num_workers=args.nw, pin_memory=True, drop_last=False,
            worker_init_fn=worker_init_fn, persistent_workers=(args.nw > 0)
        )

        total_layers = 12  # vit_tiny
        if args.layer_indices:
            layer_indices = [int(x) for x in args.layer_indices.split(",")]
        else:
            layer_indices = sorted({total_layers - 1, total_layers // 2, 0})

        model = MixedViTModel(args.timm_name, num_classes=2, layer_indices=layer_indices).to(device)

        param_groups = build_param_groups(model, backbone_lr=args.backbone_lr, head_lr=args.head_lr, weight_decay=args.wd)
        optimizer = optim.AdamW(param_groups, weight_decay=args.wd)

        steps_per_epoch = max(1, len(train_loader))
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = int(args.warmup_ratio * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        use_amp = (device.type == "cuda")
        scaler = GradScaler(enabled=use_amp)  # <-- FIXED (no device_type)
        ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

        fold_tag = f"{test_pig}"
        log_csv = os.path.join(args.out_dir, f"lopo_{fold_tag}.csv")
        best_ckpt = os.path.join(args.ckpt_dir, f"best_{fold_tag}.pth")
        best_temp = os.path.join(args.ckpt_dir, f"best_{fold_tag}_temp.pt")
        best_thr_file = os.path.join(args.ckpt_dir, f"best_{fold_tag}_thr.txt")

        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_loss","train_acc","val_best_macroF1","val_best_thr","lr_backbone","lr_head"])

        print(f"\n===== LOPO {fold_idx}/{len(pigs_to_run)} | TEST pig={test_pig} =====")
        print(f"Sizes: train={len(train_idx)} | val_tune={len(val_tune_idx)} | val_full={len(val_full_idx)} | test={len(test_idx)}")
        print(f"Val pigs={sorted(list(val_pigs))}")
        print(f"ROI: use={use_roi}, cfg={'yes' if args.roi_cfg else 'no'}, fallback={args.roi_fallback}, center={roi_fallback_center}, jitter={args.roi_jitter}")

        stopper = EarlyStopperMax(patience=args.es_patience, min_delta=1e-6)
        best_score = -1.0
        best_epoch = 0
        best_thr = 0.5
        best_temp_state = None

        for ep in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, device, optimizer, scheduler, scaler,
                num_classes=2,
                mixup_alpha=args.mixup,
                label_smoothing=args.label_smoothing,
                clip_grad=args.clip_grad,
                ema=ema
            )

            if ema is not None:
                ema.apply_shadow(model)

            temp_model = None
            if args.temp_mode == "epoch":
                temp_model = fit_temperature(model, val_tune_loader, device)

            yv, pv = predict_probs(model, val_tune_loader, device, temp=temp_model, tta_hflip=args.tta_hflip)
            sweep = threshold_sweep_macro_f1(yv, pv, args.thr_min, args.thr_max, args.thr_steps)
            score = sweep["macro_f1"]
            thr = sweep["thr"]

            if args.temp_mode == "savebest":
                if score > best_score + 1e-6:
                    tmp = fit_temperature(model, val_tune_loader, device)
                    if tmp is not None:
                        best_temp_state = tmp.state_dict()
                        torch.save(best_temp_state, best_temp)
                        yv2, pv2 = predict_probs(model, val_tune_loader, device, temp=tmp, tta_hflip=args.tta_hflip)
                        sweep2 = threshold_sweep_macro_f1(yv2, pv2, args.thr_min, args.thr_max, args.thr_steps)
                        score, thr = sweep2["macro_f1"], sweep2["thr"]

            improved, should_stop = stopper.step(score)

            if score > best_score + 1e-6:
                best_score = score
                best_epoch = ep
                best_thr = thr
                torch.save(model.state_dict(), best_ckpt)
                with open(best_thr_file, "w", encoding="utf-8") as f:
                    f.write(str(best_thr))

            lr_b = optimizer.param_groups[0]["lr"]
            lr_h = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr_b

            print(f"[{ep:03d}/{args.epochs}] "
                  f"Train L={tr_loss:.4f} A={tr_acc:.2f}% | "
                  f"Val_tune bestMacroF1={score:.4f} thr={thr:.3f} (best={best_score:.4f}@{best_thr:.3f}) | "
                  f"LRb={lr_b:.3g} LRh={lr_h:.3g}")

            with open(log_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([ep, f"{tr_loss:.6f}", f"{tr_acc:.2f}", f"{score:.6f}", f"{thr:.4f}", f"{lr_b:.8f}", f"{lr_h:.8f}"])

            if ema is not None:
                ema.restore(model)

            if should_stop:
                print(f"[EarlyStop] patience={args.es_patience} at epoch={ep}")
                break

        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state)

        best_temp_model = None
        if args.temp_mode in ("epoch", "savebest") and os.path.exists(best_temp):
            best_temp_model = _Temp().to(device)
            try:
                best_temp_model.load_state_dict(torch.load(best_temp, map_location=device, weights_only=True))
            except TypeError:
                best_temp_model.load_state_dict(torch.load(best_temp, map_location=device))

        if os.path.exists(best_thr_file):
            with open(best_thr_file, "r", encoding="utf-8") as f:
                try:
                    best_thr = float(f.read().strip())
                except:
                    best_thr = 0.5

        yvf, pvf = predict_probs(model, val_full_loader, device, temp=best_temp_model, tta_hflip=args.tta_hflip)
        yte, pte = predict_probs(model, test_loader, device, temp=best_temp_model, tta_hflip=args.tta_hflip)

        m_val = compute_metrics_binary(yvf, pvf, best_thr) if len(yvf) else compute_metrics_binary(np.array([0]), np.array([0.0]), best_thr)
        m_te  = compute_metrics_binary(yte, pte, best_thr) if len(yte) else compute_metrics_binary(np.array([0]), np.array([0.0]), best_thr)
        auc_val = auc_safe(yvf, pvf) if len(yvf) else 0.0
        auc_te  = auc_safe(yte, pte) if len(yte) else 0.0

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        latency_ms, fps = measure_latency_fps(model, device, args.img_size, iters=args.lat_iters, warmup=args.lat_warmup)

        print(f"[BEST] epoch={best_epoch} bestMacroF1={best_score:.4f} bestThr={best_thr:.3f}")
        print(f"[VAL_FULL] Acc={m_val['acc']*100:.2f}% F1={m_val['f1']:.4f} Spec={m_val['specificity']:.4f} AUC={auc_val:.4f}")
        print(f"[TEST]     Acc={m_te['acc']*100:.2f}% F1={m_te['f1']:.4f} Spec={m_te['specificity']:.4f} AUC={auc_te:.4f}")
        print(f"[MODEL] Params={params:,} | Latency={latency_ms:.2f} ms | FPS={fps:.2f}")

        tn, fp, fn, tp = m_te["tn"], m_te["fp"], m_te["fn"], m_te["tp"]
        save_confusion_csv(os.path.join(args.out_dir, f"cm_{test_pig}_raw.csv"), tn, fp, fn, tp, normalize=False)
        save_confusion_csv(os.path.join(args.out_dir, f"cm_{test_pig}_rownorm.csv"), tn, fp, fn, tp, normalize=True)

        all_true.append(yte)
        all_prob.append(pte)

        fold_rows.append({
            "pig": test_pig,
            "thr": best_thr,
            "acc": m_te["acc"],
            "precision": m_te["precision"],
            "recall": m_te["recall"],
            "f1": m_te["f1"],
            "specificity": m_te["specificity"],
            "auc": auc_te,
            "params": params,
            "latency_ms": latency_ms,
            "fps": fps
        })

    y_all = np.concatenate(all_true) if all_true else np.array([])
    p_all = np.concatenate(all_prob) if all_prob else np.array([])

    thr_avg = float(np.mean([r["thr"] for r in fold_rows])) if len(fold_rows) > 0 else 0.5

    m_all = compute_metrics_binary(y_all, p_all, thr_avg) if len(y_all) else compute_metrics_binary(np.array([0]), np.array([0.0]), thr_avg)
    auc_all = auc_safe(y_all, p_all) if len(y_all) else 0.0

    save_confusion_csv(os.path.join(args.out_dir, "cm_ALL_raw.csv"),
                       m_all["tn"], m_all["fp"], m_all["fn"], m_all["tp"], normalize=False)
    save_confusion_csv(os.path.join(args.out_dir, "cm_ALL_rownorm.csv"),
                       m_all["tn"], m_all["fp"], m_all["fn"], m_all["tp"], normalize=True)

    summary_csv = os.path.join(args.out_dir, "lopo_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pig","thr","acc","precision","recall","f1","specificity","auc","params","latency_ms","fps"])
        for r in fold_rows:
            w.writerow([r["pig"], f"{r['thr']:.4f}",
                        f"{r['acc']*100:.2f}", f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                        f"{r['f1']:.4f}", f"{r['specificity']:.4f}", f"{r['auc']:.4f}",
                        f"{r['params']}", f"{r['latency_ms']:.3f}", f"{r['fps']:.3f}"])
        w.writerow([])
        w.writerow(["ALL(avg_thr)", f"{thr_avg:.4f}",
                    f"{m_all['acc']*100:.2f}", f"{m_all['precision']:.4f}", f"{m_all['recall']:.4f}",
                    f"{m_all['f1']:.4f}", f"{m_all['specificity']:.4f}", f"{auc_all:.4f}",
                    "", "", ""])

    print("\n===== LOPO SUMMARY (ALL pigs) =====")
    print(f"thr(avg folds)={thr_avg:.3f} | Acc={m_all['acc']*100:.2f}% | "
          f"Prec={m_all['precision']:.4f} Rec={m_all['recall']:.4f} F1={m_all['f1']:.4f} "
          f"Spec={m_all['specificity']:.4f} AUC={auc_all:.4f}")
    print(f"[Saved] {summary_csv}")


# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser("LOPO unified protocol (timm ViT mixedhook)")

    ap.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="LOPO dataset root using the <class>/<pig_id>/<image> layout",
    )
    ap.add_argument("--out_dir", type=str, default="./Result/lopo_vit_timm")
    ap.add_argument("--ckpt_dir", type=str, default="./Model/lopo_vit_timm")
    ap.add_argument("--pigs", type=str, default="", help="comma-separated pig ids, empty=all")

    ap.add_argument("--roi_cfg", type=str, default="", help="JSON: {pig:[x0,y0,x1,y1]} (0~1)")
    ap.add_argument("--roi_center", type=float, default=0.99)
    ap.add_argument("--roi_fallback", type=str, default="center", choices=["center","none"])
    ap.add_argument("--roi_jitter", type=float, default=0.02)
    ap.add_argument("--no_roi", action="store_true")

    ap.add_argument("--vr", type=float, default=0.25, help="val pig ratio")
    ap.add_argument("--val_cap_per_class_per_pig", type=int, default=300)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--nw", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=224)

    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--gamma", type=float, default=0.5)

    ap.add_argument("--backbone_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.20)

    ap.add_argument("--mixup", type=float, default=0.05)
    ap.add_argument("--label_smoothing", type=float, default=0.03)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--es_patience", type=int, default=14)

    ap.add_argument("--thr_min", type=float, default=0.25)
    ap.add_argument("--thr_max", type=float, default=0.75)
    ap.add_argument("--thr_steps", type=int, default=41)

    ap.add_argument("--temp_mode", type=str, default="savebest", choices=["off","epoch","savebest"])
    ap.add_argument("--tta_hflip", action="store_true")

    ap.add_argument("--timm_name", type=str, default="vit_tiny_patch16_224")
    ap.add_argument("--layer_indices", type=str, default="", help="comma list, e.g., 0,6,11")

    ap.add_argument("--lat_iters", type=int, default=200)
    ap.add_argument("--lat_warmup", type=int, default=30)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    run_lopo(args)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
