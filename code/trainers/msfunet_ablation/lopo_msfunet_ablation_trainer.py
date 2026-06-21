# Organized filename: lopo_msfunet_trainer.py
# Purpose: LOPO trainer for E1, E3, E4, and E5 MSFUNet ablations.
# Original source: train_lopo_generic.py

# train_lopo_generic.py
# -*- coding: utf-8 -*-
"""
train_lopo_generic.py

Generic pluggable LOPO trainer with E4/E5 suite execution.

The trainer reports Accuracy, Precision, Recall, F1, Specificity, and AUC per
held-out pig and as fold-level aggregates. It supports ROI cropping, letterbox
resizing, augmentation, weighted sampling, EMA, temperature scaling, validation
threshold selection, and optional adaptation procedures. Feature taps are
configured through ``tap_idx_z`` and ``tap_idx_y``; ``-1`` disables a tap.

IMPORTANT:
- AUC requires sklearn.metrics. If not available, AUC will be N/A.
"""

import os, json, time, csv, argparse, random
from collections import Counter
from typing import Optional, Tuple, List, Dict, Set

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel

from torchvision import datasets, transforms
from transformers.optimization import get_cosine_schedule_with_warmup

_HAS_SK = False
_HAS_IH = False
_CM_AVAILABLE = False
_HAS_SK_METRICS = False

try:
    from sklearn.metrics import confusion_matrix
    _HAS_SK = True
    _CM_AVAILABLE = True
except Exception:
    pass

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SK_METRICS = True
except Exception:
    _HAS_SK_METRICS = False

try:
    import imagehash
    _HAS_IH = True
except Exception:
    pass


# ===================== seed =====================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(os.environ.get("MSFUNET_DETERMINISTIC", "0") == "1", warn_only=True)

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# ===================== path/pig id =====================
def pig_id_of(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3:
        return parts[-2]
    return parts[-1] if parts else "unknown_pig"


# ===================== ROI utils =====================
def load_roi_cfg(path: Optional[str]):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    ok = {}
    for k, v in cfg.items():
        if isinstance(v, (list, tuple)) and len(v) == 4:
            ok[k] = [float(max(0.0, min(1.0, x))) for x in v]
    return ok

def crop_by_roi(img: Image.Image, roi: Tuple[float,float,float,float]) -> Image.Image:
    w, h = img.size
    x0,y0,x1,y1 = roi
    left   = max(0, min(int(x0*w), w-1))
    top    = max(0, min(int(y0*h), h-1))
    right  = max(left+1, min(int(x1*w), w))
    bottom = max(top+1,  min(int(y1*h), h))
    return img.crop((left, top, right, bottom))

def center_roi_box(img: Image.Image, keep_ratio: float):
    keep_ratio = max(0.1, min(keep_ratio, 1.0))
    w, h = img.size
    nw, nh = int(w*keep_ratio), int(h*keep_ratio)
    left = (w-nw)//2; top=(h-nh)//2
    return left/w, top/h, (left+nw)/w, (top+nh)/h

def _clamp01(x): return max(0.0, min(1.0, float(x)))

def jitter_roi_box(roi, j: float):
    if not j or j <= 0:
        return roi
    x0,y0,x1,y1 = map(float, roi)
    w = max(1e-6, x1-x0); h = max(1e-6, y1-y0)
    tx = random.uniform(-j, j) * w
    ty = random.uniform(-j, j) * h
    sx = random.uniform(1.0-j, 1.0+j)
    sy = random.uniform(1.0-j, 1.0+j)
    cx = (x0+x1)*0.5 + tx
    cy = (y0+y1)*0.5 + ty
    nw = w*sx; nh = h*sy
    nx0, ny0 = _clamp01(cx-nw*0.5), _clamp01(cy-nh*0.5)
    nx1, ny1 = _clamp01(cx+nw*0.5), _clamp01(cy+nh*0.5)
    if nx1-nx0 < 0.02: nx1 = _clamp01(nx0+0.02)
    if ny1-ny0 < 0.02: ny1 = _clamp01(ny0+0.02)
    return (nx0, ny0, nx1, ny1)


# ===================== Letterbox =====================
class Letterbox:
    def __init__(self, out_size: int, pad_color=(114,114,114), scale_jitter=0.0):
        self.out = int(out_size)
        self.pad_color = pad_color
        self.scale_jitter = float(scale_jitter)

    def __call__(self, img: Image.Image):
        w, h = img.size
        s = min(self.out / w, self.out / h)
        if self.scale_jitter > 0:
            lo = max(0.5, 1.0 - self.scale_jitter)
            s = s * random.uniform(lo, 1.0)
            s = min(s, 1.0)
        nw = max(1, int(round(w*s)))
        nh = max(1, int(round(h*s)))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.out, self.out), self.pad_color)
        left = (self.out - nw)//2
        top  = (self.out - nh)//2
        canvas.paste(img, (left, top))
        return canvas


# ===================== custom aug =====================
class ChannelDrop:
    def __init__(self, p=0.12): self.p=p
    def __call__(self, img: Image.Image):
        if random.random() < self.p:
            r,g,b = img.split()
            zero = Image.new("L", img.size, 0)
            k = random.choice([0,1,2])
            ch = [r,g,b]; ch[k]=zero
            img = Image.merge("RGB", ch)
        return img

def build_transforms_train(aug_preset: str, img_size: int):
    mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
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
    aug_list.append(transforms.RandomGrayscale(
        p=0.25 if aug_preset in ("color_robust","pig_color") else 0.15
    ))
    return transforms.Compose([
        *aug_list,
        Letterbox(img_size, pad_color=(114,114,114), scale_jitter=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.35),
    ])

class ColorAlignToTrain:
    def __init__(self, mu_tr, sd_tr, mu_ev, sd_ev):
        self.mu_tr = mu_tr.view(-1,1,1)
        self.sd_tr = sd_tr.view(-1,1,1).clamp_min(1e-6)
        self.mu_ev = mu_ev.view(-1,1,1)
        self.sd_ev = sd_ev.view(-1,1,1).clamp_min(1e-6)
    def __call__(self, x):
        return (x-self.mu_ev)/self.sd_ev * self.sd_tr + self.mu_tr

def build_transforms_eval(img_size: int, aligner: Optional[ColorAlignToTrain]=None):
    mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
    ops = [Letterbox(img_size, pad_color=(114,114,114), scale_jitter=0.0),
           transforms.ToTensor()]
    if aligner is not None:
        ops.append(aligner)
    ops.append(transforms.Normalize(mean,std))
    return transforms.Compose(ops)


# ===================== dataset (ROI then transform) =====================
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
            if self.roi_jitter > 0:
                roi = jitter_roi_box(roi, self.roi_jitter)
            img = crop_by_roi(img, roi)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


# ===================== sampler =====================
def build_weighted_sampler(dataset: PigImageFolder, indices: List[int],
                           alpha=0.5, beta=0.7, gamma=0.5):
    labels, pigs, pigcls = [], [], []
    for i in indices:
        path, y = dataset.samples[i]
        pid = pig_id_of(path)
        labels.append(y); pigs.append(pid); pigcls.append((pid,y))

    cls_counter = Counter(labels)
    pig_counter = Counter(pigs)
    pigcls_counter = Counter(pigcls)

    weights = []
    for i in indices:
        path, y = dataset.samples[i]
        pid = pig_id_of(path)
        wc = 1.0 / max(1, cls_counter[y])
        wp = 1.0 / max(1, pig_counter[pid])
        wg = 1.0 / max(1, pigcls_counter[(pid,y)])
        weights.append((wc**alpha)*(wp**beta)*(wg**gamma))

    w = torch.as_tensor(weights, dtype=torch.double)
    if not torch.isfinite(w).all() or float(w.sum()) == 0.0:
        w = torch.ones_like(w, dtype=torch.double)

    return WeightedRandomSampler(w, num_samples=len(indices), replacement=True)


# ===================== mixup/cutmix =====================
def one_hot(labels, num_classes, smoothing=0.0):
    with torch.no_grad():
        y = torch.empty((labels.size(0), num_classes), device=labels.device)
        if num_classes > 1:
            y.fill_(smoothing/(num_classes-1))
            y.scatter_(1, labels.unsqueeze(1), 1.0-smoothing)
        else:
            y.fill_(1.0)
    return y

def soft_cross_entropy(logits, target_prob):
    log_prob = F.log_softmax(logits, dim=1)
    return -(target_prob * log_prob).sum(dim=1).mean()

def apply_mixup_cutmix(x, y, num_classes, mixup_alpha=0.05, cutmix_alpha=0.0):
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return x, one_hot(y, num_classes, 0.0), 1.0
    y1 = one_hot(y, num_classes, 0.0)

    use_cutmix = (random.random() < 0.5) and (cutmix_alpha > 0)
    if use_cutmix:
        beta = torch.distributions.Beta(cutmix_alpha, cutmix_alpha)
        lam = float(beta.sample())
        B,C,H,W = x.size()
        index = torch.randperm(B, device=x.device)
        cx, cy = random.randint(0, W-1), random.randint(0, H-1)
        bw = int(W*(1-lam)**0.5); bh=int(H*(1-lam)**0.5)
        x0,y0 = max(0,cx-bw//2), max(0,cy-bh//2)
        x1,y1b= min(W,cx+bw//2), min(H,cy+bh//2)
        x[:,:,y0:y1b,x0:x1] = x[index,:,y0:y1b,x0:x1]
        lam = 1 - ((x1-x0)*(y1b-y0)/(W*H))
        y2 = one_hot(y[index], num_classes, 0.0)
        y_soft = lam*y1 + (1-lam)*y2
        return x, y_soft, lam

    beta = torch.distributions.Beta(mixup_alpha, mixup_alpha)
    lam = float(beta.sample())
    index = torch.randperm(x.size(0), device=x.device)
    x = lam*x + (1-lam)*x[index]
    y2 = one_hot(y[index], num_classes, 0.0)
    y_soft = lam*y1 + (1-lam)*y2
    return x, y_soft, lam


# ===================== temp scaling =====================
class _Temp(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))
    def forward(self, z):
        return z / self.log_t.exp()

@torch.no_grad()
def _gather_logits_targets(model, loader, device):
    model.eval()
    zs, ys = [], []
    for x,y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = model(x)
        zs.append(z); ys.append(y)
    return torch.cat(zs,0), torch.cat(ys,0)

def fit_temperature(model, val_loader, device):
    z, y = _gather_logits_targets(model, val_loader, device)
    if z.numel() == 0:
        return None
    t = _Temp().to(device)
    nll = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([t.log_t], lr=0.1, max_iter=50)
    def closure():
        opt.zero_grad(set_to_none=True)
        loss = nll(t(z), y)
        loss.backward()
        return loss
    try:
        opt.step(closure)
    except Exception:
        opt = torch.optim.Adam([t.log_t], lr=1e-2)
        for _ in range(200):
            opt.zero_grad(set_to_none=True)
            loss = nll(t(z), y)
            loss.backward()
            opt.step()
    return t


# ===================== EM prior correction =====================
@torch.no_grad()
def prior_correction_em(model, loader, device, max_iter=20, eps=1e-5):
    model.eval()
    ps=[]
    for x,_ in loader:
        x = x.to(device, non_blocking=True)
        z = model(x)
        ps.append(z.softmax(1).detach())
    P = torch.cat(ps,0)
    pi = (P.mean(0)/P.mean(0).sum()).clamp_min(1e-6)
    for _ in range(max_iter):
        W = P * pi
        W = W / (W.sum(1, keepdim=True) + 1e-12)
        pi_new = (W.mean(0)/W.mean(0).sum()).clamp_min(1e-6)
        if (pi_new-pi).abs().max().item() < eps:
            pi = pi_new
            break
        pi = pi_new
    bias = torch.log(pi + 1e-12)
    bias = bias - bias.mean()
    return bias.to(device)

class _WithBiasScaled(nn.Module):
    def __init__(self, base, bias: Optional[torch.Tensor], alpha=1.0):
        super().__init__()
        self.base = base
        self.bias = bias
        self.alpha = float(alpha)
    def forward(self, x):
        z = self.base(x)
        if self.bias is not None:
            z = z + self.alpha * self.bias
        return z


# ===================== eval & threshold =====================
@torch.no_grad()
def evaluate(model, loader, criterion_hard, device, class_names,
             tta_hflip=False, temp_model: Optional[_Temp]=None, threshold: Optional[float]=None):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    per_cls_c = [0]*len(class_names)
    per_cls_t = [0]*len(class_names)
    n_batches=0

    for x,y in loader:
        n_batches += 1
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = model(x)
        if tta_hflip:
            z = 0.5*(z + model(torch.flip(x, dims=[-1])))
        if temp_model is not None:
            z = temp_model(z)

        loss = criterion_hard(z, y)
        loss_sum += float(loss.item())

        if threshold is not None and z.shape[1] == 2:
            p1 = z.softmax(1)[:,1]
            pred = (p1 >= threshold).long()
        else:
            pred = z.argmax(1)

        correct += (pred==y).sum().item()
        total += y.size(0)

        for yy, pp in zip(y, pred):
            per_cls_t[yy.item()] += 1
            if yy==pp:
                per_cls_c[yy.item()] += 1

    acc = 100.0*correct/max(1,total)
    cls_acc = {class_names[i]:(100.0*per_cls_c[i]/per_cls_t[i] if per_cls_t[i] else 0.0)
               for i in range(len(class_names))}
    macro = sum(cls_acc.values())/max(1,len(cls_acc))
    return (loss_sum/max(1,n_batches)), acc, cls_acc, macro

@torch.no_grad()
def _collect_probs_binary(model, loader, device, temp_model=None, tta=False):
    model.eval()
    ps, ys = [], []
    for x,y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x)
        if tta:
            z = 0.5*(z + model(torch.flip(x, dims=[-1])))
        if temp_model is not None:
            z = temp_model(z)
        p1 = z.softmax(1)[:,1].detach().cpu().numpy()
        ps.append(p1)
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)

def tune_threshold_macro_acc(p1: np.ndarray, y: np.ndarray, lo=0.25, hi=0.75, n=41):
    ts = np.linspace(lo, hi, n)
    best_t, best_macro = 0.5, -1.0
    for t in ts:
        pred = (p1 >= t).astype(np.int32)
        a0 = (pred[y==0]==0).mean() if (y==0).any() else 0.0
        a1 = (pred[y==1]==1).mean() if (y==1).any() else 0.0
        macro = 0.5*(a0+a1)
        if macro > best_macro:
            best_macro, best_t = macro, t
    return float(best_t), float(best_macro*100.0)

def quantile_map_threshold(t_star_val: float, val_probs: np.ndarray, test_probs: np.ndarray):
    q = float((val_probs <= t_star_val).mean())
    q = min(max(q, 1e-4), 1-1e-4)
    t_test = float(np.quantile(test_probs, q))
    return t_test, q


# ===================== train epoch =====================
def train_epoch(model, loader, num_classes, criterion_hard, device,
                mixup_alpha, cutmix_alpha, label_smoothing,
                optimizer, scheduler=None,
                ema: Optional[AveragedModel]=None,
                clip_grad: float=1.0,
                scheduler_is_plateau: bool=False):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    n_batches=0

    for x,y in loader:
        n_batches += 1
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        x_mix, y_soft, _ = apply_mixup_cutmix(x, y, num_classes, mixup_alpha, cutmix_alpha)
        z = model(x_mix)

        if mixup_alpha > 0 or cutmix_alpha > 0 or label_smoothing > 0:
            if label_smoothing > 0 and (mixup_alpha<=0 and cutmix_alpha<=0):
                y_soft = one_hot(y, num_classes, smoothing=label_smoothing)
            loss = soft_cross_entropy(z, y_soft)
        else:
            loss = criterion_hard(z, y)

        loss.backward()
        if clip_grad and clip_grad > 0:
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
        pred = z.detach().argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)

    return loss_sum/max(1,n_batches), 100.0*correct/max(1,total)


# ===================== early stopping =====================
class EarlyStopper:
    def __init__(self, patience=14, min_delta=0.0, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -float("inf")
        self.count = 0
        self.best_epoch = 0

    def step(self, value, epoch):
        improved = value > (self.best + self.min_delta)
        if improved:
            self.best = value
            self.count = 0
            self.best_epoch = epoch
        else:
            self.count += 1
        return improved, (self.count >= self.patience)


# ===================== split safety =====================
def _pigs_of_indices(ds: PigImageFolder, idxs: List[int]) -> Set[str]:
    return set(pig_id_of(ds.samples[i][0]) for i in idxs)

def _assert_disjoint(a: Set[str], b: Set[str], msg: str):
    inter = a & b
    assert len(inter)==0, f"[SplitError] {msg} 交集={sorted(list(inter))}"

def _assert_no_leak(ds_base: PigImageFolder, train_idx, val_idx, test_idx, test_pig: str):
    pt = _pigs_of_indices(ds_base, train_idx)
    pv = _pigs_of_indices(ds_base, val_idx)
    ps = _pigs_of_indices(ds_base, test_idx)
    _assert_disjoint(pt, pv, "train vs val")
    _assert_disjoint(pt, ps, "train vs test")
    _assert_disjoint(pv, ps, "val vs test")
    assert test_pig in ps, f"[SplitError] test_pig='{test_pig}' not in test split!"
    assert test_pig not in pt and test_pig not in pv, f"[Leakage] test_pig leaked into train/val!"


# ===================== confusion matrix =====================
def _save_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, K: int,
                             out_raw_csv: str, out_row_csv: str, title: str):
    if _CM_AVAILABLE:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
    else:
        cm = np.zeros((K,K), dtype=np.int64)
        for t,p in zip(y_true, y_pred):
            if 0 <= t < K and 0 <= p < K:
                cm[t,p] += 1
    row_sum = cm.sum(1, keepdims=True)
    cm_row = cm / np.maximum(row_sum, 1)

    os.makedirs(os.path.dirname(out_raw_csv), exist_ok=True)
    np.savetxt(out_raw_csv, cm, fmt="%d", delimiter=",")
    np.savetxt(out_row_csv, cm_row, fmt="%.6f", delimiter=",")

    print(f"\n===== Confusion Matrix — {title} =====")
    print(cm)
    print(f"[SAVE] {out_raw_csv}")
    print("===== Row-normalized =====")
    print(cm_row)
    print(f"[SAVE] {out_row_csv}")

@torch.no_grad()
def _collect_preds(model, loader, device, temp_model=None, threshold=None, tta=False):
    model.eval()
    ys, preds = [], []
    for x,y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x)
        if tta:
            z = 0.5*(z + model(torch.flip(x, dims=[-1])))
        if temp_model is not None:
            z = temp_model(z)
        if z.shape[1]==2 and threshold is not None:
            p1 = z.softmax(1)[:,1]
            pred = (p1 >= threshold).long()
        else:
            pred = z.argmax(1)
        ys.append(y.numpy())
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds)


# ===================== Binary metrics (Acc/Prec/Rec/F1/Spec/AUC) =====================
def _binary_metrics_from_cm(cm2x2: np.ndarray):
    tn, fp, fn, tp = cm2x2.ravel().astype(np.float64)
    acc  = (tp + tn) / max(1.0, (tp + tn + fp + fn))
    prec = tp / max(1.0, (tp + fp))
    rec  = tp / max(1.0, (tp + fn))
    f1   = (2.0 * prec * rec) / max(1e-12, (prec + rec))
    spec = tn / max(1.0, (tn + fp))
    return acc, prec, rec, f1, spec

@torch.no_grad()
def compute_test_metrics_binary(model, loader, device, temp_model=None, threshold=None, tta=False):
    # preds (for cm + thresholded metrics)
    y_true, y_pred = _collect_preds(model, loader, device, temp_model=temp_model, threshold=threshold, tta=tta)
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    # probs for AUC (does NOT use threshold)
    p1, y_true2 = _collect_probs_binary(model, loader, device, temp_model=temp_model, tta=tta)
    p1 = p1.astype(np.float64)

    if _CM_AVAILABLE:
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    else:
        cm = np.zeros((2,2), dtype=np.int64)
        for t,p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1

    acc, prec, rec, f1, spec = _binary_metrics_from_cm(cm)

    auc = None
    if _HAS_SK_METRICS and (y_true == 0).any() and (y_true == 1).any():
        try:
            auc = float(roc_auc_score(y_true, p1))
        except Exception:
            auc = None

    return {
        "acc": 100.0*acc,
        "prec": 100.0*prec,
        "rec": 100.0*rec,
        "f1": 100.0*f1,
        "spec": 100.0*spec,
        "auc": (100.0*auc if auc is not None else None),
        "y_true": y_true,
        "y_pred": y_pred,
        "cm": cm,
    }


# ===================== AdaBN =====================
@torch.no_grad()
def adabn_calibrate(model: nn.Module, loader: DataLoader, device: torch.device):
    was_training = model.training
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for x,_ in loader:
        _ = model(x.to(device, non_blocking=True))
    if not was_training:
        model.eval()


# ===================== ColorAlign stats =====================
def _estimate_tensor_stats(image_paths: List[str], roi_cfg, default_center, img_size: int, take: int=1200):
    rnd = random.Random(2025)
    samp = rnd.sample(image_paths, min(take, len(image_paths)))
    lb = Letterbox(img_size, pad_color=(114,114,114), scale_jitter=0.0)

    means, sqs = [], []
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
        x = transforms.functional.to_tensor(im)
        C = x.shape[0]
        means.append(x.view(C,-1).mean(1))
        sqs.append((x**2).view(C,-1).mean(1))

    if not means:
        mu = torch.tensor([0.485,0.456,0.406])
        sd = torch.tensor([0.229,0.224,0.225])
        return mu, sd

    mu = torch.stack(means,0).mean(0)
    ex2= torch.stack(sqs,0).mean(0)
    var= (ex2 - mu**2).clamp_min(1e-9)
    sd = var.sqrt()
    return mu, sd

def _paths(ds: PigImageFolder, indices: List[int]) -> List[str]:
    return [ds.samples[i][0] for i in indices]


# ===================== model loader (pluggable) =====================
def load_model_from_py(model_py: str, fn_name: str, num_classes: int, model_kwargs: dict):
    import importlib.util
    assert os.path.isfile(model_py), f"model_py not found: {model_py}"
    spec = importlib.util.spec_from_file_location("user_model_mod", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    assert hasattr(mod, fn_name), f"Function '{fn_name}' not found in {model_py}"
    fn = getattr(mod, fn_name)
    model = fn(num_classes=num_classes, **(model_kwargs or {}))
    assert isinstance(model, nn.Module), "build_model must return nn.Module"
    return model


# ===================== single run LOPO =====================
def run_lopo(args):
    set_seed(args.seed)

    result_dir = os.path.join(args.result_root, args.exp_name)
    model_dir  = os.path.join(args.model_root,  args.exp_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(result_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi_cfg = load_roi_cfg(args.roi_cfg)

    default_center = None if args.roi_fallback == "none" else args.roi_center
    if args.no_roi:
        default_center = None

    base = PigImageFolder(args.data_root, transform=None, roi_cfg=roi_cfg,
                          default_center=default_center, roi_jitter=0.0)
    class_names = base.classes
    num_classes = len(class_names)

    all_pigs = sorted(set(pig_id_of(p) for p,_ in base.samples))
    if len(all_pigs) == 0:
        raise RuntimeError("No pig groups were detected. Verify the required <class>/<pig_id>/<image> layout.")

    pigs_to_run = all_pigs if not args.pigs else [p for p in args.pigs.split(",") if p in all_pigs]
    if not pigs_to_run:
        raise RuntimeError("No eligible pig groups were selected. Verify --pigs and the dataset contents.")

    print(f"[DATA] classes={class_names} (K={num_classes})")
    print(f"[DATA] pigs={len(all_pigs)} → {all_pigs}")
    print(f"[RUN ] exp_name='{args.exp_name}'")
    print(f"[OUT ] result_dir={result_dir}")
    print(f"[OUT ] model_dir ={model_dir}")

    tf_train = build_transforms_train(args.aug, args.img_size)

    if args.enable_temp_scaling and args.temp_mode == "off":
        print("[TempScaling] enable_temp_scaling → temp_mode='savebest'")
        args.temp_mode = "savebest"

    ALL_Y_TRUE, ALL_Y_PRED = [], []
    fold_rows = []  # per pig metrics rows

    for fold_idx, test_pig in enumerate(pigs_to_run, start=1):
        test_idx, trainval_idx = [], []
        for i,(path,_) in enumerate(base.samples):
            (test_idx if pig_id_of(path)==test_pig else trainval_idx).append(i)

        trainval_pigs = sorted(set(pig_id_of(base.samples[i][0]) for i in trainval_idx))
        rng = random.Random(args.seed + fold_idx)
        rng.shuffle(trainval_pigs)

        n_val = max(2, int(len(trainval_pigs)*args.vr))
        n_val = min(n_val, max(1, len(trainval_pigs)-1))
        val_pigs = set(trainval_pigs[:n_val])
        train_pigs= set(trainval_pigs[n_val:])

        val_idx   = [i for i in trainval_idx if pig_id_of(base.samples[i][0]) in val_pigs]
        train_idx = [i for i in trainval_idx if pig_id_of(base.samples[i][0]) in train_pigs]

        _assert_no_leak(base, train_idx, val_idx, test_idx, test_pig)

        ds_train = PigImageFolder(args.data_root, transform=tf_train, roi_cfg=roi_cfg,
                                  default_center=default_center, roi_jitter=args.roi_jitter)

        aligner = None
        if args.align_color_to_train:
            tr_paths = _paths(base, train_idx)
            ev_paths = _paths(base, sorted(set(val_idx) | set(test_idx)))
            mu_tr, sd_tr = _estimate_tensor_stats(tr_paths, roi_cfg, default_center, args.img_size, take=args.align_sample)
            mu_ev, sd_ev = _estimate_tensor_stats(ev_paths, roi_cfg, default_center, args.img_size, take=args.align_sample)
            print(f"[ColorAlign] train μ={mu_tr.tolist()} σ={sd_tr.tolist()}")
            print(f"[ColorAlign] eval  μ={mu_ev.tolist()} σ={sd_ev.tolist()}")
            aligner = ColorAlignToTrain(mu_tr, sd_tr, mu_ev, sd_ev)

        tf_eval = build_transforms_eval(args.img_size, aligner=aligner)
        ds_eval = PigImageFolder(args.data_root, transform=tf_eval, roi_cfg=roi_cfg,
                                 default_center=default_center, roi_jitter=0.0)

        use_simple_shuffle = (args.alpha==0 and args.beta==0 and args.gamma==0)
        if use_simple_shuffle:
            train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=args.batch,
                                      shuffle=True, drop_last=True,
                                      num_workers=args.nw, pin_memory=True,
                                      worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))
        else:
            sampler = build_weighted_sampler(ds_train, train_idx, args.alpha, args.beta, args.gamma)
            train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=args.batch,
                                      sampler=sampler, drop_last=True,
                                      num_workers=args.nw, pin_memory=True,
                                      worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))

        if len(train_loader) == 0:
            raise RuntimeError(f"[Data] train_loader has 0 batches. train_idx={len(train_idx)} batch={args.batch}")

        val_loader  = DataLoader(Subset(ds_eval, val_idx), batch_size=args.batch, shuffle=False,
                                 drop_last=False, num_workers=args.nw, pin_memory=True,
                                 worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))
        test_loader = DataLoader(Subset(ds_eval, test_idx), batch_size=args.batch, shuffle=False,
                                 drop_last=False, num_workers=args.nw, pin_memory=True,
                                 worker_init_fn=worker_init_fn, persistent_workers=(args.nw>0))

        y_train = [ds_train.samples[i][1] for i in train_idx]
        cls_freq = Counter(y_train)
        ce_w = torch.tensor(
            [max(1.0, sum(cls_freq.values())/(len(cls_freq)*max(1, cls_freq.get(c,0)))) for c in range(num_classes)],
            dtype=torch.float, device=device
        )
        criterion_train = nn.CrossEntropyLoss(weight=ce_w if args.use_class_weight else None)
        criterion_eval  = nn.CrossEntropyLoss()

        model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
        model = load_model_from_py(args.model_py, args.model_fn, num_classes, model_kwargs).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        steps_per_epoch = max(1, len(train_loader))
        total_steps = steps_per_epoch * args.epochs
        if args.lr_sched == "cosine":
            warmup_steps = int(max(0.0, args.warmup_ratio) * total_steps)
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
            is_plateau = False
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3, cooldown=1)
            is_plateau = True

        ema = None
        if args.ema_decay > 0:
            ema = AveragedModel(model, avg_fn=lambda avg_p, p, n: args.ema_decay*avg_p + (1.0-args.ema_decay)*p)

        fold_tag = f"fold{fold_idx:02d}_{test_pig}"
        log_txt = os.path.join(result_dir, f"{fold_tag}.txt")
        log_csv = os.path.join(result_dir, f"{fold_tag}.csv")
        best_ckpt = os.path.join(model_dir, f"best_{test_pig}.pth")
        final_ckpt= os.path.join(model_dir, f"final_{test_pig}.pth")
        best_temp_path = os.path.join(model_dir, f"best_{test_pig}_temp.pt")

        with open(log_txt, "w", encoding="utf-8") as f:
            f.write(f"exp_name={args.exp_name}\n")
            f.write(f"fold={fold_idx}/{len(pigs_to_run)} test_pig={test_pig}\n")
            f.write(f"train_pigs={sorted(list(train_pigs))}\n")
            f.write(f"val_pigs={sorted(list(val_pigs))}\n")
            f.write(f"sizes train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}\n")

        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","val_macro_argmax","val_macro_thr","thr","lr","is_best"])

        stopper = EarlyStopper(patience=args.es_patience, mode="max")
        best_score = -float("inf")
        best_epoch = 0
        best_temp_state = None

        print(f"\n===== LOPO {fold_idx}/{len(pigs_to_run)} | test_pig={test_pig} =====")
        print(f"Sizes train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

        for ep in range(1, args.epochs+1):
            sched_for_train = None if is_plateau else scheduler

            tr_loss, tr_acc = train_epoch(
                model, train_loader, num_classes, criterion_train, device,
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, label_smoothing=args.ls,
                optimizer=optimizer, scheduler=sched_for_train, ema=ema,
                clip_grad=args.clip_grad, scheduler_is_plateau=is_plateau
            )

            eval_model = ema if ema is not None else model

            temp_for_eval = None
            if args.temp_mode == "epoch":
                temp_for_eval = fit_temperature(eval_model, val_loader, device)
            elif args.temp_mode == "savebest" and best_score > -float("inf") and best_temp_state is not None:
                temp_for_eval = _Temp().to(device)
                temp_for_eval.load_state_dict(best_temp_state)

            eval_for_val = eval_model
            if args.prior_correction == "em" and args.em_scope in ("val","both"):
                bias_val = prior_correction_em(eval_model, val_loader, device)
                eval_for_val = _WithBiasScaled(eval_model, bias_val, alpha=args.em_alpha_val).to(device)

            va_loss, va_acc, _, va_macro_argmax = evaluate(
                eval_for_val, val_loader, criterion_eval, device, class_names,
                tta_hflip=args.tta_hflip, temp_model=temp_for_eval, threshold=None
            )

            tuned_thr = None
            macro_thr = va_macro_argmax
            if num_classes == 2:
                p1_val, y_val = _collect_probs_binary(eval_for_val, val_loader, device, temp_model=temp_for_eval, tta=args.tta_hflip)
                tuned_thr, macro_thr = tune_threshold_macro_acc(p1_val, y_val, lo=args.thr_min, hi=args.thr_max, n=args.thr_steps)

            if is_plateau:
                if args.select_by == "macro":
                    scheduler.step(1.0 - macro_thr/100.0)
                else:
                    scheduler.step(va_loss)

            lr_now = optimizer.param_groups[0]["lr"]
            score_now = macro_thr if args.select_by == "macro" else (-va_loss)

            if args.temp_mode == "savebest" and score_now > best_score:
                temp_best = fit_temperature(eval_for_val, val_loader, device)
                if temp_best is not None:
                    best_temp_state = temp_best.state_dict()
                    torch.save(best_temp_state, best_temp_path)

            improved, stop = stopper.step(score_now, ep)
            is_best = False
            if improved:
                is_best = True
                best_score = score_now
                best_epoch = ep
                torch.save(model.state_dict(), best_ckpt)

            line = (f"[{ep}/{args.epochs}] "
                    f"Tr L:{tr_loss:.4f} A:{tr_acc:.2f}% | "
                    f"Va L:{va_loss:.4f} A:{va_acc:.2f}% "
                    f"Macro(argmax):{va_macro_argmax:.2f}% "
                    f"Macro@thr:{macro_thr:.2f}% thr={tuned_thr if tuned_thr is not None else 'N/A'} "
                    f"{'(best)' if is_best else ''} | LR:{lr_now:.3g}")
            print(line)

            with open(log_txt, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            with open(log_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([ep, f"{tr_loss:.6f}", f"{tr_acc:.2f}",
                            f"{va_loss:.6f}", f"{va_acc:.2f}",
                            f"{va_macro_argmax:.2f}", f"{macro_thr:.2f}",
                            f"{tuned_thr:.6f}" if tuned_thr is not None else "",
                            f"{lr_now:.8f}", int(is_best)])

            if stop:
                print(f"[EarlyStopping] epoch={ep} (best_epoch={best_epoch})")
                break

        torch.save(model.state_dict(), final_ckpt)
        print(f"[CKPT] best={best_ckpt} final={final_ckpt} best_epoch={best_epoch}")

        model.load_state_dict(torch.load(best_ckpt, map_location=device), strict=False)

        best_temp_for_test = None
        if args.temp_mode in ("epoch","savebest") and os.path.exists(best_temp_path):
            best_temp_for_test = _Temp().to(device)
            best_temp_for_test.load_state_dict(torch.load(best_temp_path, map_location=device))

        eval_for_test = model
        if args.prior_correction == "em" and args.em_scope in ("test","both"):
            bias_test = prior_correction_em(model, test_loader, device)
            eval_for_test = _WithBiasScaled(model, bias_test, alpha=args.em_alpha_test).to(device)

        final_thr = None
        if num_classes == 2 and args.tune_threshold:
            p1_val, y_val = _collect_probs_binary(eval_for_test, val_loader, device, temp_model=best_temp_for_test, tta=args.tta_hflip)
            thr_val, macro_val = tune_threshold_macro_acc(p1_val, y_val, lo=args.thr_min, hi=args.thr_max, n=args.thr_steps)
            final_thr = float(min(args.thr_max, max(args.thr_min, thr_val)))
            print(f"[ThrTune] val best thr={thr_val:.3f} -> clamp={final_thr:.3f}, macro={macro_val:.2f}%")

            if args.enable_thr_quantile_map:
                p1_test, _ = _collect_probs_binary(eval_for_test, test_loader, device, temp_model=best_temp_for_test, tta=args.tta_hflip)
                mapped_thr, q = quantile_map_threshold(final_thr, p1_val, p1_test)
                mapped_thr = float(min(args.thr_max, max(args.thr_min, mapped_thr)))
                print(f"[QMap] q≈{q:.4f} thr_test≈{mapped_thr:.3f}")
                final_thr = mapped_thr

        if args.adabn:
            print("[AdaBN] calibrating BN on test loader...")
            adabn_calibrate(eval_for_test, test_loader, device)

        # Preserve the standard accuracy and macro-accuracy evaluation output.
        te_loss, te_acc, _, te_macro = evaluate(
            eval_for_test, test_loader, criterion_eval, device, class_names,
            tta_hflip=args.tta_hflip, temp_model=best_temp_for_test, threshold=final_thr
        )

        # Binary classification metrics.
        if num_classes == 2:
            m = compute_test_metrics_binary(
                eval_for_test, test_loader, device,
                temp_model=best_temp_for_test, threshold=final_thr, tta=args.tta_hflip
            )
            auc_str = f"{m['auc']:.2f}%" if m["auc"] is not None else "N/A"
            print(f"[TEST] Acc={m['acc']:.2f}% Prec={m['prec']:.2f}% Rec={m['rec']:.2f}% "
                  f"F1={m['f1']:.2f}% Spec={m['spec']:.2f}% AUC={auc_str} "
                  f"(thr={final_thr if final_thr is not None else 'argmax'})")
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write(f"[TEST] loss={te_loss:.6f} "
                        f"acc={m['acc']:.4f} prec={m['prec']:.4f} rec={m['rec']:.4f} "
                        f"f1={m['f1']:.4f} spec={m['spec']:.4f} auc={m['auc']} "
                        f"macro={te_macro:.4f} thr={final_thr}\n")
            y_true, y_pred = m["y_true"], m["y_pred"]
        else:
            print(f"[TEST] A={te_acc:.2f}% MacroAcc={te_macro:.2f}% (thr={final_thr if final_thr is not None else 'argmax'})")
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write(f"[TEST] loss={te_loss:.6f} acc={te_acc:.2f} macro={te_macro:.2f} thr={final_thr}\n")
            y_true, y_pred = _collect_preds(eval_for_test, test_loader, device,
                                            temp_model=best_temp_for_test, threshold=final_thr, tta=args.tta_hflip)

        y_true = np.clip(y_true.astype(np.int32), 0, num_classes-1)
        y_pred = np.clip(y_pred.astype(np.int32), 0, num_classes-1)

        cm_raw = os.path.join(result_dir, f"confusion_matrix_{test_pig}.csv")
        cm_row = os.path.join(result_dir, f"confusion_matrix_{test_pig}_row_normalized.csv")
        _save_confusion_matrices(y_true, y_pred, num_classes, cm_raw, cm_row, title=f"{test_pig} (TEST)")

        ALL_Y_TRUE.extend(y_true.tolist())
        ALL_Y_PRED.extend(y_pred.tolist())

        if num_classes == 2:
            fold_rows.append({
                "pig": test_pig,
                "acc": m["acc"],
                "prec": m["prec"],
                "rec": m["rec"],
                "f1": m["f1"],
                "spec": m["spec"],
                "auc": m["auc"],
                "macro_acc": te_macro,
            })
        else:
            fold_rows.append({
                "pig": test_pig,
                "acc": te_acc,
                "prec": None,
                "rec": None,
                "f1": None,
                "spec": None,
                "auc": None,
                "macro_acc": te_macro,
            })

    # ===== summary (per pig) =====
    print("\n===== LOPO summary =====")
    if fold_rows:
        for r in fold_rows:
            if num_classes == 2:
                auc_str = f"{r['auc']:.2f}%" if r["auc"] is not None else "N/A"
                print(f"{r['pig']}: Acc={r['acc']:.2f}% Prec={r['prec']:.2f}% Rec={r['rec']:.2f}% "
                      f"F1={r['f1']:.2f}% Spec={r['spec']:.2f}% AUC={auc_str} MacroAcc={r['macro_acc']:.2f}%")
            else:
                print(f"{r['pig']}: acc={r['acc']:.2f}% macro={r['macro_acc']:.2f}%")
    else:
        print("No folds?")

    # ===== all pigs merged CM + metrics =====
    if len(ALL_Y_TRUE) > 0:
        y_true_all = np.asarray(ALL_Y_TRUE, dtype=np.int32)
        y_pred_all = np.asarray(ALL_Y_PRED, dtype=np.int32)
        out_raw = os.path.join(result_dir, "confusion_matrix_all_pigs.csv")
        out_row = os.path.join(result_dir, "confusion_matrix_all_pigs_row_normalized.csv")
        _save_confusion_matrices(y_true_all, y_pred_all, num_classes, out_raw, out_row, title="ALL PIGS (merged)")

    # ===== write summary.csv =====
    summary_path = os.path.join(result_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if num_classes == 2:
            w.writerow(["pig","acc","prec","rec","f1","spec","auc","macro_acc"])
            for r in fold_rows:
                w.writerow([
                    r["pig"],
                    f"{r['acc']:.6f}",
                    f"{r['prec']:.6f}" if r["prec"] is not None else "",
                    f"{r['rec']:.6f}"  if r["rec"]  is not None else "",
                    f"{r['f1']:.6f}"   if r["f1"]   is not None else "",
                    f"{r['spec']:.6f}" if r["spec"] is not None else "",
                    f"{r['auc']:.6f}"  if r["auc"]  is not None else "",
                    f"{r['macro_acc']:.6f}",
                ])
            # average row (ignore None auc)
            accs = [r["acc"] for r in fold_rows]
            precs= [r["prec"] for r in fold_rows if r["prec"] is not None]
            recs = [r["rec"] for r in fold_rows if r["rec"] is not None]
            f1s  = [r["f1"] for r in fold_rows if r["f1"] is not None]
            specs= [r["spec"] for r in fold_rows if r["spec"] is not None]
            aucs = [r["auc"] for r in fold_rows if r["auc"] is not None]
            macros=[r["macro_acc"] for r in fold_rows]
            w.writerow([
                "AVERAGE",
                f"{sum(accs)/len(accs):.6f}" if accs else "",
                f"{sum(precs)/len(precs):.6f}" if precs else "",
                f"{sum(recs)/len(recs):.6f}" if recs else "",
                f"{sum(f1s)/len(f1s):.6f}" if f1s else "",
                f"{sum(specs)/len(specs):.6f}" if specs else "",
                f"{sum(aucs)/len(aucs):.6f}" if aucs else "",
                f"{sum(macros)/len(macros):.6f}" if macros else "",
            ])
        else:
            w.writerow(["pig","acc","macro_acc"])
            for r in fold_rows:
                w.writerow([r["pig"], f"{r['acc']:.6f}", f"{r['macro_acc']:.6f}"])
            accs = [r["acc"] for r in fold_rows]
            macros = [r["macro_acc"] for r in fold_rows]
            w.writerow(["AVERAGE",
                        f"{sum(accs)/len(accs):.6f}" if accs else "",
                        f"{sum(macros)/len(macros):.6f}" if macros else ""])

    print(f"[SAVE] {summary_path}")


# ===================== SUITE runner for E4/E5 =====================
def _parse_kwargs(s: str) -> dict:
    return json.loads(s) if s and s.strip() else {}

def run_suite(args):
    """
    Runs all variants for E4/E5 and stores to separate roots.
    Reuse the base arguments and change only:
      - result_root/model_root (E4 vs E5)
      - exp_name (suffix)
      - model_kwargs (tap indices)

    Feature variants are expressed through ``tap_idx_z`` and ``tap_idx_y``;
    a value of ``-1`` disables the corresponding feature tap.
    """
    base_kwargs = _parse_kwargs(args.model_kwargs)

    def run_one(tag: str, result_root: str, model_root: str, patch_kwargs: dict):
        new_args = argparse.Namespace(**vars(args))
        new_args.suite = "off"  # Each generated case executes one LOPO run.
        new_args.result_root = result_root
        new_args.model_root = model_root
        new_args.exp_name = f"{args.exp_name}_{tag}"
        kw = dict(base_kwargs)
        kw.update(patch_kwargs)
        new_args.model_kwargs = json.dumps(kw, ensure_ascii=False)
        print("\n" + "="*80)
        print(f"[SUITE] {tag}")
        print(f"[SUITE] result_root={new_args.result_root}")
        print(f"[SUITE] model_root ={new_args.model_root}")
        print(f"[SUITE] exp_name   ={new_args.exp_name}")
        print(f"[SUITE] model_kwargs={new_args.model_kwargs}")
        print("="*80 + "\n")
        run_lopo(new_args)

    # Resolve experiment-specific output roots.
    r_e4 = args.suite_result_root_e4 or args.result_root
    m_e4 = args.suite_model_root_e4  or args.model_root
    r_e5 = args.suite_result_root_e5 or args.result_root
    m_e5 = args.suite_model_root_e5  or args.model_root

    if args.suite in ("E4", "E4E5"):
        # E4: deep fixed Fire9 (backbone output is Fire9 anyway), incremental add:
        # Pool3+Fire9     => mid disabled (-1)
        # Pool5+Fire9     => shallow disabled (-1)
        # Pool3+Pool5+F9  => shallow=5, mid=8
        run_one("E4_Pool3_Fire9", r_e4, m_e4, {"tap_idx_z": 5, "tap_idx_y": -1})
        run_one("E4_Pool5_Fire9", r_e4, m_e4, {"tap_idx_z": -1, "tap_idx_y": 8})
        run_one("E4_Pool3_Pool5_Fire9_Full", r_e4, m_e4, {"tap_idx_z": 5, "tap_idx_y": 8})

    if args.suite in ("E5", "E4E5"):
        # E5(I): fix mid=Pool5(8), vary shallow: Fire3(4), Fire4(6), Pool3(5)
        run_one("E5I_Fire3_Pool5_Fire9", r_e5, m_e5, {"tap_idx_z": 4, "tap_idx_y": 8})
        run_one("E5I_Fire4_Pool5_Fire9", r_e5, m_e5, {"tap_idx_z": 6, "tap_idx_y": 8})
        run_one("E5I_Pool3_Pool5_Fire9", r_e5, m_e5, {"tap_idx_z": 5, "tap_idx_y": 8})

        # E5(II): fix shallow=Pool3(5), vary mid: Fire5(7), Pool5(8), Fire6(9), Fire7(10)
        run_one("E5II_Pool3_Fire5_Fire9", r_e5, m_e5, {"tap_idx_z": 5, "tap_idx_y": 7})
        run_one("E5II_Pool3_Pool5_Fire9", r_e5, m_e5, {"tap_idx_z": 5, "tap_idx_y": 8})
        run_one("E5II_Pool3_Fire6_Fire9", r_e5, m_e5, {"tap_idx_z": 5, "tap_idx_y": 9})
        run_one("E5II_Pool3_Fire7_Fire9", r_e5, m_e5, {"tap_idx_z": 5, "tap_idx_y": 10})


# ===================== CLI =====================
def build_argparser():
    ap = argparse.ArgumentParser("Generic LOPO Trainer (pluggable model)")

    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--result_root", type=str, required=True)
    ap.add_argument("--model_root", type=str, required=True)
    ap.add_argument("--exp_name", type=str, required=True)
    ap.add_argument("--pigs", type=str, default="")

    ap.add_argument("--model_py", type=str, required=True)
    ap.add_argument("--model_fn", type=str, default="build_model")
    ap.add_argument("--model_kwargs", type=str, default="")

    ap.add_argument("--roi_cfg", type=str, default="")
    ap.add_argument("--roi_center", type=float, default=0.90)
    ap.add_argument("--roi_fallback", type=str, default="center", choices=["center","none"])
    ap.add_argument("--roi_jitter", type=float, default=0.02)
    ap.add_argument("--no_roi", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--nw", type=int, default=4)
    ap.add_argument("--vr", type=float, default=0.25)

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--aug", type=str, default="color_robust",
                    choices=["A","B","C","light","heavy","default","color_robust","pig_color"])

    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--lr_sched", type=str, default="cosine", choices=["cosine","plateau"])
    ap.add_argument("--warmup_ratio", type=float, default=0.20)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta",  type=float, default=0.7)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--mixup", type=float, default=0.05)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--ls", type=float, default=0.03)

    ap.add_argument("--select_by", type=str, default="macro", choices=["loss","macro"])
    ap.add_argument("--es_patience", type=int, default=14)

    ap.add_argument("--tta_hflip", action="store_true")
    ap.add_argument("--use_class_weight", action="store_true")

    ap.add_argument("--tune_threshold", action="store_true")
    ap.add_argument("--thr_min", type=float, default=0.25)
    ap.add_argument("--thr_max", type=float, default=0.75)
    ap.add_argument("--thr_steps", type=int, default=41)
    ap.add_argument("--enable_thr_quantile_map", action="store_true")

    ap.add_argument("--ema_decay", type=float, default=0.999)

    ap.add_argument("--temp_mode", type=str, default="savebest", choices=["off","epoch","savebest"])
    ap.add_argument("--enable_temp_scaling", action="store_true")

    ap.add_argument("--prior_correction", type=str, default="off", choices=["off","em"])
    ap.add_argument("--em_scope", type=str, default="test", choices=["off","val","test","both"])
    ap.add_argument("--em_alpha_val", type=float, default=1.0)
    ap.add_argument("--em_alpha_test", type=float, default=1.0)

    ap.add_argument("--align_color_to_train", action="store_true")
    ap.add_argument("--align_sample", type=int, default=1200)
    ap.add_argument("--adabn", action="store_true")

    # ===== suite =====
    ap.add_argument("--suite", type=str, default="off", choices=["off","E4","E5","E4E5"])
    ap.add_argument("--suite_result_root_e4", type=str, default="")
    ap.add_argument("--suite_model_root_e4", type=str, default="")
    ap.add_argument("--suite_result_root_e5", type=str, default="")
    ap.add_argument("--suite_model_root_e5", type=str, default="")

    return ap


if __name__ == "__main__":
    ap = build_argparser()
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = os.environ.get("MSFUNET_DETERMINISTIC", "0") != "1"
    torch.backends.cuda.matmul.allow_tf32 = os.environ.get("MSFUNET_DETERMINISTIC", "0") != "1"

    t0 = time.time()
    if args.suite != "off":
        run_suite(args)
    else:
        run_lopo(args)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
