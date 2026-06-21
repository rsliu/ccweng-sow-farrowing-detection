# Organized filename: benchmark_cnn_models_efficiency.py
# Purpose: K-fold efficiency benchmark for SqNet, MSANet, and MSFUNet.
# Original source: train_mo_1pig_sque_fps.py

import pathlib as _pathlib
import sys as _sys
_PROJECT_ROOT = _pathlib.Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

# -*- coding: utf-8 -*-
"""
kfold_squeezenet_benchmark.py

Unified K-Fold benchmark for SqueezeNet-family models (ImageFolder version)

Dataset structure:
Dataset/full/
  exposed/*.jpg
  not_exposed/*.jpg

Metrics:
- Accuracy / Precision / Recall / F1 / Specificity / AUC
- Parameter Count (M)
- Inference Time (ms/img)  [batch=1, warmup=30, iters=200, cuda sync]
- Throughput (FPS)
"""

import os, time, argparse, random, csv
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

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

# Vanilla squeezenet (torchvision)
from torchvision.models import squeezenet1_1


# ---------------- Repro ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# ---------------- Transforms ----------------
def build_tf(img_size: int):
    # Preserve the preprocessing used by the original efficiency benchmark.
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


# ---------------- Model builders ----------------
def build_vanilla_squeezenet(num_classes: int = 2) -> nn.Module:
    m = squeezenet1_1(weights=None)  # if you want pretrained, set weights="DEFAULT"
    m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    m.num_classes = num_classes
    return m


def build_model(args) -> nn.Module:
    if args.model == "squeezenet":
        return build_vanilla_squeezenet(num_classes=2)
    elif args.model == "msanet35":
        return MSA_Addition_Pool35(num_classes=2)
    elif args.model == "msanet53":
        return MSA_Addition_Pool53(num_classes=2)
    elif args.model == "msfu":
        return SqueezeNetWithMSFU(
            num_classes=2,
            tap_idx_z=args.tap_idx_z,
            tap_idx_y=args.tap_idx_y,
            topk_ratio=args.topk_ratio,
            style_p=args.style_p,
            style_alpha=args.style_alpha,
            use_style_norm=(not args.no_style_norm),
            msfu_bg_scale=args.msfu_bg_scale,
            msfu_init_gamma=args.msfu_init_gamma,
            softk_tau=args.softk_tau,
            softk_alpha=args.softk_alpha,
            use_coord_score=(not args.no_coord_score),
            use_local_refine=(not args.no_local_refine),
            pool_type=args.pool_type
        )
    else:
        raise ValueError(f"Unknown --model {args.model}")


def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


# ---------------- Inference Benchmark (Unified) ----------------
@torch.no_grad()
def measure_latency(model: nn.Module, device: torch.device, img_size: int,
                    warmup: int = 30, iters: int = 200):
    """
    Unified latency measurement:
    - batch=1 synthetic tensor
    - exclude dataloader/preprocess
    - CUDA synchronize
    """
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


# ---------------- Train / Eval ----------------
def train_one_epoch(model, loader, device, criterion, optimizer, clip_grad: float = 0.0):
    model.train()
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        # torchvision squeezenet returns [B, C, 1, 1] sometimes (if classifier conv)
        if logits.ndim == 4:
            logits = logits.squeeze(-1).squeeze(-1)

        loss = criterion(logits, y)
        loss.backward()
        if clip_grad and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        loss_sum += float(loss.item())

    return loss_sum / max(1, len(loader))


@torch.no_grad()
def eval_probs(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        if logits.ndim == 4:
            logits = logits.squeeze(-1).squeeze(-1)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
    if not ys:
        return np.array([]), np.array([])
    return np.concatenate(ys), np.concatenate(ps)


def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    except Exception:
        auc = float("nan")

    return {
        "cm": cm, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "acc": acc, "precision": prec, "recall": rec, "f1": f1,
        "specificity": spec, "auc": auc
    }


# ---------------- Main ----------------
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    tf = build_tf(args.img_size)
    ds = datasets.ImageFolder(args.data_dir, transform=tf)

    if len(ds.classes) != 2:
        raise ValueError(f"Binary only. Got classes={ds.classes}")

    X = np.arange(len(ds))
    y = np.array([label for _, label in ds.samples], dtype=np.int64)

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    rows = []
    all_true, all_prob = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        print(f"\n===== Fold {fold}/{args.k} =====")

        train_loader = DataLoader(
            Subset(ds, tr),
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.nw,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=(args.nw > 0)
        )
        test_loader = DataLoader(
            Subset(ds, te),
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.nw,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            persistent_workers=(args.nw > 0)
        )

        model = build_model(args).to(device)

        params_m = count_params_m(model)
        print(f"[Model] {args.model} | Params={params_m:.3f} M")

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        criterion = nn.CrossEntropyLoss()

        # ---- train ----
        for ep in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, device, criterion, optimizer, clip_grad=args.clip_grad)
            if ep == 1 or ep == args.epochs or (ep % args.print_every == 0):
                y_true, y_prob = eval_probs(model, test_loader, device)
                m = compute_metrics(y_true, y_prob, thr=args.thr) if len(y_true) else None
                if m:
                    print(f"[{ep:03d}/{args.epochs}] loss={tr_loss:.4f} | "
                          f"Acc={m['acc']*100:.2f}% F1={m['f1']*100:.2f}% AUC={m['auc']:.4f}")

        # ---- final eval ----
        y_true, y_prob = eval_probs(model, test_loader, device)
        m = compute_metrics(y_true, y_prob, thr=args.thr)

        # ---- unified latency ----
        latency_ms, fps = measure_latency(model, device, args.img_size, warmup=args.lat_warmup, iters=args.lat_iters)

        print(f"[Fold {fold}] Acc={m['acc']*100:.2f}% | F1={m['f1']*100:.2f}% | "
              f"Params={params_m:.3f}M | Lat={latency_ms:.2f}ms | FPS={fps:.2f}")

        rows.append([
            fold, params_m, latency_ms, fps,
            m["acc"], m["precision"], m["recall"], m["f1"], m["specificity"], m["auc"]
        ])

        all_true.append(y_true)
        all_prob.append(y_prob)

    # ---- overall ----
    y_all = np.concatenate(all_true) if all_true else np.array([])
    p_all = np.concatenate(all_prob) if all_prob else np.array([])
    m_all = compute_metrics(y_all, p_all, thr=args.thr) if len(y_all) else None

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"kfold_{args.model}_benchmark.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold","params(M)","latency(ms/img)","fps",
                    "acc","precision","recall","f1","specificity","auc"])
        for r in rows:
            w.writerow([
                r[0],
                f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}",
                f"{r[4]:.6f}", f"{r[5]:.6f}", f"{r[6]:.6f}",
                f"{r[7]:.6f}", f"{r[8]:.6f}", f"{r[9]:.6f}"
            ])
        if m_all:
            w.writerow([])
            w.writerow(["ALL","","","","","","","","",""])
            w.writerow(["ALL",
                        "", "", "",
                        f"{m_all['acc']:.6f}", f"{m_all['precision']:.6f}", f"{m_all['recall']:.6f}",
                        f"{m_all['f1']:.6f}", f"{m_all['specificity']:.6f}", f"{m_all['auc']:.6f}"])

    print(f"\n[SAVED] {out_csv}")
    if m_all:
        print(f"[OVERALL] Acc={m_all['acc']*100:.2f}% | F1={m_all['f1']*100:.2f}% | "
              f"Spec={m_all['specificity']*100:.2f}% | AUC={m_all['auc']:.4f}")


# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("K-Fold benchmark for SqueezeNet-family (ImageFolder)")

    ap.add_argument("--data_dir", required=True, help="Dataset/full")
    ap.add_argument("--out_dir", default="./Result/kfold_squeezenet")

    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--nw", type=int, default=4)

    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--print_every", type=int, default=5)

    # unified latency settings
    ap.add_argument("--lat_warmup", type=int, default=30)
    ap.add_argument("--lat_iters", type=int, default=200)

    # route
    ap.add_argument("--model", type=str, default="msfu",
                    choices=["squeezenet", "msanet35", "msanet53", "msfu"])

    # MSFUNet architecture parameters used by the comparison benchmark.
    ap.add_argument("--topk_ratio", type=float, default=0.0)
    ap.add_argument("--style_p", type=float, default=0.0)
    ap.add_argument("--style_alpha", type=float, default=0.0)
    ap.add_argument("--msfu_init_gamma", type=float, default=0.05)
    ap.add_argument("--msfu_bg_scale", type=float, default=0.0)
    ap.add_argument("--tap_idx_z", type=int, default=5)
    ap.add_argument("--tap_idx_y", type=int, default=8)
    ap.add_argument("--no_style_norm", action="store_true")
    ap.add_argument("--pool_type", type=str, default="guided", choices=["guided", "gap", "attn", "gem"])
    ap.add_argument("--softk_tau", type=float, default=0.5)
    ap.add_argument("--softk_alpha", type=float, default=1.0)
    ap.add_argument("--no_coord_score", action="store_true")
    ap.add_argument("--no_local_refine", action="store_true")

    args = ap.parse_args()
    main(args)
