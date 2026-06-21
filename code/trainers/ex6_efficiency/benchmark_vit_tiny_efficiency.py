# Organized filename: benchmark_vit_tiny_efficiency.py
# Purpose: K-fold efficiency benchmark for ViT-Tiny.
# Original source: train_mo_1pig_vit_timm_fps.py

# -*- coding: utf-8 -*-
"""
kfold_vit_timm_benchmark.py

Unified K-Fold benchmark for timm ViT (ImageFolder version)

Dataset structure:
Dataset/full/
  exposed/*.jpg
  not_exposed/*.jpg

Metrics:
- Accuracy / Precision / Recall / F1 / Specificity / AUC
- Parameter Count (M)
- Inference Time (ms/img)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import timm


# ---------------- Repro ----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- Model ----------------
class MixedViT(nn.Module):
    def __init__(self, timm_name, num_classes, layer_indices):
        super().__init__()
        self.backbone = timm.create_model(
            timm_name, pretrained=True, num_classes=0, global_pool=""
        )
        self.embed_dim = self.backbone.embed_dim
        self.layer_indices = sorted(set(layer_indices))
        self._feats = []

        def hook(_, __, output):
            self._feats.append(output)

        for i in self.layer_indices:
            self.backbone.blocks[i].register_forward_hook(hook)

        self.head = nn.Linear(len(self.layer_indices) * self.embed_dim, num_classes)

    def forward(self, x):
        self._feats = []
        _ = self.backbone(x)
        pooled = [f.mean(dim=1) for f in self._feats]
        feat = torch.cat(pooled, dim=1)
        return self.head(feat)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# ---------------- Inference Benchmark ----------------
@torch.no_grad()
def measure_latency(model, device, img_size, warmup=30, iters=200):
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

    dt = (t1 - t0) / iters
    return 1000 * dt, 1.0 / dt


# ---------------- Main ----------------
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    ds = datasets.ImageFolder(args.data_dir, transform=tf)
    X = np.arange(len(ds))
    y = np.array([label for _, label in ds.samples])

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    rows = []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        print(f"\n===== Fold {fold}/{args.k} =====")

        train_loader = DataLoader(
            Subset(ds, tr),
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.nw,
            pin_memory=True
        )
        test_loader = DataLoader(
            Subset(ds, te),
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.nw,
            pin_memory=True
        )

        model = MixedViT(
            args.timm_name,
            num_classes=2,
            layer_indices=args.layers
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        criterion = nn.CrossEntropyLoss()

        # ---- train ----
        model.train()
        for ep in range(args.epochs):
            for x, yb in train_loader:
                x, yb = x.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), yb)
                loss.backward()
                optimizer.step()

        # ---- eval ----
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, yb in test_loader:
                x = x.to(device)
                prob = torch.softmax(model(x), dim=1)[:, 1]
                ys.append(yb.numpy())
                ps.append(prob.cpu().numpy())

        y_true = np.concatenate(ys)
        y_prob = np.concatenate(ps)
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)

        params_m = count_params_m(model)
        latency, fps = measure_latency(model, device, args.img_size)

        print(f"[Fold {fold}] Params={params_m:.2f}M | Lat={latency:.2f}ms | FPS={fps:.1f}")

        rows.append([fold, params_m, latency, fps, acc, prec, rec, f1, auc])

    # ---- save ----
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "kfold_vit_benchmark.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold","params(M)","latency(ms)","fps","acc","precision","recall","f1","auc"])
        for r in rows:
            w.writerow(r)

    print(f"\n[SAVED] {out_csv}")


# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="./Result/kfold_vit")
    ap.add_argument("--timm_name", default="vit_tiny_patch16_224")
    ap.add_argument("--layers", type=int, nargs="+", default=[0,6,11])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--nw", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    main(args)
