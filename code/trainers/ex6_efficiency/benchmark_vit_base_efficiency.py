# Organized filename: benchmark_vit_base_efficiency.py
# Purpose: Efficiency benchmark for ViT-B.
# Original source: train_mo_1pig_msfrt_fps.py

# -*- coding: utf-8 -*-
import time
import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from tqdm import tqdm

from transformers import ViTForImageClassification
from transformers.modeling_outputs import SequenceClassifierOutput

# ===================== 記錄訓練次數 =====================
RUN_COUNT_FILE = "run_count.txt"

def get_run_count():
    if os.path.exists(RUN_COUNT_FILE):
        with open(RUN_COUNT_FILE, "r", encoding="utf-8") as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0

def update_run_count(count: int):
    with open(RUN_COUNT_FILE, "w", encoding="utf-8") as f:
        f.write(str(int(count)))

# ===================== AMP 相容 =====================
try:
    from torch.cuda.amp import autocast, GradScaler  # 舊版最穩
    AMP_NEW_API = False
except Exception:
    from torch.amp import autocast, GradScaler       # 新版 torch
    AMP_NEW_API = True

# ===================== Repro =====================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

def worker_init_fn(worker_id):
    s = torch.initial_seed() % (2**32)
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)

# ===================== Stratified split (train/val/test) =====================
def stratified_split_indices(y: np.ndarray, val_ratio: float, test_ratio: float, seed: int):
    """
    y: (N,)
    return: train_idx, val_idx, test_idx (list of int)
    """
    assert 0 < val_ratio < 1
    assert 0 < test_ratio < 1
    assert (val_ratio + test_ratio) < 1.0

    N = len(y)
    idx_all = np.arange(N)

    rng = np.random.RandomState(seed)

    # 先切 test
    test_idx = []
    remain_idx = []

    for c in np.unique(y):
        idx_c = idx_all[y == c]
        rng.shuffle(idx_c)
        n_test = max(1, int(round(len(idx_c) * test_ratio)))
        test_idx.extend(idx_c[:n_test].tolist())
        remain_idx.extend(idx_c[n_test:].tolist())

    remain_idx = np.array(remain_idx, dtype=int)

    # 再從 remain 切 val（比例是針對 remain 的相對比例）
    y_remain = y[remain_idx]
    val_idx = []
    train_idx = []

    for c in np.unique(y_remain):
        idx_c = remain_idx[y_remain == c]
        rng.shuffle(idx_c)
        n_val = max(1, int(round(len(idx_c) * val_ratio / (1.0 - test_ratio))))
        val_idx.extend(idx_c[:n_val].tolist())
        train_idx.extend(idx_c[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx

# ===================== ViT 模型定義 =====================
class CustomViTForImageClassification(ViTForImageClassification):
    """
    HF ViTForImageClassification wrapper:
    - output_hidden_states=True
    - select multiple layers, mean-pool tokens, concat, then custom classifier
    """
    def __init__(self, config, layer_indices):
        super().__init__(config)
        self.layer_indices = list(layer_indices)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states  # tuple length = num_hidden_layers + 1

        L = len(hidden_states)
        idxs = []
        for i in self.layer_indices:
            ii = int(i)
            if ii < 0:
                ii = L + ii
            ii = max(0, min(L - 1, ii))
            idxs.append(ii)

        selected = [hidden_states[i] for i in idxs]                 # (B, N, D)
        concatenated = torch.cat([h.mean(dim=1) for h in selected], dim=1)  # (B, K*D)

        logits = self.classifier(concatenated)
        return SequenceClassifierOutput(logits=logits)

# ===================== 訓練/評估 =====================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler: GradScaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = (device.type == "cuda") and (scaler is not None) and scaler.is_enabled()

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            if AMP_NEW_API:
                with autocast("cuda", enabled=True):
                    logits = model(images).logits
                    loss = criterion(logits, labels)
            else:
                with autocast(enabled=True):
                    logits = model(images).logits
                    loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item())
        pred = logits.argmax(1)
        correct += int((pred == labels).sum().item())
        total += int(labels.size(0))

    return running_loss / max(1, len(loader)), 100.0 * correct / max(1, total)

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)

    use_amp = (device.type == "cuda")

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            if AMP_NEW_API:
                with autocast("cuda", enabled=True):
                    logits = model(images).logits
                    loss = criterion(logits, labels)
            else:
                with autocast(enabled=True):
                    logits = model(images).logits
                    loss = criterion(logits, labels)
        else:
            logits = model(images).logits
            loss = criterion(logits, labels)

        running_loss += float(loss.item())
        pred = logits.argmax(1)

        correct += int((pred == labels).sum().item())
        total += int(labels.size(0))

        for y, p in zip(labels.tolist(), pred.tolist()):
            class_total[y] += 1
            if p == y:
                class_correct[y] += 1

    class_acc = {
        class_names[i]: (100.0 * class_correct[i] / class_total[i]) if class_total[i] else 0.0
        for i in range(len(class_names))
    }
    return running_loss / max(1, len(loader)), 100.0 * correct / max(1, total), class_acc

# ===================== 模型建構 =====================
def setup_model(num_classes, freeze_option, base_ckpt="google/vit-base-patch16-224-in21k"):
    """
    freeze_option:
      0: no freeze
      1: freeze all except classifier
      2: freeze backbone then unfreeze last 4 blocks + layernorm + classifier
    """
    total_layers = 12  # vit-base
    layer_indices = sorted({
        0,                          # embedding output
        1 + total_layers // 5,
        1 + 2 * total_layers // 5,
        1 + 3 * total_layers // 5,
        1 + total_layers            # last layer output
    })

    model = CustomViTForImageClassification.from_pretrained(
        base_ckpt,
        num_labels=num_classes,
        layer_indices=layer_indices,
        ignore_mismatched_sizes=True
    )

    hidden_dim = int(model.config.hidden_size)
    input_dim = len(layer_indices) * hidden_dim

    model.classifier = nn.Sequential(
        nn.Linear(input_dim, 3072), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(3072, 1536), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(1536, num_classes)
    )

    if freeze_option == 1:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True

    elif freeze_option == 2:
        for n, p in model.named_parameters():
            if n.startswith("vit."):
                p.requires_grad = False

        unfreeze_last_n = 4
        for i in range(total_layers - unfreeze_last_n, total_layers):
            for p in model.vit.encoder.layer[i].parameters():
                p.requires_grad = True

        for p in model.vit.layernorm.parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True

    return model, layer_indices

# ===================== Params / Latency / FPS =====================
def count_params_m(model: nn.Module, trainable_only: bool = True) -> float:
    if trainable_only:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        n = sum(p.numel() for p in model.parameters())
    return float(n) / 1e6

@torch.no_grad()
def measure_latency_fps(model: nn.Module, device: torch.device, img_size: int,
                        warmup: int = 30, iters: int = 200) -> tuple:
    """
    returns: (latency_ms_per_img, fps)
    """
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # warmup
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

# ===================== 主訓練流程（full/ exposed+not_exposed） =====================
def Model(args, runs, PATH_D, PATH_R, PATH_M, device):
    os.makedirs(PATH_R, exist_ok=True)
    os.makedirs(PATH_M, exist_ok=True)

    output = os.path.join(PATH_R, f"MHvit{runs}_result.txt")

    transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.0, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # base dataset for split
    base_ds = datasets.ImageFolder(root=PATH_D, transform=None)
    class_names = base_ds.classes
    if len(class_names) != 2:
        raise ValueError(f"Binary only expected, got classes={class_names}")

    y = np.array([label for _, label in base_ds.samples], dtype=np.int64)

    train_idx, val_idx, test_idx = stratified_split_indices(
        y=y, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    # apply transforms per split
    train_ds = datasets.ImageFolder(root=PATH_D, transform=transform_train)
    eval_ds  = datasets.ImageFolder(root=PATH_D, transform=transform_eval)

    train_subset = Subset(train_ds, train_idx)
    val_subset   = Subset(eval_ds,  val_idx)
    test_subset  = Subset(eval_ds,  test_idx)

    train_loader = DataLoader(
        train_subset, batch_size=args.b, shuffle=True,
        num_workers=args.nw, pin_memory=True,
        worker_init_fn=worker_init_fn, persistent_workers=(args.nw > 0)
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.b, shuffle=False,
        num_workers=args.nw, pin_memory=True,
        worker_init_fn=worker_init_fn, persistent_workers=(args.nw > 0)
    )
    test_loader = DataLoader(
        test_subset, batch_size=args.b, shuffle=False,
        num_workers=args.nw, pin_memory=True,
        worker_init_fn=worker_init_fn, persistent_workers=(args.nw > 0)
    )

    num_classes = len(class_names)
    model, layer_indices = setup_model(num_classes, args.f, base_ckpt=args.ckpt)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.wd
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    use_amp = (device.type == "cuda")
    if AMP_NEW_API:
        scaler = GradScaler("cuda", enabled=use_amp)
    else:
        scaler = GradScaler(enabled=use_amp)

    best_val_acc = -1.0
    best_val_epoch = -1
    max_test_acc = 0.0
    sum_test_acc = 0.0

    best_path = os.path.join(PATH_M, f"MHvit{runs}_bestval.pth")
    last_path = os.path.join(PATH_M, f"MHvit{runs}.pth")

    # ===== log header =====
    with open(output, "w", encoding="utf-8") as f:
        f.write(f"PATH_D = {PATH_D}\n")
        f.write(f"Classes = {class_names}\n")
        f.write(f"Split ratios: val_ratio={args.val_ratio}, test_ratio={args.test_ratio}\n")
        f.write(f"Sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}\n\n")
        f.write(f"Epoch: {args.e}, Batch: {args.b}, Freeze: {args.f}\n")
        f.write(f"Backbone: {args.ckpt}\n")
        f.write(f"Layer indices: {layer_indices}\n")
        f.write(f"LR: {args.lr}, WD: {args.wd}, Seed: {args.seed}\n")
        f.write(f"AMP(cuda only): {use_amp} | AMP_NEW_API={AMP_NEW_API}\n\n")

    # ===== train loop =====
    for epoch in range(args.e):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)

        val_loss, val_acc, val_cls_acc = eval_one_epoch(model, val_loader, criterion, device, class_names)
        test_loss, test_acc, test_cls_acc = eval_one_epoch(model, test_loader, criterion, device, class_names)

        scheduler.step(val_loss)

        sum_test_acc += test_acc
        if test_acc > max_test_acc:
            max_test_acc = test_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)

        lr_now = optimizer.param_groups[0]["lr"]

        with open(output, "a", encoding="utf-8") as f:
            f.write(f"[{epoch+1}/{args.e}] "
                    f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                    f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%\n")
            f.write(f"LR: {lr_now:.16f}\n")
            f.write("Val each accuracy:\n" +
                    ", ".join(f"c{name}: {acc:.2f}%" for name, acc in val_cls_acc.items()) + "\n")
            f.write("Test each accuracy:\n" +
                    ", ".join(f"c{name}: {acc:.2f}%" for name, acc in test_cls_acc.items()) + "\n\n")

        print(f"[{epoch+1}/{args.e}] "
              f"Train {tr_acc:.2f}% | Val {val_acc:.2f}% | Test {test_acc:.2f}% | "
              f"LR={lr_now:.3e}")

        torch.cuda.empty_cache()

    # save final
    torch.save(model.state_dict(), last_path)

    # ===== Params / Latency / FPS =====
    params_m_trainable = count_params_m(model, trainable_only=True)
    params_m_all = count_params_m(model, trainable_only=False)
    latency_ms, fps = measure_latency_fps(model, device, args.img_size, warmup=args.lat_warmup, iters=args.lat_iters)

    with open(output, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Best_Val_Acc = {best_val_acc:.2f} (epoch {best_val_epoch})\n")
        f.write(f"Max_Test_Acc = {max_test_acc:.2f}\n")
        f.write(f"Avg_Test_Acc = {sum_test_acc / max(1, args.e):.2f}\n")
        f.write("\n[MODEL STATS]\n")
        f.write(f"Params (trainable) = {params_m_trainable:.3f} M\n")
        f.write(f"Params (all)       = {params_m_all:.3f} M\n")
        f.write(f"Inference latency  = {latency_ms:.3f} ms/img\n")
        f.write(f"Throughput         = {fps:.3f} FPS\n")
        f.write(f"Saved best: {best_path}\n")
        f.write(f"Saved last: {last_path}\n")
        f.write("=" * 60 + "\n")

    print("\n" + "=" * 60)
    print(f"Best_Val_Acc = {best_val_acc:.2f} (epoch {best_val_epoch})")
    print(f"Max_Test_Acc = {max_test_acc:.2f}")
    print(f"Avg_Test_Acc = {sum_test_acc / max(1, args.e):.2f}")
    print("\n[MODEL STATS]")
    print(f"Parameter Count (trainable) = {params_m_trainable:.3f} M")
    print(f"Parameter Count (all)       = {params_m_all:.3f} M")
    print(f"Inference Time (ms/img)     = {latency_ms:.3f}")
    print(f"Throughput (FPS)            = {fps:.3f}")
    print(f"[SAVED] {output}")
    print(f"[SAVED] {best_path}")
    print(f"[SAVED] {last_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MHViT on ImageFolder (exposed/not_exposed) with internal split")
    parser.add_argument("--data_dir", default="Dataset/full", help="Flat binary ImageFolder dataset")
    parser.add_argument("--out_dir", default="Result/E6_vit_base", help="Result directory")
    parser.add_argument("--model_dir", default="Model/E6_vit_base", help="Checkpoint directory")
    parser.add_argument("--run_id", type=int, default=1, help="Stable output suffix; avoids mutable global run_count")
    parser.add_argument("-e", type=int, default=2, help="Epoch")
    parser.add_argument("-b", type=int, default=4, help="Batch")
    parser.add_argument("-f", type=int, default=0, help="freeze: 0 none, 1 head-only, 2 last-blocks+head")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--nw", type=int, default=4)

    parser.add_argument("--val_ratio", type=float, default=0.25, help="val ratio (stratified)")
    parser.add_argument("--test_ratio", type=float, default=0.20, help="test ratio (stratified)")

    parser.add_argument("--ckpt", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    # latency benchmark args
    parser.add_argument("--lat_warmup", type=int, default=30)
    parser.add_argument("--lat_iters", type=int, default=200)

    args = parser.parse_args()

    PATH_D = args.data_dir
    PATH_R = args.out_dir
    PATH_M = args.model_dir
    runs = args.run_id
    print(f"\033[33m{PATH_D}\033[0m")

    set_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    Model(args, runs, PATH_D, PATH_R, PATH_M, device)
    end_time = time.time() - start_time
    print(f"Total training time: {end_time // 60:.0f} mins {end_time % 60:.0f} seconds")
