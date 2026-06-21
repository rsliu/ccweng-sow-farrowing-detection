# Organized filename: measure_resnet18_efficiency.py
# Purpose: Measure ResNet-18 params, FLOPs, latency, FPS, and model size.
# Original source: measure.py

# measure_resnet18.py

import torch
import torch.nn as nn
import time
import os
from thop import profile
import torchvision.models as models
import argparse
import csv


# -------------------------
# ResNet-18 construction
# -------------------------
def build_resnet18(num_classes=2):
    model = models.resnet18(weights=None)  # 不用 pretrained，公平測架構本身
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# -------------------------
# Efficiency measurement
# -------------------------
def measure_resnet18(device="cuda",
                     img_size=224,
                     warmup=50,
                     iters=200,
                     use_amp=False,
                     out_dir="Result/E6_resnet18"):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = build_resnet18().to(device).eval()

    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    # -------------------------
    # Params
    # -------------------------
    params = sum(p.numel() for p in model.parameters())
    params_m = params / 1e6

    # -------------------------
    # FLOPs
    # -------------------------
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    flops_g = flops / 1e9

    # -------------------------
    # Warmup
    # -------------------------
    with torch.no_grad():
        for _ in range(warmup):
            if use_amp and device.type == "cuda":
                with torch.autocast("cuda"):
                    _ = model(dummy)
            else:
                _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # -------------------------
    # Latency
    # -------------------------
    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            if use_amp and device.type == "cuda":
                with torch.autocast("cuda"):
                    _ = model(dummy)
            else:
                _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    latency_ms = (end - start) / iters * 1000
    fps = 1000 / latency_ms

    # -------------------------
    # Model size
    # -------------------------
    tmp_path = "tmp_resnet18.pth"
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)

    # -------------------------
    # Print
    # -------------------------
    print("\n===== ResNet-18 Efficiency =====")
    print(f"Params (M)   : {params_m:.3f}")
    print(f"FLOPs (G)    : {flops_g:.3f}")
    print(f"Latency (ms) : {latency_ms:.3f}")
    print(f"FPS          : {fps:.2f}")
    print(f"Size (MB)    : {size_mb:.2f}")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "resnet18_efficiency.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "params_m", "flops_g", "latency_ms", "fps", "size_mb", "device", "img_size", "warmup", "iters", "amp"])
        writer.writerow(["resnet18", f"{params_m:.6f}", f"{flops_g:.6f}", f"{latency_ms:.6f}", f"{fps:.6f}", f"{size_mb:.6f}", str(device), img_size, warmup, iters, int(use_amp)])
    print(f"Saved       : {out_csv}")
    return params_m, flops_g, latency_ms, fps, size_mb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproducible ResNet-18 efficiency benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out_dir", default="Result/E6_resnet18")
    args = parser.parse_args()
    measure_resnet18(args.device, args.img_size, args.warmup, args.iters, args.amp, args.out_dir)
