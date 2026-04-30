"""
Compute model complexity: Parameters, FLOPs, FPS.
Generates a comparison table for the paper.
"""

from __future__ import annotations

import argparse
import time
import json
import os

import sys
from pathlib import Path

import torch
import pandas as pd
from thop import profile, clever_format

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attxnet.models import build_model


def measure_complexity(backbone, attention, img_size=224, device="cuda:0", n_warmup=50, n_runs=300):
    model = build_model(backbone, attention, pretrained=False).to(device).eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    flops, params = profile(model, inputs=(dummy,), verbose=False)
    flops_str, params_str = clever_format([flops, params], "%.2f")

    for _ in range(n_warmup):
        model(dummy)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        model(dummy)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    fps = n_runs / (time.time() - t0)

    return {
        "backbone": backbone,
        "attention": attention,
        "params": params,
        "params_str": params_str,
        "flops": flops,
        "flops_str": flops_str,
        "fps": round(fps, 1),
        "latency_ms": round(1000.0 / fps, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="./runs/complexity")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    configs = [
        # our lightweight models
        ("resnet18", "none"),
        ("resnet18", "cbam"),
        ("resnet18", "ca"),
        ("mobilenetv3", "none"),
        ("mobilenetv3", "cbam"),
        ("mobilenetv3", "ca"),
        ("efficientnet", "none"),
        ("efficientnet", "cbam"),
        ("efficientnet", "ca"),
    ]

    # also compare against heavier baselines (via timm names directly)
    heavy_configs = [
        ("resnet50", "none"),
        ("resnet101", "none"),
    ]

    results = []
    for backbone, attn in configs:
        print(f"Measuring {backbone} + {attn} ...")
        r = measure_complexity(backbone, attn, args.img_size, device)
        results.append(r)

    for backbone, attn in heavy_configs:
        print(f"Measuring heavy baseline {backbone} ...")
        try:
            r = measure_complexity(backbone, attn, args.img_size, device)
            results.append(r)
        except Exception as e:
            print(f"  Skipped: {e}")

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(args.output_dir, "complexity_table.csv"), index=False)

    with open(os.path.join(args.output_dir, "complexity.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
