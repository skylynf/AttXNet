"""
Grad-CAM visualization for crack detection models.
Compares baseline (no attention) vs full model (with attention) on the same images.
Generates side-by-side heatmap figures for the paper.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attxnet.models import build_model
from attxnet.dataset import get_val_transform, IMG_MEAN, IMG_STD


def denormalize(tensor, mean=IMG_MEAN, std=IMG_STD):
    """Convert normalized tensor back to [0,1] numpy image."""
    img = tensor.cpu().clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img.permute(1, 2, 0).clamp(0, 1).numpy()


def load_model(ckpt_dir: str, device: str):
    """Load model from experiment directory using saved config."""
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)
    model = build_model(cfg["backbone"], cfg["attention"], pretrained=False)
    ckpt = os.path.join(ckpt_dir, "best_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    return model, cfg


def _run_gradcam(model, target_layers, input_tensor):
    """Run GradCAM with proper gradient handling."""
    cam = GradCAM(model=model, target_layers=target_layers)
    result = cam(input_tensor=input_tensor, targets=None)
    del cam
    return result


def generate_gradcam_comparison(
    image_paths: list,
    baseline_dir: str,
    ours_dir: str,
    output_dir: str,
    device: str = "cuda:0",
    img_size: int = 224,
):
    """Generate side-by-side Grad-CAM: Original | Baseline | Ours."""
    os.makedirs(output_dir, exist_ok=True)
    transform = get_val_transform(img_size)

    model_base, cfg_base = load_model(baseline_dir, device)
    model_ours, cfg_ours = load_model(ours_dir, device)

    target_base = [model_base.get_cam_target_layer()]
    target_ours = [model_ours.get_cam_target_layer()]

    for idx, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Cannot read {img_path}, skipping.")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))
        img_float = img_resized.astype(np.float32) / 255.0

        augmented = transform(image=img_rgb)
        input_tensor = augmented["image"].unsqueeze(0).to(device)

        grayscale_base = _run_gradcam(model_base, target_base, input_tensor)
        grayscale_ours = _run_gradcam(model_ours, target_ours, input_tensor)

        vis_base = show_cam_on_image(img_float, grayscale_base[0], use_rgb=True)
        vis_ours = show_cam_on_image(img_float, grayscale_ours[0], use_rgb=True)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_float)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")

        base_label = f"Baseline ({cfg_base['backbone']})"
        axes[1].imshow(vis_base)
        axes[1].set_title(base_label, fontsize=12)
        axes[1].axis("off")

        ours_label = f"Ours ({cfg_ours['backbone']}+{cfg_ours['attention'].upper()})"
        axes[2].imshow(vis_ours)
        axes[2].set_title(ours_label, fontsize=12)
        axes[2].axis("off")

        plt.tight_layout()
        fname = f"gradcam_{idx:03d}"
        plt.savefig(os.path.join(output_dir, f"{fname}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"{fname}.pdf"))
        plt.close()
        print(f"[Saved] {fname}")


def auto_select_images(data_root: str, category: str = "D", n: int = 8, seed: int = 42):
    """Automatically select crack images for visualization."""
    import random
    random.seed(seed)
    crack_dir = os.path.join(data_root, category, f"C{category}")
    files = [os.path.join(crack_dir, f) for f in os.listdir(crack_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)
    return files[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--baseline_dir", type=str, required=True,
                   help="Path to baseline experiment (e.g., runs/exp1_mobilenetv3_baseline)")
    p.add_argument("--ours_dir", type=str, required=True,
                   help="Path to full model experiment (e.g., runs/exp2_ablation_full_cbam)")
    p.add_argument("--output_dir", type=str, default="./runs/figures/gradcam")
    p.add_argument("--category", type=str, default="D")
    p.add_argument("--n_images", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--image_paths", type=str, nargs="*", default=None,
                   help="Specific image paths (auto-select if not given)")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    if args.image_paths:
        img_paths = args.image_paths
    else:
        img_paths = auto_select_images(args.data_root, args.category, args.n_images)

    print(f"Generating Grad-CAM for {len(img_paths)} images ...")
    generate_gradcam_comparison(
        img_paths, args.baseline_dir, args.ours_dir,
        args.output_dir, device, args.img_size,
    )
    print(f"[Done] Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
