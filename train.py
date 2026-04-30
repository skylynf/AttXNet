"""
Training & evaluation engine for bridge crack classification.
Supports: multiple backbones, attention modules, loss functions, augmentation strategies.
Logs to TensorBoard and saves checkpoints + metrics JSON.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from tqdm import tqdm

from attxnet.dataset import build_dataloaders
from attxnet.models import build_model
from attxnet.losses import build_loss


def parse_args():
    p = argparse.ArgumentParser(description="Train crack classifier")
    # data
    p.add_argument("--data_root", type=str, required=True,
                   help="Path to SDNET2018 root (containing D/, P/, W/)")
    p.add_argument("--categories", type=str, default="D",
                   help="Comma-separated categories: D,P,W")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    # model
    p.add_argument("--backbone", type=str, default="mobilenetv3",
                   choices=["resnet18", "mobilenetv3", "efficientnet"])
    p.add_argument("--attention", type=str, default="cbam",
                   choices=["cbam", "ca", "none"])
    p.add_argument("--pretrained", action="store_true", default=True)
    # loss
    p.add_argument("--loss", type=str, default="focal", choices=["focal", "ce", "wce"],
                   help="ce=CrossEntropy; wce=class-weighted CE (inverse freq., use with --no_weighted_sampler)")
    p.add_argument("--focal_alpha", type=float, default=0.75)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    # augmentation
    p.add_argument("--use_robust_aug", action="store_true", default=False,
                   help="Use inspection-scene robust augmentation")
    p.add_argument("--no_weighted_sampler", action="store_true", default=False,
                   help="Disable WeightedRandomSampler (recommended when using Focal Loss)")
    # training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    # cross-validation (stratified K-fold; fixed cv_seed ⇒ same folds across runs)
    p.add_argument("--fold", type=int, default=None,
                   help="0-based stratified CV fold index. When set, data split follows K-fold (see --n_folds).")
    p.add_argument("--n_folds", type=int, default=5, help="Number of folds when --fold is set.")
    p.add_argument("--cv_seed", type=int, default=42,
                   help="Random seed for stratified K-fold splitter (reuse across experiments for pairing).")
    # output
    p.add_argument("--output_dir", type=str, default="./runs")
    p.add_argument("--exp_name", type=str, default=None,
                   help="Experiment name (auto-generated if not set)")
    return p.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return {
        "loss": epoch_loss, "accuracy": acc,
        "precision": prec, "recall": rec, "f1": f1,
        "confusion_matrix": cm.tolist(),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    categories = tuple(args.categories.upper().split(","))
    args.use_weighted_sampler = not args.no_weighted_sampler

    if args.exp_name is None:
        aug_tag = "robust" if args.use_robust_aug else "std"
        args.exp_name = f"{args.backbone}_{args.attention}_{args.loss}_{aug_tag}"

    if args.fold is not None:
        exp_dir = os.path.join(args.output_dir, args.exp_name, f"fold_{args.fold}")
    else:
        exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, "tb_logs"))

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Config] {args.exp_name} | device={device}")

    loader_kw = dict(
        data_root=args.data_root,
        categories=categories,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_robust_aug=args.use_robust_aug,
        use_weighted_sampler=args.use_weighted_sampler,
        seed=args.seed,
    )
    if args.fold is not None:
        loader_kw.update(
            fold=args.fold,
            n_folds=args.n_folds,
            cv_seed=args.cv_seed,
        )
    loaders, ds_info = build_dataloaders(**loader_kw)

    model = build_model(args.backbone, args.attention, args.pretrained).to(device)
    print(f"[Model] {model}")

    cw = None
    if args.loss == "wce":
        cw = ds_info["class_weights"].to(device)

    criterion = build_loss(args.loss, args.focal_alpha, args.focal_gamma, class_weights=cw)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # warmup
    warmup_scheduler = None
    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs
        )

    best_f1 = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_precision": [], "val_recall": [], "val_f1": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device, epoch)
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        elapsed = time.time() - t0

        if epoch <= args.warmup_epochs and warmup_scheduler is not None:
            warmup_scheduler.step()
        else:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])

        writer.add_scalars("Loss", {"train": train_loss, "val": val_metrics["loss"]}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_metrics["accuracy"]}, epoch)
        writer.add_scalar("Val/Precision", val_metrics["precision"], epoch)
        writer.add_scalar("Val/Recall", val_metrics["recall"], epoch)
        writer.add_scalar("Val/F1", val_metrics["f1"], epoch)
        writer.add_scalar("LR", lr_now, epoch)

        is_best = val_metrics["f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train L={train_loss:.4f} A={train_acc:.4f} | "
              f"Val L={val_metrics['loss']:.4f} A={val_metrics['accuracy']:.4f} "
              f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} "
              f"F1={val_metrics['f1']:.4f} | LR={lr_now:.6f} | {elapsed:.1f}s"
              + (" *best*" if is_best else ""))

    torch.save(model.state_dict(), os.path.join(exp_dir, "last_model.pth"))

    # ─── Final test evaluation ───
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pth")))
    test_metrics = evaluate(model, loaders["test"], criterion, device)
    print("\n" + "=" * 60)
    print(f"[TEST] Acc={test_metrics['accuracy']:.4f}  P={test_metrics['precision']:.4f}  "
          f"R={test_metrics['recall']:.4f}  F1={test_metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # measure inference speed
    model.eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    # warmup
    for _ in range(20):
        model(dummy)
    torch.cuda.synchronize()
    t0 = time.time()
    n_runs = 200
    for _ in range(n_runs):
        model(dummy)
    torch.cuda.synchronize()
    fps = n_runs / (time.time() - t0)
    test_metrics["fps"] = fps
    test_metrics["inference_ms"] = 1000.0 / fps
    print(f"[Speed] FPS={fps:.1f}  Latency={1000.0/fps:.2f}ms")

    results = {
        "config": vars(args),
        "best_val_f1": best_f1,
        "test_metrics": test_metrics,
        "history": history,
        "cv": None
        if args.fold is None
        else {
            "fold": args.fold,
            "n_folds": args.n_folds,
            "cv_seed": args.cv_seed,
        },
    }
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    writer.close()
    print(f"\n[Done] Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
