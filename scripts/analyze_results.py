"""
Aggregate experiment results and generate publication-quality tables and figures.
Reads results.json from each experiment run directory.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_all_results(runs_dir: str):
    """Load results.json from all experiment sub-directories."""
    records = {}
    for d in sorted(Path(runs_dir).iterdir()):
        rfile = d / "results.json"
        if rfile.exists():
            with open(rfile) as f:
                records[d.name] = json.load(f)
    return records


# ─── Table 1: Backbone Comparison ───

def make_backbone_table(records, output_dir):
    rows = []
    for name in sorted(records):
        if not name.startswith("exp1_"):
            continue
        r = records[name]
        tm = r["test_metrics"]
        cfg = r["config"]
        rows.append({
            "Backbone": cfg["backbone"],
            "Accuracy": f"{tm['accuracy']:.4f}",
            "Precision": f"{tm['precision']:.4f}",
            "Recall": f"{tm['recall']:.4f}",
            "F1-score": f"{tm['f1']:.4f}",
            "FPS": f"{tm.get('fps', '-')}",
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n=== Table 1: Backbone Comparison ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(output_dir, "table1_backbone.csv"), index=False)
    df.to_latex(os.path.join(output_dir, "table1_backbone.tex"), index=False,
                caption="Backbone comparison on bridge deck crack classification.",
                label="tab:backbone")


# ─── Table 2: Ablation Study ───

def make_ablation_table(records, output_dir):
    ablation_order = [
        ("exp1_mobilenetv3_baseline", "Baseline"),
        ("exp2_ablation_robust", "+Robust Aug"),
        ("exp2_ablation_robust_focal", "+Robust Aug +Focal"),
        ("exp2_ablation_full_cbam", "+Robust Aug +Focal +CBAM (Ours)"),
    ]
    rows = []
    for key, label in ablation_order:
        if key not in records:
            continue
        r = records[key]
        tm = r["test_metrics"]
        rows.append({
            "Configuration": label,
            "Accuracy": f"{tm['accuracy']:.4f}",
            "Precision": f"{tm['precision']:.4f}",
            "Recall": f"{tm['recall']:.4f}",
            "F1-score": f"{tm['f1']:.4f}",
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n=== Table 2: Ablation Study ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(output_dir, "table2_ablation.csv"), index=False)
    df.to_latex(os.path.join(output_dir, "table2_ablation.tex"), index=False,
                caption="Ablation study on the proposed method components.",
                label="tab:ablation")


# ─── Table 3: Attention Module Comparison ───

def make_attention_table(records, output_dir):
    attn_keys = [
        ("exp2_ablation_robust_focal", "No Attention"),
        ("exp2_ablation_full_cbam", "CBAM"),
        ("exp2b_ablation_full_ca", "CA"),
    ]
    rows = []
    for key, label in attn_keys:
        if key not in records:
            continue
        r = records[key]
        tm = r["test_metrics"]
        rows.append({
            "Attention": label,
            "Accuracy": f"{tm['accuracy']:.4f}",
            "Precision": f"{tm['precision']:.4f}",
            "Recall": f"{tm['recall']:.4f}",
            "F1-score": f"{tm['f1']:.4f}",
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n=== Table 3: Attention Module Comparison ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(output_dir, "table3_attention.csv"), index=False)


# ─── Figure 1: Training Curves ───

def plot_training_curves(records, output_dir):
    """Plot train/val loss and accuracy curves for key experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    key_experiments = {
        "exp1_mobilenetv3_baseline": "Baseline",
        "exp2_ablation_full_cbam": "Ours (Full)",
    }

    colors = {"Baseline": "#2196F3", "Ours (Full)": "#E53935"}
    for key, label in key_experiments.items():
        if key not in records:
            continue
        h = records[key]["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        c = colors[label]

        axes[0].plot(epochs, h["train_loss"], "-", color=c, alpha=0.6, label=f"{label} Train")
        axes[0].plot(epochs, h["val_loss"], "--", color=c, label=f"{label} Val")

        axes[1].plot(epochs, h["train_acc"], "-", color=c, alpha=0.6, label=f"{label} Train")
        axes[1].plot(epochs, h["val_acc"], "--", color=c, label=f"{label} Val")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_training_curves.png"))
    plt.savefig(os.path.join(output_dir, "fig_training_curves.pdf"))
    plt.close()
    print("[Saved] fig_training_curves.png/pdf")


# ─── Figure 2: Ablation Bar Chart ───

def plot_ablation_bars(records, output_dir):
    """Bar chart comparing Precision, Recall, F1 across ablation configs."""
    ablation_order = [
        ("exp1_mobilenetv3_baseline", "Baseline"),
        ("exp2_ablation_robust", "+RA"),
        ("exp2_ablation_robust_focal", "+RA+FL"),
        ("exp2_ablation_full_cbam", "Ours"),
    ]

    labels, precs, recs, f1s = [], [], [], []
    for key, label in ablation_order:
        if key not in records:
            continue
        tm = records[key]["test_metrics"]
        labels.append(label)
        precs.append(tm["precision"])
        recs.append(tm["recall"])
        f1s.append(tm["f1"])

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.22
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, precs, w, label="Precision", color="#42A5F5")
    ax.bar(x, recs, w, label="Recall", color="#EF5350")
    ax.bar(x + w, f1s, w, label="F1-score", color="#66BB6A")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study Results")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_ablation_bars.png"))
    plt.savefig(os.path.join(output_dir, "fig_ablation_bars.pdf"))
    plt.close()
    print("[Saved] fig_ablation_bars.png/pdf")


# ─── Figure 3: F1 & Recall across Epochs ───

def plot_metric_curves(records, output_dir):
    """Plot val F1 and Recall curves for ablation experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    experiments = {
        "exp1_mobilenetv3_baseline": ("Baseline", "#9E9E9E"),
        "exp2_ablation_robust": ("+RA", "#42A5F5"),
        "exp2_ablation_robust_focal": ("+RA+FL", "#FF9800"),
        "exp2_ablation_full_cbam": ("Ours", "#E53935"),
    }

    for key, (label, color) in experiments.items():
        if key not in records:
            continue
        h = records[key]["history"]
        epochs = range(1, len(h["val_f1"]) + 1)
        axes[0].plot(epochs, h["val_f1"], label=label, color=color)
        axes[1].plot(epochs, h["val_recall"], label=label, color=color)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("F1-score")
    axes[0].set_title("Validation F1-score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Validation Recall")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_metric_curves.png"))
    plt.savefig(os.path.join(output_dir, "fig_metric_curves.pdf"))
    plt.close()
    print("[Saved] fig_metric_curves.png/pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="./runs")
    p.add_argument("--output_dir", type=str, default="./runs/figures")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = load_all_results(args.runs_dir)
    if not records:
        print(f"No results found in {args.runs_dir}")
        return

    print(f"Found {len(records)} experiments: {list(records.keys())}")

    make_backbone_table(records, args.output_dir)
    make_ablation_table(records, args.output_dir)
    make_attention_table(records, args.output_dir)
    plot_training_curves(records, args.output_dir)
    plot_ablation_bars(records, args.output_dir)
    plot_metric_curves(records, args.output_dir)

    print(f"\n[Done] All figures and tables saved to {args.output_dir}")


if __name__ == "__main__":
    main()
