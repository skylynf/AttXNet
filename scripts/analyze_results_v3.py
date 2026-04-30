"""
Aggregate V3 experiment results: tables and publication-quality figures.
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
    records = {}
    for d in sorted(Path(runs_dir).iterdir()):
        rfile = d / "results.json"
        if rfile.exists():
            with open(rfile) as f:
                records[d.name] = json.load(f)
    return records


def _fmt(v, is_pct=True):
    if isinstance(v, float):
        return f"{v:.4f}" if is_pct else f"{v:.1f}"
    return str(v)


# ─── Table 1: Backbone Comparison (D only, CE baseline) ───

def make_backbone_table(records, output_dir):
    rows = []
    for name in sorted(records):
        if not name.startswith("exp1_"):
            continue
        r = records[name]
        tm = r["test_metrics"]
        cfg = r["config"]
        rows.append({
            "Backbone": cfg["backbone"].replace("mobilenetv3", "MobileNetV3").replace("resnet18", "ResNet18").replace("efficientnet", "EfficientNet-B0"),
            "Accuracy": float(f"{tm['accuracy']:.4f}"),
            "Precision": float(f"{tm['precision']:.4f}"),
            "Recall": float(f"{tm['recall']:.4f}"),
            "F1-score": float(f"{tm['f1']:.4f}"),
            "FPS": float(f"{tm.get('fps', 0):.1f}"),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n=== Table 1: Backbone Comparison (D only) ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(output_dir, "table1_backbone.csv"), index=False)
    df.to_latex(os.path.join(output_dir, "table1_backbone.tex"), index=False,
                caption="Performance comparison of lightweight backbones on bridge deck crack classification (SDNET2018-D).",
                label="tab:backbone", float_format="%.4f")


# ─── Table 2: Ablation Study (ResNet18, D only) ───

def make_ablation_table(records, output_dir):
    ablation_order = [
        ("exp1_resnet18_baseline", "Baseline (CE)"),
        ("exp2_ablation_robust", "+ Robust Aug"),
        ("exp2_ablation_robust_focal", "+ Robust Aug + Focal Loss"),
        ("exp2_ablation_full_cbam", "+ Robust Aug + Focal Loss + CBAM (Ours)"),
    ]
    rows = []
    base_f1 = None
    for key, label in ablation_order:
        if key not in records:
            continue
        r = records[key]
        tm = r["test_metrics"]
        f1 = tm["f1"]
        if base_f1 is None:
            base_f1 = f1
            delta = "-"
        else:
            delta = f"{(f1 - base_f1)*100:+.2f}%"
        rows.append({
            "Configuration": label,
            "Accuracy": float(f"{tm['accuracy']:.4f}"),
            "Precision": float(f"{tm['precision']:.4f}"),
            "Recall": float(f"{tm['recall']:.4f}"),
            "F1-score": float(f"{f1:.4f}"),
            "ΔF1": delta,
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n=== Table 2: Ablation Study (ResNet18) ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(output_dir, "table2_ablation.csv"), index=False)
    df_tex = df.drop(columns=["ΔF1"])
    df_tex.to_latex(os.path.join(output_dir, "table2_ablation.tex"), index=False,
                    caption="Ablation study of the proposed method on ResNet18 backbone (SDNET2018-D).",
                    label="tab:ablation", float_format="%.4f")


# ─── Table 3: Attention Module Comparison ───

def make_attention_table(records, output_dir):
    attn_keys = [
        ("exp2_ablation_robust_focal", "None"),
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
            "Accuracy": float(f"{tm['accuracy']:.4f}"),
            "Precision": float(f"{tm['precision']:.4f}"),
            "Recall": float(f"{tm['recall']:.4f}"),
            "F1-score": float(f"{tm['f1']:.4f}"),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n=== Table 3: Attention Module Comparison (ResNet18) ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(output_dir, "table3_attention.csv"), index=False)


# ─── Table 4: Cross-backbone full method ───

def make_crossbackbone_table(records, output_dir):
    pairs = [
        ("exp1_resnet18_baseline", "exp2_ablation_full_cbam", "ResNet18"),
        ("exp1_mobilenetv3_baseline", "exp3_mobilenetv3_full", "MobileNetV3"),
        ("exp1_efficientnet_baseline", "exp3_efficientnet_full", "EfficientNet-B0"),
    ]
    rows = []
    for base_key, full_key, name in pairs:
        if base_key not in records or full_key not in records:
            continue
        tb = records[base_key]["test_metrics"]
        tf = records[full_key]["test_metrics"]
        rows.append({
            "Backbone": name,
            "Baseline F1": float(f"{tb['f1']:.4f}"),
            "Ours F1": float(f"{tf['f1']:.4f}"),
            "ΔF1": f"{(tf['f1']-tb['f1'])*100:+.2f}%",
            "Baseline Recall": float(f"{tb['recall']:.4f}"),
            "Ours Recall": float(f"{tf['recall']:.4f}"),
            "ΔRecall": f"{(tf['recall']-tb['recall'])*100:+.2f}%",
            "Built-in SE": "Yes" if name != "ResNet18" else "No",
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n=== Table 4: Cross-backbone Full Method ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(output_dir, "table4_crossbackbone.csv"), index=False)


# ─── Reviewer: Baseline / WCE / Robust / Focal+CBAM ───

def make_reviewer_ablation_table(records, output_dir):
    order = [
        ("rev_baseline_ce", "ResNet18 + CE + standard aug (+WeightedSampler)"),
        ("rev_wce", "ResNet18 + weighted CE (inverse freq., no sampler)"),
        ("rev_robust_aug", "ResNet18 + CE + robust aug (+WeightedSampler)"),
        ("rev_focal_no_ra", "ResNet18 + Focal Loss + standard aug (γ=2, α=0.75; no RA, no CBAM)"),
        ("rev_cbam_only", "ResNet18 + CE + CBAM + standard aug (+WeightedSampler)"),
        ("rev_focal_gamma_2", "ResNet18 + Focal Loss + CBAM + robust aug (γ=2, α=0.75)"),
    ]
    rows = []
    for key, label in order:
        if key not in records:
            continue
        tm = records[key]["test_metrics"]
        rows.append({
            "Setting": label,
            "Accuracy": float(f"{tm['accuracy']:.4f}"),
            "Precision": float(f"{tm['precision']:.4f}"),
            "Recall": float(f"{tm['recall']:.4f}"),
            "F1-score": float(f"{tm['f1']:.4f}"),
            "FPS": float(f"{tm.get('fps', 0):.1f}"),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n=== Reviewer Table: Ablation (loss & augmentation) ===")
    print(df.to_string(index=False))
    out = os.path.join(output_dir, "table_reviewer_ablation.csv")
    df.to_csv(out, index=False)


# ─── Reviewer: Focal Loss hyper-parameter sweep (gamma) ───

def make_reviewer_focal_sweep_table(records, output_dir):
    order = [
        ("rev_focal_gamma_1", "gamma=1.0"),
        ("rev_focal_gamma_2", "gamma=2.0"),
        ("rev_focal_gamma_3", "gamma=3.0"),
        ("rev_focal_gamma_5", "gamma=5.0"),
    ]
    rows = []
    for key, label in order:
        if key not in records:
            continue
        tm = records[key]["test_metrics"]
        cfg = records[key].get("config", {})
        fa = cfg.get("focal_alpha", "-")
        fg = cfg.get("focal_gamma", "-")
        rows.append({
            "Run": label,
            "alpha": fa,
            "gamma": fg,
            "Accuracy": float(f"{tm['accuracy']:.4f}"),
            "Precision": float(f"{tm['precision']:.4f}"),
            "Recall": float(f"{tm['recall']:.4f}"),
            "F1-score": float(f"{tm['f1']:.4f}"),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n=== Reviewer Table: Focal Loss parameter sweep (fixed alpha=0.75) ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(output_dir, "table_reviewer_focal_sweep.csv"), index=False)


# ─── Table 5: Cross-category (D+P+W) ───

def make_crosscategory_table(records, output_dir):
    keys = [
        ("exp4_resnet18_DPW_baseline", "ResNet18 Baseline"),
        ("exp4_resnet18_DPW_full", "ResNet18 + Ours"),
        ("exp4_mobilenetv3_DPW_full", "MobileNetV3 + Ours"),
    ]
    rows = []
    for key, label in keys:
        if key not in records:
            continue
        r = records[key]
        tm = r["test_metrics"]
        rows.append({
            "Model": label,
            "Accuracy": float(f"{tm['accuracy']:.4f}"),
            "Precision": float(f"{tm['precision']:.4f}"),
            "Recall": float(f"{tm['recall']:.4f}"),
            "F1-score": float(f"{tm['f1']:.4f}"),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n=== Table 5: Cross-category (D+P+W) ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(output_dir, "table5_crosscategory.csv"), index=False)
    df.to_latex(os.path.join(output_dir, "table5_crosscategory.tex"), index=False,
                caption="Generalization study on full SDNET2018 dataset (Deck + Pavement + Wall).",
                label="tab:crosscat", float_format="%.4f")


# ─── Figure 1: Training curves (Baseline vs Ours, ResNet18) ───

def plot_training_curves(records, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    key_experiments = {
        "exp1_resnet18_baseline": ("Baseline", "#2196F3", "-"),
        "exp2_ablation_full_cbam": ("Ours (Full)", "#E53935", "-"),
    }
    for key, (label, color, ls) in key_experiments.items():
        if key not in records:
            continue
        h = records[key]["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], ls, color=color, alpha=0.5, label=f"{label} Train")
        axes[0].plot(epochs, h["val_loss"], "--", color=color, label=f"{label} Val")
        axes[1].plot(epochs, h["train_acc"], ls, color=color, alpha=0.5, label=f"{label} Train")
        axes[1].plot(epochs, h["val_acc"], "--", color=color, label=f"{label} Val")

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_training_curves.png"))
    plt.savefig(os.path.join(output_dir, "fig_training_curves.pdf"))
    plt.close()
    print("[Saved] fig_training_curves.png/pdf")


# ─── Figure 2: Ablation bar chart ───

def plot_ablation_bars(records, output_dir):
    ablation_order = [
        ("exp1_resnet18_baseline", "Baseline"),
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
    x = np.arange(len(labels)); w = 0.22
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, precs, w, label="Precision", color="#42A5F5")
    ax.bar(x, recs, w, label="Recall", color="#EF5350")
    ax.bar(x + w, f1s, w, label="F1-score", color="#66BB6A")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_title("Ablation Study (ResNet18)")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_ablation_bars.png"))
    plt.savefig(os.path.join(output_dir, "fig_ablation_bars.pdf"))
    plt.close()
    print("[Saved] fig_ablation_bars.png/pdf")


# ─── Figure 3: Val F1 & Recall curves ───

def plot_metric_curves(records, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    experiments = {
        "exp1_resnet18_baseline": ("Baseline", "#9E9E9E"),
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
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("F1-score")
    axes[0].set_title("Validation F1-score"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Recall")
    axes[1].set_title("Validation Recall"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_metric_curves.png"))
    plt.savefig(os.path.join(output_dir, "fig_metric_curves.pdf"))
    plt.close()
    print("[Saved] fig_metric_curves.png/pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="./runs_v3")
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.runs_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    records = load_all_results(args.runs_dir)
    if not records:
        print(f"No results found in {args.runs_dir}"); return

    print(f"Found {len(records)} experiments: {list(records.keys())}")

    make_backbone_table(records, args.output_dir)
    make_ablation_table(records, args.output_dir)
    make_attention_table(records, args.output_dir)
    make_crossbackbone_table(records, args.output_dir)
    make_crosscategory_table(records, args.output_dir)
    make_reviewer_ablation_table(records, args.output_dir)
    make_reviewer_focal_sweep_table(records, args.output_dir)
    plot_training_curves(records, args.output_dir)
    plot_ablation_bars(records, args.output_dir)
    plot_metric_curves(records, args.output_dir)
    print(f"\n[Done] All figures and tables saved to {args.output_dir}")


if __name__ == "__main__":
    main()
