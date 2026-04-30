"""Generate numbered publication figures under runs_dir/figures_pub (fig1–fig10)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_all_results(runs_dir: str) -> dict:
    records = {}
    for d in sorted(Path(runs_dir).iterdir()):
        if not d.is_dir():
            continue
        rfile = d / "results.json"
        if rfile.exists():
            with open(rfile) as f:
                records[d.name] = json.load(f)
    return records


def _save(fig: plt.Figure, output_dir: str, stem: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{stem}.{ext}"))
    plt.close(fig)


def fig1_ablation_bars(records: dict, output_dir: str) -> None:
    order = [
        ("exp1_resnet18_baseline", "Baseline"),
        ("exp2_ablation_robust", "+RA"),
        ("exp2_ablation_robust_focal", "+RA+FL"),
        ("exp2_ablation_full_cbam", "Ours"),
    ]
    labels, precs, recs, f1s = [], [], [], []
    for key, label in order:
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
    ax.set_title("Ablation Study (ResNet18)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "fig1_ablation_bars")
    print("[Saved] fig1_ablation_bars")


def fig2_training_curves(records: dict, output_dir: str) -> None:
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
    fig.tight_layout()
    _save(fig, output_dir, "fig2_training_curves")
    print("[Saved] fig2_training_curves")


def fig3_metric_curves(records: dict, output_dir: str) -> None:
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
    fig.tight_layout()
    _save(fig, output_dir, "fig3_metric_curves")
    print("[Saved] fig3_metric_curves")


def fig4_backbone_tradeoff(records: dict, output_dir: str) -> None:
    pts = []
    for name in sorted(records):
        if not name.startswith("exp1_") or not name.endswith("_baseline"):
            continue
        tm = records[name]["test_metrics"]
        bb = records[name]["config"].get("backbone", name)
        pts.append((bb, tm["f1"], tm.get("fps", 0)))
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    disp_map = {"mobilenetv3": "MobileNetV3", "resnet18": "ResNet18", "efficientnet": "EfficientNet-B0"}
    for bb, f1, fps in pts:
        ax.scatter(fps, f1, s=140, alpha=0.88)
        ax.annotate(disp_map.get(bb, bb), (fps, f1), textcoords="offset points", xytext=(6, 4), fontsize=10)
    ax.set_xlabel("Throughput (FPS, test)")
    ax.set_ylabel("Test F1-score")
    ax.set_title("Backbone trade-off (CE baseline)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "fig4_backbone_tradeoff")
    print("[Saved] fig4_backbone_tradeoff")


def fig5_attention_radar(records: dict, output_dir: str) -> None:
    specs = [
        ("exp2_ablation_robust_focal", "None"),
        ("exp2_ablation_full_cbam", "CBAM"),
        ("exp2b_ablation_full_ca", "CA"),
    ]
    metrics_list = []
    labels = []
    for key, lab in specs:
        if key not in records:
            continue
        tm = records[key]["test_metrics"]
        metrics_list.append([tm["precision"], tm["recall"], tm["f1"]])
        labels.append(lab)
    if len(metrics_list) < 2:
        return
    M = np.array(metrics_list)
    cats = ["Precision", "Recall", "F1"]
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    for i, row in enumerate(M):
        vals = np.concatenate([row, row[:1]])
        ax.plot(angles, vals, "o-", linewidth=1.5, label=labels[i])
        ax.fill(angles, vals, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 1)
    ax.set_title("Attention module comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    _save(fig, output_dir, "fig5_attention_radar")
    print("[Saved] fig5_attention_radar")


def fig6_cross_backbone(records: dict, output_dir: str) -> None:
    pairs = [
        ("exp1_resnet18_baseline", "exp2_ablation_full_cbam", "ResNet18"),
        ("exp1_mobilenetv3_baseline", "exp3_mobilenetv3_full", "MobileNetV3"),
        ("exp1_efficientnet_baseline", "exp3_efficientnet_full", "EfficientNet-B0"),
    ]
    names, base_f1, full_f1 = [], [], []
    for bk, fk, disp in pairs:
        if bk not in records or fk not in records:
            continue
        names.append(disp)
        base_f1.append(records[bk]["test_metrics"]["f1"])
        full_f1.append(records[fk]["test_metrics"]["f1"])
    if not names:
        return
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, base_f1, w, label="Baseline", color="#90CAF9")
    ax.bar(x + w / 2, full_f1, w, label="Ours", color="#E53935")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Test F1-score")
    ax.set_title("Cross-backbone: baseline vs full method")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "fig6_cross_backbone")
    print("[Saved] fig6_cross_backbone")


def fig7_complexity_scatter(records: dict, output_dir: str) -> None:
    """Scatter test FPS vs F1 for representative configs (efficiency vs accuracy)."""
    keys = [
        "exp1_resnet18_baseline",
        "exp1_mobilenetv3_baseline",
        "exp1_efficientnet_baseline",
        "exp2_ablation_full_cbam",
        "exp3_mobilenetv3_full",
        "exp3_efficientnet_full",
    ]
    xs, ys, labs = [], [], []
    for k in keys:
        if k not in records:
            continue
        tm = records[k]["test_metrics"]
        xs.append(tm.get("fps", 0))
        ys.append(tm["f1"])
        labs.append(
            k.replace("exp1_", "")
            .replace("exp2_ablation_", "")
            .replace("exp3_", "")
            .replace("_", " ")
        )
    if len(xs) < 3:
        return
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(xs, ys, s=110, alpha=0.85, c="#1976D2")
    for x, y, lab in zip(xs, ys, labs):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Throughput (FPS, test)")
    ax.set_ylabel("Test F1-score")
    ax.set_title("Efficiency vs accuracy (selected runs)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "fig7_complexity_scatter")
    print("[Saved] fig7_complexity_scatter")


def fig8_ablation_heatmap(records: dict, output_dir: str) -> None:
    order = [
        ("exp1_resnet18_baseline", "Baseline"),
        ("exp2_ablation_robust", "+RA"),
        ("exp2_ablation_robust_focal", "+RA+FL"),
        ("exp2_ablation_full_cbam", "Ours"),
    ]
    rows = []
    labels = []
    for key, lab in order:
        if key not in records:
            continue
        tm = records[key]["test_metrics"]
        rows.append([tm["accuracy"], tm["precision"], tm["recall"], tm["f1"]])
        labels.append(lab)
    if len(rows) < 2:
        return
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Ablation heatmap (test metrics)")
    fig.tight_layout()
    _save(fig, output_dir, "fig8_ablation_heatmap")
    print("[Saved] fig8_ablation_heatmap")


def fig9_ablation_waterfall(records: dict, output_dir: str) -> None:
    order = [
        ("exp1_resnet18_baseline", "Baseline"),
        ("exp2_ablation_robust", "+RA"),
        ("exp2_ablation_robust_focal", "+RA+FL"),
        ("exp2_ablation_full_cbam", "Ours"),
    ]
    f1s = []
    labels = []
    for key, lab in order:
        if key not in records:
            continue
        f1s.append(records[key]["test_metrics"]["f1"])
        labels.append(lab)
    if len(f1s) < 2:
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, f1s, color="#42A5F5", alpha=0.85)
    ax.plot(x, f1s, "o-", color="#C62828", linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test F1-score")
    ax.set_title("Ablation: F1 progression")
    ax.set_ylim(0, max(1.05, max(f1s) * 1.08))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "fig9_ablation_waterfall")
    print("[Saved] fig9_ablation_waterfall")


def fig10_pr_tradeoff(records: dict, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
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
        vp = h["val_precision"]
        vr = h["val_recall"]
        ax.plot(vp, vr, color=color, label=label, linewidth=1.8)
        if vp and vr:
            ax.scatter(vp[-1], vr[-1], color=color, s=42, zorder=5)
    ax.set_xlabel("Validation precision")
    ax.set_ylabel("Validation recall")
    ax.set_title("Precision–recall trajectories (validation)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "fig10_pr_tradeoff")
    print("[Saved] fig10_pr_tradeoff")


def main():
    p = argparse.ArgumentParser(description="Generate publication-quality numbered figures")
    p.add_argument("--runs_dir", type=str, default="./runs_v3")
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()
    output_dir = args.output_dir or os.path.join(args.runs_dir, "figures_pub")
    os.makedirs(output_dir, exist_ok=True)
    records = load_all_results(args.runs_dir)
    if not records:
        print(f"No results found in {args.runs_dir}")
        return
    print(f"Found {len(records)} experiments: {list(records.keys())}")
    print(f"Output directory: {output_dir}\n")
    print("=" * 60)
    print("Generating publication-quality figures...")
    print("=" * 60)
    fig1_ablation_bars(records, output_dir)
    fig2_training_curves(records, output_dir)
    fig3_metric_curves(records, output_dir)
    fig4_backbone_tradeoff(records, output_dir)
    fig5_attention_radar(records, output_dir)
    fig6_cross_backbone(records, output_dir)
    fig7_complexity_scatter(records, output_dir)
    fig8_ablation_heatmap(records, output_dir)
    fig9_ablation_waterfall(records, output_dir)
    fig10_pr_tradeoff(records, output_dir)
    print(f"\n{'=' * 60}")
    print(f"All figures saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
