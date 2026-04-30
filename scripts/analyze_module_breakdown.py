"""
Quantify per-module Params and FLOPs (Backbone / Attention / Head)
for CrackClassifier configs used in runs_v3. Generates publication-style figures.

Usage (from repository root):
  python scripts/analyze_module_breakdown.py --output_dir ./runs_v3/figures/module_breakdown
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from thop import profile

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attxnet.models import build_model


@dataclass
class BreakdownRow:
    backbone: str
    attention: str
    params_bb: int
    params_attn: int
    params_head: int
    flops_bb: int
    flops_attn: int
    flops_head: int

    @property
    def params_total(self) -> int:
        return self.params_bb + self.params_attn + self.params_head

    @property
    def flops_total(self) -> int:
        return self.flops_bb + self.flops_attn + self.flops_head


class _BackboneFeatures(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def measure_breakdown(
    backbone: str,
    attention: str,
    img_size: int,
    device: str,
) -> BreakdownRow:
    model = build_model(backbone, attention, pretrained=False).to(device)
    model.eval()

    dummy_img = torch.randn(1, 3, img_size, img_size, device=device)

    with torch.no_grad():
        feat = model.backbone.forward_features(dummy_img)

    params_bb = _count_params(model.backbone)
    params_attn = _count_params(model.attn)
    params_head = _count_params(model.head)

    # FLOPs: backbone features
    bb_wrap = _BackboneFeatures(model.backbone).to(device).eval()
    flops_bb, _ = profile(bb_wrap, inputs=(dummy_img.clone(),), verbose=False)

    # Attention (spatial); Identity → 0 FLOPs from thop
    if feat.dim() == 4:
        dummy_feat = torch.randn(1, feat.shape[1], feat.shape[2], feat.shape[3], device=device)
        flops_attn, _ = profile(model.attn, inputs=(dummy_feat,), verbose=False)
    else:
        flops_attn = 0

    dummy_vec = torch.randn(1, model._feat_dim, device=device)
    flops_head, _ = profile(model.head, inputs=(dummy_vec,), verbose=False)

    torch.cuda.empty_cache() if device.startswith("cuda") else None

    return BreakdownRow(
        backbone=backbone,
        attention=attention,
        params_bb=int(params_bb),
        params_attn=int(params_attn),
        params_head=int(params_head),
        flops_bb=int(flops_bb),
        flops_attn=int(flops_attn),
        flops_head=int(flops_head),
    )


def _pct_part(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * part / total


def rows_to_records(rows: List[BreakdownRow]) -> List[Dict]:
    out = []
    for r in rows:
        pt, ft = r.params_total, r.flops_total
        out.append({
            "backbone": r.backbone,
            "attention": r.attention,
            "params_total": pt,
            "params_backbone": r.params_bb,
            "params_attention": r.params_attn,
            "params_head": r.params_head,
            "params_pct_backbone": round(_pct_part(r.params_bb, pt), 2),
            "params_pct_attention": round(_pct_part(r.params_attn, pt), 2),
            "params_pct_head": round(_pct_part(r.params_head, pt), 2),
            "flops_total": ft,
            "flops_backbone": r.flops_bb,
            "flops_attention": r.flops_attn,
            "flops_head": r.flops_head,
            "flops_pct_backbone": round(_pct_part(r.flops_bb, ft), 2),
            "flops_pct_attention": round(_pct_part(r.flops_attn, ft), 2),
            "flops_pct_head": round(_pct_part(r.flops_head, ft), 2),
        })
    return out


def setup_paper_mpl():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


# Colorblind-friendly palette (Okabe–Ito inspired)
COLORS = {
    "backbone": "#0072B2",
    "attention": "#E69F00",
    "head": "#009E73",
}


def plot_stacked_bars(
    rows: List[BreakdownRow],
    out_base: str,
    metric: str,  # "params" | "flops"
):
    """Horizontal stacked bars: one row per config (backbone — attention)."""
    labels = [f"{r.backbone}\n({r.attention})" for r in rows]
    n = len(rows)
    if metric == "params":
        a = np.array([r.params_bb for r in rows], dtype=np.float64)
        b = np.array([r.params_attn for r in rows], dtype=np.float64)
        c = np.array([r.params_head for r in rows], dtype=np.float64)
        title = "Parameter count by module"
        xlabel = "Parameters (millions)"
        scale = 1e6
    else:
        a = np.array([r.flops_bb for r in rows], dtype=np.float64)
        b = np.array([r.flops_attn for r in rows], dtype=np.float64)
        c = np.array([r.flops_head for r in rows], dtype=np.float64)
        title = "MACs by module (single 224×224 forward, thop convention)"
        xlabel = "MACs (GFLOPs)"
        scale = 1e9

    a_s, b_s, c_s = a / scale, b / scale, c / scale
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(6.8, max(2.4, 0.38 * n + 0.8)))
    ax.barh(y, a_s, color=COLORS["backbone"], label="Backbone", height=0.65)
    ax.barh(y, b_s, left=a_s, color=COLORS["attention"], label="Attention", height=0.65)
    ax.barh(y, c_s, left=a_s + b_s, color=COLORS["head"], label="Head", height=0.65)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.92)

    # Annotate total at bar end
    totals = a_s + b_s + c_s
    xmax = float(totals.max()) * 1.12 if n else 1.0
    ax.set_xlim(0, xmax)
    for i, t in enumerate(totals):
        ax.text(t + xmax * 0.01, y[i], f"{t:.2f}", va="center", fontsize=7, color="#333333")

    fig.tight_layout()
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", bbox_inches="tight")
    plt.close(fig)


def plot_share_heatmap(
    rows: List[BreakdownRow],
    out_base: str,
):
    """Heatmap of percentage share (Params and FLOPs) per row."""
    backbones = sorted({r.backbone for r in rows})
    attentions = ["none", "cbam", "ca"]
    # matrix shape: len(backbones) x len(attentions) x 2 metrics x 3 parts — use faceted small multiples instead

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.0), sharex=False)

    for ax, metric, name in zip(
        axes,
        ["params", "flops"],
        ["Parameter share (%)", "MACs share (%)"],
    ):
        mat = []
        ylabels = []
        for bb in backbones:
            for att in attentions:
                row = next((r for r in rows if r.backbone == bb and r.attention == att), None)
                if row is None:
                    continue
                if metric == "params":
                    total = row.params_total
                    vec = [
                        _pct_part(row.params_bb, total),
                        _pct_part(row.params_attn, total),
                        _pct_part(row.params_head, total),
                    ]
                else:
                    total = row.flops_total
                    vec = [
                        _pct_part(row.flops_bb, total),
                        _pct_part(row.flops_attn, total),
                        _pct_part(row.flops_head, total),
                    ]
                mat.append(vec)
                ylabels.append(f"{bb} | {att}")

        M = np.array(mat, dtype=np.float64)  # (R, 3)
        im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=100)
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Backbone", "Attention", "Head"])
        ax.set_title(name)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                tc = "#f8f8f8" if v > 55 else "#111111"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6, color=tc)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")

    fig.tight_layout()
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", bbox_inches="tight")
    plt.close(fig)


def plot_attention_overhead_vs_baseline(
    rows: List[BreakdownRow],
    out_base: str,
):
    """Grouped bars: incremental Params / MACs of CBAM and CA vs same backbone with attention=none."""
    backbones = ["resnet18", "mobilenetv3", "efficientnet"]
    row_map = {(r.backbone, r.attention): r for r in rows}

    x = np.arange(len(backbones))
    w = 0.35

    def deltas_for(attn: str) -> Tuple[List[float], List[float]]:
        dp, df = [], []
        for bb in backbones:
            base = row_map[(bb, "none")]
            cur = row_map[(bb, attn)]
            dp.append(100.0 * (cur.params_total - base.params_total) / max(base.params_total, 1))
            df.append(100.0 * (cur.flops_total - base.flops_total) / max(base.flops_total, 1))
        return dp, df

    d_cbam_p, d_cbam_f = deltas_for("cbam")
    d_ca_p, d_ca_f = deltas_for("ca")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axp, axf = axes

    axp.bar(x - w / 2, d_cbam_p, width=w, label="CBAM", color=COLORS["attention"], edgecolor="black", linewidth=0.4)
    axp.bar(x + w / 2, d_ca_p, width=w, label="CA", color="#CC79A7", edgecolor="black", linewidth=0.4)
    axp.set_xticks(x)
    axp.set_xticklabels([b.replace("mobilenetv3", "MobileNetV3").replace("efficientnet", "Eff.-B0").replace("resnet18", "ResNet18") for b in backbones])
    axp.set_ylabel("Δ Parameters (%) vs. no attention")
    axp.set_title("(a) Relative parameter overhead")
    axp.legend(framealpha=0.92)
    axp.grid(axis="y")

    axf.bar(x - w / 2, d_cbam_f, width=w, label="CBAM", color=COLORS["attention"], edgecolor="black", linewidth=0.4)
    axf.bar(x + w / 2, d_ca_f, width=w, label="CA", color="#CC79A7", edgecolor="black", linewidth=0.4)
    axf.set_xticks(x)
    axf.set_xticklabels([b.replace("mobilenetv3", "MobileNetV3").replace("efficientnet", "Eff.-B0").replace("resnet18", "ResNet18") for b in backbones])
    axf.set_ylabel("Δ MACs (%) vs. no attention")
    axf.set_title("(b) Relative MAC overhead (thop)")
    axf.legend(framealpha=0.92)
    axf.grid(axis="y")

    fig.tight_layout()
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--output_dir",
        type=str,
        default="./runs_v3/figures/module_breakdown",
    )
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    backbones = ["resnet18", "mobilenetv3", "efficientnet"]
    attentions = ["none", "cbam", "ca"]

    rows: List[BreakdownRow] = []
    for bb in backbones:
        for att in attentions:
            print(f"Profiling {bb} + {att} ...")
            rows.append(measure_breakdown(bb, att, args.img_size, device))

    recs = rows_to_records(rows)
    json_path = os.path.join(args.output_dir, "module_breakdown.json")
    with open(json_path, "w") as f:
        json.dump(recs, f, indent=2)

    csv_path = os.path.join(args.output_dir, "module_breakdown.csv")
    if recs:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)

    setup_paper_mpl()
    plot_stacked_bars(rows, os.path.join(args.output_dir, "fig_stacked_params"), "params")
    plot_stacked_bars(rows, os.path.join(args.output_dir, "fig_stacked_flops"), "flops")
    plot_share_heatmap(rows, os.path.join(args.output_dir, "fig_share_heatmap"))
    plot_attention_overhead_vs_baseline(rows, os.path.join(args.output_dir, "fig_attention_overhead"))

    # Console summary
    print("\n" + "=" * 72)
    for r in rows:
        pt, ft = r.params_total, r.flops_total
        print(
            f"{r.backbone:14s} {r.attention:5s} | "
            f"P: BB {_pct_part(r.params_bb, pt):5.1f}%  Attn {_pct_part(r.params_attn, pt):5.1f}%  Head {_pct_part(r.params_head, pt):5.1f}% | "
            f"F: BB {_pct_part(r.flops_bb, ft):5.1f}%  Attn {_pct_part(r.flops_attn, ft):5.1f}%  Head {_pct_part(r.flops_head, ft):5.1f}%"
        )
    print("=" * 72)
    print(f"Saved figures and tables under: {args.output_dir}")


if __name__ == "__main__":
    main()
