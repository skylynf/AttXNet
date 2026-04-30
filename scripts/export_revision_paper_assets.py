#!/usr/bin/env python3
"""
Build revision/paper_additional_results.md and reviewer figures from runs_v3/*/results.json.
Writes supplementary markdown and figures under outputs/revision/.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ATT_ROOT = Path(__file__).resolve().parent.parent
RUNS = ATT_ROOT / "runs_v3"
REVISION = ATT_ROOT / "outputs" / "revision"
FIG_OUT = REVISION / "figures"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Liberation Sans",
            "Arial",
            "Helvetica",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
    }
)


def load_metrics(exp_name: str) -> dict:
    path = RUNS / exp_name / "results.json"
    with open(path) as f:
        r = json.load(f)
    tm = r["test_metrics"]
    return {
        "accuracy": tm["accuracy"],
        "precision": tm["precision"],
        "recall": tm["recall"],
        "f1": tm["f1"],
        "fps": tm.get("fps", float("nan")),
    }


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}"


def fmt_fps(x: float) -> str:
    if x != x:
        return "—"
    return f"{x:.0f}"


def plot_reviewer_ablation(rows: list[tuple[str, dict]]) -> None:
    labels = [r[0] for r in rows]
    precs = [r[1]["precision"] for r in rows]
    recs = [r[1]["recall"] for r in rows]
    f1s = [r[1]["f1"] for r in rows]

    x = np.arange(len(labels))
    w = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.bar(x - w, precs, w, label="Precision", color="#42A5F5")
    ax.bar(x, recs, w, label="Recall", color="#EF5350")
    ax.bar(x + w, f1s, w, label="F1-score", color="#66BB6A")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.35)
    ax.set_title("Ablation on SDNET2018-D (ResNet18): loss & augmentation")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_OUT / f"fig_R1_reviewer_ablation.{ext}")
    plt.close()


def plot_gamma_sweep(gamma_vals: list[float], metrics_by_gamma: list[dict]) -> None:
    """Twin y-axis: left = Acc/Prec/Rec/F1 (%); right = FPS. Font: Arial."""
    gammas = np.asarray(gamma_vals, dtype=float)
    acc = np.array([m["accuracy"] * 100.0 for m in metrics_by_gamma])
    prec = np.array([m["precision"] * 100.0 for m in metrics_by_gamma])
    rec = np.array([m["recall"] * 100.0 for m in metrics_by_gamma])
    f1 = np.array([m["f1"] * 100.0 for m in metrics_by_gamma])
    fps_vals = np.array([m.get("fps", float("nan")) for m in metrics_by_gamma])

    fig, ax_l = plt.subplots(figsize=(7.2, 4.2))
    ax_r = ax_l.twinx()

    styles = [
        ("Accuracy", acc, "#1565C0", "o", "-", 7),
        ("Precision", prec, "#00897B", "s", "-", 6),
        ("Recall", rec, "#FB8C00", "^", "--", 6),
        ("F1-score", f1, "#C62828", "D", "-", 6),
    ]
    for label, ys, color, mkr, ls, ms in styles:
        ax_l.plot(
            gammas,
            ys,
            marker=mkr,
            linestyle=ls,
            color=color,
            linewidth=2.0,
            markersize=ms,
            label=label,
            clip_on=False,
        )

    ax_r.plot(
        gammas,
        fps_vals,
        marker="*",
        linestyle=":",
        color="#6A1B9A",
        linewidth=2.2,
        markersize=11,
        label="FPS",
        clip_on=False,
    )

    ax_l.set_xlabel(r"Focal loss $\gamma$ (fixed $\alpha=0.75$)")
    ax_l.set_ylabel("Test metric (%)", color="#37474F")
    ax_r.set_ylabel("FPS", color="#6A1B9A")
    ax_r.tick_params(axis="y", labelcolor="#6A1B9A")
    ax_l.tick_params(axis="y", labelcolor="#37474F")
    ax_r.spines["right"].set_color("#6A1B9A")
    ax_r.spines["left"].set_visible(False)

    ax_l.set_xticks(gammas)
    ax_l.set_xticklabels(
        [str(int(g)) if abs(g - round(g)) < 1e-9 else str(g) for g in gammas]
    )
    ax_l.set_xlim(float(gammas.min()) - 0.35, float(gammas.max()) + 0.35)

    ymin = float(min(acc.min(), prec.min(), rec.min(), f1.min())) - 2.0
    ymax = float(max(acc.max(), prec.max(), rec.max(), f1.max())) + 2.0
    ax_l.set_ylim(max(0.0, ymin), min(100.0, ymax))

    fps_ok = fps_vals[~np.isnan(fps_vals)]
    if fps_ok.size:
        fp_min, fp_max = float(fps_ok.min()), float(fps_ok.max())
        pad = max(8.0, (fp_max - fp_min) * 0.25)
        ax_r.set_ylim(fp_min - pad, fp_max + pad)

    ax_l.grid(True, axis="both", alpha=0.32, linestyle="-", linewidth=0.6)
    ax_l.set_axisbelow(True)

    handles_l, labels_l = ax_l.get_legend_handles_labels()
    handles_r, labels_r = ax_r.get_legend_handles_labels()
    fig.legend(
        handles_l + handles_r,
        labels_l + labels_r,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        fontsize=9,
        columnspacing=1.0,
    )

    ax_l.set_title(
        r"Focal $\gamma$ sweep — ResNet18 + CBAM + robust aug., SDNET2018-D "
        r"(no weighted sampler)",
        fontsize=10,
        pad=30,
    )

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_OUT / f"fig_R2_focal_gamma_sweep.{ext}")
    plt.close()


def main():
    REVISION.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # (markdown row label, short label for matplotlib x-axis, exp_name)
    ablation_specs = [
        ("Baseline CE + std aug (+sampler)", "Baseline CE\n(+sampler)", "rev_baseline_ce"),
        ("Weighted CE (inv. freq.)", "Weighted CE", "rev_wce"),
        ("CE + robust augmentation", "CE + robust aug.", "rev_robust_aug"),
        ("Focal, std aug ($\\gamma{=}2$, no RA/CBAM)", "Focal\n(no RA)", "rev_focal_no_ra"),
        ("CE + CBAM + std aug", "CE+CBAM\n(std aug)", "rev_cbam_only"),
        ("Focal + CBAM + robust ($\\gamma{=}2$)", "Focal+CBAM\n($\\gamma{=}2$)", "rev_focal_gamma_2"),
    ]

    table_r1_rows = [
        "| Configuration | Acc. (%) | Prec. (%) | Rec. (%) | F1 (%) | FPS |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    rows_plot: list[tuple[str, dict]] = []
    for md_label, plot_label, exp in ablation_specs:
        m = load_metrics(exp)
        table_r1_rows.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                md_label,
                fmt_pct(m["accuracy"]),
                fmt_pct(m["precision"]),
                fmt_pct(m["recall"]),
                fmt_pct(m["f1"]),
                fmt_fps(m["fps"]),
            )
        )
        rows_plot.append((plot_label, m))

    plot_reviewer_ablation(rows_plot)

    focal_specs = [
        ("rev_focal_gamma_1", 1.0),
        ("rev_focal_gamma_2", 2.0),
        ("rev_focal_gamma_3", 3.0),
        ("rev_focal_gamma_5", 5.0),
    ]
    gammas: list[float] = []
    mlist: list[dict] = []
    table_r2_rows = [
        "| $\\gamma$ | $\\alpha$ | Acc. (%) | Prec. (%) | Rec. (%) | F1 (%) | FPS |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for exp, g in focal_specs:
        m = load_metrics(exp)
        gammas.append(g)
        mlist.append(m)
        table_r2_rows.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                int(g) if g == int(g) else g,
                0.75,
                fmt_pct(m["accuracy"]),
                fmt_pct(m["precision"]),
                fmt_pct(m["recall"]),
                fmt_pct(m["f1"]),
                fmt_fps(m["fps"]),
            )
        )

    plot_gamma_sweep(gammas, mlist)

    copy_pairs = [
        (RUNS / "figures" / "fig_training_curves.pdf", "fig_sup_training_curves.pdf"),
        (RUNS / "figures" / "fig_training_curves.png", "fig_sup_training_curves.png"),
        (RUNS / "figures" / "fig_metric_curves.pdf", "fig_sup_metric_curves.pdf"),
        (RUNS / "figures" / "fig_metric_curves.png", "fig_sup_metric_curves.png"),
        (RUNS / "figures" / "fig_ablation_bars.pdf", "fig_sup_legacy_ablation_bars.pdf"),
        (RUNS / "figures" / "fig_ablation_bars.png", "fig_sup_legacy_ablation_bars.png"),
        (RUNS / "figures_pub" / "fig6_cross_backbone.pdf", "fig_sup_cross_backbone.pdf"),
        (RUNS / "figures_pub" / "fig6_cross_backbone.png", "fig_sup_cross_backbone.png"),
        (RUNS / "figures_pub" / "fig7_complexity_scatter.pdf", "fig_sup_complexity.pdf"),
        (RUNS / "figures_pub" / "fig7_complexity_scatter.png", "fig_sup_complexity.png"),
        (
            RUNS / "figures" / "dataset_stats" / "fig_dataset_crack_share_by_category.pdf",
            "fig_sup_dataset_distribution.pdf",
        ),
        (
            RUNS / "figures" / "dataset_stats" / "fig_dataset_crack_share_by_category.png",
            "fig_sup_dataset_distribution.png",
        ),
    ]
    for src, dst in copy_pairs:
        if src.exists():
            shutil.copy2(src, FIG_OUT / dst)

    b0 = load_metrics("rev_baseline_ce")
    latex_example = (
        "Baseline CE + std aug (+sampler) & "
        f"{b0['accuracy']:.4f} & {b0['precision']:.4f} & "
        f"{b0['recall']:.4f} & {b0['f1']:.4f} \\\\"
    )

    md = f"""# Additional Experimental Results（审稿补充材料）

面向综述意见的额外实验（**SDNET2018-D**，二分类图像块）。划分与训练脚本一致：`seed=42`，与 `dataset.split_dataset` 逻辑相同。表中为 **test** 指标（与 `runs_v3/*/results.json` 一致）。

配图位于 `outputs/revision/figures/`（PDF 用于 LaTeX 排版，PNG 便于 Word/Markdown 预览）。

---

## 表 R1 — 损失与增强消融（ResNet18）

{chr(10).join(table_r1_rows)}

**说明**：*Weighted CE* 为训练集逆频加权交叉熵，且关闭 `WeightedRandomSampler`；*Baseline CE* 为标准增强 + 加权采样。

---

## 图 R1 — 对应表 R1（Precision / Recall / F1）

![图 R1](figures/fig_R1_reviewer_ablation.png)

LaTeX: `\\includegraphics[width=\\linewidth]{{figures/fig_R1_reviewer_ablation.pdf}}`

---

## 表 R2 — Focal Loss $\\gamma$ 参数扫描

固定 $\\alpha{{=}}0.75$，结构为 ResNet18 + CBAM + 稳健增强，`no_weighted_sampler`。

{chr(10).join(table_r2_rows)}

---

## 图 R2 — $\\gamma$ 与 Acc / Prec / Rec / F1 / FPS（对应表 R2）

![图 R2](figures/fig_R2_focal_gamma_sweep.png)

LaTeX: `\\includegraphics[width=0.48\\linewidth]{{figures/fig_R2_focal_gamma_sweep.pdf}}`

---

## 其它可直接用于论文的补充图（由既有流水线复制）

| 文件 | 内容概要 |
| :--- | :--- |
| `fig_sup_training_curves.pdf` | 训练/验证 loss 与 accuracy |
| `fig_sup_metric_curves.pdf` | 验证集 F1、Recall 随 epoch |
| `fig_sup_legacy_ablation_bars.pdf` | 原 `exp2_*` 阶梯消融柱状图 |
| `fig_sup_cross_backbone.pdf` | 不同 backbone 上完整方法对比 |
| `fig_sup_complexity.pdf` | 参数量–FPS 散点 |
| `fig_sup_dataset_distribution.pdf` | 各类别裂缝占比 |

---

## LaTeX 行示例（小数形式，非百分比）

```latex
{latex_example}
```

以上可与 booktabs 表格合并使用；若稿件使用百分比列，请将数值 $\\times 100$ 并配合 `siunitx` 格式化。

---

## English caption stubs

**Table R1.** Test-set performance on SDNET2018-D for cross-entropy baselines, class-weighted CE, robust augmentation, and the proposed Focal loss + CBAM stack ($\\gamma{{=}}2$, $\\alpha{{=}}0.75$).

**Figure R1.** Precision, recall, and F1-score corresponding to Table R1.

**Table R2.** Sensitivity of focal loss focusing parameter $\\gamma$ with fixed $\\alpha{{=}}0.75$.

**Figure R2.** Test accuracy, precision, recall, F1 (left axis, \\%) and throughput FPS (right axis) versus focal loss $\\gamma$ with fixed $\\alpha{{=}}0.75$.
"""

    footer = """

---

*表中数值由脚本生成：`python scripts/export_revision_paper_assets.py`（读取 `runs_v3/*/results.json`）。*
"""

    md_path = REVISION / "paper_additional_results.md"
    md_path.write_text(md.rstrip() + footer, encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Figures -> {FIG_OUT}")


if __name__ == "__main__":
    main()
