"""
Aggregate all runs_v3/rev_* experiment results: performance table + confusion matrix grid.
"""

import json
import math
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

RUNS_V3 = Path(__file__).resolve().parent.parent / "runs_v3"
OUT_DIR = RUNS_V3 / "rev_aggregate"

# Suggested display order (baseline first, then grouped by theme)
ORDER = [
    "rev_baseline_ce",
    "rev_cbam_only",
    "rev_wce",
    "rev_robust_aug",
    "rev_focal_gamma_1",
    "rev_focal_gamma_2",
    "rev_focal_gamma_3",
    "rev_focal_gamma_5",
    "rev_focal_no_ra",
]

CLASS_NAMES = ["Non-crack", "Crack"]

# Loss-strategy comparison (matches revision Table R1 first four rows: CE / WCE / robust CE / Focal)
LOSS_STRATEGY_SPECS: list[tuple[str, str]] = [
    ("rev_baseline_ce", "Baseline CE\n(+sampler)"),
    ("rev_wce", "Weighted CE"),
    ("rev_robust_aug", "CE + robust aug."),
    ("rev_focal_gamma_2", r"Focal + CBAM + robust\n($\gamma{=}2$)"),
]


def discover_rev_runs():
    names = sorted(d.name for d in RUNS_V3.iterdir() if d.is_dir() and d.name.startswith("rev_"))
    # Place known order first; append any extra rev_* folders not in ORDER
    ordered = [n for n in ORDER if n in names]
    ordered += [n for n in names if n not in ordered]
    return ordered


def load_results(exp_name: str):
    path = RUNS_V3 / exp_name / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def build_table(exp_names):
    rows = []
    for name in exp_names:
        r = load_results(name)
        if r is None:
            continue
        tm = r.get("test_metrics", {})
        cfg = r.get("config", {})
        rows.append(
            {
                "run": name,
                "loss": tm.get("loss"),
                "accuracy": tm.get("accuracy"),
                "precision": tm.get("precision"),
                "recall": tm.get("recall"),
                "f1": tm.get("f1"),
                "fps": tm.get("fps"),
                "inference_ms": tm.get("inference_ms"),
                "best_val_f1": r.get("best_val_f1"),
                "backbone": cfg.get("backbone"),
                "loss_fn": cfg.get("loss"),
                "attention": cfg.get("attention"),
                "use_robust_aug": cfg.get("use_robust_aug"),
            }
        )
    return pd.DataFrame(rows)


def plot_confusion_grid(
    exp_names: list[str],
    out_prefix: Path,
    *,
    subplot_titles: Optional[List[str]] = None,
    suptitle: Optional[str] = None,
):
    """Plot confusion matrices side-by-side. `subplot_titles` overrides auto titles from folder names."""
    n = len(exp_names)
    if n == 0:
        return
    if subplot_titles is not None and len(subplot_titles) != n:
        raise ValueError("subplot_titles must match len(exp_names)")

    # Default wide 3-column grid; use 2×2 when ≤4 runs for readability
    if n <= 4:
        ncols, nrows = 2, int(math.ceil(n / 2))
    else:
        ncols = 3
        nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.85 * nrows))
    axes = np.atleast_2d(axes)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes[0, 0]]])
    flat = axes.flatten()

    vmax_global = 0
    cms = []
    for name in exp_names:
        r = load_results(name)
        if r is None:
            cms.append(None)
            continue
        cm = np.array(r["test_metrics"]["confusion_matrix"], dtype=float)
        cms.append(cm)
        vmax_global = max(vmax_global, cm.max())

    for idx, name in enumerate(exp_names):
        ax = flat[idx]
        cm = cms[idx]
        if cm is None:
            ax.set_visible(False)
            continue
        ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax_global)
        title = subplot_titles[idx] if subplot_titles else name.replace("rev_", "")
        ax.set_title(title, fontsize=10)
        tick_marks = np.arange(cm.shape[0])
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{int(cm[i, j])}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10,
                )

    for j in range(len(exp_names), len(flat)):
        flat[j].set_visible(False)

    norm = mpl.colors.Normalize(vmin=0, vmax=vmax_global)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="Blues")
    sm.set_array([])
    visible_axes = [flat[i] for i in range(len(exp_names))]
    cbar = fig.colorbar(sm, ax=visible_axes, fraction=0.02, pad=0.03)
    cbar.set_label("Count")

    fig.suptitle(
        suptitle or "Test confusion matrices (SDNET2018-D, rev_* runs)",
        fontsize=14,
        y=1.02,
    )
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.06, top=0.92, hspace=0.4, wspace=0.35)
    for ext in ("png", "pdf"):
        p = out_prefix.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exp_names = discover_rev_runs()
    if not exp_names:
        raise SystemExit(f"No rev_* folders under {RUNS_V3}")

    df = build_table(exp_names)
    csv_path = OUT_DIR / "rev_performance_table.csv"
    df.to_csv(csv_path, index=False)

    print(df.to_string(index=False))
    print(f"\nWrote {csv_path}")

    loss_names = [e for e, _ in LOSS_STRATEGY_SPECS]
    loss_titles = [t for _, t in LOSS_STRATEGY_SPECS]
    out_cm = OUT_DIR / "rev_confusion_matrices_grid"
    plot_confusion_grid(
        loss_names,
        out_cm,
        subplot_titles=loss_titles,
        suptitle="Loss-strategy comparison — test confusion matrices (SDNET2018-D)",
    )
    print(f"Wrote {out_cm.with_suffix('.png')} and .pdf (CE / WCE / CE+robust aug / focal γ=2)")


if __name__ == "__main__":
    main()
