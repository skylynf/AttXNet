#!/usr/bin/env python3
"""
Print SDNET/Maguiere subset statistics: class balance per category (D,P,W) and
stratified train/val/test splits (same logic as dataset.split_dataset).

Optional: save a simple bar chart for papers (--save_fig DIR).

Example (run from repository root):
  python scripts/summarize_dataset_domain.py --data_root ../dataset/DATA_Maguire_20180517_ALL
  python scripts/summarize_dataset_domain.py --data_root ... --save_fig ./runs_v3/figures
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attxnet.dataset import collect_sdnet_files, split_dataset


def _counts(samples: List[Tuple[str, int]]) -> Tuple[int, int]:
    crack = sum(1 for _, l in samples if l == 1)
    return crack, len(samples) - crack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_fig", type=str, default=None, help="Directory to save PNG/PDF figures")
    args = ap.parse_args()

    random.seed(args.seed)

    print("=== Per-category totals (all images before split) ===\n")
    for cat, name in [("D", "Deck"), ("P", "Pavement"), ("W", "Wall")]:
        samples = collect_sdnet_files(args.data_root, (cat,))
        c, u = _counts(samples)
        tot = c + u
        pct = 100.0 * c / tot if tot else 0.0
        print(f"  {cat} ({name}): crack={c}, non-crack={u}, total={tot}")
        print(f"           crack proportion = {pct:.2f}%   (non-crack : crack ≈ {u/c:.3f}:1)\n")

    for label, cats in [
        ("Deck-only (main experiments)", ("D",)),
        ("Deck+Pavement+Wall (cross-category)", ("D", "P", "W")),
    ]:
        samples = collect_sdnet_files(args.data_root, cats)
        c_all, u_all = _counts(samples)
        print(f"=== {label} ===")
        print(
            f"  Pooled: crack={c_all}, non-crack={u_all}, "
            f"crack%={100*c_all/(c_all+u_all):.2f}\n"
        )
        train, val, test = split_dataset(samples, seed=args.seed)
        for split_name, sp in [("train", train), ("val", val), ("test", test)]:
            c, u = _counts(sp)
            print(
                f"  {split_name}: n={len(sp)}, crack={c}, non-crack={u}, "
                f"crack%={100*c/(c+u):.2f},  non:crack={u/c:.3f}:1"
            )
        print()

    if args.save_fig:
        os.makedirs(args.save_fig, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as e:
            print("matplotlib/numpy required for --save_fig:", e)
            return

        cats = ["D", "P", "W"]
        names = ["Deck", "Pavement", "Wall"]
        crack_pct = []
        for cat in cats:
            samples = collect_sdnet_files(args.data_root, (cat,))
            c, u = _counts(samples)
            crack_pct.append(100.0 * c / (c + u))

        fig, ax = plt.subplots(figsize=(5, 3.5))
        x = np.arange(len(cats))
        ax.bar(x, crack_pct, color=["#2c7fb8", "#7fcdbb", "#f0ad4e"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n({n})" for c, n in zip(cats, names)])
        ax.set_ylabel("Crack samples (%)")
        ax.set_title("Class balance: crack proportion per structural category")
        ax.set_ylim(0, max(crack_pct) * 1.25)
        for i, v in enumerate(crack_pct):
            ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
        fig.tight_layout()
        base = os.path.join(args.save_fig, "fig_dataset_crack_share_by_category")
        fig.savefig(base + ".png", dpi=150)
        fig.savefig(base + ".pdf")
        plt.close()
        print(f"Saved {base}.png / .pdf")


if __name__ == "__main__":
    main()
