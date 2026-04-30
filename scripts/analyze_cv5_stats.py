"""
Aggregate stratified K-fold CV results (train.py with --fold) and paired tests.

Reports mean ± std across folds, Wilcoxon signed-rank & paired t-test for
paired fold scores (recommended for n=5: report both; primary metric: F1).
Friedman test applies when comparing 3+ methods on the same folds — enabled
automatically if three or more experiment directories are detected.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


METRIC_KEYS = ("f1", "accuracy", "precision", "recall")


def load_fold_metrics(runs_dir: Path, exp_name: str) -> Tuple[np.ndarray, List[int]]:
    """Return (matrix n_folds x len(METRIC_KEYS)), fold_indices in sorted order."""
    exp_path = runs_dir / exp_name
    if not exp_path.is_dir():
        raise FileNotFoundError(f"Missing experiment directory: {exp_path}")

    fold_dirs = sorted(
        [p for p in exp_path.iterdir() if p.is_dir() and p.name.startswith("fold_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* under {exp_path}")

    rows = []
    fold_ids = []
    for fd in fold_dirs:
        rf = fd / "results.json"
        if not rf.exists():
            raise FileNotFoundError(f"Missing {rf}")
        with open(rf) as f:
            data = json.load(f)
        tm = data["test_metrics"]
        rows.append([float(tm[k]) for k in METRIC_KEYS])
        fold_ids.append(int(fd.name.split("_")[1]))

    return np.asarray(rows, dtype=np.float64), fold_ids


def cohens_dz_paired(diff: np.ndarray) -> float:
    """Cohen's d_z for paired samples: mean(d) / std(d)."""
    d = np.asarray(diff, dtype=np.float64)
    sd = float(np.std(d, ddof=1))
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(d) / sd)


def paired_tests(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """a, b shape (n,); paired tests B vs A on the same folds (alternative: two-sided)."""
    d = b - a
    n = len(d)
    out: Dict[str, float] = {
        "n_pairs": float(n),
        "mean_diff": float(np.mean(d)),
        "std_diff": float(np.std(d, ddof=1)) if n > 1 else 0.0,
        "dz": cohens_dz_paired(d),
        "wilcoxon_stat": float("nan"),
    }
    if n < 2:
        out["wilcoxon_p"] = float("nan")
        out["ttest_p"] = float("nan")
        out["wilcoxon_stat"] = float("nan")
        return out
    if np.allclose(d, 0.0, atol=0.0):
        out["ttest_p"] = 1.0
    else:
        try:
            tt = stats.ttest_rel(b, a, alternative="two-sided")
            out["ttest_p"] = float(tt.pvalue)
        except TypeError:
            tt = stats.ttest_rel(b, a)
            out["ttest_p"] = float(tt.pvalue)
        if math.isnan(out["ttest_p"]):
            out["ttest_p"] = 1.0

    try:
        wr = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
    except TypeError:
        try:
            wr = stats.wilcoxon(d, zero_method="wilcox")
        except Exception:
            wr = stats.wilcoxon(d)
    except ValueError:
        # zero diffs across folds
        out["wilcoxon_p"] = 1.0
        out["wilcoxon_stat"] = 0.0
        return out
    out["wilcoxon_p"] = float(wr.pvalue)
    out["wilcoxon_stat"] = float(getattr(wr, "statistic", float("nan")))
    return out


def friedman_blocking(matrices: List[np.ndarray], metric_col: int) -> Optional[Dict[str, float]]:
    """Matrices: same n_folds x n_metrics from each method."""
    if len(matrices) < 3:
        return None
    x = np.column_stack([m[:, metric_col] for m in matrices])
    stat, p = stats.friedmanchisquare(*[x[:, j] for j in range(x.shape[1])])
    return {"statistic": float(stat), "p_value": float(p), "df": float(x.shape[1] - 1)}


def discover_experiment_names(runs_dir: Path) -> List[str]:
    names = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        if list(child.glob("fold_*/results.json")):
            names.append(child.name)
    return names


def main():
    ap = argparse.ArgumentParser(description="Analyze stratified CV runs and paired significance tests.")
    ap.add_argument("--runs_dir", type=str, default="./runs_cv5", help="Same as train --output_dir for CV.")
    ap.add_argument(
        "--methods",
        type=str,
        default=None,
        help=(
            'Comma-separated exp folder names under runs_dir (default: auto-detect dirs with fold_*/results.json). '
            'Example: cv5_resnet18_baseline,cv5_resnet18_full_cbam'
        ),
    )
    ap.add_argument("--out_csv", type=str, default=None, help="Write summary CSV (default: <runs>/cv5_summary.csv).")
    ap.add_argument("--out_tests_csv", type=str, default=None, help="Paired-tests CSV.")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.is_dir():
        raise SystemExit(f"runs_dir does not exist: {runs_dir}")

    if args.methods:
        method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        method_names = discover_experiment_names(runs_dir)

    if len(method_names) < 2:
        raise SystemExit("Need at least two experiment folders with CV results (--methods or auto-discovery).")

    matrices: Dict[str, np.ndarray] = {}
    fold_ids_all: Optional[List[int]] = None

    for m in method_names:
        mtx, folds = load_fold_metrics(runs_dir, m)
        matrices[m] = mtx
        if fold_ids_all is None:
            fold_ids_all = folds
        elif tuple(folds) != tuple(fold_ids_all):
            raise SystemExit(f"Fold mismatch for '{m}' vs baseline listing: {folds} vs {fold_ids_all}")

    n_fold = matrices[method_names[0]].shape[0]
    summary_rows = []

    labels_map = {}
    print("\n=== Per-fold metrics (same row index = paired by fold ID) ===\n")
    for name in method_names:
        X = matrices[name]
        print(f"-- {name}")
        folds = fold_ids_all or list(range(len(X)))
        for i in range(len(X)):
            fid = folds[i]
            rr = dict(zip(METRIC_KEYS, X[i].tolist()))
            print(
                f"  fold_{fid}: "
                + "  ".join(f"{k}={rr[k]:.4f}" for k in METRIC_KEYS),
            )

        for j, key in enumerate(METRIC_KEYS):
            col = X[:, j]
            summary_rows.append(
                {
                    "method": name,
                    "label": name,
                    "metric": key,
                    "mean": float(np.mean(col)),
                    "std": float(np.std(col, ddof=1)) if n_fold > 1 else 0.0,
                    "n_folds": n_fold,
                }
            )

    df_sum = pd.DataFrame(summary_rows)
    print("\n=== Mean ± std across folds ===\n")
    pt = df_sum.pivot_table(index="method", columns="metric", values=["mean", "std"])
    print(pt.to_string())

    out_csv = args.out_csv or str(runs_dir / "cv5_summary.csv")
    df_sum.to_csv(out_csv, index=False)
    print(f"\n[Saved] {out_csv}")

    # Paired tests: first method as reference (e.g. baseline), rest vs first
    ref = method_names[0]
    A = matrices[ref]
    test_rows = []
    for other in method_names[1:]:
        B = matrices[other]
        for j, key in enumerate(METRIC_KEYS):
            pr = paired_tests(A[:, j], B[:, j])
            test_rows.append(
                {
                    "reference": ref,
                    "compare": other,
                    "metric": key,
                    **pr,
                }
            )

    df_tests = pd.DataFrame(test_rows)
    out_t = args.out_tests_csv or str(runs_dir / "cv5_paired_tests.csv")
    df_tests.to_csv(out_t, index=False)

    print("\n=== Paired significance (same folds): Wilcoxon & paired t-test ===")
    print(f"Reference: {ref}\n")
    for other in method_names[1:]:
        sub = df_tests[df_tests["compare"] == other]
        print(f"vs {other}")
        for key in METRIC_KEYS:
            row = sub[sub["metric"] == key].iloc[0]
            dzv = row["dz"]
            dz_str = "nan" if (isinstance(dzv, float) and math.isnan(dzv)) else f"{dzv:.3f}"
            print(
                f"  {key:9s}  mean_diff={row['mean_diff']:+.6f}  "
                f"Wilcoxon p={row['wilcoxon_p']:.4g}  t-test p={row['ttest_p']:.4g}  "
                f"d_z={dz_str}",
            )
        print()

    print(f"[Saved] {out_t}")

    fr = friedman_blocking([matrices[m] for m in method_names], METRIC_KEYS.index("f1"))
    if fr is not None:
        print("=== Friedman test on F1 across methods (same folds) ===")
        print(f"  chi^2={fr['statistic']:.4f}  p={fr['p_value']:.4g}  (k={len(method_names)} methods, n={n_fold} folds)\n")

    print("Note: With only two methods, prefer Wilcoxon / paired t on fold scores; Friedman needs 3+ methods.")


if __name__ == "__main__":
    main()
