"""
Summary table for three Exp1 baselines (ResNet18 / MobileNetV3 / EfficientNet-B0, D+CE):
- From results.json: test FPS, per-image latency, final-epoch train/val loss, test loss and classification metrics
- From complexity.json (thop + pure forward GPU timing): params, FLOPs, forward FPS/latency (may differ from train GPU; see table notes)
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def df_to_markdown(df: pd.DataFrame) -> str:
    """GitHub-style markdown table without tabulate."""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


BASELINES = [
    ("exp1_resnet18_baseline", "ResNet18"),
    ("exp1_mobilenetv3_baseline", "MobileNetV3"),
    ("exp1_efficientnet_baseline", "EfficientNet-B0"),
]


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_complexity_map(complexity_path: Path) -> dict:
    """backbone -> row with attention=='none'"""
    if not complexity_path.exists():
        return {}
    data = load_json(complexity_path)
    out = {}
    for row in data:
        if row.get("attention") == "none":
            out[row["backbone"]] = row
    return out


def build_rows(runs_dir: Path, complexity_map: dict):
    rows = []
    for exp_name, display in BASELINES:
        rpath = runs_dir / exp_name / "results.json"
        if not rpath.exists():
            continue
        r = load_json(rpath)
        cfg = r["config"]
        backbone = cfg["backbone"]
        tm = r["test_metrics"]
        h = r["history"]
        train_loss = h["train_loss"]
        val_loss = h["val_loss"]
        cx = complexity_map.get(backbone, {})

        row = {
            "Model": display,
            "Backbone_key": backbone,
            "Params": cx.get("params_str", "—"),
            "Params_raw": cx.get("params"),
            "FLOPs": cx.get("flops_str", "—"),
            "FLOPs_raw": cx.get("flops"),
            "FPS_fwd": cx.get("fps"),
            "Latency_fwd_ms": cx.get("latency_ms"),
            "FPS_test": round(float(tm.get("fps", 0)), 2),
            "Latency_test_ms": round(float(tm.get("inference_ms", 0)), 3),
            "Train_loss_final": round(float(train_loss[-1]), 6),
            "Val_loss_final": round(float(val_loss[-1]), 6),
            "Test_loss": round(float(tm["loss"]), 6),
            "Accuracy": round(float(tm["accuracy"]), 4),
            "Precision": round(float(tm["precision"]), 4),
            "Recall": round(float(tm["recall"]), 4),
            "F1": round(float(tm["f1"]), 4),
            "Best_val_F1": round(float(r.get("best_val_f1", 0)), 4),
        }
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="./runs_v3")
    p.add_argument(
        "--complexity_json",
        type=str,
        default=None,
        help="complexity.json from python complexity.py (default: runs_v2/complexity/complexity.json)",
    )
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = root / runs_dir

    comp_path = Path(args.complexity_json) if args.complexity_json else root / "runs_v3/complexity/complexity.json"
    if not comp_path.is_absolute():
        comp_path = root / comp_path

    out_dir = Path(args.output_dir) if args.output_dir else runs_dir / "figures"
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cx_map = load_complexity_map(comp_path)
    rows = build_rows(runs_dir, cx_map)
    if not rows:
        print(f"No baseline results under {runs_dir} (expected exp1_*_baseline).")
        return

    df_full = pd.DataFrame(rows)
    df_pub = df_full[
        [
            "Model",
            "Params",
            "FLOPs",
            "FPS_fwd",
            "Latency_fwd_ms",
            "FPS_test",
            "Latency_test_ms",
            "Train_loss_final",
            "Val_loss_final",
            "Test_loss",
            "Accuracy",
            "F1",
        ]
    ].copy()
    df_pub.columns = [
        "Model",
        "Params",
        "FLOPs",
        "FPS (fwd)",
        "Lat_fwd (ms)",
        "FPS (test)",
        "Lat_test (ms)",
        "Train loss (last)",
        "Val loss (last)",
        "Test loss",
        "Acc",
        "F1",
    ]

    csv_path = out_dir / "table_baseline_three_models.csv"
    md_path = out_dir / "baseline_three_models.md"
    df_pub.to_csv(csv_path, index=False)

    note = (
        "- **Params / FLOPs / FPS (fwd)**: from `complexity.py` (thop, `attention=none`, input 224×224), "
        f"file `{comp_path.name}`.\n"
        "- **FPS (test) / Lat_test**: measured during the test phase in each run's `results.json` (GPU/driver dependent).\n"
    )
    df_detail = df_full[
        [
            "Model",
            "Params",
            "FLOPs",
            "FPS_test",
            "Latency_test_ms",
            "Train_loss_final",
            "Val_loss_final",
            "Test_loss",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Best_val_F1",
        ]
    ]
    md = (
        "# Exp1 three-backbone baseline summary (SDNET2018-D, CE)\n\n"
        + note
        + "\n"
        + df_to_markdown(df_pub)
        + "\n\n"
        + "## Full columns (Precision / Recall / best_val_F1)\n\n"
        + df_to_markdown(df_detail)
        + "\n"
    )
    md_path.write_text(md, encoding="utf-8")

    print("\n=== Baseline three models (D + CE) ===\n")
    print(df_pub.to_string(index=False))
    print(f"\n[Saved] {csv_path}")
    print(f"[Saved] {md_path}")
    if not cx_map:
        print(
            f"\n[Warn] {comp_path} not found; Params/FLOPs/FPS(fwd) are placeholders. Run: "
            "python scripts/complexity.py --output_dir runs_v3/complexity"
        )


if __name__ == "__main__":
    main()
