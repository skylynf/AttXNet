#!/bin/bash
# ============================================================================
# Post-experiment V3: generate all tables, figures, Grad-CAM, and complexity.
# ============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="../dataset/DATA_Maguire_20180517_ALL"
RUNS="./runs_v3"

echo "=== Generating tables & figures ==="
python scripts/analyze_results_v3.py --runs_dir $RUNS --output_dir $RUNS/figures

echo ""
echo "=== Generating Grad-CAM (ResNet18: Baseline vs Ours) ==="
python scripts/gradcam_vis.py \
    --data_root $DATA_ROOT \
    --baseline_dir $RUNS/exp1_resnet18_baseline \
    --ours_dir $RUNS/exp2_ablation_full_cbam \
    --output_dir $RUNS/figures/gradcam \
    --n_images 8 \
    --gpu 0

echo ""
echo "============================================"
echo "All outputs saved to $RUNS/figures/"
echo "  Tables: table1-5 (.csv / .tex)"
echo "  Figures: fig_training_curves, fig_ablation_bars, fig_metric_curves"
echo "  Grad-CAM: gradcam/gradcam_*.png"
echo "============================================"
