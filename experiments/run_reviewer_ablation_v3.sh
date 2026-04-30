#!/bin/bash
# ============================================================================
# Reviewer-requested experiments (revision):
#   1) ResNet18 + Cross-Entropy (+WeightedSampler, standard aug)
#   2) ResNet18 + weighted CE (inverse frequency class weights, no sampler)
#   3) ResNet18 + CE + robust augmentation (+WeightedSampler)
#   4) ResNet18 + Focal Loss only (no robust aug, no CBAM; γ=2, α=0.75, +WeightedSampler)
#   5) ResNet18 + CE + CBAM only (standard aug, +WeightedSampler)
#   6) ResNet18 + Focal Loss + CBAM + robust aug (full branch, via γ=2 in sweep)
#   7–10) Focal Loss gamma sweep: 1.0, 2.0, 3.0, 5.0 (alpha=0.75 fixed)
#
# To parallelize “Focal-only (no RA)” vs “CBAM-only” on two GPUs, use the echo commands at the end.
#
# After completion:
#   ./.venv/bin/python scripts/analyze_results_v3.py --runs_dir ./runs_v3
# Tables: table_reviewer_ablation.csv, table_reviewer_focal_sweep.csv
# ============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

DATA_ROOT="../dataset/DATA_Maguire_20180517_ALL"
EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-64}"
IMG_SIZE="${IMG_SIZE:-224}"
OUTPUT="./runs_v3"
GPU="${GPU:-0}"

cd "$ROOT"

run_train() {
  "$PYTHON" train.py "$@"
}

echo "=== Reviewer ablation (first 3 runs; full method γ=2 is shared with focal sweep) ==="

run_train \
  --data_root "$DATA_ROOT" --categories D \
  --backbone resnet18 --attention none --loss ce \
  --epochs "$EPOCHS" --batch_size "$BATCH" --img_size "$IMG_SIZE" \
  --gpu "$GPU" --output_dir "$OUTPUT" \
  --exp_name rev_baseline_ce

run_train \
  --data_root "$DATA_ROOT" --categories D \
  --backbone resnet18 --attention none --loss wce \
  --no_weighted_sampler \
  --epochs "$EPOCHS" --batch_size "$BATCH" --img_size "$IMG_SIZE" \
  --gpu "$GPU" --output_dir "$OUTPUT" \
  --exp_name rev_wce

run_train \
  --data_root "$DATA_ROOT" --categories D \
  --backbone resnet18 --attention none --loss ce \
  --use_robust_aug \
  --epochs "$EPOCHS" --batch_size "$BATCH" --img_size "$IMG_SIZE" \
  --gpu "$GPU" --output_dir "$OUTPUT" \
  --exp_name rev_robust_aug

echo "=== Extra reviewer controls: Focal-only (no RA / no CBAM); CBAM-only (CE + std aug) ==="

run_train \
  --data_root "$DATA_ROOT" --categories D \
  --backbone resnet18 --attention none --loss focal \
  --focal_alpha 0.75 --focal_gamma 2.0 \
  --epochs "$EPOCHS" --batch_size "$BATCH" --img_size "$IMG_SIZE" \
  --gpu "$GPU" --output_dir "$OUTPUT" \
  --exp_name rev_focal_no_ra

run_train \
  --data_root "$DATA_ROOT" --categories D \
  --backbone resnet18 --attention cbam --loss ce \
  --epochs "$EPOCHS" --batch_size "$BATCH" --img_size "$IMG_SIZE" \
  --gpu "$GPU" --output_dir "$OUTPUT" \
  --exp_name rev_cbam_only

echo "=== Focal Loss gamma sweep (4 runs, α=0.75); γ=2 corresponds to reviewer 'Focal+CBAM' row ==="

declare -a GAMMA_EXP=(
  "1.0:rev_focal_gamma_1"
  "2.0:rev_focal_gamma_2"
  "3.0:rev_focal_gamma_3"
  "5.0:rev_focal_gamma_5"
)
for item in "${GAMMA_EXP[@]}"; do
  GAMMA="${item%%:*}"
  EXPNAME="${item##*:}"
  run_train \
    --data_root "$DATA_ROOT" --categories D \
    --backbone resnet18 --attention cbam --loss focal \
    --focal_alpha 0.75 --focal_gamma "$GAMMA" \
    --use_robust_aug --no_weighted_sampler \
    --epochs "$EPOCHS" --batch_size "$BATCH" --img_size "$IMG_SIZE" \
    --gpu "$GPU" --output_dir "$OUTPUT" \
    --exp_name "$EXPNAME"
done

echo ""
echo "============================================"
echo "Reviewer experiments completed."
echo "Aggregate: \"$PYTHON\" scripts/analyze_results_v3.py --runs_dir $OUTPUT"
echo ""
echo "Parallel rerun example (terminal A: --gpu 1 Focal-only no RA; terminal B: --gpu 2 CBAM-only):"
echo "  cd \"$ROOT\" && \"$PYTHON\" train.py --data_root \"$DATA_ROOT\" --categories D --backbone resnet18 --attention none --loss focal --focal_alpha 0.75 --focal_gamma 2.0 --epochs \"$EPOCHS\" --batch_size \"$BATCH\" --img_size \"$IMG_SIZE\" --gpu 1 --output_dir \"$OUTPUT\" --exp_name rev_focal_no_ra"
echo "  cd \"$ROOT\" && \"$PYTHON\" train.py --data_root \"$DATA_ROOT\" --categories D --backbone resnet18 --attention cbam --loss ce --epochs \"$EPOCHS\" --batch_size \"$BATCH\" --img_size \"$IMG_SIZE\" --gpu 2 --output_dir \"$OUTPUT\" --exp_name rev_cbam_only"
echo "============================================"
