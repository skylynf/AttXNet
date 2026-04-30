#!/bin/bash
# ============================================================================
# Stratified 5-fold CV — minimal key comparisons (SDNET2018-D only)
#
# Runs only the principal ablation pair aligned with Exp2 in run_experiments_v3.sh:
#   - Baseline: ResNet18 + CE + weighted sampler + standard aug
#   - Full:     ResNet18 + Robust aug + Focal + CBAM (no weighted sampler)
#
# Same folds for all configs (fixed --cv_seed). After completion:
#   python scripts/analyze_cv5_stats.py --runs_dir $OUTPUT
# ============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="../dataset/DATA_Maguire_20180517_ALL"
EPOCHS=40
BATCH=64
IMG_SIZE=224
OUTPUT="./runs_cv5"
N_FOLDS=5
CV_SEED=42
GPU=0

mkdir -p "$OUTPUT"

echo "=== Stratified ${N_FOLDS}-fold CV | key configs | output: $OUTPUT ==="

for ((FOLD=0; FOLD<N_FOLDS; FOLD++)); do
  echo ""
  echo "--- Fold $((FOLD+1))/${N_FOLDS} (index=${FOLD}) ---"

  python train.py \
    --data_root "$DATA_ROOT" --categories D \
    --backbone resnet18 --attention none --loss ce \
    --epochs $EPOCHS --batch_size $BATCH --img_size $IMG_SIZE \
    --gpu $GPU --output_dir "$OUTPUT" \
    --exp_name cv5_resnet18_baseline \
    --fold $FOLD --n_folds $N_FOLDS --cv_seed $CV_SEED

  python train.py \
    --data_root "$DATA_ROOT" --categories D \
    --backbone resnet18 --attention cbam --loss focal \
    --focal_alpha 0.75 --focal_gamma 2.0 \
    --use_robust_aug --no_weighted_sampler \
    --epochs $EPOCHS --batch_size $BATCH --img_size $IMG_SIZE \
    --gpu $GPU --output_dir "$OUTPUT" \
    --exp_name cv5_resnet18_full_cbam \
    --fold $FOLD --n_folds $N_FOLDS --cv_seed $CV_SEED
done

echo ""
echo "============================================"
echo "5-fold CV finished."
echo "Statistical summary: python scripts/analyze_cv5_stats.py --runs_dir $OUTPUT"
echo "============================================"
