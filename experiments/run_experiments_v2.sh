#!/bin/bash
# ============================================================================
# Bridge Crack Detection - Experiment Suite V2 (Fixed)
#
# Key fix: When using Focal Loss, DISABLE WeightedRandomSampler to avoid
# double class-balancing (Sampler already rebalances to 50/50, then
# alpha=0.75 further weights crack 3x → massive over-correction).
#
# Strategy: Focal Loss alone handles BOTH class imbalance (via alpha)
# AND hard-sample mining (via gamma). No sampler needed.
# ============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="../dataset/DATA_Maguire_20180517_ALL"
CATEGORIES="D"
EPOCHS=40
BATCH=64
IMG_SIZE=224
OUTPUT="./runs_v2"

# ────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Backbone Comparison (CE + WeightedSampler, standard aug)
# ────────────────────────────────────────────────────────────────────────────

echo "=== Experiment 1: Backbone Comparison ==="

python train.py \
    --data_root $DATA_ROOT --categories $CATEGORIES \
    --backbone resnet18 --attention none --loss ce \
    --epochs $EPOCHS --batch_size $BATCH --img_size $IMG_SIZE \
    --gpu 0 --output_dir $OUTPUT \
    --exp_name exp1_resnet18_baseline &

python train.py \
    --data_root $DATA_ROOT --categories $CATEGORIES \
    --backbone mobilenetv3 --attention none --loss ce \
    --epochs $EPOCHS --batch_size $BATCH --img_size $IMG_SIZE \
    --gpu 1 --output_dir $OUTPUT \
    --exp_name exp1_mobilenetv3_baseline &

python train.py \
    --data_root $DATA_ROOT --categories $CATEGORIES \
    --backbone efficientnet --attention none --loss ce \
    --epochs $EPOCHS --batch_size $BATCH --img_size $IMG_SIZE \
    --gpu 2 --output_dir $OUTPUT \
    --exp_name exp1_efficientnet_baseline &

wait
echo "=== Experiment 1 Done ==="

# ────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Ablation Study (MobileNetV3)
#
# Baseline:           CE + WeightedSampler + standard aug
# +Robust Aug:        CE + WeightedSampler + robust aug
# +RA +Focal:         Focal(α=0.75,γ=2) + NO sampler + robust aug
# +RA +Focal +CBAM:   Focal(α=0.75,γ=2) + NO sampler + robust aug + CBAM
# ────────────────────────────────────────────────────────────────────────────

echo "=== Experiment 2: Ablation Study ==="

# Baseline already covered by exp1_mobilenetv3_baseline

# +Robust Aug (CE + sampler still on)
python train.py \
    --data_root $DATA_ROOT --categories $CATEGORIES \
    --backbone mobilenetv3 --attention none --loss ce \
    --use_robust_aug \
    --epochs $EPOCHS --batch_size $BATCH --img_size $IMG_SIZE \
    --gpu 0 --output_dir $OUTPUT \
    --exp_name exp2_ablation_robust &

# +Robust Aug + Focal Loss (NO sampler → let Focal handle balance)
python train.py \
    --data_root $DATA_ROOT --categories $CATEGORIES \
    --backbone mobilenetv3 --attention none --loss focal \
    --focal_alpha 0.75 --focal_gamma 2.0 \
    --use_robust_aug --no_weighted_sampler \
    --epochs $EPOCHS --batch_size $BATCH --img_size $IMG_SIZE \
    --gpu 1 --output_dir $OUTPUT \
    --exp_name exp2_ablation_robust_focal &

# +Robust Aug + Focal + CBAM (Ours full method)
python train.py \
    --data_root $DATA_ROOT --categories $CATEGORIES \
    --backbone mobilenetv3 --attention cbam --loss focal \
    --focal_alpha 0.75 --focal_gamma 2.0 \
    --use_robust_aug --no_weighted_sampler \
    --epochs $EPOCHS --batch_size $BATCH --img_size $IMG_SIZE \
    --gpu 2 --output_dir $OUTPUT \
    --exp_name exp2_ablation_full_cbam &

wait
echo "=== Experiment 2 Done ==="

# ────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2b: Attention comparison & cross-backbone full method
# ────────────────────────────────────────────────────────────────────────────

echo "=== Experiment 2b: Attention + Cross-backbone ==="

# CA variant on MobileNetV3
python train.py \
    --data_root $DATA_ROOT --categories $CATEGORIES \
    --backbone mobilenetv3 --attention ca --loss focal \
    --focal_alpha 0.75 --focal_gamma 2.0 \
    --use_robust_aug --no_weighted_sampler \
    --epochs $EPOCHS --batch_size $BATCH --img_size $IMG_SIZE \
    --gpu 0 --output_dir $OUTPUT \
    --exp_name exp2b_ablation_full_ca &

# Full method on ResNet18
python train.py \
    --data_root $DATA_ROOT --categories $CATEGORIES \
    --backbone resnet18 --attention cbam --loss focal \
    --focal_alpha 0.75 --focal_gamma 2.0 \
    --use_robust_aug --no_weighted_sampler \
    --epochs $EPOCHS --batch_size $BATCH --img_size $IMG_SIZE \
    --gpu 1 --output_dir $OUTPUT \
    --exp_name exp2b_resnet18_full &

# Full method on EfficientNet
python train.py \
    --data_root $DATA_ROOT --categories $CATEGORIES \
    --backbone efficientnet --attention cbam --loss focal \
    --focal_alpha 0.75 --focal_gamma 2.0 \
    --use_robust_aug --no_weighted_sampler \
    --epochs $EPOCHS --batch_size $BATCH --img_size $IMG_SIZE \
    --gpu 2 --output_dir $OUTPUT \
    --exp_name exp2b_efficientnet_full &

wait
echo "=== Experiment 2b Done ==="

# ────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: Complexity Analysis
# ────────────────────────────────────────────────────────────────────────────

echo "=== Experiment 3: Complexity Analysis ==="
python scripts/complexity.py --gpu 0 --output_dir $OUTPUT/complexity
echo "=== Experiment 3 Done ==="

echo ""
echo "============================================"
echo "All experiments completed!"
echo "Run: python scripts/analyze_results.py --runs_dir $OUTPUT --output_dir $OUTPUT/figures"
echo "============================================"
