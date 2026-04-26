#!/bin/bash
set -e

echo "====== EXPERIMENT 1: BASELINE (delta_gamma=0.5) ======"
echo "Start time: $(date)"
python train.py \
  --problem_type SC \
  --gamma_init 0.0 \
  --delta_gamma 0.5 \
  --num_epochs 80 \
  --val_every 5 \
  --n_eval_samples 100 \
  --seed 0 \
  2>&1 | tee exp1_baseline.log

echo ""
echo "====== EXPERIMENT 2: omw=0.5 + lema=0.3 ======"
echo "Start time: $(date)"
python train.py \
  --problem_type SC \
  --gamma_init 0.0 \
  --delta_gamma 0.5 \
  --obj_margin_weight 0.5 \
  --lambda_ema_alpha 0.3 \
  --num_epochs 80 \
  --val_every 5 \
  --n_eval_samples 100 \
  --seed 0 \
  2>&1 | tee exp2_omw_lema.log

echo ""
echo "====== EXPERIMENT 3: omw=0.5 + lema=0.3 + cln ======"
echo "Start time: $(date)"
python train.py \
  --problem_type SC \
  --gamma_init 0.0 \
  --delta_gamma 0.5 \
  --obj_margin_weight 0.5 \
  --lambda_ema_alpha 0.3 \
  --cons_loss_normalize \
  --num_epochs 80 \
  --val_every 5 \
  --n_eval_samples 100 \
  --seed 0 \
  2>&1 | tee exp3_omw_lema_cln.log

echo ""
echo "====== ALL EXPERIMENTS DONE ======"
echo "End time: $(date)"
