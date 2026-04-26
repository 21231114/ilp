#!/bin/bash
set -e

# Match existing experiment parameters:
# tau=1.0, tau_min=0.9999, inner_steps=240, entropy=0.0
# es_xi_threshold=1.1, es_xi_threshold2=1.1, threshold2_on=valid
# Use gamma_init=15.0 + delta_gamma=0.3 to reach phase transition (gamma~19.56) by epoch ~15
# and get 65 epochs of meaningful margin territory by epoch 80

COMMON="--problem_type SC --tau 1.0 --tau_min 0.9999 --inner_steps 240 \
  --entropy_weight 0.0 --es_xi_threshold 1.1 --es_xi_threshold2 1.1 \
  --threshold2_on valid --gamma_init 15.0 --delta_gamma 0.3 \
  --num_epochs 80 --val_every 5 --n_eval_samples 100 --seed 0"

echo "====== EXP1: BASELINE ======"
echo "Start: $(date)"
python train.py $COMMON 2>&1 | tee exp1_baseline_v2.log
echo "End: $(date)"

echo ""
echo "====== EXP2: omw=0.5 + lema=0.3 ======"
echo "Start: $(date)"
python train.py $COMMON --obj_margin_weight 0.5 --lambda_ema_alpha 0.3 2>&1 | tee exp2_omw_lema_v2.log
echo "End: $(date)"

echo ""
echo "====== EXP3: omw=0.5 + lema=0.3 + cln ======"
echo "Start: $(date)"
python train.py $COMMON --obj_margin_weight 0.5 --lambda_ema_alpha 0.3 --cons_loss_normalize 2>&1 | tee exp3_omw_lema_cln_v2.log
echo "End: $(date)"

echo ""
echo "====== ALL DONE ======"
