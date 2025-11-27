#!/bin/bash


# change the "train-path-label" and "train-path-unlabel" accordingly (for different label-unlabel data ratio): 
# Here, 
# train-path-label & train-path-unlabel for 10% labeled data is: "ssl_cifar10_label_split_10_pct.pt" & "ssl_cifar10_unlabel_split_10_pct.pt"
# train-path-label & train-path-unlabel for 1% labeled data is: "ssl_cifar10_label_split_1_pct.pt" & "ssl_cifar10_unlabel_split_1_pct.pt"
# train-path-label & train-path-unlabel for 0.08%(approx.) labeled data is: "ssl_cifar10_label_split_0.08_pct.pt" & "ssl_cifar10_unlabel_split_0.08_pct.pt"


python train_mean_teacher.py \
  --train-path-label '../data/cifer_10/ssl_cifar10_label_split_10_pct.pt' \
  --train-path-unlabel '../data/cifer_10/ssl_cifar10_unlabel_split_10_pct.pt' \
  --val-path '../data/cifer_10/ssl_cifar10_val_split.pt' \
  --test-path '../data/cifer_10/cifar10_test.pt' \
  --model-name "wrn" \
  --epochs 180 \
  --train-step 149 \
  --batch-size 128 \
  --label-unlabel-ratio 4 \
  --learning-rate 0.1 \
  --initial-learning-rate 0.0 \
  --lr-rampup 5 \
  --lr-rampdown-epochs 210 \
  --ema-decay 0.999 \
  --consistency 100.0 \
  --consistency-rampup 5 \
  --beta 0.9 \
  --weight-decay 0.0001 \
  --save-dir './saved_models' \
  --seed 41 \
  --num-class 10 \
  --model-weight './saved_models/mean_teacher_wrn_ema_best_model_10_pct.pt'
