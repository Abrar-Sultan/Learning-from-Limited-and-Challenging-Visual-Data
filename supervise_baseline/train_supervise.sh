#!/bin/bash

# use this setup for CIFAR-10 dataset
python train_supervise.py \
  --train-path '../data/cifer_10/cifar10_train_split.pt' \
  --val-path '../data/cifer_10/cifar10_val_split.pt' \
  --test-path '../data/cifer_10/cifar10_test.pt' \
  --model-name "wrn" \
  --epochs 10 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --save-dir './saved_models'\
  --seed 41\
  --num-class 100 \
  --model-weight './saved_models/wrn_best_model_cifar10.pt'  


# use this setup for CIFAR-100 dataset
  # python train_supervise.py \
  # --train-path '../data/cifer_100/cifar100_train_split.pt' \
  # --val-path '../data/cifer_100/cifar100_val_split.pt' \
  # --test-path '../data/cifer_100/cifar100_test.pt' \
  # --model-name "wrn" \
  # --epochs 100 \
  # --batch-size 128 \
  # --learning-rate 0.001 \
  # --save-dir './saved_models'\
  # --seed 41\
  # --num-class 100 \
  # --model-weight './saved_models/wrn_best_model_cifar100.pt' 
