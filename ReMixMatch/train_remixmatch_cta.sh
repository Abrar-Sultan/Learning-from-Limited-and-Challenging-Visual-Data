#!/bin/bash


# note: before traning or testing using this code, change the normalization value to CIFAR-10 or CIFAR-100 based on the dataset(just uncomment)
# in remixmatch_transform.py (inside "normalize_img" function) and 
# in remixmatch_dataset.py (inside "get_val_dataloader_labeled" function) 


# use the following configuration for the CIFAR-10 dataset. 
 
# change the "train-path-label" and "train-path-unlabel" accordingly (for different label-unlabel data ratio): 
# Here, 
# train-path-label & train-path-unlabel for 10% labeled data is: "ssl_cifar10_label_split_10_pct.pt" & "ssl_cifar10_unlabel_split_10_pct.pt"
# train-path-label & train-path-unlabel for 1% labeled data is: "ssl_cifar10_label_split_1_pct.pt" & "ssl_cifar10_unlabel_split_1_pct.pt"
# train-path-label & train-path-unlabel for 0.08%(approx.) labeled data is: "ssl_cifar10_label_split_0.08_pct.pt" & "ssl_cifar10_unlabel_split_0.08_pct.pt"


python train_remixmatch_cta.py \
  --train-path-label '../data/cifer_10/ssl_cifar10_label_split_10_pct.pt' \
  --train-path-unlabel '../data/cifer_10/ssl_cifar10_unlabel_split_10_pct.pt' \
  --val-path '../data/cifer_10/ssl_cifar10_val_split.pt' \
  --test-path '../data/cifer_10/cifar10_test.pt' \
  --model-name "wrn" \
  --seed 41 \
  --epochs 256 \
  --train-step 1024 \
  --batch-size 64 \
  --num-class 10 \
  --learning-rate 0.002 \
  --k 2 \
  --T 0.5 \
  --lambda_u 1.5 \
  --lambda_u1 0.5 \
  --lambda_r 0.5 \
  --alpha 0.75 \
  --weight-decay 0.001 \
  --ema-decay 0.999 \
  --cta-update 64 \
  --use-cta True \
  --save-dir './saved_models'\
  --model-weight './saved_models/ReMixMatch_wrn_best_model_10_pct_cta_cifar10.pt' 




# use the following configuration for the CIFAR-100 dataset. 

# change the "train-path-label" and "train-path-unlabel" accordingly (for different label-unlabel data ratio):
# Here, 
# train-path-label & train-path-unlabel for 20% labeled data is: "ssl_cifar100_label_split_20_pct.pt" & "ssl_cifar10_unlabel_split_20_pct.pt"
# train-path-label & train-path-unlabel for 10% labeled data is: "ssl_cifar100_label_split_10_pct.pt" & "ssl_cifar10_unlabel_split_10_pct.pt"
# train-path-label & train-path-unlabel for 1% labeled data is: "ssl_cifar100_label_split_1_pct.pt" & "ssl_cifar10_unlabel_split_1_pct.pt"

python train_remixmatch_cta.py \
  --train-path-label '../data/cifer_100/ssl_cifar100_label_split_20_pct.pt' \
  --train-path-unlabel '../data/cifer_100/ssl_cifar100_unlabel_split_20_pct.pt' \
  --val-path '../data/cifer_100/ssl_cifar100_val_split.pt' \
  --test-path '../data/cifer_100/cifar100_test.pt' \
  --model-name "wrn" \
  --seed 41 \
  --epochs 256 \
  --train-step 1024 \
  --batch-size 64 \
  --num-class 100 \
  --learning-rate 0.002 \
  --k 2 \
  --T 0.5 \
  --lambda_u 1.5 \
  --lambda_u1 0.5 \
  --lambda_r 0.5 \
  --alpha 0.75 \
  --weight-decay 0.001 \
  --ema-decay 0.999 \
  --cta-update 64 \
  --use-cta True \
  --save-dir './saved_models'\
  --model-weight './saved_models/ReMixMatch_wrn_best_model_20_pct_cta_cifar100.pt' 