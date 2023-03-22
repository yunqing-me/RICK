#!/usr/bin/env bash

export PYTHONPATH=.


# Adaptation-Aware Kernel Modulation (AdAM)

# step 2. Main Adaptation

# FFHQ - Babies
CUDA_VISIBLE_DEVICES=0 python AdAM_main_adaptation.py \
 --exp adam_ffhq-babies --data_path babies --n_sample_train 10 \
 --iter 1500 --batch 4 --eval_in_training --eval_in_training_freq 100 --n_sample_test 5000 \
 --store_samples --samples_freq 100 --fisher_quantile 50 \
 --eval_in_training \
 --wandb --wandb_project_name null --wandb_run_name null_adam_babies --method adam 
 
# FFHQ - AFHQ-Cat
CUDA_VISIBLE_DEVICES=7 python AdAM_main_adaptation.py \
 --exp adam_ffhq-afhq_cat --data_path afhq_cat --n_sample_train 10 \
 --iter 2500 --batch 4 --eval_in_training --eval_in_training_freq 100 --n_sample_test 5000 \
 --store_samples --samples_freq 100 --fisher_quantile 75 \
 --eval_in_training \
 --wandb --wandb_project_name null --wandb_run_name null_adam_afhq_cat --method adam