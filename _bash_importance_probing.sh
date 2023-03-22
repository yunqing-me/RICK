#!/usr/bin/env bash

export PYTHONPATH=.


# Adaptation-Aware Kernel Modulation (AdAM)

# step 1. Importance Probing

# FFHQ - Babies
CUDA_VISIBLE_DEVICES=0 python ${method_0}.py \
 --exp _low_rank_probing_ffhq-babies --data_path babies --n_sample_train 10 \
 --fisher_iter 500 --fisher_freq 500 --num_batch_fisher 250 --source_key ffhq --ckpt_source style_gan_source_ffhq.pt

# FFHQ - AFHQ-Cat
CUDA_VISIBLE_DEVICES=7 python ${method_0}.py \
 --exp _low_rank_probing_ffhq-afhq_cat --data_path afhq_cat --n_sample_train 10 \
 --fisher_iter 500 --fisher_freq 500 --num_batch_fisher 250 --source_key ffhq --ckpt_source style_gan_source_ffhq.pt
