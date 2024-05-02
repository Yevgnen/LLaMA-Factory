#!/bin/bash

CUDA_VISIBLE_DEVICES=0 rye run python ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --adapter_name_or_path ../../saves/phi3/lora/sft \
    --dataset dispatcher_test \
    --dataset_dir ../../data \
    --template phi \
    --finetuning_type lora \
    --output_dir ../../saves/phi3/lora/predict_test \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate
