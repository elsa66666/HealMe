#! /bin/bash


# model_name_or_path=/data/xiaomengxi/psychology/openai-quickstart-python/llama-2-7b-chat-hf
# lora_path=/data/xiaomengxi/psychology/BELLE-main/train/round1_output/BELLE/saved_models/llama2_conversation_round2/checkpoint-432
# output_path=/data/xiaomengxi/psychology/BELLE-main/train/round1_output/round2_432

CUDA_VISIBLE_DEVICES=0 python src/merge_llama_with_lora.py \
    --model_name_or_path /data/xiaomengxi/psychology/openai-quickstart-python/llama-2-7b-chat-hf \
    --output_path /data/xiaomengxi/psychology/BELLE-main/train/round1_output/round2_432 \
    --lora_path /data/xiaomengxi/psychology/BELLE-main/train/round1_output/BELLE/saved_models/llama2_conversation_round2/checkpoint-432 \
    --llama