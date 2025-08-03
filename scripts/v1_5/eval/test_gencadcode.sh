#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# CKPT="CADCODER/CAD-Coder"
# SPLIT="cadquery_test_data_subset100"

# First argument is checkpoint, can be either the HF model name or a local path
MODEL=$1

# Second argument is the name of the split
SPLIT=$2

CKPT="${MODEL##*/}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL \
        --question-file ./inference/$SPLIT.jsonl \
        --image-folder ./inference/test100_images \
        --answers-file ./inference/inference_results/$CKPT/$SPLIT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --max_new_tokens 10\
        --conv-mode vicuna_v1 &
done

wait

output_file=./inference/inference_results/$CKPT/$SPLIT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./inference/inference_results/$CKPT/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

