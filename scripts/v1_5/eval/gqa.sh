#!/bin/bash
model_path="/mnt/bn/bohanzhainas1/shijiay/exp_ckpts/05-21-24_XL_llava_vicuna7b_ft"
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
CKPT="diffllavaXL_7bvi"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="/mnt/bn/bohanzhainas1/flamingo_data/data/evaluation/gqa/"
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 -m llava.eval.model_vqa_loader \
        --model-path $model_path \
        --question-file /mnt/bn/bohanzhainas1/yiqi.wang/data/llava_eval/gqa/$SPLIT.jsonl \
        --image-folder /mnt/bn/bohanzhainas1/flamingo_data/data/evaluation/gqa/images \
        --answers-file /mnt/bn/bohanzhainas1/shijiay/llava_eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait
output_file=/mnt/bn/bohanzhainas1/shijiay/llava_eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /mnt/bn/bohanzhainas1/shijiay/llava_eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
sudo python3 scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json
cd $GQADIR
sudo python3 eval_gqa.py --tier testdev_balanced