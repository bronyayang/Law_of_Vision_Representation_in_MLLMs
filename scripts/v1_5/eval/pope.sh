#!/bin/bash
python3 -m llava.eval.model_vqa_loader \
    --model-path /mnt/bn/bohanzhainas1/shijiay/exp_ckpts/05-28-24_sd2.1_7bv1_diffllava_ft \
    --question-file /mnt/bn/bohanzhainas1/yiqi.wang/data/llava_eval/pope/llava_pope_test.jsonl \
    --image-folder /mnt/bn/bohanzhainas1/flamingo_data/data/evaluation/mscoco_karpathy/val2014 \
    --answers-file /mnt/bn/bohanzhainas1/shijiay/llava_eval/pope/answers/llava_diff2.17bvi.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
    
python3 llava/eval/eval_pope.py \
    --annotation-dir /mnt/bn/bohanzhainas1/flamingo_data/data/evaluation/mscoco_karpathy/annotations/coco \
    --question-file /mnt/bn/bohanzhainas1/yiqi.wang/data/llava_eval/pope/llava_pope_test.jsonl \
    --result-file /mnt/bn/bohanzhainas1/shijiay/llava_eval/pope/answers/llava_diff2.17bvi.jsonl