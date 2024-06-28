#!/bin/bash

# Define arrays for tasks, models, and their corresponding paths
# tasks=("mmbench_en" "mme" "ok_vqa" "textvqa_val" "vizwiz_vqa_val" "scienceqa_img" "seed_all" "mmmu_val")
tasks=("seedbench")
# tasks=("mmmu_val")
# models=("clip" "clip224" "openclip" "dinov2" "imsd" "sd1.5" "sdxl" "dit" "sd3" "sd2.1" "siglip" "clipdino")
models=("clipdino336")
# models=("sdxl")
# paths=(
#     "/mnt/bn/shijiaynas/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/llava_clip_stage1"
#     "/mnt/bn/bohanzhainas1/bohan/exp/2024-01-25/LLaVA_1.5_7B_224/llava_clip224_stage1"
#     "/mnt/bn/bohanzhainas1/shijiay/exp_ckpts/05-11-24_llava_openclip/llava_openclip_stage1"
#     "/mnt/bn/bohanzhainas1/shijiay/exp_ckpts/05-20-24_llava_dinov2_7b/llava_dinov2_stage1"
#     "/mnt/bn/bohanzhainas1/shijiay/exp_ckpts/05-22-24_imdiffLLaVA_7bvi"
#     "/mnt/bn/bohanzhainas1/shijiay/exp_ckpts/04_18_7B_pretrain_diff_llava"
#     "/mnt/bn/bohanzhainas1/shijiay/exp_ckpts/06-19-24_llavasdxl_7b_vi"
#     "/mnt/bn/bohanzhainas1/shijiay/exp_ckpts/06-19-24_llava_dit_7b_vi_img_size_fix"
#     "/mnt/bn/bohanzhainas1/shijiay/exp_ckpts/06-19-24_llavasd3_7b_vi_2"
#     "/mnt/bn/bohanzhainas1/shijiay/exp_ckpts/05-28-24_sd2.1_7bv1_diffllava"
#     "/mnt/bn/bohanzhainas1/shijiay/06-25-24_llava_siglip_7b_vi_2"
#     "/mnt/bn/bohanzhainas1/shijiay/06-25-24_llava_clipdino_7b_vi"
# )
paths=(
    "/mnt/bn/bohanzhainas1/shijiay/06-25-24_llava_clipdino336_7b_vi"
)

# Ensure the paths array has the same length as models array
if [ ${#models[@]} -ne ${#paths[@]} ]; then
    echo "Error: The number of models and paths does not match."
    exit 1
fi

# Iterate over each task
for task in "${tasks[@]}"; do
    # Iterate over each model and corresponding path
    for i in "${!models[@]}"; do
        model=${models[$i]}
        path=${paths[$i]}
        
        # Print the current task and model
        echo "Running task: $task with model: $model"

        # Run the command
        accelerate launch --main_process_port 25000 --num_processes=1 -m lmms_eval --model llava --model_args pretrained="$path" --tasks "$task" --batch_size 1
        
        # Define the target directory
        target_dir="/mnt/bn/shijiaynas/${task}_tensors/${model}"
        
        # Ensure the target directory exists
        mkdir -p "$target_dir"
        
        # Move the created folder to the appropriate directory
        mv /mnt/bn/shijiaynas/temp_tensors/* "$target_dir/"
        
        # Check if the move was successful
        if [ $? -ne 0 ]; then
            echo "Failed to move temp_tensors for task: $task with model: $model"
        else
            echo "Successfully moved temp_tensors for task: $task with model: $model to $target_dir"
        fi
    done
done
