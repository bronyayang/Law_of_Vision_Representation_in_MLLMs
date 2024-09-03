#!/bin/bash

# Define arrays for tasks, models, and their corresponding paths
tasks=("mmbench_en" "mme" "ok_vqa" "textvqa_val" "vizwiz_vqa_val" "scienceqa_img" "seed_all" "mmmu_val")
models=("clip" "clip224" "openclip" "dinov2" "imsd" "sd1.5" "sdxl" "dit" "sd3" "sd2.1" "siglip" "clipdino")
paths=()

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
        target_dir="/any/path/${task}/${model}"
        
        # Ensure the target directory exists
        mkdir -p "$target_dir"
        
        # Move the created folder to the appropriate directory
        mv /any/path/temp_tensors/* "$target_dir/"
        
        # Check if the move was successful
        if [ $? -ne 0 ]; then
            echo "Failed to move temp_tensors for task: $task with model: $model"
        else
            echo "Successfully moved temp_tensors for task: $task with model: $model to $target_dir"
        fi
    done
done
