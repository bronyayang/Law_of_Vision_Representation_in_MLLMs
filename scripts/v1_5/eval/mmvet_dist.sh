accelerate launch --num_processes 8 \
        --config_file /mnt/bd/bohanzhaiv0/AML/Inference/LLaVA1.5/default_config.yaml \
        --main_process_port 23786 \
        llava/eval/model_mmvet_dist.py \
        --model_name_or_path ./ckpts/llava-v1.5-7b \
        --version vicuna_v1 \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --do_train False \
        --do_predict True \
        --per_device_eval_batch_size 1 \
        --predict_with_generate True \
        --dataloader_drop_last False \
        --image_aspect_ratio pad \
        --bf16 True \
        --output_dir ./outputs/ \