accelerate launch --num_processes 2 \
        --config_file /mnt/bd/bohanzhaiv1/AML/LLaVA1.5/default_config.yaml \
        --main_process_port 23786 \
        llava/eval/model_mmcore_dist.py \
        --model_name_or_path /mnt/bn/yukunfeng-nasdrive/bohan/weights/LLaVA_weights/ORI_CKPTs/llava-v1.5-7b \
        --version vicuna_v1 \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --data_path /mnt/bd/bohanzhaiv1/AML/core-mm-1102.json \
        --image_folder /mnt/bd/bohanzhaiv1/AML/images \
        --do_train False \
        --do_predict True \
        --per_device_eval_batch_size 1 \
        --predict_with_generate True \
        --dataloader_drop_last False \
        --image_aspect_ratio pad \
        --bf16 True \
        --output_file /mnt/bd/bohanzhaiv1/AML/LLaVA1.5/outputs/llava1.5_llama2_7b_chat_coremm_cot_list.json \
        --special_prompt " let's think step by step." \
        --output_dir ./outputs/ \