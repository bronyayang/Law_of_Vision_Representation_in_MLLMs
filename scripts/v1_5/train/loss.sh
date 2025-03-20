echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" >> ~/.deepspeed_env
export NCCL_NVLS_ENABLE=0


deepspeed --master_port 25001 --num_nodes=1 --num_gpus=1 llava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path /data/bohan/llava_data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /data/bohan/llava_data/LLaVA-Pretrain \
    --vision_tower google/siglip2-large-patch16-256 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --freeze_mm_mlp_adapter False \
    --pretrain_mm_mlp_adapter /data/bohan/llava_ckpts_final/siglip2_256_vicuna7b_llava/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /data/bohan/llava_ckpts_final/test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 0 \
    --weight_decay 0 \
    --warmup_ratio 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True