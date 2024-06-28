master_addr=${master_addr:=$ARNOLD_WORKER_0_HOST}
master_port=${master_port:=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)}
nnodes="${nnodes:=$ARNOLD_WORKER_NUM}"
node_rank="${node_rank:=$ARNOLD_ID}"
nproc_per_node="${nproc_per_node:=$ARNOLD_WORKER_GPU}"
node_rank="${node_rank:=$ARNOLD_ID}"
trial_id="${trial_id:=$ARNOLD_TRIAL_ID}"

torchrun --node_rank=$node_rank --nproc_per_node=$nproc_per_node --nnodes=$nnodes --rdzv_endpoint="${master_addr}:${master_port}" llava/train/train_mem.py \
    --deepspeed /opt/tiger/LLaVA1.5/scripts/zero2.json \
    --model_name_or_path /mnt/bn/bohanzhainas1/Public_Models/vicuna-13b-v1.5 \
    --version plain \
    --data_path /mnt/bn/bohanzhainas1/Public_data/blip_laion_cc_sbu_558k/blip_laion_cc_sbu_558k.json\
    --image_folder /mnt/bn/shijiaynas/LLaVA_pretrain_sd1.5 \
    --vision_tower runwayml/stable-diffusion-v1-5_feature \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /mnt/bn/bohanzhainas1/shijiay/exp_ckpts/debug \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True 