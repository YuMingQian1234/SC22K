#!/bin/bash

# Set up the data folder
IMAGE_FOLDER="XXX"
VIDEO_FOLDER="XXX"
DATA_YAML="scripts/video/train/exp.yaml"

############### Prepare Envs #################
python3 -m pip install flash-attn --no-build-isolation
alias python=python3
############### Show Envs ####################

nvidia-smi

################ LoRA Fine-tuning Configuration ################

# 模型版本设置 (LLaVA-Next-Video 通常基于 Qwen 或 Llama)
# 这里以 Qwen2-7B 为例，您可以根据实际模型路径修改
LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# 运行名称和检查点
MID_RUN_NAME="llava-next-video-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-lora"
PREV_STAGE_CHECKPOINT="lmms-lab/llava-next-video-7b-qwen2" # 目标模型：LLaVA-Next-Video

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# LoRA 核心命令
deepspeed --master_port 30001 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version qwen_1_5 \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./work_dirs/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 64 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2

exit 0;
