#!/bin/bash
ROOT_DIR="/online1/zhaopl/zhaopl/yumingqian/LLaVA-Next"
cd $ROOT_DIR
export PYTHONPATH=$PYTHONPATH:"/online1/zhaopl/zhaopl/yumingqian/LLaVA-Next"
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT="/online1/zhaopl/zhaopl/yumingqian/LLaVA-Next/models/Video-LLaVA-7B-hf"
CONV_MODE="video_llava"
FRAMES=30
POOL_STRIDE=1024
POOL_MODE="average"
NEWLINE_POSITION="no_token"
OVERWRITE=True
VIDEO_PATH="/online1/zhaopl/zhaopl/yumingqian/Sports_Commentary_Datasets/test/100m/Mens_100m_Final_Paris/Second_Level/Videos/"


if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_frames_${FRAMES}_stride_${POOL_STRIDE}
fi
    
python3 playground/demo/video_demo.py \
    --model-path $CKPT \
    --video_path ${VIDEO_PATH} \
    --output_dir "inference_results/Video-LLaVA-7B/100m/Mens_100m_Final_Paris" \
    --output_name pred \
    --overwrite ${OVERWRITE} \
    --mm_spatial_pool_stride ${POOL_STRIDE:-4} \
    --for_get_frames_num $FRAMES \
    --conv-mode "video_llava" \
    --mm_spatial_pool_mode ${POOL_MODE:-average} \
    --mm_newline_position ${NEWLINE_POSITION:-grid} \
    --prompt "<video>\nYou are now a professional sports commentator, providing live commentary for the women's 100m race at the Paris Olympics. Please commentate on the given video."
