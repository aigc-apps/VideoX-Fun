#!/usr/bin/env bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="${DATASET_NAME:-datasets/X-Fun-Videos-Demo}"
export DATASET_META_NAME="${DATASET_META_NAME:-$DATASET_NAME/metadata_add_width_height.json}"
export TAE_PATH="models/Diffusion_Transformer/taew2_2_super.pth"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/taehv/train_taehv_super.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --tae_path="$TAE_PATH" \
  --train_data_dir="$DATASET_NAME" \
  --train_data_meta="$DATASET_META_NAME" \
  --fix_sample_size 720 1280 \
  --video_sample_strides 1 2 3 \
  --video_sample_n_frames=25 \
  --validation_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --max_train_steps=10000 \
  --checkpointing_steps=1000 \
  --validation_steps=1000 \
  --validation_max_samples=0 \
  --learning_rate=5e-06 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=200 \
  --lr_min_ratio=0.2 \
  --l1_loss_weight=0.1 \
  --lpips_loss_weight=0.1 \
  --dino_loss_weight=0.02 \
  --grid_loss_weight=1.0 \
  --grid_period=16 \
  --feature_batch_size=13 \
  --use_ema \
  --ema_decay=0.999 \
  --selection_lpips_ratio=1.02 \
  --selection_grid_ratio=1.05 \
  --seed=42 \
  --output_dir="output_dir_taehv_w2.2_super" \
  --mixed_precision="bf16" \
  --adam_beta1=0.9 \
  --adam_beta2=0.99 \
  --adam_weight_decay=1e-2 \
  --adam_epsilon=1e-08 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0
