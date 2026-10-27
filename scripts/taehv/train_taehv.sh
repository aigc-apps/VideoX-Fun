export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export TAE_PATH="models/Diffusion_Transformer/taew2_2.safetensors"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/taehv/train_taehv.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --tae_path="$TAE_PATH" \
  --train_data_dir="$DATASET_NAME" \
  --train_data_meta="$DATASET_META_NAME" \
  --fix_sample_size 720 1280 \
  --video_sample_strides 1 2 3 \
  --video_sample_n_frames=25 \
  --validation_n_frames=49 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --max_train_steps=10000 \
  --checkpointing_steps=1000 \
  --validation_steps=1000 \
  --validation_paths "asset/inpaint_video.mp4" \
  --learning_rate=1e-04 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=200 \
  --latent_loss_weight=1.0 \
  --use_ema \
  --ema_decay=0.999 \
  --seed=42 \
  --output_dir="output_dir_taehv_w2.2" \
  --mixed_precision="bf16" \
  --adam_beta1=0.9 \
  --adam_beta2=0.99 \
  --adam_weight_decay=1e-2 \
  --adam_epsilon=1e-08 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --enable_bucket \
  --trainable_modules "." \
  --resume_from_checkpoint=latest
