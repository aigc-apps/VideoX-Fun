export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets_lmdb"
export STAGE1_CKPT="output_dir_wan2.1_causal_forcing_ar_diffusion/checkpoint-2000/diffusion_pytorch_model.safetensors"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# Causal-Forcing Stage 2 (Option B): Causal Consistency Distillation Initialization.
# - The dataset MUST be a Causal-Forcing-style Latent LMDB (use preprocess_lmdb.py).
# - --transformer_path / --teacher_transformer_path point to the Stage 1 AR-diffusion ckpt.
# - num_frame_per_block=3 -> chunkwise; set to 1 for the framewise variant.
# - discrete_cd_N=48 mirrors the official `causal_cd_chunkwise.yaml`.
accelerate launch --mixed_precision="bf16" scripts/wan2.1_causal_forcing/train_causal_consistency_distill.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_meta=$DATASET_NAME \
  --train_data_format=latent_lmdb \
  --transformer_path=$STAGE1_CKPT \
  --teacher_transformer_path=$STAGE1_CKPT \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_causal_forcing_ccd" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=0.0 \
  --adam_beta1=0.0 \
  --adam_beta2=0.999 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=10.0 \
  --num_frame_per_block=3 \
  --shift=5.0 \
  --discrete_cd_N=48 \
  --guidance_scale=3.0 \
  --ema_weight=0.99 \
  --ema_start_step=200 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --low_vram
