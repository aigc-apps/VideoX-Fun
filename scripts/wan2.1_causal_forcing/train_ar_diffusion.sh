export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_META_NAME="datasets/internal_datasets_lmdb"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# Causal-Forcing Stage 1: Autoregressive Diffusion Training (teacher forcing).
# - num_frame_per_block=3 -> chunkwise; set to 1 for the framewise variant.
# - shift=5.0 mirrors the Causal-Forcing default scheduler shift.
accelerate launch \
  --num_processes=1 \
  --mixed_precision="bf16" \
  scripts/wan2.1_causal_forcing/train_ar_diffusion.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_meta="$DATASET_META_NAME" \
  --train_data_format=latent_lmdb \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=2 \
  --num_train_epochs=999999 \
  --max_train_steps=20 \
  --checkpointing_steps=999999 \
  --learning_rate=2.0e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="$OUTPUT_DIR" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=10.0 \
  --num_frame_per_block=3 \
  --train_sampling_steps=1000 \
  --shift=5.0 \
  --use_timestep_weight \
  --trainable_modules "."
