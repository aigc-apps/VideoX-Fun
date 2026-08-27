export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
export TAE_PATH="models/Diffusion_Transformer/taew2_2.pth"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/taehv/train_taehv.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --tae_path=$TAE_PATH \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=33 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --latent_loss_weight=1.0 \
  --seed=42 \
  --output_dir="output_dir_taehv_w2.2" \
  --mixed_precision="bf16" \
  --adam_weight_decay=1e-2 \
  --adam_epsilon=1e-08 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --enable_bucket \
  --trainable_modules "." \
  --resume_from_checkpoint=latest
