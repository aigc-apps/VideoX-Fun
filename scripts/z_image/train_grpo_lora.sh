export MODEL_NAME="models/Diffusion_Transformer/Z-Image"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/z_image/train_grpo_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1328 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=10 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_z_image_grpo_lora" \
  --validation_steps=10 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1 \
  --enable_bucket \
  --uniform_sampling \
  --rank=128 \
  --network_alpha=64 \
  --noise_level 1.2 \
  --grpo_num_steps=20 \
  --grpo_cfg_scale=6 \
  --sde_window_size 5 \
  --sde_window_range 0 10 \
  --num_image_per_prompt=16 \
  --num_batches_per_epoch=16 \
  --reward_fn="HPSv3Reward" \
  --reward_fn_kwargs='{"HPSv3Reward": {"checkpoint_path": "models/Diffusion_Transformer/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Diffusion_Transformer/Qwen2-VL-7B-Instruct"}}' \
  --multi_reward_weights='{"HPSv3Reward": 1}' \
  --target_name="to_q,to_k,to_v,feed_forward.w1,feed_forward.w2,feed_forward.w3"