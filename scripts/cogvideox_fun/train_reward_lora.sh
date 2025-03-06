export MODEL_NAME="alibaba-pai/CogVideoX-Fun-V1.1-2b-InP"
export TRAIN_PROMPT_PATH="MovieGenVideoBench_train.txt"
# Performing validation simultaneously with training will increase time and GPU memory usage.
export VALIDATION_PROMPT_PATH="MovieGenVideoBench_val.txt"
# Use 49 for V1 and V1.1; Use 85 for V1.5.
export VIDEO_LENGTH=49

accelerate launch --num_processes=1 --mixed_precision="bf16" scripts/train_reward_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --rank=32 \
  --network_alpha=16 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=10000 \
  --checkpointing_steps=100 \
  --learning_rate=1e-05 \
  --seed=42 \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --vae_gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.3 \
  --prompt_path=$TRAIN_PROMPT_PATH \
  --train_sample_height=224 \
  --train_sample_width=224 \
  --video_length=49 \
  --num_decoded_latents=1 \
  --num_sampled_frames=1 \
  --reward_fn="HPSReward" \
  --reward_fn_kwargs='{"version": "v2.1"}' \
  --backprop

# Training command for CogVideoX-Fun-V1.1-2b-InP-HPS2.1.safetensors (with 8 A100 GPUs)
# accelerate launch --num_processes=8 --mixed_precision="bf16" --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json scripts/train_reward_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --rank=128 \
#   --network_alpha=64 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --max_train_steps=10000 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-05 \
#   --seed=42 \
#   --output_dir="output_dir" \
#   --gradient_checkpointing \
#   --mixed_precision="bf16" \
#   --adam_weight_decay=3e-2 \
#   --adam_epsilon=1e-10 \
#   --max_grad_norm=0.3 \
#   --prompt_path=$TRAIN_PROMPT_PATH \
#   --train_sample_height=256 \
#   --train_sample_width=256 \
#   --video_length=49 \
#   --validation_prompt_path=$VALIDATION_PROMPT_PATH \
#   --validation_steps=100 \
#   --validation_batch_size=8 \
#   --num_decoded_latents=1 \
#   --num_sampled_frames=1 \
#   --reward_fn="HPSReward" \
#   --reward_fn_kwargs='{"version": "v2.1"}' \
#   --backprop