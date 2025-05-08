export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP"
export TRAIN_PROMPT_PATH="MovieGenVideoBench_train.txt"
# Performing validation simultaneously with training will increase time and GPU memory usage.
export VALIDATION_PROMPT_PATH="MovieGenVideoBench_val.txt"
# Set 1 for Wan2.1-Fun-14B-InP
export BACKPROP_NUM_STEPS=5

accelerate launch --mixed_precision="bf16" scripts/wan2.1_fun/train_reward_lora.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=10000 \
  --checkpointing_steps=100 \
  --learning_rate=1e-05 \
  --seed=42 \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.3 \
  --low_vram \
  --prompt_path=$TRAIN_PROMPT_PATH \
  --train_sample_height=256 \
  --train_sample_width=256 \
  --num_inference_steps=50 \
  --video_length=81 \
  --validation_prompt_path=$VALIDATION_PROMPT_PATH \
  --validation_steps=10000 \
  --num_decoded_latents=1 \
  --reward_fn="HPSReward" \
  --reward_fn_kwargs='{"version": "v2.1"}' \
  --backprop_strategy="tail" \
  --backprop_num_steps=5 \
  --backprop