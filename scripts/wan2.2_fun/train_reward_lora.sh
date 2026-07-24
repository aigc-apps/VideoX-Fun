export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
export TRAIN_PROMPT_PATH="MovieGenVideoBench_train.txt"

# Train HPSv2.1 reward LoRA for the low noise model of Wan2.2-Fun-A14B-InP
accelerate launch --mixed_precision="bf16" --num-processes=8 --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json scripts/wan2.2_fun/train_reward_lora.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
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
  --boundary_type="low" \
  --lora_skip_name="ffn" \
  --low_vram \
  --use_deepspeed \
  --prompt_path=$TRAIN_PROMPT_PATH \
  --train_sample_height=256 \
  --train_sample_width=256 \
  --num_inference_steps=40 \
  --video_length=81 \
  --num_decoded_latents=1 \
  --reward_fn="HPSReward" \
  --reward_fn_kwargs='{"version": "v2.1"}' \
  --backprop_strategy="tail" \
  --backprop_num_steps=1 \
  --backprop

# Train MPS reward LoRA for the high noise model of Wan2.2-Fun-A14B-InP
# accelerate launch --mixed_precision="bf16" --num-processes=8 --use_deepspeed --deepspeed_config_file config/zero_stage3_config_cpu_offload.json scripts/wan2.2_fun/train_reward_lora.py \
#   --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
#   --pretrained_model_name_or_path=$MODEL_NAME \
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
#   --boundary_type="high" \
#   --lora_skip_name="ffn" \
#   --low_vram \
#   --use_deepspeed \
#   --prompt_path=$TRAIN_PROMPT_PATH \
#   --train_sample_height=256 \
#   --train_sample_width=256 \
#   --num_inference_steps=40 \
#   --video_length=81 \
#   --num_decoded_latents=1 \
#   --reward_fn="MPSReward" \
#   --backprop_strategy="tail" \
#   --backprop_num_steps=1 \
#   --backprop