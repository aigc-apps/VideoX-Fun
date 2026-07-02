export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export REAL_SCORE_MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-14B"
export DATASET_NAME="prompts/vidprom_filtered_extended.txt"
export STAGE2_CKPT="models/stage2.pt"
export OUTPUT_DIR="output_dir"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# Causal-Forcing Stage 3: Distribution Matching Distillation (DMD), frame-wise 2-step variant.
#
# Mirrors configs/causal_forcing_dmd_framewise_2step.yaml from upstream CF.
#   real_name: Wan2.1-T2V-14B            -> --real_score_pretrained_model_name_or_path=14B path
#   generator_ckpt: causal_cd.pt (stage2) -> --transformer_path=STAGE2_CKPT (CF .pt with generator_ema)
#   num_frame_per_block: 1               -> --num_frame_per_block=1 (frame-wise)
#   denoising_step_list: [1000, 500]     -> --denoising_step_indices_list 1000 500 (2-step)
#   timestep_shift: 5.0                  -> --shift=5.0 (matches CF flow scheduler shift)
#   lr / lr_critic                       -> --learning_rate=2e-6 / --learning_rate_critic=4e-7
#   beta1 / beta2 (gen + critic)         -> --adam_beta1=0.0 / --adam_beta2=0.999
#   dfake_gen_update_ratio: 5            -> --gen_update_interval=5 (existing arg, same semantics)
#   guidance_scale: 3.0                  -> --real_guidance_scale=3.0 (DMD internal CFG)
#   fake_guidance_scale: 0.0             -> --fake_guidance_scale=0.0 (CF default)
#   ema_weight / ema_start_step          -> --ema_weight=0.99 / --ema_start_step=200
#   batch_size 1, total_batch_size 8     -> 4 GPUs * bs=1 = total batch 4 (no grad-accum)
#   mixed_precision: true                -> --mixed_precision="bf16" (autocast; fp32 master)
#   gradient_checkpointing: true         -> --gradient_checkpointing
#
# 14B teacher + 1.3B generator + 1.3B critic on 4xGB200 (~189GB each). FSDP/ZeRO shards
# the 14B real_score (~56GB fp32 → ~14GB per GPU); generator + critic + Adam moments ~30GB
# per GPU. Comfortable headroom for the backward-simulation generator pass.
#
# Inference (2-step DMD, no external CFG): predict_t2v.py with
#   guidance_scale=1.0; num_inference_steps=2; stochastic_sampling=True.
accelerate launch \
  --num_processes=4 \
  --mixed_precision="bf16" \
  scripts/wan2.1_causal_forcing/train_causal_dmd.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --real_score_pretrained_model_name_or_path="$REAL_SCORE_MODEL_NAME" \
  --train_data_meta="$DATASET_NAME" \
  --transformer_path="$STAGE2_CKPT" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=2 \
  --num_train_epochs=999999 \
  --max_train_steps=10000 \
  --checkpointing_steps=500 \
  --learning_rate=2.0e-06 \
  --learning_rate_critic=4.0e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --adam_beta1=0.0 \
  --adam_beta2=0.999 \
  --adam_weight_decay=0.0 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=10.0 \
  --seed=0 \
  --output_dir="$OUTPUT_DIR" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --num_frame_per_block=1 \
  --use_kv_cache_training \
  --denoising_step_indices_list 1000 500 \
  --real_guidance_scale=3.0 \
  --fake_guidance_scale=0.0 \
  --gen_update_interval=5 \
  --ema_weight=0.99 \
  --ema_start_step=200 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "."
