export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export REAL_SCORE_MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-14B"
# One-Forcing needs REAL videos to build the adversarial critic's clean_latents, so the dataset
# must be a video meta (type=video + file_path), NOT a prompt-only json. The official demo
# dataset X-Fun-Videos-Demo ships captioned videos and is normalized by --enable_bucket
# --random_hw_adapt --training_with_video_token_length.
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# Warm-start generator: a distilled/ODE checkpoint that the DMD + adversarial critic then polish.
# Default: the official One-Forcing ODE-initialized checkpoint, downloaded via
#   hf download JiaqiFeng/OneForcing checkpoints/framewise/causal_ode.pt --local-dir models/Diffusion_Transformer/One-Forcing
# You can also point it at your own Causal-Forcing Stage-2 CCD checkpoint.
export STAGE2_CKPT="models/Diffusion_Transformer/One-Forcing/checkpoints/framewise/causal_ode.pt"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# One-Forcing = Causal-Forcing Stage-3 DMD + a shared noised-latent adversarial critic on fake_score.
# - 1-step frame-wise generation: --denoising_step_indices_list 1000, --num_frame_per_block=1;
#   21 latent frames from --video_sample_n_frames=81 @ 480x832.
# - GAN critic: gan_g_weight/gan_d_weight=0.03, feature layers [21, 29], ffn 2048, heads 12,
#   non-relativistic, R1/R2 off. The head rides on fake_score and is training-only.
# - GAN requires real videos, --gradient_accumulation_steps=1, no DeepSpeed, and --video_sample_stride=1
#   (stride=2 on 81-frame sources yields ~10 latent frames < score_num_frames=21 and aborts).
# - guidance: local CFG is cond + scale*(cond - uncond), so --real_guidance_scale=6.0.
# - optimizer/EMA/steps: lr=lr_critic=1e-5, betas (0.0, 0.999), weight_decay=0.01,
#   --use_ema --ema_decay=0.99, gen_update_interval=5, max_steps=200.
accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/wan2.1_one_forcing/train_one_forcing.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --real_score_pretrained_model_name_or_path=$REAL_SCORE_MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --ode_transformer_path=$STAGE2_CKPT \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --fix_sample_size 480 832 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --max_train_steps=200 \
  --checkpointing_steps=100 \
  --learning_rate=1.0e-05 \
  --learning_rate_critic=1.0e-05 \
  --lr_scheduler="constant" \
  --seed=42 \
  --output_dir="output_dir_wan2.1_one_forcing_framewise_1step" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --fsdp_reduce_fp32 \
  --adam_weight_decay=0.01 \
  --adam_beta1=0.0 \
  --adam_beta2=0.999 \
  --adam_epsilon=1e-08 \
  --vae_mini_batch=1 \
  --max_grad_norm=10.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --num_frame_per_block=1 \
  --use_kv_cache_training \
  --use_ema \
  --ema_decay=0.99 \
  --denoising_step_indices_list 1000 \
  --real_guidance_scale=6.0 \
  --score_timestep_clamp \
  --fake_guidance_scale=0.0 \
  --gen_update_interval=5 \
  --gan_g_weight=0.03 \
  --gan_d_weight=0.03 \
  --gan_feature_layers 21 29 \
  --gan_block_ffn_dim=2048 \
  --gan_block_num_heads=12 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "."
