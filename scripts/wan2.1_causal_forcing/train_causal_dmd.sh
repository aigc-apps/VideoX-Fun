export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export REAL_SCORE_MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-14B"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export STAGE2_CKPT="output_dir_wan2.1_causal_forcing_ccd/checkpoint-5000/transformer/diffusion_pytorch_model.safetensors"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# Causal-Forcing Stage 3: Distribution Matching Distillation (DMD), frame-wise 2-step variant.
# - --real_score_pretrained_model_name_or_path -> Wan2.1-T2V-14B non-causal teacher (DMD real_score).
# - --transformer_path -> Stage 2 CCD ckpt (generator/critic init).
# - num_frame_per_block=3 -> frame-wise; --denoising_step_indices_list 1000 500 -> 2-step DMD.
# - train_mode="normal" (default) -> TextDataset (prompt-only); generation shape from video_sample_* / fix_sample_size.
# - Note: DMD parser has no --shift; the flow scheduler shift is fixed inside the training loop.
accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/wan2.1_causal_forcing/train_causal_dmd.py \
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
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=200 \
  --learning_rate=2.0e-06 \
  --learning_rate_critic=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_causal_forcing_dmd" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=0.0 \
  --adam_beta1=0.0 \
  --adam_beta2=0.999 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=10.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --num_frame_per_block=3 \
  --use_kv_cache_training \
  --denoising_step_indices_list 1000 667 334 1 \
  --real_guidance_scale=6.0 \
  --randomize_step_indices \
  --fake_guidance_scale=0.0 \
  --gen_update_interval=5 \
  --trainable_modules "."
