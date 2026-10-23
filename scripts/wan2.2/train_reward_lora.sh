export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 与 NCCL_P2P_DISABLE=1 用于没有 RDMA 的多机环境。
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
  --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" \
  --fsdp_cpu_ram_efficient_loading False \
  scripts/wan2.2/train_reward_lora.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_reward_lora" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --enable_bucket \
  --boundary_type="high" \
  --rank=128 \
  --network_alpha=64 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --train_mode="normal" \
  --guidance_scale=5.0 \
  --num_inference_steps=40 \
  --num_decoded_latents=3 \
  --reward_fn="HPSReward" \
  --reward_fn_kwargs='{"version": "v2.1"}' \
  --backprop \
  --backprop_strategy="tail" \
  --backprop_num_steps=1 \
  --low_vram