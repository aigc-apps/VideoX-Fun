export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# CFG distillation of the control branch (mirrors scripts/flux2_fun/train_control_distill.py): both the student
# and the frozen teacher load the same trained control branch, so point --transformer_path at it, either the whole
# transformer safetensors or the control-only file of scripts/minimax_h3_fun/extract_control_weights.py, e.g.
# --transformer_path="output_dir_minimax_h3_control_inpaint/checkpoint-xxx/transformer/diffusion_pytorch_model_control.safetensors"
accelerate launch --mixed_precision="bf16" \
    --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap BaseMiniMaxH3TransformerBlock,MiniMaxH3ControlTransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
      scripts/minimax_h3_fun/train_control_distill.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=311 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_control_distill" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --enable_inpaint \
  --transformer_path="output_dir_minimax_h3_control_inpaint/checkpoint-6000/transformer/diffusion_pytorch_model_control.safetensors" \
  --uniform_sampling \
  --trainable_modules "control" \
  --real_guidance_scale=3.5 \
  --resume_from_checkpoint=latest