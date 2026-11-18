export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# SparkDiffusion Stage-1 (sparse warm-up) for the HIGH-noise expert: insert the compensating sparse
# self-attention (RoLa: blocks.*.self_attn.attn_op.{proj_q,proj_k,gate_proj,gate_bias}) into the dense
# Wan2.2 backbone and briefly warm it up with the plain step-local flow-matching objective, so Stage-2
# distillation starts from a usable coarse prior instead of a zero-initialized sparse branch.
# --boundary_type="high" picks the high-noise expert from the config boundary (config/wan2.2/wan_civitai_t2v.yaml,
# boundary:0.875); the low expert is warmed up separately by train_finetune_sparse_wan2.2_low.sh (Sec 4.3: each MoE
# expert group is warmed up and distilled on its own noise range). --student_topk=0.05 (~95% sparsity) must
# match the value used later in train_distill_wan2.2_high.sh. Stage A trains only "attn_op"; for Stage B (full sparse
# finetune) set --trainable_modules "." plus --transformer_path=<stage-A ckpt> and a new --output_dir.
accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" \
    --fsdp_cpu_ram_efficient_loading False scripts/sparkdiffusion/train_finetune_sparse_wan2.2.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --max_train_steps=2000 \
  --checkpointing_steps=200 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_sparkdiffusion_wan2.2_14b_high_sparse_rola" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=1e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "attn_op" \
  --student_topk=0.05 \
  --train_mode="normal" \
  --boundary_type="high" \
  --resume_from_checkpoint="latest"
