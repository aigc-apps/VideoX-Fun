export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# Stage-1 RoLa sparse warm-up product for the LOW expert (train_finetune_sparse_wan2.2_low.sh ->
# _low_sparse_rola/checkpoint-400).
export SPARSE_STUDENT_CKPT="output_dir_sparkdiffusion_wan2.2_14b_low_sparse_rola/checkpoint-400/diffusion_pytorch_model.safetensors"
# Optional warm-start of the backward-simulation starting latent from the trained HIGH distilled student
# (train_distill_wan2.2_high.sh -> _high_sparse_rola_distill/checkpoint-<step>/diffusion_pytorch_model.safetensors).
# Empty = cold start from pure noise; set it to the latest HIGH distill checkpoint for a proper relay.
# --init_rollout_path is fault-tolerant (auto-finds the safetensors and strips generator/transformer/model
# prefixes, so a distill checkpoint dir also works).
export INIT_ROLLOUT_CKPT="output_dir_sparkdiffusion_wan2.2_14b_high_sparse_rola_distill/checkpoint-4000/diffusion_pytorch_model.safetensors"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# SparkDiffusion Stage-2 (CrossDistill) for the LOW-noise expert: same sparse-student DMD as
# train_distill_wan2.2_high.sh, with two low-specific differences. (1) --boundary_type="low" starts the few-step
# trajectory at the MoE boundary and keeps only the low-segment knots, while --rf_t_max=1.0 lets the
# DMD/critic comparison time s_D span the FULL noise range -- mirrors official wan2.2_joint.sh's
# joint_low_d_time_full_range=true (the low expert still emits a complete x0, so its fake/teacher scores
# are matched at all levels, not only s < boundary). (2) The low expert is div-free (no --div_mode /
# --loss_scale_div): the CrossDistill
# recipe keeps the PCM diversity term on the high expert only. The low segment of the few-step trajectory
# starts at the MoE boundary rather than pure noise, so --init_rollout_path warm-starts it from the frozen
# trained HIGH student (INIT_ROLLOUT_CKPT); leaving it empty makes the untrained low student roll the
# high-noise segment itself and the samples degrade. Memory levers are identical to train_distill_wan2.2_high.sh
# (--fsdp_offload_params true + --gradient_checkpointing; lower --fix_sample_size, not frame count, on OOM).
accelerate launch --main_process_port=0 --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_POST" \
    --fsdp_cpu_ram_efficient_loading False scripts/sparkdiffusion/train_distill_wan2.2.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --fix_sample_size 480 832 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=200 \
  --learning_rate=2e-06 \
  --learning_rate_critic=4e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_sparkdiffusion_wan2.2_14b_low_sparse_rola_distill" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --trainable_modules "." \
  --train_mode="normal" \
  --distill_loss="dmd" \
  --max_simulation_steps_fake=2 \
  --boundary_type="low" \
  --rf_t_max=1.0 \
  --real_guidance_scale=5.0 \
  --sparse_student \
  --student_topk=0.03 \
  --generator_transformer_path=$SPARSE_STUDENT_CKPT \
  --init_rollout_path=$INIT_ROLLOUT_CKPT \
  --resume_from_checkpoint="latest"
