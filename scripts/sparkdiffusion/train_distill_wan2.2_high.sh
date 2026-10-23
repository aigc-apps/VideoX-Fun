export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export SPARSE_STUDENT_CKPT="output_dir_sparkdiffusion_wan2.2_14b_high_sparse_rola/checkpoint-2000/diffusion_pytorch_model.safetensors"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# SparkDiffusion Stage-2 (CrossDistill) for the HIGH-noise expert of Wan2.2-T2V-A14B (an MoE T2V model):
# distill the sparse student into a few-step, CFG-free generator against the dense teacher. --sparse_student
# turns the generator (and the init-rollout model) into SparkWan2_2Transformer3DModel while teacher + critic
# stay dense; --generator_transformer_path loads the Stage-1 sparse warm-up ckpt (no prefix stripping,
# unexpected keys must be 0). --boundary_type="high" restricts the student to the high-noise rf segment: the
# few-step grid is derived IN-SCRIPT by splitting --joint_backward_timesteps at the config boundary (0.875) and
# keeping the knots >= boundary, so the high expert rolls [1.0 -> 0.933781 -> 0.875] with rf-native Euler steps
# (--flow_euler_rollout) -- no contiguous backward simulation. The host is a plain discrete-grid DMD
# (--uniform_sampling over that grid, mean reduction, generator/critic alternating via --gen_update_interval 5).
# CrossDistill layers the official PCM diversity term on the HIGH expert only (--div_mode pcm_random
# --loss_scale_div 10.0): a phase-0 boundary-alignment MSE anchored on the paired real-video latent, rolled on
# its own continuous rf grid (--dcm_total_steps 48 --dcm_skipping_interval_steps 1, --teacher_timestep_shift 12.0
# = the Wan2.2 scheduler shift, --div_rf_boundary 0.933781 = the high-segment first node, --div_teacher_guidance
# 5.0). Set --loss_scale_div 0 to recover the plain-DMD baseline. Memory: --fsdp_offload_params true +
# --gradient_checkpointing are mandatory (three 14B nets are optimizer-state dominated); resolution
# (--fix_sample_size), not frame count, is the OOM lever. No --max_train_steps: runs until you kill the procs.
accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_POST" \
    --fsdp_offload_params true \
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
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_sparkdiffusion_wan2.2_14b_high_sparse_rola_distill" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --flow_euler_rollout \
  --gen_update_interval 5 \
  --joint_backward_timesteps 0.933781 0.875 0.608979 \
  --div_mode pcm_random \
  --loss_scale_div 10.0 \
  --dcm_total_steps 48 \
  --dcm_skipping_interval_steps 1 \
  --teacher_timestep_shift 12.0 \
  --div_rf_boundary 0.933781 \
  --div_teacher_guidance 5.0 \
  --boundary_type="high" \
  --real_guidance_scale=5.0 \
  --trainable_modules "." \
  --train_mode="normal" \
  --sparse_student \
  --student_topk=0.05 \
  --generator_transformer_path=$SPARSE_STUDENT_CKPT \
  --resume_from_checkpoint="latest"
