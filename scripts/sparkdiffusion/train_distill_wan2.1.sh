export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# Stage-1 RoLa sparse warm-up product (train_finetune_sparse_wan2.1.sh -> _1.3b_sparse_rola/checkpoint-400).
# --student_topk below MUST match the Stage-1 value (0.10, i.e. sparsity 0.9).
export SPARSE_STUDENT_CKPT="output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola/checkpoint-2000/diffusion_pytorch_model.safetensors"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# SparkDiffusion Stage-2 (CrossDistill) for the Wan2.1 single-expert T2V model: distill the sparse student into a
# few-step, CFG-free generator against the dense teacher. --sparse_student turns the generator into
# SparkWanTransformer3DModel while teacher + critic stay dense; --generator_transformer_path loads the Stage-1
# sparse warm-up ckpt (no prefix stripping, unexpected keys must be 0). Wan2.1-T2V-1.3B is a single full-range
# expert, so the student distills over the whole rf-time range [0, 1] (no MoE boundary, no init-rollout relay).
# The host is a plain discrete-grid DMD (--denoising_step_indices_list 1000 750 500 250 + --flow_euler_rollout, mean
# reduction). CrossDistill layers the official PCM diversity term on top as a self-contained additive loss
# (--div_mode pcm_random --loss_scale_div 5.0): a phase-0 boundary-alignment MSE anchored on the paired real video
# latent, rolled on its own continuous rf grid (--dcm_total_steps 48 --dcm_skipping_interval_steps 1,
# --teacher_timestep_shift 5.0 = the Wan2.1 scheduler shift, --div_rf_boundary 0.933781 = phase-0 right edge,
# --div_teacher_guidance 5.0). --loss_scale_div is in per-element-mean space (the official uses 10.0 in sum space);
# set it to 0 to recover the plain-DMD baseline. Memory: --gradient_checkpointing keeps the three nets
# (student/teacher/critic) off the activation OOM path (1.3B single expert needs no CPU offload; add
# --fsdp_offload_params true only if it OOMs). No --max_train_steps: runs until you kill the procs.
accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_POST" \
    --fsdp_cpu_ram_efficient_loading False scripts/sparkdiffusion/train_distill_wan2.1.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
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
  --checkpointing_steps=200 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola_distill" \
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
  --denoising_step_indices_list 1000 750 500 250 \
  --flow_euler_rollout \
  --randomize_step_indices \
  --gen_update_interval 5 \
  --div_mode pcm_random \
  --loss_scale_div 5.0 \
  --dcm_total_steps 48 \
  --dcm_skipping_interval_steps 1 \
  --teacher_timestep_shift 5.0 \
  --div_rf_boundary 0.933781 \
  --div_teacher_guidance 5.0 \
  --trainable_modules "." \
  --train_mode="normal" \
  --sparse_student \
  --student_topk=0.10 \
  --generator_transformer_path=$SPARSE_STUDENT_CKPT \
  --resume_from_checkpoint="latest"
