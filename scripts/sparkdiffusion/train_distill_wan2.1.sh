export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# Stage-1 RoLa sparse warm-up product (train_finetune_sparse_wan2.1.sh -> _1.3b_sparse_rola/checkpoint-400).
# Env-overridable. --student_topk below MUST match the Stage-1 value (0.03).
export SPARSE_STUDENT_CKPT="output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola/checkpoint-400/diffusion_pytorch_model.safetensors"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# SparkDiffusion Stage-2 (CrossDistill) for the Wan2.1 single-expert T2V model: distill the sparse student into a
# few-step, CFG-free generator against the dense teacher. --sparse_student turns the generator into
# SparkWanTransformer3DModel while teacher + critic stay dense; --generator_transformer_path loads the Stage-1
# sparse warm-up ckpt (no prefix stripping, unexpected keys must be 0). Wan2.1-T2V-1.3B is a single full-range
# expert, so the student distills over the whole rf-time range [0, 1] (no MoE boundary, no init-rollout relay).
# The CrossDistill recipe adds a PCM-style diversity term (--div_mode pcm_random --loss_scale_div 10.0), which needs
# --max_simulation_steps_fake >= 2 so the first backward-simulation segment stays no_grad; --teacher_timestep_shift
# matches the Wan2.1 scheduler shift (5.0). Memory: --fsdp_offload_params true + --gradient_checkpointing keep the
# three nets (student/teacher/critic) off the optimizer-state OOM path. No --max_train_steps: this phase runs until
# you stop it (kill the train_distill_wan2.1.py processes).
accelerate launch --main_process_port=0 --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
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
  --learning_rate_critic=4e-07 \
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
  --trainable_modules "." \
  --train_mode="normal" \
  --distill_loss="dmd" \
  --max_simulation_steps_fake=2 \
  --div_mode="pcm_random" \
  --loss_scale_div=10.0 \
  --teacher_timestep_shift=5.0 \
  --real_guidance_scale=5.0 \
  --sparse_student \
  --student_topk=0.03 \
  --generator_transformer_path=$SPARSE_STUDENT_CKPT \
  --resume_from_checkpoint="latest"
