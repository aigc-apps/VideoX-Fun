export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# Stage-1 RoLa sparse warm-up product for the LOW expert (train_finetune_sparse_wan2.2_low.sh ->
# _low_sparse_rola/checkpoint-400).
export SPARSE_STUDENT_CKPT="output_dir_sparkdiffusion_wan2.2_14b_low_sparse_rola/checkpoint-2000/diffusion_pytorch_model.safetensors"
# Warm-start of the low expert's rollout from the trained HIGH distilled student
# (train_distill_wan2.2_high.sh -> _high_sparse_rola_distill/checkpoint-<step>/diffusion_pytorch_model.safetensors).
# Empty = cold start from pure noise; set it to the latest HIGH distill checkpoint for a proper relay.
# --init_rollout_path is fault-tolerant (auto-finds the safetensors and strips generator/transformer/model
# prefixes, so a distill checkpoint dir also works).
export INIT_ROLLOUT_CKPT="output_dir_sparkdiffusion_wan2.2_14b_high_sparse_rola_distill/checkpoint-4000/diffusion_pytorch_model.safetensors"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# SparkDiffusion Stage-2 (CrossDistill) for the LOW-noise expert of Wan2.2-T2V-A14B: the same sparse-student
# discrete-grid DMD as train_distill_wan2.2_high.sh, with two low-specific differences. (1) --boundary_type="low"
# keeps only the --joint_backward_timesteps knots BELOW the config boundary (0.875), so the trainable low student
# rolls [0.875 -> 0.608979] with rf-native Euler steps (--flow_euler_rollout). Because that segment starts at the
# MoE boundary rather than pure noise, --init_rollout_path warm-starts it: a FROZEN copy of the trained HIGH
# student (INIT_ROLLOUT_CKPT) first rolls the high segment [1.0 -> 0.933781 -> 0.875] to produce the boundary
# latent, then the low student takes over. Leaving INIT_ROLLOUT_CKPT empty = cold start (the untrained low student
# rolls the high segment itself and the samples degrade). (2) The low expert is div-free (no --div_mode /
# --loss_scale_div): the PCM diversity term is gated on --loss_scale_div > 0 and lives on the HIGH expert only.
# Memory levers are identical to train_distill_wan2.2_high.sh (--fsdp_offload_params true + --gradient_checkpointing;
# lower --fix_sample_size, not frame count, on OOM). No --max_train_steps: runs until you kill the procs.
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
  --uniform_sampling \
  --flow_euler_rollout \
  --gen_update_interval 5 \
  --boundary_type="low" \
  --real_guidance_scale=5.0 \
  --init_rollout_path=$INIT_ROLLOUT_CKPT \
  --trainable_modules "." \
  --train_mode="normal" \
  --sparse_student \
  --student_topk=0.05 \
  --generator_transformer_path=$SPARSE_STUDENT_CKPT \
  --resume_from_checkpoint="latest"
