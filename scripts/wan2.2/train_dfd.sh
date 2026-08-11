export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.jsonl"

# DFD is a post-training stage and depends on a generator trained with DMD first.
# Do not initialize DFD directly from the base model. Point this variable to the
# DMD generator checkpoint; optionally provide its dedicated fake-score checkpoint.
export GENERATOR_TRANSFORMER_PATH="${GENERATOR_TRANSFORMER_PATH:?Set GENERATOR_TRANSFORMER_PATH to a DMD generator checkpoint}"
export FAKE_SCORE_TRANSFORMER_PATH="${FAKE_SCORE_TRANSFORMER_PATH:-$GENERATOR_TRANSFORMER_PATH}"

# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.2/train_distill.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --generator_transformer_path="$GENERATOR_TRANSFORMER_PATH" \
  --fake_score_transformer_path="$FAKE_SCORE_TRANSFORMER_PATH" \
  --train_data_dir="$DATASET_NAME" \
  --train_data_meta="$DATASET_META_NAME" \
  --fix_sample_size=704 1280 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=4 \
  --max_train_steps=200 \
  --checkpointing_steps=20 \
  --learning_rate=1e-5 \
  --learning_rate_critic=1e-5 \
  --seed=0 \
  --output_dir="output_dir_wan2.2_dfd" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --allow_tf32 \
  --adam_beta1=0.9 \
  --adam_beta2=0.999 \
  --adam_weight_decay=0.01 \
  --adam_epsilon=1e-8 \
  --vae_mini_batch=1 \
  --tokenizer_max_length=512 \
  --boundary_type="full" \
  --train_mode="normal" \
  --gen_update_interval=5 \
  --real_guidance_scale=5.0 \
  --denoising_step_indices_list 1000 750 500 250 \
  --dfd \
  --dfd_teacher_replace_prob=0.5 \
  --report_to="none"
