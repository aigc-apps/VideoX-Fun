## Training Code

The default training commands for the different versions are as follows:

We can choose whether to use fsdp in FantasyTalking, which can save a lot of video memory. 

The metadata_control.json is a little different from normal json in FantasyTalking, you need to add a audio_path.

```json
[
    {
      "file_path": "train/00000001.mp4",
      "audio_path": "wav/00000001.wav",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    .....
]
```

Some parameters in the sh file can be confusing, and they are explained in this document:

- `enable_bucket` is used to enable bucket training. When enabled, the model does not crop the videos at the center, but instead, it trains the videos after grouping them into buckets based on resolution.
- `random_frame_crop` is used for random cropping on video frames to simulate videos with different frame counts.
- `random_hw_adapt` is used to enable automatic height and width scaling for videos. When `random_hw_adapt` is enabled, for training videos, the height and width will be set to `video_sample_size` as the maximum and `512` as the minimum.
  - For example, when `random_hw_adapt` is enabled, with `video_sample_n_frames=49`, `video_sample_size=512`, the resolution of video inputs for training is `512x512x49`.
- `training_with_video_token_length` specifies training the model according to token length. For training videos, the height and width will be set to `video_sample_size` as the maximum and `256` as the minimum.
  - For example, when `training_with_video_token_length` is enabled, with `video_sample_n_frames=49`, `token_sample_size=512`, `video_sample_size=256`, the resolution of image inputs for training is `256x256` to `1024x1024`, and the resolution of video inputs for training is `256x256x49` to `512x512x21`.
  - The token length for a video with dimensions 512x512 and 49 frames is 13,312. We need to set the `token_sample_size = 512`.
    - At 512x512 resolution, the number of video frames is 49 (~= 512 * 512 * 49 / 512 / 512).
    - At 768x768 resolution, the number of video frames is 21 (~= 512 * 512 * 49 / 768 / 768).
    - At 1024x1024 resolution, the number of video frames is 9 (~= 512 * 512 * 49 / 1024 / 1024).
    - These resolutions combined with their corresponding lengths allow the model to generate videos of different sizes.
- `resume_from_checkpoint` is used to set the training should be resumed from a previous checkpoint. Use a path or `"latest"` to automatically select the last available checkpoint.

FantasyTalking without deepspeed:

```sh
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata_control.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/fantasytalking/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir" \
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
  --low_vram \
  --transformer_path="models/FantasyTalking/fantasytalking_model.ckpt" \
  --trainable_modules "processor." "proj_model."
```

FantasyTalking with deepspeed zero-2:

```sh
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata_control.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/fantasytalking/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir" \
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
  --low_vram \
  --transformer_path="models/FantasyTalking/fantasytalking_model.ckpt" \
  --trainable_modules "processor." "proj_model."
```

FantasyTalking with deepspeed zero-3:

```sh
python scripts/zero_to_bf16.py output_dir/checkpoint-{our-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

Training shell command is as follows:
```sh
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata_control.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/fantasytalking/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir" \
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
  --low_vram \
  --transformer_path="models/FantasyTalking/fantasytalking_model.ckpt" \
  --trainable_modules "processor." "proj_model."
```

FantasyTalking with FSDP:

Wan with FSDP is suitable for 14B Wan at high resolutions. Training shell command is as follows:
```sh
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata_control.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=AudioAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False  scripts/fantasytalking/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir" \
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
  --low_vram \
  --transformer_path="models/FantasyTalking/fantasytalking_model.ckpt" \
  --trainable_modules "processor." "proj_model."
```