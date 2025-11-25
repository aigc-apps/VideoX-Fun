## Lora Training Code

We can choose whether to use deepspeed and fsdp in CogVideoX-Fun, which can save a lot of video memory. 

Some parameters in the sh file can be confusing, and they are explained in this document:

- `enable_bucket` is used to enable bucket training. When enabled, the model does not crop the images and videos at the center, but instead, it trains the entire images and videos after grouping them into buckets based on resolution.
- Sample size Configuration Guide
  - `video_sample_size` represents the resolution size of videos; when `random_hw_adapt` is True, it represents the minimum value between video and image resolutions.
  - `image_sample_size` represents the resolution size of images; when `random_hw_adapt` is True, it represents the maximum value between video and image resolutions.
  - `token_sample_size` represents the resolution corresponding to the maximum token length when `training_with_video_token_length` is True.
  - Due to potential confusion in configuration, **if you don't require arbitrary resolution for finetuning**, it is recommended to set `video_sample_size`, `image_sample_size`, and `token_sample_size` to the same fixed value, such as **(320, 480, 512, 640, 960)**.
    - **All set to 320** represents **240P**.
    - **All set to 480** represents **320P**.
    - **All set to 640** represents **480P**.
    - **All set to 960** represents **720P**.
- `random_frame_crop` is used for random cropping on video frames to simulate videos with different frame counts.
- `random_hw_adapt` is used to enable automatic height and width scaling for images and videos. When `random_hw_adapt` is enabled, the training images will have their height and width set to `image_sample_size` as the maximum and `min(video_sample_size, 512)` as the minimum. For training videos, the height and width will be set to `image_sample_size` as the maximum and `min(video_sample_size, 512)` as the minimum.
  - For example, when `random_hw_adapt` is enabled, with `video_sample_n_frames=49`, `video_sample_size=1024`, and `image_sample_size=1024`, the resolution of image inputs for training is `512x512` to `1024x1024`, and the resolution of video inputs for training is `512x512x49` to `1024x1024x49`.
  - For example, when `random_hw_adapt` is enabled, with `video_sample_n_frames=49`, `video_sample_size=256`, and `image_sample_size=1024`, the resolution of image inputs for training is `256x256` to `1024x1024`, and the resolution of video inputs for training is `256x256x49`.
- `training_with_video_token_length` specifies training the model according to token length. For training images and videos, the height and width will be set to `image_sample_size` as the maximum and `video_sample_size` as the minimum.
  - For example, when `training_with_video_token_length` is enabled, with `video_sample_n_frames=49`, `token_sample_size=1024`, `video_sample_size=256`, and `image_sample_size=1024`, the resolution of image inputs for training is `256x256` to `1024x1024`, and the resolution of video inputs for training is `256x256x49` to `1024x1024x49`.
  - For example, when `training_with_video_token_length` is enabled, with `video_sample_n_frames=49`, `token_sample_size=512`, `video_sample_size=256`, and `image_sample_size=1024`, the resolution of image inputs for training is `256x256` to `1024x1024`, and the resolution of video inputs for training is `256x256x49` to `1024x1024x9`.
  - The token length for a video with dimensions 512x512 and 49 frames is 13,312. We need to set the `token_sample_size = 512`.
    - At 512x512 resolution, the number of video frames is 49 (~= 512 * 512 * 49 / 512 / 512).
    - At 768x768 resolution, the number of video frames is 21 (~= 512 * 512 * 49 / 768 / 768).
    - At 1024x1024 resolution, the number of video frames is 9 (~= 512 * 512 * 49 / 1024 / 1024).
    - These resolutions combined with their corresponding lengths allow the model to generate videos of different sizes.
- `train_mode` is used to specify the training mode, which can be either normal or i2v. Since CogVideoX-Fun uses the inpaint model to achieve image-to-video generation, the default is set to inpaint mode. If you only wish to achieve text-to-video generation, you can remove this line, and it will default to the text-to-video mode.
- `resume_from_checkpoint` is used to set the training should be resumed from a previous checkpoint. Use a path or `"latest"` to automatically select the last available checkpoint.
- `target_name` represents the components/modules to which LoRA will be applied, separated by commas.
- `use_peft_lora` indicates whether to use the PEFT module for adding LoRA. Using this module will be more memory-efficient.
- `rank` means the dimension of the LoRA update matrices.
- `network_alpha` means the scale of the LoRA update matrices.

CogVideoX-Fun without deepspeed:

```sh
export MODEL_NAME="models/Diffusion_Transformer/CogVideoX-Fun-2b-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/cogvideox_fun/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
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
  --rank=64 \
  --network_alpha=32 \
  --target_name="to_q,to_k,to_v,ff.0,ff.2" \
  --use_peft_lora \
  --low_vram \
  --train_mode="inpaint" 
```

CogVideoX-Fun with Deepspeed Zero-2:
```sh
export MODEL_NAME="models/Diffusion_Transformer/CogVideoX-Fun-2b-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/cogvideox_fun/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
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
  --rank=64 \
  --network_alpha=32 \
  --target_name="to_q,to_k,to_v,ff.0,ff.2" \
  --use_peft_lora \
  --low_vram \
  --train_mode="inpaint" 
```

With FSDP:

```sh
export MODEL_NAME="models/Diffusion_Transformer/CogVideoX-Fun-2b-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap CogVideoXBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/cogvideox_fun/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
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
  --rank=64 \
  --network_alpha=32 \
  --target_name="to_q,to_k,to_v,ff.0,ff.2" \
  --use_peft_lora \
  --low_vram \
  --train_mode="inpaint" 
```