## Lora Training Code

The default training commands for MOVA are as follows:

We can use FSDP in MOVA training, which can save a lot of video memory. 

The metadata.json is a little different from normal json in VideoX-Fun, you need to add a audio_path.

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
  - For example, when `random_hw_adapt` is enabled, with `video_sample_n_frames=49`, `video_sample_size=768`, the resolution of video inputs for training is `512x512x49`, `768x768x49`.
- `training_with_video_token_length` specifies training the model according to token length. For training videos, the height and width will be set to `video_sample_size` as the maximum and `256` as the minimum.
  - For example, when `training_with_video_token_length` is enabled, with `video_sample_n_frames=49`, `token_sample_size=512`, `video_sample_size=768`, the resolution of video inputs for training is `256x256x49`, `512x512x49`, `768x768x21`.
  - The token length for a video with dimensions 512x512 and 49 frames is 13,312. We need to set the `token_sample_size = 512`.
    - At 512x512 resolution, the number of video frames is 49 (~= 512 * 512 * 49 / 512 / 512).
    - At 768x768 resolution, the number of video frames is 21 (~= 512 * 512 * 49 / 768 / 768).
    - At 1024x1024 resolution, the number of video frames is 9 (~= 512 * 512 * 49 / 1024 / 1024).
    - These resolutions combined with their corresponding lengths allow the model to generate videos of different sizes.
- `resume_from_checkpoint` is used to set the training should be resumed from a previous checkpoint. Use a path or `"latest"` to automatically select the last available checkpoint.
- `target_name` represents the components/modules to which LoRA will be applied, separated by commas (e.g., "q,k,v,ffn.0,ffn.2").
- `use_peft_lora` indicates whether to use the PEFT module for adding LoRA. Using this module will be more memory-efficient.
- `rank` means the dimension of the LoRA update matrices (default: 128).
- `network_alpha` means the scaling factor for LoRA update matrices (default: 64).
- `boundary_type` specifies which DiT to train LoRA on: "low" = only low-noise DiT, "high" = only high-noise DiT, "full" = both DiTs.
- `train_components` specifies which components to train LoRA on. Comma-separated list of: "transformer", "transformer_2", "transformer_audio", "dual_tower_bridge", or "all". This affects which LoRA weights are saved during checkpointing.
- `i2v_ratio` is the ratio of I2V samples in training. 0.0 = pure T2V, 1.0 = pure I2V, 0.5 = 50% T2V + 50% I2V (default).
- `low_vram` enables low VRAM mode to reduce memory usage.

When train model with multi machines, please set the params as follows:
```sh
export MASTER_ADDR="your master address"
export MASTER_PORT=10086
export WORLD_SIZE=1 # The number of machines
export NUM_PROCESS=8 # The number of processes, such as WORLD_SIZE * 8
export RANK=0 # The rank of this machine

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK scripts/xxx/xxx.py
```

MOVA without deepspeed:

```sh
export MODEL_NAME="models/Diffusion_Transformer/MOVA-360p"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/mova/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=360 \
  --video_sample_size=360 \
  --token_sample_size=360 \
  --video_sample_stride=1 \
  --video_sample_n_frames=193 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_mova_lora" \
  --validation_steps=500 \
  --validation_epochs=500 \
  --validation_paths "asset/single_person.jpg" \
  --validation_prompts="A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks. At one point, he says, \"I would also say that this election in Germany wasn't surprising.\"" \
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
  --rank=64 \
  --network_alpha=64 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --boundary_type="high" \
  --use_peft_lora \
  --train_components="transformer,transformer_2"
```

MOVA with Deepspeed Zero-2:

```sh
export MODEL_NAME="models/Diffusion_Transformer/MOVA-360p"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/mova/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=360 \
  --video_sample_size=360 \
  --token_sample_size=360 \
  --video_sample_stride=1 \
  --video_sample_n_frames=193 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_mova_lora" \
  --validation_steps=500 \
  --validation_epochs=500 \
  --validation_paths "asset/single_person.jpg" \
  --validation_prompts="A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks. At one point, he says, \"I would also say that this election in Germany wasn't surprising.\"" \
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
  --rank=64 \
  --network_alpha=64 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --boundary_type="high" \
  --use_peft_lora \
  --train_components="transformer,transformer_2"
```

MOVA with FSDP:

```sh
export MODEL_NAME="models/Diffusion_Transformer/MOVA-360p"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock,AudioWanAttentionBlock,ConditionalCrossAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/mova/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=360 \
  --video_sample_size=360 \
  --token_sample_size=360 \
  --video_sample_stride=1 \
  --video_sample_n_frames=193 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_mova_lora" \
  --validation_steps=500 \
  --validation_epochs=500 \
  --validation_paths "asset/single_person.jpg" \
  --validation_prompts="A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks. At one point, he says, \"I would also say that this election in Germany wasn't surprising.\"" \
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
  --rank=64 \
  --network_alpha=64 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --boundary_type="high" \
  --use_peft_lora \
  --train_components="transformer,transformer_2"
```