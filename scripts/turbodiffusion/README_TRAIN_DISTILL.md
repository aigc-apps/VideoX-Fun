# TurboDiffusion Distill Training Guide

This document provides a complete workflow for distilling Wan2.1 into TurboWan2.1, including environment configuration, data preparation, distributed training, and inference testing.

> **Note**: TurboDiffusion is a knowledge distillation approach that reduces inference steps (e.g., from 25-50 steps to 4-8 steps) while maintaining video generation quality.

---

## Table of Contents
- [1. Environment Configuration](#1-environment-configuration)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. Distill Training](#3-distill-training)
  - [3.1 Download Pretrained Model](#31-download-pretrained-model)
  - [3.2 Quick Start (Without DeepSpeed/FSDP)](#32-quick-start-without-deepspeedfsdp)
  - [3.3 Common Training Parameters](#33-common-training-parameters)
  - [3.4 Training with DeepSpeed Zero-2](#34-training-with-deepspeed-zero-2)
  - [3.5 Training with FSDP](#35-training-with-fsdp)
  - [3.6 Training with DeepSpeed Zero-3](#36-training-with-deepspeed-zero-3)
  - [3.7 Multi-Machine Distributed Training](#37-multi-machine-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameters](#41-inference-parameters)
  - [4.2 Text-to-Video (T2V) Inference](#42-text-to-video-t2v-inference)
  - [4.3 Image-to-Video (I2V) Inference](#43-image-to-video-i2v-inference)
  - [4.4 Multi-GPU Parallel Inference](#44-multi-gpu-parallel-inference)
- [5. Additional Resources](#5-additional-resources)

---

## 1. Environment Configuration

**Method 1: Using requirements.txt**

```bash
pip install -r requirements.txt
```

**Method 2: Manual Dependency Installation**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl deepspeed==0.17.0 numpy==1.26.4
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

**Method 3: Using Docker**

When using Docker, please ensure that the GPU driver and CUDA environment are correctly installed on your machine, then execute the following commands:

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset containing several training samples.

```bash
# Download official demo dataset
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 2.2 Dataset Structure

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json Format

**Relative Path Format** (example):
```json
[
  {
    "file_path": "train/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "type": "video",
    "width": 1024,
    "height": 1024
  },
  {
    "file_path": "train/video002.mp4",
    "text": "A person walking through a forest, cinematic view",
    "type": "video",
    "width": 1328,
    "height": 1328
  }
]
```

**Absolute Path Format**:
```json
[
  {
    "file_path": "/mnt/data/videos/sunset.mp4",
    "text": "A beautiful sunset over the ocean",
    "type": "video",
    "width": 1024,
    "height": 1024
  }
]
```

**Key Field Descriptions**:
- `file_path`: Video path (relative or absolute path)
- `text`: Video description (English prompt)
- `type`: Data type, fixed as `"video"`
- `width` / `height`: Video dimensions (**recommended** to provide for bucket training. If not provided, they will be automatically read during training, which may slow down training when data is stored on slow systems like OSS).
  - You can use `scripts/process_json_add_width_and_height.py` to extract width and height fields for JSON files without them, supporting both images and videos.
  - Usage: `python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Demo/metadata.json --output_file datasets/X-Fun-Videos-Demo/metadata_add_width_height.json`.

### 2.4 Relative vs Absolute Path Usage

**Relative Path**:

If your data uses relative paths, configure in the training script:

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

**Absolute Path**:

If your data uses absolute paths, configure in the training script:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **Recommendation**: If the dataset is small and stored locally, use relative paths. If the dataset is stored on external storage (e.g., NAS, OSS) or shared across multiple machines, use absolute paths.

---

## 3. Distill Training

### 3.1 Download Pretrained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download Wan2.1 official weights
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B
```

### 3.2 Quick Start (DeepSpeed-Zero-2)

After downloading the data as per **2.1 Quick Test Dataset** and the weights as per **3.1 Download Pretrained Model**, you can directly copy and run the quick start command.

DeepSpeed-Zero-2 and FSDP are recommended for training. Here we use DeepSpeed-Zero-2 as an example.

The difference between DeepSpeed-Zero-2 and FSDP is whether the model weights are sharded. **If you experience insufficient GPU memory when using multiple GPUs with DeepSpeed-Zero-2**, you can switch to FSDP for training.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/turbodiffusion/train_distill.py \
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
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_distill" \
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
  --train_mode="normal" \
  --trainable_modules "." \
  --low_vram
```

### 3.3 Common Training Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--config_path` | Config file path | `config/wan2.1/wan_civitai.yaml` |
| `--pretrained_model_name_or_path` | Pretrained model path | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B/` |
| `--train_data_dir` | Training data directory | `datasets/internal_datasets/` |
| `--train_data_meta` | Training data metadata file | `datasets/internal_datasets/metadata.json` |
| `--train_batch_size` | Batch size per GPU | 1 |
| `--image_sample_size` | Maximum image training resolution | 640 |
| `--video_sample_size` | Maximum video training resolution | 640 |
| `--token_sample_size` | Token sample size | 640 |
| `--video_sample_stride` | Video sampling stride | 2 |
| `--video_sample_n_frames` | Video sampling frames | 81 |
| `--gradient_accumulation_steps` | Gradient accumulation steps (effectively increases batch) | 1 |
| `--dataloader_num_workers` | DataLoader worker processes | 8 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 50 |
| `--learning_rate` | Initial learning rate (generator) | 2e-06 |
| `--learning_rate_critic` | Initial learning rate (critic) | 2e-07 |
| `--lr_scheduler` | Learning rate scheduler | `constant_with_warmup` |
| `--lr_warmup_steps` | Learning rate warmup steps | 100 |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir_distill` |
| `--gradient_checkpointing` | Enable gradient checkpointing | - |
| `--mixed_precision` | Mixed precision: `fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon | 1e-10 |
| `--vae_mini_batch` | VAE encoding mini-batch size | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--enable_bucket` | Enable bucket training, no cropping, group by resolution | - |
| `--random_hw_adapt` | Auto-scale images/videos to random size in `[min_size, max_size]` range | - |
| `--training_with_video_token_length` | Train based on token length, supports arbitrary resolutions | - |
| `--uniform_sampling` | Uniform timestep sampling | - |
| `--low_vram` | Low VRAM mode | - |
| `--train_mode` | Training mode: `normal` (standard) or `i2v` (image-to-video) | `normal` |
| `--resume_from_checkpoint` | Resume training path, use `"latest"` to auto-select latest checkpoint | None |
| `--validation_steps` | Run validation every N steps | 2000 |
| `--validation_epochs` | Run validation every N epochs | 5 |
| `--validation_prompts` | Prompts for video generation validation | `"A dog shaking head..."` |
| `--trainable_modules` | Trainable modules (`"."` means all modules) | `"."` |

**Distill-Specific Parameters**:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--denoising_step_indices_list` | Denoising step list (core distill parameter) | `1000 750 500 250` |
| `--real_guidance_scale` | Real guidance scale for scoring | 6.0 |
| `--fake_guidance_scale` | Fake guidance scale for scoring | 0.0 |
| `--gen_update_interval` | Generator update interval | 5 |
| `--negative_prompt` | Negative prompt for distillation | Chinese negative prompt |
| `--validation_paths` | Validation image paths for I2V mode | Image path list |
| `--train_sampling_steps` | Training sampling steps | 1000 |

**Sample Size Configuration Guide**:
- `video_sample_size` represents the resolution size of videos; when `random_hw_adapt` is True, it represents the minimum value between video and image resolutions.
- `image_sample_size` represents the resolution size of images; when `random_hw_adapt` is True, it represents the maximum value between video and image resolutions.
- `token_sample_size` represents the resolution corresponding to the maximum token length when `training_with_video_token_length` is True.
- Due to potential confusion in configuration, **if you don't require arbitrary resolution for finetuning**, it is recommended to set `video_sample_size`, `image_sample_size`, and `token_sample_size` to the same fixed value, such as **(320, 480, 512, 640, 960)**.
  - **All set to 320** represents **240P**.
  - **All set to 480** represents **320P**.
  - **All set to 640** represents **480P**.
  - **All set to 960** represents **720P**.

**Token Length Training Guide**:
- When `training_with_video_token_length` is enabled, the model trains based on token length.
- For example: a video with 512x512 resolution and 49 frames has a token length of 13,312, requiring `token_sample_size = 512`.
  - At 512x512 resolution, the number of video frames is 49 (~= 512 * 512 * 49 / 512 / 512).
  - At 768x768 resolution, the number of video frames is 21 (~= 512 * 512 * 49 / 768 / 768).
  - At 1024x1024 resolution, the number of video frames is 9 (~= 512 * 512 * 49 / 1024 / 1024).
  - These resolutions combined with their corresponding frame counts allow the model to generate videos of different sizes.

**Other Parameter Explanations**:
- `enable_bucket` is used to enable bucket training. When enabled, the model does not crop the images and videos at the center, but instead, it trains the entire images and videos after grouping them into buckets based on resolution.
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
- `train_mode` is used to specify the training mode, which can be either normal or i2v. Since Wan uses the inpaint model to achieve image-to-video generation, the default is set to inpaint mode. If you only wish to achieve text-to-video generation, you can remove this line, and it will default to the text-to-video mode.
- `resume_from_checkpoint` is used to set the training should be resumed from a previous checkpoint. Use a path or `"latest"` to automatically select the last available checkpoint.

### 3.4 Training Validation

You can configure validation parameters to periodically generate test videos during training, allowing you to monitor training progress and model quality.

**Validation Parameter Descriptions**:

| Parameter | Description | Recommended Value |
|------|------|--------|
| `--validation_steps` | Run validation every N steps | 2000 |
| `--validation_epochs` | Run validation every N epochs | 5 |
| `--validation_prompts` | Prompts for video generation validation | English prompts |
| `--validation_paths` | Validation image paths for I2V mode (i2v mode only) | `"asset/1.png"` |

**Normal Mode Example** (T2V Validation):

```bash
  --validation_steps=2000 \
  --validation_epochs=5 \
  --validation_prompts="A dog shaking head. The video is of high quality, and the view is very clear."
```

**I2V Mode Example** (I2V Validation):

```bash
  --validation_paths "asset/1.png" \
  --validation_steps=2000 \
  --validation_epochs=5 \
  --validation_prompts="A dog shaking head. The video is of high quality, and the view is very clear."
```

**Notes**:
- Validation videos will be saved to the `output_dir` directory
- Multiple prompts validation format: `--validation_prompts "prompt1" "prompt2" "prompt3"`
- `i2v` mode must provide `--validation_paths` parameter
- Distill model validation will use the steps defined in `denoising_step_indices_list` for inference

### 3.5 Training with FSDP

**If you experience insufficient GPU memory when using multiple GPUs with DeepSpeed-Zero-2**, you can switch to FSDP for training.

```sh
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/turbodiffusion/train_distill.py \
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
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_distill" \
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
  --train_mode="normal" \
  --trainable_modules "." \
  --low_vram
```

### 3.6 Other Backends

#### 3.6.1 Training with DeepSpeed-Zero-3

DeepSpeed Zero-3 is not highly recommended at the moment. In this repository, using FSDP has fewer errors and is more stable.

DeepSpeed Zero-3 is suitable for 14B Wan at high resolutions. After training, you can use the following command to get the final model:
```bash
python scripts/zero_to_bf16.py output_dir/checkpoint-{our-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

Training shell command is as follows:
```sh
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/turbodiffusion/train_distill.py \
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
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_distill" \
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
  --train_mode="normal" \
  --trainable_modules "." \
  --low_vram
```

#### 3.6.2 Training Without DeepSpeed and FSDP

**This approach is not recommended because without memory-saving backends, it easily causes insufficient GPU memory**. This is only provided as a reference shell for training.

```sh
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/turbodiffusion/train_distill.py \
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
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_distill" \
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
  --train_mode="normal" \
  --trainable_modules "." \
  --low_vram
```

### 3.7 Multi-Machine Distributed Training

**Suitable for**: Ultra-large datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines, each with 8 GPUs:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/turbodiffusion/train_distill.py \
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
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_distill" \
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
  --train_mode="normal" \
  --trainable_modules "." \
  --low_vram
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
export MASTER_ADDR="192.168.1.100"  # Same as Master
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # Note this is 1
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# Use the same accelerate launch command as Machine 0
```

#### 3.7.2 Multi-Machine Training Notes

- **Network Requirements**:
   - RDMA/InfiniBand recommended (high performance)
   - Without RDMA, add environment variables:
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **Data Synchronization**: All machines must be able to access the same data paths (NFS/shared storage)

---

## 4. Inference Testing

### 4.1 Inference Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|------|------|-------|
| `GPU_memory_mode` | GPU memory mode, see table below for options | `sequential_cpu_offload` |
| `ulysses_degree` | Ulysses parallelism degree for multi-GPU inference | 1 |
| `ring_degree` | Ring parallelism degree for multi-GPU inference | 1 |
| `fsdp_dit` | Use FSDP for Transformer during multi-GPU inference to save memory | `False` |
| `fsdp_text_encoder` | Use FSDP for text encoder during multi-GPU inference | `True` |
| `compile_dit` | Compile Transformer for faster inference (effective at fixed resolution) | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `sampler_name` | Sampler type: `Flow`, `Flow_Unipc`, `Flow_DPM++` | `Flow` |
| `transformer_path` | Path to trained Transformer weights | `models/Personalized_Model/TurboWan2.1-T2V-1.3B-480P.pth` |
| `vae_path` | Path to trained VAE weights | `None` |
| `lora_path` | LoRA weights path | `None` |
| `sample_size` | Generated video resolution `[height, width]` | `[480, 832]` |
| `video_length` | Number of generated frames | `81` |
| `fps` | Frames per second | `16` |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 support | `torch.bfloat16` |
| `validation_image_start` | Reference image path for I2V mode | `"asset/1.png"` |
| `prompt` | Positive prompt describing generated content | `"A stylish woman walks..."` |
| `negative_prompt` | Negative prompt to avoid certain content | Chinese negative prompt |
| `guidance_scale` | Guidance strength (distill models typically use 1.0) | 1.0 |
| `seed` | Random seed for reproducibility | 43 |
| `num_inference_steps` | Number of inference steps (typically 4 for distill models) | 4 |
| `lora_weight` | LoRA weight strength | 0.55 |
| `save_path` | Path to save generated videos | `samples/turbowan-videos-t2v` |

**GPU Memory Mode Descriptions**:

| Mode | Description | Memory Usage |
|------|------|---------|
| `model_full_load` | Full model loaded to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer groups switch between CPU/CUDA | Low |
| `sequential_cpu_offload` | Layer-by-layer offload (slowest) | Lowest |

### 4.2 Text-to-Video (T2V) Inference

Run single-GPU inference:

```bash
python examples/turbodiffusion/predict_t2v_wan2.1.py
```

Edit `examples/turbodiffusion/predict_t2v_wan2.1.py` according to your needs. For first-time inference, focus on the following key parameters. For other parameters, refer to the inference parameter descriptions above.

```python
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# Your actual model path
model_name = "models/Diffusion_Transformer/Wan2.1-T2V-1.3B"  
# Path to trained weights, e.g., "models/Personalized_Model/TurboWan2.1-T2V-1.3B-480P.pth"
transformer_path = "models/Personalized_Model/TurboWan2.1-T2V-1.3B-480P.pth"  
# Write based on your generation content
prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage..."  
# Distill models typically use 4 steps
num_inference_steps = 4
# ...
```

### 4.3 Image-to-Video (I2V) Inference

Run single-GPU inference:

```bash
python examples/turbodiffusion/predict_i2v_wan2.2.py
```

Edit `examples/turbodiffusion/predict_i2v_wan2.2.py` according to your needs. For first-time inference, focus on the following key parameters. For other parameters, refer to the inference parameter descriptions above.

```python
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# Your actual model path
model_name = "models/Diffusion_Transformer/Wan2.1-T2V-1.3B"  
# Path to trained weights
transformer_path = "models/Personalized_Model/TurboWan2.1-T2V-1.3B-480P.pth"  
# Reference image path
validation_image_start = "asset/1.png"
# Write based on your generation content
prompt = "The dog is shaking head. The video is of high quality, and the view is very clear..."  
# Distill models typically use 4 steps
num_inference_steps = 4
# ...
```

### 4.4 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/turbodiffusion/predict_t2v_wan2.1.py` or `examples/turbodiffusion/predict_i2v_wan2.2.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs used
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelism
ring_degree = 1     # Sequence dimension parallelism
```

**Configuration Principles**:
- `ulysses_degree` must evenly divide the model's head count
- `ring_degree` splits on the sequence dimension, which affects communication overhead. Try to avoid using it when heads can be evenly divided.

**Configuration Examples**:

| GPU Count | ulysses_degree | ring_degree | Description |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 2 | 2 | 1 | Head parallelism |
| 4 | 4 | 1 | Head parallelism |
| 8 | 8 | 1 | Head parallelism |
| 8 | 4 | 2 | Hybrid parallelism |

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/turbodiffusion/predict_t2v_wan2.1.py
```

---

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
- **TurboDiffusion Paper**: https://arxiv.org/abs/2411.19823