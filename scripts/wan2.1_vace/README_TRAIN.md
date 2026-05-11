# Wan2.1 VACE Training Guide

This document provides a complete workflow for training Wan2.1 VACE (Unified Video Generation and Editing Model), including environment setup, data preparation, distributed training strategies, and inference testing.

> **Note**: Wan2.1 VACE is a unified video generation and editing model based on the Wan2.1 architecture, supporting multiple tasks such as Image-to-Video (I2V), Subject Reference Video Generation (S2V), Controllable Video Generation (V2V Control), and Masked Video-to-Video (MV2V). This guide covers the training workflow for Wan2.1 VACE, supporting both 1.3B and 14B model variants.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. VACE Training](#3-vace-training)
  - [3.1 Download Pretrained Model](#31-download-pretrained-model)
  - [3.2 Quick Start (DeepSpeed-Zero-2)](#32-quick-start-deepspeed-zero-2)
  - [3.3 VACE-Specific Parameter Reference](#33-vace-specific-parameter-reference)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training with FSDP](#35-training-with-fsdp)
  - [3.6 Other Backends](#36-other-backends)
  - [3.7 Multi-Node Distributed Training](#37-multi-node-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameter Reference](#41-inference-parameter-reference)
  - [4.2 VACE Video Generation Inference](#42-vace-video-generation-inference)
  - [4.3 Multi-GPU Parallel Inference](#43-multi-gpu-parallel-inference)
- [5. Additional Resources](#5-additional-resources)

---

## 1. Environment Setup

**Option 1: Using requirements.txt**

```bash
pip install -r requirements.txt
```

**Option 2: Manual Installation**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl deepspeed==0.17.0 numpy==1.26.4
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

**Option 3: Using Docker**

When using Docker, ensure that GPU drivers and CUDA are properly installed, then execute:

```bash
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset with control signals and subject reference images containing several training samples.

```bash
# Download official demo dataset (with control signals + subject reference images)
modelscope download --dataset PAI/X-Fun-Videos-Controls-Demo --local_dir ./datasets/X-Fun-Videos-Controls-Demo
```

After downloading, the dataset contains the following metadata files:
- `metadata.json`: Basic format (control video paths only)
- `metadata_add_width_height.json`: With width/height info
- `metadata_add_width_height_add_objects.json`: With width/height + subject reference images (recommended for S2V training)

### 2.2 Dataset Structure

VACE training datasets require both original videos and corresponding control signal videos (e.g., canny edge videos, pose videos, depth videos, etc.). For S2V (Subject Reference Video Generation) training, subject reference images are also needed.

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/                    # Original training videos
│   │   ├── 📄 00000000.mp4
│   │   ├── 📄 00000001.mp4
│   │   └── 📄 ...
│   ├── 📂 canny/                    # Control signal videos (e.g., canny edge detection)
│   │   ├── 📄 00000000.mp4
│   │   ├── 📄 00000001.mp4
│   │   └── 📄 ...
│   ├── 📂 object/                   # Subject reference images (optional, for S2V training)
│   │   ├── 📂 00000000/
│   │   │   └── 📄 0-0.jpg
│   │   ├── 📂 00000001/
│   │   │   └── 📄 1-0.jpg
│   │   └── 📂 ...
│   ├── 📄 metadata.json
│   └── 📄 metadata_add_width_height.json
```

> **Note**:
> - `train/` directory stores original videos
> - `canny/` (or `pose/`, `depth/`, etc.) directory stores control signal videos that correspond one-to-one with the original videos. Filenames should match the original videos
> - `object/` directory stores subject reference images (optional). Each video has a subdirectory containing its subject reference images
> - Control signal directory name is customizable, as long as `control_file_path` in `metadata.json` correctly points to it

### 2.3 metadata.json Format

**Basic format** (control videos only):
```json
[
  {
    "file_path": "train/00000000.mp4",
    "text": "A young woman gently turns her head to the right...",
    "type": "video",
    "control_file_path": "canny/00000000.mp4"
  },
  {
    "file_path": "train/00000001.mp4",
    "text": "A young woman parts her lips slightly...",
    "type": "video",
    "control_file_path": "canny/00000001.mp4"
  }
]
```

**Format with width/height** (recommended, corresponds to `metadata_add_width_height.json`):
```json
[
  {
    "file_path": "train/00000000.mp4",
    "text": "A young woman gently turns her head to the right...",
    "type": "video",
    "control_file_path": "canny/00000000.mp4",
    "height": 480,
    "width": 832
  }
]
```

**Format with subject reference images** (for S2V training, corresponds to `metadata_add_width_height_add_objects.json`):
```json
[
  {
    "file_path": "train/00000000.mp4",
    "text": "A young woman gently turns her head to the right...",
    "type": "video",
    "control_file_path": "canny/00000000.mp4",
    "height": 480,
    "width": 832,
    "object_file_path": [
      "object/00000000/0-0.jpg"
    ]
  }
]
```

**Absolute path format**:
```json
[
  {
    "file_path": "/mnt/data/train/00000000.mp4",
    "text": "A beautiful sunset over the ocean",
    "type": "video",
    "control_file_path": "/mnt/data/canny/00000000.mp4",
    "height": 480,
    "width": 832,
    "object_file_path": [
      "/mnt/data/object/00000000/0-0.jpg"
    ]
  }
]
```

**Key Field Descriptions**:
- `file_path`: Original video path (relative or absolute)
- `text`: Video description (English prompt)
- `type`: Data type, fixed as `"video"`
- `control_file_path`: Control signal video path (relative or absolute, **required for VACE training**)
- `object_file_path`: Subject reference image path list (optional, for S2V subject reference training). Each element is a path to a subject reference image; the order is randomly shuffled during training
- `width` / `height`: Video dimensions (**recommended to provide**, used for bucket training. If not provided, they will be read automatically during training, which may affect training speed when data is stored on slower systems like OSS).
  - Use `scripts/process_json_add_width_and_height.py` to extract width and height for JSON files without these fields, supporting both images and videos.
  - Usage: `python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Controls-Demo/metadata.json --output_file datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json`.

### 2.4 Relative vs Absolute Path Usage

**Relative paths**:

If data paths are relative, set in the training script:

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

**Absolute paths**:

If data paths are absolute, set in the training script:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **Recommendation**: Use relative paths for small local datasets; use absolute paths for external storage (NAS, OSS) or shared storage across multiple machines.

---

## 3. VACE Training

### 3.1 Download Pretrained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download Wan2.1 VACE official weights
# 1.3B model
modelscope download --model Wan-AI/Wan2.1-VACE-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-VACE-1.3B
# or 14B model
# modelscope download --model Wan-AI/Wan2.1-VACE-14B --local_dir models/Diffusion_Transformer/Wan2.1-VACE-14B
```

### 3.2 Quick Start (DeepSpeed-Zero-2)

If you have downloaded data from **Quick Test Dataset** and weights from **Download Pretrained Model**, you can directly copy the quick start command to launch training.

We recommend training with DeepSpeed-Zero-2 and FSDP. Here is an example shell configuration using DeepSpeed-Zero-2.

**Wan2.1 VACE Training Example (DeepSpeed-Zero-2)**:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-VACE-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_objects.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_vace/train.py \
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
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_vace" \
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
  --control_ref_image="random" \
  --trainable_modules "vace" \
  --low_vram
```

> **Note**: The `train.sh` script provides a basic template without DeepSpeed. For better multi-GPU training performance and memory efficiency, use the DeepSpeed-Zero-2 command above.

### 3.3 VACE-Specific Parameter Reference

**VACE Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config_path` | Config file path | `config/wan2.1/wan_civitai.yaml` |
| `--pretrained_model_name_or_path` | Pretrained model path | `models/Diffusion_Transformer/Wan2.1-VACE-1.3B` |
| `--train_data_dir` | Training data directory | `datasets/X-Fun-Videos-Controls-Demo/` |
| `--train_data_meta` | Training data metadata file | `datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_objects.json` |
| `--train_batch_size` | Batch size per step | 1 |
| `--image_sample_size` | Maximum training resolution for images | 640 |
| `--video_sample_size` | Maximum training resolution for videos | 640 |
| `--token_sample_size` | Token sample size | 640 |
| `--video_sample_stride` | Video sampling stride | 2 |
| `--video_sample_n_frames` | Number of video frames to sample | 81 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 1 |
| `--dataloader_num_workers` | Number of DataLoader workers | 8 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 50 |
| `--learning_rate` | Initial learning rate | 2e-05 |
| `--lr_scheduler` | LR scheduler: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup` | `constant_with_warmup` |
| `--lr_warmup_steps` | Learning rate warmup steps | 100 |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir_wan2.1_vace` |
| `--gradient_checkpointing` | Enable gradient checkpointing to save memory | - |
| `--mixed_precision` | Mixed precision: `no`, `fp16`, `bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon | 1e-10 |
| `--vae_mini_batch` | Mini-batch size for VAE encoding | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--enable_bucket` | Enable bucket training, no cropping of images/videos, train by resolution grouping | - |
| `--random_hw_adapt` | Randomly scale images/videos to random sizes in `[min_size, max_size]` | - |
| `--training_with_video_token_length` | Train based on token length, supports arbitrary resolutions | - |
| `--uniform_sampling` | Uniform timestep sampling (recommended) | - |
| `--low_vram` | Low VRAM mode for better memory efficiency | - |
| `--control_ref_image` | Reference image source: `first_frame`, `random` | `random` |
| `--trainable_modules` | Trainable modules (`"vace"` means only train VACE modules) | `"vace"` |
| `--trainable_modules_low_learning_rate` | Modules trained with lower learning rate | `[]` |
| `--resume_from_checkpoint` | Resume training path, use `"latest"` for auto-select | None |
| `--validation_steps` | Run validation every N steps | 2000 |
| `--validation_epochs` | Run validation every N epochs | 5 |
| `--validation_prompts` | Prompts for validation video generation | `"A brown dog shaking its head..."` |
| `--validation_paths` | Control video paths for validation | `"asset/pose.mp4"` |
| `--use_8bit_adam` | Use 8-bit Adam optimizer to save memory | - |
| `--use_came` | Use CAME optimizer | - |

**Sample Size Configuration Guide**:
- `video_sample_size` represents video resolution; when `random_hw_adapt` is True, it is the minimum resolution.
- `image_sample_size` represents image resolution; when `random_hw_adapt` is True, it is the maximum resolution.
- `token_sample_size` represents the resolution corresponding to the max token length when `training_with_video_token_length` is True.
- To avoid confusion, **if you don't need arbitrary resolution finetuning**, set `video_sample_size`, `image_sample_size`, and `token_sample_size` to the same fixed value, such as **(320, 480, 512, 640, 960)**.
  - **All 320** = **240P**
  - **All 480** = **320P**
  - **All 640** = **480P**
  - **All 960** = **720P**

**Token Length Training Explanation**:
- When `training_with_video_token_length` is enabled, the model trains based on token length.
- For example: a 512x512 video with 49 frames has a token length of 13,312, requiring `token_sample_size = 512`.
  - At 512x512 resolution, video frames = 49 (~= 512 * 512 * 49 / 512 / 512).
  - At 768x768 resolution, video frames = 21 (~= 512 * 512 * 49 / 768 / 768).
  - At 1024x1024 resolution, video frames = 9 (~= 512 * 512 * 49 / 1024 / 1024).
  - These resolution-frame combinations enable generating videos of different sizes.

### 3.4 Training Validation

You can configure validation parameters to periodically generate test videos during training to monitor progress and model quality.

**Validation Parameter Descriptions**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--validation_steps` | Run validation every N steps | 2000 |
| `--validation_epochs` | Run validation every N epochs | 5 |
| `--validation_prompts` | Prompts for validation video generation | None |
| `--validation_paths` | Control video paths for validation | None |

**Validation Example**:

```bash
  --validation_paths "asset/pose.mp4" \
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="In this sunlit outdoor garden, a beautiful woman wears a knee-length white sleeveless dress, its hem swaying gently with her graceful movements like a dancing butterfly. Sunlight filters through the leaves, casting dappled shadows that highlight her soft features and clear eyes, enhancing her elegance. Every motion seems to speak of youth and vitality as she spins on the grass, her skirt fluttering around her, as if the entire garden rejoices in her dance. Colorful flowers all around—roses, chrysanthemums, lilies—sway in the breeze, releasing their fragrances and creating a relaxed and joyful atmosphere."
```

**Notes**:
- Validation videos are saved to the `output_dir/sample` directory
- Multi-prompt format: `--validation_prompts "prompt1" "prompt2" "prompt3"`
- `validation_paths` should correspond one-to-one with `validation_prompts`, pointing to control video files
- VACE validation uses `WanVacePipeline` with `guidance_scale=4.5` and `num_inference_steps=25`

### 3.5 Training with FSDP

**If GPU memory is insufficient with DeepSpeed-Zero-2 on multiple GPUs**, switch to FSDP.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-VACE-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_objects.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=VaceWanAttentionBlock,BaseWanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.1_vace/train.py \
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
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_vace" \
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
  --control_ref_image="random" \
  --trainable_modules "vace" \
  --low_vram
```

> **Note**: In this repository, FSDP is more stable and has fewer errors than DeepSpeed-Zero-3. Use FSDP when DeepSpeed-Zero-2 runs into memory issues on multi-GPU setups.

### 3.6 Other Backends

#### 3.6.1 Training with DeepSpeed-Zero-3

DeepSpeed Zero-3 is not highly recommended. FSDP has fewer errors and is more stable in this repository.

DeepSpeed Zero-3 is suitable for high-resolution 14B Wan. After training, use the following command to get the final model:
```bash
python scripts/zero_to_bf16.py output_dir/checkpoint-{our-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

Training shell command:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-VACE-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_objects.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_vace/train.py \
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
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_vace" \
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
  --control_ref_image="random" \
  --trainable_modules "vace" \
  --low_vram
```

#### 3.6.2 Training Without DeepSpeed and FSDP

**This approach is not recommended as it lacks memory-saving backends**. Provided for reference only.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-VACE-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_objects.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.1_vace/train.py \
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
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_vace" \
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
  --control_ref_image="random" \
  --trainable_modules "vace" \
  --low_vram
```

> **Note**: Similar to `train.sh`, but with correct dataset paths. `train.sh` can be used as a starting point for single-GPU training.

### 3.7 Multi-Node Distributed Training

**Suitable for**: Ultra-large-scale datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines with 8 GPUs each:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-VACE-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_objects.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total machines
export NUM_PROCESS=16                # Total processes = machines x 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_vace/train.py \
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
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_vace" \
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
  --control_ref_image="random" \
  --trainable_modules "vace" \
  --low_vram
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-VACE-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_objects.json"
export MASTER_ADDR="192.168.1.100"  # Same as Master
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # Note: this is 1
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

# Use the same accelerate launch command as Machine 0
```

#### 3.7.2 Multi-Node Training Notes

- **Network Requirements**:
  - Recommended: RDMA/InfiniBand (high performance)
  - Without RDMA, add environment variables:
    ```bash
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    ```

- **Data Sync**: All machines must access the same data path (NFS/shared storage)

---

## 4. Inference Testing

### 4.1 Inference Parameter Reference

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `GPU_memory_mode` | GPU memory management mode | `model_group_offload` |
| `ulysses_degree` | Head dimension parallelism, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelism, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP for Transformer during multi-GPU inference | `False` |
| `fsdp_text_encoder` | Use FSDP for text encoder during multi-GPU inference | `True` |
| `compile_dit` | Compile Transformer for faster inference | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/Wan2.1-VACE-1.3B` |
| `sampler_name` | Sampler type: `Flow`, `Flow_Unipc`, `Flow_DPM++` | `Flow_Unipc` |
| `transformer_path` | Path to trained Transformer weights | `None` |
| `vae_path` | Path to trained VAE weights | `None` |
| `lora_path` | LoRA weights path | `None` |
| `sample_size` | Generated video resolution `[height, width]` | `[480, 832]` (I2V) or `[832, 480]` (S2V/V2V) |
| `video_length` | Number of video frames | `81` |
| `fps` | Frames per second | `16` |
| `weight_dtype` | Model weight precision | `torch.bfloat16` |
| `control_video` | Control signal video path (V2V Control task) | `"asset/pose.mp4"` |
| `start_image` | Starting image path (I2V task) | `"asset/1.png"` |
| `end_image` | Ending image path (optional) | `None` |
| `subject_ref_images` | Subject reference image list (S2V task) | `["asset/ref_1.png", "asset/ref_2.png"]` |
| `vace_context_scale` | VACE context scale factor | `1.00` |
| `prompt` | Positive prompt | `"A young woman standing on a sunny coastline..."` |
| `negative_prompt` | Negative prompt | `"low resolution, low quality..."` |
| `guidance_scale` | Guidance strength | 5.0 |
| `seed` | Random seed | 43 |
| `num_inference_steps` | Number of inference steps | 40 |
| `lora_weight` | LoRA weight strength | 0.55 |
| `save_path` | Path to save generated video | `samples/vace-videos` |

**GPU Memory Management Modes**:

| Mode | Description | Memory Usage |
|------|-------------|--------------|
| `model_full_load` | Full model loaded to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer groups switch between CPU/CUDA | Low |
| `sequential_cpu_offload` | Layer-by-layer offload (slowest) | Lowest |

### 4.2 VACE Video Generation Inference

#### 4.2.1 Inference Script Selection

Wan2.1 VACE provides multiple inference scripts. Choose based on your task type:

| Script | Task Type | Main Purpose |
|--------|-----------|--------------|
| `predict_i2v.py` | I2V (Image-to-Video) | Generate video from starting image |
| `predict_s2v.py` | S2V (Subject Reference Video) | Generate video with subject reference |
| `predict_v2v_control.py` | V2V Control (Controllable Video) | Generate video guided by control signal |

> **Note**:
> - Wan2.1 VACE uses single-Transformer architecture, only need to configure `transformer_path`, no high-noise Transformer path needed
> - `vace_context_scale` adjusts VACE editing intensity, default is 1.00
> - I2V uses `start_image`, S2V uses `subject_ref_images`, V2V Control uses `control_video`

#### 4.2.2 I2V Inference (Image-to-Video)

Single-GPU inference:

```bash
python examples/wan2.1_vace/predict_i2v.py
```

Edit `examples/wan2.1_vace/predict_i2v.py`. For first-time inference, focus on the following parameters. For other parameters, see the Inference Parameter Reference above.

```python
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# Based on actual model path
model_name = "models/Diffusion_Transformer/Wan2.1-VACE-1.3B"
# Trained weights path, e.g., "output_dir_wan2.1_vace/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = None
# Starting image path
start_image = "asset/1.png"
# Write according to generation content
prompt = "A young woman standing on a sunny coastline, wearing a dark blue vest and a crisp white shirt..."
# ...
```

> **Note**: I2V task uses `start_image` as the video starting frame. The model generates a continuation video based on the image content and prompt.

#### 4.2.3 S2V Inference (Subject Reference Video)

```bash
python examples/wan2.1_vace/predict_s2v.py
```

```python
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# Based on actual model path
model_name = "models/Diffusion_Transformer/Wan2.1-VACE-1.3B"
# Trained weights path
transformer_path = None
# Subject reference image list
subject_ref_images = ["asset/ref_1.png", "asset/ref_2.png"]
# Write according to generation content
prompt = "Warm sunlight spreads over the grass, a little girl with pigtails and a green bow..."
# ...
```

> **Note**: S2V task uses `subject_ref_images` to provide subject reference images. The model generates a video containing the specified subject.

#### 4.2.4 V2V Control Inference (Controllable Video)

```bash
python examples/wan2.1_vace/predict_v2v_control.py
```

```python
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# Based on actual model path
model_name = "models/Diffusion_Transformer/Wan2.1-VACE-1.3B"
# Trained weights path
transformer_path = None
# Control signal video (e.g., pose video)
control_video = "asset/pose.mp4"
# Write according to generation content
prompt = "A young woman standing on a sunny coastline..."
# ...
```

> **Note**: V2V Control task provides `control_video` as control signal. The model guides video generation according to the control signal.

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/wan2.1_vace/predict_v2v_control.py` (or other inference scripts):

```python
# Ensure ulysses_degree x ring_degree = number of GPUs used
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelism
ring_degree = 1     # Sequence dimension parallelism
```

**Configuration Principles**:
- `ulysses_degree` must divide the model's number of heads
- `ring_degree` splits along the sequence dimension and affects communication overhead; avoid when heads can be evenly divided

**Configuration Examples**:

| GPU Count | ulysses_degree | ring_degree | Description |
|-----------|---------------|-------------|-------------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelism |
| 8 | 8 | 1 | Head parallelism |
| 8 | 4 | 2 | Hybrid parallelism |

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/wan2.1_vace/predict_v2v_control.py
```

---

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
