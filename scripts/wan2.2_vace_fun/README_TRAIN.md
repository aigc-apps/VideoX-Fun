# Wan2.2 VACE Fun Training Guide

This document provides a complete workflow for Wan2.2 VACE Fun (Video Generation and Editing Unified Framework) training, including environment setup, data preparation, various distributed training strategies, and inference testing.

> **Note**: Wan2.2 VACE Fun is a unified video generation and editing model based on the Wan2.2 architecture, supporting I2V (Image-to-Video), S2V (Subject Reference Video Generation), V2V Control (Controllable Video Generation), V2V Mask (Video Inpainting), and other tasks. Wan2.2 adopts a dual-Transformer architecture (high-noise/low-noise models). This guide covers the VACE module training workflow for Wan2.2 VACE Fun, supporting only the A14B model variant.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. VACE Module Training](#3-vace-module-training)
  - [3.1 Download Pre-trained Model](#31-download-pre-trained-model)
  - [3.2 Quick Start (DeepSpeed-Zero-2)](#32-quick-start-deepspeed-zero-2)
  - [3.3 VACE-specific Parameter Reference](#33-vace-specific-parameter-reference)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training with FSDP](#35-training-with-fsdp)
  - [3.6 Other Backends](#36-other-backends)
    - [3.6.1 Training with DeepSpeed-Zero-3](#361-training-with-deepspeed-zero-3)
    - [3.6.2 Training without DeepSpeed or FSDP](#362-training-without-deepspeed-or-fsdp)
  - [3.7 Multi-node Distributed Training](#37-multi-node-distributed-training)
    - [3.7.1 Environment Configuration](#371-environment-configuration)
    - [3.7.2 Multi-node Training Notes](#372-multi-node-training-notes)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameter Reference](#41-inference-parameter-reference)
  - [4.2 VACE Video Generation Inference](#42-vace-video-generation-inference)
    - [4.2.1 Inference Script Selection](#421-inference-script-selection)
    - [4.2.2 I2V Inference (Image-to-Video)](#422-i2v-inference-image-to-video)
    - [4.2.3 S2V Inference (Subject Reference Video Generation)](#423-s2v-inference-subject-reference-video-generation)
    - [4.2.4 V2V Control Inference (Controllable Video Generation)](#424-v2v-control-inference-controllable-video-generation)
    - [4.2.5 V2V Control + Ref Inference (Control + Reference Image)](#425-v2v-control--ref-inference-control--reference-image)
    - [4.2.6 V2V Mask Inference (Video Inpainting)](#426-v2v-mask-inference-video-inpainting)
  - [4.3 Multi-GPU Parallel Inference](#43-multi-gpu-parallel-inference)
- [5. More Resources](#5-more-resources)

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

When using Docker, please ensure that the GPU driver and CUDA environment are correctly installed on the machine, then execute the following commands:

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset with control signals and subject reference images containing some training data.

```bash
# Download the official example dataset (with control signals + subject reference images)
modelscope download --dataset PAI/X-Fun-Videos-Controls-Demo --local_dir ./datasets/X-Fun-Videos-Controls-Demo
```

After downloading, the dataset contains the following metadata files:
- `metadata.json`: Basic format (control video paths only)
- `metadata_add_width_height.json`: With width/height info (recommended for V2V Control training)
- `metadata_add_width_height_add_objects.json`: With width/height + subject reference images (recommended for S2V training)

### 2.2 Dataset Structure

VACE training datasets require original videos with corresponding control signal videos (e.g., canny edge videos, pose videos, depth videos, etc.). For S2V (Subject Reference Video Generation) training, subject reference images are also needed.

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

**Key field descriptions**:
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

## 3. VACE Module Training

### 3.1 Download Pre-trained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download Wan2.2 VACE Fun official weights
modelscope download --model PAI/Wan2.2-VACE-Fun-A14B --local_dir models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B
```

### 3.2 Quick Start (DeepSpeed-Zero-2)

If you have downloaded data from **Quick Test Dataset** and weights from **Download Pre-trained Model**, you can directly copy the quick start command to launch training.

We recommend training with DeepSpeed-Zero-2 and FSDP. Here is an example shell configuration using DeepSpeed-Zero-2.

**Wan2.2 VACE Fun Training Example (DeepSpeed-Zero-2)**:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.2_vace_fun/train.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
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
  --output_dir="output_dir_wan2.2_vace_fun" \
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
  --boundary_type="low" \
  --trainable_modules "vace" \
  --low_vram
```

### 3.3 VACE-specific Parameter Reference

**Wan2.2 Dual-Transformer Architecture**:

Wan2.2 adopts an innovative dual-Transformer architecture:
- **Low Noise Model**: Handles low-noise stages (near final output)
- **High Noise Model**: Handles high-noise stages (initial generation)
- **Boundary Type (boundary_type)**:
  - `low`: Train low-noise model, high-noise model uses pre-trained weights (recommended for VACE module training)
  - `high`: Train high-noise model, low-noise model uses pre-trained weights
  - `full`: Single model training (for single-Transformer models)

**Key VACE Parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config_path` | Config file path | `config/wan2.2/wan_civitai_t2v.yaml` |
| `--pretrained_model_name_or_path` | Pre-trained model path | `models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B` |
| `--train_data_dir` | Training data directory | `datasets/X-Fun-Videos-Controls-Demo/` |
| `--train_data_meta` | Training data metadata file | `datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json` |
| `--train_batch_size` | Batch size per device | 1 |
| `--image_sample_size` | Max image training resolution | 640 |
| `--video_sample_size` | Max video training resolution | 640 |
| `--token_sample_size` | Token sample size | 640 |
| `--video_sample_stride` | Video sampling stride | 2 |
| `--video_sample_n_frames` | Number of video sampling frames | 81 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 1 |
| `--dataloader_num_workers` | DataLoader subprocess count | 8 |
| `--num_train_epochs` | Training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 50 |
| `--learning_rate` | Initial learning rate (recommended for VACE module training) | 2e-05 |
| `--lr_scheduler` | LR scheduler: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup` | `constant_with_warmup` |
| `--lr_warmup_steps` | LR warmup steps | 100 |
| `--seed` | Random seed for reproducible training | 42 |
| `--output_dir` | Output directory | `output_dir_wan2.2_vace_fun` |
| `--gradient_checkpointing` | Enable gradient checkpointing to save memory | - |
| `--mixed_precision` | Mixed precision: `no`, `fp16`, `bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon value | 1e-10 |
| `--vae_mini_batch` | VAE encoding mini batch size | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--enable_bucket` | Enable bucket training, no cropping of images/videos, train by resolution grouping | - |
| `--random_hw_adapt` | Auto-scale images/videos to random sizes within `[min_size, max_size]` range | - |
| `--training_with_video_token_length` | Train by token length for arbitrary resolution | - |
| `--uniform_sampling` | Uniform timestep sampling (recommended) | - |
| `--low_vram` | Low VRAM mode for better memory efficiency | - |
| `--boundary_type` | Wan2.2 boundary type: `low`, `high`, `full` | `low` |
| `--control_ref_image` | Reference image source: `first_frame`, `random` | `random` |
| `--trainable_modules` | Trainable modules (`vace` trains only VACE-related modules) | `"vace"` |
| `--trainable_modules_low_learning_rate` | Modules trained with lower learning rate | `[]` |
| `--resume_from_checkpoint` | Resume training path, use `"latest"` for auto-selection | None |
| `--validation_steps` | Validate every N steps | 2000 |
| `--validation_epochs` | Validate every N epochs | 5 |
| `--validation_prompts` | Validation prompts | `"A brown dog shaking its head..."` |
| `--validation_paths` | Validation control video paths | `"asset/pose.mp4"` |
| `--use_8bit_adam` | Use 8-bit Adam optimizer to save memory | - |
| `--use_came` | Use CAME optimizer | - |

**Sample Size Configuration Guide**:
- `video_sample_size` represents the video resolution; when `random_hw_adapt` is True, it represents the minimum resolution for both video and images.
- `image_sample_size` represents the image resolution; when `random_hw_adapt` is True, it represents the maximum resolution for both video and images.
- `token_sample_size` represents the resolution corresponding to the max token length when `training_with_video_token_length` is True.
- To avoid confusion, **if you don't need arbitrary resolution finetuning**, set `video_sample_size`, `image_sample_size`, and `token_sample_size` to the same fixed value, e.g., **(320, 480, 512, 640, 960)**.
  - **All set to 320** represents **240P**.
  - **All set to 480** represents **320P**.
  - **All set to 640** represents **480P**.
  - **All set to 960** represents **720P**.

**Token Length Training**:
- When `training_with_video_token_length` is enabled, the model trains by token length.
- Example: A 512x512 resolution, 49-frame video has a token length of 13,312, requiring `token_sample_size = 512`.
  - At 512x512 resolution, video frames = 49 (~= 512 * 512 * 49 / 512 / 512).
  - At 768x768 resolution, video frames = 21 (~= 512 * 512 * 49 / 768 / 768).
  - At 1024x1024 resolution, video frames = 9 (~= 512 * 512 * 49 / 1024 / 1024).
  - These resolution-frame combinations enable generating videos of different sizes.

### 3.4 Training Validation

You can configure validation parameters to periodically generate test videos during training to monitor training progress and model quality.

**Validation Parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--validation_steps` | Validate every N steps | 2000 |
| `--validation_epochs` | Validate every N epochs | 5 |
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
- Validation videos are saved to the `output_dir` directory
- Multi-prompt format: `--validation_prompts "prompt1" "prompt2" "prompt3"`
- Wan2.2 VACE Fun validation automatically selects single or dual Transformer based on `boundary_type`
- `validation_paths` should correspond one-to-one with `validation_prompts`, pointing to control video files

### 3.5 Training with FSDP

**If DeepSpeed-Zero-2 runs out of memory on multi-GPU setups**, switch to FSDP for training.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=VaceWanAttentionBlock,BaseWanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.2_vace_fun/train.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
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
  --output_dir="output_dir_wan2.2_vace_fun" \
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
  --boundary_type="low" \
  --trainable_modules "vace" \
  --low_vram
```

> **Note**: In this repository, FSDP is more stable and has fewer errors than DeepSpeed-Zero-3. When DeepSpeed-Zero-2 encounters memory issues on multi-GPU, please use FSDP.

### 3.6 Other Backends

#### 3.6.1 Training with DeepSpeed-Zero-3

DeepSpeed Zero-3 is currently not strongly recommended. In this repository, FSDP has fewer errors and is more stable.

DeepSpeed Zero-3 is suitable for high-resolution 14B Wan training. After training, you can obtain the final model with:
```bash
python scripts/zero_to_bf16.py output_dir/checkpoint-{our-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

Training shell command:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/wan2.2_vace_fun/train.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
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
  --output_dir="output_dir_wan2.2_vace_fun" \
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
  --boundary_type="low" \
  --trainable_modules "vace" \
  --low_vram
```

#### 3.6.2 Training without DeepSpeed or FSDP

**This approach is not recommended as it lacks memory-saving backends and can easily cause out-of-memory issues**. The training shell below is provided for reference only.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.2_vace_fun/train.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
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
  --output_dir="output_dir_wan2.2_vace_fun" \
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
  --boundary_type="low" \
  --trainable_modules "vace" \
  --low_vram
```

> **Note**: This is similar to the `train.sh` script but uses the correct dataset path. The `train.sh` script can be used as a starting point for single-GPU training.

### 3.7 Multi-node Distributed Training

**Suitable for**: Ultra-large datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines, each with 8 GPUs:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total machines
export NUM_PROCESS=16                # Total processes = machines x 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.2_vace_fun/train.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
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
  --output_dir="output_dir_wan2.2_vace_fun" \
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
  --boundary_type="low" \
  --trainable_modules "vace" \
  --low_vram
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
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

#### 3.7.2 Multi-node Training Notes

- **Network Requirements**:
   - RDMA/InfiniBand recommended (high performance)
   - Without RDMA, add environment variables:
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **Data Sync**: All machines must access the same data path (NFS/shared storage)

---

## 4. Inference Testing

### 4.1 Inference Parameter Reference

**Key Parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `GPU_memory_mode` | GPU memory management mode, see table below | `model_group_offload` |
| `ulysses_degree` | Head dimension parallelism, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelism, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP on Transformer for multi-GPU inference | `False` |
| `fsdp_text_encoder` | Use FSDP on text encoder | `True` |
| `compile_dit` | Compile Transformer for faster inference (fixed resolution) | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B` |
| `sampler_name` | Sampler type: `Flow`, `Flow_Unipc`, `Flow_DPM++` | `Flow` |
| `transformer_path` | Trained low-noise Transformer weight path | `None` |
| `transformer_high_path` | Trained high-noise Transformer weight path (dual-Transformer only) | `None` |
| `vae_path` | Trained VAE weight path | `None` |
| `sample_size` | Video resolution `[height, width]` | `[480, 832]` or `[832, 480]` |
| `video_length` | Number of video frames | `81` |
| `fps` | Frames per second | `16` |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 | `torch.bfloat16` |
| `control_video` | Control signal video path (V2V Control tasks) | `"asset/pose.mp4"` |
| `start_image` | Start frame image path (I2V tasks) | `"asset/1.png"` |
| `end_image` | End frame image path (optional) | `None` |
| `inpaint_video` | Video to inpaint (V2V Mask tasks) | `"asset/inpaint_video.mp4"` |
| `inpaint_video_mask` | Inpaint mask video path (V2V Mask tasks) | `"asset/inpaint_video_mask.mp4"` |
| `subject_ref_images` | Subject reference image paths (S2V / V2V Control+Ref tasks) | `["asset/8.png", "asset/ref_1.png"]` |
| `vace_context_scale` | VACE context scale factor | `1.00` |
| `prompt` | Positive prompt describing content | `"A young woman stands on a sunny coastline..."` |
| `negative_prompt` | Negative prompt to avoid | `"Tone vivid, overexposure, static..."` |
| `guidance_scale` | Guidance scale | 5.0 |
| `seed` | Random seed for reproducibility | 43 |
| `num_inference_steps` | Inference steps | 50 |
| `save_path` | Generated video save path | `samples/vace-videos-fun` |

**GPU Memory Management Modes**:

| Mode | Description | Memory Usage |
|------|-------------|--------------|
| `model_full_load` | Entire model loaded to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer groups switch between CPU/CUDA | Low |
| `sequential_cpu_offload` | Layer-by-layer offload (slowest) | Lowest |

### 4.2 VACE Video Generation Inference

#### 4.2.1 Inference Script Selection

Wan2.2 VACE Fun provides multiple inference scripts. Choose based on your task:

| Script | Purpose | Key Input |
|--------|---------|-----------|
| `predict_i2v.py` | I2V (Image-to-Video) | `start_image` |
| `predict_s2v.py` | S2V (Subject Reference Video Generation) | `subject_ref_images` |
| `predict_v2v_control.py` | V2V Control (Controllable Video Generation) | `control_video` |
| `predict_v2v_control_ref.py` | V2V Control + Ref (Control + Reference Image) | `control_video` + `subject_ref_images` |
| `predict_v2v_mask.py` | V2V Mask (Video Inpainting) | `inpaint_video` + `inpaint_video_mask` |

> **Note**:
> - A14B models use dual-Transformer architecture (low-noise + high-noise models), requiring both `transformer_path` and `transformer_high_path`
> - `predict_v2v_control_ref.py` supports Control + Reference Image for better generation quality

#### 4.2.2 I2V Inference (Image-to-Video)

Run single-GPU inference:

```bash
python examples/wan2.2_vace_fun/predict_i2v.py
```

Modify `examples/wan2.2_vace_fun/predict_i2v.py` as needed. For first-time inference, focus on the following parameters. For other parameters, see the Inference Parameter Reference above.

```python
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# Set to your actual model path
model_name = "models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
# Trained low-noise weight path, e.g. "output_dir_wan2.2_vace_fun/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = None
# Trained high-noise weight path
transformer_high_path = None
# I2V start image
start_image = "asset/1.png"
# Video resolution [height, width]
sample_size = [480, 832]
# Write according to generation content
prompt = "A brown dog licks its tongue, sitting on a light-colored sofa in a cozy room..."
# ...
```

> **Note**: Wan2.2 VACE Fun I2V inference requires a `start_image`. The model generates video based on the input image.

#### 4.2.3 S2V Inference (Subject Reference Video Generation)

```bash
python examples/wan2.2_vace_fun/predict_s2v.py
```

```python
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# Set to your actual model path
model_name = "models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
# Trained low-noise weight path
transformer_path = None
# Trained high-noise weight path
transformer_high_path = None
# Subject reference image list
subject_ref_images = ["asset/8.png", "asset/ref_1.png"]
# Video resolution [height, width]
sample_size = [480, 832]
# VACE context scale factor
vace_context_scale = 1.00
# Write according to generation content
prompt = "The sea breeze composes, waves beat time. She holds a bright yellow camera..."
# ...
```

> **Note**: S2V inference uses `subject_ref_images` to provide subject reference images. The model generates videos that maintain subject consistency. `vace_context_scale` controls the strength of subject reference.

#### 4.2.4 V2V Control Inference (Controllable Video Generation)

Run single-GPU inference:

```bash
python examples/wan2.2_vace_fun/predict_v2v_control.py
```

Modify `examples/wan2.2_vace_fun/predict_v2v_control.py`, focusing on:

```python
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# Set to your actual model path
model_name = "models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
# Trained low-noise weight path
transformer_path = None
# Trained high-noise weight path
transformer_high_path = None
# Control signal video (e.g., pose video)
control_video = "asset/pose.mp4"
# Video resolution [height, width]
sample_size = [832, 480]
# No reference images
subject_ref_images = None
# Write according to generation content
prompt = "A young woman stands on a sunny coastline, wearing a refreshing white shirt and skirt..."
# ...
```

#### 4.2.5 V2V Control + Ref Inference (Control + Reference Image)

```bash
python examples/wan2.2_vace_fun/predict_v2v_control_ref.py
```

```python
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# Set to your actual model path
model_name = "models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
# Trained low-noise weight path
transformer_path = None
# Trained high-noise weight path
transformer_high_path = None
# Control signal video
control_video = "asset/pose.mp4"
# Reference image paths
subject_ref_images = ["asset/8.png"]
# Video resolution [height, width]
sample_size = [832, 480]
# VACE context scale factor
vace_context_scale = 1.00
# Write according to generation content
prompt = "A young woman stands on a sunny coastline, wearing a refreshing white shirt and skirt..."
# ...
```

> **Note**: V2V Control + Ref leverages both control signal videos and reference images for more precise controllable video generation.

#### 4.2.6 V2V Mask Inference (Video Inpainting)

```bash
python examples/wan2.2_vace_fun/predict_v2v_mask.py
```

```python
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# Set to your actual model path
model_name = "models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
# Trained low-noise weight path
transformer_path = None
# Trained high-noise weight path
transformer_high_path = None
# Video to inpaint
inpaint_video = "asset/inpaint_video.mp4"
# Inpaint mask video
inpaint_video_mask = "asset/inpaint_video_mask.mp4"
# Video resolution [height, width]
sample_size = [480, 832]
# No control video
control_video = None
# No reference images
subject_ref_images = None
# Write according to generation content
prompt = "A brown rabbit licks its tongue, sitting on a light-colored sofa in a cozy room..."
# ...
```

> **Note**: V2V Mask inference uses `inpaint_video` and `inpaint_video_mask` for video local inpainting. White regions in the mask video indicate areas to be inpainted.

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, faster inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit any inference script (e.g., `examples/wan2.2_vace_fun/predict_v2v_control_ref.py`):

```python
# Ensure ulysses_degree x ring_degree = number of GPUs used
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallel
ring_degree = 1     # Sequence dimension parallel
```

**Configuration Principles**:
- `ulysses_degree` must divide the model's head count
- `ring_degree` splits along sequence dimension and affects communication overhead; avoid using if heads can divide evenly

**Configuration Examples**:

| GPUs | ulysses_degree | ring_degree | Note |
|------|----------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallel |
| 8 | 8 | 1 | Head parallel |
| 8 | 4 | 2 | Mixed parallel |

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/wan2.2_vace_fun/predict_v2v_control_ref.py
```

---

## 5. More Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
