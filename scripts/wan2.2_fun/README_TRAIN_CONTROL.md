# Wan2.2 Fun Control Full Parameter Training Guide

This document provides a complete workflow for full parameter training of Wan2.2 Fun (Controllable Video Generation), including environment setup, data preparation, distributed training strategies, and inference testing.

> **Note**: Wan2.2 Fun is a controllable video generation model based on the Wan2.2 architecture, supporting guided video generation via control signals (e.g., pose video, depth maps). Wan2.2 adopts a dual-Transformer architecture (high-noise/low-noise models), while the 5B version uses a single-Transformer architecture. This guide covers the Control full parameter training workflow for Wan2.2 Fun, supporting both A14B and 5B model variants.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. Control Full Parameter Training](#3-control-full-parameter-training)
  - [3.1 Download Pretrained Model](#31-download-pretrained-model)
  - [3.2 Quick Start (DeepSpeed-Zero-2)](#32-quick-start-deepspeed-zero-2)
  - [3.3 Control-Specific Parameter Reference](#33-control-specific-parameter-reference)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training with FSDP](#35-training-with-fsdp)
  - [3.6 Other Backends](#36-other-backends)
  - [3.7 Multi-Node Distributed Training](#37-multi-node-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameter Reference](#41-inference-parameter-reference)
  - [4.2 Control Video Generation Inference](#42-control-video-generation-inference)
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

We provide a test dataset with control signals containing several training samples.

```bash
# Download official demo dataset (with control signals)
modelscope download --dataset PAI/X-Fun-Videos-Controls-Demo --local_dir ./datasets/X-Fun-Videos-Controls-Demo
```

### 2.2 Dataset Structure

Control training datasets require both original videos and corresponding control signal videos (e.g., pose video, depth video).

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   ├── 📂 control/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

> **Note**: The `train/` directory stores original videos, and the `control/` directory stores corresponding control signal videos. Control video filenames should match the original videos.

### 2.3 metadata.json Format

**Relative Path Format** (example):
```json
[
  {
    "file_path": "train/video001.mp4",
    "control_file_path": "control/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "type": "video",
    "width": 1024,
    "height": 1024
  },
  {
    "file_path": "train/video002.mp4",
    "control_file_path": "control/video002.mp4",
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
    "control_file_path": "/mnt/data/controls/sunset.mp4",
    "text": "A beautiful sunset over the ocean",
    "type": "video",
    "width": 1024,
    "height": 1024
  }
]
```

**Key Field Descriptions**:
- `file_path`: Original video path (relative or absolute)
- `control_file_path`: Control signal video path (relative or absolute, **required for Control training**)
- `text`: Video description (English prompt)
- `type`: Data type, fixed as `"video"`
- `width` / `height`: Video resolution (**recommended to provide**, used for bucket training; if omitted, they will be read automatically during training, which may slow down training when data is on slow storage like OSS)
  - Use `scripts/process_json_add_width_and_height.py` to extract width and height for JSON files without these fields. It supports both images and videos.
  - Usage: `python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Controls-Demo/metadata.json --output_file datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json`

### 2.4 Relative vs Absolute Path Usage

**Relative Path**:

If your data uses relative paths, set in the training script:

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

**Absolute Path**:

If your data uses absolute paths, set in the training script:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **Recommendation**: Use relative paths for small local datasets; use absolute paths for external storage (NAS, OSS) or shared multi-machine storage.

---

## 3. Control Full Parameter Training

### 3.1 Download Pretrained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download Wan2.2 Fun Control official weights
# A14B model (dual-Transformer architecture)
modelscope download --model PAI/Wan2.2-Fun-A14B-Control --local_dir models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control
# or 5B model (single-Transformer architecture)
# modelscope download --model PAI/Wan2.2-Fun-5B-Control --local_dir models/Diffusion_Transformer/Wan2.2-Fun-5B-Control
```

### 3.2 Quick Start (DeepSpeed-Zero-2)

After downloading the dataset as in **2.1** and the pretrained model as in **3.1**, you can directly copy and run the quick start command.

We recommend using DeepSpeed-Zero-2 or FSDP for training. Here we use DeepSpeed-Zero-2 as an example.

**Wan2.2 Fun Control Full Parameter Training Example (DeepSpeed-Zero-2)**:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.2_fun/train_control.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
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
  --output_dir="output_dir_wan2.2_fun_control" \
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
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_inpaint_info \
  --add_full_ref_image_in_self_attention \
  --boundary_type="low" \
  --trainable_modules "." \
  --low_vram
```

> **Note**: The `train_control.sh` script provides a basic template without DeepSpeed. For better multi-GPU training performance and memory efficiency, use the DeepSpeed-Zero-2 command above.

### 3.3 Control-Specific Parameter Reference

**Wan2.2 Dual-Transformer Architecture Explanation**:

Wan2.2 adopts an innovative dual-Transformer architecture:
- **Low Noise Model**: Handles the low-noise stage (closer to final output)
- **High Noise Model**: Handles the high-noise stage (initial generation stage)
- **Boundary Type (boundary_type)**:
  - `low`: Train low-noise model, high-noise model uses pretrained weights (recommended for T2V/I2V/Control full fine-tuning)
  - `high`: Train high-noise model, low-noise model uses pretrained weights
  - `full`: Single model training (for single-Transformer models like TI2V-5B)

**Control Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config_path` | Config file path | `config/wan2.2/wan_civitai_i2v.yaml` |
| `--pretrained_model_name_or_path` | Pretrained model path | `models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control` |
| `--train_data_dir` | Training data directory | `datasets/X-Fun-Videos-Controls-Demo/` |
| `--train_data_meta` | Training data metadata file | `datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json` |
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
| `--learning_rate` | Initial learning rate (recommended for full parameter training) | 2e-05 |
| `--lr_scheduler` | LR scheduler: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup` | `constant_with_warmup` |
| `--lr_warmup_steps` | Learning rate warmup steps | 100 |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir_wan2.2_fun_control` |
| `--gradient_checkpointing` | Enable gradient checkpointing to save memory | - |
| `--mixed_precision` | Mixed precision: `no`, `fp16`, `bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon | 1e-10 |
| `--vae_mini_batch` | Mini-batch size for VAE encoding | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--enable_bucket` | Enable bucket training without cropping | - |
| `--random_hw_adapt` | Randomly scale images/videos to random sizes in `[min_size, max_size]` | - |
| `--training_with_video_token_length` | Train based on token length, supports arbitrary resolutions | - |
| `--uniform_sampling` | Uniform timestep sampling (recommended) | - |
| `--low_vram` | Low VRAM mode for better memory efficiency | - |
| `--boundary_type` | Wan2.2 boundary type: `low`, `high`, `full` | `low` |
| `--train_mode` | Training mode: `control`, `control_ref`, `control_camera_ref` | `control_ref` |
| `--control_ref_image` | Reference image source: `first_frame`, `random` | `random` |
| `--add_full_ref_image_in_self_attention` | Inject full reference image info into self-attention | - |
| `--add_inpaint_info` | Inject inpainting info into self-attention | - |
| `--trainable_modules` | Trainable modules (`"."` means all modules) | `"."` |
| `--resume_from_checkpoint` | Resume training path, use `"latest"` for auto-select | None |
| `--validation_steps` | Run validation every N steps | 2000 |
| `--validation_epochs` | Run validation every N epochs | 5 |
| `--validation_prompts` | Prompts for validation video generation | `"A brown dog shaking its head..."` |
| `--validation_paths` | Control video path for validation | `"asset/pose.mp4"` |
| `--use_deepspeed` | Enable DeepSpeed distributed training | - |
| `--use_fsdp` | Enable FSDP distributed training | - |
| `--use_8bit_adam` | Use 8-bit Adam optimizer to save memory | - |
| `--use_came` | Use CAME optimizer | - |
| `--multi_stream` | Use CUDA multi-stream for better performance | - |
| `--snr_loss` | Use SNR loss | - |
| `--weighting_scheme` | Timestep weighting scheme | `none` |
| `--motion_sub_loss` | Enable motion sub-loss for better temporal consistency | - |
| `--motion_sub_loss_ratio` | Motion sub-loss ratio | 0.25 |

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
  - At 512x512, video frames = 49 (~= 512 * 512 * 49 / 512 / 512)
  - At 768x768, video frames = 21 (~= 512 * 512 * 49 / 768 / 768)
  - At 1024x1024, video frames = 9 (~= 512 * 512 * 49 / 1024 / 1024)

### 3.4 Training Validation

You can configure validation parameters to periodically generate test videos during training to monitor progress and model quality.

**Validation Parameter Descriptions**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--validation_steps` | Run validation every N steps | 2000 |
| `--validation_epochs` | Run validation every N epochs | 5 |
| `--validation_prompts` | Prompts for validation video generation | None |
| `--validation_paths` | Control video path for validation | None |

**Validation Example** (Control mode):

```bash
  --validation_paths "asset/pose.mp4" \
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="In this sunlit outdoor garden, a beautiful woman wears a knee-length white sleeveless dress, its hem swaying gently with her graceful movements like a dancing butterfly. Sunlight filters through the leaves, casting dappled shadows that highlight her soft features and clear eyes, enhancing her elegance. Every motion seems to speak of youth and vitality as she spins on the grass, her skirt fluttering around her, as if the entire garden rejoices in her dance. Colorful flowers all around—roses, chrysanthemums, lilies—sway in the breeze, releasing their fragrances and creating a relaxed and joyful atmosphere."
```

**Notes**:
- Validation videos are saved to the `output_dir` directory
- Multiple prompts: `--validation_prompts "prompt1" "prompt2" "prompt3"`
- Wan2.2 Fun validation automatically selects single or dual-Transformer based on `boundary_type`
- `validation_paths` should correspond one-to-one with `validation_prompts`, pointing to control video files
- When `train_mode="control_ref"`, validation uses both control video and reference image

### 3.5 Training with FSDP

**If GPU memory is insufficient with DeepSpeed-Zero-2 on multiple GPUs**, switch to FSDP.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.2_fun/train_control.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
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
  --output_dir="output_dir_wan2.2_fun_control" \
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
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_inpaint_info \
  --add_full_ref_image_in_self_attention \
  --boundary_type="low" \
  --trainable_modules "." \
  --low_vram
```

> **Note**: In this repository, FSDP is more stable and has fewer errors than DeepSpeed-Zero-3. Use FSDP when DeepSpeed-Zero-2 runs into memory issues on multi-GPU setups.

### 3.6 Other Backends

#### 3.6.1 Training with DeepSpeed-Zero-3

DeepSpeed Zero-3 is not highly recommended. FSDP has fewer errors and is more stable in this repository.

DeepSpeed Zero-3 is suitable for high-resolution 14B Wan training. After training, use the following command to get the final model:
```bash
python scripts/zero_to_bf16.py output_dir/checkpoint-{our-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

Training shell command:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/wan2.2_fun/train_control.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
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
  --output_dir="output_dir_wan2.2_fun_control" \
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
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_inpaint_info \
  --add_full_ref_image_in_self_attention \
  --boundary_type="low" \
  --trainable_modules "." \
  --low_vram
```

#### 3.6.2 Training Without DeepSpeed and FSDP

**This approach is not recommended as it lacks memory-saving backends**. Provided for reference only.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.2_fun/train_control.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
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
  --output_dir="output_dir_wan2.2_fun_control" \
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
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_inpaint_info \
  --add_full_ref_image_in_self_attention \
  --boundary_type="low" \
  --trainable_modules "." \
  --low_vram
```

> **Note**: Similar to `train_control.sh`, but with correct dataset paths. `train_control.sh` can be used as a starting point for single-GPU training.

### 3.7 Multi-Node Distributed Training

**Suitable for**: Ultra-large-scale datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines with 8 GPUs each:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control"
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

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.2_fun/train_control.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
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
  --output_dir="output_dir_wan2.2_fun_control" \
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
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_inpaint_info \
  --add_full_ref_image_in_self_attention \
  --boundary_type="low" \
  --trainable_modules "." \
  --low_vram
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control"
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

#### 3.7.2 Multi-Node Training Notes

- **Network Requirements**:
  - Recommended: RDMA/InfiniBand (high performance)
  - Without RDMA, add environment variables:
    ```bash
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    ```

- **Data Synchronization**: All machines must access the same data paths (NFS/shared storage)

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
| `model_name` | Model path | `models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control` |
| `sampler_name` | Sampler type: `Flow`, `Flow_Unipc`, `Flow_DPM++` | `Flow` |
| `transformer_path` | Path to trained low-noise Transformer weights | `None` |
| `transformer_high_path` | Path to trained high-noise Transformer weights (dual-Transformer only) | `None` |
| `vae_path` | Path to trained VAE weights | `None` |
| `sample_size` | Generated video resolution `[height, width]` | `[832, 480]` (A14B) or `[1280, 704]` (5B) |
| `video_length` | Number of video frames | `81` (A14B) or `121` (5B) |
| `fps` | Frames per second | `16` (A14B) or `24` (5B) |
| `weight_dtype` | Model weight precision | `torch.bfloat16` |
| `control_video` | Control signal video path (e.g., pose video) | `"asset/pose.mp4"` |
| `control_camera_txt` | Camera control text path (optional) | `None` |
| `ref_image` | Reference image path (control_ref mode) | `"asset/8.png"` |
| `start_image` | Starting frame image path (optional, for inpaint mode) | `None` |
| `end_image` | Ending frame image path (optional) | `None` |
| `prompt` | Positive prompt | `"A young woman standing on a sunny coastline..."` |
| `negative_prompt` | Negative prompt | `"low resolution, low quality..."` |
| `guidance_scale` | Guidance strength | 6.0 |
| `seed` | Random seed | 43 |
| `num_inference_steps` | Number of inference steps | 50 |
| `save_path` | Path to save generated video | `samples/wan-videos-fun-control` |

**GPU Memory Management Modes**:

| Mode | Description | Memory Usage |
|------|-------------|--------------|
| `model_full_load` | Full model loaded to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer groups switch between CPU/CUDA | Low |
| `sequential_cpu_offload` | Layer-by-layer offload (slowest) | Lowest |

### 4.2 Control Video Generation Inference

#### 4.2.1 Inference Script Selection

Wan2.2 Fun Control provides multiple inference scripts. Choose based on your model version and task:

| Script | Model Version | Architecture | Main Purpose |
|--------|--------------|--------------|--------------|
| `predict_v2v_control_ref.py` | A14B | Dual-Transformer | Control + Reference Image (recommended) |
| `predict_v2v_control.py` | A14B | Dual-Transformer | Pure Control (no reference image) |
| `predict_v2v_control_ref_5b.py` | 5B | Single-Transformer | Control + Reference Image (5B) |
| `predict_v2v_control_5b.py` | 5B | Single-Transformer | Pure Control (5B, no reference image) |

> **Note**:
> - A14B models use dual-Transformer architecture (low-noise + high-noise), requiring `transformer_path` and `transformer_high_path`
> - 5B models use single-Transformer architecture, only need `transformer_path`, keep `transformer_high_path` as `None`
> - `predict_v2v_control_ref.py` supports Control + Reference Image, usually yielding better results

#### 4.2.2 A14B Control + Ref Inference (Dual-Transformer)

Single-GPU inference:

```bash
python examples/wan2.2_fun/predict_v2v_control_ref.py
```

Edit `examples/wan2.2_fun/predict_v2v_control_ref.py`. For initial inference, focus on these parameters:

```python
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# Based on actual model path
model_name = "models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control"
# Trained low-noise weights path, e.g., "output_dir_wan2.2_fun_control/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = None
# Trained high-noise weights path
transformer_high_path = None
# Control signal video (e.g., pose video)
control_video = "asset/pose.mp4"
# Reference image path (control_ref mode)
ref_image = "asset/8.png"
# Write based on desired content
prompt = "A young woman standing on a sunny coastline, wearing a dark blue vest and a crisp white shirt..."
# ...
```

> **Note**: Wan2.2 Fun Control is mainly used for controllable video generation. After providing `control_video`, the model guides video generation according to the control signal.

#### 4.2.3 A14B Pure Control Inference (Dual-Transformer, No Reference Image)

```bash
python examples/wan2.2_fun/predict_v2v_control.py
```

```python
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# Based on actual model path
model_name = "models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control"
# Trained low-noise weights path
transformer_path = None
# Trained high-noise weights path
transformer_high_path = None
# Control signal video
control_video = "asset/pose.mp4"
# No reference image
ref_image = None
# Write based on desired content
prompt = "A young woman standing on a sunny coastline..."
# ...
```

#### 4.2.4 5B Control + Ref Inference (Single-Transformer)

Single-GPU inference:

```bash
python examples/wan2.2_fun/predict_v2v_control_ref_5b.py
```

Edit `examples/wan2.2_fun/predict_v2v_control_ref_5b.py`, focus on these parameters:

```python
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# 5B model path
model_name = "models/Diffusion_Transformer/Wan2.2-Fun-5B-Control/"
# Trained weights path (5B is single-Transformer, only set transformer_path)
transformer_path = None
# 5B models do not use high-noise Transformer, keep as None
transformer_high_path = None
# Control signal video
control_video = "asset/pose.mp4"
# Reference image path
ref_image = "asset/8.png"
# Write based on desired content
prompt = "A young woman standing on a sunny coastline..."
# ...
```

> **Note**:
> - 5B models use single-Transformer architecture, simpler configuration and lower memory usage
> - If trained with `boundary_type="full"`, inference only needs `transformer_path`, no need for `transformer_high_path`

#### 4.2.5 5B Pure Control Inference (Single-Transformer, No Reference Image)

```bash
python examples/wan2.2_fun/predict_v2v_control_5b.py
```

```python
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# 5B model path
model_name = "models/Diffusion_Transformer/Wan2.2-Fun-5B-Control/"
# Trained weights path (5B is single-Transformer, only set transformer_path)
transformer_path = None
# 5B models do not use high-noise Transformer, keep as None
transformer_high_path = None
# Control signal video
control_video = "asset/pose.mp4"
# No reference image
ref_image = None
# Write based on desired content
prompt = "A young woman standing on a sunny coastline..."
# ...
```

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/wan2.2_fun/predict_v2v_control_ref.py`:

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
torchrun --nproc-per-node=2 examples/wan2.2_fun/predict_v2v_control_ref.py
```

---

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
