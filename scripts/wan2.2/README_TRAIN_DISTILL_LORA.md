# Wan2.2 Distillation LoRA Training Guide

This document provides a complete workflow for distilling and fine-tuning Wan2.2 with LoRA, including environment setup, data preparation, distributed training, and inference testing.

> **Note**: Wan2.2 is a video generation model that supports Text-to-Video (T2V), Image-to-Video (I2V), and Text-Image-to-Video (TI2V). Wan2.2 adopts a dual-Transformer architecture (high-noise/low-noise models). This training method combines distillation (reducing inference steps) and LoRA (parameter-efficient fine-tuning) technologies. It can reduce inference steps from 25-50 to 4-8 steps with lower VRAM usage while maintaining or improving video generation quality.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. Distillation LoRA Training](#3-distillation-lora-training)
  - [3.1 Download Pretrained Model](#31-download-pretrained-model)
  - [3.2 Quick Start (DeepSpeed-Zero-2)](#32-quick-start-deepspeed-zero-2)
  - [3.3 Training Parameters Explanation](#33-training-parameters-explanation)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training with FSDP](#35-training-with-fsdp)
  - [3.6 Other Backends](#36-other-backends)
  - [3.7 Multi-Node Distributed Training](#37-multi-node-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameters Explanation](#41-inference-parameters-explanation)
  - [4.2 Text-to-Video (T2V) Inference](#42-text-to-video-t2v-inference)
  - [4.3 Image-to-Video (I2V) Inference](#43-image-to-video-i2v-inference)
  - [4.3.1 Text-Image-to-Video (TI2V) Inference](#431-text-image-to-video-ti2v-inference)
  - [4.4 Multi-GPU Parallel Inference](#44-multi-gpu-parallel-inference)
- [5. More Resources](#5-more-resources)

---

## 1. Environment Setup

**Method 1: Using requirements.txt**

```bash
pip install -r requirements.txt
```

**Method 2: Manual Installation**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl deepspeed==0.17.0 numpy==1.26.4
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

**Method 3: Using Docker**

When using Docker, please ensure that the graphics card driver and CUDA environment are correctly installed, then execute the following commands:

```
# Pull the image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# Enter the container
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset containing several training samples.

```bash
# Download the official demo dataset
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

**Relative Path Format** (example format):
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
- `width` / `height`: Video width and height (**recommended to provide**, used for bucket training. If not provided, it will be automatically read during training, which may affect training speed when data is stored on slower systems like OSS).
  - You can use the `scripts/process_json_add_width_and_height.py` file to extract width and height from JSON files without these fields. It supports processing both images and videos.
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

## 3. Distillation LoRA Training

### 3.1 Download Pretrained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download official Wan2.2 weights
# T2V model (Text-to-Video)
modelscope download --model Wan-AI/Wan2.2-T2V-A14B --local_dir models/Diffusion_Transformer/Wan2.2-T2V-A14B
# Or I2V model (Image-to-Video)
# modelscope download --model Wan-AI/Wan2.2-I2V-A14B --local_dir models/Diffusion_Transformer/Wan2.2-I2V-A14B
# Or TI2V model (Text-Image-to-Video)
# modelscope download --model Wan-AI/Wan2.2-TI2V-5B --local_dir models/Diffusion_Transformer/Wan2.2-TI2V-5B
```

### 3.2 Quick Start (DeepSpeed-Zero-2)

After following **2.1 Quick Test Dataset** to download data and **3.1 Download Pretrained Model** to download weights, directly copy and run the quick start command.

It is recommended to use DeepSpeed-Zero-2 or FSDP for training. Here we use DeepSpeed-Zero-2 as an example to configure the shell file.

The difference between DeepSpeed-Zero-2 and FSDP in this repository is whether model weights are sharded. **If VRAM is insufficient when using multiple GPUs with DeepSpeed-Zero-2**, you can switch to FSDP for training.

**Wan2.2 Dual-Transformer Architecture Explanation**:

Wan2.2 adopts an innovative dual-Transformer architecture:
- **Low Noise Model**: Responsible for processing the low-noise stage (close to final output)
- **High Noise Model**: Responsible for processing the high-noise stage (initial generation stage)
- **Boundary Type (boundary_type)**:
  - `low`: Train the low-noise model, high-noise model uses pretrained weights (recommended for T2V/I2V distillation)
  - `high`: Train the high-noise model, low-noise model uses pretrained weights
  - `full`: Single model training (for single-Transformer models like TI2V-5B)

**Wan2.2 T2V Distillation LoRA Training Example**:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.2/train_distill_lora.py \
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
  --learning_rate=1e-05 \
  --learning_rate_critic=1e-06 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_distill_lora" \
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
  --boundary_type="low" \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --train_mode="normal" \
  --low_vram
```

**Wan2.2 I2V Distillation LoRA Training Example**:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-I2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.2/train_distill_lora.py \
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
  --learning_rate=1e-05 \
  --learning_rate_critic=1e-06 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_distill_lora" \
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
  --boundary_type="low" \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --train_mode="i2v" \
  --low_vram
```

### 3.3 Training Parameters Explanation

**LoRA-Specific Parameters**:

In addition to distillation training, LoRA training adds the following specific parameters:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--use_peft_lora` | Whether to use PEFT module to add LoRA, this module saves more VRAM | - |
| `--rank` | Dimension (rank) of LoRA update matrix | 64 |
| `--network_alpha` | Scaling coefficient of LoRA update matrix | 32 |
| `--target_name` | Components/modules where LoRA is applied, comma-separated | `"q,k,v,ffn.0,ffn.2"` |
| `--lora_skip_name` | Modules skipped by LoRA (not trained) | None |

**LoRA Configuration Recommendations**:
- **rank=64, network_alpha=32**: Suitable for most scenarios, balances quality and VRAM
- **rank=128, network_alpha=64**: Higher quality fine-tuning, but requires more VRAM
- **target_name="q,k,v,ffn.0,ffn.2"**: Fine-tunes attention layers and feed-forward networks, this is a common configuration
- **use_peft_lora**: Strongly recommended to enable, can significantly reduce VRAM usage

**Key Parameters Explanation**:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--pretrained_model_name_or_path` | Pretrained model path | `models/Diffusion_Transformer/Wan2.2-T2V-A14B` |
| `--train_data_dir` | Training data directory | `datasets/internal_datasets/` |
| `--train_data_meta` | Training data metadata file | `datasets/internal_datasets/metadata.json` |
| `--train_batch_size` | Number of samples per batch | 1 |
| `--image_sample_size` | Maximum training resolution for images | 640 |
| `--video_sample_size` | Maximum training resolution for videos | 640 |
| `--token_sample_size` | Token sampling size | 640 |
| `--video_sample_stride` | Video sampling stride | 2 |
| `--video_sample_n_frames` | Number of video frames sampled | 81 |
| `--gradient_accumulation_steps` | Gradient accumulation steps (effectively increases batch) | 1 |
| `--dataloader_num_workers` | Number of DataLoader subprocesses | 8 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 50 |
| `--learning_rate` | Initial learning rate (generator) | 1e-05 |
| `--learning_rate_critic` | Initial learning rate (discriminator) | 1e-06 |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir_wan2.2_distill_lora` |
| `--gradient_checkpointing` | Activation recomputation | - |
| `--mixed_precision` | Mixed precision: `fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon value | 1e-10 |
| `--vae_mini_batch` | Mini-batch size for VAE encoding | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--enable_bucket` | Enable bucket training, no cropping of images/videos, grouped by resolution | - |
| `--random_hw_adapt` | Automatically scale images/videos to random sizes within `[min_size, max_size]` range | - |
| `--training_with_video_token_length` | Train based on token length, supports any resolution | - |
| `--uniform_sampling` | Uniform timestep sampling | - |
| `--low_vram` | Low VRAM mode | - |
| `--boundary_type` | Wan2.2 dual-Transformer boundary type: `low` (train low-noise model), `high` (train high-noise model), `full` (train single model like TI2V-5B) | `low` |
| `--train_mode` | Training mode: `normal` (standard T2V) or `i2v` (image-to-video) | `normal` |
| `--resume_from_checkpoint` | Resume training path, use `"latest"` to automatically select the latest checkpoint | None |
| `--validation_steps` | Run validation every N steps | 2000 |
| `--validation_epochs` | Run validation every N epochs | 5 |
| `--validation_prompts` | Prompts for validation video generation | `"A brown dog shaking its head..."` |
| `--validation_paths` | Reference image paths for I2V validation (i2v mode only) | `"asset/1.png"` |

**Distillation-Specific Parameters**:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--denoising_step_indices_list` | Denoising step list (core distillation parameter) | `1000 750 500 250` |
| `--real_guidance_scale` | Real guidance scale for scoring | 6.0 |
| `--fake_guidance_scale` | Fake guidance scale for scoring | 0.0 |
| `--gen_update_interval` | Generator update interval | 5 |
| `--train_sampling_steps` | Training sampling steps | 1000 |

**Sample Size Configuration Guide**:
- `video_sample_size` represents the video resolution size; when `random_hw_adapt` is True, it represents the minimum resolution for videos and images.
- `image_sample_size` represents the image resolution size; when `random_hw_adapt` is True, it represents the maximum resolution for videos and images.
- `token_sample_size` represents the resolution corresponding to the maximum token length when `training_with_video_token_length` is True.
- Since configurations may cause confusion, **if you don't need arbitrary resolution fine-tuning**, it is recommended to set `video_sample_size`, `image_sample_size`, and `token_sample_size` to the same fixed value, such as **(320, 480, 512, 640, 960)**.
  - **All set to 320** represents **240P**.
  - **All set to 480** represents **320P**.
  - **All set to 640** represents **480P**.
  - **All set to 960** represents **720P**.

**Token Length Training Explanation**:
- When `training_with_video_token_length` is enabled, the model trains based on token length.
- For example: a video with 512x512 resolution and 49 frames has a token length of 13,312, requiring `token_sample_size = 512`.
  - At 512x512 resolution, video frames are 49 (~= 512 * 512 * 49 / 512 / 512).
  - At 768x768 resolution, video frames are 21 (~= 512 * 512 * 49 / 768 / 768).
  - At 1024x1024 resolution, video frames are 9 (~= 512 * 512 * 49 / 1024 / 1024).
  - These combinations of resolutions and corresponding frame numbers enable the model to generate videos of different sizes.

### 3.4 Training Validation

You can configure validation parameters to regularly generate test videos during training, allowing you to monitor training progress and model quality.

**Validation Parameters Explanation**:

| Parameter | Description | Recommended Value |
|------|------|--------|
| `--validation_steps` | Run validation every N steps | 2000 |
| `--validation_epochs` | Run validation every N epochs | 5 |
| `--validation_prompts` | Prompts for validation video generation | English prompts |
| `--validation_paths` | Reference image paths for I2V validation (i2v/inpaint mode only) | `"asset/1.png"` |

**normal Mode Example** (T2V Validation):

```bash
  --validation_steps=2000 \
  --validation_epochs=5 \
  --validation_prompts="A brown dog shaking its head, sitting on a light-colored sofa in a cozy room. Behind the dog, there's a framed painting on a shelf, surrounded by pink flowers. The soft, warm lighting in the room creates a comfortable atmosphere."
```

**i2v/inpaint Mode Example** (I2V Validation):

```bash
  --validation_paths "asset/1.png" \
  --validation_steps=2000 \
  --validation_epochs=5 \
  --validation_prompts="A brown dog shaking its head, sitting on a light-colored sofa in a cozy room. Behind the dog, there's a framed picture on a shelf surrounded by pink flowers. The soft, warm lighting in the room creates a comfortable atmosphere."
```

**Notes**:
- Validation videos will be saved to the `output_dir` directory
- Multi-prompt validation format: `--validation_prompts "prompt1" "prompt2" "prompt3"`
- `i2v` or `inpaint` mode must provide the `--validation_paths` parameter

### 3.5 Training with FSDP

If VRAM is insufficient when using multiple GPUs with DeepSpeed-Zero-2, you can switch to FSDP for training.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.2/train_distill_lora.py \
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
  --learning_rate=1e-05 \
  --learning_rate_critic=1e-06 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_distill_lora" \
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
  --boundary_type="low" \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --train_mode="normal" \
  --low_vram
```

### 3.6 Other Backends

#### 3.6.1 Training with DeepSpeed-Zero-3

DeepSpeed Zero-3 is not highly recommended at present. In this repository, using FSDP has fewer errors and is more stable.

It is known that DeepSpeed Zero-3 is incompatible with PEFT.

DeepSpeed Zero-3 is suitable for high-resolution 14B Wan models. After training, you can use the following command to obtain the final model:
```bash
python scripts/zero_to_bf16.py output_dir/checkpoint-{your-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

The training shell command is as follows:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/wan2.2/train_distill_lora.py \
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
  --learning_rate=1e-05 \
  --learning_rate_critic=1e-06 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_distill_lora" \
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
  --boundary_type="low" \
  --train_mode="normal" \
  --low_vram
```

#### 3.6.2 Training without DeepSpeed and FSDP

**This approach is not recommended because without VRAM-saving backends, it easily causes VRAM shortages**. This is only provided as a reference for training.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.2/train_distill_lora.py \
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
  --learning_rate=1e-05 \
  --learning_rate_critic=1e-06 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_distill_lora" \
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
  --boundary_type="low" \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --train_mode="normal" \
  --low_vram
```

### 3.7 Multi-Node Distributed Training

**Suitable for**: Ultra-large-scale datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines, each with 8 GPUs:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.2/train_distill_lora.py \
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
  --learning_rate=1e-05 \
  --learning_rate_critic=1e-06 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_distill_lora" \
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
  --boundary_type="low" \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --train_mode="normal" \
  --low_vram
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
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

#### 3.7.2 Multi-Node Training Notes

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

### 4.1 Inference Parameters Explanation

**Key Parameters Explanation**:

| Parameter | Description | Example Value |
|------|------|-------|
| `GPU_memory_mode` | VRAM management mode, see table below for options | `model_group_offload` |
| `ulysses_degree` | Head dimension parallelism, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelism, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP for Transformer during multi-GPU inference to save VRAM | `False` |
| `fsdp_text_encoder` | Use FSDP for text encoder during multi-GPU inference | `True` |
| `compile_dit` | Compile Transformer to accelerate inference (effective at fixed resolution) | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/Wan2.2-T2V-A14B` |
| `sampler_name` | Sampler type: `Flow`, `Flow_Unipc`, `Flow_DPM++` | `Flow_Unipc` |
| `transformer_path` | Path to load trained low-noise Transformer weights | `None` or base model weights |
| `transformer_high_path` | Path to load trained high-noise Transformer weights (dual-Transformer models only) | `None` |
| `vae_path` | Path to load trained VAE weights | `None` |
| `lora_path` | LoRA weights path for low-noise model (distillation LoRA training output) | `output_dir_wan2.2_distill_lora/checkpoint-xxx/pytorch_lora_weights.safetensors` |
| `lora_high_path` | LoRA weights path for high-noise model (dual-Transformer models only) | `None` |
| `sample_size` | Generated video resolution `[height, width]` | `[480, 832]` or `[832, 480]` |
| `video_length` | Number of generated video frames | `81` |
| `fps` | Frames per second | `16` |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs that don't support bf16 | `torch.bfloat16` |
| `validation_image_start` | Reference image path for image-to-video (I2V mode) | `"asset/1.png"` |
| `prompt` | Positive prompt, describes the content to generate | `"A brown dog shaking its head..."` |
| `negative_prompt` | Negative prompt, content to avoid | `"Low resolution, low quality..."` |
| `guidance_scale` | Guidance strength (distillation models typically use 1.0) | 1.0 |
| `seed` | Random seed, for reproducing results | 43 |
| `num_inference_steps` | Number of inference steps (distillation models typically use 4) | 4 |
| `lora_weight` | LoRA weight strength for low-noise model | 0.55 |
| `lora_high_weight` | LoRA weight strength for high-noise model (dual-Transformer models only) | 0.55 |
| `save_path` | Path to save generated videos | `samples/wan-videos-i2v` or `samples/wan-videos-t2v` |

**VRAM Management Mode Explanation**:

| Mode | Description | VRAM Usage |
|------|------|---------|
| `model_full_load` | Entire model loaded to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer groups switch between CPU/CUDA | Low |
| `sequential_cpu_offload` | Layer-by-layer offload (slowest) | Lowest |

### 4.2 Text-to-Video (T2V) Inference

Run the following command for single-GPU inference:

```bash
python examples/wan2.2/predict_t2v.py
```

Edit `examples/wan2.2/predict_t2v.py` according to your needs. For initial inference, focus on the following parameters. If you're interested in other parameters, please refer to the inference parameters explanation above.

```python
# Select based on GPU VRAM
GPU_memory_mode = "sequential_cpu_offload"
# Based on actual model path
model_name = "models/Diffusion_Transformer/Wan2.2-T2V-A14B"  
# Base model weight path (if you have trained full weights)
transformer_path = None  
# Trained high-noise weight path (if trained dual-Transformer)
transformer_high_path = None  
# LoRA weight path, e.g., "output_dir_wan2.2_distill_lora/checkpoint-xxx/pytorch_lora_weights.safetensors"
lora_path = None  
# LoRA weight path for high-noise model (if trained)
lora_high_path = None  
# Distillation models typically use 4 steps
num_inference_steps = 4
# Distillation models guidance_scale is usually 1.0
guidance_scale = 1.0
# LoRA weight strength
lora_weight = 0.55
# LoRA weight strength for high-noise model
lora_high_weight = 0.55
# Write based on generated content
prompt = "A brown dog shaking its head, sitting on a light-colored sofa in a cozy room. Behind the dog, there's a framed picture on a shelf surrounded by pink flowers. The soft, warm lighting in the room creates a comfortable atmosphere."  
# ...
```

### 4.3 Image-to-Video (I2V) Inference

Run the following command for single-GPU inference:

```bash
python examples/wan2.2/predict_i2v.py
```

Edit `examples/wan2.2/predict_i2v.py` according to your needs. For initial inference, focus on the following parameters. If you're interested in other parameters, please refer to the inference parameters explanation above.

```python
# Select based on GPU VRAM
GPU_memory_mode = "sequential_cpu_offload"
# Based on actual model path
model_name = "models/Diffusion_Transformer/Wan2.2-I2V-A14B"  
# Base model weight path (if you have trained full weights)
transformer_path = None  
# Trained high-noise weight path
transformer_high_path = None  
# LoRA weight path, e.g., "output_dir_wan2.2_distill_lora/checkpoint-xxx/pytorch_lora_weights.safetensors"
lora_path = None  
# LoRA weight path for high-noise model
lora_high_path = None  
# Distillation models typically use 4 steps
num_inference_steps = 4
# Distillation models guidance_scale is usually 1.0
guidance_scale = 1.0
# LoRA weight strength
lora_weight = 0.55
# LoRA weight strength for high-noise model
lora_high_weight = 0.55
# Starting image for image-to-video
validation_image_start = "asset/1.png"
# Write based on generated content
prompt = "A brown dog shaking its head, sitting on a light-colored sofa in a cozy room. Behind the dog, there's a framed picture on a shelf surrounded by pink flowers. The soft, warm lighting in the room creates a comfortable atmosphere."  
# ...
```

### 4.3.1 Text-Image-to-Video (TI2V) Inference

Run the following command for single-GPU inference:

```bash
python examples/wan2.2/predict_ti2v.py
```

Edit `examples/wan2.2/predict_ti2v.py` according to your needs. For initial inference, focus on the following parameters. If you're interested in other parameters, please refer to the inference parameters explanation above.

```python
# Select based on GPU VRAM
GPU_memory_mode = "sequential_cpu_offload"
# Based on actual model path (TI2V single model)
model_name = "models/Diffusion_Transformer/Wan2.2-TI2V-5B"  
# Trained weight path, e.g., "output_dir_wan2.2_distill_lora/checkpoint-xxx/pytorch_lora_weights.safetensors"
transformer_path = None  
# TI2V has only one model, transformer_high_path is not used
transformer_high_path = None  
# LoRA weight path
lora_path = None  
# LoRA weight path for high-noise model (not used for TI2V)
lora_high_path = None  
# Distillation models typically use 4 steps
num_inference_steps = 4
# Distillation models guidance_scale is usually 1.0
guidance_scale = 1.0
# LoRA weight strength
lora_weight = 0.55
# Starting image for image-to-video
validation_image_start = "asset/1.png"
# Write based on generated content
prompt = "A brown dog shaking its head, sitting on a light-colored sofa in a cozy room. Behind the dog, there's a framed picture on a shelf surrounded by pink flowers. The soft, warm lighting in the room creates a comfortable atmosphere."  
# ...
```

### 4.4 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/wan2.2/predict_t2v.py`, `examples/wan2.2/predict_i2v.py` or `examples/wan2.2/predict_ti2v.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs used
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelism
ring_degree = 1     # Sequence dimension parallelism
```

**Configuration Principles**:
- `ulysses_degree` must be divisible by the model's head count
- `ring_degree` splits along the sequence dimension, which affects communication overhead. Try to avoid using it if heads are evenly divisible

**Configuration Examples**:

| Number of GPUs | ulysses_degree | ring_degree | Description |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelism |
| 8 | 8 | 1 | Head parallelism |
| 8 | 4 | 2 | Hybrid parallelism |

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/wan2.2/predict_t2v.py
```

---

## 5. More Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
