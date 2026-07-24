# Wan2.1 Causal-Forcing Stage 1: Autoregressive Diffusion Training Guide

This document provides the complete workflow for **Causal-Forcing Stage 1 — Autoregressive Diffusion (AR Diffusion) training** on Wan2.1-T2V-1.3B.

> **What is Causal-Forcing?**
>
> Causal-Forcing is a three-stage pipeline that compresses a video diffusion model into a **few-step causal autoregressive generator**:
>
> 1. **Stage 1 — AR Diffusion** (`train_ar_diffusion.py`): Train the model with **teacher forcing** on clean video latents, learning to denoise chunk-by-chunk (or frame-by-frame) in a causal autoregressive manner. This produces a strong AR backbone.
> 2. **Stage 2 — Causal Consistency Distillation** (`train_causal_consistency_distill.py`): Distill the multi-step AR model into a one-step-per-block consistency model.
> 3. **Stage 3 — Causal DMD** (`train_causal_dmd.py`): Further distill to a **2-step** generator using distribution matching with a 14B teacher.
>
> This README covers **Stage 1 only**.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Download Pretrained Models](#2-download-pretrained-models)
- [3. Prepare Training Data](#3-prepare-training-data)
  - [3.1 Quick Demo Dataset](#31-quick-demo-dataset)
  - [3.2 Dataset Structure](#32-dataset-structure)
  - [3.3 metadata.json Format](#33-metadatajson-format)
- [4. Training](#4-training)
  - [4.1 Quick Start](#41-quick-start)
  - [4.2 Key Parameters](#42-key-parameters)
  - [4.3 Causal-Forcing-Specific Parameters](#43-causal-forcing-specific-parameters)
  - [4.4 Training with FSDP](#44-training-with-fsdp)
- [5. Use the Trained Checkpoint](#5-use-the-trained-checkpoint)
- [6. Additional Resources](#6-additional-resources)

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
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**Method 3: Using Docker**

```bash
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Download Pretrained Models

Stage 1 initializes the Wan2.1-T2V-1.3B model and trains it with causal teacher forcing.

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download Wan2.1 T2V base model
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B
```

---

## 3. Prepare Training Data

Stage 1 trains directly on **raw videos** with online VAE encoding — no ODE trajectory pairs are needed.

### 3.1 Quick Demo Dataset

We provide a small demo dataset for quick testing:

```bash
# Download official demo dataset
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 3.2 Dataset Structure

```
📦 datasets/
├── 📂 X-Fun-Videos-Demo/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 3.3 metadata.json Format

**Relative paths** (recommended):
```json
[
  {
    "file_path": "train/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "type": "video"
  },
  {
    "file_path": "train/video002.mp4",
    "text": "A person walking through a forest, cinematic view",
    "type": "video"
  }
]
```

**Absolute paths**:
```json
[
  {
    "file_path": "/mnt/data/videos/sunset.mp4",
    "text": "A beautiful sunset over the ocean",
    "type": "video"
  }
]
```

---

## 4. Training

### 4.1 Quick Start

The ready-to-use launcher is [train_ar_diffusion.sh](./train_ar_diffusion.sh):

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/wan2.1_causal_forcing/train_ar_diffusion.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --fix_sample_size 480 832 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=200 \
  --learning_rate=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_causal_forcing_ar_diffusion" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --num_frame_per_block=3 \
  --train_sampling_steps=1000 \
  --shift=5.0 \
  --use_timestep_weight \
  --trainable_modules "."
```

Or simply:

```bash
bash scripts/wan2.1_causal_forcing/train_ar_diffusion.sh
```

### 4.2 Key Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--pretrained_model_name_or_path` | Base model to initialize | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B/` |
| `--config_path` | Model config YAML | `config/wan2.1/wan_civitai.yaml` |
| `--train_data_dir` | Video root directory (prepended to relative `file_path`) | `""` |
| `--train_data_meta` | Path to `metadata.json` | `datasets/X-Fun-Videos-Demo/metadata.json` |
| `--train_batch_size` | Per-GPU batch size | 1 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 1 |
| `--dataloader_num_workers` | DataLoader workers | 8 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 200 |
| `--learning_rate` | Initial learning rate | 2e-06 |
| `--lr_scheduler` | LR scheduler type | `constant_with_warmup` |
| `--lr_warmup_steps` | LR warmup steps | 100 |
| `--output_dir` | Output directory | `output_dir_wan2.1_causal_forcing_ar_diffusion` |
| `--gradient_checkpointing` | Enable activation checkpointing | - |
| `--mixed_precision` | `fp16` / `bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon | 1e-10 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--trainable_modules` | Trainable modules (`"."` = all) | `"."` |

**Video Sampling Parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--image_sample_size` | Image sampling size | 640 |
| `--video_sample_size` | Video sampling size | 640 |
| `--token_sample_size` | Token sampling size | 640 |
| `--fix_sample_size` | Fixed `[height, width]` for output | `480 832` |
| `--video_sample_stride` | Frame sampling stride | 2 |
| `--video_sample_n_frames` | Number of video frames | 81 |
| `--random_hw_adapt` | Enable random resolution adaptation | - |
| `--training_with_video_token_length` | Enable token-length-based training | - |
| `--enable_bucket` | Enable aspect-ratio bucket sampling | - |
| `--vae_mini_batch` | VAE encoding mini-batch size (1 to avoid OOM) | 1 |

### 4.3 Causal-Forcing-Specific Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--num_frame_per_block` | Frames per causal block. `3` = chunkwise, `1` = framewise | 3 |
| `--independent_first_frame` | First frame is independent (`[1, N, N, ...]` block pattern, useful for I2V) | - |
| `--shift` | `FlowMatchEulerDiscreteScheduler` shift (Causal-Forcing default: 5.0) | 5.0 |
| `--train_sampling_steps` | Total scheduler timesteps for flow matching | 1000 |
| `--use_timestep_weight` | Apply per-timestep Gaussian loss weight (centered at T/2) | - |
| `--no_teacher_forcing` | Disable teacher forcing (use diffusion forcing instead) | - |
| `--noise_augmentation_max_timestep` | Add light noise to clean context tokens during teacher forcing (0 = off) | 0 |

> **Teacher Forcing**: By default, Stage 1 uses teacher forcing — the model receives the **clean GT latent** as context (`clean_x`) when predicting the current block. This stabilizes early training. Disable with `--no_teacher_forcing` to train under diffusion forcing instead.

### 4.4 Training with FSDP

The script above already uses FSDP with `CasualWanAttentionBlock` auto-wrapping. For single-GPU training without FSDP:

```bash
accelerate launch --mixed_precision="bf16" \
    scripts/wan2.1_causal_forcing/train_ar_diffusion.py \
    ...
```

---

## 5. Use the Trained Checkpoint

The Stage 1 checkpoint is used to initialize **Stage 2 (Causal Consistency Distillation)**. Pass its path to `train_causal_consistency_distill.py` via `--transformer_path` and `--teacher_transformer_path`:

```bash
--transformer_path="output_dir_wan2.1_causal_forcing_ar_diffusion/checkpoint-{N}/diffusion_pytorch_model.safetensors" \
--teacher_transformer_path="output_dir_wan2.1_causal_forcing_ar_diffusion/checkpoint-{N}/diffusion_pytorch_model.safetensors"
```

See [train_causal_consistency_distill.sh](./train_causal_consistency_distill.sh) for the full Stage 2 workflow.

---

## 6. Additional Resources

- **Causal-Forcing Paper**: https://github.com/thu-ml/Causal-Forcing
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
