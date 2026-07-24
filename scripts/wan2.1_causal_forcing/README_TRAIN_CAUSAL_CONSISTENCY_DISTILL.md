# Wan2.1 Causal-Forcing Stage 2: Causal Consistency Distillation (CCD) Training Guide

This document provides the complete workflow for **Causal-Forcing Stage 2 — Causal Consistency Distillation (CCD)** on Wan2.1-T2V-1.3B.

> **What is Causal Consistency Distillation?**
>
> CCD is the **second stage** of the Causal-Forcing pipeline, which compresses a video diffusion model into a **few-step causal autoregressive generator**:
>
> 1. **Stage 1 — AR Diffusion** (`train_ar_diffusion.py`): Train the model with teacher forcing on clean video latents, producing a strong AR backbone.
> 2. **Stage 2 — Causal Consistency Distillation** (`train_causal_consistency_distill.py`): Distill the multi-step AR model into a **one-step-per-block** consistency model using an EMA teacher with CFG guidance.
> 3. **Stage 3 — Causal DMD** (`train_causal_dmd.py`): Further distill to a **2-step** generator using distribution matching with a 14B teacher.
>
> This README covers **Stage 2 only**. See [README_TRAIN_AR_DIFFUSION.md](./README_TRAIN_AR_DIFFUSION.md) for Stage 1.

---

## Table of Contents
- [1. Prerequisites](#1-prerequisites)
- [2. Environment Setup](#2-environment-setup)
- [3. Download Pretrained Models](#3-download-pretrained-models)
- [4. Prepare Training Data](#4-prepare-training-data)
  - [4.1 Quick Demo Dataset](#41-quick-demo-dataset)
  - [4.2 Dataset Structure](#42-dataset-structure)
  - [4.3 metadata.json Format](#43-metadatajson-format)
- [5. Training](#5-training)
  - [5.1 Quick Start](#51-quick-start)
  - [5.2 Key Parameters](#52-key-parameters)
  - [5.3 CCD-Specific Parameters](#53-ccd-specific-parameters)
- [6. Use the Trained Checkpoint](#6-use-the-trained-checkpoint)
- [7. Additional Resources](#7-additional-resources)

---

## 1. Prerequisites

Stage 2 requires a **Stage 1 AR Diffusion checkpoint** to initialize both the generator and the EMA teacher.

```bash
# Example: Stage 1 checkpoint from AR Diffusion training
export STAGE1_CKPT="output_dir_wan2.1_causal_forcing_ar_diffusion/checkpoint-8000/diffusion_pytorch_model.safetensors"
```

See [README_TRAIN_AR_DIFFUSION.md](./README_TRAIN_AR_DIFFUSION.md) for how to produce this checkpoint.

---

## 2. Environment Setup

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

## 3. Download Pretrained Models

CCD initializes the generator and teacher from Wan2.1-T2V-1.3B, then loads the Stage 1 checkpoint on top.

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download Wan2.1 T2V base model
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B
```

---

## 4. Prepare Training Data

CCD trains directly on **raw videos** with online VAE encoding — same data format as Stage 1.

### 4.1 Quick Demo Dataset

We provide a small demo dataset for quick testing:

```bash
# Download official demo dataset
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 4.2 Dataset Structure

```
📦 datasets/
├── 📂 X-Fun-Videos-Demo/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 4.3 metadata.json Format

**Relative paths** (recommended):
```json
[
  {
    "file_path": "train/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
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

## 5. Training

### 5.1 Quick Start

The ready-to-use launcher is [train_causal_consistency_distill.sh](./train_causal_consistency_distill.sh):

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export STAGE1_CKPT="output_dir_wan2.1_causal_forcing_ar_diffusion/checkpoint-8000/diffusion_pytorch_model.safetensors"

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/wan2.1_causal_forcing/train_causal_consistency_distill.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --transformer_path=$STAGE1_CKPT \
  --teacher_transformer_path=$STAGE1_CKPT \
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
  --learning_rate=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_causal_forcing_ccd" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=0.0 \
  --adam_beta1=0.0 \
  --adam_beta2=0.999 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=10.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --num_frame_per_block=3 \
  --shift=5.0 \
  --discrete_cd_N=48 \
  --guidance_scale=3.0 \
  --ema_weight=0.99 \
  --ema_start_step=200 \
  --trainable_modules "."
```

Or simply:

```bash
bash scripts/wan2.1_causal_forcing/train_causal_consistency_distill.sh
```

### 5.2 Key Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--pretrained_model_name_or_path` | Base model to initialize | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B/` |
| `--config_path` | Model config YAML | `config/wan2.1/wan_civitai.yaml` |
| `--train_data_dir` | Video root directory (prepended to relative `file_path`) | `""` |
| `--train_data_meta` | Path to `metadata.json` | `datasets/X-Fun-Videos-Demo/metadata.json` |
| `--transformer_path` | Stage 1 checkpoint for generator init | `$STAGE1_CKPT` |
| `--teacher_transformer_path` | Stage 1 checkpoint for teacher init (defaults to generator) | `$STAGE1_CKPT` |
| `--train_batch_size` | Per-GPU batch size | 1 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 1 |
| `--dataloader_num_workers` | DataLoader workers | 8 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 200 |
| `--learning_rate` | Initial learning rate | 2e-06 |
| `--lr_scheduler` | LR scheduler type | `constant_with_warmup` |
| `--lr_warmup_steps` | LR warmup steps | 100 |
| `--output_dir` | Output directory | `output_dir_wan2.1_causal_forcing_ccd` |
| `--gradient_checkpointing` | Enable activation checkpointing | - |
| `--mixed_precision` | `fp16` / `bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 0.0 |
| `--adam_beta1` | AdamW beta1 (CCD uses 0.0) | 0.0 |
| `--adam_beta2` | AdamW beta2 | 0.999 |
| `--adam_epsilon` | AdamW epsilon | 1e-10 |
| `--max_grad_norm` | Gradient clipping threshold | 10.0 |
| `--trainable_modules` | Trainable modules (`"."` = all) | `"."` |
| `--low_vram` | Enable low VRAM mode (offload VAE/text encoder) | - |

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

### 5.3 CCD-Specific Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--num_frame_per_block` | Frames per causal block. `3` = chunkwise, `1` = framewise | 3 |
| `--independent_first_frame` | First frame is independent (`[1, N, N, ...]` block pattern, useful for I2V) | - |
| `--shift` | `FlowMatchEulerDiscreteScheduler` shift (Causal-Forcing default: 5.0) | 5.0 |
| `--discrete_cd_N` | Number of discrete timesteps for the consistency schedule | 48 |
| `--guidance_scale` | CFG guidance scale used by the EMA teacher | 3.0 |
| `--ema_weight` | EMA decay for the consistency-target generator copy (set <=0 to disable) | 0.99 |
| `--ema_start_step` | Steps to wait before EMA tracking starts | 200 |

> **How CCD works**: For each training sample, CCD:
> 1. Loads clean video latents (via online VAE encoding).
> 2. Samples a timestep index from the discrete consistency schedule `[0, N-2]`.
> 3. Adds noise to the clean latents at timestep `t` and `t_next`.
> 4. Runs the **generator** on `x_t` to predict `x0`.
> 5. Runs the **EMA teacher** (with CFG guidance) on `x_{t_next}` to produce the consistency target.
> 6. Minimizes the L2 loss between the generator prediction and the teacher target.

> **EMA Teacher**: The EMA copy tracks the generator with polyak updates (`ema = decay*ema + (1-decay)*gen`). Before `--ema_start_step`, the EMA mirrors the live generator. The teacher uses CFG with `--guidance_scale=3.0` to produce higher-quality targets.

---

## 6. Use the Trained Checkpoint

The Stage 2 CCD checkpoint is used to initialize **Stage 3 (Causal DMD)**. Pass its path to `train_causal_dmd.py`:

```bash
--ode_transformer_path="output_dir_wan2.1_causal_forcing_ccd/checkpoint-{N}/diffusion_pytorch_model.safetensors"
```

See `train_causal_dmd.sh` for the full Stage 3 workflow.

---

## 7. Additional Resources

- **Causal-Forcing Paper**: https://github.com/thu-ml/Causal-Forcing
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
