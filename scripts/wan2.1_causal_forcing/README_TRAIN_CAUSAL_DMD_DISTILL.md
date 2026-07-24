# Wan2.1 Causal-Forcing Stage 3: Distribution Matching Distillation (DMD) Training Guide

This document provides the complete workflow for **Causal-Forcing Stage 3 — Distribution Matching Distillation (DMD)** on Wan2.1-T2V-1.3B.

> **What is Distribution Matching Distillation?**
>
> DMD is the **third and final stage** of the Causal-Forcing pipeline, which further compresses a CCD model into a **few-step causal autoregressive generator** using a large (14B) teacher:
>
> 1. **Stage 1 — AR Diffusion** (`train_ar_diffusion.py`): Train the model with teacher forcing on clean video latents, producing a strong AR backbone.
> 2. **Stage 2 — Causal Consistency Distillation** (`train_causal_consistency_distill.py`): Distill the multi-step AR model into a **one-step-per-block** consistency model using an EMA teacher with CFG guidance.
> 3. **Stage 3 — Distribution Matching Distillation** (`train_causal_dmd.py`): Further distill to a **few-step** generator using distribution matching with a **14B real-score teacher**.
>
> This README covers **Stage 3 only**. See [README_TRAIN_CAUSAL_CONSISTENCY_DISTILL.md](./README_TRAIN_CAUSAL_CONSISTENCY_DISTILL.md) for Stage 2 and [README_TRAIN_AR_DIFFUSION.md](./README_TRAIN_AR_DIFFUSION.md) for Stage 1.

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
  - [5.3 DMD-Specific Parameters](#53-dmd-specific-parameters)
- [6. Use the Trained Checkpoint](#6-use-the-trained-checkpoint)
- [7. Additional Resources](#7-additional-resources)

---

## 1. Prerequisites

Stage 3 requires:

1. A **Stage 2 CCD checkpoint** to initialize the generator (and critic).
2. A **Wan2.1-T2V-14B** model as the DMD real-score teacher.

```bash
# Example: Stage 2 checkpoint from CCD training
export STAGE2_CKPT="output_dir_wan2.1_causal_forcing_ccd/checkpoint-5000/transformer/diffusion_pytorch_model.safetensors"
```

See [README_TRAIN_CAUSAL_CONSISTENCY_DISTILL.md](./README_TRAIN_CAUSAL_CONSISTENCY_DISTILL.md) for how to produce this checkpoint.

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

DMD requires **two** pretrained models:

- **Wan2.1-T2V-1.3B**: base model for the generator/critic.
- **Wan2.1-T2V-14B**: the non-causal real-score teacher used by DMD to compute the real distribution score.

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download Wan2.1 T2V 1.3B (student base model)
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B

# Download Wan2.1 T2V 14B (DMD real-score teacher)
modelscope download --model Wan-AI/Wan2.1-T2V-14B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-14B
```

The Stage 2 CCD checkpoint (generator/critic init) is loaded via `--ode_transformer_path`.

---

## 4. Prepare Training Data

DMD uses a **TextDataset** (`train_mode="normal"`) — it only needs prompts, not video data, because the generator creates its own training samples via autoregressive rollout. However, a `metadata.json` with prompts is still required.

### 4.1 Quick Demo Dataset

```bash
# Download official demo dataset
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 4.2 Dataset Structure

Same as Stage 1/2. See [README_TRAIN_CAUSAL_CONSISTENCY_DISTILL.md](./README_TRAIN_CAUSAL_CONSISTENCY_DISTILL.md#42-dataset-structure) for details.

### 4.3 metadata.json Format

DMD only uses the `"text"` field from each entry. The `file_path` and other fields are ignored when `--train_mode="normal"` (TextDataset mode).

```json
[
  {
    "text": "A beautiful sunset over the ocean, golden hour lighting"
  },
  {
    "text": "A person walking through a forest, cinematic view"
  }
]
```

> **Note**: You can reuse any video metadata.json — only the `text` field matters for DMD.

---

## 5. Training

### 5.1 Quick Start

The ready-to-use launcher is [train_causal_dmd.sh](./train_causal_dmd.sh):

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export REAL_SCORE_MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export STAGE2_CKPT="output_dir_wan2.1_causal_forcing_ccd/checkpoint-5000/transformer/diffusion_pytorch_model.safetensors"

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/wan2.1_causal_forcing/train_causal_dmd.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --real_score_pretrained_model_name_or_path=$REAL_SCORE_MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --ode_transformer_path=$STAGE2_CKPT \
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
  --learning_rate_critic=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_causal_forcing_dmd" \
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
  --use_kv_cache_training \
  --denoising_step_indices_list 1000 667 334 1 \
  --real_guidance_scale=6.0 \
  --randomize_step_indices \
  --fake_guidance_scale=0.0 \
  --gen_update_interval=5 \
  --trainable_modules "."
```

Or simply:

```bash
bash scripts/wan2.1_causal_forcing/train_causal_dmd.sh
```

### 5.2 Key Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--pretrained_model_name_or_path` | Base model (1.3B) for generator/critic init | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B/` |
| `--real_score_pretrained_model_name_or_path` | 14B real-score teacher for DMD | `models/Diffusion_Transformer/Wan2.1-T2V-14B` |
| `--config_path` | Model config YAML | `config/wan2.1/wan_civitai.yaml` |
| `--train_data_dir` | Data root directory | `""` |
| `--train_data_meta` | Path to `metadata.json` (prompts only) | `datasets/X-Fun-Videos-Demo/metadata.json` |
| `--ode_transformer_path` | Stage 2 CCD checkpoint for generator/critic init | `$STAGE2_CKPT` |
| `--train_batch_size` | Per-GPU batch size | 1 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 1 |
| `--dataloader_num_workers` | DataLoader workers | 8 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 200 |
| `--learning_rate` | Generator learning rate | 2e-06 |
| `--learning_rate_critic` | Critic learning rate | 2e-06 |
| `--lr_scheduler` | LR scheduler type | `constant_with_warmup` |
| `--lr_warmup_steps` | LR warmup steps | 100 |
| `--output_dir` | Output directory | `output_dir_wan2.1_causal_forcing_dmd` |
| `--gradient_checkpointing` | Enable activation checkpointing | - |
| `--mixed_precision` | `fp16` / `bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 0.0 |
| `--adam_beta1` | AdamW beta1 (DMD uses 0.0) | 0.0 |
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

### 5.3 DMD-Specific Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--denoising_step_indices_list` | Denoising step indices (DMD core param). Example runs 4-step; parser default `[1000, 500]` = 2-step | `1000 667 334 1` |
| `--real_guidance_scale` | CFG scale for the real-score (14B teacher) | 6.0 |
| `--randomize_step_indices` | Randomize the denoising step indices during training | - |
| `--fake_guidance_scale` | CFG scale for the fake-score (generator). 0.0 = no CFG | 0.0 |
| `--gen_update_interval` | Generator update interval (generator updates every N critic steps) | 5 |
| `--num_frame_per_block` | Frames per causal block. `3` = chunkwise (default), `1` = frame-wise | 3 |
| `--use_kv_cache_training` | Use KV cache block-by-block training (matches original Self-Forcing) | - |
| `--independent_first_frame` | First frame is independent (`[1, N, N, ...]` block pattern, useful for I2V) | - |
| `--context_noise` | Context noise level for KV cache update | 0 |
| `--use_teacher_forcing` | Enable teacher forcing (pass clean_x to transformer) | - |
| `--teacher_forcing_prob` | Probability of applying teacher forcing per step (1.0 = always) | 1.0 |
| `--train_mode` | Training mode: `normal` (TextDataset, prompt-only) or `i2v` | `normal` |
| `--resume_from_checkpoint` | Resume from checkpoint. Use `"latest"` to auto-select | `"latest"` |

---

## 6. Use the Trained Checkpoint

The Stage 3 DMD checkpoint is the **final model** in the Causal-Forcing pipeline. Use it for inference:

```python
# In examples/wan2.1_causal_forcing/predict_t2v.py
transformer_path = "output_dir_wan2.1_causal_forcing_dmd/checkpoint-{N}/diffusion_pytorch_model.safetensors"

# DMD Stage 3 inference config
guidance_scale      = 1.0    # CFG is baked into distilled weights
num_inference_steps = 4      # 4-step DMD
stochastic_sampling = True
num_frame_per_block = 3      # Chunk-wise generation
```

Or run:

```bash
python examples/wan2.1_causal_forcing/predict_t2v.py
```

> **Stage Selector** in `predict_t2v.py` provides preset configs for all stages:
> - **Stage 1 (AR Diffusion)**: `guidance_scale=3.0`, `num_inference_steps=50`, `stochastic_sampling=False`
> - **Stage 2 (CCD)**: `guidance_scale=1.0`, `num_inference_steps=4`, `stochastic_sampling=True`
> - **Stage 3 (DMD)**: `guidance_scale=1.0`, `num_inference_steps=4`, `stochastic_sampling=True`

---

## 7. Additional Resources

- **Causal-Forcing Paper**: https://github.com/thu-ml/Causal-Forcing
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
