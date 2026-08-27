# TAE (Tiny AutoEncoder) Training Guide

This document provides a complete workflow for training / fine-tuning the TAE (Tiny AutoEncoder, `AutoencoderTinyWan`) against the full Wan VAE, including environment setup, data preparation, training, and using the trained checkpoint for inference.

> **Note**: The TAE ([madebyollin/taehv](https://github.com/madebyollin/taehv)) is a ~20MB distilled VAE that shares the **exact latent space** of the full-size Wan VAEs. It decodes ~100x faster than the full VAE at slightly lower reconstruction quality, and is mainly used as a fast preview / low-memory decoder during diffusion sampling. Training is a plain reconstruction distillation:
>
> ```
> x (video, [-1, 1])
>   -> teacher (full VAE, frozen) : z_full = teacher.encode(x).mode()
>   -> TAE encoder                : z_tae  = tae.encode(x).mode()
>   -> TAE decoder                : x_hat  = tae.decode(z_tae).sample
> loss = pixel L1(x_hat, x) + latent_loss_weight * MSE(z_tae, z_full)
> ```
>
> The latent MSE anchors the TAE latents to the native VAE latent space, which is what keeps TAE latents interchangeable with the diffusion model latents.

Two TAE families are supported (selected via `--config_path`, which determines the teacher VAE):

| Family | Latent | Teacher full VAE | `--config_path` | Models |
|--------|--------|------------------|-----------------|--------|
| taew2_1 | 16ch, patch_size=1 | `AutoencoderKLWan` (Wan2.1_VAE.pth) | `config/wan2.1/wan_civitai.yaml` | Wan2.1, Wan2.2 14B |
| taew2_2 | 48ch, patch_size=2 | `AutoencoderKLWan3_8` (Wan2.2_VAE.pth) | `config/wan2.2/wan_civitai_t2v_2.2vae.yaml` | Wan2.2 TI2V-5B / Fun-2.2VAE |

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. TAE Training](#3-tae-training)
  - [3.1 Download Pretrained Model](#31-download-pretrained-model)
  - [3.2 Quick Start](#32-quick-start)
  - [3.3 Training Parameter Reference](#33-training-parameter-reference)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training Tips](#35-training-tips)
  - [3.6 Multi-Node Distributed Training](#36-multi-node-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Checkpoint Layout](#41-checkpoint-layout)
  - [4.2 Use the Trained TAE in Predict Scripts](#42-use-the-trained-tae-in-predict-scripts)
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
```

> The TAE itself only has ~20MB of weights, so **plain data parallelism is enough** — DeepSpeed / FSDP is not required (but still supported by the script). The only large model in memory is the frozen teacher full VAE (~1.5GB for the 2.2 VAE); use `--low_vram` if it does not fit together with the training activations.

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

**Relative Path Format** (example format):
```json
[
  {
    "file_path": "train/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "type": "video",
    "width": 1024,
    "height": 1024
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
- `text`: Video description (not used by the TAE loss, kept for meta format compatibility)
- `type`: Data type, should be `"video"`
- `width` / `height`: Video width and height (**recommended to provide**, used for bucket training).
  - You can use `scripts/process_json_add_width_and_height.py` to extract width and height from JSON files without these fields.

### 2.4 Relative vs Absolute Path Usage

**Relative Path**:

```bash
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
```

**Absolute Path**:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **Recommendation**: If the dataset is small and stored locally, use relative paths. If the dataset is stored on external storage (e.g., NAS, OSS) or shared across multiple machines, use absolute paths.

---

## 3. TAE Training

### 3.1 Download Pretrained Model

The training script only needs the **full VAE weights** (used as the frozen teacher), which ship inside the model directory:

```bash
mkdir -p models/Diffusion_Transformer

# taew2_2 family (Wan2.2 TI2V-5B, 48ch latent, contains Wan2.2_VAE.pth)
modelscope download --model Wan-AI/Wan2.2-TI2V-5B --local_dir models/Diffusion_Transformer/Wan2.2-TI2V-5B

# taew2_1 family (Wan2.1, 16ch latent, contains Wan2.1_VAE.pth)
# modelscope download --model Wan-AI/Wan2.1-T2V-14B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-14B
```

Optionally download the released TAE weights to warm-start instead of training from scratch:

```bash
# from https://github.com/madebyollin/taehv
wget https://github.com/madebyollin/taehv/raw/main/taew2_2.safetensors
# wget https://github.com/madebyollin/taehv/raw/main/taew2_1.safetensors
```

### 3.2 Quick Start

**Wan2.2 TI2V-5B / Fun-2.2VAE (taew2_2) Example**:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
# Optional: warm-start from released TAE weights instead of training from scratch.
export TAE_PATH="taew2_2.safetensors"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/taehv/train_taehv.py \
  --config_path="config/wan2.2/wan_civitai_t2v_2.2vae.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --tae_path=$TAE_PATH \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=33 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --latent_loss_weight=1.0 \
  --seed=42 \
  --output_dir="output_dir_taehv_w2.2" \
  --mixed_precision="bf16" \
  --adam_weight_decay=1e-2 \
  --adam_epsilon=1e-08 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --enable_bucket \
  --trainable_modules "." \
  --resume_from_checkpoint=latest
```

**Wan2.1 / Wan2.2 14B (taew2_1) Example**:

Same as above, with these changes:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-14B"
export TAE_PATH="taew2_1.safetensors"
# ...
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --output_dir="output_dir_taehv_w2.1" \
# ...
```

### 3.3 Training Parameter Reference

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config_path` | Model config yaml; its `vae_kwargs.vae_type` selects the teacher full VAE family | `config/wan2.2/wan_civitai_t2v_2.2vae.yaml` |
| `--pretrained_model_name_or_path` | Model directory containing the full VAE weights | `models/Diffusion_Transformer/Wan2.2-TI2V-5B` |
| `--tae_path` | Optional TAE weights to warm-start from (file / directory). Omit to train from scratch | `taew2_2.safetensors` |
| `--tae_arch_variant` | TAE decoder variant when training from scratch: base (`None`) or `super` (~2x decoder params) | `None` |
| `--vae_path` | Optional hot-load path for other full VAE weights (teacher) | `None` |
| `--freeze_tae_encoder` | Train the TAE decoder only (decoder-only distillation) | - |
| `--use_taehv_sequential` | Run the TAE in O(1)-memory sequential mode instead of parallel | - |
| `--latent_loss_weight` | Weight of the latent MSE (TAE latent vs. full VAE latent) relative to pixel L1 | 1.0 |
| `--train_data_dir` | Training data directory | `datasets/X-Fun-Videos-Demo/` |
| `--train_data_meta` | Training data metadata file | `datasets/X-Fun-Videos-Demo/metadata.json` |
| `--train_batch_size` | Batch size (per device) | 1 |
| `--video_sample_size` | Training resolution | 512 |
| `--video_sample_stride` | Video sample stride | 1 |
| `--video_sample_n_frames` | Number of frames to sample. **Must be 4k+1** (33, 49, 81, ...) | 33 |
| `--vae_mini_batch` | Mini batch size for teacher VAE encoding | 1 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 1 |
| `--dataloader_num_workers` | Number of DataLoader workers | 4 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save a checkpoint every N steps | 500 |
| `--checkpoints_total_limit` | Max number of checkpoints to store | `None` |
| `--learning_rate` | Initial learning rate | 1e-4 |
| `--lr_scheduler` | Learning rate scheduler | `constant_with_warmup` |
| `--lr_warmup_steps` | Learning rate warmup steps | 100 |
| `--use_8bit_adam` / `--use_came` | Alternative optimizers | - |
| `--use_ema` | Keep an EMA copy of the TAE (used for validation and final save) | - |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir_taehv_w2.2` |
| `--mixed_precision` | Mixed precision: `fp16/bf16` | `bf16` |
| `--max_grad_norm` | Maximum gradient norm for clipping | 1.0 |
| `--enable_bucket` | Enable bucket training without cropping, groups by resolution | - |
| `--random_hw_adapt` | Randomly scale videos to a range of resolutions | - |
| `--low_vram` | Keep the teacher VAE on CPU and move it to GPU only when encoding | - |
| `--trainable_modules` | Trainable modules (`"."` means all modules) | `"."` |
| `--trainable_modules_low_learning_rate` | Trainable modules with lr/2 | `[]` |
| `--resume_from_checkpoint` | Resume training from checkpoint, use `"latest"` to auto-select | `latest` |
| `--validation_steps` / `--validation_epochs` | Run validation every N steps / epochs | 2000 / 5 |
| `--validation_paths` | Video paths for validation reconstruction comparison | `"asset/1.mp4"` |

**Sample Size Configuration Guide**:
- `video_sample_size` represents the training resolution; when `random_hw_adapt` is enabled, it represents the minimum resolution.
- `video_sample_n_frames` must satisfy `4k+1` (e.g. 33, 49, 81) because both the full VAE and the TAE are causal 4x temporal compressors.

### 3.4 Training Validation

You can configure validation parameters to periodically reconstruct test videos with both the TAE and the full VAE during training, so you can visually monitor reconstruction quality.

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--validation_steps` | Run validation every N steps | 2000 |
| `--validation_epochs` | Run validation every N epochs | 5 |
| `--validation_paths` | Validation video paths | `"asset/1.mp4"` |

```bash
  --validation_paths "asset/1.mp4" \
  --validation_steps=2000 \
  --validation_epochs=5
```

**Notes**:
- Validation saves three videos per sample into `output_dir/sample/`: `*_input.mp4` (resized input), `*_taehv.mp4` (TAE reconstruction), `*_fullvae.mp4` (full VAE reconstruction for reference).
- When `--use_ema` is enabled, validation runs with the EMA weights.

### 3.5 Training Tips

- **Warm-start vs. from scratch**: The released TAE weights are already well distilled; fine-tuning from `taew2_x.safetensors` with a small lr (1e-5) is usually enough for domain adaptation. Training from scratch converges but requires much more data/steps.
- **Loss balance**: `latent_loss_weight=1.0` keeps the TAE latent aligned with the diffusion latent space. Set it to `0.0` for pure pixel reconstruction (not recommended if you use the TAE latents in the sampling loop).
- **Decoder-only distillation**: Add `--freeze_tae_encoder` to only improve decoding quality.
- **Memory**: The teacher full VAE is the main memory consumer. Use `--low_vram` to keep it on CPU between encoding steps, or reduce `--video_sample_n_frames` / `--video_sample_size`.
- **Sequential TAE**: `--use_taehv_sequential` trades speed for O(1) activation memory w.r.t. video length.
- **EMA**: `--use_ema` is recommended for the final deliverable; the final saved `taehv` directory contains the EMA weights.

### 3.6 Multi-Node Distributed Training

**Suitable for**: Large-scale datasets, faster training speed

Assuming 2 machines with 8 GPUs each:

**Machine 0 (Master)**:
```bash
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Current machine rank (0 or 1)
# Without RDMA:
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK scripts/taehv/train_taehv.py \
  <same training arguments as the Quick Start>
```

**Machine 1 (Worker)**: use the same command with `export RANK=1`.

**Notes**:
- Without RDMA, add `NCCL_IB_DISABLE=1` and `NCCL_P2P_DISABLE=1`.
- All machines must have access to the same data / model paths (NFS/shared storage).

---

## 4. Inference Testing

### 4.1 Checkpoint Layout

Each checkpoint is written as `output_dir/checkpoint-{step}/`, containing:

```
📦 output_dir_taehv_w2.2/
├── 📂 checkpoint-500/
│   ├── 📂 taehv/            # TAE weights + config.json (save_pretrained format)
│   ├── 📂 taehv_ema/        # only when --use_ema
│   └── 📄 sampler_pos_start.pkl
├── 📂 sample/               # validation videos
└── 📂 logs/                 # tensorboard
```

The `taehv` subdirectory is a standard diffusers directory checkpoint and can be loaded directly by `AutoencoderTinyWan.from_pretrained`.

### 4.2 Use the Trained TAE in Predict Scripts

Point `tae_path` of any TAE predict script to the `taehv` subdirectory of your checkpoint:

| Script | Family |
|--------|--------|
| `examples/wan2.2/predict_ti2v_tae.py` | taew2_2 |
| `examples/wan2.2_fun/predict_t2v_2.2vae_tae.py` | taew2_2 |
| `examples/wan2.2_fun/predict_i2v_2.2vae_tae.py` | taew2_2 |
| `examples/wan2.1/predict_t2v_tae.py` | taew2_1 |
| `examples/wan2.1/predict_i2v_tae.py` | taew2_1 |

```python
# e.g. in examples/wan2.2_fun/predict_t2v_2.2vae_tae.py
tae_path = "output_dir_taehv_w2.2/checkpoint-500/taehv"
```

The trained TAE is interchangeable with the released one: it keeps the same latent space as the full VAE, so it can be used for fast preview decoding in the diffusion pipeline exactly like `taew2_2.safetensors`.

---

## 5. Additional Resources

- **TAE reference implementation**: https://github.com/madebyollin/taehv
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
