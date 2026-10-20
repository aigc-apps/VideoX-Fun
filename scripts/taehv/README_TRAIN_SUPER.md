# TAE Super (Tiny AutoEncoder) Training Guide

This document provides a complete workflow for training / fine-tuning the Wan2.2
Super decoder (`AutoencoderTinyWan`), including environment setup, data
preparation, training, and using the trained checkpoint for inference.

> **Note**: Training uses the frozen official Wan2.2 encoder and updates only the
> Super decoder's approximately 19M parameters. The tiny encoder is frozen and is
> not used in the training forward pass.
>
> ```text
> x (video, [-1, 1])
>   -> official Wan2.2 encoder (frozen) : z = vae.encode(x).latent_dist.mode()
>   -> Super decoder                  : x_hat = tae(z).sample
> loss = 0.1 L1 + 0.1 LPIPS + 0.02 DINO + 1.0 grid
> ```
>
> The losses compare reconstruction and input after mapping RGB to [0,1]. The
> decoder learns to consume the official Wan latent space directly.

The script supports the Wan2.2 5B Super model:

| Family | Latent | Official full VAE | `--config_path` | Models |
|--------|--------|-------------------|-----------------|--------|
| taew2_2_super | 48ch, patch_size=2 | `AutoencoderKLWan3_8` | `config/wan2.2/wan_civitai_5b.yaml` | Wan2.2 TI2V-5B / Fun-2.2VAE |

**Differences from [train_taehv.py](train_taehv.py)**:

| Item | `train_taehv.py` | `train_taehv_super.py` |
|------|-----------------|-----------------------|
| Reconstruction path | Tiny encoder → tiny decoder; official encoder supplies latent targets | Frozen official encoder → Super decoder |
| Trainable parameters | Tiny encoder and decoder by default, controlled by `--trainable_modules` | Super decoder only; official and tiny encoders are frozen |
| Losses | Pixel L1 + latent MSE | Pixel L1 + LPIPS + DINO + grid |

---

## Table of Contents

- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. TAE Super Training](#3-tae-super-training)
  - [3.1 Download Pretrained Model](#31-download-pretrained-model)
  - [3.2 Quick Start](#32-quick-start)
  - [3.3 Training Parameter Reference](#33-training-parameter-reference)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training Tips](#35-training-tips)
  - [3.6 Further Tuning](#36-further-tuning)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Checkpoint Layout](#41-checkpoint-layout)
  - [4.2 Use the Trained TAE in Predict Scripts](#42-use-the-trained-tae-in-predict-scripts)
- [5. Additional Resources](#5-additional-resources)

---

## 1. Environment Setup

Run the commands from the VideoX-Fun repository root. Install dependencies into
your Python environment with pip.

**Option 1: Using requirements.txt**

```bash
pip install -r requirements.txt
pip install lpips torchvision av modelscope opencv-python
```

**Option 2: Manual Installation**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install lpips torchvision av modelscope opencv-python
```

LPIPS's AlexNet backbone and DINOv2 download automatically when missing and reuse
the PyTorch cache. LPIPS's calibration weights ship with the package. For offline
DINOv2 loading, use `--dino_repo` and `--dino_weights`.

> The script supports a single device and ordinary DDP through Accelerate. The
> official Wan VAE and the frozen LPIPS/DINOv2 feature networks remain on the GPU.

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset containing several training samples.

```bash
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
- `width` / `height`: Video width and height (recommended metadata). The Super
  training entry uses a fixed resize/crop canvas.

### 2.4 Relative vs Absolute Path Usage

**Relative Path**:

```bash
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
```

**Absolute Path**:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **Recommendation**: If the dataset is small and stored locally, use relative paths. If the dataset is stored on external storage (e.g., NAS, OSS) or shared across multiple machines, use absolute paths.

---

## 3. TAE Super Training

### 3.1 Download Pretrained Model

The training script needs the official Wan2.2 VAE weights, which ship inside the
model directory. Download the released Super weights for fine-tuning:

```bash
mkdir -p models/Diffusion_Transformer

modelscope download --model Wan-AI/Wan2.2-TI2V-5B --local_dir models/Diffusion_Transformer/Wan2.2-TI2V-5B

wget -O models/Diffusion_Transformer/taew2_2_super.pth \
  https://github.com/madebyollin/taehv/raw/main/taew2_2_super.pth
```

LPIPS and DINOv2 weights are loaded separately by the loss implementation.

### 3.2 Quick Start

**Wan2.2 TI2V-5B / Fun-2.2VAE (taew2_2_super) Example**:

The launcher uses a 720x1280 canvas, 25 training frames, 81 validation frames,
10,000 updates, and EMA. Set the GPU/process count with `accelerate config`;
four GPUs with batch size 1 and accumulation 1 give an effective batch of 4.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="${DATASET_NAME:-datasets/X-Fun-Videos-Demo}"
export DATASET_META_NAME="${DATASET_META_NAME:-$DATASET_NAME/metadata_add_width_height.json}"
export TAE_PATH="models/Diffusion_Transformer/taew2_2_super.pth"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/taehv/train_taehv_super.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --tae_path="$TAE_PATH" \
  --train_data_dir="$DATASET_NAME" \
  --train_data_meta="$DATASET_META_NAME" \
  --fix_sample_size 720 1280 \
  --video_sample_strides 1 2 3 \
  --video_sample_n_frames=25 \
  --validation_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --max_train_steps=10000 \
  --checkpointing_steps=1000 \
  --validation_steps=1000 \
  --validation_max_samples=0 \
  --learning_rate=5e-06 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=200 \
  --lr_min_ratio=0.2 \
  --l1_loss_weight=0.1 \
  --lpips_loss_weight=0.1 \
  --dino_loss_weight=0.02 \
  --grid_loss_weight=1.0 \
  --grid_period=16 \
  --feature_batch_size=13 \
  --use_ema \
  --ema_decay=0.999 \
  --selection_lpips_ratio=1.02 \
  --selection_grid_ratio=1.05 \
  --seed=42 \
  --output_dir="output_dir_taehv_w2.2_super" \
  --mixed_precision="bf16" \
  --adam_beta1=0.9 \
  --adam_beta2=0.99 \
  --adam_weight_decay=1e-2 \
  --adam_epsilon=1e-08 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0
```

These settings are also provided in `scripts/taehv/train_taehv_super.sh`. Edit its
training arguments and run `bash scripts/taehv/train_taehv_super.sh`.

**Train from scratch**:

```bash
accelerate launch --mixed_precision="bf16" scripts/taehv/train_taehv_super.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$DATASET_NAME" \
  --train_data_meta="$DATASET_META_NAME" \
  --from_scratch \
  --fix_sample_size 720 1280 \
  --video_sample_n_frames=25 \
  --video_sample_strides 1 2 3 \
  --max_train_steps=10000 \
  --learning_rate=5e-6 --lr_warmup_steps=200 --lr_min_ratio=0.2 \
  --use_ema --ema_decay=0.999 \
  --dataloader_num_workers=8 --checkpointing_steps=1000 --validation_steps=1000 \
  --output_dir="output_dir_taehv_w2.2_super_scratch"
```

Omit `--tae_path` when using `--from_scratch`. Only the decoder is trained; the
saved tiny encoder remains randomly initialized in a scratch run.

**Resume interrupted training**:

```bash
accelerate launch --mixed_precision="bf16" scripts/taehv/train_taehv_super.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$DATASET_NAME" \
  --train_data_meta="$DATASET_META_NAME" \
  --fix_sample_size 720 1280 --video_sample_n_frames=25 \
  --video_sample_strides 1 2 3 --max_train_steps=10000 \
  --learning_rate=5e-6 --lr_warmup_steps=200 --lr_min_ratio=0.2 \
  --use_ema --ema_decay=0.999 \
  --dataloader_num_workers=8 --checkpointing_steps=1000 --validation_steps=1000 \
  --output_dir="output_dir_taehv_w2.2_super" \
  --resume_from_checkpoint=latest
```

Restore with the same settings used to create the checkpoint, including validation
metadata when supplied. `--max_train_steps` is the total update budget. Keep the
original schedule horizon. The checkpoint restores optimizer, scheduler, EMA, RNG,
step and batch cursor; changing workers or sampling settings can change the resumed
sample sequence. Use a new output directory for a new run.

### 3.3 Training Parameter Reference

Values below describe the launcher configuration; they are not all Python CLI defaults.

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config_path` | Official Wan VAE component configuration | `config/wan2.2/wan_civitai_5b.yaml` |
| `--pretrained_model_name_or_path` | Model directory containing the official VAE weights | `models/Diffusion_Transformer/Wan2.2-TI2V-5B` |
| `--tae_path` | Super weights, file or Diffusers directory; omit to initialize randomly | `models/Diffusion_Transformer/taew2_2_super.pth` |
| `--from_scratch` | Randomly initialize the Super decoder; excludes weight initialization and resume | - |
| `--train_data_dir` / `--train_data_meta` | Video root and JSON metadata array | `datasets/X-Fun-Videos-Demo` |
| `--fix_sample_size` | Fixed `[height, width]`, overriding square `--video_sample_size` | `720 1280` |
| `--video_sample_strides` | Choose one sampling stride per training clip | `1 2 3` |
| `--video_sample_n_frames` | Training clip length, must be 4k+1 | 25 |
| `--train_batch_size` / `--gradient_accumulation_steps` | Batch per device / gradient accumulation | 1 / 1 |
| `--vae_mini_batch` | Official VAE encoding mini-batch size | 1 |
| `--dataloader_num_workers` | DataLoader workers | 8 |
| `--max_train_steps` | Total optimizer updates; overrides `--num_train_epochs` | 10000 |
| `--checkpointing_steps` / `--checkpoints_total_limit` | Save interval / checkpoint count limit | 1000 / `None` |
| `--learning_rate` / `--lr_scheduler` | Peak learning rate / scheduler | `5e-6` / `cosine` |
| `--lr_warmup_steps` / `--lr_min_ratio` | Warmup updates / final-to-peak ratio for cosine decay | 200 / 0.2 |
| `--l1_loss_weight` / `--lpips_loss_weight` | Pixel L1 / LPIPS loss weights | 0.1 / 0.1 |
| `--dino_loss_weight` / `--grid_loss_weight` | DINO / periodic grid loss weights | 0.02 / 1.0 |
| `--feature_batch_size` / `--feature_checkpointing` | Perceptual frame batch / activation checkpointing | 13 / off |
| `--use_ema` / `--ema_decay` | Maintain decoder EMA for validation and selection | enabled / 0.999 |
| `--seed` / `--mixed_precision` | Random seed / mixed precision | 42 / `bf16` |
| `--output_dir` | Output directory | `output_dir_taehv_w2.2_super` |
| `--resume_from_checkpoint` | Restore a checkpoint directory or select the latest | `latest` |

**Sample Size Configuration Guide**:

- `--fix_sample_size 720 1280` resizes to cover the canvas and center crops while preserving aspect ratio. Without it, `--video_sample_size` selects a square canvas.
- A 25-frame clip at stride 3 needs at least 73 source frames; 81-frame validation needs at least 81. Short inputs shorten the batch to the shortest 4k+1 clip length, recorded as `frames` in the training log.
- Training uses random temporal starts. Validation starts at frame 0 with stride 1. Data loading retains the original random replacement/retry behavior on decode failures.

**Loss Configuration Guide**:

- The frozen official encoder supplies latents; only the Super decoder is optimized.
- LPIPS uses AlexNet; DINOv2 uses intermediate ViT-B/14 patch features. Frozen feature networks still pass reconstruction gradients to the decoder.
- Grid loss measures residual means by spatial phase with each frame/channel's DC removed. Inspect local crops alongside this metric.

### 3.4 Training Validation

Provide a separate JSON array through `--validation_data_meta` in the training
command, or add that argument to the launcher. Relative video paths resolve under
`--train_data_dir`, so use absolute video paths when validation data has another root.

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--validation_data_meta` | Held-out video metadata array | Your validation index |
| `--validation_steps` | Evaluate every N optimizer updates | 1000 |
| `--validation_n_frames` | Validation clip length, 4k+1 | 81 |
| `--validation_max_samples` | Validation clip cap; zero uses the complete index | 0 |

Validation runs at initialization, periodically, and at the final step. It reports
PSNR, SSIM, LPIPS and grid, and saves the first sample's input/reconstruction preview
as `sample/step-N.mp4`. Metrics use floating-point output before video encoding,
averaging frames within a clip and then clips equally.

With EMA enabled, validation and best-weight selection use EMA; otherwise they
use raw weights. Selection maximizes PSNR subject to LPIPS/grid ratios relative
to the first validation, controlled by `--selection_lpips_ratio` and
`--selection_grid_ratio`. The initialization is also a candidate. Best inference
weights go to `best/taehv`; resumable checkpoints retain raw training weights.

For a short execution check, use the official demo:

```bash
accelerate launch --mixed_precision="bf16" scripts/taehv/train_taehv_super.py \
  --pretrained_model_name_or_path="$MODEL_NAME" --tae_path="$TAE_PATH" \
  --train_data_dir="datasets/X-Fun-Videos-Demo" \
  --train_data_meta="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json" \
  --validation_data_meta="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json" \
  --fix_sample_size 64 96 --video_sample_n_frames=5 --validation_n_frames=5 \
  --video_sample_strides 1 --max_train_steps=2 --lr_warmup_steps=0 \
  --checkpointing_steps=1 --validation_steps=1 --validation_max_samples=1 \
  --dataloader_num_workers=0 --use_ema --ema_decay=0.999 \
  --output_dir="output_dir_taehv_w2.2_super_smoke"
```

Sharing the demo across training/validation checks execution only. Use independent
video sources for actual experiments, and use native-resolution data to assess
720p quality.

### 3.5 Training Tips

- **Warm-start vs. from scratch**: Fine-tune from `taew2_2_super.pth` or a previous Super directory; use `--from_scratch` for a new decoder. Initialization, dataset and budget all affect the result.
- **Loss balance**: Start with the four-loss settings in the launcher and inspect both reconstruction metrics and native-resolution crops.
- **Memory**: Lower the training frame count, canvas size or feature batch size, or enable `--feature_checkpointing` if needed. Perceptual networks are frozen but their prediction branches still need activation memory.
- **EMA**: The launcher enables EMA. `best/taehv` holds selected inference weights; `checkpoint-N/taehv` holds raw weights for training recovery.
- **Distributed training**: Use Accelerate's single-device or ordinary DDP configuration. Global batch equals device count times per-device batch times accumulation.

### 3.6 Further Tuning

- Try longer temporal context and check both the initial frames and later frames, accounting for the extra compute.
- Explore pixel MSE or official-decoder output supervision, balancing pixel accuracy, perceptual quality and teacher compute cost.
- Adjust LPIPS/grid weights with controlled comparisons, checking fine textures, text and local grid artifacts alongside average metrics.

These are possible extensions; the current training entry uses the four-loss recipe above.

---

## 4. Inference Testing

### 4.1 Checkpoint Layout

Each resumable checkpoint is written as `output_dir/checkpoint-{step}/`:

```text
output_dir_taehv_w2.2_super/
  args-step-N.yaml          # resolved invocation record
  train.jsonl              # total loss, four terms, LR and actual frame count
  validation.jsonl         # metrics and selection decisions, when validation is enabled
  best/taehv/              # selected inference weights, when validation is enabled
  checkpoint-1000/
    taehv/                 # raw Super weights and config.json
    optimizer.bin
    custom_checkpoint_*.pkl # scheduler and optional EMA state
    random_states_*.pkl
    trainer_state.json
  sample/                  # validation videos
  logs/                    # TensorBoard logs
```

The `taehv` directories use Diffusers `save_pretrained` format and load through
`AutoencoderTinyWan.from_pretrained`. `args-step-N.yaml` records the invocation;
training arguments are configured directly in the shell launcher.

### 4.2 Use the Trained TAE in Predict Scripts

Point `tae_path` in a compatible Wan2.2 5B TAE predict script to the selected
`best/taehv` directory:

| Script | Family |
|--------|--------|
| `examples/wan2.2/predict_ti2v_tae.py` | taew2_2 |
| `examples/wan2.2_fun/predict_t2v_2.2vae_tae.py` | taew2_2 |
| `examples/wan2.2_fun/predict_i2v_2.2vae_tae.py` | taew2_2 |

```python
# e.g. in examples/wan2.2_fun/predict_t2v_2.2vae_tae.py
tae_path = "output_dir_taehv_w2.2_super/best/taehv"
```

The trained decoder consumes official Wan2.2 5B latents. A scratch-trained
checkpoint has no trained tiny encoder; retain the official Wan encoder for
image/video conditioning when using such a checkpoint.
