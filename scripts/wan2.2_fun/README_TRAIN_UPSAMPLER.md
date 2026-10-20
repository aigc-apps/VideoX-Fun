# Wan Latent Upsampler Training Guide

This document provides a complete workflow for training the Wan Latent Upsampler (`WanLatentUpsamplerModel`) against the Wan2.2 VAE, including environment setup, data preparation, training, and using the trained checkpoint for inference.

> **Note**: The Wan Latent Upsampler is a lightweight 3D-convolution model that **spatially upsamples Wan2.2 VAE latents** (default `1.5x`, controlled by `rational_spatial_scale`) before VAE decoding. It lets the diffusion model generate at a lower latent resolution and then upsample the latents to a higher resolution, reducing the compute cost of high-resolution video generation. Unlike diffusion training, the upsampler is trained with **pure supervised MSE regression on paired low/high-resolution latents**:
>
> ```
> x_hr (high-res video, [-1, 1])
>   x_lr   = spatial_downsample(x_hr, scale)   # scale = rational_spatial_scale (1.5)
>   z_hr   = vae.encode(x_hr).mode()           # frozen VAE -> target latent
>   z_lr   = vae.encode(x_lr).mode()           # frozen VAE -> input latent
>   z_pred = upsampler(z_lr)
> loss = MSE(z_pred, z_hr)                     # latent-space MSE (default)
> ```
>
> The VAE is **frozen**; only the upsampler is trained. The low/high-resolution pairs are created on-the-fly from ordinary videos, so no pre-paired data is needed. Optional switches add degradation, flow-matching noise, or a pixel-space loss (see [3.5](#35-advanced-training-options)).

The upsampler targets the Wan2.2 2.2VAE latent space (48 channels, `16x` spatial compression). The `--config_path` must contain a `latent_upsampler_kwargs` block, which fixes the upsampler architecture and spatial scale:

| Latent | Full VAE | `--config_path` | Spatial scale | Used at inference by |
|--------|----------|-----------------|---------------|----------------------|
| 48ch | `AutoencoderKLWan3_8` (Wan2.2_VAE.pth) | `config/wan2.2/wan_civitai_t2v_2.2vae.yaml` | 1.5 | `Wan2.2-Fun-*-A14B-2.2VAE` predict scripts |

> `config/wan2.2/wan_civitai_i2v_2.2vae.yaml` shares the same `latent_upsampler_kwargs` and can be used interchangeably for training.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. Latent Upsampler Training](#3-latent-upsampler-training)
  - [3.1 Download Pretrained Model](#31-download-pretrained-model)
  - [3.2 Quick Start](#32-quick-start)
  - [3.3 Training Parameter Reference](#33-training-parameter-reference)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Advanced Training Options](#35-advanced-training-options)
  - [3.6 Training Tips](#36-training-tips)
  - [3.7 Multi-Node Distributed Training](#37-multi-node-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Checkpoint Layout](#41-checkpoint-layout)
  - [4.2 Use the Trained Upsampler in Predict Scripts](#42-use-the-trained-upsampler-in-predict-scripts)
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

> The upsampler is a lightweight convolutional model (far smaller than the diffusion transformer), so **plain data parallelism is enough** — DeepSpeed / FSDP is not required (but still supported by the script). The only large model in memory is the frozen Wan2.2 VAE; use `--low_vram` to keep it on CPU between encode/decode steps if it does not fit together with the training activations.

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
- `text`: Video description (not used by the upsampler loss, kept for meta format compatibility)
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

## 3. Latent Upsampler Training

### 3.1 Download Pretrained Model

The training script only needs the **Wan2.2 VAE weights** (used as the frozen encoder/decoder), which ship inside the model directory:

```bash
mkdir -p models/Diffusion_Transformer

# Wan2.2 TI2V-5B (48ch latent, contains Wan2.2_VAE.pth) — only the frozen VAE is used for upsampler training
modelscope download --model Wan-AI/Wan2.2-TI2V-5B --local_dir models/Diffusion_Transformer/Wan2.2-TI2V-5B
```

> At inference the upsampler is used with the `Wan2.2-Fun-*-A14B-2.2VAE` models. They share the **same Wan2.2 VAE latent space**, so an upsampler trained against the `Wan2.2-TI2V-5B` VAE is directly usable there. Any model directory that contains `Wan2.2_VAE.pth` works as `--pretrained_model_name_or_path`.

### 3.2 Quick Start

After downloading the dataset as in **2.1** and the pretrained model as in **3.1**, copy and run the quick start command directly.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.2_fun/train_upsampler.py \
  --config_path="config/wan2.2/wan_civitai_t2v_2.2vae.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_fun_upsampler" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --enable_bucket \
  --low_vram \
  --trainable_modules "."
```

> These hyper-parameters follow the standard Wan2.2 Fun video training settings. `scripts/wan2.2_fun/train_upsampler.sh` contains the same command.

### 3.3 Training Parameter Reference

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config_path` | Model config yaml; must contain `latent_upsampler_kwargs` (fixes the upsampler architecture and spatial scale) | `config/wan2.2/wan_civitai_t2v_2.2vae.yaml` |
| `--pretrained_model_name_or_path` | Model directory containing `Wan2.2_VAE.pth` (only the frozen VAE is used) | `models/Diffusion_Transformer/Wan2.2-TI2V-5B` |
| `--latent_upsampler_path` | Optional upsampler weights to warm-start / resume from (file or directory) | `None` |
| `--vae_path` | Optional path to other full VAE weights | `None` |
| `--train_data_dir` | Training data directory | `datasets/X-Fun-Videos-Demo/` |
| `--train_data_meta` | Training data metadata file | `datasets/X-Fun-Videos-Demo/metadata.json` |
| `--train_batch_size` | Batch size (per device) | 1 |
| `--video_sample_size` | Target **high-resolution** size; the low-res input is `high_res / rational_spatial_scale` | 640 |
| `--video_sample_stride` | Video sample stride | 2 |
| `--video_sample_n_frames` | Number of frames to sample. **Must be 4k+1** (33, 49, 81, ...) | 81 |
| `--vae_mini_batch` | Mini batch size for VAE encoding | 1 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 1 |
| `--dataloader_num_workers` | Number of DataLoader workers | 8 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save a checkpoint every N steps | 50 |
| `--checkpoints_total_limit` | Max number of checkpoints to store | `None` |
| `--learning_rate` | Initial learning rate | 2e-05 |
| `--lr_scheduler` | Learning rate scheduler | `constant_with_warmup` |
| `--lr_warmup_steps` | Learning rate warmup steps | 100 |
| `--use_8bit_adam` / `--use_came` | Alternative optimizers | - |
| `--use_ema` | Keep an EMA copy of the upsampler (used for validation and final save) | - |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir_wan2.2_fun_upsampler` |
| `--gradient_checkpointing` | Enable activation recompute for the upsampler (and the VAE decoder when `--enable_pixel_loss`) | - |
| `--mixed_precision` | Mixed precision: `fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon value | 1e-10 |
| `--max_grad_norm` | Maximum gradient norm for clipping | 0.05 |
| `--enable_bucket` | Enable bucket training without cropping, groups by resolution | - |
| `--random_hw_adapt` | Randomly scale videos to a range of resolutions | - |
| `--low_vram` | Keep the frozen VAE on CPU and move it to GPU only when encoding/decoding | - |
| `--trainable_modules` | Trainable modules (`"."` means all modules) | `"."` |
| `--trainable_modules_low_learning_rate` | Trainable modules with lr/2 | `[]` |
| `--resume_from_checkpoint` | Resume training from checkpoint, use `"latest"` to auto-select | `None` |
| `--validation_steps` / `--validation_epochs` | Run validation every N steps / epochs | 2000 / 5 |
| `--validation_paths` | Video paths for validation (encode low-res -> upsample -> decode) | `"asset/inpaint_video.mp4"` |

**Sample Size Configuration Guide**:
- `video_sample_size` is the **high-resolution target**. When `random_hw_adapt` is enabled, it represents the minimum resolution and the video may be scaled up to a larger bucket size.
- The low-resolution input is derived automatically: `low_res = high_res / rational_spatial_scale`, aligned to the VAE spatial compression ratio (`16`), so both resolutions are VAE-encodable.
- `video_sample_n_frames` must satisfy `4k+1` (e.g. 33, 49, 81) because the Wan2.2 VAE is a causal `4x` temporal compressor.

### 3.4 Training Validation

You can configure validation parameters to periodically run **encode low-res -> upsample -> decode** on test videos during training, so you can visually monitor super-resolution quality.

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--validation_steps` | Run validation every N steps | 2000 |
| `--validation_epochs` | Run validation every N epochs | 5 |
| `--validation_paths` | Validation video paths | `"asset/inpaint_video.mp4"` |

```bash
  --validation_paths "asset/inpaint_video.mp4" \
  --validation_steps=2000 \
  --validation_epochs=5
```

**Notes**:
- Each validation video is resized to a square target derived from `--video_sample_size`, then encoded, downsampled to the low-res input, upsampled by the model, and decoded.
- Validation saves two videos per sample into `output_dir/sample/`: `step{N}_val{i}_upsampled.mp4` (upsampler result) and `step{N}_val{i}_lowres.mp4` (decoded low-res input, for comparison).
- When `--use_ema` is enabled, validation runs with the EMA weights.

### 3.5 Advanced Training Options

These switches are off by default; the plain latent-MSE regression in **3.2** is the recommended starting point.

**Degradation (restoration + super-resolution)**:

Apply random degradation to the high-res frames **before** the spatial downsample, so the upsampler learns to restore as well as upsample.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--enable_degradation` | Enable random degradation of the high-res frames | off |
| `--degradation_ops_range MIN MAX` | Range of degradation operation count per clip | `1 4` |

**Noisy training (match inference denoising steps)**:

At inference the upsampler may run on partially-denoised latents. Noisy training adds flow-matching noise to the latents so the upsampler works at arbitrary denoising steps.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--enable_noisy_training` | Add flow-matching noise to the latents during training | off |
| `--noise_sample_mode` | `scheduler` samples sigma from the real schedule; `uniform` samples in `[sigma_min, sigma_max]` | `scheduler` |
| `--noise_num_inference_steps` | Inference steps simulated when sampling sigma (scheduler mode) | 50 |
| `--noise_step_range MIN MAX` | Restrict sampling to a sub-range of step indices (scheduler mode) | `None` |
| `--noise_sigma_max` / `--noise_sigma_min` | Sigma range for `uniform` mode | 0.25 / 0.0 |

**Pixel-space loss**:

Decode the predicted latents through the frozen VAE decoder and compute an MSE in pixel space; the gradient flows through the frozen decoder back to the upsampler.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--enable_pixel_loss` | Enable pixel-space MSE loss (replaces the latent-space MSE) | off |
| `--pixel_loss_weight` | Weight of the pixel-space loss | 1.0 |

> `--enable_pixel_loss` keeps the VAE on GPU (it is not offloaded under `--low_vram`) and needs the VAE decoder gradients, so it uses more memory. Combine it with `--gradient_checkpointing` to enable VAE-decoder activation recompute.

### 3.6 Training Tips

- **Default loss**: latent-space MSE against the frozen VAE encoding of the high-res frames. The VAE is never updated.
- **Memory**: memory scales with `video_sample_size` and `video_sample_n_frames` because the frozen VAE encodes/decodes full-resolution video. If you hit OOM, keep `--low_vram` + `--gradient_checkpointing` and reduce `--video_sample_size` or `--video_sample_n_frames`.
- **Warm start**: `--latent_upsampler_path` loads existing upsampler weights. Keys whose shape mismatches the current architecture (e.g. after changing `rational_spatial_scale`) are skipped and re-initialized automatically.
- **EMA**: `--use_ema` keeps an EMA copy used for validation and the final save. It is not supported with DeepSpeed ZeRO-3.
- **Spatial scale**: fixed by `latent_upsampler_kwargs.rational_spatial_scale` in the config; it **must match** the inference config, otherwise the upsampler output resolution will be wrong.

### 3.7 Multi-Node Distributed Training

**Suitable for**: Large-scale datasets, faster training speed.

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

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK scripts/wan2.2_fun/train_upsampler.py \
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
📦 output_dir_wan2.2_fun_upsampler/
├── 📂 checkpoint-50/
│   ├── 📂 latent_upsampler/       # upsampler weights + config.json (save_pretrained format)
│   ├── 📂 latent_upsampler_ema/   # only when --use_ema
│   └── 📄 sampler_pos_start.pkl
├── 📂 sample/                     # validation videos
└── 📂 logs/                       # tensorboard
```

The `latent_upsampler` subdirectory is a standard diffusers directory checkpoint and can be loaded directly by `WanLatentUpsamplerModel.from_pretrained`.

> With DeepSpeed ZeRO-3 or FSDP `FULL_SHARD`, the upsampler weights are saved as a single `diffusion_pytorch_model.safetensors` inside the checkpoint directory instead of the `latent_upsampler/` subfolder.

### 4.2 Use the Trained Upsampler in Predict Scripts

The Wan2.2 2.2VAE predict scripts support latent upsampling. Set `enable_latent_upsample = True` and point `latent_upsampler_path` to the `latent_upsampler` subdirectory of your checkpoint:

| Script | Task |
|--------|------|
| `examples/wan2.2_fun/predict_t2v_2.2vae.py` | Text-to-Video |
| `examples/wan2.2_fun/predict_i2v_2.2vae.py` | Image-to-Video |
| `examples/wan2.2_fun/predict_t2v_2.2vae_tae.py` | Text-to-Video + TAE fast decode |
| `examples/wan2.2_fun/predict_i2v_2.2vae_tae.py` | Image-to-Video + TAE fast decode |

```python
# e.g. in examples/wan2.2_fun/predict_t2v_2.2vae_tae.py
enable_latent_upsample = True
latent_upsampler_path  = "output_dir_wan2.2_fun_upsampler/checkpoint-50/latent_upsampler"
```

If `latent_upsampler_path` is `None`, the script falls back to the `latent_upsampler` subfolder of `model_name`. During inference the pipeline generates latents at the base resolution and the upsampler spatially upsamples them (`1.5x`) before VAE decoding.

---

## 5. Additional Resources

- **Wan2.2 Fun Full Parameter Training Guide**: [README_TRAIN.md](README_TRAIN.md)
- **TAE (Tiny AutoEncoder) Training Guide**: [../taehv/README_TRAIN.md](../taehv/README_TRAIN.md)
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
