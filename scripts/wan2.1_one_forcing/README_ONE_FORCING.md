# Wan2.1 One-Forcing: Stable One-Step Autoregressive Video Generation

This document provides the complete workflow for **One-Forcing** on Wan2.1-T2V-1.3B, a recipe ported from [Aurora-edu/One-Forcing](https://github.com/Aurora-edu/One-Forcing).

> **What is One-Forcing?**
>
> One-Forcing (*Towards Stable One-Step Autoregressive Video Generation*) is **Causal-Forcing Stage 3 — Distribution Matching Distillation (DMD)** *plus* a **shared noised-latent adversarial critic**. The warm-start generator is produced by the first two Causal-Forcing stages, and One-Forcing then polishes it into a **stable one-step, frame-wise autoregressive** video model:
>
> 1. **Stage 1 — AR Diffusion** (`scripts/wan2.1_causal_forcing/train_ar_diffusion.py`): Train the model with **teacher forcing** on clean video latents, learning to denoise chunk-by-chunk (or frame-by-frame) in a causal autoregressive manner. This produces a strong AR backbone. See [README_TRAIN_AR_DIFFUSION.md](../wan2.1_causal_forcing/README_TRAIN_AR_DIFFUSION.md).
> 2. **Stage 2 — Causal Consistency Distillation (CCD)** (`scripts/wan2.1_causal_forcing/train_causal_consistency_distill.py`): Distill the multi-step AR model into a **one-step-per-block** consistency model using an EMA teacher with CFG guidance. Its output checkpoint is the **warm-start generator** for One-Forcing. See [README_TRAIN_CAUSAL_CONSISTENCY_DISTILL.md](../wan2.1_causal_forcing/README_TRAIN_CAUSAL_CONSISTENCY_DISTILL.md).
> 3. **Stage 3 — One-Forcing** (`train_one_forcing.py`, this document), which adds two mechanisms on top of DMD:
>    - **DMD core**: the generator matches the real distribution through a 14B real-score teacher and a 1.3B fake-score critic (`train_causal_dmd.py`).
>    - **Adversarial critic**: a discriminator head rides on the **fake-score backbone** (one register token per selected feature layer). Both the DMD denoising loss and the adversarial loss operate on the **same noised generated latent**, and the critic uses **one shared noise/timestep** for fake and real samples.
>    - **One step**: `denoising_step_list = [1000]` — each frame is generated in a single denoising step (the "One" in One-Forcing).
>
> Instead of training Stage 1/2 yourself, you can also warm-start from the **official ODE-initialized checkpoint** released by the One-Forcing authors (see Section 3).
>
> **The trainer** `scripts/wan2.1_one_forcing/train_one_forcing.py` implements the One-Forcing generator/critic GAN losses, the relativistic option, and finite-difference R1/R2 penalties. For the underlying DMD stage, see [README_TRAIN_CAUSAL_DMD_DISTILL.md](../wan2.1_causal_forcing/README_TRAIN_CAUSAL_DMD_DISTILL.md).

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
  - [5.2 Recipe Parameters](#52-recipe-parameters)
  - [5.3 GAN Prerequisites](#53-gan-prerequisites)
  - [5.4 Optional Regularizers (Relativistic / R1 / R2)](#54-optional-regularizers-relativistic--r1--r2)
- [6. Inference](#6-inference)
  - [6.1 Quick Start](#61-quick-start)
  - [6.2 1-Step vs 4-Step Selector](#62-1-step-vs-4-step-selector)
  - [6.3 First-Frame Schedule](#63-first-frame-schedule)
- [7. Checkpoint Round-Trip](#7-checkpoint-round-trip)
- [8. Additional Resources](#8-additional-resources)

---

## 1. Prerequisites

One-Forcing is the final polish on top of a distilled generator, so it needs:

1. A **warm-start generator checkpoint** — a distilled/ODE generator that the DMD + adversarial critic then polishes (e.g. a **Causal-Forcing Stage-2 CCD** checkpoint, or the official ODE-initialized checkpoint in Section 3).
2. A **Wan2.1-T2V-14B** model as the DMD real-score teacher.
3. **Captioned real videos** — the adversarial critic needs clean real latents, so a prompt-only JSON is **not** sufficient (unlike plain DMD).

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

One-Forcing requires **two** pretrained models:

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

The official One-Forcing checkpoints are hosted on HuggingFace (`JiaqiFeng/OneForcing`):

```bash
# Warm-start generator for training (the official ODE-initialized checkpoint),
# loaded via --ode_transformer_path.
hf download JiaqiFeng/OneForcing checkpoints/framewise/causal_ode.pt --local-dir models/Diffusion_Transformer/One-Forcing

# Trained One-Forcing generator for inference (see Section 6).
hf download JiaqiFeng/OneForcing checkpoints/one_forcing.pt --local-dir models/Diffusion_Transformer/One-Forcing
```

If you trained Causal-Forcing Stage 1/2 yourself, you can instead point `--ode_transformer_path` at your own Stage-2 CCD checkpoint.

---

## 4. Prepare Training Data

One-Forcing enables the GAN branch by default (`--gan_g_weight=0.03 --gan_d_weight=0.03`), so the data loader always uses an **ImageVideoDataset**: the adversarial critic needs **captioned real videos** to build its clean latents. A prompt-only JSON is **not** sufficient (unlike plain DMD).

### 4.1 Quick Demo Dataset

```bash
# Download official demo dataset
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 4.2 Dataset Structure

Same as Causal-Forcing Stage 1/2. See [README_TRAIN_CAUSAL_CONSISTENCY_DISTILL.md](../wan2.1_causal_forcing/README_TRAIN_CAUSAL_CONSISTENCY_DISTILL.md#42-dataset-structure) for details.

### 4.3 metadata.json Format

Each entry needs `file_path`, `type="video"`, and `text`; the launcher uses the `metadata_add_width_height.json` variant, which additionally carries `width`/`height` so bucketing does not need to decode the media files:

```json
[
  {
    "file_path": "train/00000000.mp4",
    "text": "A young woman gently turns her head to the right, tilting it slightly as she offers a soft greeting...",
    "type": "video",
    "width": 832,
    "height": 480
  }
]
```

> **Note**: Keep enough frames per video for `--video_sample_stride=1` and `--video_sample_n_frames=81`: the GAN branch aborts when the source yields fewer latent frames than `score_num_frames=21` (see Section 5.3).

---

## 5. Training

### 5.1 Quick Start

The ready-to-use launcher is [train_one_forcing.sh](./train_one_forcing.sh). It calls the One-Forcing trainer (`train_one_forcing.py`) with the GAN recipe enabled:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export REAL_SCORE_MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export STAGE2_CKPT="models/Diffusion_Transformer/One-Forcing/checkpoints/framewise/causal_ode.pt"

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/wan2.1_one_forcing/train_one_forcing.py \
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
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --max_train_steps=200 \
  --checkpointing_steps=100 \
  --learning_rate=1.0e-05 \
  --learning_rate_critic=1.0e-05 \
  --lr_scheduler="constant" \
  --seed=42 \
  --output_dir="output_dir_wan2.1_one_forcing_framewise_1step" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --fsdp_reduce_fp32 \
  --adam_weight_decay=0.01 \
  --adam_beta1=0.0 \
  --adam_beta2=0.999 \
  --adam_epsilon=1e-08 \
  --vae_mini_batch=1 \
  --max_grad_norm=10.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --num_frame_per_block=1 \
  --use_kv_cache_training \
  --use_ema \
  --ema_decay=0.99 \
  --denoising_step_indices_list 1000 \
  --real_guidance_scale=6.0 \
  --score_timestep_clamp \
  --fake_guidance_scale=0.0 \
  --gen_update_interval=5 \
  --gan_g_weight=0.03 \
  --gan_d_weight=0.03 \
  --gan_feature_layers 21 29 \
  --gan_block_ffn_dim=2048 \
  --gan_block_num_heads=12 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "."
```

Or simply:

```bash
bash scripts/wan2.1_one_forcing/train_one_forcing.sh
```

### 5.2 Recipe Parameters

The One-Forcing recipe sets these launcher flags:

| Parameter | Launcher flag | Value |
|------------------------|---------------|-------|
| `denoising_step_list: [1000]` | `--denoising_step_indices_list` | `1000` (one step) |
| `num_frame_per_block: 1` | `--num_frame_per_block` | `1` (frame-wise) |
| `num_training_frames: 21` / `score_num_frames: 21` | `--video_sample_n_frames=81` @ 480×832 | 21 latent frames |
| `guidance_scale: 5.0` | `--real_guidance_scale` | `6.0` (see CFG note below) |
| `lr: 1e-5` / `lr_critic: 1e-5` | `--learning_rate` / `--learning_rate_critic` | `1.0e-05` |
| `beta1: 0.0`, `beta2: 0.999`, `weight_decay: 0.01` | `--adam_beta1`, `--adam_beta2`, `--adam_weight_decay` | `0.0`, `0.999`, `0.01` |
| `ema_weight: 0.99` | `--use_ema`, `--ema_decay` | on, `0.99` |
| `max_steps: 200` | `--max_train_steps` | `200` |
| `dfake_gen_update_ratio: 5` | `--gen_update_interval` | `5` |
| `gan_feature_layers: [21, 29]` | `--gan_feature_layers` | `21 29` |
| `gan_block_ffn_dim: 2048` | `--gan_block_ffn_dim` | `2048` |
| `gan_block_num_heads: 12` | `--gan_block_num_heads` | `12` |
| `gan_g_weight: 0.03`, `gan_d_weight: 0.03` | `--gan_g_weight`, `--gan_d_weight` | `0.03`, `0.03` |
| `relativistic_discriminator: false` | `--relativistic_discriminator` | omitted (off) |
| `r1_weight: 0.0`, `r2_weight: 0.0` | `--r1_weight`, `--r2_weight` | omitted (off) |
| `context_noise: 0` | `--context_noise` | `0` (default) |

> **CFG convention**: this repo computes guidance as `cond + scale * (cond - uncond)`, and this recipe uses `--real_guidance_scale=6.0`.

For the general training/video-sampling parameters shared with the DMD stage, see [README_TRAIN_CAUSAL_DMD_DISTILL.md](../wan2.1_causal_forcing/README_TRAIN_CAUSAL_DMD_DISTILL.md#52-key-parameters) Section 5.2.

### 5.3 GAN Prerequisites

The GAN branch is validated at startup in `train_one_forcing.py`. It requires:

- **Captioned real videos** (`type="video"` + `file_path` + `text`), not a prompt-only JSON. Enabling any positive GAN/R1/R2 weight switches the loader from `TextDataset` to `ImageVideoDataset`.
- **`--gradient_accumulation_steps=1`** and **no DeepSpeed** — the trainer raises `ValueError` otherwise.
- **`--video_sample_stride=1`**: with `stride=2`, an 81-frame source yields ~10 latent frames, which is below `score_num_frames=21` and aborts.
- **Valid feature-layer indices** for the fake-score backbone. The defaults `21 29` (layers) and `12` (heads) fit Wan2.1-T2V-1.3B; the head count must divide the fake-score hidden dimension.

Trainable fake-score and head master weights stay **FP32**; the forward pass uses the configured autocast/FSDP precision, and logits/softplus/finite-difference reductions are evaluated in FP32. `--fsdp_reduce_fp32` changes gradient-communication precision only.

### 5.4 Optional Regularizers (Relativistic / R1 / R2)

These regularizers are **off** by default. To experiment, append to the launcher:

```bash
# Relativistic discriminator: use paired real/fake logit differences
--relativistic_discriminator

# Finite-difference gradient penalties (APT form, not double-backward)
--r1_weight=1.0 --r1_sigma=0.1     # on noisy real latents:   r1_weight * mean((D_real' - D_real)^2)
--r2_weight=0.001 --r2_sigma=0.01  # on noisy gen latents: 0.5 * r2_weight * mean(((D_fake' - D_fake)/sigma)^2)
```

For logits `D_fake` and `D_real`, ordinary GAN losses are `softplus(-D_fake)` (generator) and `softplus(-D_real) + softplus(D_fake)` (discriminator); relativistic losses use the logit differences instead. Real/fake samples share timestep and noise; perturbations use independent noise.

---

## 6. Inference

### 6.1 Quick Start

The inference entry is [examples/wan2.1_one_forcing/predict_t2v.py](../../examples/wan2.1_one_forcing/predict_t2v.py). It loads the **1.3B student** (the 14B model is only the DMD teacher at training time) and runs frame-wise one-step generation. By default it uses the **official One-Forcing checkpoint**:

```bash
hf download JiaqiFeng/OneForcing checkpoints/one_forcing.pt --local-dir models/Diffusion_Transformer/One-Forcing
```

```python
# examples/wan2.1_one_forcing/predict_t2v.py
model_name          = "models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
transformer_path    = "models/Diffusion_Transformer/One-Forcing/checkpoints/one_forcing.pt"
use_ema             = False   # only applies to trainer-output checkpoint dirs
stochastic_sampling = True    # DMD-distilled student
guidance_scale      = 1.0     # CFG is baked into the distilled weights
num_inference_steps = 1       # One-Forcing one-step generation
num_frame_per_block = 1       # frame-wise
sample_size         = [480, 832]
video_length        = 81
```

Run it:

```bash
python examples/wan2.1_one_forcing/predict_t2v.py
```

`transformer_path` may be the official checkpoint file, a plain `.pt`/`.safetensors` file, or a trainer-output **checkpoint dir** (its live/EMA weights are auto-resolved). The GAN discriminator head is a training-only submodule of fake_score, so inference needs **no discriminator wiring**.

### 6.2 1-Step vs 4-Step Selector

With `stochastic_sampling=True`, `num_inference_steps` selects the warped timestep schedule:

| `num_inference_steps` | Denoise timesteps | Use case |
|-----------------------|-------------------|----------|
| `1` (One-Forcing default) | `[1000]` | Fastest; frame-wise one-step generation |
| `4` | `[1000, 750, 500, 250]` | More stable, slower |

### 6.3 First-Frame Schedule

The shared `WanSelfForcingPipeline` applies **one denoising schedule to all blocks**. A separate first-frame schedule (e.g. `[1000, 750, 500, 250]` for the first frame and `[1000]` for later frames) is therefore not applied; the pipeline is shared by multiple examples and is intentionally left unmodified.

---

## 7. Checkpoint Round-Trip

The FSDP trainer writes each `checkpoint-{step}/` as follows, so training output feeds inference with no key surgery:

| Component | Saved to | Contains `gan_head`? |
|-----------|----------|----------------------|
| **Generator** (live) | `checkpoint-{step}/diffusion_pytorch_model.safetensors` (root, via FSDP hook) | No |
| **Generator** (EMA) | `checkpoint-{step}/transformer_ema/` | No |
| **Fake-score + discriminator head** | `checkpoint-{step}/fake_score/` | Yes (FP32 head tensors) |

- The **generator never contains `gan_head`** keys (the head lives on fake-score), so `predict_t2v.py` loads it cleanly with `strict=False`.
- `_resolve_transformer_path` in the inference script resolves, in order: `transformer_ema/`, `ema_transformer/` (compat alias), `transformer/`, then the root safetensors.
- **Resume** GAN training with the same GAN/head configuration: the head is recreated from CLI flags before loading and restored from `fake_score/`.

---

## 8. Additional Resources

- **One-Forcing (source recipe)**: https://github.com/Aurora-edu/One-Forcing
- **Causal-Forcing (DMD stage)**: https://github.com/thu-ml/Causal-Forcing
- **Stage 1 (AR Diffusion) training guide**: [README_TRAIN_AR_DIFFUSION.md](../wan2.1_causal_forcing/README_TRAIN_AR_DIFFUSION.md)
- **Stage 2 (CCD) training guide**: [README_TRAIN_CAUSAL_CONSISTENCY_DISTILL.md](../wan2.1_causal_forcing/README_TRAIN_CAUSAL_CONSISTENCY_DISTILL.md)
- **Underlying DMD trainer guide**: [README_TRAIN_CAUSAL_DMD_DISTILL.md](../wan2.1_causal_forcing/README_TRAIN_CAUSAL_DMD_DISTILL.md)
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
