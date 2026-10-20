# SparkDiffusion Sparse Distillation Training Guide

This document provides the complete workflow for **SparkDiffusion** on Wan2.2-T2V-A14B: a two-stage pipeline that first warms up a **sparse (RoLa) student** and then **distills** it into a few-step, CFG-free generator. It covers environment setup, data preparation, the four launchers (Stage-1 warm-up + Stage-2 CrossDistill, per MoE expert), the one-shot orchestrator, and the `predict_t2v.py` inference acceptance test.

> **What is SparkDiffusion?**
>
> SparkDiffusion (arXiv:2609.23153, *Mitigating the High-Sparsity Trap*) combines a **compensating sparse attention** (RoLa) with a **CrossDistill** few-step recipe. Wan2.2-T2V-A14B is a two-expert MoE (a high-noise expert and a low-noise expert, split at the config `boundary: 0.875`); each expert is warmed up and distilled **on its own noise range**, never jointly.
>
> - **Stage 1 — sparse warm-up** ([`train_finetune_sparse_wan2.2.py`](./train_finetune_sparse_wan2.2.py)): insert the RoLa sparse self-attention into the dense backbone and briefly fine-tune it with the plain step-local flow-matching objective, so distillation starts from a usable coarse prior instead of a zero-initialized sparse branch. This stage is intentionally **short**.
> - **Stage 2 — CrossDistill** ([`train_distill_wan2.2.py`](./train_distill_wan2.2.py)): distill the sparse student into a few-step generator against the **dense** teacher. It is Distribution Matching Distillation (DMD) with three co-resident transformers — a trainable sparse `generator`, a frozen dense `real_score` teacher, and a trainable dense `fake_score` critic. The **high-noise** expert adds a PCM-style diversity term; the **low-noise** expert is div-free.
>
> | Stage | High-noise expert | Low-noise expert | Output |
> |-------|-------------------|------------------|--------|
> | **1 warm-up** | [`train_finetune_sparse_wan2.2_high.sh`](./train_finetune_sparse_wan2.2_high.sh) | [`train_finetune_sparse_wan2.2_low.sh`](./train_finetune_sparse_wan2.2_low.sh) | `output_dir_sparkdiffusion_wan2.2_14b_{high,low}_sparse_rola` |
> | **2 distill** | [`train_distill_wan2.2_high.sh`](./train_distill_wan2.2_high.sh) | [`train_distill_wan2.2_low.sh`](./train_distill_wan2.2_low.sh) | `output_dir_sparkdiffusion_wan2.2_14b_{high,low}_sparse_rola_distill` |
>
> The two experts are distilled separately and both are loaded at inference to form the full few-step MoE student.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
- [3. Training](#3-training)
  - [3.1 Download Pretrained Models](#31-download-pretrained-models)
  - [3.2 Stage 1 — Sparse Warm-up](#32-stage-1--sparse-warm-up)
  - [3.3 Stage 2 — CrossDistill](#33-stage-2--crossdistill)
  - [3.4 Common Training Parameters](#34-common-training-parameters)
  - [3.5 FSDP Backend and Memory Notes](#35-fsdp-backend-and-memory-notes)
  - [3.6 Other Backends](#36-other-backends)
  - [3.7 Multi-Node Distributed Training](#37-multi-node-distributed-training)
- [4. Inference — Sparse Few-Step Student](#4-inference--sparse-few-step-student)
- [5. Additional Resources](#5-additional-resources)

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

> 💡 The RoLa sparse attention relies on the in-repo Triton block-sparse kernel (`videox_fun/models/attention_utils.py`), so a working Triton install is required for the sparse student.

**Method 3: Using Docker**

```bash
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

SparkDiffusion consumes ordinary video-caption data (the same format as the standard Wan2.2 training).

### 2.1 Quick Test Dataset

```bash
# Download official example dataset
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 2.2 metadata.json Format

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

- `file_path`: video path (relative to `DATASET_NAME`, or absolute).
- `text`: caption / prompt.
- `type`: fixed as `"video"`.
- `width` / `height`: recommended for bucket training; add them with `scripts/process_json_add_width_and_height.py` if absent.

### 2.3 Relative vs Absolute Paths

```bash
# relative paths
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"

# absolute paths (data on NAS/OSS shared across machines)
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata_add_width_height.json"
```

---

## 3. Training

The full order is **Stage-1 high → Stage-2 high → Stage-1 low → Stage-2 low**. Run the four launchers by hand, in that order.

### 3.1 Download Pretrained Models

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download Wan2.2 T2V official weights
modelscope download --model Wan-AI/Wan2.2-T2V-A14B --local_dir models/Diffusion_Transformer/Wan2.2-T2V-A14B
```

The base checkpoint seeds all three Stage-2 transformers (the sparse `generator` is additionally initialized from the Stage-1 warm-up product). Its layout is `high_noise_model/`, `low_noise_model/`, the VAE and the UMT5 text encoder; the MoE boundary is read from `config/wan2.2/wan_civitai_t2v.yaml` (`boundary: 0.875`).

### 3.2 Stage 1 — Sparse Warm-up

Stage 1 inserts the RoLa sparse self-attention and briefly warms it up with the step-local flow-matching objective. It is a **short** stage — a brief warm-up is enough to establish a usable prior for distillation, so the launchers cap it at `--max_train_steps=2000`. Do **not** judge this stage by its own sample quality; the sharpness is established in Stage 2.

Run the high-noise expert with [scripts/sparkdiffusion/train_finetune_sparse_wan2.2_high.sh](./train_finetune_sparse_wan2.2_high.sh):

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" \
    --fsdp_cpu_ram_efficient_loading False scripts/sparkdiffusion/train_finetune_sparse_wan2.2.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=100 \
  --max_train_steps=2000 \
  --checkpointing_steps=200 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_sparkdiffusion_wan2.2_14b_high_sparse_rola" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "attn_op" \
  --student_topk=0.05 \
  --train_mode="normal" \
  --boundary_type="high" \
  --resume_from_checkpoint="latest"
```

Or simply:

```bash
bash scripts/sparkdiffusion/train_finetune_sparse_wan2.2_high.sh
```

The low-noise expert uses the same recipe with `--boundary_type="low"` and its own `--output_dir`:

```bash
bash scripts/sparkdiffusion/train_finetune_sparse_wan2.2_low.sh
```

Stage-1 keys:

| Flag | Meaning |
|------|---------|
| `--boundary_type` | `high` / `low` — selects the MoE expert to warm up (from the config boundary). |
| `--student_topk` | Ratio of key blocks kept per query in the sparse attention (sparsity = `1 - topk`). `0.05` ≈ 95% sparse. **Must match** the value used in Stage 2. |
| `--trainable_modules` | `"attn_op"` = Stage A (train only the sparse branch); `"."` = Stage B (full sparse finetune, load Stage A via `--transformer_path`). |
| `--max_train_steps` | Warm-up budget; keep it short. |

On completion the warm-up product is `output_dir_sparkdiffusion_wan2.2_14b_{high,low}_sparse_rola/checkpoint-2000/diffusion_pytorch_model.safetensors`.

### 3.3 Stage 2 — CrossDistill

Stage 2 distills the sparse student into a few-step, CFG-free generator against the dense teacher. `--sparse_student` turns the `generator` (and the `init_rollout` model) into `SparkWan2_2Transformer3DModel` while the teacher and critic stay dense. `--generator_transformer_path` loads the Stage-1 warm-up checkpoint. The host is a plain **discrete-grid DMD**: the few-step grid is derived in-script by splitting `--joint_backward_timesteps` at the config `boundary` (`0.875`) and keeping only this expert's segment (`--boundary_type`), then rolled out with rf-native Euler steps (`--flow_euler_rollout`), the generator and critic alternating via `--gen_update_interval`. On top of this host the **high** expert adds an optional CrossDistill PCM diversity term (`--loss_scale_div > 0`, see §3.3.1); the **low** expert is div-free. An opt-in `--tdm` switch (§3.3.2) replaces the plain-DMD student loss with the official SparkDiffusion **TDM** loss; it is not enabled in the published launchers, so the default run stays plain DMD.

Run the high-noise expert with [scripts/sparkdiffusion/train_distill_wan2.2_high.sh](./train_distill_wan2.2_high.sh):

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# Stage-1 warm-up product.
export SPARSE_STUDENT_CKPT="output_dir_sparkdiffusion_wan2.2_14b_high_sparse_rola/checkpoint-2000/diffusion_pytorch_model.safetensors"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_POST" \
    --fsdp_offload_params true \
    --fsdp_cpu_ram_efficient_loading False scripts/sparkdiffusion/train_distill_wan2.2.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_n_frames=81 \
  --fix_sample_size 480 832 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=100 \
  --checkpointing_steps=200 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_sparkdiffusion_wan2.2_14b_high_sparse_rola_distill" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --enable_bucket \
  --trainable_modules "." \
  --train_mode="normal" \
  --uniform_sampling \
  --flow_euler_rollout \
  --gen_update_interval 5 \
  --joint_backward_timesteps 0.933781 0.875 0.608979 \
  --div_mode pcm_random \
  --loss_scale_div 10.0 \
  --dcm_total_steps 48 \
  --dcm_skipping_interval_steps 1 \
  --teacher_timestep_shift 12.0 \
  --div_rf_boundary 0.933781 \
  --div_teacher_guidance 5.0 \
  --boundary_type="high" \
  --real_guidance_scale=5.0 \
  --sparse_student \
  --student_topk=0.05 \
  --generator_transformer_path=$SPARSE_STUDENT_CKPT \
  --resume_from_checkpoint="latest"
```

Or simply:

```bash
bash scripts/sparkdiffusion/train_distill_wan2.2_high.sh
```

The low-noise expert ([scripts/sparkdiffusion/train_distill_wan2.2_low.sh](./train_distill_wan2.2_low.sh)) differs in three ways:

```bash
bash scripts/sparkdiffusion/train_distill_wan2.2_low.sh
```

| Difference | High expert | Low expert |
|------------|-------------|------------|
| `--boundary_type` | `high` | `low` |
| Diversity term | `--div_mode="pcm_random" --loss_scale_div=10.0` | div-free (no div flags) |
| `--init_rollout_path` | not used | warm-starts the rollout's boundary latent from the trained **high** student (`INIT_ROLLOUT_CKPT`) |

> ⚠️ The low-noise segment of the few-step trajectory starts at the MoE boundary, not at pure noise. Point `INIT_ROLLOUT_CKPT` at the latest high-distill checkpoint (`output_dir_sparkdiffusion_wan2.2_14b_high_sparse_rola_distill/checkpoint-<step>/diffusion_pytorch_model.safetensors`) so the frozen high student rolls the high-noise segment; leaving it empty makes the untrained low student roll that segment itself and the samples degrade.

> ⚠️ **The distillation phases have no step cap.** They run until you stop them manually (kill the `train_distill_wan2.2.py` processes); checkpoints are written every `--checkpointing_steps=200`.

#### 3.3.1 CrossDistill Diversity (PCM, high expert only)

On the **high**-noise expert, setting `--loss_scale_div > 0` layers the official SparkDiffusion **CrossDistill** PCM diversity term on top of the discrete-grid DMD host, to counter the diversity loss that pure distribution matching can cause. It is a self-contained additive loss that leaves the DMD rollout untouched (the **low** expert keeps `--loss_scale_div 0` and is div-free):

- **What it does** — a phase-0 boundary-alignment MSE: the student's one-step jump from a noisy anchor to the phase-0 boundary (`--div_rf_boundary`, the high segment's first node `0.933781`) must match a teacher K-step Euler rollout (`--dcm_skipping_interval_steps` sub-steps) followed by the student's own target jump. It runs on its **own continuous rf grid** (`--dcm_total_steps` subdivisions shifted by `--teacher_timestep_shift 12.0`, the Wan2.2 scheduler shift), decoupled from the discrete DMD grid.
- **Anchor** — the paired **real video latent** (faithful to the official `_compute_pcm_div_loss`, which distills on real `x0`), encoded on demand only when diversity is on.
- **Merge** — `generator_loss = dmd_loss + loss_scale_div * div_loss` in a **single** `accelerator.backward` (FSDP allows one backward per wrapped module per iteration); the teacher rollout and the student target run under `no_grad` before the graded student forward.
- **Scale & tuning** — `--loss_scale_div` is in **per-element-mean** space (aligned with the mean-reduced `dmd_loss`); the launchers start at `10.0` on the high expert. On the first run watch `train_div_loss` vs `train_dmd_loss` and adjust so diversity does not overwhelm DMD; `--div_teacher_guidance` is the diversity teacher's CFG scale, independent of `--real_guidance_scale`. Set `--loss_scale_div 0` to recover the exact plain-DMD baseline.

#### 3.3.2 TDM Loss (opt-in)

An opt-in `--tdm` switch replaces the plain-DMD student loss with **TDM** (Trajectory Distribution Matching, the official SparkDiffusion student loss); **it is not enabled in the published `train_distill_wan2.2_{high,low}.sh`**, so the default run stays plain DMD. Three deltas from DMD: (1) the last rollout segment does not collapse to `x0` but takes one more rf-Euler step to `s_{i-1}` (the next grid knot), so the anchor is `G_x_{t_{i-1}}`; (2) `D_time` is clamped strictly above `s_{i-1}`; (3) the fake/teacher `x0` preds are mapped back to `s_{i-1}` via the rf PF-ODE analytic step and the surrogate is compared in the `x_{t_{i-1}}` domain. When `s_{i-1}==0` (grid exhausted) it reduces **exactly** to DMD. Requires `--flow_euler_rollout`; mutually exclusive with `--dfd` / `--decoupled_dmd`; compatible with the diversity term on the high expert.

### 3.4 Common Training Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config_path` | Wan2.2 config; carries the MoE `boundary` | `config/wan2.2/wan_civitai_t2v.yaml` |
| `--pretrained_model_name_or_path` | Wan2.2-T2V-A14B base (both experts + VAE + UMT5) | `models/Diffusion_Transformer/Wan2.2-T2V-A14B` |
| `--train_data_dir` / `--train_data_meta` | Dataset root / metadata JSON | see §2 |
| `--boundary_type` | `high` / `low` MoE expert selector | `high` |
| `--train_mode` | SparkDiffusion supports `normal` (T2V) only | `normal` |
| `--student_topk` | Sparse attention keep-ratio (sparsity = 1 − topk); Stage 1 and Stage 2 must match | `0.05` |
| `--sparse_student` | (Stage 2) build generator + init_rollout as the sparse student | - |
| `--generator_transformer_path` | (Stage 2) Stage-1 warm-up checkpoint loaded into the generator | `.../checkpoint-2000/diffusion_pytorch_model.safetensors` |
| `--uniform_sampling` | Sample the DMD noise level uniformly over the training schedule | - |
| `--flow_euler_rollout` | Roll the generator in flow/velocity space with deterministic rf-native Euler steps (no x0 re-noising) | - |
| `--tdm` | (opt-in) Use TDM — the official SparkDiffusion student loss — instead of plain DMD; reduces exactly to DMD when `s_{i-1}==0`. Requires `--flow_euler_rollout`, mutually exclusive with `--dfd` / `--decoupled_dmd` | - |
| `--gen_update_interval` | Update the generator once every N steps (the critic updates every step) | `5` |
| `--joint_backward_timesteps` | Joint few-step rf grid (descending), split at the config `boundary` into per-expert segments by `--boundary_type`; passed explicitly in the launchers (equals the built-in default), and validation reuses it | `0.933781 0.875 0.608979` |
| `--div_mode` | CrossDistill diversity mode: `pcm_random` (random phase-0 offset) or `pcm_fixed` (high expert only) | `pcm_random` |
| `--loss_scale_div` | Weight of the CrossDistill PCM diversity term (per-element-mean space); `0` disables it | `10.0` (high) / `0` (low) |
| `--dcm_total_steps` | Subdivision of the diversity term's continuous rf grid (`dw = 1 / N`) | `48` |
| `--dcm_skipping_interval_steps` | Teacher Euler sub-steps `K` within phase 0 | `1` |
| `--teacher_timestep_shift` | rf timestep shift for the diversity grid (matches the Wan2.2 scheduler shift) | `12.0` |
| `--div_rf_boundary` | Phase-0 right boundary `rf_b` (the high segment's first node) | `0.933781` |
| `--div_teacher_guidance` | CFG scale of the diversity teacher rollout (independent of `--real_guidance_scale`) | `5.0` |
| `--real_guidance_scale` | CFG scale of the frozen dense teacher | `5.0` |
| `--init_rollout_path` | (low expert) trained high student used to roll the rollout's boundary start latent | high-distill ckpt |
| `--learning_rate` / `--learning_rate_critic` | Generator / critic learning rate | `2e-06` / `2e-06` |
| `--checkpointing_steps` | Save checkpoint every N steps | `200` |
| `--max_grad_norm` | Gradient clipping threshold | `0.05` |
| `--resume_from_checkpoint` | Resume path or `"latest"` | `latest` |

> 💡 The per-expert few-step grid is **auto-derived**: `--joint_backward_timesteps` (the joint rf grid, passed explicitly in the launchers) is split at the config `boundary` by `--boundary_type`, and `--init_rollout_timesteps` (the low expert's frozen high-segment relay) is filled from the knots above the boundary. Leave `--init_rollout_timesteps` at its default; validation reuses the same grid, so `num_inference_steps` = `len(--joint_backward_timesteps) + 1` (`4` here).

### 3.5 FSDP Backend and Memory Notes

Stage 2 keeps **three** 14B transformers co-resident (sparse generator + dense teacher + dense critic; the low expert also parks a frozen `init_rollout` high-noise relay, sharded and `no_grad`), all optimizer-state dominated, so a sharding backend is mandatory — FSDP `FULL_SHARD` here. The distill launchers combine these memory levers:

- `--fsdp_sharding_strategy "FULL_SHARD"` — shard params/grads/optimizer across GPUs.
- `--fsdp_backward_prefetch "BACKWARD_POST"` — lower median residency during backward.
- `--gradient_checkpointing` — recompute activations in backward.
- **Resolution, not frame count, is the OOM lever.** Each step rolls the few-step grid and backprops a grad-carrying 14B forward, so memory scales with `seq_len`; the launchers use `--fix_sample_size 480 832` with `--video_sample_n_frames=81`. If it OOMs, lower `--fix_sample_size` (e.g. to `320 512`) rather than dropping frames.
- Frame count must satisfy `(N − 1) % 4 == 0`.

> 💡 The distill launchers already pass `--fsdp_offload_params true` (mandatory at 14B: it offloads params/grads/optimizer to CPU, at some speed cost); drop it only if you have ample VRAM. `--low_vram` additionally parks the frozen side (`real_score` / `init_rollout` / text encoder) on CPU; it complements, but does not replace, sharding.

### 3.6 Other Backends

The Stage-2 launchers default to FSDP `FULL_SHARD`. The same [`train_distill_wan2.2.py`](./train_distill_wan2.2.py) also runs under DeepSpeed Zero-2 or without a sharding backend. All examples below use the **high**-noise expert; for the low expert flip `--boundary_type="low"`, drop the diversity flags (`--div_mode` … `--div_teacher_guidance`), add `--init_rollout_path`, and change `--output_dir` / `SPARSE_STUDENT_CKPT` as in §3.3.

#### 3.6.1 Training with DeepSpeed Zero-2

DeepSpeed Zero-2 shards optimizer states and gradients but **not** parameters. Since Stage 2 keeps three 14B transformers co-resident, Zero-2 only fits at low resolution and may OOM where FSDP `FULL_SHARD` (which also shards parameters) does not — prefer FSDP for the 14B experts.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# Stage-1 warm-up product.
export SPARSE_STUDENT_CKPT="output_dir_sparkdiffusion_wan2.2_14b_high_sparse_rola/checkpoint-2000/diffusion_pytorch_model.safetensors"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/sparkdiffusion/train_distill_wan2.2.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_n_frames=81 \
  --fix_sample_size 480 832 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=100 \
  --checkpointing_steps=200 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_sparkdiffusion_wan2.2_14b_high_sparse_rola_distill" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --enable_bucket \
  --trainable_modules "." \
  --train_mode="normal" \
  --uniform_sampling \
  --flow_euler_rollout \
  --gen_update_interval 5 \
  --joint_backward_timesteps 0.933781 0.875 0.608979 \
  --div_mode pcm_random \
  --loss_scale_div 10.0 \
  --dcm_total_steps 48 \
  --dcm_skipping_interval_steps 1 \
  --teacher_timestep_shift 12.0 \
  --div_rf_boundary 0.933781 \
  --div_teacher_guidance 5.0 \
  --boundary_type="high" \
  --real_guidance_scale=5.0 \
  --sparse_student \
  --student_topk=0.05 \
  --generator_transformer_path=$SPARSE_STUDENT_CKPT \
  --resume_from_checkpoint="latest"
```

#### 3.6.2 Training Without DeepSpeed and FSDP

**This approach is not recommended: with three co-resident 14B transformers and no sharding backend it easily OOMs.** It is provided only as a reference shell.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# Stage-1 warm-up product.
export SPARSE_STUDENT_CKPT="output_dir_sparkdiffusion_wan2.2_14b_high_sparse_rola/checkpoint-2000/diffusion_pytorch_model.safetensors"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/sparkdiffusion/train_distill_wan2.2.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_n_frames=81 \
  --fix_sample_size 480 832 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=100 \
  --checkpointing_steps=200 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_sparkdiffusion_wan2.2_14b_high_sparse_rola_distill" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --enable_bucket \
  --trainable_modules "." \
  --train_mode="normal" \
  --uniform_sampling \
  --flow_euler_rollout \
  --gen_update_interval 5 \
  --joint_backward_timesteps 0.933781 0.875 0.608979 \
  --div_mode pcm_random \
  --loss_scale_div 10.0 \
  --dcm_total_steps 48 \
  --dcm_skipping_interval_steps 1 \
  --teacher_timestep_shift 12.0 \
  --div_rf_boundary 0.933781 \
  --div_teacher_guidance 5.0 \
  --boundary_type="high" \
  --real_guidance_scale=5.0 \
  --sparse_student \
  --student_topk=0.05 \
  --generator_transformer_path=$SPARSE_STUDENT_CKPT \
  --resume_from_checkpoint="latest"
```

### 3.7 Multi-Node Distributed Training

The launchers are single-node. For multiple machines, add the following to the `accelerate launch` line (all machines must share the same dataset / base weights / `output_dir` over NFS/shared storage):

```bash
--main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
--num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK
```

Without RDMA, also set `NCCL_IB_DISABLE=1` and `NCCL_P2P_DISABLE=1`.

---

## 4. Inference — Sparse Few-Step Student

Acceptance is a standalone run of [examples/sparkdiffusion/predict_t2v.py](../../examples/sparkdiffusion/predict_t2v.py), which rebuilds each MoE expert as a sparse `SparkWan2_2Transformer3DModel` and loads the distilled checkpoints. Edit the top of the script:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `GPU_memory_mode` | Memory mode | `model_cpu_offload` |
| `model_name` | Wan2.2-T2V-A14B base (VAE / UMT5 / schedulers) | `models/Diffusion_Transformer/Wan2.2-T2V-A14B` |
| `student_topk` | **Must equal** the training `--student_topk` | `0.05` |
| `transformer_path` | **Low**-noise sparse student checkpoint | `output_dir_sparkdiffusion_wan2.2_14b_low_sparse_rola_distill/checkpoint-<step>/diffusion_pytorch_model.safetensors` |
| `transformer_high_path` | **High**-noise sparse student checkpoint | `output_dir_sparkdiffusion_wan2.2_14b_high_sparse_rola_distill/checkpoint-<step>/diffusion_pytorch_model.safetensors` |
| `sampler_name` | `Flow` matches the few-step CFG-free student | `Flow` |
| `num_inference_steps` | The student's few-step count | `4` |
| `guidance_scale` | Distilled student is CFG-free | `1.0` |
| `sample_size` / `video_length` / `fps` | `[height, width]` / frames / fps | `[240, 336]` / `81` / `16` |

```bash
python examples/sparkdiffusion/predict_t2v.py
```

> 💡 Per-expert loading order is `from_pretrained → set_student_topk → load_state_dict`; reversing it re-zeroes the trained RoLa weights. An expert whose path is `None` stays dense — for the full SparkDiffusion student, set **both** paths (prefer each expert's Stage-2 distill product).

> 📌 **Stage 3** of the paper (the FP8/INT8 fused quantized inference kernel) is intentionally not integrated in this repository.

---

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
- **Paper**: SparkDiffusion, *Mitigating the High-Sparsity Trap* (arXiv:2609.23153)
- **Sparse student model**: `videox_fun/models/spark_transformer3d.py` (`SparkWan2_2Transformer3DModel`)
- **RoLa sparse attention kernel**: `videox_fun/models/attention_utils.py`
