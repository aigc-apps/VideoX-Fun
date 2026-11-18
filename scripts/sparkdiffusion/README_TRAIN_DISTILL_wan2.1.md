# SparkDiffusion Sparse Distillation Training Guide (Wan2.1-T2V-1.3B)

This document provides the complete workflow for **SparkDiffusion** on Wan2.1-T2V-1.3B: a two-stage pipeline that first warms up a **sparse (RoLa) student** and then **distills** it into a few-step, CFG-free generator. It covers environment setup, data preparation, the two launchers (Stage-1 warm-up + Stage-2 DMD distill), and sparse few-step inference. Wan2.1-T2V-1.3B is a **single full-range expert**, so this is the leaner counterpart of the Wan2.2 MoE guide ([README_TRAIN_DISTILL_wan2.2.md](./README_TRAIN_DISTILL_wan2.2.md)): no MoE boundary, no high/low expert split, and no init-rollout relay.

> **What is SparkDiffusion?**
>
> SparkDiffusion (arXiv:2609.23153, *Mitigating the High-Sparsity Trap*) combines a **compensating sparse attention** (RoLa) with a **DMD** few-step distillation recipe. Wan2.1-T2V-1.3B is a single-expert diffusion transformer that spans the whole noise range, so the student is warmed up and distilled **once over the full rf-time range `[0, 1]`** — there is no MoE boundary to split on.
>
> - **Stage 1 — sparse warm-up** ([`train_finetune_sparse_wan2.1.py`](./train_finetune_sparse_wan2.1.py)): insert the RoLa sparse self-attention into the dense backbone and briefly fine-tune it with the plain step-local flow-matching objective, so distillation starts from a usable coarse prior instead of a zero-initialized sparse branch. This stage is intentionally **short**.
> - **Stage 2 — DMD distill** ([`train_distill_wan2.1.py`](./train_distill_wan2.1.py)): distill the sparse student into a few-step generator against the **dense** teacher. It is plain Distribution Matching Distillation (DMD) with three co-resident transformers — a trainable sparse `generator`, a frozen dense `real_score` teacher, and a trainable dense `fake_score` critic — rolled out over a discrete few-step grid. An optional **CrossDistill** PCM diversity term (`--loss_scale_div > 0`, see §3.3.1) can be layered on top to preserve sample diversity.
>
> | Stage | Launcher | Output |
> |-------|----------|--------|
> | **1 warm-up** | [`train_finetune_sparse_wan2.1.sh`](./train_finetune_sparse_wan2.1.sh) | `output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola` |
> | **2 distill** | [`train_distill_wan2.1.sh`](./train_distill_wan2.1.sh) | `output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola_distill` |
>
> Run the two launchers in order: the Stage-2 generator is seeded from the Stage-1 warm-up product, and the single distilled checkpoint is the whole few-step student at inference.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
- [3. Training](#3-training)
  - [3.1 Download Pretrained Models](#31-download-pretrained-models)
  - [3.2 Stage 1 — Sparse Warm-up](#32-stage-1--sparse-warm-up)
  - [3.3 Stage 2 — DMD Distill](#33-stage-2--dmd-distill)
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

SparkDiffusion consumes ordinary video-caption data (the same format as the standard Wan2.1 training).

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

The full order is **Stage-1 warm-up → Stage-2 DMD distill**. Run the two launchers by hand, in that order.

### 3.1 Download Pretrained Models

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download Wan2.1 T2V official weights
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B
```

The base checkpoint seeds all three Stage-2 transformers (the sparse `generator` is additionally initialized from the Stage-1 warm-up product). Its layout is the civitai single-file form read by `config/wan2.1/wan_civitai.yaml`: the transformer sits at the model root (`transformer_subpath: ./`), alongside `Wan2.1_VAE.pth`, the UMT5 encoder `models_t5_umt5-xxl-enc-bf16.pth` and the `google/umt5-xxl` tokenizer. The scheduler `shift` is `5.0`, supplied by the config `scheduler_kwargs`; the distill launcher does not override it.

### 3.2 Stage 1 — Sparse Warm-up

Stage 1 inserts the RoLa sparse self-attention and briefly warms it up with the step-local flow-matching objective. It is a **short** stage — a brief warm-up is enough to establish a usable prior for distillation, so the launcher caps it at `--max_train_steps=2000`. Do **not** judge this stage by its own sample quality; the sharpness is established in Stage 2.

Run [scripts/sparkdiffusion/train_finetune_sparse_wan2.1.sh](./train_finetune_sparse_wan2.1.sh):

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" \
    --fsdp_cpu_ram_efficient_loading False scripts/sparkdiffusion/train_finetune_sparse_wan2.1.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
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
  --output_dir="output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --random_hw_adapt \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "attn_op" \
  --student_topk=0.10 \
  --train_mode="normal" \
  --resume_from_checkpoint="latest"
```

Or simply:

```bash
bash scripts/sparkdiffusion/train_finetune_sparse_wan2.1.sh
```

Stage-1 keys:

| Flag | Meaning |
|------|---------|
| `--student_topk` | Ratio of key blocks kept per query in the sparse attention (sparsity = `1 - topk`). `0.10` ≈ 90% sparse. **Must match** the value used in Stage 2. |
| `--trainable_modules` | `"attn_op"` = Stage A (train only the sparse branch); `"."` = Stage B (full sparse finetune, load Stage A via `--transformer_path`). |
| `--max_train_steps` | Warm-up budget; keep it short. |

On completion the warm-up product is `output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola/checkpoint-2000/diffusion_pytorch_model.safetensors`.

### 3.3 Stage 2 — DMD Distill

Stage 2 distills the sparse student into a few-step, CFG-free generator against the dense teacher. `--sparse_student` turns the `generator` into `SparkWanTransformer3DModel` while the teacher and critic stay dense. `--generator_transformer_path` loads the Stage-1 warm-up checkpoint. Being a single full-range expert, the student distills over the whole rf-time range `[0, 1]` — there is no boundary segment and no init-rollout relay. This is the plain-DMD baseline: the generator rolls out a discrete few-step grid (`--denoising_step_indices_list 1000 750 500 250` with `--flow_euler_rollout`), and the generator/critic alternate via `--gen_update_interval`. On top of this discrete DMD host, an optional CrossDistill PCM diversity term (`--loss_scale_div > 0`, see §3.3.1 below) adds a phase-0 boundary-alignment loss to preserve sample diversity. Two further opt-in switches (§3.3.2) replace parts of this baseline — `--tdm` (the official SparkDiffusion TDM student loss) and `--joint_backward_timesteps` (define the few-step grid directly with rf knots) — but neither is enabled in the launcher, so the default run stays plain DMD.

Run [scripts/sparkdiffusion/train_distill_wan2.1.sh](./train_distill_wan2.1.sh):

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# Stage-1 warm-up product. Env-overridable; --student_topk below MUST match Stage 1.
export SPARSE_STUDENT_CKPT="output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola/checkpoint-2000/diffusion_pytorch_model.safetensors"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_POST" \
    --fsdp_cpu_ram_efficient_loading False scripts/sparkdiffusion/train_distill_wan2.1.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=100 \
  --checkpointing_steps=200 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola_distill" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --random_hw_adapt \
  --enable_bucket \
  --trainable_modules "." \
  --train_mode="normal" \
  --uniform_sampling \
  --denoising_step_indices_list 1000 750 500 250 \
  --flow_euler_rollout \
  --randomize_step_indices \
  --gen_update_interval 5 \
  --div_mode pcm_random \
  --loss_scale_div 5.0 \
  --dcm_total_steps 48 \
  --dcm_skipping_interval_steps 1 \
  --teacher_timestep_shift 5.0 \
  --div_rf_boundary 0.933781 \
  --div_teacher_guidance 5.0 \
  --sparse_student \
  --student_topk=0.10 \
  --generator_transformer_path=$SPARSE_STUDENT_CKPT \
  --resume_from_checkpoint="latest"
```

Or simply:

```bash
bash scripts/sparkdiffusion/train_distill_wan2.1.sh
```

> ⚠️ **The distillation phase has no step cap.** It runs until you stop it manually (kill the `train_distill_wan2.1.py` processes); checkpoints are written every `--checkpointing_steps=200`.

#### 3.3.1 CrossDistill Diversity (PCM)

Setting `--loss_scale_div > 0` layers the official SparkDiffusion **CrossDistill** PCM diversity term on top of the discrete DMD host, to counter the diversity loss that pure distribution matching can cause. It is a self-contained additive loss that leaves the DMD rollout untouched:

- **What it does** — a phase-0 boundary-alignment MSE: the student's one-step jump from a noisy anchor to the phase-0 boundary (`--div_rf_boundary`, the official `backward_timesteps[0]`) must match a teacher K-step Euler rollout (`--dcm_skipping_interval_steps` sub-steps) followed by the student's own target jump. It runs on its **own continuous rf grid** (`--dcm_total_steps` subdivisions shifted by `--teacher_timestep_shift`), decoupled from the discrete DMD grid.
- **Anchor** — the paired **real video latent** (faithful to the official `_compute_pcm_div_loss`, which distills on real `x0`), encoded on demand only when diversity is on.
- **Merge** — `generator_loss = dmd_loss + loss_scale_div * div_loss` in a **single** `accelerator.backward` (FSDP allows one backward per wrapped module per iteration); the teacher rollout and the student target run under `no_grad` before the graded student forward.
- **Scale & tuning** — `--loss_scale_div` is in **per-element-mean** space (aligned with the mean-reduced `dmd_loss`); the official uses `10.0` in sum space, so `5.0` is the recommended start. On the first run watch `train_div_loss` vs `train_dmd_loss` in the logs and adjust so diversity does not overwhelm DMD; `--div_teacher_guidance` is the diversity teacher's CFG scale, independent of `--real_guidance_scale`. Set `--loss_scale_div 0` to recover the exact plain-DMD baseline.

#### 3.3.2 TDM Loss & Direct rf Grid (opt-in)

Two opt-in switches replace parts of the plain-DMD baseline above; **neither is enabled in `train_distill_wan2.1.sh`**, so the default run stays plain DMD.

- **`--tdm`** — replaces the plain-DMD student loss with **TDM** (Trajectory Distribution Matching, the official SparkDiffusion student loss). Three deltas from DMD: (1) the last rollout segment does not collapse to `x0` but takes one more rf-Euler step to `s_{i-1}` (the next grid knot, `0.0` when exhausted), so the anchor is `G_x_{t_{i-1}}`; (2) `D_time` is clamped strictly above `s_{i-1}`; (3) the fake/teacher `x0` preds are mapped back to `s_{i-1}` via the rf PF-ODE analytic step and the surrogate is compared in the `x_{t_{i-1}}` domain. When `s_{i-1}==0` (grid exhausted) it reduces **exactly** to DMD. Requires `--flow_euler_rollout`; mutually exclusive with `--dfd` / `--decoupled_dmd`; compatible with the diversity term.
- **`--joint_backward_timesteps`** — descending rf knots (in `[0,1]`) that define the few-step grid **directly**, replacing `--denoising_step_indices_list`. Wan2.1 is a single expert (no MoE boundary split), so the whole list is used: the grid becomes `[1.0] + [k for k in knots if 0<k<1]` and the rf values serve directly as Euler sigmas (`sigma == rf == timestep/1000`). Official Wan2.1 rCM knots: `0.933781 0.852895 0.608979`. When omitted, the grid falls back to `--denoising_step_indices_list`.

### 3.4 Common Training Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config_path` | Wan2.1 config (single expert, no `boundary`) | `config/wan2.1/wan_civitai.yaml` |
| `--pretrained_model_name_or_path` | Wan2.1-T2V-1.3B base (transformer + VAE + UMT5) | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `--train_data_dir` / `--train_data_meta` | Dataset root / metadata JSON | see §2 |
| `--train_mode` | SparkDiffusion supports `normal` (T2V) only | `normal` |
| `--student_topk` | Sparse attention keep-ratio (sparsity = 1 − topk); Stage 1 and Stage 2 must match | `0.10` |
| `--sparse_student` | (Stage 2) build the generator as the sparse student | - |
| `--generator_transformer_path` | (Stage 2) Stage-1 warm-up checkpoint loaded into the generator | `.../checkpoint-2000/diffusion_pytorch_model.safetensors` |
| `--denoising_step_indices_list` | Discrete few-step grid the generator rolls out; validation reuses it (`num_inference_steps` = its length) | `1000 750 500 250` |
| `--flow_euler_rollout` | Roll the generator in flow/velocity space with deterministic Euler steps (LightX2V-style), no x0 re-noising | - |
| `--tdm` | (opt-in) Use TDM — the official SparkDiffusion student loss — instead of plain DMD; reduces exactly to DMD when `s_{i-1}==0`. Requires `--flow_euler_rollout`, mutually exclusive with `--dfd` / `--decoupled_dmd` | - |
| `--joint_backward_timesteps` | (opt-in) Descending rf knots that define the few-step grid directly, replacing `--denoising_step_indices_list` (single expert: the whole list, no MoE split) | `0.933781 0.852895 0.608979` |
| `--uniform_sampling` | Sample the DMD noise level uniformly over the training schedule | - |
| `--randomize_step_indices` | Jitter the grid indices each step for robustness | - |
| `--gen_update_interval` | Update the generator once every N steps (the critic updates every step) | `5` |
| `--real_guidance_scale` | CFG scale of the frozen dense teacher (not overridden; uses the script default) | `6.0` |
| `--div_mode` | CrossDistill diversity mode: `pcm_random` (random phase-0 offset) or `pcm_fixed` | `pcm_random` |
| `--loss_scale_div` | Weight of the CrossDistill PCM diversity term (per-element-mean space); `0` disables it and recovers plain DMD | `5.0` |
| `--dcm_total_steps` | Subdivision of the diversity term's continuous rf grid (`dw = 1 / N`) | `48` |
| `--dcm_skipping_interval_steps` | Teacher Euler sub-steps `K` within phase 0 | `1` |
| `--teacher_timestep_shift` | rf timestep shift for the diversity grid (matches the Wan2.1 scheduler shift) | `5.0` |
| `--div_rf_boundary` | Phase-0 right boundary `rf_b` (official `backward_timesteps[0]`) | `0.933781` |
| `--div_teacher_guidance` | CFG scale of the diversity teacher rollout (independent of `--real_guidance_scale`) | `5.0` |
| `--learning_rate` / `--learning_rate_critic` | Generator / critic learning rate | `2e-06` / `2e-06` |
| `--checkpointing_steps` | Save checkpoint every N steps | `200` |
| `--max_grad_norm` | Gradient clipping threshold | `0.05` |
| `--resume_from_checkpoint` | Resume path or `"latest"` | `latest` |

> 💡 The few-step grid is set by `--denoising_step_indices_list` (indices into the `--train_sampling_steps=1000` schedule); validation reuses the same grid, so `num_inference_steps` equals its length (`4` here).

### 3.5 FSDP Backend and Memory Notes

Stage 2 keeps **three** transformers co-resident (sparse generator + dense teacher + dense critic). At 1.3B each this is far lighter than the Wan2.2 14B experts, but the optimizer states still dominate, so the distill launcher keeps FSDP `FULL_SHARD` plus `--gradient_checkpointing`:

- `--fsdp_sharding_strategy "FULL_SHARD"` — shard params/grads/optimizer across GPUs.
- `--fsdp_backward_prefetch "BACKWARD_POST"` — lower median residency during backward.
- `--gradient_checkpointing` — recompute activations in backward.
- **Resolution, not frame count, is the OOM lever.** Memory scales with `seq_len`; the launchers bucket at `--image_sample_size=640 --video_sample_size=640 --token_sample_size=640` with `--random_hw_adapt --enable_bucket` at `--video_sample_n_frames=81`. If it OOMs, lower the sample sizes (e.g. `640 → 480`) rather than dropping frames.
- Frame count must satisfy `(N − 1) % 4 == 0`.

> 💡 At 1.3B the three transformers usually fit without CPU offload; if it OOMs, add `--fsdp_offload_params true` to the `accelerate launch` line to offload params/grads/optimizer to CPU (slower). `--low_vram` only parks the frozen side on CPU; it complements, but does not replace, sharding.

### 3.6 Other Backends

The Stage-2 launcher defaults to FSDP `FULL_SHARD`. The same [`train_distill_wan2.1.py`](./train_distill_wan2.1.py) also runs under DeepSpeed Zero-2 or without a sharding backend; at 1.3B these are more viable than for the Wan2.2 14B experts. Keep every script argument identical to §3.3 — only the `accelerate launch` header changes.

#### 3.6.1 Training with DeepSpeed Zero-2

DeepSpeed Zero-2 shards optimizer states and gradients but **not** parameters. For three co-resident 1.3B transformers it fits comfortably at the default resolution:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export SPARSE_STUDENT_CKPT="output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola/checkpoint-2000/diffusion_pytorch_model.safetensors"
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/sparkdiffusion/train_distill_wan2.1.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=100 \
  --checkpointing_steps=200 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola_distill" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --random_hw_adapt \
  --enable_bucket \
  --trainable_modules "." \
  --train_mode="normal" \
  --uniform_sampling \
  --denoising_step_indices_list 1000 750 500 250 \
  --flow_euler_rollout \
  --randomize_step_indices \
  --gen_update_interval 5 \
  --div_mode pcm_random \
  --loss_scale_div 5.0 \
  --dcm_total_steps 48 \
  --dcm_skipping_interval_steps 1 \
  --teacher_timestep_shift 5.0 \
  --div_rf_boundary 0.933781 \
  --div_teacher_guidance 5.0 \
  --sparse_student \
  --student_topk=0.10 \
  --generator_transformer_path=$SPARSE_STUDENT_CKPT \
  --resume_from_checkpoint="latest"
```

#### 3.6.2 Training Without DeepSpeed and FSDP

Drop the sharding backend entirely — three co-resident 1.3B transformers fit on a single high-memory GPU, though FSDP is still recommended for multi-GPU. Replace the `accelerate launch` header with the plain form and keep the same script arguments as §3.6.1:

```bash
accelerate launch --mixed_precision="bf16" \
  scripts/sparkdiffusion/train_distill_wan2.1.py \
  <the same --flag block as §3.6.1>
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

The Stage-2 product is a few-step, CFG-free sparse student. Being single-expert, inference loads **one** checkpoint (there is no high-noise expert). Build the wan2.1 sparse class `SparkWanTransformer3DModel`, set the training top-k, then load the distilled weights — the order matters because `set_student_topk` (re)creates the `attn_op` kernels that receive the trained RoLa params:

```python
from videox_fun.models import SparkWanTransformer3DModel
transformer = SparkWanTransformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True, torch_dtype=weight_dtype,
)
transformer.set_student_topk(student_topk)          # BEFORE load_state_dict
# transformer.load_state_dict(load_file(transformer_path), strict=False)
```

The Wan2.1 T2V entry [examples/wan2.1/predict_t2v.py](../../examples/wan2.1/predict_t2v.py) already runs few-step CFG-free sampling (`sampler_name="Flow"`, `num_inference_steps=4`, `guidance_scale=1.0`) under `WanPipeline`; adapt it by swapping the dense `WanTransformer3DModel` for the sparse build above. Set:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `GPU_memory_mode` | Memory mode | `model_full_load` |
| `config_path` | Wan2.1 config | `config/wan2.1/wan_civitai.yaml` |
| `model_name` | Wan2.1-T2V-1.3B base (transformer / VAE / UMT5) | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `student_topk` | **Must equal** the training `--student_topk` | `0.10` |
| `transformer_path` | Sparse student checkpoint (single expert) | `output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola_distill/checkpoint-<step>/diffusion_pytorch_model.safetensors` |
| `sampler_name` | `Flow` matches the few-step CFG-free student | `Flow` |
| `num_inference_steps` | The student's few-step count | `4` |
| `guidance_scale` | Distilled student is CFG-free | `1.0` |
| `sample_size` / `video_length` / `fps` | `[height, width]` / frames / fps | `[480, 832]` / `81` / `16` |

```bash
python examples/wan2.1/predict_t2v.py
```

> 💡 Load order is `from_pretrained → set_student_topk → load_state_dict`; reversing it re-zeroes the trained RoLa weights. A sparse checkpoint is confirmed when the `attn_op.*` keys load with **0** unexpected keys.

> 📌 The bundled [examples/sparkdiffusion/predict_t2v.py](../../examples/sparkdiffusion/predict_t2v.py) targets the Wan2.2 two-expert MoE student (`SparkWan2_2Transformer3DModel` + `Wan2_2Pipeline`); the single-expert Wan2.1 student uses `SparkWanTransformer3DModel` + `WanPipeline` with one checkpoint.

> 📌 **Stage 3** of the paper (the FP8/INT8 fused quantized inference kernel) is intentionally not integrated in this repository.

---

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
- **Paper**: SparkDiffusion, *Mitigating the High-Sparsity Trap* (arXiv:2609.23153)
- **Sparse student model**: `videox_fun/models/spark_transformer3d.py` (`SparkWanTransformer3DModel`)
- **RoLa sparse attention kernel**: `videox_fun/models/attention_utils.py`
- **Wan2.2 MoE counterpart**: [README_TRAIN_DISTILL_wan2.2.md](./README_TRAIN_DISTILL_wan2.2.md)
