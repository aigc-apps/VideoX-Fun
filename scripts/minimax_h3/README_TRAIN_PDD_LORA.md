# MiniMax-H3 PDD LoRA Training Guide

This document provides a complete workflow for Parallel Decoding Distillation (PDD, [arXiv 2607.26004](https://arxiv.org/abs/2607.26004)) LoRA training of MiniMax-H3, including environment configuration, conditioning-cache preparation, distributed training, and inference testing.

> **Note**: MiniMax-H3 is an audio-visual generative video model that can simultaneously generate video and corresponding audio. PDD training is **data-free**: it never reads target videos. Each rank carries one trajectory, rolls it forward with the student's own predictions, and is supervised by a frozen teacher on the same backbone. Only cached Qwen3-VL conditioning is needed, which keeps the ~62 GB text encoder out of the training run.

PDD turns the pre-trained flow model into a *parallel decoder*. The sampling interval is discretized into `N` intervals grouped into blocks of size `L`; one network evaluation predicts the mean velocity of every interval of the next block, so generation advances `L` intervals per evaluation (`NFE = N / L`). The default recipe is `N = 32`, `L = 4` (8 NFE). The student is the teacher's own transformer with the two final heads (`proj_out` / `audio_proj_out`) repeated `N` times; switching LoRA off is still the teacher, so there is no second copy of the 33 B backbone.

---

## Table of Contents
- [1. Environment Configuration](#1-environment-configuration)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Data-free Conditioning](#21-data-free-conditioning)
  - [2.2 Cache Structure](#22-cache-structure)
  - [2.3 Generating the Cache](#23-generating-the-cache)
  - [2.4 Annotation JSON Format](#24-annotation-json-format)
  - [2.5 Ref2VA Request Cache](#25-ref2va-request-cache)
- [3. PDD LoRA Training](#3-pdd-lora-training)
  - [3.1 Download Pretrained Model](#31-download-pretrained-model)
  - [3.2 Quick Start (FSDP)](#32-quick-start-fsdp)
    - [3.2.1 Ref2VA Training](#321-ref2va-training)
  - [3.3 PDD Training Parameters](#33-pdd-training-parameters)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Checkpoint Layout](#35-checkpoint-layout)
  - [3.6 Training with DeepSpeed-Zero-2](#36-training-with-deepspeed-zero-2)
  - [3.7 Training Without DeepSpeed or FSDP](#37-training-without-deepspeed-or-fsdp)
  - [3.8 Multi-Machine Distributed Training](#38-multi-machine-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameters](#41-inference-parameters)
  - [4.2 Single GPU Inference](#42-single-gpu-inference)
  - [4.3 Multi-GPU Parallel Inference](#43-multi-gpu-parallel-inference)
- [5. Additional Resources](#5-additional-resources)

---

## 1. Environment Configuration

**Method 1: Using requirements.txt**

```bash
pip install -r requirements.txt
```

**Method 2: Manual Dependency Installation**

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

When using Docker, please ensure that the GPU driver and CUDA environment are correctly installed on your machine, then execute the following commands:

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

PDD is **data-free**: the student trajectory is sampled from noise, so no target video/audio media is ever read for the loss. Training consumes only *conditioning* — the prompt for `fl2va`, and the prompt plus reference media for `ref2va`. Both `--train_mode`s support both conditioning routes, selected by `--enable_preprocess_training`:

| `--train_mode` | Conditions on | **Cache route** (`--enable_preprocess_training`) | **Direct-load route** (flag omitted) |
|----------------|---------------|--------------------------------------------------|--------------------------------------|
| `fl2va` | prompt | Pre-encode with `generate_prompt_cache.py`, read the `outputs.json` | Read a `{"text": ...}` annotation (`TextDataset`), encode on the fly |
| `ref2va` | prompt + reference media | Pre-encode with `generate_ref2va_request_cache.py`, read the `outputs.json` | Read a request annotation (`load_requests`), encode the prompt + reference latents on the fly |

- **Cache route** (recommended for long / repeated runs): the Qwen3-VL embeddings — and, for `ref2va`, the VAE-encoded reference latents — are pre-encoded to safetensors **once**, so the ~62 GB conditioner never loads during training.
- **Direct-load route** (recommended to start, or for a small request set): there is no separate preprocessing step; the run loads the ~62 GB conditioner (and, for `ref2va`, the two VAEs) and encodes each entry on the fly. Keep `--low_vram` on so they move onto the GPU only while encoding.

> 💡 Either route feeds **both** the train trajectories and the validation renders — data-free PDD has no train/val split, so `--train_data_meta` and `--val_data_meta` usually point at the same annotation (e.g. the official `datasets/X-Fun-Videos-Audios-Demo`, the standard test data — never an ad-hoc prompt set).

### 2.1 Data-free Conditioning

`--train_mode=fl2va` (the default recipe, FL2VA / t2va packed layout) needs the prompt conditioning; `--train_mode=ref2va` additionally needs reference media (images / videos / audio) and loads `transformer_ref` by default. On the **cache route**, generate the cache **once** with `scripts/minimax_h3/generate_prompt_cache.sh` (`fl2va`) or `scripts/minimax_h3/generate_ref2va_request_cache.sh` (`ref2va`); both run multi-GPU under `accelerate launch`, and the ~62 GB Qwen3-VL conditioner is then not loaded during PDD training. On the **direct-load route** there is no preprocessing step: point `--train_data_meta` straight at the annotation and the run encodes it on the fly (see **3.2.1** for both `ref2va` routes end to end).

### 2.2 Cache Structure

```
📦 datasets/
├── 📂 X-Fun-Videos-Audios-Demo/          # official demo dataset (source of the `text` captions)
│   └── 📄 metadata_add_width_height.json
└── 📂 minimax_h3_pdd_prompt_cache/       # generated once; feeds both train and validation
    ├── 📄 outputs.json
    ├── 📄 00000.safetensors
    ├── 📄 00001.safetensors
    └── 📄 ...
```

`outputs.json` is a list of `{"file_path": ".../00000.safetensors"}` records that `ImageVideoSafetensorsDataset` reads. `--train_data_dir` is the optional root prepended to each `file_path`; leave it empty when `outputs.json` already stores repo-relative (or absolute) paths, as the generators do. Each `fl2va` `.safetensors` holds:

| Field | Description |
|-------|-------------|
| `prompt_embeds` | Qwen3-VL hidden states at the MiniMax-H3 text-encoder layer (bfloat16) |
| `text_token_tags` | Per-token tags for the packed sequence (int64) |

### 2.3 Generating the Cache

This step is only for the **cache route**; the direct-load route (**3.2.1 Route A**) reads the annotation directly and skips it.

```bash
# fl2va: cache the prompt conditioning of the official demo dataset (multi-GPU)
bash scripts/minimax_h3/generate_prompt_cache.sh

# ref2va: cache the request conditioning (prompt embeds + reference latents)
bash scripts/minimax_h3/generate_ref2va_request_cache.sh
```

Each launcher runs `accelerate launch ... generate_*_cache.py` once: every rank walks an interleaved slice of the annotation, `.safetensors` that already exist are skipped (resume), and rank0 finally writes `outputs.json`. Edit the `MODEL_NAME` / `DATASET_META` / `CACHE_ROOT` variables at the top of each `.sh` first.

> 💡 `--pretrained_model_name_or_path` may be either the converted diffusers layout or an original MiniMax-H3 partition (e.g. `MiniMax-H3/FL2VA`). The tokenizer is read from `tokenizer/`, the processor from `processor/`, and the conditioner from `text_encoder/`.

### 2.4 Annotation JSON Format

The `fl2va` generator reads the official demo dataset's annotation JSON — a list of records whose `text` caption is the only field PDD uses (the `file_path` / `audio_path` / `control_file_path` / `width` / `height` fields the audio-visual dataset carries are ignored). A bare list of strings, or `{"prompt": ...}` / `{"examples": [...]}` jobs, are accepted too:

```json
[
  {"file_path": "train/00000001.mp4", "text": "A young woman gently turns her head to the right ...", "audio_path": "wav/00000001.wav", "type": "video"}
]
```

The `ref2va` route — both `generate_ref2va_request_cache.py` (cache) and direct-load training — reads a list of request records instead. Either an explicit `{"prompt": ..., "references": [...]}` request, where each reference is an `"image=..."` / `"video=..."` / `"audio=..."` string (the `predict_ref2va.py` schema, in the order the model reads them):

```json
[
  {"prompt": "The character turns and waves", "references": ["image=ref/face.png", "audio=ref/voice.wav"]}
]
```

or — the default — the *same* audio-visual demo annotation as `fl2va` above: `load_requests` derives the `video=<file_path>` + `audio=<audio_path>` references from each record (relative media paths resolve against the annotation's directory) and uses its `text` as the prompt.

Only the conditioning is encoded. Resolution and duration are training/inference flags (`--video_sample_size` / `--fix_sample_size` / `--video_sample_n_frames`), not cache fields.

### 2.5 Ref2VA Request Cache

`--train_mode=ref2va` loads `transformer_ref` by default. On the **cache route** it reads the request cache written by `generate_ref2va_request_cache.py`: besides `prompt_embeds` / `text_token_tags`, each request `.safetensors` also carries the reference latents for the Ref2VA packed layout, flattened into tensors — `reference_kind_ids` / `reference_has_audio` (per-reference kind and has-audio flag), `num_condition_latents` / `num_audio_condition_latents`, and the indexed `condition_latents_{i}` / `audio_condition_latents_{i}`. On the **direct-load route** the very same latents are produced on the fly instead — `train_pdd_lora.py` reads the request annotation with `load_requests`, encodes the prompt with Qwen3-VL and the references with the two VAEs — so no `.safetensors` cache is needed (`--train_mode=ref2va` without `--enable_preprocess_training`).

---

## 3. PDD LoRA Training

### 3.1 Download Pretrained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download MiniMax-H3 official weights
hf download MiniMax-AI/MiniMax-H3 --local-dir models/Diffusion_Transformer/MiniMax-H3
```

> 💡 The loader accepts either the converted diffusers layout above or an *original* MiniMax-H3 partition (e.g. `MiniMax-H3/FL2VA`); the original shards are converted on the fly while loading, with no intermediate copy on disk.

### 3.2 Quick Start (FSDP)

If you have generated a conditioning cache as per **2.3** and downloaded the weights as per **3.1**, you can copy and run the command below. `scripts/minimax_h3/train_pdd_lora.sh` is the same launch.

FSDP is recommended: even though PDD does not load Qwen3-VL, the frozen transformer is still about 62 GB in bfloat16 and must be sharded — which FSDP (`FULL_SHARD`) does but DeepSpeed-Zero-2 does not.

`--mixed_precision=no` is required. The released checkpoint already pins `proj_out` / `audio_proj_out` in float32 (`_keep_in_fp32_modules`); the parallel heads built from them stay float32 master weights over a bfloat16 backbone, and the run does not use autocast.

**fl2va (from a prompt cache — the default recipe):**

```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATA_DIR=""
export PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export VAL_PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_preprocess_training \
  --train_data_dir=$DATA_DIR \
  --train_data_meta=$PROMPT_CACHE_META \
  --val_data_meta=$VAL_PROMPT_CACHE_META \
  --video_sample_n_frames=124 \
  --fix_sample_size 768 1344 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --checkpointing_steps=200 \
  --learning_rate=1e-5 \
  --lora_learning_rate=1e-4 \
  --seed=43 \
  --output_dir="output_dir_minimax_h3_pdd_lora" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="no" \
  --adam_weight_decay=0.0 \
  --max_grad_norm=1.0 \
  --rank=64 \
  --network_alpha=64 \
  --low_vram \
  --target_name="to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear" \
  --train_mode="fl2va" \
  --pdd_num_steps=32 \
  --pdd_block_size=4 \
  --validation_steps=200 \
  --resume_from_checkpoint=latest
```

#### 3.2.1 Ref2VA Training

`--train_mode=ref2va` conditions on reference media (images / videos / audio) in addition to the prompt, and loads `transformer_ref` by default. It runs on either conditioning route; both commands below are the FSDP Quick Start with only the mode / data flags changed.

**Route A — direct load (no request cache).** Point `--train_data_meta` straight at a request annotation: either the explicit `{"prompt", "references"}` schema or the official `X-Fun-Videos-Audios-Demo` annotation, whose own video + audio become the references (see **2.4**). The run loads the Qwen3-VL conditioner and the two VAEs and encodes every request on the fly, so keep `--low_vram` — they are onloaded only while encoding.

```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export REQUEST_META="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_mode="ref2va" \
  --train_data_meta=$REQUEST_META \
  --val_data_meta=$REQUEST_META \
  --video_sample_n_frames=124 \
  --fix_sample_size 768 1344 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --checkpointing_steps=200 \
  --learning_rate=1e-5 \
  --lora_learning_rate=1e-4 \
  --seed=43 \
  --output_dir="output_dir_minimax_h3_pdd_ref2va_lora" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="no" \
  --adam_weight_decay=0.0 \
  --max_grad_norm=1.0 \
  --rank=64 \
  --network_alpha=64 \
  --low_vram \
  --target_name="to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear" \
  --pdd_num_steps=32 \
  --pdd_block_size=4 \
  --validation_steps=200 \
  --resume_from_checkpoint=latest
```

> ⚠️ Route A keeps the ~62 GB Qwen3-VL conditioner resident in the run (sharded by FSDP alongside the transformer) and VAE-encodes each request's references on every trajectory reset. For long or repeated runs, pre-encode once with Route B so training never loads the conditioner.

**Route B — request cache (recommended for long / repeated runs).** Pre-encode the requests once with `generate_ref2va_request_cache.sh` (**2.3**), then train with `--enable_preprocess_training`; the ~62 GB conditioner stays out of the run:

```bash
# 1) Pre-encode the ref2va requests once (multi-GPU): prompt embeds + reference latents → safetensors
bash scripts/minimax_h3/generate_ref2va_request_cache.sh

# 2) Train off the cache — Route A with these flags changed
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export REQUEST_CACHE_META="datasets/minimax_h3_pdd_request_cache/outputs.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_mode="ref2va" \
  --enable_preprocess_training \
  --train_data_meta=$REQUEST_CACHE_META \
  --val_data_meta=$REQUEST_CACHE_META \
  ... # the remaining arguments identical to Route A
```

### 3.3 PDD Training Parameters

**PDD / LoRA parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|----------------|
| `--pdd_num_steps` | Grid size `N` | 32 |
| `--pdd_block_size` | `L_min`: intervals the carried state advances by (`NFE = N / L`) | 4 |
| `--pdd_max_block_size` | `L_max`: widest block a loss target is drawn from. Defaults to `--pdd_block_size` | 4 |
| `--pdd_solver` | Runge-Kutta method for the teacher's mean velocity: `euler` or `midpoint` | `midpoint` |
| `--pdd_num_targets` | How many intra-block indices `k` one student evaluation is supervised at | 2 |
| `--rank` | Dimension of LoRA update matrices | 64 |
| `--network_alpha` | Scale of LoRA update matrices | 64 |
| `--target_name` | Modules to apply LoRA (comma-separated) | `to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear` |
| `--learning_rate` | Learning rate of the parallel heads | 1e-5 |
| `--lora_learning_rate` | Learning rate of the low-rank updates | 1e-4 |
| `--use_ema` | Keep an EMA of the trainable set; validation and `pdd_ema.safetensors` use it | off |
| `--ema_decay` | EMA decay | 0.99 |

**Common training parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|----------------|
| `--pretrained_model_name_or_path` | Path to pretrained model | `models/Diffusion_Transformer/MiniMax-H3` |
| `--enable_preprocess_training` | Train on the pre-processed safetensors cache instead of encoding the conditioning on the fly; both `fl2va` and `ref2va` support either route | on |
| `--train_data_dir` | Optional root prepended to each `file_path` of `--train_data_meta`; empty when it stores repo-relative/absolute paths | `""` |
| `--train_data_meta` | With the flag: the cache `outputs.json`. Without it: the on-the-fly annotation (`fl2va`: `{"text": ...}`; `ref2va`: the request list) | `datasets/minimax_h3_pdd_prompt_cache/outputs.json` |
| `--val_data_meta` | Mirrors `--train_data_meta`: with the flag the cache `outputs.json`, without it the on-the-fly annotation (`fl2va`: `{"text": ...}`; `ref2va`: the request list). Skipped when empty | `datasets/minimax_h3_pdd_prompt_cache/outputs.json` |
| `--train_mode` | `fl2va` (t2va layout) or `ref2va` (`transformer_ref` + reference media); both run on the cache or the direct-load route | `fl2va` |
| `--transformer_subfolder` | Transformer subfolder. Default: `transformer_ref` for `ref2va`, else `transformer` | None |
| `--train_batch_size` | Must be 1: each rank carries one trajectory | 1 |
| `--num_train_epochs` | Training epochs when `--max_train_steps` is omitted. One epoch is one pass through the conditioning set | 100 |
| `--max_train_steps` | Total optimization steps. If set, overrides `--num_train_epochs` | 3000 |
| `--video_sample_n_frames` | Number of frames, must follow the `17*n+5` form of the video VAE (duration stays between 5 and 15 seconds) | 124 |
| `--video_sample_size` | Square canvas size (height = width); must be a multiple of 32 | 1280 |
| `--fix_sample_size` | Fixed `[height, width]` overriding `--video_sample_size` for a non-square canvas; both must be multiples of 32 | 768 1344 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 1 |
| `--checkpointing_steps` | Save checkpoint every N steps | 200 |
| `--seed` | Random seed | 43 |
| `--output_dir` | Output directory | `output_dir_minimax_h3_pdd_lora` |
| `--gradient_checkpointing` | Enable activation checkpointing | - |
| `--gradient_checkpointing_save_on_cpu` | Offload the activations saved for backward of the transformer blocks to CPU memory | - |
| `--mixed_precision` | Use `no`. The parallel heads stay float32 over a bfloat16 backbone | `no` |
| `--adam_weight_decay` | AdamW weight decay | 0.0 |
| `--max_grad_norm` | Gradient clipping threshold | 1.0 |
| `--low_vram` | Keep the VAEs on CPU; they move to GPU only inside validation decode | - |
| `--resume_from_checkpoint` | Resume from a checkpoint path, or `"latest"` to auto-select | `latest` |
| `--validation_steps` | Run validation every N steps | 200 |
| `--validation_nfe` | Student NFE during validation; must divide `--pdd_num_steps` | 8 |
| `--video_loss_weight` / `--audio_loss_weight` | Weights of the joint video + audio MSE | 0.5 / 0.5 |

### 3.4 Training Validation

Validation does **not** take `--validation_prompts`. It generates every entry of `--val_data_meta`, sharded across ranks, at `--validation_nfe`: the cache `outputs.json` under `--enable_preprocess_training`, or the on-the-fly annotation without it — mirroring the training route for both `fl2va` (prompts) and `ref2va` (requests). Validation is skipped when `--val_data_meta` is empty.

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--validation_steps` | Validate every N steps | 200 |
| `--validation_nfe` | Student evaluations per clip (`N / NFE` must be an integer) | 8 |

Videos are saved under `output_dir/sample/` as `sample-{step}-prompt{index}-{train_mode}-nfe{nfe}.mp4` (with audio).

### 3.5 Checkpoint Layout

Each `checkpoint-{step}/` holds:

| File | Role |
|------|------|
| `pdd.safetensors` | Live (non-EMA) gathered trainable tensors (parallel heads + LoRA). Used to resume DDP |
| `pdd_ema.safetensors` | EMA export when `--use_ema` is on; this is the inference file |
| `pdd_config.json` | Rank / alpha / targets / grid, read by `examples/minimax_h3/predict_t2v.py` |
| `optimizer.pt` / `scheduler.pt` / `scaler.pt` / `ema.pt` | DDP trainer state (`optimizer.bin` / `scheduler.bin` from Accelerate `--save_state` are also accepted on resume) |

FSDP stage 3 / ZeRO-3 also write `accelerator.save_state` into the same folder (auto `--save_state`) and still export a gathered `pdd.safetensors` (live) plus `pdd_ema.safetensors` when EMA is on. Checkpoints written before the rename stored live weights in `pdd_live.safetensors`; DDP resume still loads that file when it is present.

### 3.6 Training with DeepSpeed-Zero-2

> ⚠️ **Warning**: DeepSpeed-Zero-2 only partitions optimizer states and gradients, **not the model weights**. The MiniMax-H3 transformer is about 62 GB, so each GPU still holds a full weight replica and this setup usually runs out of memory. Prefer FSDP (**3.2**); the command below is provided for reference only.

```sh
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATA_DIR=""
export PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export VAL_PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_preprocess_training \
  --train_data_dir=$DATA_DIR \
  --train_data_meta=$PROMPT_CACHE_META \
  --val_data_meta=$VAL_PROMPT_CACHE_META \
  ... # the same train_pdd_lora.py arguments as the Quick Start
```

### 3.7 Training Without DeepSpeed or FSDP

**This approach is not recommended on 80 GB cards**: every GPU keeps a full ~62 GB transformer replica. PDD does not load Qwen3-VL, so DDP is lighter than `scripts/minimax_h3/train_lora.py`, but FSDP (**3.2**) is still the default. Drop `--use_fsdp` and the FSDP wrap flags from the Quick Start command; DDP resume reads `pdd.safetensors` plus `optimizer.pt` / `optimizer.bin`.

```sh
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATA_DIR=""
export PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export VAL_PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_preprocess_training \
  --train_data_dir=$DATA_DIR \
  --train_data_meta=$PROMPT_CACHE_META \
  --val_data_meta=$VAL_PROMPT_CACHE_META \
  ... # the same train_pdd_lora.py arguments as the Quick Start
```

### 3.8 Multi-Machine Distributed Training

**Suitable for**: more GPUs, faster training

#### 3.8.1 Environment Configuration

Assuming 2 machines with 8 GPUs each:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATA_DIR=""
export PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export VAL_PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_preprocess_training \
  --train_data_dir=$DATA_DIR \
  --train_data_meta=$PROMPT_CACHE_META \
  --val_data_meta=$VAL_PROMPT_CACHE_META \
  ... # the same train_pdd_lora.py arguments as the Quick Start
```

**Machine 1 (Worker)**:
```bash
export RANK=1  # Note this is 1
# Other environment variables identical to Machine 0

# Use the same accelerate launch command as Machine 0
```

#### 3.8.2 Multi-Machine Training Notes

- **Network Requirements**:
   - RDMA/InfiniBand recommended (high performance)
   - Without RDMA, add environment variables:
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **Data Synchronization**: All machines must be able to access the same conditioning cache and model paths (NFS/shared storage)

## 4. Inference Testing

PDD inference attaches the parallel heads and LoRA from `pdd_ema.safetensors` (falling back to `pdd.safetensors` when EMA was not saved), then samples at `num_inference_steps` NFE. Use `examples/minimax_h3/predict_t2v.py`; do not set `lora_path` at the same time (`lora_path` and `pdd_lora_path` cannot be used together).

The default recipe (`N = 32`, `L = 4`) runs at **8** inference steps. `num_inference_steps` must divide `pdd_num_steps` from `pdd_config.json`. If it is left at the teacher default of 40, the script snaps it to `N / L`.

### 4.1 Inference Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|------|------|-------|
| `GPU_memory_mode` | GPU memory mode, see table below for options | `model_cpu_offload` |
| `ulysses_degree` | Head dimension parallelization degree, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelization degree, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP for Transformer in multi-GPU inference to save VRAM | `False` |
| `fsdp_text_encoder` | Use FSDP for the Qwen3-VL text encoder in multi-GPU inference to save VRAM | `False` |
| `compile_dit` | Compile Transformer to accelerate inference (effective at fixed resolution) | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/MiniMax-H3` |
| `transformer_path` | Path to trained Transformer weights | `None` |
| `vae_path` | Path to trained VAE weights | `None` |
| `pdd_lora_path` | PDD checkpoint directory (loads `pdd_ema.safetensors` if present, else `pdd.safetensors`, plus `pdd_config.json`) or a `.safetensors` file | `output_dir_minimax_h3_pdd_lora/checkpoint-3000` |
| `lora_path` | Turbo/PEFT LoRA; **cannot** be combined with `pdd_lora_path` | `None` |
| `sample_size` | Generated video resolution `[height, width]`; height/width must be multiples of 32. `None` uses MiniMax-H3's own 16:9 canvas (768x1344) | `[768, 1344]` |
| `video_length` | Number of frames to generate, snapped up to the next `17*n+5` the video VAE can decode (duration stays between 5 and 15 seconds) | 124 |
| `fps` | Frames per second (MiniMax-H3 generates at a fixed 24 fps) | 24 |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 support | `torch.bfloat16` |
| `prompt` | Positive prompt describing the content to generate | `"A red fox trotting..."` |
| `seed` | Random seed for reproducibility | 43 |
| `num_inference_steps` | Student NFE. Default PDD recipe uses 8 (not the teacher's 40) | 8 |
| `guidance_scale` | Guidance strength. The released checkpoint is guidance-distilled: keep it at 1 to run one forward pass per step with no CFG | 1 |
| `flow_shift` | Exponential sigma shift of the video schedule, `None` keeps the one of the checkpoint (12.0) | `None` |
| `audio_flow_shift` | Exponential sigma shift of the audio schedule, `None` keeps the one of the checkpoint (3.0) | `None` |
| `save_path` | Generated video save path | `samples/minimax-h3-videos-t2v` |

**GPU Memory Mode Description**:

| Mode | Description | VRAM Usage |
|------|------|---------|
| `model_full_load` | Load entire model to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer group offload between CPU/CUDA | Low |
| `sequential_cpu_offload` | Offload each layer individually (slowest) | Lowest |

> 💡 The transformer alone is 61.7 GB in bfloat16 and the Qwen3-VL conditioner is another 62.1 GB, so a single 80 GB card needs `model_cpu_offload` or `model_group_offload`. Inference *does* load the text encoder; training does not.

### 4.2 Single GPU Inference

Run single GPU inference with:

```bash
python examples/minimax_h3/predict_t2v.py
```

Edit `examples/minimax_h3/predict_t2v.py` according to your needs. For PDD inference, focus on these parameters:

```python
# Choose based on your GPU VRAM
GPU_memory_mode = "model_cpu_offload"
# Your actual model path
model_name = "models/Diffusion_Transformer/MiniMax-H3"
# PDD checkpoint directory or weights file; a directory prefers pdd_ema.safetensors. Rank / alpha / targets / grid are read from pdd_config.json
pdd_lora_path = "output_dir_minimax_h3_pdd_lora/checkpoint-3000"
# Must stay None when pdd_lora_path is set
lora_path = None
# Student NFE; 8 for the default N=32 / L=4 recipe. Left at 40, the script snaps to N / L
num_inference_steps = 8
# Write based on content to generate
prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"
# ...
```

Image-to-video and Ref2VA use the same `pdd_lora_path` field in `examples/minimax_h3/predict_i2v.py` and `examples/minimax_h3/predict_ref2va.py`. Ref2VA needs a checkpoint trained with `--train_mode=ref2va`.

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser yunchang
```

#### Configure Parallel Strategy

Edit `examples/minimax_h3/predict_t2v.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelization
ring_degree = 1     # Sequence dimension parallelization
```

**Configuration Principles**:
- `ulysses_degree` must evenly divide the model's number of heads
- `ring_degree` splits on sequence dimension, affecting communication overhead; avoid using it when heads can be divided
- Multi-GPU runs through the xfuser sequence-parallel path and is **incompatible with the `*cpu_offload*` memory modes** (accelerate offload hooks own a single device); use `model_full_load` / `model_full_load_and_qfloat8` across GPUs there, with `fsdp_dit` / `fsdp_text_encoder` to save memory
- Sequence parallel needs a working FlashAttention. Without it, run independent single-GPU jobs (`CUDA_VISIBLE_DEVICES=i`, `ulysses_degree = 1`, `ring_degree = 1`) instead

**Example Configurations**:

| GPU Count | ulysses_degree | ring_degree | Description |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelization |
| 8 | 2 | 4 | Hybrid parallelization |
| 8 | 8 | 1 | Head parallelization |

#### Run Multi-GPU Inference

```bash
torchrun --nproc_per_node=2 examples/minimax_h3/predict_t2v.py
```

## 5. Additional Resources

- **PDD paper**: https://arxiv.org/abs/2607.26004
- **MiniMax-H3 Official GitHub**: https://github.com/MiniMax-AI/MiniMax-H3
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
- **Base MiniMax-H3 LoRA training**: `scripts/minimax_h3/README_TRAIN_LORA.md`
- **MiniMax-H3 Fun control training**: `scripts/minimax_h3_fun/README_TRAIN.md`
