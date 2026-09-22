# Wan2.1 Flex-Forcing Stage 1: ODE Regression Training Guide

This document provides the complete workflow for **Flex-Forcing Stage 1 — ODE Regression** ([arXiv 2607.03509](https://arxiv.org/abs/2607.03509)) on Wan2.1-T2V-1.3B, including environment setup, ODE trajectory pair generation, and ODE regression training.

> **What is ODE Regression Training?**
>
> ODE regression is the **pre-training stage** of the two-stage Flex-Forcing pipeline. The pipeline is:
>
> 1. **Step 1 — Generate ODE pairs** (`generate_ode_pairs.py`): Use the **bidirectional teacher** (Wan2.1-T2V-1.3B) to perform full multi-step CFG denoising on a list of text prompts. Save the intermediate latents along the ODE trajectory together with the encoded prompt embeddings as `.safetensors` files.
> 2. **Step 2 — Train ODE regression** (`train_ode.py`): Load the generated ODE pairs and train a **flexible-chunking generator** to predict the clean endpoint `x0` at multiple sampled trajectory points, drawing a fresh random frame partition every iteration. The output checkpoint serves as the few-step initialization for the subsequent **Stage 2 DMD distillation** (`train_distill.py`, see [README_TRAIN_FLEX_DMD_DISTILL.md](./README_TRAIN_FLEX_DMD_DISTILL.md)).
>
> Flex-Forcing itself replaces Self-Forcing's scalar `num_frame_per_block` with a **partition of the frame axis** `a = (a_0, ..., a_K)`: attention is bidirectional *inside* a chunk and autoregressive *across* chunks. `[1] * F` recovers pure autoregression, `[F]` recovers pure bidirectional attention, and everything in between is covered by one model. The random partitions are first seen in this stage.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Download Pretrained Models](#2-download-pretrained-models)
- [3. Step 1 — Generate ODE Trajectory Pairs](#3-step-1--generate-ode-trajectory-pairs)
  - [3.1 Download Prompt File](#31-download-prompt-file)
  - [3.2 Run ODE Pair Generation](#32-run-ode-pair-generation)
  - [3.3 Output Format](#33-output-format)
  - [3.4 Generation Parameters](#34-generation-parameters)
  - [3.5 Multi-GPU Generation](#35-multi-gpu-generation)
- [4. Step 2 — Train ODE Regression](#4-step-2--train-ode-regression)
  - [4.1 Quick Start](#41-quick-start)
  - [4.2 Common Training Parameters](#42-common-training-parameters)
  - [4.3 Flex-Forcing Parameters](#43-flex-forcing-parameters)
  - [4.4 Training with DeepSpeed-Zero-2 / FSDP](#44-training-with-deepspeed-zero-2--fsdp)
  - [4.5 Multi-Node Distributed Training](#45-multi-node-distributed-training)
- [5. Use the Trained ODE Weights](#5-use-the-trained-ode-weights)
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

When using Docker, please ensure that your machine has correctly installed GPU drivers and CUDA environment, then execute the following commands:

```bash
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

> The flexible attention masks are built with **FlexAttention** (`torch.nn.attention.flex_attention`). That dependency is inherited from the Self-Forcing backbone rather than added here, so nothing new to install — but it does put `torch>=2.5` as the practical floor.

---

## 2. Download Pretrained Models

Stage 1 uses a single **Wan2.1-T2V-1.3B** model: it is both the student to train and the bidirectional teacher whose ODE trajectories are generated in Step 1.

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download Wan2.1 T2V 1.3B (teacher for generation, init for training)
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B
```

Then point `MODEL_NAME` in `train_ode.sh` at the 1.3B directory.

> Loading a plain Self-Forcing / CausVid checkpoint into `WanTransformer3DModel_FlexForcing` reports the `flex_kproj.*` keys as **missing**. That is expected: the K-Projection (§3.3 of the paper, a cache-alignment projection) is identity-initialised, so a checkpoint that predates it behaves exactly like the identity until it is trained.

---

## 3. Step 1 — Generate ODE Trajectory Pairs

The trajectories are a property of the *teacher*, not of the chunking, so this step is shared with the Self-Forcing pipeline as-is: it uses the bidirectional teacher to run **48-step CFG denoising** on each prompt and saves the resulting ODE trajectory together with the prompt embedding into a `.safetensors` file. After all prompts are processed, an `outputs.json` annotation file is produced for the training stage to consume.

### 3.1 Download Prompt File

The official Self-Forcing prompt list is recommended:

```bash
mkdir -p datasets

# Download vidprom_filtered_extended.txt from the official Self-Forcing repo
hf download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir datasets/
# Final path: datasets/vidprom_filtered_extended.txt
```

You can also use any plain-text file with one prompt per line.

### 3.2 Run ODE Pair Generation

The ready-to-use launcher is [scripts/wan2.1_self_forcing/generate_ode_pairs.sh](../wan2.1_self_forcing/generate_ode_pairs.sh):

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.1_self_forcing/generate_ode_pairs.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --video_sample_n_frames=81 \
  --height=480 \
  --width=832 \
  --guidance_scale=6.0 \
  --shift=8.0 \
  --num_inference_steps=48 \
  --caption_path="datasets/vidprom_filtered_extended.txt" \
  --output_folder="datasets/ode_pairs_output" \
  --sample_every_n_prompts=50
```

Or simply run the shell script:

```bash
bash scripts/wan2.1_self_forcing/generate_ode_pairs.sh
```

### 3.3 Output Format

After generation, `--output_folder` will contain:

```
📦 datasets/ode_pairs_output/
├── 📄 00000.safetensors    # Per-prompt ODE trajectory + prompt embeds
├── 📄 00001.safetensors
├── 📄 ...
├── 📂 sample/              # Optional preview videos (when sample_every_n_prompts > 0)
│   └── 📄 00000_clean.mp4
└── 📄 outputs.json         # Annotation file consumed by train_ode.py
```

Each `.safetensors` file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `latents` | `[5, C, F, H, W]` | Sparse 5-point sampling of the 48-step ODE trajectory: indices `[0, 12, 24, 36, -1]` (initial noise → 3 mid-points → clean endpoint) |
| `prompt_embeds` | `[512, D]` | Padded T5 prompt embeddings (max length 512) |
| `prompt_attention_mask` | `[512]` | Attention mask for the prompt embeddings |

The auto-generated `outputs.json` follows the same format as a standard `metadata.json`:

```json
[
  { "file_path": "datasets/ode_pairs_output/00000.safetensors" },
  { "file_path": "datasets/ode_pairs_output/00001.safetensors" }
]
```

### 3.4 Generation Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--pretrained_model_name_or_path` | Path to Wan2.1-T2V-1.3B teacher | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `--config_path` | Model config YAML | `config/wan2.1/wan_civitai.yaml` |
| `--caption_path` | Plain-text file with one prompt per line | `datasets/vidprom_filtered_extended.txt` |
| `--output_folder` | Output directory for `.safetensors` files and `outputs.json` | `datasets/ode_pairs_output` |
| `--guidance_scale` | CFG guidance scale used by the teacher | 6.0 |
| `--num_inference_steps` | Number of teacher denoising steps (must be ≥ 37 because indices `[0,12,24,36,-1]` are sampled) | 48 |
| `--shift` | Shift value for `FlowMatchEulerDiscreteScheduler` (must match training!) | 8.0 |
| `--video_sample_n_frames` | Pixel frame count of generated video | 81 |
| `--height` / `--width` | Video resolution in pixels | 480 / 832 |
| `--negative_prompt` | Negative prompt used for CFG | (Chinese default) |
| `--sample_every_n_prompts` | Decode and save preview MP4 every N prompts (0 to disable) | 50 |
| `--mixed_precision` | `no` / `fp16` / `bf16` | `bf16` |

> ⚠️ Keep `--shift` **identical** between generation and ODE training. Both default to `8.0` in the provided scripts.

### 3.5 Multi-GPU Generation

`generate_ode_pairs.py` is built on `accelerate`. Each rank automatically processes an interleaved subset of prompts (`prompt_index = index * world_size + rank`) and skips already-existing files, so the job is **resumable** and **parallelizable** out of the box:

```bash
# 8-GPU generation
accelerate launch --multi_gpu --num_processes=8 --mixed_precision="bf16" \
    scripts/wan2.1_self_forcing/generate_ode_pairs.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --config_path="config/wan2.1/wan_civitai.yaml" \
    --caption_path="datasets/vidprom_filtered_extended.txt" \
    --output_folder="datasets/ode_pairs_output" \
    --num_inference_steps=48 --guidance_scale=6.0 --shift=8.0 \
    --height=480 --width=832 --video_sample_n_frames=81
```

Only the main process writes the final `outputs.json`.

---

## 4. Step 2 — Train ODE Regression

After Step 1 completes and `datasets/ode_pairs_output/outputs.json` is generated, train the flexible-chunking generator (`WanTransformer3DModel_FlexForcing`) to regress the ODE trajectory.

For each training sample the script:
1. Loads the saved sparse trajectory (5 points) and prompt embedding from one `.safetensors` file.
2. Draws the frame-axis partition and gives **one timestep per chunk** (`get_timestep_for_ode_flexible`) — this is what makes a chunk internally bidirectional. 90% of iterations draw a **fresh random partition** (chunk sizes `--flex_chunk_min`..`--flex_chunk_max`); the other `UNIFORM_BLOCK_PROB = 0.1` pin it to the uniform `--num_frame_per_block` layout, which a random draw would reach with probability `1/183708` at 21 latent frames and which is exactly what `log_validation` renders.
3. Feeds the noisy latent and the per-frame timestep through the generator, converts the predicted flow into an `x0` prediction, and computes MSE loss against the **clean endpoint** of the trajectory.

### 4.1 Quick Start

The ready-to-use launcher is [train_ode.sh](./train_ode.sh):

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME=""
export ODE_DATA_META="datasets/ode_pairs_output/outputs.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/wan2.1_flex_forcing/train_ode.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$ODE_DATA_META \
  --fix_sample_size 432 832 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_ode_regression" \
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about." \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.05 \
  --train_sampling_steps=1000 \
  --denoising_step_indices_list 1000 750 500 250 \
  --shift=8.0 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --num_frame_per_block=3
```

Or simply:

```bash
bash scripts/wan2.1_flex_forcing/train_ode.sh
```

> 💡 Because the ODE trajectory and prompt embeddings are pre-computed in Step 1, **no VAE / text encoder is invoked during ODE training** — training is fast, memory-efficient, and `train_data_dir` can be left empty when `outputs.json` already contains absolute paths.

> Stage 1 carries `--flex_forcing` and `--flex_chunk_min/max`, but **no** `--flex_pyramid_levels`: it regresses `x0` in a single forward per block, so there is no multi-step schedule to ladder over. The pyramid timestep schedule arrives in Stage 2.

Output: `output_dir_wan2.1_flex_forcing_ode_regression/`.

### 4.2 Common Training Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--pretrained_model_name_or_path` | Base model used to initialize the generator (1.3B) | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `--config_path` | Model config YAML | `config/wan2.1/wan_civitai.yaml` |
| `--train_data_dir` | Optional root prepended to `file_path`; can be empty when `outputs.json` stores absolute paths | `""` |
| `--train_data_meta` | Annotation JSON produced by Step 1 | `datasets/ode_pairs_output/outputs.json` |
| `--fix_sample_size` | Fixed `H W`; the paper's 5 s clip is 432×832 | `432 832` |
| `--train_batch_size` | Per-GPU batch size | 1 |
| `--gradient_accumulation_steps` | 8 GPUs × 8 = the paper's batch 64 | 8 |
| `--dataloader_num_workers` | DataLoader workers | 8 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 500 |
| `--learning_rate` | Initial learning rate | 2.0e-06 |
| `--lr_scheduler` | LR scheduler type | `constant_with_warmup` |
| `--lr_warmup_steps` | LR warmup steps | 100 |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir_wan2.1_flex_forcing_ode_regression` |
| `--gradient_checkpointing` | Trade compute for memory | - |
| `--mixed_precision` | `fp16` / `bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon | 1e-10 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--trainable_modules` | Substring match for the trainable group (`"."` = all) | `"."` |
| `--resume_from_checkpoint` | `"latest"` to auto-select | `"latest"` |

**ODE-specific parameters** (must match Step 1 unless you understand the consequences):

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--train_sampling_steps` | Size of the scheduler grid that `denoising_step_indices_list` indexes into | 1000 |
| `--denoising_step_indices_list` | Discrete timestep indices used during ODE regression (corresponds to the 5 sparse points sampled in Step 1). Read off the scheduler's own grid rather than a `set_timesteps()` rebuild, so at `shift=8.0` they map to t = `[1000, 960, 888.89, 727.27]` - bit-identical to what `stochastic_sampling_timesteps` feeds validation and inference. The rebuild was off by up to `5.44`, because it ends its grid at a different `sigma_min`. | `1000 750 500 250` |
| `--shift` | Shift for `FlowMatchEulerDiscreteScheduler` — **must match `--shift` used in Step 1** | 8.0 |
| `--num_frame_per_block` | The uniform fallback for the inherited path; the partition `UNIFORM_BLOCK_PROB = 0.1` of iterations pin, so that layout is genuinely trained rather than merely reachable; and the *fixed* partition `log_validation` samples with so checkpoints stay comparable | 3 |
| `--independent_first_frame` | First frame is independent (`[1, N, N, ...]` block pattern) | - |
| `--context_noise` | Context noise level (matches downstream distillation config) | 0 |

**Validation Parameters (Optional)**:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--validation_steps` | Run validation every N steps | 100 |
| `--validation_epochs` | Run validation every N epochs | 500 |
| `--validation_prompts` | Prompts used for validation video generation | English prompt |

### 4.3 Flex-Forcing Parameters

| Parameter | Paper | Description | Default |
|-----------|-------|-------------|---------|
| `--flex_forcing` | §3.1 | Instantiate `WanTransformer3DModel_FlexForcing` and validate with `WanFlexForcingPipeline`. Off = the inherited path, unchanged. | off |
| `--flex_chunk_min` | §3.1 | Smallest chunk drawn per iteration. `1` is legal and lets single-frame (strictly causal) chunks appear mid-clip; `2` is the paper's and keeps the sampler's tail from degenerating into a sliver. | 2 |
| `--flex_chunk_max` | §3.1 | Largest chunk drawn per iteration. Set equal to `--flex_chunk_min` to pin one fixed layout. Unlike stage 2 there is no coarse mixture here, so raising this to the latent frame count is the **only** way stage 1 ever regresses the whole clip as one fully bidirectional chunk (about 5% of draws at 21 latent frames); below the frame count that layout is never drawn. | 10 |

> The §3.3 K-Projection has no launcher flag either: the variant comes from the model config (`transformer_additional_kwargs.flex_kproj_mode`, default `diag_rank1`; set `none` there to ablate). It is identity-initialised, so Stage 1 simply carries it and trains it like any other layer at `--learning_rate`.

> Every rank must train the same layout, otherwise the FlexAttention mask and the `num_frame_per_block` derived from it disagree across the SP/FSDP group. The trainer therefore passes the drawn partition through `broadcast_chunk_sizes` before using it.

### 4.4 Training with DeepSpeed-Zero-2 / FSDP

For multi-GPU training, the same memory-saving backends as the Self-Forcing stages are supported. The Quick Start above already uses **FSDP**; use **DeepSpeed-Zero-2** instead by swapping the launch prefix:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME=""
export ODE_DATA_META="datasets/ode_pairs_output/outputs.json"
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_flex_forcing/train_ode.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$ODE_DATA_META \
  --fix_sample_size 432 832 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_ode_regression" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.05 \
  --train_sampling_steps=1000 \
  --denoising_step_indices_list 1000 750 500 250 \
  --shift=8.0 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --num_frame_per_block=3
```

### 4.5 Multi-Node Distributed Training

Assuming 2 machines × 8 GPUs:

**Machine 0 (Master)**:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME=""
export ODE_DATA_META="datasets/ode_pairs_output/outputs.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Rank of this machine (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_flex_forcing/train_ode.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$ODE_DATA_META \
  --fix_sample_size 432 832 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_ode_regression" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.05 \
  --train_sampling_steps=1000 \
  --denoising_step_indices_list 1000 750 500 250 \
  --shift=8.0 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --num_frame_per_block=3
```

**Machine 1 (Worker)**: use the same command with `RANK=1`.

**Notes**:
- Use RDMA / InfiniBand whenever possible. Without RDMA, set `NCCL_IB_DISABLE=1` and `NCCL_P2P_DISABLE=1`.
- All machines must share the same `outputs.json` and the underlying `.safetensors` files (NFS / shared storage).

---

## 5. Use the Trained ODE Weights

Stage 1's output is the few-step initialization for Stage 2, not a final model. Copy the checkpoint path into `--ode_transformer_path` in [train_distill.sh](./train_distill.sh):

```bash
# Example: pick the saved weight file (e.g. diffusion_pytorch_model.safetensors)
--ode_transformer_path="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

Then continue with [README_TRAIN_FLEX_DMD_DISTILL.md](./README_TRAIN_FLEX_DMD_DISTILL.md).

---

## 6. Additional Resources

- **Flex-Forcing paper**: https://arxiv.org/abs/2607.03509
- **Stage 2 — DMD distillation**: [README_TRAIN_FLEX_DMD_DISTILL.md](./README_TRAIN_FLEX_DMD_DISTILL.md)
- **Shared ODE-pair generation**: [README_TRAIN_ODE.md](../wan2.1_self_forcing/README_TRAIN_ODE.md)
- **Shared chunking helpers**: `videox_fun/utils/flex_chunking.py`
- **Model / pipeline**: `videox_fun/models/wan_transformer3d_flex_forcing.py`, `videox_fun/pipeline/pipeline_wan_flex_forcing.py`
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
