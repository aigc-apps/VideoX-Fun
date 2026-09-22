# Wan2.1 Flex-Forcing Stage 2: DMD Distillation Training Guide

This document provides the complete workflow for **Flex-Forcing Stage 2 — Distribution Matching Distillation (DMD)** ([arXiv 2607.03509](https://arxiv.org/abs/2607.03509)) on Wan2.1-T2V-1.3B.

> **What is Flex-Forcing?**
>
> Self-Forcing fixes one scalar `num_frame_per_block` for the whole model, so a checkpoint is either chunkwise-causal or bidirectional and cannot be both. Flex-Forcing replaces that scalar with a **partition of the frame axis**, `a = (a_0, ..., a_K)`: attention is bidirectional *inside* a chunk and autoregressive *across* chunks. `[1] * F` recovers pure autoregression, `[F]` recovers pure bidirectional attention, and everything in between is one model.
>
> Three ideas make that work, and each maps onto a flag below:
>
> 1. **§3.1 Flexible frame chunking** — the partition is a first-class input rather than a build-time constant.
> 2. **§3.2 Pyramid timestep chunking** — after finishing step `t+1`, the partition for step `t` is *refined* by inserting boundaries while preserving the existing ones. The step-`t` result is buffered over the whole original chunk, then each sub-chunk resumes autoregressively once the KV it needs is available. Coarse planning first, fine detail later.
> 3. **§3.3 K-Projection** — clean-context keys live in a different space from noisy queries at timestep `t`, so an identity-initialised, timestep-conditioned projection maps the cache into the current noisy latent space. It is applied on the fly, never mutates the cache, and trains together with the generator.
>
> Training draws a **fresh random partition every iteration** (chunk sizes 2–10), which is what lets one set of weights cover the whole causal↔bidirectional spectrum. §4.2 of the paper then reuses the same machinery for **any-order editing**.
>
> The Flex-Forcing pipeline has two stages:
>
> 1. **Stage 1 — ODE Regression** (`train_ode.py`): regress the teacher's ODE trajectory so the model becomes a competent few-step generator. This is where the random partitions are first seen.
> 2. **Stage 2 — DMD Distillation** (`train_distill.py`, this README): compress to a **4-step** generator with a **Wan2.1-T2V-14B** real-score teacher, and add the pyramid timestep schedule.
>
> This README covers **Stage 2 only**. See [README_TRAIN_ODE.md](./README_TRAIN_ODE.md) for Stage 1.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 metadata.json Format](#22-metadatajson-format)
- [3. Distillation Training](#3-distillation-training)
  - [3.1 Download Pretrained Models](#31-download-pretrained-models)
  - [3.2 Quick Start (DeepSpeed-Zero-2)](#32-quick-start-deepspeed-zero-2)
  - [3.3 Common Training Parameters](#33-common-training-parameters)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training with FSDP](#35-training-with-fsdp)
  - [3.6 Other Backends](#36-other-backends)
  - [3.7 Multi-Node Distributed Training](#37-multi-node-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameters](#41-inference-parameters)
  - [4.2 Text-to-Video (T2V) Inference](#42-text-to-video-t2v-inference)
  - [4.3 Multi-GPU Parallel Inference](#43-multi-gpu-parallel-inference)
  - [4.4 Any-Order Video Editing](#44-any-order-video-editing)
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

**Method 3: Using Docker**

When using Docker, please ensure that your machine has correctly installed GPU drivers and CUDA environment, then execute the following commands:

```bash
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

> Both stages build their attention masks with **FlexAttention** (`torch.nn.attention.flex_attention`). That dependency is inherited from the Self-Forcing backbone rather than added here, so nothing new to install — but it does put `torch>=2.5` as the practical floor.

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset that contains several training data samples.

```bash
# Download official example dataset
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 2.2 metadata.json Format

DMD uses a **TextDataset** (`--train_mode="normal"`) — the generator creates its own training samples by rollout, so only the `"text"` field of each entry is read and any video metadata works as a prompt list:

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

`train_distill.sh` defaults to `datasets/internal_datasets/metadata.json`. For a real run, swap in the **VidProM extended** prompts the paper uses.

---

## 3. Distillation Training

### 3.1 Download Pretrained Models

Stage 2 needs **two** pretrained models:

- **Wan2.1-T2V-1.3B**: base model for the generator/critic.
- **Wan2.1-T2V-14B**: the non-causal real-score teacher DMD uses for the real distribution score.

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download Wan2.1 T2V 1.3B (student base model)
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B

# Download Wan2.1 T2V 14B (DMD real-score teacher)
modelscope download --model Wan-AI/Wan2.1-T2V-14B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-14B
```

Stage 2 also requires a **Stage 1 ODE checkpoint** to initialise the generator (and critic):

```bash
# Example: Stage 1 checkpoint from ODE regression
export STAGE1_CKPT="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

See [README_TRAIN_ODE.md](./README_TRAIN_ODE.md) for how to produce this checkpoint. Then point `MODEL_NAME` in `train_distill.sh` at the 1.3B directory.

> Loading a plain Self-Forcing / CausVid checkpoint into `WanTransformer3DModel_FlexForcing` reports the `flex_kproj.*` keys as **missing**. That is expected: §3.3's K-Projection is identity-initialised, so a checkpoint that predates it behaves exactly like the identity until it is trained.

### 3.2 Quick Start (DeepSpeed-Zero-2)

After downloading data according to **2.1 Quick Test Dataset** and downloading weights according to **3.1 Download Pretrained Models**, you can directly copy and run the quick start command.

We recommend using DeepSpeed-Zero-2 and FSDP for training. Here we use DeepSpeed-Zero-2 as an example to configure the shell file.

The difference between DeepSpeed-Zero-2 and FSDP lies in whether to shard model weights. **If you use multiple GPUs and encounter insufficient GPU memory with DeepSpeed-Zero-2**, you can switch to FSDP for training.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_flex_forcing/train_distill.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --fix_sample_size 480 832 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --score_num_frames=21 \
  --video_repeat=1 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_distill" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --train_mode="normal" \
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --flex_pyramid_levels=4 \
  --flex_min_num_frame_per_block=1 \
  --flex_self_generated_context \
  --denoising_step_indices_list 1000 750 500 250 \
  --randomize_step_indices \
  --flow_euler_rollout \
  --use_kv_cache_training \
  --num_frame_per_block=3 \
  --ode_transformer_path="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

The launcher pins: 81 pixel frames = **21 latent frames** (a 5 s clip) at **480×832**, a **4-step** model (`--denoising_step_indices_list 1000 750 500 250`, jittered per iteration by `--randomize_step_indices`), `--flex_pyramid_levels=4` - which over 21 latent frames reproduces the paper's ladder `[[21], [11, 10], [6, 5, 5, 5], [3, 3, 3, 2, 3, 2, 3, 2]]` - partitions drawn in **2..10** (plus, on half the iterations, the whole-clip coarse layout of §3.2 - see below), `--flow_euler_rollout`, `diag_rank1` K-Projection, batch 8 (8 GPUs × `--gradient_accumulation_steps=1`).

Output: `output_dir_wan2.1_flex_forcing_distill/`.

### 3.3 Common Training Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config_path` | Component subpaths / `transformer_additional_kwargs` | `config/wan2.1/wan_civitai.yaml` |
| `--pretrained_model_name_or_path` | Base model directory (1.3B) | `$MODEL_NAME` |
| `--train_data_meta` | Prompt JSON (only the `"text"` field is read) | `$DATASET_META_NAME` |
| `--ode_transformer_path` | Stage 1 ODE checkpoint for generator/critic init | `$STAGE1_CKPT` |
| `--fix_sample_size` | Fixed `H W` of the training clip | `480 832` |
| `--video_sample_n_frames` | Pixel frames; 81 → 21 latent frames | 81 |
| `--score_num_frames` | Latent frames the real-score teacher sees | 21 |
| `--train_batch_size` | Per-GPU batch size | 1 |
| `--gradient_accumulation_steps` | 8 GPUs × 1 = batch 8 | 1 |
| `--dataloader_num_workers` | DataLoader workers | 8 |
| `--num_train_epochs` | Number of training epochs (overridden by `--max_train_steps`) | 100 |
| `--max_train_steps` | Hard step cap; the paper uses 600 | 600 |
| `--checkpointing_steps` | Save checkpoint every N steps | 50 |
| `--learning_rate` | Generator learning rate | `2e-06` |
| `--learning_rate_critic` | Critic learning rate | `2e-06` |
| `--lr_scheduler` | LR scheduler type | `constant_with_warmup` |
| `--lr_warmup_steps` | LR warmup steps | 100 |
| `--real_guidance_scale` | CFG scale for the real-score (14B teacher) | 4.5 |
| `--fake_guidance_scale` | CFG scale for the fake-score (generator). 0.0 = no CFG | 0.0 |
| `--gen_update_interval` | Generator updates every N critic steps | 5 |
| `--denoising_step_indices_list` | Denoising step indices (DMD core param). Mapped through `timesteps[train_sampling_steps - idx]`, so `1000 750 500 250` becomes t = `[1000, 937.5, 833.33, 625]` at `shift=5.0` - bit-identical to what `stochastic_sampling_timesteps(4, shift)` feeds the pipeline. | `1000 750 500 250` |
| `--randomize_step_indices` | Jitter every index but the first symmetrically within `--index_jitter_ratio` of its neighbouring gap each iteration. The expected schedule is unchanged and monotonicity is enforced by construction, so inference's fixed t stays inside every step's trained envelope. | off |
| `--index_jitter_ratio` | Jitter budget as a fraction of the neighbouring gap. At `1000 750 500 250` with 0.3, 20 000 draws give idx `[675,825] / [425,575] / [175,325]`, i.e. t `[912.2,959.3] / [787.0,871.2] / [514.7,706.5]`; the inference points 937.5 / 833.33 / 625 all fall inside. | 0.3 |
| `--flow_euler_rollout` | Advance the generator's self-rollout with a deterministic Euler ODE step in flow space (`x_next = x_t + (sigma_next - sigma_t) * v`, fp32) instead of converting to x0 and re-noising with fresh noise. Algebraically this *is* re-noising with the same implied noise - measured `8.9e-16` apart in fp64, versus up to `6.4` for a fresh draw - so it removes the rollout's stochasticity and matches normal flow-matching inference. The final step still converts to x0, which is what DMD is defined on. The critic re-rolls the generator and follows the same switch. | off |
| `--output_dir` | Output directory | `output_dir_wan2.1_flex_forcing_distill` |
| `--gradient_checkpointing` | Trade compute for memory | - |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--trainable_modules` | Substring match for the trainable group (`"."` = all) | `"."` |
| `--train_mode` | `normal` (TextDataset, prompt-only) | `normal` |
| `--resume_from_checkpoint` | `"latest"` to auto-select | `"latest"` |

**Flex-Forcing-Specific Parameters**:

| Parameter | Paper | Description | Default |
|-----------|-------|-------------|---------|
| `--flex_forcing` | §3.1 | Instantiate `WanTransformer3DModel_FlexForcing` and validate with `WanFlexForcingPipeline`. Off = the inherited path, unchanged. | off |
| `--flex_chunk_min` | §3.1 | Smallest chunk drawn per iteration. `1` is legal but adds no coverage under a pyramid - refining a 2-frame chunk already yields 1-frame leaves - and it would let single-frame chunks appear mid-clip. | 2 |
| `--flex_chunk_max` | §3.1 | Largest chunk drawn per iteration. Set equal to `--flex_chunk_min` to pin one fixed layout. **Keep it below the latent frame count**: the whole-clip layout is already covered by the mixture below, so raising this to the frame count only spends draws on a layout you already have and thins out §3.1 (at 21 latent frames, `max=21` leaves 3649 distinct partitions versus 4882 at `max=10`). | 10 |
| `--flex_pyramid_levels` | §3.2 | `1` = one partition per iteration. `>1` = nest it into a coarse-to-fine ladder, one level per denoising step; both training paths roll it out. **Stage 2 only.** | 1 |
| `--flex_min_num_frame_per_block` | §3.2 | Block size the refinement stops at; `1` is fully causal at the leaves. | 1 |
| `--flex_self_generated_context` | §3.3 | Feeds the K-Projection the model's own previous-step `x0` prediction as context, so it enters the autograd graph under prompt-only training. Costs a doubled attention sequence; from denoising step 1 on. Inert under `--use_kv_cache_training` - that path passes no `flex_state`, so the K-Projection is never invoked and stays at its identity init. | off |
| `--num_frame_per_block` | — | The uniform fallback for the inherited (non-Flex) path; with Flex on it is also one of the three level-0 bands (`UNIFORM_BLOCK_PROB = 0.1` of iterations pin level 0 to its uniform partition), and it is what `log_validation` samples with at `--flex_pyramid_levels=1` - see the two bullets below. | 3 |
| `--independent_first_frame` | — | Prepend a length-1 chunk (`[1, N, N, ...]`); honoured by the partition sampler. | off |

Seven interactions are worth stating explicitly:

- **Level 0 is a three-band mixture decided by one draw.** `denoise_mode="pyramid"` at inference builds its level 0 as the whole clip in one chunk (fully bidirectional planning), so a partition drawn only from 2..10 would leave that step untrained - and the uniform `--num_frame_per_block` layout is just as unreachable by chance alone (at 21 latent frames `[3]*7` has probability `1/183708`, i.e. 0.11 expected hits in 20 000 iterations). `train_distill.py` therefore splits level 0 with a single uniform draw into three bands, neither constant a CLI flag: `COARSE_GLOBAL_PROB = 0.5` for the whole clip as one chunk, `UNIFORM_BLOCK_PROB = 0.1` pinned to the uniform `--num_frame_per_block` partition, and the remaining `0.4` for §3.1's random draw; all three refine into the same ladder. Measured over 200 000 iterations at 21 latent frames with `--flex_pyramid_levels=4`: whole clip `49.999%`, `[3]*7` `10.034%`, random `39.968%`, **4078** distinct level-0 layouts, and the inference ladder `[[21], [11, 10], [6, 5, 5, 5], [3, 3, 3, 2, 3, 2, 3, 2]]` at `49.999%`. At `--flex_pyramid_levels=1` the whole-clip band is empty and the split becomes `10% / 90%`, which keeps that mode as the paper describes it. Peak memory is set by the densest case (level 0 = whole clip), so budget for it.
- The ladder is **clamped to the number of trained denoising steps**. A 4-level ladder under a 2-step schedule can never reach its last two levels, so `sample_flex_partitions` drops the unreachable tail rather than silently reporting it as trained. Keep `len(--denoising_step_indices_list) >= --flex_pyramid_levels`; the launcher's 4 indices and `--flex_pyramid_levels=4` are matched on purpose. The converse also wastes compute: more steps than levels leaves `install_flex_partition` clamped on its finest level for the extra steps.
- **`--flow_euler_rollout` keeps the prediction in flow space between steps, so §3.3's context needs its own x0.** `--flex_self_generated_context` feeds the K-Projection the previous step's *clean* prediction; under an Euler rollout that variable is a velocity, so both rollout loops convert a detached copy for the context while the rollout state itself advances by Euler. The generator and the critic convert identically, otherwise the critic would score a rollout that is not the one being trained.
- **`--randomize_step_indices` does not break the pyramid.** The jitter moves the *timesteps*, not the partitions, and `sample_flex_partitions` keys off `len(denoising_step_list)`, which the jitter preserves. Every level of the ladder still lines up with its denoising step.
- **`log_validation` renders the layout training renders.** With `--flex_pyramid_levels > 1` it passes `chunk_spec=None` + `denoise_mode="pyramid"`, so its ladder is identical to `predict_t2v.py`'s and measured in `49.999%` of iterations. It used to pin `chunk_spec` to `--num_frame_per_block`, which nested `[[3]*7, [2,1]*7, [1]*21]` - a **3-level** ladder under a 4-step schedule (refinement hits the all-ones fixed point) that the trainer drew **0 times in 20 000** iterations, so those clips said nothing about the model being trained. At `--flex_pyramid_levels=1` the conclusion reverses: the whole-clip band is empty there and `[21]` is never drawn, so validation uses the uniform `--num_frame_per_block` partition - the one band that is both fixed across checkpoints and genuinely trained (`10.066%` of iterations, against ~`1.3%` for the most common random partition). On that 3-level ladder, now covered `10.034%` of the time: step 4 reuses the finest level `[1]*21` through `install_flex_partition`'s index clamp (`min(step_index, len(partitions) - 1)`), so nothing runs off the end.
- §3.3 has **no launcher flag**: the config yaml chooses the variant through `transformer_additional_kwargs.flex_kproj_mode` (default `diag_rank1`; set it to `none` there to ablate Pi, `diag` for the cheaper head-only map). Its parameters are named `flex_kproj.proj_out.*` on purpose - that name is what lets `initialize_missing_parameters()` zero-fill them when a checkpoint predates §3.3, which is how identity init survives `low_cpu_mem_usage=True` - and they otherwise train like any other layer, following `--learning_rate`. The trainer forces them `requires_grad` whenever the built model carries them, so a narrowed `--trainable_modules` cannot silently freeze Pi at its identity. Under `--use_kv_cache_training` it is never invoked and stays at its identity init as well.
- Every rank must train the same layout, otherwise the FlexAttention mask and the `num_frame_per_block` derived from it disagree across the SP/FSDP group. The trainer therefore passes the drawn partition through `broadcast_chunk_sizes` before using it, and `randomize_denoising_step_indices` broadcasts its jittered indices from rank 0 for the same reason.

### 3.4 Training Validation

You can configure validation parameters to periodically generate test videos during training, allowing you to monitor training progress and model quality.

**Validation Parameter Descriptions**:

| Parameter | Description | Recommended Value |
|------|------|--------|
| `--validation_steps` | Run validation every N steps | 2000 |
| `--validation_epochs` | Run validation every N epochs | 5 |
| `--validation_prompts` | Prompts for video generation validation | English prompts |

**T2V Validation Example**:

```bash
  --validation_steps=2000 \
  --validation_epochs=5 \
  --validation_prompts="A brown dog shaking its head, sitting on a light-colored sofa in a cozy room. Behind the dog, there's a framed picture on the shelf, surrounded by pink flowers. The soft and warm lighting in the room creates a comfortable atmosphere."
```

**Notes**:
- Validation videos will be saved to the `sample` subdirectory of `output_dir`
- Multi-prompt validation format: `--validation_prompts "prompt1" "prompt2" "prompt3"`
- With `--flex_pyramid_levels > 1`, validation renders the pyramid layout (`chunk_spec=None` + `denoise_mode="pyramid"`), the same ladder `predict_t2v.py` uses; at `--flex_pyramid_levels=1` it samples the uniform `--num_frame_per_block` partition instead. See **3.3 Common Training Parameters** for why those are the trained layouts.

### 3.5 Training with FSDP

**If you use multiple GPUs and encounter insufficient GPU memory with DeepSpeed-Zero-2**, you can switch to FSDP for training. The ready-to-use launcher is [train_distill.sh](./train_distill.sh):

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/wan2.1_flex_forcing/train_distill.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --fix_sample_size 480 832 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --score_num_frames=21 \
  --video_repeat=1 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_distill" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --train_mode="normal" \
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --flex_pyramid_levels=4 \
  --flex_min_num_frame_per_block=1 \
  --flex_self_generated_context \
  --denoising_step_indices_list 1000 750 500 250 \
  --randomize_step_indices \
  --flow_euler_rollout \
  --use_kv_cache_training \
  --num_frame_per_block=3 \
  --ode_transformer_path="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

Or simply:

```bash
bash scripts/wan2.1_flex_forcing/train_distill.sh
```

### 3.6 Other Backends

#### 3.6.1 Training with DeepSpeed-Zero-3

DeepSpeed Zero-3 is not highly recommended at the moment. In this repository, using FSDP has fewer errors and is more stable.

DeepSpeed Zero-3 is suitable for 14B Wan at high resolutions. After training, you can use the following command to get the final model:
```bash
python scripts/zero_to_bf16.py output_dir/checkpoint-{our-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

Training shell command is as follows:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_flex_forcing/train_distill.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --fix_sample_size 480 832 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --score_num_frames=21 \
  --video_repeat=1 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_distill" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --train_mode="normal" \
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --flex_pyramid_levels=4 \
  --flex_min_num_frame_per_block=1 \
  --flex_self_generated_context \
  --denoising_step_indices_list 1000 750 500 250 \
  --randomize_step_indices \
  --flow_euler_rollout \
  --use_kv_cache_training \
  --num_frame_per_block=3 \
  --ode_transformer_path="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

#### 3.6.2 Training without DeepSpeed and FSDP

**This approach is not recommended because there is no memory-saving backend, which can easily cause out-of-memory errors**. We only provide the training shell for reference.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.1_flex_forcing/train_distill.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --fix_sample_size 480 832 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --score_num_frames=21 \
  --video_repeat=1 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_distill" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --train_mode="normal" \
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --flex_pyramid_levels=4 \
  --flex_min_num_frame_per_block=1 \
  --flex_self_generated_context \
  --denoising_step_indices_list 1000 750 500 250 \
  --randomize_step_indices \
  --flow_euler_rollout \
  --use_kv_cache_training \
  --num_frame_per_block=3 \
  --ode_transformer_path="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

### 3.7 Multi-Node Distributed Training

**Suitable for**: Ultra-large-scale datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines, each with 8 GPUs:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Rank of this machine (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_flex_forcing/train_distill.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --fix_sample_size 480 832 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --score_num_frames=21 \
  --video_repeat=1 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_distill" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --train_mode="normal" \
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --flex_pyramid_levels=4 \
  --flex_min_num_frame_per_block=1 \
  --flex_self_generated_context \
  --denoising_step_indices_list 1000 750 500 250 \
  --randomize_step_indices \
  --flow_euler_rollout \
  --use_kv_cache_training \
  --num_frame_per_block=3 \
  --ode_transformer_path="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export MASTER_ADDR="192.168.1.100"  # Same as Master
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # Note this is 1
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# Use the same accelerate launch command as Machine 0
```

#### 3.7.2 Multi-Node Training Notes

- **Network Requirements**:
   - RDMA/InfiniBand recommended (high performance)
   - Without RDMA, add environment variables:
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **Data Synchronization**: All machines must be able to access the same data paths (NFS/shared storage)

---

## 4. Inference Testing

### 4.1 Inference Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|------|------|-------|
| `GPU_memory_mode` | GPU memory mode, see options below | `model_full_load` |
| `ulysses_degree` | Ulysses parallelism degree for multi-GPU inference | 1 |
| `ring_degree` | Ring parallelism degree for multi-GPU inference | 1 |
| `fsdp_dit` | Use FSDP for Transformer during multi-GPU inference to save memory | `False` |
| `fsdp_text_encoder` | Use FSDP for text encoder during multi-GPU inference | `True` |
| `compile_dit` | Compile Transformer for faster inference (effective at fixed resolution) | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `sampler_name` | Sampler type: `Flow`, `Flow_Unipc`, `Flow_DPM++` | `Flow` |
| `shift` | Noise schedule shift parameter, used when the sampler is `Flow_Unipc` / `Flow_DPM++` | 5 |
| `transformer_path` | Trained Transformer weight path | `"output_dir_wan2.1_flex_forcing_distill/checkpoint-1000/diffusion_pytorch_model.safetensors"` |
| `vae_path` | Trained VAE weight path | `None` |
| `lora_path` | LoRA weight path | `None` |
| `sample_size` | Generated video resolution `[height, width]` | `[432, 832]` |
| `video_length` | Number of frames to generate | `81` |
| `fps` | Frames per second | `16` |
| `denoise_mode` | Flex-Forcing partition schedule: `"fixed"` holds one partition for every step; `"pyramid"` runs one nested level per denoising step; an int pins a truncated pyramid | `"pyramid"` |
| `min_num_frame_per_block` | Block size the pyramid refinement stops at (1 = fully causal leaves) | 1 |
| `num_frame_per_block` | Number of frames to generate per block in the fixed schedule | 3 |
| `independent_first_frame` | Whether first frame is generated independently | `False` |
| `context_noise` | Context noise level for generation | 0.0 |
| `local_attn_size` | Local attention window size (-1 for global attention) | -1 |
| `sink_size` | Attention sink size used with the local attention window | 0 |
| `weight_dtype` | Model weight dtype, use `torch.float16` for GPUs that don't support bf16 | `torch.bfloat16` |
| `prompt` | Positive prompt describing what to generate | `"A stylish woman walks down a Tokyo street..."` |
| `negative_prompt` | Negative prompt to avoid certain content | Chinese negative prompt |
| `guidance_scale` | Guidance strength (distillation models typically use 1.0) | 1.0 |
| `seed` | Random seed for reproducibility | 43 |
| `num_inference_steps` | Number of inference steps (typically 4 for distillation models) | 4 |
| `lora_weight` | LoRA weight strength | 0.55 |
| `save_path` | Path to save generated videos | `samples/wan-videos-flex-forcing-t2v` |

**GPU Memory Mode Descriptions**:

| Mode | Description | Memory Usage |
|------|------|---------|
| `model_full_load` | Entire model loaded to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer groups switch between CPU/CUDA | Low |
| `sequential_cpu_offload` | Layer-by-layer offload (slowest) | Lowest |

### 4.2 Text-to-Video (T2V) Inference

The Stage 2 checkpoint is the final model. Point `transformer_path` in the inference entries at it:

```bash
# 3.1 flexible chunking / 3.2 pyramid
python examples/wan2.1_flex_forcing/predict_t2v.py
```

```python
# In examples/wan2.1_flex_forcing/predict_t2v.py
transformer_path = "output_dir_wan2.1_flex_forcing_distill/checkpoint-1000/diffusion_pytorch_model.safetensors"

num_inference_steps = 4          # matches --denoising_step_indices_list
video_length        = 81         # 21 latent frames = a 5 s clip
sample_size         = [432, 832]

# --- 3.1 / 3.2 inference knobs --------------------------------------------
# chunk_spec accepts None / int / "11-10" / "uniform:3" / "ar" / "bidir".
# denoise_mode "fixed" holds one partition for every step; "pyramid" runs one
# nested level per denoising step, its depth taken from num_inference_steps.
# An int pins a truncated pyramid instead: 4 over 21 latent frames reproduces
# the paper's [[21], [11, 10], [6, 5, 5, 5], [3, 3, 3, 2, 3, 2, 3, 2]].
chunk_spec          = None
denoise_mode        = "pyramid"
min_num_frame_per_block = 1

# 3.3 K-Projection: nothing to set - the model builds Pi (diag_rank1) and applies
# it on every call.
```

`predict_t2v.py` prints `PAPER_CHUNK_CONFIGS` at startup — the fifteen layouts §3.1 evaluates, from `[21]` (fully bidirectional) through `[11, 10]` and `[7, 7, 7]` down to `[3, 2, 2, 2, 2, 2, 2, 2, 2, 2]`. Feed any of them to `chunk_spec` as a dash-separated string.

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/wan2.1_flex_forcing/predict_t2v.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs used
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelism
ring_degree = 1     # Sequence dimension parallelism
```

**Configuration Principles**:
- `ulysses_degree` must be divisible by the model's head count
- `ring_degree` splits on the sequence dimension, which affects communication overhead. Try to avoid using it when heads are evenly divisible.

**Configuration Examples**:

| GPU Count | ulysses_degree | ring_degree | Description |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelism |
| 8 | 8 | 1 | Head parallelism |
| 8 | 4 | 2 | Hybrid parallelism |

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/wan2.1_flex_forcing/predict_t2v.py
```

### 4.4 Any-Order Video Editing

```bash
# 4.2 any-order, any-timestep editing
python examples/wan2.1_flex_forcing/predict_t2v_edit.py
```

For editing, `predict_t2v_edit.py` **edits only and generates nothing**, and restricts regeneration to the **refinement** timesteps (`edit_steps = 1`) while the planning timesteps stay untouched, which is what lets a middle span condition on clean context from the future:

```python
input_video_path    = "samples/wan-videos-flex-forcing-t2v/00000001.mp4"  # clip to edit, required
edit_span           = (8, 15)    # half-open range of *latent* frames
edit_steps          = 1          # low-level refinement only
num_frame_per_block = 7          # clean-context commit granularity; None = one bidirectional pass
```

This `num_frame_per_block` is the same knob as Self-Forcing's: the clean context is committed in uniform blocks of that many frames, in temporal order. It differs in exactly two ways - the trailing chunk absorbs the remainder so divisibility is not required (the inherited rollout asserts it), and the transformer's block width becomes `max(it, edit span width)`.

Any-order editing needs the full KV cache: both entries set `local_attn_size = -1`. A pyramid needs the same, plus `stochastic_sampling = True` — the buffered step between two levels *is* the schedule's own re-noising. The pipeline raises rather than silently degrading if either is violated.

---

## 5. Additional Resources

- **Flex-Forcing paper**: https://arxiv.org/abs/2607.03509
- **Stage 1**: [README_TRAIN_ODE.md](./README_TRAIN_ODE.md)
- **Shared chunking helpers**: `videox_fun/utils/flex_chunking.py` — `normalize_chunk_spec`, `sample_flexible_chunks`, `refine_partition`, `build_pyramid_partitions`, `validate_nested_partitions`, `broadcast_chunk_sizes`
- **Model / pipeline**: `videox_fun/models/wan_transformer3d_flex_forcing.py`, `videox_fun/pipeline/pipeline_wan_flex_forcing.py`
- **Upstream stages**: [README_TRAIN_ODE.md](../wan2.1_self_forcing/README_TRAIN_ODE.md), [README_TRAIN_DISTILL.md](../wan2.1_self_forcing/README_TRAIN_DISTILL.md)
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
