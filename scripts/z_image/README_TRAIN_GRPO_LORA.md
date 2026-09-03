# Z-Image GRPO LoRA Training Guide

This document provides a complete workflow for Z-Image GRPO (Group Relative Policy Optimization) LoRA reinforcement fine-tuning, including environment configuration, prompt data preparation, reward model configuration, multiple distributed training strategies, and inference testing.

> **Note**: Z-Image has two model variants: `Z-Image` (standard version) and `Z-Image-Turbo` (fast inference version). This guide uses `Z-Image` by default. To use `Z-Image-Turbo`, simply replace the model path accordingly.

**Difference from supervised LoRA fine-tuning**: GRPO does not need any ground-truth images, only a pool of prompts. The current model samples images by itself, a reward model scores them, and the group-relative advantage is used as the policy gradient signal to update the LoRA weights. Compared with reward backpropagation (`train_reward_lora.py`), GRPO does not require a differentiable reward model, and it avoids the VRAM cost and adversarial-patch artifacts caused by backpropagating through the reward model.

| Item | LoRA Fine-Tuning (`train_lora.py`) | GRPO LoRA Training (this document) |
|--|--|--|
| Training data | Images + prompts | **Prompts only** |
| Supervision signal | Denoising reconstruction error | Group-normalized advantage of reward model scores |
| Ground-truth images required | Yes | No |
| Differentiable reward model required | - | No |
| Typical use | Learn a specific style / concept / subject | Improve human preference alignment (aesthetics, prompt-image consistency) |

---

## Table of Contents
- [1. Environment Configuration](#1-environment-configuration)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Prompt Data Format](#22-prompt-data-format)
  - [2.3 Relative vs Absolute Path Usage](#23-relative-vs-absolute-path-usage)
  - [2.4 Prompt Pool Design Recommendations](#24-prompt-pool-design-recommendations)
- [3. GRPO Training](#3-grpo-training)
  - [3.1 Download Pretrained Model and Reward Model](#31-download-pretrained-model-and-reward-model)
  - [3.2 Quick Start (DeepSpeed-Zero-2)](#32-quick-start-deepspeed-zero-2)
  - [3.3 GRPO-Specific Parameters](#33-grpo-specific-parameters)
  - [3.4 Reward Model Configuration](#34-reward-model-configuration)
  - [3.5 Training Validation and Metrics Monitoring](#35-training-validation-and-metrics-monitoring)
  - [3.6 Training with FSDP](#36-training-with-fsdp)
  - [3.7 Other Backends](#37-other-backends)
  - [3.8 Multi-Machine Distributed Training](#38-multi-machine-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameter Parsing](#41-inference-parameter-parsing)
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

**Extra Dependencies for GRPO**

The dependencies required by the reward model depend on the chosen `--reward_fn`:

```bash
# HPSv3Reward (the default example in this document) relies on a Qwen2-VL base model, which needs a recent transformers
pip install "transformers>=4.49.0" "peft>=0.17.0" huggingface_hub

# Only required when using HPSReward (HPS v2/v2.1)
pip install hpsv2
```

- `MPSReward` automatically downloads `MPS_overall.pth` from the PAI OSS into the `torch.hub` cache directory on first run.
- `PickScoreReward` and `AestheticReward` automatically download their CLIP/SigLIP base models from Hugging Face on first run.
- If the training machine has no network access, download the weights in advance and fill in local paths in `--reward_fn_kwargs`, see [3.4 Reward Model Configuration](#34-reward-model-configuration).

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

### 2.1 Quick Test Dataset

We provide a test dataset containing several training samples. GRPO training only reads the `text` field; the image files are never loaded.

```bash
# Download official demo dataset
modelscope download --dataset PAI/X-Fun-Images-Demo --local_dir ./datasets/X-Fun-Images-Demo
```

### 2.2 Prompt Data Format

GRPO uses a text-only dataset (`TextDataset`). The annotation file is a JSON array in which **`text` is the only required field**:

```json
[
  {
    "text": "1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body"
  },
  {
    "text": "A beautiful sunset over the ocean, golden hour lighting"
  }
]
```

**Field Description**:
- `text`: The prompt used to sample images and to be scored by the reward model, **the only required field**
- `file_path` / `width` / `height` / `type`: All ignored, so `datasets/X-Fun-Images-Demo/metadata.json` (used for supervised LoRA training) can be reused directly as a prompt pool
- There is no need to run `scripts/process_json_add_width_and_height.py`, and no images need to be prepared

**Generation Resolution**: Since there are no ground-truth images, the sampling resolution is determined by the training arguments and defaults to a square of `--image_sample_size`. For non-square outputs, use `--fix_sample_size height width` to fix the resolution, or add `--random_hw_adapt` so that each batch randomly picks one aspect ratio.

### 2.3 Relative vs Absolute Path Usage

**Relative Paths**:

If your data uses relative paths, configure the training script as follows:

```bash
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata.json"
```

**Absolute Paths**:

If your data uses absolute paths, configure the training script as follows:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/prompts.json"
```

### 2.4 Prompt Pool Design Recommendations

The learning signal of GRPO comes from the relative quality among the multiple samples generated for the same prompt, so the size and diversity of the prompt pool directly determine the training result:

- **Prompt count**: It is recommended to have no fewer than the order of `num_batches_per_epoch × train_batch_size ÷ num_image_per_prompt` prompts, otherwise the same prompts keep repeating within one epoch. With the example configuration (`16 × 1 ÷ 16 = 1`), a single update cycle covers only 1 prompt; if the pool is too small, the model easily overfits to a handful of prompts.
- **Monitor `zero_std_ratio`**: This metric is the proportion of prompts whose within-group reward standard deviation is zero. When it stays high, the samples of the same prompt cannot be distinguished (either the model already converged on that prompt, or the reward model is not discriminative enough), and the advantage signal degenerates; expand the prompt pool or change/combine reward models in that case.
- **Prompt style**: Reward models (especially the HPS family) are sensitive to the style distribution they are applied to. The prompt pool should cover the target distribution you want to improve, rather than copying the validation prompts.

---

## 3. GRPO Training

### 3.1 Download Pretrained Model and Reward Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer
mkdir -p models/Diffusion_Transformer/HPSv3

# Download Z-Image official weights
modelscope download --model Tongyi-MAI/Z-Image --local_dir models/Diffusion_Transformer/Z-Image

# (Optional) Download Z-Image-Turbo fast inference version
modelscope download --model Tongyi-MAI/Z-Image-Turbo --local_dir models/Diffusion_Transformer/Z-Image-Turbo

# Download the reward models used by the example: HPSv3 scoring weights + its Qwen2-VL base model (HuggingFace only)
huggingface-cli download MizzenAI/HPSv3 --local-dir models/Diffusion_Transformer/HPSv3
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir models/Diffusion_Transformer/Qwen2-VL-7B-Instruct
```

> **Note**: If `checkpoint_path` of `HPSv3Reward` is left empty, it is downloaded online via `hf_hub_download("MizzenAI/HPSv3", "HPSv3.safetensors")`; the `Qwen2-VL-7B-Instruct` base model is also fetched online by default. In an offline environment, specify local paths explicitly.

### 3.2 Quick Start (DeepSpeed-Zero-2)

If you have downloaded the data as per **2.1 Quick Test Dataset** and the weights as per **3.1 Download Pretrained Model and Reward Model**, you can directly copy and run the quick start command.

DeepSpeed-Zero-2 and FSDP are recommended for training. Here we use DeepSpeed-Zero-2 as an example.

The difference between DeepSpeed-Zero-2 and FSDP lies in whether the model weights are sharded. **If VRAM is insufficient when using multiple GPUs with DeepSpeed-Zero-2**, you can switch to FSDP.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Z-Image"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/z_image/train_grpo_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1328 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=10 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_z_image_grpo_lora" \
  --validation_steps=10 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1 \
  --enable_bucket \
  --uniform_sampling \
  --rank=128 \
  --network_alpha=64 \
  --target_name="to_q,to_k,to_v,feed_forward.w1,feed_forward.w2,feed_forward.w3" \
  --noise_level 1.2 \
  --grpo_num_steps=20 \
  --grpo_cfg_scale=6 \
  --sde_window_size 5 \
  --sde_window_range 0 10 \
  --num_image_per_prompt=16 \
  --num_batches_per_epoch=16 \
  --reward_fn="HPSv3Reward" \
  --reward_fn_kwargs='{"HPSv3Reward": {"checkpoint_path": "models/Diffusion_Transformer/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Diffusion_Transformer/Qwen2-VL-7B-Instruct"}}' \
  --multi_reward_weights='{"HPSv3Reward": 1}'
```

> **Note**: Frequently tuned arguments that do not appear above are `--clip_range` (default `1e-5`), `--adv_clip_max` (default `5.0`), `--grpo_beta` (default `0.0`, i.e. no KL regularization), `--use_peft_lora` and `--low_vram`. Append them as needed; see [3.3 GRPO-Specific Parameters](#33-grpo-specific-parameters) for their meaning.

### 3.3 GRPO-Specific Parameters

**Core GRPO Parameters**:

| Parameter | Description | Default | Example Value |
|-----|------|--------|-------|
| `--grpo_num_steps` | Number of denoising steps during sampling | 20 | 20 |
| `--grpo_cfg_scale` | CFG scale used for sampling and for recomputing log-probs during training; when set to ≤ 1.0 the script skips CFG (only the conditional branch is run), which greatly reduces the cost for Turbo models | 4.5 | 6 |
| `--noise_level` | Noise strength injected at each step inside the SDE window; larger values give more exploration and better reward separation, but deviate further from the base distribution | 1.2 | 1.2 |
| `--sde_window_size` | Number of trained timesteps participating in the policy update; 0 means using all steps (`grpo_num_steps - 1`) | 2 | 5 |
| `--sde_window_range` | Candidate range `[start, end]` for the window start, must satisfy `end - sde_window_size >= start` (checked by an assertion) | `0 5` | `0 10` |
| `--clip_range` | PPO clipping range, controls the trust region of a single update; larger values speed up reward growth but make the base model easier to break | 1e-5 | 1e-5 |
| `--adv_clip_max` | Symmetric clipping bound `±adv_clip_max` of the advantage, prevents outliers from dominating the gradient | 5.0 | 5.0 |
| `--grpo_beta` | KL regularization coefficient; `0.0` means no KL constraint at all (the reference model is still loaded but not used in the forward pass); increase it when reward hacking appears | 0.0 | 0.0 |
| `--num_image_per_prompt` | Number of samples generated per prompt, i.e. the GRPO group size, which determines the variance of the advantage estimate | 16 | 16 |
| `--num_batches_per_epoch` | How many sampling batches are collected before one policy update; all batches are sampled with the same weights and concatenated for advantage computation | 16 | 16 |
| `--reward_fn` | Reward model class name; use comma-separated names for multiple models, e.g. `HPSv3Reward,MPSReward` | `MPSReward` | `HPSv3Reward` |
| `--reward_fn_kwargs` | JSON string of reward model constructor kwargs, see [3.4 Reward Model Configuration](#34-reward-model-configuration) | None | see example |
| `--multi_reward_weights` | JSON weights used to combine the per-reward advantages; equal weights if not provided, and the values are automatically normalized | None | `'{"HPSv3Reward": 1}'` |
| `--per_prompt_stat_tracking` | Track per-prompt mean/std for advantage normalization (within-group comparison). It is a flag argument and is enabled by default; the script disables it automatically when `num_image_per_prompt=1` | True | - |
| `--global_std` | Use the global std instead of the within-group std as the normalization denominator (flag argument, enabled by default), which keeps a gradient even when all samples of a group are identical | True | - |

**LoRA and Common Training Parameters**:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--pretrained_model_name_or_path` | Path to pretrained model | `models/Diffusion_Transformer/Z-Image` |
| `--train_data_meta` | Path of the prompt annotation JSON | `datasets/X-Fun-Images-Demo/metadata.json` |
| `--train_batch_size` | Samples per batch | 1 |
| `--image_sample_size` | Sampling resolution (square); combine with `--fix_sample_size`/`--random_hw_adapt` to change the output aspect ratio | 1328 |
| `--fix_sample_size` | Fixed sampling resolution `[height, width]`; disables `random_hw_adapt` when set | None |
| `--random_hw_adapt` | Randomly pick one aspect ratio per batch | - |
| `--gradient_accumulation_steps` | Gradient accumulation factor, scales the policy-update accumulation window | 1 |
| `--dataloader_num_workers` | DataLoader subprocesses | 8 |
| `--num_train_epochs` | Number of training epochs (one epoch traverses the prompt pool once) | 100 |
| `--checkpointing_steps` | Save checkpoint every N update steps | 10 |
| `--learning_rate` | Initial learning rate | 1e-04 |
| `--max_grad_norm` | Gradient clipping threshold (1.0 is recommended for GRPO; the 0.05 commonly used by supervised LoRA clearly suppresses the policy gradient) | 1 |
| `--seed` | Random seed; all ranks traverse the same prompt sequence with different noise | 42 |
| `--output_dir` | Output directory | `output_dir_z_image_grpo_lora` |
| `--gradient_checkpointing` | Enable activation checkpointing | - |
| `--mixed_precision` | Mixed precision: `fp16/bf16` | `bf16` |
| `--rank` | Dimension of LoRA update matrices (higher rank = stronger expressiveness but more VRAM usage) | 128 |
| `--network_alpha` | Scaling factor of LoRA update matrices (typically set to half of rank or same) | 64 |
| `--target_name` | Components/modules to apply LoRA, separated by commas | `to_q,to_k,to_v,feed_forward.w1,feed_forward.w2,feed_forward.w3` |
| `--use_peft_lora` | Use PEFT module for adding LoRA (more VRAM-efficient), also exports ComfyUI-compatible weights | - |
| `--low_vram` | Keep the reference model / text encoder on CPU and move them back to GPU only when needed | - |
| `--resume_from_checkpoint` | Resume training from checkpoint path, use `"latest"` to auto-select latest | None |
| `--validation_steps` / `--validation_epochs` / `--validation_prompts` | Validation frequency and prompts | see [3.5](#35-training-validation-and-metrics-monitoring) |

### 3.4 Reward Model Configuration

`--reward_fn_kwargs` is a JSON string. For a single reward, either a flat object or a nested `{"RewardName": {...}}` object works; for multiple rewards, the nested form is required.

**Constructor Arguments of the Image Rewards (applicable to Z-Image)**:

| Reward | Key kwargs | Description |
|--------|-------------|------|
| `HPSv3Reward` | `checkpoint_path`, `model_name_or_path` | Preference model based on Qwen2-VL-7B; if `checkpoint_path` is empty, `HPSv3.safetensors` is downloaded from `MizzenAI/HPSv3` automatically, and `model_name_or_path` should point to a local `Qwen2-VL-7B-Instruct` |
| `HPSReward` | `model_path`, `version` | HPS v2 / v2.1, `version` takes `"v2.0"` or `"v2.1"`, requires the extra `hpsv2` package |
| `PickScoreReward` | `model_path`, `processor_name_or_path` | Defaults to `yuvalkirstain/PickScore_v1` + `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, downloaded automatically when online |
| `MPSReward` | `model_path`, `processor_name_or_path` | Downloads `MPS_overall.pth` automatically by default; `model_path` can point to a local `.pth` |
| `AestheticReward` | `encoder_path`, `predictor_path`, `version` | `version` takes `"v2"` (requires `clip-vit-large-patch14` + the aesthetic MLP weights) or `"v2.5"` (requires `siglip-so400m-patch14-384`); pure aesthetics scoring, prompt is not used |

> **Note**: `max_reward` and `loss_scale` only define the loss of reward-backpropagation training. GRPO only calls `get_reward` to obtain raw scores and normalizes them itself, so these two arguments do not affect GRPO results. The same applies to `differentiable` of `HPSv3Reward` / `VideoAlignReward`. `VideoAlignReward` is a video reward and is not applicable to this script.

**Single Reward Example**:

```bash
  --reward_fn="HPSv3Reward" \
  --reward_fn_kwargs='{"HPSv3Reward": {"checkpoint_path": "models/Diffusion_Transformer/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Diffusion_Transformer/Qwen2-VL-7B-Instruct"}}' \
  --multi_reward_weights='{"HPSv3Reward": 1}'
```

**Multi-Reward Example** (advantage is computed per reward first, then combined with the normalized weights):

```bash
  --reward_fn="HPSv3Reward,MPSReward" \
  --reward_fn_kwargs='{"HPSv3Reward": {"checkpoint_path": "models/Diffusion_Transformer/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Diffusion_Transformer/Qwen2-VL-7B-Instruct"}, "MPSReward": {}}' \
  --multi_reward_weights='{"HPSv3Reward": 0.7, "MPSReward": 0.3}'
```

With multiple rewards, TensorBoard additionally records `{RewardName}_reward_mean`, `{RewardName}_advantage_mean` and `{RewardName}_weight`, which makes it possible to spot conflicts between the two models.

### 3.5 Training Validation and Metrics Monitoring

You can configure validation parameters to periodically generate test images during training, in order to monitor training progress and model quality.

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--validation_steps` | Execute validation every N steps | 100 |
| `--validation_epochs` | Execute validation every N epochs | 100 |
| `--validation_prompts` | Prompts for validation image generation, separated by spaces | Multiple space-separated prompts |

Validation sampling always uses `ZImagePipeline`: `guidance_scale=4.5, num_inference_steps=25` for non-Turbo models, and `guidance_scale=0, num_inference_steps=8` when the model path contains `Turbo`.

### 3.6 Training with FSDP

**If VRAM is insufficient when using multiple GPUs with DeepSpeed-Zero-2**, you can switch to FSDP.

> ✅ **Recommended**: FSDP has been thoroughly tested in this repository, with fewer errors and greater stability.

```sh
export MODEL_NAME="models/Diffusion_Transformer/Z-Image"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=ZImageTransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/z_image/train_grpo_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1328 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=10 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_z_image_grpo_lora" \
  --validation_steps=10 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1 \
  --enable_bucket \
  --uniform_sampling \
  --rank=128 \
  --network_alpha=64 \
  --target_name="to_q,to_k,to_v,feed_forward.w1,feed_forward.w2,feed_forward.w3" \
  --noise_level 1.2 \
  --grpo_num_steps=20 \
  --grpo_cfg_scale=6 \
  --sde_window_size 5 \
  --sde_window_range 0 10 \
  --num_image_per_prompt=16 \
  --num_batches_per_epoch=16 \
  --reward_fn="HPSv3Reward" \
  --reward_fn_kwargs='{"HPSv3Reward": {"checkpoint_path": "models/Diffusion_Transformer/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Diffusion_Transformer/Qwen2-VL-7B-Instruct"}}' \
  --multi_reward_weights='{"HPSv3Reward": 1}'
```

### 3.7 Other Backends

**Training without DeepSpeed or FSDP is not recommended** as it lacks VRAM-saving backends and may easily cause out-of-memory errors (this is also the original launch command in `scripts/z_image/train_grpo_lora.sh`, which can be used directly on a single GPU). This is provided for reference only.

```sh
export MODEL_NAME="models/Diffusion_Transformer/Z-Image"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/z_image/train_grpo_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1328 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=10 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_z_image_grpo_lora" \
  --validation_steps=10 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1 \
  --enable_bucket \
  --uniform_sampling \
  --rank=128 \
  --network_alpha=64 \
  --target_name="to_q,to_k,to_v,feed_forward.w1,feed_forward.w2,feed_forward.w3" \
  --noise_level 1.2 \
  --grpo_num_steps=20 \
  --grpo_cfg_scale=6 \
  --sde_window_size 5 \
  --sde_window_range 0 10 \
  --num_image_per_prompt=16 \
  --num_batches_per_epoch=16 \
  --reward_fn="HPSv3Reward" \
  --reward_fn_kwargs='{"HPSv3Reward": {"checkpoint_path": "models/Diffusion_Transformer/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Diffusion_Transformer/Qwen2-VL-7B-Instruct"}}' \
  --multi_reward_weights='{"HPSv3Reward": 1}'
```

**DeepSpeed-Zero-3**: The only difference from DeepSpeed-Zero-2 is replacing the config with `config/zero_stage3_config.json` in the launch arguments. Zero-3 and FSDP-3 automatically set `--save_state` to True, so checkpoints are directories `output_dir/checkpoint-{global_step}/` containing the optimizer state; resume training with `--resume_from_checkpoint="latest"`.

### 3.8 Multi-Machine Distributed Training

**Suitable for**: Very large prompt pools, higher sampling throughput

#### 3.8.1 Environment Configuration

Assuming 2 machines with 8 GPUs each:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Z-Image"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/z_image/train_grpo_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1328 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=10 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_z_image_grpo_lora" \
  --validation_steps=10 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1 \
  --enable_bucket \
  --uniform_sampling \
  --rank=128 \
  --network_alpha=64 \
  --target_name="to_q,to_k,to_v,feed_forward.w1,feed_forward.w2,feed_forward.w3" \
  --noise_level 1.2 \
  --grpo_num_steps=20 \
  --grpo_cfg_scale=6 \
  --sde_window_size 5 \
  --sde_window_range 0 10 \
  --num_image_per_prompt=16 \
  --num_batches_per_epoch=16 \
  --reward_fn="HPSv3Reward" \
  --reward_fn_kwargs='{"HPSv3Reward": {"checkpoint_path": "models/Diffusion_Transformer/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Diffusion_Transformer/Qwen2-VL-7B-Instruct"}}' \
  --multi_reward_weights='{"HPSv3Reward": 1}'
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Z-Image"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata.json"
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

#### 3.8.2 Multi-Machine Training Notes

- **Network Requirements**:
   - RDMA/InfiniBand recommended (high performance)
   - Without RDMA, add environment variables:
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **Data Synchronization**: All machines must be able to access the same data paths (NFS/shared storage), including the reward model weight paths.

---

## 4. Inference Testing

### 4.1 Inference Parameter Parsing

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|------|------|-------|
| `GPU_memory_mode` | VRAM management mode, see table below for options | `model_cpu_offload` |
| `ulysses_degree` | Head dimension parallelism degree, set to 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelism degree, set to 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP for Transformer during multi-GPU inference to save VRAM | `False` |
| `fsdp_text_encoder` | Use FSDP for text encoder during multi-GPU inference | `False` |
| `compile_dit` | Compile Transformer for faster inference (effective at fixed resolution) | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/Z-Image` |
| `sampler_name` | Sampler type: `Flow`, `Flow_Unipc`, `Flow_DPM++` | `Flow` |
| `transformer_path` | Path to load trained Transformer weights | `None` |
| `vae_path` | Path to load trained VAE weights | `None` |
| `lora_path` | LoRA weights path | `None` |
| `sample_size` | Generated image resolution `[height, width]` | `[1728, 992]` |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 support | `torch.bfloat16` |
| `prompt` | Positive prompt describing the generation content | `"1girl, black_hair, ..."` |
| `negative_prompt` | Negative prompt for content to avoid | `"low resolution, low quality..."` |
| `guidance_scale` | Guidance strength, recommended 0.0 for Turbo model | 4.0 / 0.0 |
| `seed` | Random seed for reproducible results | 43 |
| `num_inference_steps` | Number of inference steps, can be greatly reduced for Turbo model | 25 / 9 |
| `lora_weight` | LoRA weight strength | 0.55 |
| `save_path` | Path to save generated images | `samples/z-image-t2i` |

**VRAM Management Mode Description**:

| Mode | Description | VRAM Usage |
|------|------|---------|
| `model_full_load` | Load entire model to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer groups switch between CPU/CUDA | Low |
| `sequential_cpu_offload` | Sequential layer offload (slowest) | Lowest |

> **Difference from LoRA fine-tuning**: GRPO weights are written directly as `output_dir/checkpoint-{global_step}.safetensors` (not a `checkpoint-{global_step}/` directory), so point `lora_path` at that `.safetensors` file. The loading logic handles both the PEFT naming and the ComfyUI (kohya) naming, so either file can be used.

### 4.2 Single GPU Inference

#### Z-Image (Standard Version)

Run the following command for single GPU inference:

```bash
python examples/z_image/predict_t2i.py
```

Edit `examples/z_image/predict_t2i.py` according to your needs. For first-time inference, focus on these parameters. For other parameters, refer to the inference parameter parsing above.

```python
# Choose based on GPU VRAM
GPU_memory_mode = "model_cpu_offload"
# Based on actual model path
model_name = "models/Diffusion_Transformer/Z-Image"  
# GRPO LoRA weights path, e.g., "output_dir_z_image_grpo_lora/checkpoint-100.safetensors"
lora_path = None
# LoRA weight strength; for GRPO weights start from 1.0 and try 0.5/0.75 to check overfitting to the reward
lora_weight = 0.55
# Write based on generation content
prompt = "A young woman standing on a sunny coastline, her white dress gently fluttering in the sea breeze."  
# ...
```

> 💡 **CFG Consistency Tip**: GRPO collects trajectories and optimizes the policy under the guidance scale given by `--grpo_cfg_scale`; if the inference `guidance_scale` deviates a lot from it, the gain may be hard to see. Start from a value close to the one used in training.

#### Z-Image-Turbo (Fast Version)

Run the following command for single GPU inference:

```bash
python examples/z_image/predict_turbo_t2i.py
```

Edit `examples/z_image/predict_turbo_t2i.py` in the same way as the standard version, only replacing `model_name` with `models/Diffusion_Transformer/Z-Image-Turbo`. When training with the Turbo model, it is also recommended to set `--grpo_cfg_scale` below `1.0` to skip CFG.

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/z_image/predict_t2i.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelization
ring_degree = 1     # Sequence dimension parallelization
```

**Configuration Principles**:
- `ulysses_degree` must evenly divide the model's number of heads
- `ring_degree` splits on sequence dimension, affecting communication overhead; avoid using it when heads can be divided

**Example Configurations**:

| GPU Count | ulysses_degree | ring_degree | Description |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelization |
| 8 | 8 | 1 | Head parallelization |
| 8 | 4 | 2 | Hybrid parallelization |

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/z_image/predict_t2i.py
```

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
- **Supervised LoRA Training**: [README_TRAIN_LORA.md](./README_TRAIN_LORA.md)
- **Full Fine-Tuning Training**: [README_TRAIN.md](./README_TRAIN.md)
