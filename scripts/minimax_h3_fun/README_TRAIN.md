# MiniMax-H3 Control Training Guide

This document provides a complete workflow for MiniMax-H3 control (VACE-style controllable video generation) training, including environment setup, data preparation, various distributed training strategies, CFG distillation, and inference testing.

> **Note**: MiniMax-H3 is an audio-visual generative video model that simultaneously generates video and corresponding audio. Control training adds a zero-initialised control side branch (`MiniMaxH3ControlTransformer3DModel`) on top of the base packed-sequence transformer: a paired control video (pose / depth / canny ...) is patchified and injected as per-layer skips, so a freshly initialised model is numerically identical to the base MiniMax-H3 model and only the control parameters (`--trainable_modules control`) need training. The joint video + audio flow-matching loss of `scripts/minimax_h3/train.py` is kept, so the training data requires video, control video **and** audio files.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. Control Branch Training](#3-control-branch-training)
  - [3.1 Download Pre-trained Model](#31-download-pre-trained-model)
  - [3.2 Control Branch YAML Configuration](#32-control-branch-yaml-configuration)
  - [3.3 Quick Start (FSDP)](#33-quick-start-fsdp)
  - [3.4 Control-specific Parameter Reference](#34-control-specific-parameter-reference)
  - [3.5 Training Validation](#35-training-validation)
  - [3.6 Training with FSDP2](#36-training-with-fsdp2)
  - [3.7 Other Backends](#37-other-backends)
    - [3.7.1 Training with DeepSpeed-Zero-2](#371-training-with-deepspeed-zero-2)
    - [3.7.2 Training without DeepSpeed or FSDP](#372-training-without-deepspeed-or-fsdp)
  - [3.8 Multi-node Distributed Training](#38-multi-node-distributed-training)
  - [3.9 CFG Distillation of the Control Branch](#39-cfg-distillation-of-the-control-branch)
  - [3.10 Extracting Control-only Weights](#310-extracting-control-only-weights)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameter Reference](#41-inference-parameter-reference)
  - [4.2 V2V Control Inference](#42-v2v-control-inference)
  - [4.3 Multi-GPU Parallel Inference](#43-multi-gpu-parallel-inference)
- [5. More Resources](#5-more-resources)

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
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**Option 3: Using Docker**

When using Docker, please ensure that the GPU driver and CUDA environment are correctly installed on the machine, then execute the following commands:

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset with control signals containing some training data.

```bash
# Download the official example dataset (with control signals)
modelscope download --dataset PAI/X-Fun-Videos-Controls-Demo --local_dir ./datasets/X-Fun-Videos-Controls-Demo
```

After downloading, the dataset contains the following metadata files:
- `metadata.json`: basic format (control video paths only)
- `metadata_add_width_height.json`: with width/height info
- `metadata_add_width_height_add_wav.json`: with width/height info + audio paths (recommended for MiniMax-H3 control training)

> 💡 The demo's control signal lives in `canny/` and the paired audio tracks live in `wav/` (referenced by `metadata_add_width_height_add_wav.json`); metadata files without `audio_path` still work, since the training dataset decodes the audio track from each video container in that case.

### 2.2 Dataset Structure

Control training datasets require original videos with corresponding control signal videos (e.g., pose videos, depth videos, canny edge videos, etc.) **and** paired audio tracks, since MiniMax-H3 keeps the joint video + audio training loss.

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/                    # Original training videos
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   ├── 📂 control/                  # Control signal videos (e.g., pose / depth / canny)
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   ├── 📂 wav/                      # Paired audio tracks
│   │   ├── 📄 audio001.wav
│   │   ├── 📄 audio002.wav
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

> **Note**:
> - `train/` directory stores original videos
> - `control/` (or `pose/`, `depth/`, `canny/`, etc.) directory stores control signal videos that correspond one-to-one with the original videos. The directory name is customizable, as long as `control_file_path` in `metadata.json` correctly points to it
> - `wav/` directory stores the paired waveforms; they are resampled to the audio VAE's sample rate (32 kHz) during training

### 2.3 metadata.json Format

> ⚠️ **Important**: unlike regular video training, control training of MiniMax-H3 requires **the `control_file_path` field** in `metadata.json` — `VideoSpeechControlDataset` reads it on top of the usual ones. The `audio_path` field is optional; when it is absent, the audio track is decoded from the video container itself.

**Relative path format** (example):
```json
[
  {
    "file_path": "train/video001.mp4",
    "control_file_path": "control/video001.mp4",
    "audio_path": "wav/audio001.wav",
    "text": "A brown dog barks on a sofa, sitting on a light-colored couch in a cozy room",
    "type": "video",
    "width": 768,
    "height": 512
  }
]
```

**Absolute path format**:
```json
[
  {
    "file_path": "/mnt/data/videos/dog.mp4",
    "control_file_path": "/mnt/data/control/dog.mp4",
    "audio_path": "/mnt/data/wavs/dog.wav",
    "text": "A brown dog barks on a sofa",
    "type": "video",
    "width": 768,
    "height": 512
  }
]
```

**Key field descriptions**:
- `file_path`: Original video path (relative or absolute)
- `control_file_path`: Control signal video path (**required for control training**)
- `audio_path`: Audio file path (**MiniMax-H3 specific, optional**). Audio files are typically in `.wav` format; the path should correspond to `file_path`. When it is absent, the audio track is decoded from the video container itself
- `text`: Video description (English prompt)
- `type`: Data type, fixed as `"video"`
- `width` / `height`: Video dimensions (**recommended to provide**, used for bucket training. If not provided, they will be read automatically during training, which may affect training speed when data is stored on slower systems like OSS).
  - Use `scripts/process_json_add_width_and_height.py` to extract width and height for JSON files without these fields, supporting both images and videos.

### 2.4 Relative vs Absolute Path Usage

**Relative paths**:

If data paths are relative, set in the training script:

```bash
export DATASET_NAME="datasets/my_dataset/"
export DATASET_META_NAME="datasets/my_dataset/metadata.json"
```

**Absolute paths**:

If data paths are absolute, set in the training script:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **Recommendation**: Use relative paths for small local datasets; use absolute paths for external storage (NAS, OSS) or shared storage across multiple machines.

---

## 3. Control Branch Training

### 3.1 Download Pre-trained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download MiniMax-H3 official weights
hf download MiniMax-AI/MiniMax-H3 --local-dir models/Diffusion_Transformer/MiniMax-H3

# Download the pretrained control branch (Controlnet-Union) weights
modelscope download --model PAI/MiniMax-H3-Fun-Controlnet-Union --local_dir models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union
```

> 💡 The loader accepts either the converted diffusers layout above or an *original* MiniMax-H3 partition (e.g. `MiniMax-H3/FL2VA`); the original shards are converted on the fly while loading, with no intermediate copy on disk. The control branch is **not** part of the base MiniMax-H3 weights: `from_pretrained` fills it in, every control block is initialised from the main block it is attached to and `control_proj_in` from `proj_in`, with `before_proj` / `after_proj` zeroed, so a freshly loaded model is numerically identical to the base MiniMax-H3 model. To train or infer with a control signal, load the released control checkpoint `models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union/MiniMax-H3-Fun-Controlnet-Union.safetensors` through `--transformer_path` (training hot start) or `transformer_path` (inference).

### 3.2 Control Branch YAML Configuration

The control branch layout is driven by a YAML config passed through `--config_path`, whose `transformer_additional_kwargs` is spread into `from_pretrained` the same way at training and inference time:

```yaml
# config/minimax_h3/minimax_h3_control.yaml (inpaint variant)
format: diffusers
pipeline: minimax-h3
transformer_additional_kwargs:
    control_blocks_places: [0, 10, 20, 30, 40]
    control_in_dim: 49
```

| Key | Description | Default |
|-----|-------------|---------|
| `control_blocks_places` | The transformer layer indices the zero-initialised control blocks attach to | - |
| `control_in_dim` | The channels the control rows carry. `49` = control latent (24) + visibility map (1) + masked-video latents (24), used with `--enable_inpaint`; use `24` for a mask-less branch (`config/minimax_h3/minimax_h3_control_only.yaml`) | - |
| `control_apply_audio` | Whether the control skips reach the audio rows. `False` zeros the audio rows out of every skip before injection, so the control video guides the video rows alone while the soundtrack stays on the base model's path | `True` (default) |

> ⚠️ **Important**: `control_in_dim` must be pinned by the YAML — it is never inferred in code. `control_in_dim=49` is only valid together with `--enable_inpaint`; without the flag use `control_in_dim=24`. A checkpoint trained with one layout cannot be loaded into a model built with the other (`control_proj_in.weight` shape mismatch), and inference must load the **same** YAML the checkpoint was trained with.

### 3.3 Quick Start (FSDP)

FSDP is recommended for training MiniMax-H3 control: the transformer alone is about 62 GB in bfloat16 and the Qwen3-VL conditioner is another 62 GB, so the model weights must be sharded across GPUs — which FSDP (`FULL_SHARD`) does but DeepSpeed-Zero-2 does not. Only the control branch is trainable (`--trainable_modules control`), the rest of the model stays frozen.

```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" \
    --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap BaseMiniMaxH3TransformerBlock,MiniMaxH3ControlTransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
      scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=311 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_control_inpaint" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --enable_inpaint \
  --low_vram \
  --validation_paths "asset/pose.mp4" \
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body" \
  --uniform_sampling \
  --trainable_modules "control" \
  --resume_from_checkpoint=latest
```

> 💡 The launch script `scripts/minimax_h3_fun/train_control.sh` can be used as a starting point. `VIDEOX_OFFLOAD_VACE_LATENTS=True` offloads the encoded control/video latents to CPU between steps.

### 3.4 Control-specific Parameter Reference

**Key Control Parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config_path` | Control branch YAML config (pins `control_blocks_places` / `control_in_dim`, see 3.2) | `config/minimax_h3/minimax_h3_control.yaml` |
| `--enable_inpaint` | Feed a random inpaint mask through the control branch alongside the control video: the control rows carry the visibility map + masked-video latents on top of `in_channels`. The YAML must pin the matching widened `control_in_dim` (49) | - |
| `--trainable_modules` | Trainable modules (`control` trains only the control side branch, the default recipe) | `"control"` |
| `--trainable_modules_low_learning_rate` | Modules trained with lower learning rate | `[]` |
| `--transformer_path` | Load transformer weights from another checkpoint (e.g. a hot start) | None |
| `--pretrained_model_name_or_path` | Pre-trained MiniMax-H3 model path | `models/Diffusion_Transformer/MiniMax-H3` |
| `--train_data_dir` | Training data directory (empty when the meta uses absolute paths) | `""` |
| `--train_data_meta` | Training data metadata file | `/mnt/data/datasets/my_dataset/metadata.json` |
| `--train_batch_size` | Batch size per device | 1 |
| `--video_sample_size` | Max video training resolution | 960 |
| `--token_sample_size` | Resolution corresponding to the max token length when `training_with_video_token_length` is on | 960 |
| `--video_sample_stride` | Frame sampling stride (MiniMax-H3 is 24 fps) | 1 |
| `--video_sample_n_frames` | Number of frames to sample, must follow the `17*n+5` form of the video VAE (duration stays between 5 and 15 seconds) | 311 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 1 |
| `--dataloader_num_workers` | DataLoader subprocess count | 8 |
| `--num_train_epochs` | Training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 50 |
| `--learning_rate` | Initial learning rate (recommended for control branch training) | 2e-05 |
| `--lr_scheduler` | LR scheduler: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup` | `constant_with_warmup` |
| `--lr_warmup_steps` | LR warmup steps | 100 |
| `--seed` | Random seed for reproducible training | 42 |
| `--output_dir` | Output directory | `output_dir_minimax_h3_control_inpaint` |
| `--gradient_checkpointing` | Enable gradient checkpointing to save memory | - |
| `--gradient_checkpointing_save_on_cpu` | Offload the activations saved for backward of the transformer blocks to CPU memory | - |
| `--mixed_precision` | Mixed precision: `no`, `fp16`, `bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon value | 1e-10 |
| `--vae_mini_batch` | VAE encoding mini batch size | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--enable_bucket` | Enable bucket training, no cropping of videos, train by resolution grouping | - |
| `--random_hw_adapt` | Auto-scale videos to random sizes within `[min_size, max_size]` range | - |
| `--training_with_video_token_length` | Train by token length for arbitrary resolution | - |
| `--uniform_sampling` | Uniform timestep sampling (recommended) | - |
| `--low_vram` | Keep VAE and conditioner on CPU, move to GPU only while encoding | - |
| `--offload_every_step` | Move the transformer through CPU between steps (cards far below 62 GB) | - |
| `--resume_from_checkpoint` | Resume training path, use `"latest"` for auto-selection | `latest` |
| `--validation_steps` | Validate every N steps | 100 |
| `--validation_epochs` | Validate every N epochs | 500 |
| `--validation_prompts` | Validation prompts | `"1girl, ..."` |
| `--validation_paths` | Validation control video paths | `"asset/pose.mp4"` |
| `--validation_sampling_steps` | Number of denoising steps of the validation sampling loop | 50 |
| `--use_8bit_adam` | Use 8-bit Adam optimizer to save memory | - |

**Training recipe notes**:
- 10% of the batches zero the control latents, keeping the unconditional path trainable (CFG)
- The audio stream keeps the joint video + audio flow-matching loss of `scripts/minimax_h3/train.py`
- Checkpoints are serialized in the diffusers layout: `<output_dir>/checkpoint-x/transformer/diffusion_pytorch_model.safetensors` plus `config.json`, so the predict scripts load them with `from_pretrained(..., subfolder="transformer")`

### 3.5 Training Validation

You can configure validation parameters to periodically generate test videos during training to monitor training progress and model quality.

**Validation Parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--validation_steps` | Validate every N steps | 2000 |
| `--validation_epochs` | Validate every N epochs | 5 |
| `--validation_prompts` | Prompts for validation video generation | None |
| `--validation_paths` | Control video paths for validation | None |
| `--validation_sampling_steps` | Denoising steps of the validation sampling loop | 50 |

**Validation Example**:

```bash
  --validation_paths "asset/pose.mp4" \
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body"
```

**Notes**:
- Validation videos are saved to the `output_dir/sample/` directory
- Multi-prompt format: `--validation_prompts "prompt1" "prompt2" "prompt3"`
- `validation_paths` should correspond one-to-one with `validation_prompts`, pointing to control video files

### 3.6 Training with FSDP2

The same `train_control.py` runs under FSDP2 by adding `--fsdp_version 2` to the accelerate launch command. Things to know before choosing it:

- `--fsdp_sharding_strategy`, `--fsdp_backward_prefetch` and `--fsdp_use_orig_params` are FSDP1-only; under FSDP2 the reshard behaviour is `--fsdp_reshard_after_forward`, which must be a boolean
- FSDP2 reshards a fully frozen unit through an autograd function on the unit's forward inputs, so the frozen main stack is released during the backward pass without the workarounds FSDP1 needs
- accelerate's FSDP2 path upcasts the model to float32 whenever mixed precision is on, so the frozen base keeps fp32 master weights: expect roughly twice the sharded parameter memory of the FSDP1 run
- `fully_shard` moves a CPU-resident model wholesale to the GPU before sharding, so a 60GB+ model needs `--fsdp_cpu_ram_efficient_loading True` (meta init + rank-0 broadcast) to avoid a per-rank VRAM OOM

```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" \
    --use_fsdp --fsdp_version 2 \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap BaseMiniMaxH3TransformerBlock,MiniMaxH3ControlTransformerBlock \
    --fsdp_reshard_after_forward true \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_cpu_ram_efficient_loading True \
      scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  ... # the same train_control.py arguments as the Quick Start
```

### 3.7 Other Backends

#### 3.7.1 Training with DeepSpeed-Zero-2

> ⚠️ **Warning**: DeepSpeed-Zero-2 only partitions optimizer states and gradients, **not the model weights**. The MiniMax-H3 transformer is about 62 GB, so each GPU still holds a full weight replica and this setup usually runs out of memory. Prefer FSDP (**3.3**) for MiniMax-H3 control training; the command below is provided for reference only.

```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  ... # the same train_control.py arguments as the Quick Start
```

#### 3.7.2 Training without DeepSpeed or FSDP

**This approach is not recommended as it lacks memory-saving backends and can easily cause out-of-memory issues**. With MiniMax-H3's ~62 GB transformer and ~62 GB Qwen3-VL conditioner replicated on every GPU, it will almost certainly run out of memory. This is provided for reference only.

```bash
accelerate launch --mixed_precision="bf16" scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  ... # the same train_control.py arguments as the Quick Start
```

### 3.8 Multi-node Distributed Training

**Suitable for**: Ultra-large datasets, faster training speed

#### 3.8.1 Environment Configuration

Assuming 2 machines, each with 8 GPUs:

**Machine 0 (Master)**:
```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total machines
export NUM_PROCESS=16                # Total processes = machines x 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK \
    --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap BaseMiniMaxH3TransformerBlock,MiniMaxH3ControlTransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
      scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  ... # the same train_control.py arguments as the Quick Start
```

**Machine 1 (Worker)**:
```bash
export RANK=1  # Note: this is 1
# Other environment variables identical to Machine 0

# Use the same accelerate launch command as Machine 0
```

#### 3.8.2 Multi-node Training Notes

- **Network Requirements**:
   - RDMA/InfiniBand recommended (high performance)
   - Without RDMA, add environment variables:
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **Data Sync**: All machines must access the same data path (NFS/shared storage)

### 3.9 CFG Distillation of the Control Branch

`scripts/minimax_h3_fun/train_control_distill.py` distills classifier-free guidance into the control branch (mirroring `scripts/flux2_fun/train_control_distill.py`): a frozen teacher copy of the control transformer runs two forward passes per step — one on the prompt, one on the empty negative prompt — and the two predictions combine with `--real_guidance_scale` into the target the trainable student regresses onto with an MSE loss over the video and audio rows alike. Both copies load the same trained control branch, so point `--transformer_path` at it — either the whole transformer safetensors saved by `train_control.py` or the control-only file extracted in **3.10** (the loader is `strict=False` on missing keys).

```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" \
    --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap BaseMiniMaxH3TransformerBlock,MiniMaxH3ControlTransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
      scripts/minimax_h3_fun/train_control_distill.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=311 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_control_distill" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --enable_inpaint \
  --uniform_sampling \
  --trainable_modules "control" \
  --transformer_path="output_dir_minimax_h3_control_inpaint/checkpoint-xxx/transformer/diffusion_pytorch_model_control.safetensors" \
  --real_guidance_scale=3.5 \
  --resume_from_checkpoint=latest
```

**Key distillation parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config_path` | The same control branch YAML the source checkpoint was trained with (see 3.2) | `config/minimax_h3/minimax_h3_control.yaml` |
| `--transformer_path` | Trained control branch checkpoint loaded by both the student and the frozen teacher; accepts the whole transformer safetensors or the control-only file of **3.10** | `output_dir_minimax_h3_control_inpaint/checkpoint-xxx/transformer/diffusion_pytorch_model_control.safetensors` |
| `--enable_inpaint` | Keep the flag of the source checkpoint: the distill recipe feeds the same mask channels through student and teacher | - |
| `--real_guidance_scale` | CFG scale of the teacher's two-pass score | 3.5 |
| `--learning_rate` | A lower learning rate than control training is recommended | 2e-06 |

> 💡 The launch script `scripts/minimax_h3_fun/train_control_distill.sh` can be used as a starting point. The released MiniMax-H3 checkpoint is guidance-distilled, so the student takes no guidance input; the guidance enters only through the teacher's two CFG forward passes. `--low_vram` of control training is supported here too, but the frozen teacher then moves through the GPU on every step, so the launch script leaves it off.

### 3.10 Extracting Control-only Weights

`train_control.py` saves the whole transformer (main branch + control branch). To ship only the control branch (`control_blocks.*` + `control_proj_in.*`) as a standalone safetensors file:

```bash
python scripts/minimax_h3_fun/extract_control_weights.py \
  --model_path output_dir_minimax_h3_control_inpaint/checkpoint-1000/transformer/diffusion_pytorch_model.safetensors \
  --output_path output_dir_minimax_h3_control_inpaint/checkpoint-1000/transformer/diffusion_pytorch_model_control.safetensors
```

> 💡 `--model_path` also accepts the `<checkpoint>/transformer` directory itself (every shard is merged, and `control_blocks_places` / `control_in_dim` are carried over from its `config.json` as safetensors metadata).

The extracted file can be re-applied onto a fresh base model with `MiniMaxH3ControlTransformer3DModel.materialize_missing_control_params(...)`, and is directly loadable through `--transformer_path` of `train_control.py` / `train_control_distill.py` and `transformer_path` of the predict scripts (`strict=False` on missing keys).

---

## 4. Inference Testing

### 4.1 Inference Parameter Reference

**Key Parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `GPU_memory_mode` | GPU memory management mode, see table below | `model_group_offload` |
| `ulysses_degree` | Head dimension parallelism, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelism, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP on Transformer (main + control blocks) for multi-GPU inference | `False` |
| `fsdp_text_encoder` | Use FSDP on the Qwen3-VL text encoder (~62 GB); wraps the inner `text_encoder.model` decoder layers | `True` |
| `compile_dit` | Compile Transformer for faster inference (fixed resolution) | `False` |
| `model_name` | Base MiniMax-H3 model path | `models/Diffusion_Transformer/MiniMax-H3` |
| `config_path` | Control branch YAML, **must match the one `train_control.py` ran with** (`control_in_dim` 49 for an inpaint checkpoint). `None` builds the default 24-channel branch, which cannot load an inpaint checkpoint | `config/minimax_h3/minimax_h3_control.yaml` |
| `transformer_path` | Control checkpoint path: the released `MiniMax-H3-Fun-Controlnet-Union` weights or a trained control checkpoint. Hand it a `.safetensors` file (the whole transformer or the control-only file of **3.10** load too, `strict=False` on missing keys); a checkpoint's `transformer` folder is **not** accepted here | `models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union/MiniMax-H3-Fun-Controlnet-Union.safetensors` |
| `vae_path` | Trained VAE weight path | `None` |
| `lora_path` | LoRA weights path | `None` |
| `sample_size` | Generated video resolution `[height, width]`; height/width must be multiples of 32. Control inference fits the control video onto this canvas with the training's resize + crop geometry, so it **cannot be None** | `[704, 1280]` |
| `video_length` | Upper bound of the generation length, capped at the video VAE's `17*n+5` grid (duration stays under 15 seconds): generation follows the control video's actual length, snapped **down** to the largest valid `17*n+5` so a short control video is never padded | 243 |
| `fps` | Frames per second (MiniMax-H3 generates at a fixed 24 fps) | 24 |
| `control_context_scale` | Scale applied to every control skip before it is added to the main branch. 0.0 switches the control branch off; values below 1.0 weaken the control guidance | 1.00 |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 | `torch.bfloat16` |
| `control_video` | Control signal video path | `"asset/pose.mp4"` |
| `inpaint_video` | Source video behind the mask (only read by `--enable_inpaint` checkpoints); requires `inpaint_video_mask` | `None` |
| `inpaint_video_mask` | Mask video marking the regions to regenerate, binarized at 0.5 — white = repaint, black = keep (only for inpaint checkpoints) | `None` |
| `prompt` | Positive prompt describing content | `"视频中，一位年轻女性站在阳光洒满的沙滩上..."` |
| `negative_prompt` | Negative prompt, only used when `guidance_scale` > 1 | `"色调艳丽，过曝，静态..."` |
| `guidance_scale` | Guidance strength. The released checkpoint is guidance-distilled: keep it at 1 to run one forward pass per step with no CFG — the distill checkpoints of `train_control_distill.py` already bake the teacher's CFG target into the weights, so any value above 1 applies guidance twice and degrades the output | 1.0 |
| `num_inference_steps` | Number of denoising steps | 40 |
| `flow_shift` | Exponential sigma shift of the video schedule, `None` keeps the one of the checkpoint (12.0) | `None` |
| `audio_flow_shift` | Exponential sigma shift of the audio schedule, `None` keeps the one of the checkpoint (3.0) | `None` |
| `seed` | Random seed for reproducibility | 43 |
| `lora_weight` | LoRA weight strength | 0.55 |
| `save_path` | Generated video save path | `samples/minimax-h3-videos-v2v-control` |

**GPU Memory Management Modes**:

| Mode | Description | Memory Usage |
|------|-------------|--------------|
| `model_full_load` | Entire model loaded to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer groups switch between CPU/CUDA | Low |
| `sequential_cpu_offload` | Layer-by-layer offload (slowest) | Lowest |

> 💡 The transformer alone is 61.7 GB in bfloat16 and the Qwen3-VL conditioner is another 62.1 GB, so a single 80 GB card needs `model_cpu_offload` or `model_group_offload`.

### 4.2 V2V Control Inference

Run single-GPU inference:

```bash
python examples/minimax_h3_fun/predict_v2v_control.py
```

Modify `examples/minimax_h3_fun/predict_v2v_control.py` as needed. For first-time inference, focus on the following parameters. For other parameters, see the Inference Parameter Reference above.

```python
# Choose based on GPU memory
GPU_memory_mode = "model_group_offload"
# Set to your actual model path
model_name = "models/Diffusion_Transformer/MiniMax-H3"
# Control branch layout: must match the yaml train_control.py ran with
# (minimax_h3_control.yaml for an --enable_inpaint checkpoint, minimax_h3_control_only.yaml otherwise)
config_path = "config/minimax_h3/minimax_h3_control.yaml"
# Control checkpoint (.safetensors file): the released MiniMax-H3-Fun-Controlnet-Union weights or a checkpoint
# trained by train_control.py (the control-only file of extract_control_weights.py loads too)
transformer_path = "models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union/MiniMax-H3-Fun-Controlnet-Union.safetensors"
# Control signal video (e.g., pose video); control inference fits it onto sample_size
# with the training's resize + crop geometry
control_video = "asset/pose.mp4"
# Generated video resolution [height, width], multiples of 32, cannot be None here
sample_size = [704, 1280]
# Strength of the control guidance, 1.00 applies every control skip at full strength
control_context_scale = 1.00
# Inpaint inputs, only read by checkpoints trained with --enable_inpaint
inpaint_video = None
inpaint_video_mask = None
# Write according to generation content
prompt = "..."
# ...
```

**Notes**:
- The control branch is not part of the base MiniMax-H3 weights: point `transformer_path` at the released `MiniMax-H3-Fun-Controlnet-Union` checkpoint (or one trained by `train_control.py`) for the control video to take effect; a base `model_name` without `transformer_path` starts the side branch as an identity (`after_proj` is zero) and the control video has no effect
- Generation follows the control video's actual length, snapped down to the largest `17*n+5` the video VAE can decode and capped by `video_length` — a short control video is never padded (below 5 frames it is raised to 5). With an inpaint checkpoint but no inpaint inputs given, the pipeline zero-pads the mask channels and the run degrades to pure generation; a mask-less checkpoint rejects inpaint inputs outright
- The mask video is binarized at 0.5 before use, so grayscale masks work directly (white = repaint, black = keep)
- The generated output carries audio: videos are saved with `save_videos_with_audio_grid`

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, faster inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser yunchang
```

#### Configure Parallel Strategy

Edit `examples/minimax_h3_fun/predict_v2v_control.py`:

```python
# Ensure ulysses_degree x ring_degree = number of GPUs used
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallel
ring_degree = 1     # Sequence dimension parallel
```

**Configuration Principles**:
- `ulysses_degree` must divide the model's head count
- `ring_degree` splits along sequence dimension and affects communication overhead; avoid using if heads can divide evenly
- Multi-GPU runs through the xfuser sequence-parallel path and is **incompatible with the `*cpu_offload*` memory modes** (accelerate offload hooks own a single device); use `model_full_load` / `model_full_load_and_qfloat8` across GPUs there, with `fsdp_dit` / `fsdp_text_encoder` to save memory

**Configuration Examples**:

| GPUs | ulysses_degree | ring_degree | Note |
|------|----------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallel |
| 8 | 8 | 1 | Head parallel |
| 8 | 2 | 4 | Mixed parallel |

#### Run Multi-GPU Inference

```bash
torchrun --nproc_per_node=2 examples/minimax_h3_fun/predict_v2v_control.py
```

---

## 5. More Resources

- **MiniMax-H3 Official GitHub**: https://github.com/MiniMax-AI/MiniMax-H3
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
