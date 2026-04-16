# MOVA Full Training Guide

This document provides a complete workflow for MOVA (Audio-Video Generation Model) full parameter training, including environment setup, data preparation, distributed training, and inference testing.

> **Note**: MOVA is an audio-video generation model that can simultaneously generate video and corresponding audio. Training data requires both video and audio files.

---
## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Using Relative and Absolute Paths](#24-using-relative-and-absolute-paths)
- [3. Full Parameter Training](#3-full-parameter-training)
  - [3.1 Download Pretrained Model](#31-download-pretrained-model)
  - [3.2 Quick Start (DeepSpeed-Zero-2)](#32-quick-start-deepspeed-zero-2)
  - [3.3 Common Training Parameters](#33-common-training-parameters)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training with FSDP](#35-training-with-fsdp)
  - [3.6 Training without DeepSpeed or FSDP](#36-training-without-deepspeed-or-fsdp)
  - [3.7 Multi-Node Distributed Training](#37-multi-node-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameters](#41-inference-parameters)
  - [4.2 Single GPU Inference](#42-single-gpu-inference)
  - [4.3 Multi-GPU Parallel Inference](#43-multi-gpu-parallel-inference)
- [5. Additional Resources](#5-additional-resources)


## 1. Environment Setup

**Option 1: Using requirements.txt**

```bash
pip install -r requirements.txt
```

**Option 2: Manual Installation**

```bash
pip install Pillow einops safetensors timm tomesd librosa transformers accelerate diffusers peft decord imageio imageio-ffmpeg moviepy ftfy tensorboard sentencepiece modelscope
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

For testing purposes, you can use the demo dataset from Hugging Face:

```bash
# Create dataset directory
mkdir -p datasets

# Download demo dataset from Hugging Face
cd datasets
git clone https://huggingface.co/datasets/modelscope/X-Fun-Videos-Audios-Demo
cd ..
```

This dataset contains 17 video-audio pairs for quick testing.

### 2.2 Dataset Structure

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   ├── 📂 wav/
│   │   ├── 📄 audio001.wav
│   │   ├── 📄 audio002.wav
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json Format

> ⚠️ **Important**: MOVA is an audio-video generation model. Unlike normal video training, **you must provide the `audio_path` field in metadata.json**.

**Relative Path Format** (Example):
```json
[
  {
    "file_path": "train/video001.mp4",
    "audio_path": "wav/audio001.wav",
    "text": "A brown dog barks on a sofa, sitting on a light-colored couch in a cozy room",
    "type": "video",
    "width": 768,
    "height": 512
  },
  {
    "file_path": "train/video002.mp4",
    "audio_path": "wav/audio002.wav",
    "text": "A group of young men in suits and sunglasses are walking down a city street",
    "type": "video",
    "width": 640,
    "height": 640
  }
]
```

**Absolute Path Format**:
```json
[
  {
    "file_path": "/mnt/data/videos/dog.mp4",
    "audio_path": "/mnt/data/wavs/dog.wav",
    "text": "A brown dog barks on a sofa",
    "type": "video",
    "width": 768,
    "height": 512
  }
]
```

**Key Fields**:
- `file_path`: Video file path (relative or absolute)
- `audio_path`: Audio file path (**MOVA-specific and required**, the main difference from regular video training)
  - Audio files are typically in `.wav` format
  - The path should correspond to `file_path`, e.g., `train/video001.mp4` corresponds to `wav/audio001.wav`
- `text`: Video description (English prompt)
- `type`: Data type, fixed as `"video"`
- `width` / `height`: Video dimensions (**recommended** to enable bucket training; if not provided, they will be automatically read during training)

### 2.4 Using Relative and Absolute Paths

**Relative Paths**:

If your data uses relative paths, configure the training script as follows:

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

**Absolute Paths**:

If your data uses absolute paths, configure the training script as follows:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **Tip**: If the dataset is small and stored locally, use relative paths. If the dataset is stored on external storage (e.g., NAS, OSS) or shared across multiple machines, use absolute paths.

---

## 3. Full Parameter Training

### 3.1 Download Pretrained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download MOVA official weights (ModelScope)
modelscope download --model modelscope/MOVA-360p --local_dir models/Diffusion_Transformer/MOVA-360p

# Or use Hugging Face
# hf download modelscope/MOVA-360p --local-dir models/Diffusion_Transformer/MOVA-360p
```

### 3.2 Quick Start (DeepSpeed-Zero-2)

If you have downloaded the data as per **2.1 Quick Test Dataset** and the weights as per **3.1 Download Pretrained Model**, you can directly copy and run the quick start command.

FSDP training is recommended as it can significantly save VRAM.

```bash
export MODEL_NAME="models/Diffusion_Transformer/MOVA-360p"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi-node environments without RDMA
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/mova/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=480 \
  --video_sample_size=480 \
  --token_sample_size=480 \
  --video_sample_stride=1 \
  --video_sample_n_frames=193 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_mova" \
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
  --low_vram \
  --trainable_modules "." \
  --boundary_type="low" \
  --boundary_ratio=0.9 \
  --train_components="transformer,transformer_2"
```

### 3.3 Common Training Parameters

**Core Parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--pretrained_model_name_or_path` | Pretrained model path | `models/Diffusion_Transformer/MOVA-360p` |
| `--train_data_dir` | Training data directory | `datasets/internal_datasets/` |
| `--train_data_meta` | Training data metadata file | `datasets/internal_datasets/metadata.json` |
| `--train_batch_size` | Number of samples per batch | 1 |
| `--image_sample_size` | Maximum training resolution for auto bucket | 480 |
| `--video_sample_size` | Maximum video training resolution | 480 |
| `--token_sample_size` | Token length sampling size | 480 |
| `--video_sample_stride` | Frame sampling stride | 1 |
| `--video_sample_n_frames` | Number of video frames to sample | 193 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--learning_rate` | Learning rate | 2e-05 |
| `--output_dir` | Output directory for checkpoints | `output_dir_mova` |
| `--checkpointing_steps` | Save checkpoint every N steps | 100 |

**Some parameters in the sh file can be confusing, and they are explained in this document:**

- `enable_bucket`: Used to enable bucket training. When enabled, the model does not crop the videos at the center, but instead trains the videos after grouping them into buckets based on resolution.
- `random_frame_crop`: Used for random cropping on video frames to simulate videos with different frame counts.
- `random_hw_adapt`: Used to enable automatic height and width scaling for videos. When `random_hw_adapt` is enabled, for training videos, the height and width will be set to `video_sample_size` as the maximum and `512` as the minimum.
  - For example, when `random_hw_adapt` is enabled, with `video_sample_n_frames=49`, `video_sample_size=768`, the resolution of video inputs for training is `512x512x49`, `768x768x49`.
- `training_with_video_token_length`: Specifies training the model according to token length. For training videos, the height and width will be set to `video_sample_size` as the maximum and `256` as the minimum.
  - For example, when `training_with_video_token_length` is enabled, with `video_sample_n_frames=49`, `token_sample_size=512`, `video_sample_size=768`, the resolution of video inputs for training is `256x256x49`, `512x512x49`, `768x768x21`.
  - The token length for a video with dimensions 512x512 and 49 frames is 13,312. We need to set the `token_sample_size = 512`.
    - At 512x512 resolution, the number of video frames is 49 (~= 512 * 512 * 49 / 512 / 512).
    - At 768x768 resolution, the number of video frames is 21 (~= 512 * 512 * 49 / 768 / 768).
    - At 1024x1024 resolution, the number of video frames is 9 (~= 512 * 512 * 49 / 1024 / 1024).
    - These resolutions combined with their corresponding lengths allow the model to generate videos of different sizes.
- `resume_from_checkpoint`: Used to set whether training should be resumed from a previous checkpoint. Use a path or `"latest"` to automatically select the last available checkpoint.
- `trainable_modules`: Represents the modules to be trained, using "." means training all modules.
- `boundary_type`: Specifies which DiT to train: "low" = only low-noise DiT, "high" = only high-noise DiT, "full" = both DiTs.
- `boundary_ratio`: Is the boundary ratio for switching between high-noise and low-noise DiT. Timesteps below this ratio use low-noise DiT (default: 0.9).
- `train_components`: Specifies which components to train. Comma-separated list of: "transformer", "transformer_2", "transformer_audio", "dual_tower_bridge", or "all".
- `i2v_ratio`: Is the ratio of I2V samples in training. 0.0 = pure T2V, 1.0 = pure I2V, 0.5 = 50% T2V + 50% I2V (default).
- `low_vram`: Enables low VRAM mode to reduce memory usage.

### 3.4 Training Validation

During training, the model will automatically generate validation videos to monitor training progress. You can find these videos in the `output_dir` directory.

To manually validate a trained model:

```bash
python examples/mova/predict_i2v.py
```

Edit the script to load your trained checkpoint and generate test videos.

### 3.5 Training with FSDP

```bash
export MODEL_NAME="models/Diffusion_Transformer/MOVA-360p"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock,AudioWanAttentionBlock,ConditionalCrossAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/mova/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=480 \
  --video_sample_size=480 \
  --token_sample_size=480 \
  --video_sample_stride=1 \
  --video_sample_n_frames=193 \
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
  --output_dir="output_dir_mova" \
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
  --low_vram \
  --trainable_modules "." \
  --boundary_type="low" \
  --boundary_ratio=0.9 \
  --train_components="transformer,transformer_2"
```

### 3.6 Training without DeepSpeed or FSDP

```bash
export MODEL_NAME="models/Diffusion_Transformer/MOVA-360p"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/mova/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=480 \
  --video_sample_size=480 \
  --token_sample_size=480 \
  --video_sample_stride=1 \
  --video_sample_n_frames=193 \
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
  --output_dir="output_dir_mova" \
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
  --low_vram \
  --trainable_modules "." \
  --boundary_type="low" \
  --boundary_ratio=0.9 \
  --train_components="transformer,transformer_2"
```

### 3.7 Multi-Node Distributed Training

When training with multiple machines, set the parameters as follows:

```bash
export MASTER_ADDR="your master address"
export MASTER_PORT=10086
export WORLD_SIZE=1 # The number of machines
export NUM_PROCESS=8 # The number of processes, such as WORLD_SIZE * 8
export RANK=0 # The rank of this machine

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK scripts/mova/train.py
```

---

## 4. Inference Testing

### 4.1 Inference Parameters

**Core Parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `GPU_memory_mode` | GPU memory mode, see table below for options | `sequential_cpu_offload` |
| `ulysses_degree` | Head dimension parallelism degree, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelism degree, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP for Transformer during multi-GPU inference to save VRAM | `False` |
| `fsdp_text_encoder` | Use FSDP for text encoder during multi-GPU inference | `True` |
| `compile_dit` | Compile Transformer for faster inference (effective for fixed resolution, not compatible with sequential_cpu_offload) | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/MOVA-360p` |
| `sampler_name` | Sampler type: `Flow`, `Flow_Unipc`, `Flow_DPM++` | `Flow` |
| `boundary_ratio` | Boundary ratio for switching between high-noise and low-noise DiT | 0.9 |
| `transformer_path` | Path to trained low-noise Transformer weights | `None` |
| `transformer_high_path` | Path to trained high-noise Transformer weights | `None` |
| `transformer_audio_path` | Path to trained audio Transformer weights | `None` |
| `bridge_path` | Path to trained dual-tower bridge weights | `None` |
| `vae_path` | Path to trained video VAE weights | `None` |
| `audio_vae_path` | Path to trained audio VAE weights | `None` |
| `lora_path` | Path to low-noise model LoRA weights | `None` |
| `lora_high_path` | Path to high-noise model LoRA weights | `None` |
| `validation_image` | Input image for I2V mode | `asset/8.png` |
| `sample_size` | Generated video resolution `[height, width]` | `[640, 352]` |
| `video_length` | Number of frames to generate | 81 |
| `fps` | Frames per second | 24 |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 (e.g., v100, 2080ti) | `torch.bfloat16` |
| `prompt` | Positive prompt describing what to generate | `"Medium shot of a girl..."` |
| `negative_prompt` | Negative prompt describing what to avoid | `"oversaturated, overexposed..."` |
| `guidance_scale` | Guidance strength | 5.0 |
| `seed` | Random seed for reproducibility | 43 |
| `num_inference_steps` | Number of inference steps | 50 |
| `lora_weight` | Low-noise model LoRA weight strength | 0.55 |
| `lora_high_weight` | High-noise model LoRA weight strength | 0.55 |
| `save_path` | Path to save generated videos | `samples/mova-videos-i2v` |
| `audio_sample_rate` | Audio sample rate (read from vocoder config) | 24000 |

**GPU Memory Mode Description**:

| Mode | Description | Memory Usage |
|------|-------------|--------------|
| `model_full_load` | Load entire model to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Transfer layer groups between CPU/CUDA | Low |
| `sequential_cpu_offload` | Offload each layer to CPU after use (slowest) | Lowest |

### 4.2 Single GPU Inference

Run single GPU inference:

```bash
python examples/mova/predict_i2v.py
```

Edit `examples/mova/predict_i2v.py` according to your needs. For first-time inference, focus on modifying the following parameters. For other parameters, see the inference parameter description above.

```python
# Select based on your GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# Your actual model path
model_name = "models/Diffusion_Transformer/MOVA-360p"
  
# I2V input image
validation_image = "asset/8.png"
# Paths to trained weights (if needed)
transformer_path = None
transformer_high_path = None
transformer_audio_path = None
bridge_path = None
# LoRA weight paths (if needed)
lora_path = None
lora_high_path = None
# Write according to what you want to generate
prompt = "Medium shot of a girl by the ocean. She starts with a bright smile, then gently nods her head while speaking. Her mouth moves naturally to say: \"Hi, nice to meet you.\" She maintains eye contact throughout. The background shows calm waves. Smooth motion, cinematic quality, realistic facial expressions."  
# ...
```

### 4.3 Multi-GPU Parallel Inference

**Applicable Scenarios**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/mova/predict_i2v.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs used
# For example, using 8 GPUs:
ulysses_degree = 2  # Head dimension parallelism
ring_degree = 4     # Sequence dimension parallelism
```

**Configuration Principles**:
- `ulysses_degree` must be divisible by the model's number of heads
- `ring_degree` splits along the sequence dimension and affects communication overhead; avoid using it when heads can be evenly divided

**Configuration Examples**:

| GPU Count | ulysses_degree | ring_degree | Description |
|-----------|---------------|-------------|-------------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelism |
| 8 | 2 | 4 | Hybrid parallelism |
| 8 | 8 | 1 | Head parallelism |

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/mova/predict_i2v.py
```

---

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun