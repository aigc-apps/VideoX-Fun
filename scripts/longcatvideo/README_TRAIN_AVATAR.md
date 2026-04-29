# LongCat-Video-Avatar Full Parameter Training Guide

This document provides a complete workflow for full parameter training of LongCat-Video-Avatar (audio-driven long-form talking head video generation model), including environment configuration, data preparation, distributed training, and inference testing.

> **Note**: LongCat-Video-Avatar is an audio-driven long-form talking head video generation model that requires both a reference image and an audio file to generate talking videos. The training data needs to include video and audio.

---

## Table of Contents
- [1. Environment Configuration](#1-environment-configuration)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. Full Parameter Training](#3-full-parameter-training)
  - [3.1 Download Pretrained Model](#31-download-pretrained-model)
  - [3.2 Quick Start (DeepSpeed-Zero-2)](#32-quick-start-deepspeed-zero-2)
  - [3.3 Common Training Parameters](#33-common-training-parameters)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training with FSDP](#35-training-with-fsdp)
  - [3.6 Training Without DeepSpeed or FSDP](#36-training-without-deepspeed-or-fsdp)
  - [3.7 Multi-Machine Distributed Training](#37-multi-machine-distributed-training)
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
pip install yunchang xfuser modelscope openpyxl deepspeed==0.17.0 numpy==1.26.4
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

**Method 3: Using Docker**

When using Docker, please ensure that the GPU driver and CUDA environment are correctly installed on your machine, then execute the following commands:

```bash
# Pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# Enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset containing several audio-video training samples.

```bash
# Download official demo dataset
modelscope download --dataset PAI/X-Fun-Videos-Audios-Demo --local_dir ./datasets/X-Fun-Videos-Audios-Demo
```

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

> ⚠️ **Important**: LongCat-Video-Avatar is an audio-driven long-form talking head video generation model. Unlike regular video training, you **must provide the `audio_path` field in metadata.json**.

**Relative Path Format** (example):
```json
[
  {
    "file_path": "train/video001.mp4",
    "audio_path": "wav/audio001.wav",
    "text": "A person talking with natural expressions",
    "type": "video",
    "width": 512,
    "height": 512
  },
  {
    "file_path": "train/video002.mp4",
    "audio_path": "wav/audio002.wav",
    "text": "A speaker delivering a speech",
    "type": "video",
    "width": 512,
    "height": 512
  }
]
```

**Absolute Path Format**:
```json
[
  {
    "file_path": "/mnt/data/videos/speaker.mp4",
    "audio_path": "/mnt/data/wavs/speaker.wav",
    "text": "A person talking with natural expressions",
    "type": "video",
    "width": 512,
    "height": 512
  }
]
```

**Key Fields Description**:
- `file_path`: Video file path (relative or absolute)
- `audio_path`: Audio file path (**required**, main difference from regular video training)
  - Audio files are typically in `.wav` format
  - Path should correspond to `file_path`, e.g., `train/video001.mp4` corresponds to `wav/audio001.wav`
- `text`: Video description (prompt, optional)
- `type`: Data type, fixed as `"video"`
- `width` / `height`: Video dimensions (**recommended** to provide for bucket training; if not provided, they will be automatically read during training, which may slow down training when data is stored on slow systems like OSS)
  - You can use `scripts/process_json_add_width_and_height.py` to add width and height fields to JSON files without these fields, supporting both images and videos
  - Usage: `python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Audios-Demo/metadata.json --output_file datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json`

**Dataset Comparison: LongCat-Video-Avatar vs Regular Video Training**:

| Model Type | Required Fields | Audio Field |
|---------|---------|---------|
| Regular Video (WAN, CogVideoX, etc.) | `file_path`, `text`, `type` | ❌ Not needed |
| **LongCat-Video-Avatar (Audio-Driven Generation)** | `file_path`, `audio_path`, `type` | ✅ **Required** |

### 2.4 Relative vs Absolute Path Usage

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

> 💡 **Recommendation**: If the dataset is small and stored locally, use relative paths. If the dataset is stored on external storage (e.g., NAS, OSS) or shared across multiple machines, use absolute paths.

---

## 3. Full Parameter Training

### 3.1 Download Pretrained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer
mkdir -p models/Personalized_Model

# Download LongCat-Video base model
modelscope download --model meituan-longcat/LongCat-Video --local_dir models/Diffusion_Transformer/LongCat-Video

# Download LongCat-Video-Avatar model
modelscope download --model meituan-longcat/LongCat-Video-Avatar --local_dir models/Diffusion_Transformer/LongCat-Video-Avatar
```

### 3.2 Quick Start (DeepSpeed-Zero-2)

If you have downloaded the data and weights as per the previous sections, you can directly copy and run the quick start command.

DeepSpeed-Zero-2 and FSDP are recommended for training. Here we use DeepSpeed-Zero-2 as an example.

The difference between DeepSpeed-Zero-2 and FSDP lies in whether the model weights are sharded. **If VRAM is insufficient when using multiple GPUs with DeepSpeed-Zero-2**, you can switch to FSDP.

```bash
export MODEL_NAME="models/Diffusion_Transformer/LongCat-Video"
export MODEL_NAME_AVATAR="models/Diffusion_Transformer/LongCat-Video-Avatar"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/longcatvideo/train_avatar.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_avatar_model_name_or_path=$MODEL_NAME_AVATAR \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
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
  --output_dir="output_dir_longcat_avatar" \
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
  --trainable_modules "."
```

### 3.3 Common Training Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--pretrained_model_name_or_path` | Path to pretrained base model | `models/Diffusion_Transformer/LongCat-Video` |
| `--pretrained_avatar_model_name_or_path` | Path to pretrained avatar model | `models/Diffusion_Transformer/LongCat-Video-Avatar` |
| `--train_data_dir` | Training data directory | `datasets/internal_datasets/` |
| `--train_data_meta` | Training data metadata file | `datasets/internal_datasets/metadata.json` |
| `--train_batch_size` | Samples per batch | 1 |
| `--video_sample_size` | Maximum video resolution for training | 640 |
| `--token_sample_size` | Token length sampling size | 640 |
| `--video_sample_stride` | Frame sampling stride | 2 |
| `--video_sample_n_frames` | Number of frames to sample | 81 |
| `--video_repeat` | Number of video repeats | 1 |
| `--gradient_accumulation_steps` | Gradient accumulation steps (equivalent to larger batch) | 1 |
| `--dataloader_num_workers` | DataLoader subprocesses | 8 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 50 |
| `--learning_rate` | Initial learning rate | 2e-05 |
| `--lr_scheduler` | Learning rate scheduler | `constant_with_warmup` |
| `--lr_warmup_steps` | Learning rate warmup steps | 100 |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir_longcat_avatar` |
| `--gradient_checkpointing` | Enable activation checkpointing | - |
| `--mixed_precision` | Mixed precision: `fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon value | 1e-10 |
| `--vae_mini_batch` | Mini-batch size for VAE encoding | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--random_hw_adapt` | Auto-scale videos to random size in range `[512, video_sample_size]` | - |
| `--training_with_video_token_length` | Train based on token length instead of fixed resolution | - |
| `--enable_bucket` | Enable bucket training: trains entire videos grouped by resolution without center cropping | - |
| `--uniform_sampling` | Uniform timestep sampling | - |
| `--low_vram` | Enable low VRAM optimizations | - |
| `--resume_from_checkpoint` | Resume training from checkpoint path, use `"latest"` to auto-select latest | None |
| `--transformer_path` | Pretrained Transformer weights path | `None` |
| `--trainable_modules` | Trainable modules (`"."` means all modules) | `"."` |
| `--validation_steps` | Execute validation every N steps | 100 |
| `--validation_epochs` | Execute validation every N epochs | 500 |
| `--validation_image_paths` | Validation reference image paths, use multiple space-separated paths | `"asset/8.png"` |
| `--validation_audio_paths` | Validation audio paths, use multiple space-separated paths | `"asset/talk.wav"` |


### 3.4 Training Validation

You can configure validation parameters to periodically generate test videos during training, allowing you to monitor training progress and model quality.

**Validation Parameters**:

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--validation_steps` | Execute validation every N steps | 100 |
| `--validation_epochs` | Execute validation every N epochs | 500 |
| `--validation_image_paths` | Reference image paths for validation, use multiple space-separated paths | Multiple space-separated image paths |
| `--validation_audio_paths` | Audio paths for validation, use multiple space-separated paths | Multiple space-separated audio paths |
| `--validation_prompts` | Prompt for validation video generation, use multiple space-separated prompts | Multiple space-separated prompts |

**Example**:

```bash
  --validation_image_paths "asset/8.png" \
  --validation_audio_paths "asset/talk.wav" \
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="A young woman with long flowing purple hair stands by the seaside on a sunny day, singing. Wearing a white sleeveless dress with a navy blue bow at the collar, her hair gently sways in the ocean breeze. The sparkling sea, blue sky with white clouds, and pink wildflowers along the shore create a beautiful and vibrant scene."
```

**Notes**:
- Validation videos will be saved to the `output_dir/sample` directory
- Image paths, audio paths, and prompts must correspond one-to-one

### 3.5 Training with FSDP

**If VRAM is insufficient when using multiple GPUs**, you can switch to FSDP.

```bash
export MODEL_NAME="models/Diffusion_Transformer/LongCat-Video"
export MODEL_NAME_AVATAR="models/Diffusion_Transformer/LongCat-Video-Avatar"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap LongCatAvatarSingleStreamBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/longcatvideo/train_avatar.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_avatar_model_name_or_path=$MODEL_NAME_AVATAR \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
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
  --output_dir="output_dir_longcat_avatar" \
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
  --trainable_modules "."
```

### 3.6 Training Without DeepSpeed or FSDP

**This approach is not recommended as it lacks VRAM-saving backends and may easily cause out-of-memory errors**. This is provided for reference only.

```bash
export MODEL_NAME="models/Diffusion_Transformer/LongCat-Video"
export MODEL_NAME_AVATAR="models/Diffusion_Transformer/LongCat-Video-Avatar"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/longcatvideo/train_avatar.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_avatar_model_name_or_path=$MODEL_NAME_AVATAR \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
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
  --output_dir="output_dir_longcat_avatar" \
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
  --trainable_modules "."
```

### 3.7 Multi-Machine Distributed Training

**Suitable for**: Ultra-large-scale datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines with 8 GPUs each:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/LongCat-Video"
export MODEL_NAME_AVATAR="models/Diffusion_Transformer/LongCat-Video-Avatar"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/longcatvideo/train_avatar.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_avatar_model_name_or_path=$MODEL_NAME_AVATAR \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
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
  --output_dir="output_dir_longcat_avatar" \
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
  --trainable_modules "."
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/LongCat-Video"
export MODEL_NAME_AVATAR="models/Diffusion_Transformer/LongCat-Video-Avatar"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
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

#### 3.7.2 Multi-Machine Training Notes

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
| `GPU_memory_mode` | GPU memory mode, see table below for options | `model_group_offload` |
| `ulysses_degree` | Head dimension parallelization degree, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelization degree, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP for Transformer in multi-GPU inference to save VRAM | `False` |
| `fsdp_text_encoder` | Use FSDP for text encoder in multi-GPU inference | `True` |
| `compile_dit` | Compile Transformer to accelerate inference (effective at fixed resolution) | `False` |
| `model_name` | Base model path | `models/Diffusion_Transformer/LongCat-Video` |
| `model_name_avatar` | Avatar model path | `models/Diffusion_Transformer/LongCat-Video-Avatar` |
| `sampler_name` | Sampler type: `Flow`, `Flow_Unipc`, `Flow_DPM++` | `Flow` |
| `transformer_path` | Path to trained Transformer weights | `None` |
| `vae_path` | Path to trained VAE weights | `None` |
| `lora_path` | LoRA weights path | `None` |
| `sample_size` | Generated video resolution `[height, width]` | `[832, 480]` |
| `video_length` | Number of frames to generate | 81 |
| `fps` | Frames per second | 16 |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 support | `torch.bfloat16` |
| `validation_image_start` | Reference image path | `"asset/8.png"` |
| `audio_path` | Input audio path | `"asset/talk.wav"` |
| `prompt` | Generation prompt | `"A young woman with long flowing purple hair..."` |
| `negative_prompt` | Negative prompt | See code |
| `guidance_scale` | Prompt guidance strength | 4.5 |
| `seed` | Random seed for reproducibility | 43 |
| `num_inference_steps` | Inference steps | 25 |
| `lora_weight` | LoRA weight strength | 0.55 |
| `save_path` | Generated video save path | `samples/longcat-avatar-videos-t2v` |

**GPU Memory Mode Descriptions**:

| Mode | Description | VRAM Usage |
|------|------|---------|
| `model_full_load` | Load entire model to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer group offload between CPU/CUDA | Low |
| `sequential_cpu_offload` | Offload each layer individually (slowest) | Lowest |

### 4.2 Single GPU Inference

Run single GPU inference with:

```bash
python examples/longcatvideo/predict_s2v_avatar.py
```

Edit `examples/longcatvideo/predict_s2v_avatar.py` according to your needs. For first-time inference, focus on these parameters. For other parameters, see the Inference Parameters section above.

```python
# Choose based on your GPU VRAM
GPU_memory_mode = "model_group_offload"
# Your actual model paths
model_name = "models/Diffusion_Transformer/LongCat-Video"
model_name_avatar = "models/Diffusion_Transformer/LongCat-Video-Avatar"
# Trained weights path (optional), e.g. "output_dir_longcat_avatar/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = None
# Reference image path
validation_image_start = "asset/8.png"
# Input audio path
audio_path = "asset/talk.wav"
# Generation prompt
prompt = "A young woman with long flowing purple hair stands by the seaside on a sunny day, singing..."
# ...
```

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/longcatvideo/predict_s2v_avatar.py`:

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
| 8 | 2 | 4 | Hybrid parallelization |
| 8 | 8 | 1 | Head parallelization |

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/longcatvideo/predict_s2v_avatar.py
```

---

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
