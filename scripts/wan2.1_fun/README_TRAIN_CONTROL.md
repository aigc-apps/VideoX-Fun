# Wan2.1 Fun Control Full Parameter Training Guide

This document provides a complete workflow for full parameter training of Wan2.1 Fun Control Diffusion Transformer, including environment configuration, data preparation, distributed training, and inference testing.

> **Note**: Wan2.1 Fun Control is a video generation model that supports controllable video generation (e.g., pose control). This document covers the full parameter training workflow for Control model.

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
  - [3.6 Training without DeepSpeed and FSDP](#36-training-without-deepspeed-and-fsdp)
  - [3.7 Multi-Machine Distributed Training](#37-multi-machine-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameters](#41-inference-parameters)
  - [4.2 Control Video Inference](#42-control-video-inference)
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

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset for Control training, containing several training videos and their corresponding control videos (e.g., pose videos).

```bash
# Download official demo dataset
modelscope download --dataset PAI/X-Fun-Videos-Controls-Demo --local_dir ./datasets/X-Fun-Videos-Controls-Demo
```

### 2.2 Dataset Structure

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   ├── 📂 control/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

> **Note**: The `control/` directory stores control signal videos (e.g., pose videos, edge detection videos) that correspond one-to-one with videos in the `train/` directory.

### 2.3 metadata.json Format

**Relative Path Format** (example):
```json
[
  {
    "file_path": "train/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "type": "video",
    "control_file_path": "control/video001.mp4",
    "width": 1024,
    "height": 1024
  },
  {
    "file_path": "train/video002.mp4",
    "text": "A person walking through a forest, cinematic view",
    "type": "video",
    "control_file_path": "control/video002.mp4",
    "width": 1328,
    "height": 1328
  }
]
```

**Absolute Path Format**:
```json
[
  {
    "file_path": "/mnt/data/videos/sunset.mp4",
    "text": "A beautiful sunset over the ocean",
    "type": "video",
    "control_file_path": "/mnt/data/control/sunset.mp4",
    "width": 1024,
    "height": 1024
  }
]
```

**Key Field Descriptions**:
- `file_path`: Video path (relative or absolute path)
- `text`: Video description (English prompt)
- `type`: Data type, fixed as `"video"`
- `control_file_path`: Path to the corresponding control signal video (e.g., pose video), path format should be consistent with `file_path`
- `width` / `height`: Video dimensions (**recommended** to provide for bucket training. If not provided, they will be automatically read during training, which may slow down training when data is stored on slow systems like OSS).
  - You can use `scripts/process_json_add_width_and_height.py` to extract width and height fields for JSON files without them, supporting both images and videos.
  - Usage: `python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Controls-Demo/metadata.json --output_file datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json`.

### 2.4 Relative vs Absolute Path Usage

**Relative Path**:

If your data uses relative paths, configure in the training script:

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata_control.json"
```

**Absolute Path**:

If your data uses absolute paths, configure in the training script:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata_control.json"
```

> 💡 **Recommendation**: If the dataset is small and stored locally, use relative paths. If the dataset is stored on external storage (e.g., NAS, OSS) or shared across multiple machines, use absolute paths.

---

## 3. Full Parameter Training

### 3.1 Download Pretrained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download Wan2.1 Fun Control official weights
modelscope download --model PAI/Wan2.1-Fun-V1.1-14B-Control --local_dir models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-Control

modelscope download --model PAI/Wan2.1-Fun-V1.1-1.3B-Control --local_dir models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control
```

### 3.2 Quick Start (DeepSpeed-Zero-2)

After downloading the data as per **2.1 Quick Test Dataset** and the weights as per **3.1 Download Pretrained Model**, you can directly copy and run the quick start command.

We recommend using DeepSpeed-Zero-2 or FSDP for training. Here we use DeepSpeed-Zero-2 as an example.

The difference between DeepSpeed-Zero-2 and FSDP lies in whether model weights are sharded. **If you run out of GPU memory with multiple GPUs using DeepSpeed-Zero-2**, you can switch to FSDP.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-Control"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_fun/train_control.py \
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
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_fun_control" \
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
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_full_ref_image_in_self_attention \
  --trainable_modules "."
```

### 3.3 Common Training Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--config_path` | Model config file path | `config/wan2.1/wan_civitai.yaml` |
| `--pretrained_model_name_or_path` | Pretrained model path | `models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-Control` |
| `--train_data_dir` | Training data directory | `datasets/internal_datasets/` |
| `--train_data_meta` | Training data metadata file | `datasets/internal_datasets/metadata_control.json` |
| `--train_batch_size` | Batch size per GPU | 1 |
| `--image_sample_size` | Maximum image training resolution | 640 |
| `--video_sample_size` | Maximum video training resolution | 640 |
| `--token_sample_size` | Token sample size | 640 |
| `--video_sample_stride` | Video sampling stride | 2 |
| `--video_sample_n_frames` | Video sampling frames | 81 |
| `--gradient_accumulation_steps` | Gradient accumulation steps (effectively increases batch) | 1 |
| `--dataloader_num_workers` | DataLoader worker processes | 8 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 50 |
| `--learning_rate` | Initial learning rate | 2e-05 |
| `--lr_scheduler` | Learning rate scheduler | `constant_with_warmup` |
| `--lr_warmup_steps` | Learning rate warmup steps | 100 |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir_wan2.1_fun_control` |
| `--gradient_checkpointing` | Enable gradient checkpointing | - |
| `--mixed_precision` | Mixed precision: `fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon | 1e-10 |
| `--vae_mini_batch` | VAE encoding mini-batch size | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--enable_bucket` | Enable bucket training, no cropping, group by resolution | - |
| `--random_hw_adapt` | Auto-scale images/videos to random size in `[min_size, max_size]` range | - |
| `--training_with_video_token_length` | Train based on token length, supports arbitrary resolutions | - |
| `--uniform_sampling` | Uniform timestep sampling (recommended) | - |
| `--low_vram` | Low VRAM mode | - |
| `--train_mode` | Training mode: `control`, `control_ref`, `control_camera_ref` | `control_ref` |
| `--control_ref_image` | Reference image source: `first_frame` or `random` | `random` |
| `--add_full_ref_image_in_self_attention` | Add full reference image in self attention | - |
| `--resume_from_checkpoint` | Resume training path, use `"latest"` to auto-select latest checkpoint | None |
| `--validation_steps` | Run validation every N steps | 100 |
| `--validation_epochs` | Run validation every N epochs | 500 |
| `--validation_prompts` | Prompts for video generation validation | `"A woman dancing..."` |
| `--validation_paths` | Control video paths for validation | `"asset/pose.mp4"` |
| `--trainable_modules` | Trainable modules (`"."` means all modules) | `"."` |

**Sample Size Configuration Guide**:
- `video_sample_size` represents the resolution size of videos; when `random_hw_adapt` is True, it represents the minimum value between video and image resolutions.
- `image_sample_size` represents the resolution size of images; when `random_hw_adapt` is True, it represents the maximum value between video and image resolutions.
- `token_sample_size` represents the resolution corresponding to the maximum token length when `training_with_video_token_length` is True.
- Due to potential confusion in configuration, **if you don't require arbitrary resolution for finetuning**, it is recommended to set `video_sample_size`, `image_sample_size`, and `token_sample_size` to the same fixed value, such as **(320, 480, 512, 640, 960)**.
  - **All set to 320** represents **240P**.
  - **All set to 480** represents **320P**.
  - **All set to 640** represents **480P**.
  - **All set to 960** represents **720P**.

**Token Length Training Guide**:
- When `training_with_video_token_length` is enabled, the model trains based on token length.
- For example: a video with 640x640 resolution and 81 frames has a token length of approximately 40,960, requiring `token_sample_size = 640`.
  - At 640x640 resolution, the number of video frames is 81.
  - At 832x480 resolution, the number of video frames is approximately 75 (~= 640 * 640 * 81 / 832 / 480).
  - These resolutions combined with their corresponding frame counts allow the model to generate videos of different sizes.

**Training Mode Guide**:
- `train_mode="control"`: Standard Control mode, uses control video to guide generation.
- `train_mode="control_ref"`: Control + Reference Image mode, adds reference image information on top of control video.
- `train_mode="control_camera_ref"`: Control + Camera Motion + Reference Image mode.
- `control_ref_image`: In `control_ref` mode, select the reference image source. `first_frame` uses the first frame of the video, `random` uses a random frame.

### 3.4 Training Validation

You can configure validation parameters to periodically generate test videos during training, allowing you to monitor training progress and model quality.

**Validation Parameter Descriptions**:

| Parameter | Description | Recommended Value |
|------|------|--------|
| `--validation_steps` | Run validation every N steps | 100 |
| `--validation_epochs` | Run validation every N epochs | 500 |
| `--validation_prompts` | Prompts for video generation validation, space-separated for multiple prompts | Multiple space-separated prompts |
| `--validation_paths` | Control video paths for validation, corresponding one-to-one with `validation_prompts` | `"asset/pose.mp4"` |

**Example**:

```bash
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="In this sunlit outdoor garden, a beautiful woman wears a knee-length white sleeveless dress, its hem swaying gently with her graceful movements like a dancing butterfly. Sunlight filters through the leaves, casting dappled shadows that highlight her soft features and clear eyes, enhancing her elegance. Every motion seems to speak of youth and vitality as she spins on the grass, her skirt fluttering around her, as if the entire garden rejoices in her dance. Colorful flowers all around—roses, chrysanthemums, lilies—sway in the breeze, releasing their fragrances and creating a relaxed and joyful atmosphere." \
  --validation_paths "asset/pose.mp4" \
```

**Notes**:
- Validation videos are saved to the `output_dir` directory
- Multiple prompts format: `--validation_prompts "prompt1" "prompt2" "prompt3"`
- Multiple control videos format: `--validation_paths "path1.mp4" "path2.mp4" "path3.mp4"`
- The number of `validation_prompts` and `validation_paths` must correspond one-to-one

### 3.5 Training with FSDP

**If you run out of GPU memory with multiple GPUs using DeepSpeed-Zero-2**, you can switch to FSDP.

```sh
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-Control"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.1_fun/train_control.py \
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
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_fun_control" \
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
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_full_ref_image_in_self_attention \
  --trainable_modules "."
```

### 3.6 Training without DeepSpeed and FSDP

**This approach is not recommended due to lack of memory-saving backends, which may easily cause out-of-memory errors**. Provided here only for reference.

```sh
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-Control"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.1_fun/train_control.py \
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
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_fun_control" \
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
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_full_ref_image_in_self_attention \
  --trainable_modules "."
```

### 3.7 Multi-Machine Distributed Training

**Suitable for**: Ultra-large datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines, each with 8 GPUs:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-Control"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_fun/train_control.py \
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
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_fun_control" \
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
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_full_ref_image_in_self_attention \
  --trainable_modules "."
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-Control"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
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
| `config_path` | Model config file path | `config/wan2.1/wan_civitai.yaml` |
| `GPU_memory_mode` | GPU memory mode, see table below for options | `sequential_cpu_offload` |
| `ulysses_degree` | Ulysses parallelism degree for multi-GPU inference | 1 |
| `ring_degree` | Ring parallelism degree for multi-GPU inference | 1 |
| `fsdp_dit` | Use FSDP for Transformer during multi-GPU inference to save memory | `False` |
| `fsdp_text_encoder` | Use FSDP for text encoder during multi-GPU inference | `True` |
| `compile_dit` | Compile Transformer for faster inference (effective at fixed resolution) | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control` |
| `sampler_name` | Sampler type: `Flow`, `Flow_Unipc`, `Flow_DPM++` | `Flow` |
| `transformer_path` | Path to trained Transformer weights | `None` |
| `vae_path` | Path to trained VAE weights | `None` |
| `lora_path` | LoRA weights path | `None` |
| `sample_size` | Generated video resolution `[height, width]` | `[832, 480]` |
| `video_length` | Number of generated frames | `49` |
| `fps` | Frames per second | `16` |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 support | `torch.bfloat16` |
| `control_video` | Control signal video path (e.g., pose video) | `"asset/pose.mp4"` |
| `control_camera_txt` | Camera motion control txt file path | `None` |
| `ref_image` | Reference image path | `"asset/6.png"` |
| `start_image` | Start image path (alternative to ref_image) | `None` |
| `prompt` | Positive prompt describing generated content | `"A young woman..."` |
| `negative_prompt` | Negative prompt to avoid certain content | `"Blurring, mutation..."` |
| `guidance_scale` | Guidance strength | 6.0 |
| `seed` | Random seed for reproducibility | 43 |
| `num_inference_steps` | Number of inference steps | 50 |
| `lora_weight` | LoRA weight strength | 0.55 |
| `save_path` | Path to save generated videos | `samples/wan-videos-fun-control` |

**GPU Memory Mode Descriptions**:

| Mode | Description | Memory Usage |
|------|------|---------|
| `model_full_load` | Load entire model to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer groups offloaded between CPU/CUDA | Low |
| `sequential_cpu_offload` | Sequential offload layer by layer (slowest) | Lowest |

### 4.2 Control Video Inference

Run single-GPU inference:

```bash
python examples/wan2.1_fun/predict_v2v_control_ref.py
```

Edit `examples/wan2.1_fun/predict_v2v_control_ref.py` according to your needs. For first-time inference, focus on the following key parameters. For other parameters, refer to the inference parameter descriptions above.

```python
# Model config file path
config_path = "config/wan2.1/wan_civitai.yaml"
# Choose based on GPU memory
GPU_memory_mode = "sequential_cpu_offload"
# Your actual model path
model_name = "models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"  
# Path to trained weights, e.g., "output_dir_wan2.1_fun_control/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = None  
# Control signal video path (e.g., pose video)
control_video = "asset/pose.mp4"
# Reference image path
ref_image = "asset/6.png"
# Write based on your generation content
prompt = "A young woman wearing a pink dress..."  
# ...
```

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/wan2.1_fun/predict_v2v_control_ref.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs used
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelism
ring_degree = 1     # Sequence dimension parallelism
```

**Configuration Principles**:
- `ulysses_degree` must evenly divide the model's head count
- `ring_degree` splits on the sequence dimension, which affects communication overhead. Try to avoid using it when heads can be evenly divided.

**Configuration Examples**:

| GPU Count | ulysses_degree | ring_degree | Description |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelism |
| 8 | 8 | 1 | Head parallelism |
| 8 | 4 | 2 | Hybrid parallelism |

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/wan2.1_fun/predict_v2v_control_ref.py
```

---

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
