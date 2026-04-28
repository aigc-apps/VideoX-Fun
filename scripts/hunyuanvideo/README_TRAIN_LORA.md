# HunyuanVideo LoRA Fine-tuning Training Guide

This document provides a complete workflow for HunyuanVideo LoRA fine-tuning training, including environment configuration, data preparation, multiple distributed training strategies, and inference testing.

> **Note**: HunyuanVideo is a video generation model that supports text-to-video (T2V) and image-to-video (I2V). This guide covers the LoRA fine-tuning training workflow, suitable for custom dataset fine-tuning scenarios.

---

## Table of Contents
- [1. Environment Configuration](#1-environment-configuration)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative and Absolute Path Usage](#24-relative-and-absolute-path-usage)
- [3. LoRA Training](#3-lora-training)
  - [3.1 Download Pre-trained Model](#31-download-pre-trained-model)
  - [3.2 Quick Start (DeepSpeed-Zero-2)](#32-quick-start-deepspeed-zero-2)
  - [3.3 LoRA-Specific Parameters](#33-lora-specific-parameters)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training with FSDP](#35-training-with-fsdp)
  - [3.6 Training without DeepSpeed and FSDP](#36-training-without-deepspeed-and-fsdp)
  - [3.7 Multi-machine Distributed Training](#37-multi-machine-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameters](#41-inference-parameters)
  - [4.2 Text-to-Video (T2V) Inference](#42-text-to-video-t2v-inference)
  - [4.3 Image-to-Video (I2V) Inference](#43-image-to-video-i2v-inference)
  - [4.4 Multi-GPU Parallel Inference](#44-multi-gpu-parallel-inference)
- [5. Additional Resources](#5-additional-resources)

---

## 1. Environment Configuration

**Option 1: Using requirements.txt**

```bash
pip install -r requirements.txt
```

**Option 2: Manual dependency installation**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl deepspeed==0.17.0 numpy==1.26.4
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

**Option 3: Using Docker**

When using Docker, please ensure that the GPU driver and CUDA environment are correctly installed, then execute the following commands:

```bash
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset that contains several training samples.

```bash
# Download official example dataset
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 2.2 Dataset Structure

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json Format

**Relative path format** (example format):
```json
[
  {
    "file_path": "train/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "type": "video",
    "width": 1024,
    "height": 1024
  },
  {
    "file_path": "train/video002.mp4",
    "text": "A person walking through a forest, cinematic view",
    "type": "video",
    "width": 1328,
    "height": 1328
  }
]
```

**Absolute path format**:
```json
[
  {
    "file_path": "/mnt/data/videos/sunset.mp4",
    "text": "A beautiful sunset over the ocean",
    "type": "video",
    "width": 1024,
    "height": 1024
  }
]
```

**Key field descriptions**:
- `file_path`: Video path (relative or absolute path)
- `text`: Video description (English prompt)
- `type`: Data type, fixed as `"video"`
- `width` / `height`: Video width and height (**recommended to provide**, used for bucket training. If not provided, they will be automatically read during training, which may affect training speed when data is stored on slow systems like OSS).
  - You can use `scripts/process_json_add_width_and_height.py` to extract width and height fields from JSON files without these fields, supporting both images and videos.
  - Usage: `python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Demo/metadata.json --output_file datasets/X-Fun-Videos-Demo/metadata_add_width_height.json`.

### 2.4 Relative and Absolute Path Usage

**Relative paths**:

If your data uses relative paths, set in the training script:

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

**Absolute paths**:

If your data uses absolute paths, set in the training script:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **Recommendation**: If the dataset is small and stored locally, relative paths are recommended. If the dataset is stored on external storage (such as NAS, OSS) or shared across multiple machines, absolute paths are recommended.

---

## 3. LoRA Training

### 3.1 Download Pre-trained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download HunyuanVideo official weights
hf download hunyuanvideo-community/HunyuanVideo --local-dir models/Diffusion_Transformer/HunyuanVideo
```

### 3.2 Quick Start (DeepSpeed-Zero-2)

After downloading data according to **2.1 Quick Test Dataset** and weights according to **3.1 Download Pre-trained Model**, you can directly copy and run the quick start command.

DeepSpeed-Zero-2 and FSDP are recommended for training. Here we use DeepSpeed-Zero-2 as an example to configure the shell file.

The difference between DeepSpeed-Zero-2 and FSDP is whether to shard model weights. **If GPU memory is insufficient when using multiple GPUs with DeepSpeed-Zero-2**, you can switch to FSDP for training.

```bash
export MODEL_NAME="models/Diffusion_Transformer/HunyuanVideo"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/hunyuanvideo/train_lora.py \
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
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_hunyuanvideo_lora" \
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
  --rank=64 \
  --network_alpha=32 \
  --target_name="to_q,to_k,to_v,ff.0,ff.2,ff_context.0,ff_context.2" \
  --use_peft_lora \
  --low_vram
```

### 3.3 LoRA-Specific Parameters

**Key LoRA Parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--pretrained_model_name_or_path` | Pre-trained model path | `models/Diffusion_Transformer/HunyuanVideo` |
| `--train_data_dir` | Training data directory | `datasets/internal_datasets/` |
| `--train_data_meta` | Training data metadata file | `datasets/internal_datasets/metadata.json` |
| `--train_batch_size` | Number of samples per batch | 1 |
| `--image_sample_size` | Maximum training resolution for images | 640 |
| `--video_sample_size` | Maximum training resolution for videos | 640 |
| `--token_sample_size` | Token sampling size | 640 |
| `--video_sample_stride` | Video sampling stride | 2 |
| `--video_sample_n_frames` | Number of video frames to sample | 81 |
| `--gradient_accumulation_steps` | Gradient accumulation steps (effectively increases batch size) | 1 |
| `--dataloader_num_workers` | Number of DataLoader workers | 8 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 50 |
| `--learning_rate` | Initial learning rate (recommended for LoRA) | 1e-04 |
| `--lr_scheduler` | Learning rate scheduler | `constant` |
| `--lr_warmup_steps` | Learning rate warmup steps | 500 |
| `--seed` | Random seed (for reproducible training) | 42 |
| `--output_dir` | Output directory | `output_dir_hunyuanvideo_lora` |
| `--gradient_checkpointing` | Enable gradient checkpointing | - |
| `--mixed_precision` | Mixed precision: `fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon value | 1e-10 |
| `--vae_mini_batch` | Mini-batch size for VAE encoding | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--enable_bucket` | Enable bucket training, no cropping, group by resolution | - |
| `--random_hw_adapt` | Auto-resize images/videos to random sizes within `[min_size, max_size]` | - |
| `--training_with_video_token_length` | Train by token length, supports arbitrary resolutions | - |
| `--uniform_sampling` | Uniform timestep sampling (recommended) | - |
| `--low_vram` | Low VRAM mode | - |
| `--train_mode` | Training mode: `normal` (standard) or `i2v` (image-to-video) | `normal` |
| `--resume_from_checkpoint` | Resume training from checkpoint, use `"latest"` to auto-select | None |
| `--rank` | LoRA update matrix dimension (higher rank = more expressive, more memory) | 128 |
| `--network_alpha` | LoRA update matrix scaling factor (usually half of rank or same) | 64 |
| `--target_name` | Components/modules to apply LoRA, comma-separated | `to_q,to_k,to_v,ff.0,ff.2,ff_context.0,ff_context.2` |
| `--use_peft_lora` | Use PEFT module for LoRA (more memory-efficient) | - |
| `--validation_steps` | Run validation every N steps | 100 |
| `--validation_epochs` | Run validation every N epochs | 100 |
| `--validation_prompts` | Prompts for validation video generation | `"A young woman..."` |

**Sample Size Configuration Guide**:
- `video_sample_size` represents the resolution size of videos; when `random_hw_adapt` is True, it represents the minimum value between video and image resolutions.
- `image_sample_size` represents the resolution size of images; when `random_hw_adapt` is True, it represents the maximum value between video and image resolutions.
- `token_sample_size` represents the resolution corresponding to the maximum token length when `training_with_video_token_length` is True.
- Due to potential confusion in configuration, **if you don't require arbitrary resolution for finetuning**, it is recommended to set `video_sample_size`, `image_sample_size`, and `token_sample_size` to the same fixed value, such as **(320, 480, 512, 640, 960)**.
  - **All set to 320** represents **240P**.
  - **All set to 480** represents **320P**.
  - **All set to 640** represents **480P**.
  - **All set to 960** represents **720P**.

**Token Length Training Notes**:
- When `training_with_video_token_length` is enabled, the model trains based on token length.
- For example: a video with 512x512 resolution and 49 frames has a token length of 13,312, requiring `token_sample_size = 512`.
  - At 512x512 resolution, the number of video frames is 49 (~= 512 * 512 * 49 / 512 / 512).
  - At 768x768 resolution, the number of video frames is 21 (~= 512 * 512 * 49 / 768 / 768).
  - At 1024x1024 resolution, the number of video frames is 9 (~= 512 * 512 * 49 / 1024 / 1024).
  - These resolutions combined with their corresponding frame counts allow the model to generate videos of different sizes.

### 3.4 Training Validation

You can configure validation parameters to periodically generate test videos during training, allowing you to monitor training progress and model quality.

**Validation Parameters**:

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--validation_steps` | Run validation every N steps | 100 |
| `--validation_epochs` | Run validation every N epochs | 100 |
| `--validation_prompts` | Prompts for validation video generation, space-separated for multiple prompts | Space-separated prompts |

**Example**:

```bash
  --validation_steps=100 \
  --validation_epochs=100 \
  --validation_prompts="A young woman standing on a sunny coastline, white dress gently blowing in the sea breeze."
```

**Notes**:
- Validation videos are saved to the `output_dir` directory
- Multi-prompt validation format: `--validation_prompts "prompt1" "prompt2" "prompt3"`

### 3.5 Training with FSDP

**If GPU memory is insufficient when using multiple GPUs with DeepSpeed-Zero-2**, you can switch to FSDP for training.

```sh
export MODEL_NAME="models/Diffusion_Transformer/HunyuanVideo"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=HunyuanVideoTransformer3DModel --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/hunyuanvideo/train_lora.py \
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
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_hunyuanvideo_lora" \
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
  --rank=64 \
  --network_alpha=32 \
  --target_name="to_q,to_k,to_v,ff.0,ff.2,ff_context.0,ff_context.2" \
  --use_peft_lora \
  --low_vram
```

### 3.6 Training without DeepSpeed and FSDP

**This approach is not recommended because it lacks memory-saving backends and may easily cause out-of-memory errors**. It is provided here for reference only.

```sh
export MODEL_NAME="models/Diffusion_Transformer/HunyuanVideo"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/hunyuanvideo/train_lora.py \
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
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_hunyuanvideo_lora" \
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
  --rank=64 \
  --network_alpha=32 \
  --target_name="to_q,to_k,to_v,ff.0,ff.2,ff_context.0,ff_context.2" \
  --use_peft_lora \
  --low_vram
```

### 3.7 Multi-machine Distributed Training

**Suitable for**: Ultra-large-scale datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines with 8 GPUs each:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/HunyuanVideo"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/hunyuanvideo/train_lora.py \
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
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_hunyuanvideo_lora" \
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
  --rank=64 \
  --network_alpha=32 \
  --target_name="to_q,to_k,to_v,ff.0,ff.2,ff_context.0,ff_context.2" \
  --use_peft_lora \
  --low_vram
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/HunyuanVideo"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
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

#### 3.7.2 Multi-machine Training Notes

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

**Key Parameters**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `GPU_memory_mode` | Memory management mode, see table below for options | `model_group_offload` |
| `ulysses_degree` | Head dimension parallelism, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelism, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP for Transformer during multi-GPU inference to save memory | `False` |
| `fsdp_text_encoder` | Use FSDP for text encoder during multi-GPU inference | `True` |
| `compile_dit` | Compile Transformer for faster inference (effective for fixed resolutions) | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/HunyuanVideo` |
| `sampler_name` | Sampler type: `Flow`, `Flow_Unipc`, `Flow_DPM++` | `Flow` |
| `transformer_path` | Path to load trained Transformer weights | `None` |
| `vae_path` | Path to load trained VAE weights | `None` |
| `lora_path` | LoRA weights path | `None` |
| `sample_size` | Generated video resolution `[height, width]` | `[480, 832]` or `[832, 480]` |
| `video_length` | Number of video frames | `81` |
| `fps` | Frames per second | `16` |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 support | `torch.bfloat16` |
| `validation_image_start` | Reference image path for I2V mode | `"asset/1.png"` |
| `prompt` | Positive prompt, describes generated content | `"The dog is shaking head..."` |
| `negative_prompt` | Negative prompt, content to avoid | `"Low resolution, low quality..."` |
| `guidance_scale` | Guidance strength (HunyuanVideo uses true_cfg_scale) | 1.0 |
| `seed` | Random seed for reproducibility | 43 |
| `num_inference_steps` | Number of inference steps | 40 |
| `lora_weight` | LoRA weight strength | 0.55 |
| `save_path` | Generated video save path | `samples/hunyuanvideo-videos-i2v` or `samples/hunyuanvideo-videos-t2v` |

**Memory Management Modes**:

| Mode | Description | Memory Usage |
|------|-------------|--------------|
| `model_full_load` | Entire model loaded to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Model offloaded to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer groups switch between CPU/CUDA | Low |
| `sequential_cpu_offload` | Layer-by-layer offload (slowest) | Lowest |

### 4.2 Text-to-Video (T2V) Inference

Run the following command for single-GPU inference:

```bash
python examples/hunyuanvideo/predict_t2v.py
```

Edit `examples/hunyuanvideo/predict_t2v.py` according to your needs. For initial inference, focus on the following parameters. If you're interested in other parameters, refer to the inference parameter analysis above.

```python
# Choose based on GPU memory
GPU_memory_mode = "model_group_offload"
# Based on actual model path
model_name = "models/Diffusion_Transformer/HunyuanVideo"  
# Path to trained weights, e.g., "output_dir_hunyuanvideo_lora/checkpoint-xxx/lora_weights.safetensors"
lora_path = None
# LoRA weight strength
lora_weight = 0.55
# Write based on generated content
prompt = "1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body"  
# ...
```

### 4.3 Image-to-Video (I2V) Inference

Run the following command for single-GPU inference:

```bash
python examples/hunyuanvideo/predict_i2v.py
```

Edit `examples/hunyuanvideo/predict_i2v.py` according to your needs. For initial inference, focus on the following parameters. If you're interested in other parameters, refer to the inference parameter analysis above.

```python
# Choose based on GPU memory
GPU_memory_mode = "model_group_offload"
# Based on actual model path
model_name = "models/Diffusion_Transformer/HunyuanVideo"  
# LoRA weights path, e.g., "output_dir_hunyuanvideo_lora/checkpoint-xxx/lora_weights.safetensors"
lora_path = None
# LoRA weight strength
lora_weight = 0.55
# Starting image for image-to-video
validation_image_start = "asset/1.png"
# Write based on generated content
prompt = "The dog is shaking head. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."  
# ...
```

### 4.4 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, faster inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/hunyuanvideo/predict_t2v.py` or `examples/hunyuanvideo/predict_i2v.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelism
ring_degree = 1     # Sequence dimension parallelism
```

**Configuration Principles**:
- `ulysses_degree` must be divisible by the model's head count
- `ring_degree` splits along the sequence dimension and affects communication overhead; avoid using it if heads are evenly divisible

**Configuration Examples**:

| GPU Count | ulysses_degree | ring_degree | Description |
|-----------|---------------|-------------|-------------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelism |
| 8 | 8 | 1 | Head parallelism |
| 8 | 4 | 2 | Hybrid parallelism |

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/hunyuanvideo/predict_t2v.py
```

---

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
