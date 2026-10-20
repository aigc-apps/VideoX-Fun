# Qwen-Image 2.1 Full Parameter Training Guide

This document provides a complete workflow for full parameter training of the Qwen-Image 2.1 Diffusion Transformer, including environment configuration, data preparation, distributed training, and inference testing.
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
  - [3.6 Other Backends](#36-other-backends)
  - [3.7 Multi-Machine Distributed Training](#37-multi-machine-distributed-training)
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

We provide a test dataset containing several training samples.

```bash
# Download official demo dataset
modelscope download --dataset PAI/X-Fun-Images-Demo --local_dir ./datasets/X-Fun-Images-Demo
```

### 2.2 Dataset Structure

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 image001.jpg
│   │   ├── 📄 image002.png
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json Format

**Relative Path Format** (example):
```json
[
  {
    "file_path": "train/image001.jpg",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "width": 1024,
    "height": 1024
  },
  {
    "file_path": "train/image002.png",
    "text": "Portrait of a young woman, studio lighting, high quality",
    "width": 1024,
    "height": 1024
  }
]
```

**Absolute Path Format**:
```json
[
  {
    "file_path": "/mnt/data/images/sunset.jpg",
    "text": "A beautiful sunset over the ocean",
    "width": 1024,
    "height": 1024
  }
]
```

**Key Fields Description**:
- `file_path`: Image path (relative or absolute)
- `text`: Image description (English prompt)
- `width` / `height`: Image dimensions (**recommended** to provide for bucket training; if not provided, they will be automatically read during training, which may slow down training when data is stored on slow systems like OSS)
  - You can use `scripts/process_json_add_width_and_height.py` to add width and height fields to JSON files without these fields, supporting both images and videos
  - Usage: `python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Images-Demo/metadata.json --output_file datasets/X-Fun-Images-Demo/metadata_add_width_height.json`

> 💡 Training images are read as RGB and automatically composited over an opaque alpha channel before VAE encoding, so you do **not** need to provide RGBA data.

### 2.4 Relative vs Absolute Path Usage

**Relative Paths**:

If your data uses relative paths, configure the training script as follows:

```bash
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
```

**Absolute Paths**:

If your data uses absolute paths, configure the training script as follows:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata_add_width_height.json"
```

> 💡 **Recommendation**: If the dataset is small and stored locally, use relative paths. If the dataset is stored on external storage (e.g., NAS, OSS) or shared across multiple machines, use absolute paths.

---

## 3. Full Parameter Training

### 3.1 Download Pretrained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download Qwen-Image 2.1 official weights
modelscope download --model Qwen/Qwen-Image-2.1 --local_dir models/Diffusion_Transformer/Qwen-Image-2.1
```

> 💡 If the ModelScope id differs from the above, adjust it to the official Qwen-Image-2.1 release. You may also point `--pretrained_model_name_or_path` to any local directory in diffusers layout that contains the `transformer/`, `vae/`, `text_encoder/`, `processor/` and `scheduler/` subfolders.

### 3.2 Quick Start (DeepSpeed-Zero-2)

If you have downloaded the data as per **2.1 Quick Test Dataset** and the weights as per **3.1 Download Pretrained Model**, you can directly copy and run the quick start command.

DeepSpeed-Zero-2 and FSDP are recommended for training. Here we use DeepSpeed-Zero-2 as an example.

The difference between DeepSpeed-Zero-2 and FSDP lies in whether the model weights are sharded. **If VRAM is insufficient when using multiple GPUs with DeepSpeed-Zero-2**, you can switch to FSDP.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/qwenimage21/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1024 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_qwenimage21" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "."
```

### 3.3 Common Training Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--pretrained_model_name_or_path` | Path to pretrained model | `models/Diffusion_Transformer/Qwen-Image-2.1` |
| `--train_data_dir` | Training data directory | `datasets/X-Fun-Images-Demo/` |
| `--train_data_meta` | Training data metadata file | `datasets/X-Fun-Images-Demo/metadata_add_width_height.json` |
| `--train_batch_size` | Samples per batch | 1 |
| `--image_sample_size` | Maximum training resolution, auto bucketing | 1024 |
| `--gradient_accumulation_steps` | Gradient accumulation steps (equivalent to larger batch) | 1 |
| `--dataloader_num_workers` | DataLoader subprocesses | 8 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 50 |
| `--learning_rate` | Initial learning rate | 2e-05 |
| `--lr_scheduler` | Learning rate scheduler | `constant_with_warmup` |
| `--lr_warmup_steps` | Learning rate warmup steps | 100 |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir_qwenimage21` |
| `--gradient_checkpointing` | Enable activation checkpointing | - |
| `--mixed_precision` | Mixed precision: `fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon value | 1e-10 |
| `--vae_mini_batch` | Mini-batch size for VAE encoding | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--enable_bucket` | Enable bucket training: trains entire images grouped by resolution without center cropping | - |
| `--random_hw_adapt` | Auto-scale images to random size in range `[512, image_sample_size]` | - |
| `--resume_from_checkpoint` | Resume training from checkpoint path, use `"latest"` to auto-select latest | None |
| `--uniform_sampling` | Uniform timestep sampling | - |
| `--trainable_modules` | Trainable modules (`"."` means all modules) | `"."` |
| `--tokenizer_max_length` | Maximum prompt token length fed to the Qwen3-VL text encoder | 1024 |
| `--validation_steps` | Execute validation every N steps | 100 |
| `--validation_epochs` | Execute validation every N epochs | 100 |
| `--validation_prompts` | Prompts used during validation | `"1girl, black_hair, ..."` |


### 3.4 Training Validation

You can configure validation parameters to periodically generate test images during training, allowing you to monitor training progress and model quality.

**Validation Parameters**:

```bash
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/qwenimage21/train.py \
  # ... (other training parameters)
  --validation_steps=100 \
  --validation_epochs=100 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body"
```

**Parameter Descriptions**:

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--validation_steps` | Execute validation every N steps. If your dataset is large and you want to save validation time, you can set a larger value (e.g., 100 or 500) | 100 |
| `--validation_epochs` | Execute validation every N epochs | 100 |
| `--validation_prompts` | Prompt for validation image generation. Use multiple space-separated prompt strings | Space-separated prompt strings |

**Notes**:
- Validation images will be saved to the `output_dir` directory
- Setting `--validation_steps=1` means validation is performed every step, which may slow down training. Adjust according to your needs
- For multi-prompt validation, use: `--validation_prompts "prompt1" "prompt2" "prompt3"`


### 3.5 Training with FSDP

**If VRAM is insufficient when using multiple GPUs with DeepSpeed-Zero-2**, you can switch to FSDP. Note that the transformer layer class to wrap for Qwen-Image 2.1 is `QwenImage21TransformerBlock`.

```sh
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=QwenImage21TransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/qwenimage21/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1024 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_qwenimage21" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "."
```

### 3.6 Other Backends

#### 3.6.1 Training with DeepSpeed-Zero-3

DeepSpeed Zero-3 is not highly recommended at the moment. In this repository, using FSDP has fewer errors and is more stable.

DeepSpeed Zero-3:

After training, you can use the following command to get the final model:

```sh
python scripts/zero_to_bf16.py output_dir/checkpoint-{our-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

Training shell command:
```sh
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/qwenimage21/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1024 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_qwenimage21" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "."
```

#### 3.6.2 Training Without DeepSpeed or FSDP

**This approach is not recommended as it lacks VRAM-saving backends and may easily cause out-of-memory errors**. This is provided for reference only.

```sh
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/qwenimage21/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1024 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_qwenimage21" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "."
```

### 3.7 Multi-Machine Distributed Training

**Suitable for**: Ultra-large-scale datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines with 8 GPUs each:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/qwenimage21/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1024 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_qwenimage21" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "."
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
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

## 4. Inference Testing

> ℹ️ **Multi-GPU (Ulysses only)**: Qwen-Image 2.1 supports Ulysses (head-parallel) sequence parallelism — set `ulysses_degree > 1` to split a single image's denoising across GPUs (lower latency and less activation memory per GPU, mathematically identical to single-GPU). `ring_degree` **must stay 1**: ring attention rotates KV chunks and cannot express 2.1's block-causal mask or its prefix KV cache. `ulysses_degree` must divide `num_attention_heads` (32). See [4.3 Multi-GPU Parallel Inference](#43-multi-gpu-parallel-inference). You can also use the VRAM management modes below (offload / FP8) when a single GPU is not enough.

### 4.1 Inference Parameter Parsing

**Key Parameter Descriptions** (see `examples/qwenimage21/predict_t2i.py`):

| Parameter | Description | Example Value |
|------|------|-------|
| `GPU_memory_mode` | VRAM management mode, see table below for options | `model_full_load` |
| `ulysses_degree` | Ulysses (head) parallelism degree. Must divide `num_attention_heads` (32): 1/2/4/8/16/32; `>1` splits one image across GPUs | 1 |
| `ring_degree` | Sequence (ring) parallelism degree. **Must stay 1** — ring cannot express the block-causal mask or prefix KV cache | 1 |
| `compile_dit` | Compile Transformer for faster inference (effective at fixed resolution) | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/Qwen-Image-2.1` |
| `sampler_name` | Sampler type. Qwen-Image 2.1 is flow-matching, only `Flow` is supported | `Flow` |
| `transformer_path` | Path to load trained Transformer weights | `None` |
| `vae_path` | Path to load trained VAE weights | `None` |
| `lora_path` | LoRA weights path | `None` |
| `sample_size` | Generated image resolution `[height, width]`, rounded down to a multiple of 32; `None` falls back to the pipeline default square | `[1024, 1024]` |
| `use_kv_cache` | Cache text/condition keys-values after the first denoising step to speed up inference | `True` |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 support | `torch.bfloat16` |
| `prompts` | Positive prompts describing the generation content | `["a young girl ..."]` |
| `negative_prompt` | Negative prompt for content to avoid | `" "` |
| `guidance_scale` | Guidance strength (passed to the pipeline as `true_cfg_scale`) | 1.0 |
| `seed` | Random seed for reproducible results | 43 |
| `num_inference_steps` | Number of inference steps | 40 |
| `lora_weight` | LoRA weight strength | 1 |
| `save_path` | Path to save generated images | `samples/qwenimage21-t2i` |

**VRAM Management Mode Description**:

| Mode | Description | VRAM Usage |
|------|------|---------|
| `model_full_load` | Load entire model to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer groups switch between CPU/CUDA | Low |
| `sequential_cpu_offload` | Sequential layer offload (slowest) | Lowest |

### 4.2 Single GPU Inference

#### Quick Start

Run the following command for single GPU inference:

```bash
python examples/qwenimage21/predict_t2i.py
```

Edit `examples/qwenimage21/predict_t2i.py` according to your needs. For first-time inference, focus on these parameters. For other parameters, refer to the inference parameter parsing above.

```python
# Choose based on GPU VRAM
GPU_memory_mode = "model_full_load"
# Based on actual model path
model_name = "models/Diffusion_Transformer/Qwen-Image-2.1"
# Path to trained weights, e.g., "output_dir_qwenimage21/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = None
# Write based on generation content
prompts = ["a young girl with flowing long hair, wearing a white halter dress"]
# ...
```

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: high-resolution generation and faster single-image inference. Qwen-Image 2.1 splits the attention **heads** across GPUs (Ulysses sequence parallelism): after an all-to-all each GPU holds the full sequence for a subset of heads, so the block-causal multi-pass prefill and the prefix KV cache run unchanged and the output is **mathematically identical** to single-GPU inference.

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/qwenimage21/predict_t2i.py`:

```python
# ulysses_degree × ring_degree = number of GPUs; ring_degree MUST stay 1 for Qwen-Image 2.1
# For example, using 2 GPUs:
ulysses_degree = 2  # Head (Ulysses) parallelism
ring_degree = 1     # Must be 1
```

**Configuration Principles**:
- `ulysses_degree` must evenly divide `num_attention_heads` (32), i.e. one of 1/2/4/8/16/32.
- `ring_degree` must stay **1**: ring attention rotates KV chunks and cannot express 2.1's block-causal mask or its prefix KV cache (the script asserts this).
- The joint (text + image) sequence is padded internally to a multiple of `ulysses_degree`; padded keys are masked out, so results match single-GPU exactly.
- Ulysses replicates the weights on every GPU (it splits activations, not parameters). If VRAM is tight, also set `fsdp_dit = True` to shard the Transformer.

**Example Configurations**:

| GPU Count | ulysses_degree | ring_degree | Description |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 2 | 2 | 1 | Head parallelism |
| 4 | 4 | 1 | Head parallelism |
| 8 | 8 | 1 | Head parallelism |

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/qwenimage21/predict_t2i.py
```

Set `--nproc-per-node` equal to `ulysses_degree`.

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
