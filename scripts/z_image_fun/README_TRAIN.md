# Z-Image Control Full Parameter Training Guide

This document provides a complete workflow for training Z-Image Control models, including environment setup, data preparation, distributed training, and inference testing.

In Z-Image training, you can choose to use DeepSpeed or FSDP to save a significant amount of GPU memory.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. Control Training](#3-control-training)
  - [3.1 Download Pretrained Models](#31-download-pretrained-models)
  - [3.2 Quick Start (DeepSpeed-Zero-2)](#32-quick-start-deepspeed-zero-2)
  - [3.3 Training Parameters Explanation](#33-training-parameters-explanation)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training with FSDP](#35-training-with-fsdp)
  - [3.6 Other Backends](#36-other-backends)
  - [3.7 Multi-Node Distributed Training](#37-multi-node-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameters Explanation](#41-inference-parameters-explanation)
  - [4.2 Single GPU Inference](#42-single-gpu-inference)
  - [4.3 Multi-GPU Parallel Inference](#43-multi-gpu-parallel-inference)
- [5. Additional Resources](#5-additional-resources)

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
pip install yunchang xfuser modelscope openpyxl deepspeed==0.17.0 numpy==1.26.4
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

**Option 3: Using Docker**

When using Docker, please ensure that the GPU drivers and CUDA environment are correctly installed, then execute the following commands:

```bash
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset containing several training images and corresponding control files.

```bash
# Download official example dataset
modelscope download --dataset PAI/X-Fun-Images-Controls-Demo --local_dir ./datasets/X-Fun-Images-Controls-Demo
```

### 2.2 Dataset Structure

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 image001.jpg
│   │   ├── 📄 image002.png
│   │   └── 📄 ...
│   ├── 📂 control/
│   │   ├── 📄 image001.jpg
│   │   ├── 📄 image002.png
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json Format

The metadata.json for Control mode is slightly different from regular Z-Image json, requiring an additional `control_file_path` field.

It is recommended to use tools like [DWPose](https://github.com/IDEA-Research/DWPose) to generate control files (such as pose estimation maps).

**Relative Path Format** (example format):
```json
[
    {
      "file_path": "train/image001.jpg",
      "control_file_path": "control/image001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "width": 1024,
      "height": 1024,
      "type": "image"
    },
    {
      "file_path": "train/image002.jpg",
      "control_file_path": "control/image002.jpg",
      "text": "A beautiful woman standing on the beach at sunset.",
      "width": 1328,
      "height": 1328,
      "type": "image"
    }
]
```

**Absolute Path Format**:
```json
[
    {
      "file_path": "/mnt/data/images/image001.jpg",
      "control_file_path": "/mnt/data/controls/image001.jpg",
      "text": "A group of young men in suits and sunglasses.",
      "width": 1024,
      "height": 1024,
      "type": "image"
    }
]
```

**Key Fields Description**:
- `file_path`: Original image path (relative or absolute)
- `control_file_path`: Control file path, such as pose maps, edge detection maps, etc.
- `text`: Image description (English prompt)
- `width` / `height`: Image width and height (**recommended to provide**, used for bucket training; if not provided, will be automatically read during training)
- `type`: Data type, `"image"` for image data

> 💡 **Tip**: You can use `scripts/process_json_add_width_and_height.py` to extract width and height fields for json files without them.

### 2.4 Relative vs Absolute Path Usage

**Option 1: Using Relative Paths (Recommended)**

When data paths are not fixed or you need to train on different machines, it is recommended to use relative paths.

Configure relative paths in `metadata.json`, then specify the dataset root directory via `--train_data_dir` in the training script:

```json
[
  {
    "file_path": "train/image001.jpg",
    "control_file_path": "control/image001.jpg",
    "text": "A group of young men in suits and sunglasses are walking down a city street.",
    "width": 1024,
    "height": 1024,
    "type": "image"
  }
]
```

During training, the system will automatically search for files corresponding to relative paths under `--train_data_dir`.

**Option 2: Using Absolute Paths**

If the dataset path is fixed, you can directly configure absolute paths in `metadata.json`:

```json
[
  {
    "file_path": "/mnt/data/images/image001.jpg",
    "control_file_path": "/mnt/data/controls/image001.jpg",
    "text": "A group of young men in suits and sunglasses.",
    "width": 1024,
    "height": 1024,
    "type": "image"
  }
]
```

When using absolute paths, the `--train_data_dir` parameter serves only as a default path, and the absolute paths in json will take precedence.

---

## 3. Control Training

### 3.1 Download Pretrained Models

**ModelScope Download**:

```bash
# Create model directories
mkdir -p models/Diffusion_Transformer
mkdir -p models/Personalized_Model

# Download Z-Image official weights
modelscope download --model Tongyi-MAI/Z-Image --local_dir models/Diffusion_Transformer/Z-Image

# Download Z-Image-Turbo for fast inference
modelscope download --model Tongyi-MAI/Z-Image-Turbo --local_dir models/Diffusion_Transformer/Z-Image-Turbo

# Download Z-Image Control pretrained weights
modelscope download --model PAI/Z-Image-Fun-Controlnet-Union-2.1 --local_dir models/Personalized_Model/Z-Image-Fun-Controlnet-Union-2.1

# Download Z-Image-Turbo Control pretrained weights
modelscope download --model PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1 --local_dir models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.1
```

**HuggingFace Download**:

```bash
# Create model directories
mkdir -p models/Diffusion_Transformer
mkdir -p models/Personalized_Model

# Download Z-Image official weights
hf download Tongyi-MAI/Z-Image --local-dir models/Diffusion_Transformer/Z-Image

# Download Z-Image-Turbo for fast inference
hf download Tongyi-MAI/Z-Image-Turbo --local-dir models/Diffusion_Transformer/Z-Image-Turbo

# Download Z-Image Control pretrained weights
hf download alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1 --local-dir models/Personalized_Model/Z-Image-Fun-Controlnet-Union-2.1

# Download Z-Image-Turbo Control pretrained weights
hf download alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 --local-dir models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.1
```

### 3.2 Quick Start (DeepSpeed-Zero-2)

It is recommended to use DeepSpeed-Zero-2 or FSDP for training, which can save a significant amount of GPU memory.

After downloading data according to **2.1 Quick Test Dataset** and weights according to **3.1 Download Pretrained Models**, you can directly copy and run the following command:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Z-Image-Turbo"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/z_image_fun/train_control.py \
  --config_path="config/z_image/z_image_control_2.1.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1328 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_z_image_control" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --add_inpaint_info \
  --transformer_path="models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors" \
  --trainable_modules "control"
```

### 3.3 Training Parameters Explanation

**Key Parameters Description**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--pretrained_model_name_or_path` | Pretrained model path | `models/Diffusion_Transformer/Z-Image-Turbo` |
| `--train_data_dir` | Training data directory | `datasets/internal_datasets/` |
| `--train_data_meta` | Training data metadata file | `datasets/internal_datasets/metadata.json` |
| `--train_batch_size` | Batch size per device | 1 |
| `--image_sample_size` | Maximum training resolution, automatic bucketing | 1328 |
| `--gradient_accumulation_steps` | Gradient accumulation steps (equivalent to larger batch) | 1 |
| `--dataloader_num_workers` | DataLoader worker processes | 8 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 50 |
| `--learning_rate` | Initial learning rate | 2e-05 |
| `--lr_scheduler` | Learning rate scheduler | `constant_with_warmup` |
| `--lr_warmup_steps` | Learning rate warmup steps | 100 |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir_z_image_control` |
| `--gradient_checkpointing` | Enable gradient checkpointing | - |
| `--mixed_precision` | Mixed precision: `fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon value | 1e-10 |
| `--vae_mini_batch` | Mini batch size for VAE encoding | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--enable_bucket` | Enable bucket training, train full images grouped by resolution | - |
| `--uniform_sampling` | Uniform timestep sampling | - |
| `--transformer_path` | Load pretrained Control weights | `models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors` |
| `--trainable_modules` | Trainable modules (`"control"` means only train control module) | `"control"` |
| `--validation_steps` | Run validation every N steps | 50 |
| `--validation_epochs` | Run validation every N epochs | 500 |
| `--validation_prompts` | Prompts used for validation | `"1girl, black_hair, ..."` |

### 3.4 Training Validation

During training, you can set validation parameters to periodically evaluate model performance:

```bash
  --validation_paths "asset/pose.jpg" \
  --validation_steps=50 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body"
```

**Validation Parameters Description**:
- `--validation_paths`: Control image paths, used as control conditions during validation (supports multiple images)
- `--validation_steps`: Run validation every N steps (triggers when either this or `--validation_epochs` is met)
- `--validation_epochs`: Run validation every N epochs (triggers when either this or `--validation_steps` is met)
- `--validation_prompts`: Prompts used for validation (supports multiple prompts, corresponding one-to-one with `--validation_paths`)

Validation results will be saved in the `{output_dir}/sample/` directory, with filenames formatted as `sample-{global_step}-rank{process_index}-image-{index}.jpg`.

### 3.5 Training with FSDP

If DeepSpeed-Zero-2 runs out of GPU memory, you can switch to FSDP for training:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Z-Image-Turbo"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap BaseZImageTransformerBlock,ZImageControlTransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/z_image_fun/train_control.py \
  --config_path="config/z_image/z_image_control_2.1.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1328 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_z_image_control" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --add_inpaint_info \
  --transformer_path="models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors" \
  --trainable_modules "control"
```

### 3.6 Other Backends

#### 3.6.1 Training without DeepSpeed and FSDP

Training without DeepSpeed or FSDP may result in insufficient GPU memory. Only recommended when GPU memory is sufficient:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Z-Image-Turbo"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/z_image_fun/train_control.py \
  --config_path="config/z_image/z_image_control_2.1.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1328 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_z_image_control" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --add_inpaint_info \
  --transformer_path="models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors" \
  --trainable_modules "control"
```

### 3.7 Multi-Node Distributed Training

**Suitable for**: Ultra-large-scale datasets, faster training speed

#### 3.7.1 Environment Configuration

When using multi-node training, please set the following environment variables:

```bash
export MASTER_ADDR="your master address"
export MASTER_PORT=10086
export WORLD_SIZE=1 # The number of machines
export NUM_PROCESS=8 # The number of processes, such as WORLD_SIZE * 8
export RANK=0 # The rank of this machine

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK scripts/z_image_fun/train_control.py \
  [other training parameters...]
```

#### 3.7.2 Multi-Node Training Notes

- **Network Requirements**:
   - RDMA/InfiniBand recommended (high performance)
   - Add environment variables when without RDMA:
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **Data Synchronization**: All machines must be able to access the same data paths (NFS/shared storage)

---

## 4. Inference Testing

### 4.1 Inference Parameters Explanation

**Key Parameters Description**:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `GPU_memory_mode` | GPU memory management mode, see table below for options | `model_group_offload` |
| `ulysses_degree` | Head dimension parallelism, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelism, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP for Transformer in multi-GPU inference to save memory | `False` |
| `fsdp_text_encoder` | Use FSDP for text encoder in multi-GPU inference | `False` |
| `compile_dit` | Compile Transformer for faster inference (effective for fixed resolution) | `False` |
| `enable_teacache` | Enable TeaCache for faster inference | `True` |
| `teacache_threshold` | TeaCache threshold, recommended 0.05~0.30, higher is faster but quality may decrease | 0.30 |
| `num_skip_start_steps` | Number of steps to skip at the beginning to reduce impact on quality | 5 |
| `teacache_offload` | Offload TeaCache tensors to CPU to save memory | `False` |
| `cfg_skip_ratio` | Skip some CFG steps for faster inference, recommended 0.00~0.25 | 0 |
| `config_path` | Configuration file path | `config/z_image/z_image_control.yaml` |
| `model_name` | Model path | `models/Diffusion_Transformer/Z-Image-Turbo` |
| `sampler_name` | Sampler type: `Flow`, `Flow_Unipc`, `Flow_DPM++` | `Flow` |
| `transformer_path` | Path to trained Transformer weights | `models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union.safetensors` |
| `vae_path` | Path to trained VAE weights | `None` |
| `lora_path` | LoRA weights path | `None` |
| `sample_size` | Generated image resolution `[height, width]` | `[1728, 992]` |
| `weight_dtype` | Model weight precision, use `torch.float16` if GPU doesn't support bf16 | `torch.bfloat16` |
| `control_image` | Control image path (e.g., pose map) | `asset/pose.jpg` |
| `inpaint_image` | Inpaint input image (optional) | `asset/8.png` |
| `mask_image` | Mask image (optional) | `asset/mask.png` |
| `control_context_scale` | Control condition weight, recommended value 0.80 | 0.80 |
| `prompt` | Positive prompt, describing the content to generate | `"A young girl in the center..."` |
| `negative_prompt` | Negative prompt, content to avoid | `" "` |
| `guidance_scale` | Guidance strength | 4.0 |
| `seed` | Random seed for reproducibility | 43 |
| `num_inference_steps` | Number of inference steps | 50 |
| `lora_weight` | LoRA weight intensity | 0.55 |
| `save_path` | Path to save generated images | `samples/z-image-t2i-control` |

**GPU Memory Management Mode Description**:

| Mode | Description | Memory Usage |
|------|-------------|--------------|
| `model_full_load` | Entire model loaded to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer groups switch between CPU/CUDA | Low |
| `sequential_cpu_offload` | Layer-by-layer offload (slowest) | Lowest |

### 4.2 Single GPU Inference

#### Quick Start

Run the following command for single GPU inference:

```bash
python examples/z_image_fun/predict_turbo_t2i_control.py
```

Edit `examples/z_image_fun/predict_turbo_t2i_control.py` according to your needs. For initial inference, focus on the following parameters. If you're interested in other parameters, please refer to the inference parameters explanation above.

```python
# Choose based on GPU memory
GPU_memory_mode = "model_group_offload"
# Based on actual model path
model_name = "models/Diffusion_Transformer/Z-Image-Turbo"  
# Path to trained weights, e.g., "output_dir_z_image_control/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = "models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union.safetensors"  
# Control image path
control_image = "asset/pose.jpg"
# Write based on the content to generate
prompt = "A young girl in the center..."  
# ...
```

Generated results will be saved in the `samples/z-image-t2i-control` directory.

**Image Inpainting Inference**:

If you want to use the image inpainting feature, you can run:

```bash
python examples/z_image_fun/predict_turbo_i2i_inpaint_2.1.py
```

This script supports using both control images and inpainting masks for image generation.

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/z_image_fun/predict_turbo_t2i_control.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelism
ring_degree = 1     # Sequence dimension parallelism
```

**Configuration Principles**:
- `ulysses_degree` must be divisible by the model's number of heads.
- `ring_degree` splits on sequence dimension, affecting communication overhead. Try to avoid using it when heads can be split.

**Example Configurations**:

| GPU Count | ulysses_degree | ring_degree | Description |
|-----------|----------------|-------------|-------------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelism |
| 8 | 8 | 1 | Head parallelism |
| 8 | 4 | 2 | Hybrid parallelism |

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/z_image_fun/predict_turbo_t2i_control.py
```

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
