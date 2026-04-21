# Qwen-Image-Edit Full Parameter Training Guide

This document provides a complete workflow for full parameter training of Qwen-Image-Edit Diffusion Transformer, including environment configuration, data preparation, distributed training, and inference testing.

---

## Table of Contents
- [1. Environment Configuration](#1-environment-configuration)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. Full Parameter Training](#3-full-parameter-training)
  - [3.1 Download Pre-trained Model](#31-download-pre-trained-model)
  - [3.2 Quick Start (DeepSpeed-Zero-2)](#32-quick-start-deepspeed-zero-2)
  - [3.3 Training Parameters Explanation](#33-training-parameters-explanation)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training with FSDP](#35-training-with-fsdp)
  - [3.6 Other Backends](#36-other-backends)
  - [3.7 Multi-machine Distributed Training](#37-multi-machine-distributed-training)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameters Explanation](#41-inference-parameters-explanation)
  - [4.2 Single GPU Inference](#42-single-gpu-inference)
  - [4.3 Multi-GPU Parallel Inference](#43-multi-gpu-parallel-inference)
- [5. Additional Resources](#5-additional-resources)

---

## 1. Environment Configuration

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

When using Docker, please ensure that the graphics driver and CUDA environment are correctly installed, then execute the following commands:

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
# Download official example dataset
modelscope download --dataset PAI/X-Fun-Images-Edit-Demo --local_dir ./datasets/X-Fun-Images-Edit-Demo
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

**Edit Model Data Format**:

The metadata.json for Edit model is different from the normal version, requiring the addition of a `source_file_path` field.

- Qwen-Image-Edit: Only needs one file in source_file_path
- Qwen-Image-Edit-2509: Needs one or more files in source_file_path

**Relative Path Format** (Edit Model):
```json
[
    {
      "file_path": "train/00000001.jpg",
      "source_file_path": ["source/00000001.jpg"],
      "text": "A young woman stands on a sunny coastline, wearing a refreshing white shirt and skirt",
      "type": "image"
    },
    {
      "file_path": "train/00000002.jpg",
      "source_file_path": ["source/00000002.jpg"],
      "text": "A young woman with purple hair stands on the coastline, with the vast sea in the background",
      "type": "image"
    }
]
```

**Key Fields Description**:
- `file_path`: Target image path (the image to be generated after training)
- `source_file_path`: Source image path array (original images for editing)
  - Edit model will edit based on source_file_path images according to text description to generate file_path images
  - Qwen-Image-Edit only needs one source file, Qwen-Image-Edit-2509 supports multiple source files
- `text`: Image description (prompt, describing the expected generated content)
- `type`: Data type ("image" or "video")
- `width` / `height`: Image width and height (**recommended to provide**, used for bucket training. If not provided, it will be automatically read during training, which may affect training speed when data is stored on slow systems like OSS).
  - You can use `scripts/process_json_add_width_and_height.py` to extract width and height fields from json without these fields, supporting both images and videos.
  - Usage: `python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Images-Demo/metadata.json --output_file datasets/X-Fun-Images-Demo/metadata_add_width_height.json`.

### 2.4 Relative vs Absolute Path Usage

**Relative Paths**:

If your data uses relative paths, set in the training script:

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

**Absolute Paths**:

If your data uses absolute paths, set in the training script:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **Recommendation**: If the dataset is small and stored locally, relative paths are recommended; if the dataset is stored on external storage (such as NAS, OSS) or shared across multiple machines, absolute paths are recommended.

---

## 3. Full Parameter Training

### 3.1 Download Pre-trained Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download official Qwen-Image-Edit weights
modelscope download --model Qwen/Qwen-Image-Edit --local_dir models/Diffusion_Transformer/Qwen-Image-Edit
```

### 3.2 Quick Start (DeepSpeed-Zero-2)

If you have downloaded data according to **2.1 Quick Test Dataset** and downloaded weights according to **3.1 Download Pre-trained Model**, you can directly copy the quick start command to launch.

**Training Notes**:
- **Warning Without DeepSpeed**: Training Qwen-Image-Edit without DeepSpeed may result in insufficient GPU memory. DeepSpeed-Zero-2 or FSDP is recommended.
- **DeepSpeed Zero-3 Recommendation**: DeepSpeed Zero-3 is not highly recommended at the moment. In this repository, using FSDP has fewer errors and is more stable.
- If using DeepSpeed Zero-3, after training you need to use the following command to get the final model:
  ```bash
  python scripts/zero_to_bf16.py output_dir/checkpoint-{step-number} output_dir/checkpoint-{step-number}-outputs --max_shard_size 80GB --safe_serialization
  ```

DeepSpeed-Zero-2 and FSDP are recommended for training. Here we use DeepSpeed-Zero-2 as an example to configure the shell file.

The difference between DeepSpeed-Zero-2 and FSDP in this document is whether the model weights are sharded. **If using multiple GPUs and encountering insufficient memory with DeepSpeed-Zero-2**, you can switch to FSDP for training.

```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-Edit"
export DATASET_NAME="datasets/X-Fun-Images-Edit-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Edit-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/qwenimage/train_edit.py \
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
  --output_dir="output_dir_qwenimage_edit" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "." \
  --train_mode "qwen_image_edit"
```

### 3.3 Training Parameters Explanation

**Key Parameters Description**:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--pretrained_model_name_or_path` | Pre-trained model path | `models/Diffusion_Transformer/Qwen-Image-Edit` |
| `--train_data_dir` | Training data directory | `datasets/internal_datasets/` |
| `--train_data_meta` | Training data metadata file | `datasets/internal_datasets/metadata.json` |
| `--train_batch_size` | Number of samples per batch | 1 |
| `--image_sample_size` | Maximum training resolution, code will automatically bucket | 1328 |
| `--gradient_accumulation_steps` | Gradient accumulation steps (equivalent to increasing batch) | 1 |
| `--dataloader_num_workers` | Number of DataLoader subprocesses | 8 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 50 |
| `--learning_rate` | Initial learning rate | 2e-05 |
| `--lr_scheduler` | Learning rate scheduler | `constant_with_warmup` |
| `--lr_warmup_steps` | Learning rate warmup steps | 100 |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir` |
| `--gradient_checkpointing` | Activation recomputation | - |
| `--mixed_precision` | Mixed precision: `fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon value | 1e-10 |
| `--vae_mini_batch` | Mini batch size for VAE encoding | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 0.05 |
| `--enable_bucket` | Enable bucket training, no image cropping, train entire images grouped by resolution | - |
| `--random_hw_adapt` | Auto-scale images to random sizes in range `[512, image_sample_size]` | - |
| `--resume_from_checkpoint` | Resume training path, use `"latest"` to auto-select latest checkpoint | None |
| `--uniform_sampling` | Uniform sampling of timesteps | - |
| `--trainable_modules` | Trainable modules (`.` means all modules) | `"."` |
| `--train_mode` | Training mode: `qwen_image_edit` for Qwen-Image-Edit, `qwen_image_edit_plus` for Qwen-Image-Edit-2509 | `"qwen_image_edit"` |
| `--validation_steps` | Run validation every N steps | 100 |
| `--validation_epochs` | Run validation every N epochs | 500 |
| `--validation_prompts` | Prompts used during validation | `"1girl, black_hair, ..."` |
| `--validation_image_paths` | Source image paths used during validation (Edit model specific) | `"asset/8.png"` |

**random_hw_adapt Detailed Explanation**:
- When `random_hw_adapt` is enabled and `image_sample_size=1024`, the resolution range of training images is `512x512` to `1024x1024`
- Can be used with `enable_bucket` for more flexible handling of different image resolutions
- For example: `random_hw_adapt=true`, `image_sample_size=1024`, images will be randomly scaled to sizes between 512 and 1024 during training


### 3.4 Training Validation

You can configure validation parameters to regularly generate test images during training to monitor training progress and model quality.

**Validation Parameters Configuration**:

```bash
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/qwenimage/train_edit.py \
  # ... (other training parameters)
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body" \
  --validation_image_paths="asset/8.png"
```

**Parameters Description**:

| Parameter | Description | Recommended Value |
|------|------|--------|
| `--validation_steps` | Run validation every N steps. If dataset is large and you want to save validation time, you can set a larger value (e.g., 100 or 500) | 100 |
| `--validation_epochs` | Run validation every N epochs | 500 |
| `--validation_prompts` | Prompts for validation image generation. Multiple prompts can be set, separated by spaces | Multiple space-separated prompts |
| `--validation_image_paths` | Source image paths used during validation (Edit model specific) | `asset/8.png` |

**Notes**:
- Validation images will be saved to the `output_dir` directory
- Setting `--validation_steps=1` means validation at every step, which may slow down training. Adjust according to actual needs.
- Multi-prompt validation format: `--validation_prompts "prompt1" "prompt2" "prompt3"`
- Edit model validation must provide `--validation_image_paths` parameter to specify source images for editing

### 3.5 Training with FSDP

**If using multiple GPUs and encountering insufficient memory with DeepSpeed-Zero-2**, you can switch to FSDP for training.

```sh
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-Edit"
export DATASET_NAME="datasets/X-Fun-Images-Edit-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Edit-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=QwenImageTransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/qwenimage/train_edit.py \
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
  --output_dir="output_dir_qwenimage_edit" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "." \
  --train_mode "qwen_image_edit"
```

### 3.6 Other Backends

#### 3.6.1 Training with DeepSpeed-Zero-3

DeepSpeed Zero-3 is not highly recommended at the moment. In this repository, using FSDP has fewer errors and is more stable.

DeepSpeed Zero-3:

After training, you can use the following command to get the final model:

```sh
python scripts/zero_to_bf16.py output_dir/checkpoint-{step-number} output_dir/checkpoint-{step-number}-outputs --max_shard_size 80GB --safe_serialization
```

Training shell command is as follows:

```sh
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-Edit"
export DATASET_NAME="datasets/X-Fun-Images-Edit-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Edit-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/qwenimage/train_edit.py \
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
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "." \
  --train_mode "qwen_image_edit"
```

#### 3.6.2 Training Without DeepSpeed and FSDP

**This approach is not recommended because without memory-saving backends, it easily causes insufficient GPU memory**. This is only provided as a reference shell for training.

```sh
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-Edit"
export DATASET_NAME="datasets/X-Fun-Images-Edit-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Edit-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/qwenimage/train_edit.py \
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
  --output_dir="output_dir_qwenimage_edit" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "." \
  --train_mode "qwen_image_edit"
```

### 3.7 Multi-machine Distributed Training

**Suitable for**: Ultra-large-scale datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines, each with 8 GPUs:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-Edit"
export DATASET_NAME="datasets/X-Fun-Images-Edit-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Edit-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/qwenimage/train_edit.py \
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
  --output_dir="output_dir_qwenimage_edit" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "." \
  --train_mode "qwen_image_edit"
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-Edit"
export DATASET_NAME="datasets/X-Fun-Images-Edit-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Edit-Demo/metadata_add_width_height.json"
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

## 4. Inference Testing

### 4.1 Inference Parameters Explanation

**Key Parameters Description**:

| Parameter | Description | Example Value |
|------|------|-------|
| `GPU_memory_mode` | Memory management mode, see table below for options | `model_group_offload` |
| `ulysses_degree` | Head dimension parallelism, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelism, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP for Transformer during multi-GPU inference to save memory | `False` |
| `fsdp_text_encoder` | Use FSDP for text encoder during multi-GPU inference | `False` |
| `compile_dit` | Compile Transformer for faster inference (effective at fixed resolution) | `False` |
| `enable_teacache` | Enable TeaCache for faster inference | `True` |
| `teacache_threshold` | TeaCache threshold, recommended 0.05~0.30, larger is faster but quality may decrease | 0.25 |
| `num_skip_start_steps` | Steps to skip at the beginning of inference to reduce impact on generation quality | 5 |
| `teacache_offload` | Offload TeaCache tensors to CPU to save memory | `False` |
| `cfg_skip_ratio` | Skip some CFG steps for faster inference, recommended 0.00~0.25 | 0 |
| `model_name` | Model path | `models/Diffusion_Transformer/Qwen-Image-Edit` |
| `sampler_name` | Sampler type: `Flow`, `Flow_Unipc`, `Flow_DPM++` | `Flow` |
| `transformer_path` | Path to load trained Transformer weights | `None` |
| `vae_path` | Path to load trained VAE weights | `None` |
| `lora_path` | LoRA weights path | `None` |
| `sample_size` | Generated image resolution `[height, width]` | `[1344, 768]` |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs that don't support bf16 | `torch.bfloat16` |
| `prompt` | Positive prompt, describes content to generate | `"1girl, black_hair..."` |
| `negative_prompt` | Negative prompt, content to avoid | `" "` |
| `guidance_scale` | Guidance strength | 4.0 |
| `seed` | Random seed, for reproducibility | 43 |
| `num_inference_steps` | Inference steps | 50 |
| `lora_weight` | LoRA weight intensity | 0.55 |
| `save_path` | Generated image save path | `samples/qwenimage-t2i` |

**Memory Management Mode Description**:

| Mode | Description | Memory Usage |
|------|------|---------|
| `model_full_load` | Entire model loaded to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer groups switch between CPU/CUDA | Low |
| `sequential_cpu_offload` | Layer-by-layer offload (slowest) | Lowest |

### 4.2 Single GPU Inference

#### Quick Start

Run single GPU inference with the following command:

```bash
python examples/qwenimage/predict_t2i_edit.py
```

Edit `examples/qwenimage/predict_t2i_edit.py` according to your needs. For first-time inference, focus on the following parameters. If you're interested in other parameters, please refer to the inference parameters explanation above.

```python
# Select based on GPU memory
GPU_memory_mode = "model_group_offload"
# Based on actual model path
model_name = "models/Diffusion_Transformer/Qwen-Image-Edit"  
# Trained weights path, e.g., "output_dir_qwenimage_edit/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = None  
# Write based on content to generate
prompt = "A young woman stands on a sunny coastline, wearing a refreshing white shirt and skirt"  
# ...
```

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/qwenimage/predict_t2i_edit.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelism
ring_degree = 1     # Sequence dimension parallelism
```

**Configuration Principles**:
- `ulysses_degree` must be divisible by the model's head count.
- `ring_degree` splits on sequence dimension, affecting communication overhead. Try not to use it when head count can be split.

**Example Configurations**:

| GPU Count | ulysses_degree | ring_degree | Description |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelism |
| 8 | 8 | 1 | Head parallelism |
| 8 | 4 | 2 | Hybrid parallelism |

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/qwenimage/predict_t2i_edit.py
```

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun