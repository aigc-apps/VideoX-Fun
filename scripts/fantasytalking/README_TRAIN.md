# FantasyTalking-S2V Full Parameter Training Guide

This document provides a complete workflow for full parameter training of FantasyTalking-S2V (an audio-driven digital human video generation model), including environment configuration, data preparation, distributed training, and inference testing.

> **Note**: FantasyTalking is an audio-driven digital human video generation model that requires both a reference image and an audio file to generate talking videos. Training data needs to include videos, audios, and reference images.

---

## Table of Contents
- [1. Environment Configuration](#1-environment-configuration)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Paths](#24-relative-vs-absolute-paths)
- [3. Full Parameter Training](#3-full-parameter-training)
  - [3.1 Download Pretrained Models](#31-download-pretrained-models)
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

When using Docker, first ensure that GPU drivers and CUDA environment are properly installed on your machine, then execute the following commands:

```bash
# Pull the image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# Run the container
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

We provide a test dataset containing several audio-video training samples.

```bash
# Download the official demo dataset
modelscope download --dataset PAI/X-Fun-Videos-Audios-Demo --local_dir ./datasets/X-Fun-Videos-Audios-Demo
```

### 2.2 Dataset Structure

```text
datasets/internal_datasets/
├── train/
│   ├── 00000001.mp4
│   ├── 00000002.mp4
│   └── ...
├── wav/
│   ├── 00000001.wav
│   ├── 00000002.wav
│   └── ...
└── metadata.json
```

**Directory Description**:
- `train/`: Training video files (MP4 format)
- `wav/`: Corresponding audio files (WAV format)
- `metadata.json`: Dataset metadata file describing the path and text description for each sample

### 2.3 metadata.json Format

The `metadata.json` for FantasyTalking is slightly different from the normal JSON format in VideoX-Fun. **You need to add an `audio_path` field**.

**Example Format**:

```json
[
    {
      "file_path": "train/00000001.mp4",
      "audio_path": "wav/00000001.wav",
      "text": "A girl talking by the sea.",
      "type": "video"
    },
    {
      "file_path": "train/00000002.mp4",
      "audio_path": "wav/00000002.wav",
      "text": "A man talking in a room.",
      "type": "video"
    }
]
```

**Field Description**:
- `file_path`: Relative or absolute path to the video file
- `audio_path`: Relative or absolute path to the audio file (**Required field for FantasyTalking**)
- `text`: Text description (prompt) for the video
- `type`: Data type, usually `"video"`

### 2.4 Relative vs Absolute Paths

**Relative Paths (Recommended)**:

Paths in `metadata.json` are relative to the dataset root directory:

```json
{
  "file_path": "train/00000001.mp4",
  "audio_path": "wav/00000001.wav"
}
```

Specify the dataset root directory in the training command:

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

**Absolute Paths**:

You can also use absolute paths:

```json
{
  "file_path": "/path/to/your/dataset/train/00000001.mp4",
  "audio_path": "/path/to/your/dataset/wav/00000001.wav"
}
```

---

## 3. Full Parameter Training

### 3.1 Download Pretrained Models

Before training, you need to download the following pretrained models:

**1. Base Video Generation Model**

```bash
# Download Wan2.1-I2V-14B-720P model
modelscope download --model Wan-AI/Wan2.1-I2V-14B-720P --local_dir models/Diffusion_Transformer/Wan2.1-I2V-14B-720P/
```

**2. Audio Encoder**

```bash
# Download wav2vec2 audio encoder
modelscope download --model AI-ModelScope/wav2vec2-base-960h --local_dir models/Diffusion_Transformer/wav2vec2-base-960h
```

**3. FantasyTalking Pretrained Weights**

```bash
modelscope download --model amap_cvlab/FantasyTalking --local_dir models/Personalized_Model/FantasyTalking/
```

### 3.2 Quick Start (DeepSpeed-Zero-2)

Here is the recommended configuration for distributed training with DeepSpeed-Zero-2:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
export MODEL_NAME_AUDIO="models/Diffusion_Transformer/wav2vec2-base-960h"  # If None, will use $MODEL_NAME/audio_encoder
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# Uncomment the following two lines for multi-node training without RDMA
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/fantasytalking/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_audio_model_name_or_path=$MODEL_NAME_AUDIO \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_fantasytalking" \
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
  --transformer_path="models/Personalized_Model/FantasyTalking/fantasytalking_model.ckpt" \
  --trainable_modules "processor." "proj_model."
```

### 3.3 Common Training Parameters

Here is a detailed explanation of the key parameters in the training script:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `config_path` | Model configuration file path | `config/wan2.1/wan_civitai.yaml` |
| `pretrained_model_name_or_path` | Pretrained model path | `models/Diffusion_Transformer/Wan2.1-I2V-14B-720P` |
| `pretrained_audio_model_name_or_path` | Audio encoder path | `None` (automatically uses $MODEL_NAME/audio_encoder) |
| `train_data_dir` | Training dataset directory | `datasets/internal_datasets/` |
| `train_data_meta` | Training dataset metadata file | `datasets/internal_datasets/metadata.json` |
| `video_sample_size` | Video sample size (maximum resolution) | `512` |
| `token_sample_size` | Token sample size | `512` |
| `video_sample_stride` | Video sample stride | `1` |
| `video_sample_n_frames` | Video sample frame count | `81` |
| `train_batch_size` | Training batch size | `1` |
| `gradient_accumulation_steps` | Gradient accumulation steps | `1` |
| `dataloader_num_workers` | Data loader worker threads | `8` |
| `num_train_epochs` | Number of training epochs | `100` |
| `checkpointing_steps` | Steps to save checkpoint | `50` |
| `learning_rate` | Learning rate | `2e-05` |
| `lr_scheduler` | Learning rate scheduler | `constant_with_warmup` |
| `lr_warmup_steps` | Learning rate warmup steps | `100` |
| `seed` | Random seed | `42` |
| `output_dir` | Output directory | `output_dir_fantasytalking` |
| `gradient_checkpointing` | Enable gradient checkpointing to save memory | `True` |
| `mixed_precision` | Mixed precision training: `bf16` or `fp16` | `bf16` |
| `adam_weight_decay` | Adam weight decay | `3e-2` |
| `adam_epsilon` | Adam epsilon | `1e-10` |
| `vae_mini_batch` | VAE mini-batch size | `1` |
| `max_grad_norm` | Maximum gradient norm | `0.05` |
| `transformer_path` | Pretrained Transformer weights path | `models/FantasyTalking/fantasytalking_model.ckpt` |
| `trainable_modules` | List of trainable modules | `"processor." "proj_model."` |

**Advanced Parameters Explanation**:

The following parameters in the training script might be confusing, here's a detailed explanation:

- **`enable_bucket`**: Enable bucket training. When enabled, the model does not crop videos at the center, but instead groups videos into different buckets based on resolution for training. This allows the model to better adapt to videos of different resolutions.

- **`random_frame_crop`**: Random cropping on video frames to simulate videos with different frame counts. This helps the model better generalize to videos of varying lengths.

- **`random_hw_adapt`**: Enable automatic height and width scaling. When enabled, training video dimensions will be set to:
  - Maximum: `video_sample_size`
  - Minimum: `512`
  
  **Example**: With `random_hw_adapt` enabled, `video_sample_n_frames=81`, `video_sample_size=768`, the training input video resolutions can be `512x512x81` or `768x768x81`.

- **`training_with_video_token_length`**: Train the model based on token length. When enabled, training video dimensions will be set to:
  - Maximum: `video_sample_size`
  - Minimum: `256`
  
  **Example**: With `training_with_video_token_length` enabled, `video_sample_n_frames=81`, `token_sample_size=512`, `video_sample_size=768`, the training input video resolutions can be `256x256x81`, `512x512x81`, or `768x768x37`.
  
  **Token Length Calculation**:
  - For a 512x512 resolution video with 81 frames, the token length is approximately 21,952
  - We need to set `token_sample_size = 512`
    - At 512x512 resolution, the number of video frames is 81 (≈ 512 * 512 * 81 / 512 / 512)
    - At 768x768 resolution, the number of video frames is 37 (≈ 512 * 512 * 81 / 768 / 768)
    - At 1024x1024 resolution, the number of video frames is 16 (≈ 512 * 512 * 81 / 1024 / 1024)
    - These resolutions combined with their corresponding lengths allow the model to generate videos of different sizes.

- **`resume_from_checkpoint`**: Resume training from a previous checkpoint. Use a path or `"latest"` to automatically select the last available checkpoint.

- **`low_vram`**: Enable low VRAM mode to reduce memory usage through memory optimization.

- **`uniform_sampling`**: Use uniform sampling strategy for timestep sampling.

### 3.4 Training Validation

You can configure validation parameters to periodically generate test videos during training, allowing you to monitor training progress and model quality.

**Validation Parameters**:

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--validation_steps` | Execute validation every N steps | 100 |
| `--validation_epochs` | Execute validation every N epochs | 500 |
| `--validation_image_paths` | List of reference image paths for validation, multiple paths separated by spaces | Multiple space-separated image paths |
| `--validation_audio_paths` | List of audio paths for validation, multiple paths separated by spaces | Multiple space-separated audio paths |
| `--validation_prompts` | List of prompts for validation, multiple prompts separated by spaces | Multiple space-separated prompts |

**Example**:

```bash
  --validation_image_paths="asset/8.png" \
  --validation_audio_paths="asset/talk.wav" \
  --validation_prompts="A girl talking by the sea." \
  --validation_steps=100 \
  --validation_epochs=500
```

**Notes**:
- The number of `validation_image_paths`, `validation_audio_paths`, and `validation_prompts` must be consistent
- When both `validation_steps` and `validation_epochs` are set, validation is triggered when either condition is met

### 3.5 Training with FSDP

FSDP (Fully Sharded Data Parallel) is suitable for training 14B models at high resolutions and can effectively save VRAM.

**Training Command**:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
export MODEL_NAME_AUDIO="models/Diffusion_Transformer/wav2vec2-base-960h"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=AudioAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/fantasytalking/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_audio_model_name_or_path=$MODEL_NAME_AUDIO \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_fantasytalking" \
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
  --transformer_path="models/Personalized_Model/FantasyTalking/fantasytalking_model.ckpt" \
  --trainable_modules "processor." "proj_model."
```

**FSDP Key Parameters**:

| Parameter | Description |
|-----------|-------------|
| `--use_fsdp` | Enable FSDP |
| `--fsdp_auto_wrap_policy` | Auto wrap policy: `TRANSFORMER_BASED_WRAP` |
| `--fsdp_transformer_layer_cls_to_wrap` | Transformer layer class name to wrap: `AudioAttentionBlock` |
| `--fsdp_sharding_strategy` | Sharding strategy: `FULL_SHARD` |
| `--fsdp_state_dict_type` | State dict type: `SHARDED_STATE_DICT` |
| `--fsdp_backward_prefetch` | Backward prefetch: `BACKWARD_PRE` |
| `--fsdp_cpu_ram_efficient_loading` | CPU RAM efficient loading: `False` |

### 3.6 Training without DeepSpeed or FSDP

If you only have one GPU or don't want to use distributed training, you can run directly:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
export MODEL_NAME_AUDIO="models/Diffusion_Transformer/wav2vec2-base-960h"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/fantasytalking/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_audio_model_name_or_path=$MODEL_NAME_AUDIO \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_fantasytalking" \
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
  --transformer_path="models/Personalized_Model/FantasyTalking/fantasytalking_model.ckpt" \
  --trainable_modules "processor." "proj_model."
```

### 3.7 Multi-Node Distributed Training

**Use Case**: Ultra-large-scale datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines with 8 GPUs each:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
export MODEL_NAME_AUDIO="models/Diffusion_Transformer/wav2vec2-base-960h"  # If None, will use $MODEL_NAME/audio_encoder
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # Total number of machines
export NUM_PROCESS=16                # Total processes = machines × 8
export RANK=0                        # Current machine rank (0 or 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 for multi-node environment without RDMA
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/fantasytalking/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_audio_model_name_or_path=$MODEL_NAME_AUDIO \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_fantasytalking" \
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
  --transformer_path="models/Personalized_Model/FantasyTalking/fantasytalking_model.ckpt" \
  --trainable_modules "processor." "proj_model."
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
export MODEL_NAME_AUDIO="models/Diffusion_Transformer/wav2vec2-base-960h"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export MASTER_ADDR="192.168.1.100"  # Same as Master
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # Note this is 1
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 for multi-node environment without RDMA
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

After training is complete, you can use the inference script to test the generated model.

### 4.1 Inference Parameters

Main parameters in the inference script [predict_s2v.py](file:///home/zhoumo.xjq/disk1/Code/VideoX-Fun-Github/examples/fantasytalking/predict_s2v.py):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `GPU_memory_mode` | GPU memory mode: `model_full_load`, `model_full_load_and_qfloat8`, `model_cpu_offload`, `model_cpu_offload_and_qfloat8`, `sequential_cpu_offload` | `sequential_cpu_offload` |
| `ulysses_degree` | Ulysses parallelism degree for multi-GPU inference | `1` |
| `ring_degree` | Ring parallelism degree for multi-GPU inference | `1` |
| `fsdp_dit` | Use FSDP for Transformer during multi-GPU inference to save memory | `False` |
| `compile_dit` | Compile Transformer for faster inference (fixed resolution only) | `False` |
| `config_path` | Model configuration file path | `config/wan2.1/wan_civitai.yaml` |
| `model_name` | Model path | `models/Diffusion_Transformer/Wan2.1-I2V-14B-720P` |
| `model_name_audio` | Audio encoder path | `None` |
| `sampler_name` | Sampler type: `Flow`, `Flow_Unipc`, `Flow_DPM++` | `Flow` |
| `shift` | Sampler shift parameter | 5.0 |
| `transformer_path` | Trained Transformer weights path | `models/Personalized_Model/FantasyTalking/fantasytalking_model.ckpt` |
| `vae_path` | Trained VAE weights path | `None` |
| `lora_path` | LoRA weights path | `None` |
| `sample_size` | Generated video resolution `[height, width]` | `[832, 480]` |
| `video_length` | Generated video frame count | `81` |
| `fps` | Frames per second | `23` |
| `weight_dtype` | Model weight dtype, use `torch.float16` for GPUs without bf16 | `torch.bfloat16` |
| `validation_image_start` | Reference image path | `"asset/8.png"` |
| `audio_path` | Input audio path | `"asset/talk.wav"` |
| `prompt` | Generation prompt | `"A girl talking by the sea."` |
| `negative_prompt` | Negative prompt | See code |
| `guidance_scale` | Prompt guidance strength | `4.5` |
| `audio_guide_scale` | Audio guidance strength | `4.0` |
| `seed` | Random seed for reproducibility | `43` |
| `num_inference_steps` | Inference steps | `40` |
| `lora_weight` | LoRA weight strength | `0.55` |
| `save_path` | Generated video save path | `samples/fantasy-talking-videos-speech2v` |

**TeaCache Acceleration Configuration**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enable_teacache` | Enable TeaCache acceleration | `True` |
| `teacache_threshold` | TeaCache threshold (recommended 0.05~0.30) | `0.10` |
| `num_skip_start_steps` | Initial steps to skip TeaCache | `5` |
| `teacache_offload` | Offload TeaCache tensors to CPU to save memory | `False` |

**GPU Memory Mode Description**:

| Mode | Description | Memory Usage |
|------|-------------|--------------|
| `model_full_load` | Load entire model to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `sequential_cpu_offload` | Offload each layer to CPU after use (slowest) | Lowest |

### 4.2 Single GPU Inference

Run single GPU inference:

```bash
python examples/fantasytalking/predict_s2v.py
```

Edit [examples/fantasytalking/predict_s2v.py](file:///home/zhoumo.xjq/disk1/Code/VideoX-Fun-Github/examples/fantasytalking/predict_s2v.py) according to your needs. For first-time inference, focus on modifying the following parameters. For other parameters, see the inference parameters description above.

```python
# Choose based on your GPU memory
GPU_memory_mode = "model_full_load"
# Model configuration file path
config_path = "config/wan2.1/wan_civitai.yaml"
# Your actual model path
model_name = "models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
# Audio encoder path
model_name_audio = None  # or "models/Diffusion_Transformer/wav2vec2-base-960h"
# Trained weights path, e.g., "output_dir_fantasytalking/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = "models/Personalized_Model/FantasyTalking/fantasytalking_model.ckpt"
# Reference image path
validation_image_start = "asset/8.png"
# Input audio path
audio_path = "asset/talk.wav"
# Generation prompt
prompt = "A girl talking by the sea."
# ...
```

### 4.3 Multi-GPU Parallel Inference

**Use Case**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit [examples/fantasytalking/predict_s2v.py](file:///home/zhoumo.xjq/disk1/Code/VideoX-Fun-Github/examples/fantasytalking/predict_s2v.py):

```python
# Ensure ulysses_degree × ring_degree = number of GPUs used
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelism
ring_degree = 1     # Sequence dimension parallelism
```

**Configuration Principles**:
- `ulysses_degree` must be divisible by the model's head count
- `ring_degree` splits along the sequence dimension and affects communication overhead. Try to avoid using it if heads are divisible.

**Configuration Examples**:

| GPU Count | ulysses_degree | ring_degree | Description |
|-----------|---------------|-------------|-------------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelism |
| 8 | 2 | 4 | Hybrid parallelism |
| 8 | 8 | 1 | Head parallelism |

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/fantasytalking/predict_s2v.py
```

---

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
