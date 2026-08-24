# MiniMax-H3 Full Parameter Training Guide

This document provides a complete workflow for full parameter training of MiniMax-H3, including environment configuration, data preparation, distributed training, and inference testing.

> **Note**: MiniMax-H3 is an audio-visual generative video model that can simultaneously generate video and corresponding audio. The training data requires both video and audio files.

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
  - [3.2 Quick Start (FSDP)](#32-quick-start-fsdp)
  - [3.3 Common Training Parameters](#33-common-training-parameters)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training with DeepSpeed-Zero-2](#35-training-with-deepspeed-zero-2)
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

> ⚠️ **Important**: MiniMax-H3 is an audio-visual generative model. Unlike regular video training, you **must provide the `audio_path` field in metadata.json**. The paired waveform is resampled to the audio VAE's sample rate (32 kHz) during training.

**Relative Path Format** (example):
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

**Key Fields Description**:
- `file_path`: Video file path (relative or absolute)
- `audio_path`: Audio file path (**MiniMax-H3 specific and required**, main difference from regular video training)
  - Audio files are typically in `.wav` format
  - Path should correspond to `file_path`, e.g., `train/video001.mp4` corresponds to `wav/audio001.wav`
- `text`: Video description (English prompt)
- `type`: Data type, fixed as `"video"`
- `width` / `height`: Video dimensions (**recommended** to provide for bucket training; if not provided, they will be automatically read during training, which may slow down training when data is stored on slow systems like OSS)
  - You can use `scripts/process_json_add_width_and_height.py` to add width and height fields to JSON files without these fields, supporting both images and videos
  - Usage: `python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Audios-Demo/metadata.json --output_file datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json`

**Dataset Comparison: MiniMax-H3 vs Regular Video Training**:

| Model Type | Required Fields | Audio Field |
|---------|---------|---------|
| Regular Video (WAN, CogVideoX, etc.) | `file_path`, `text`, `type` | ❌ Not needed |
| **MiniMax-H3 (Audio-Visual Generation)** | `file_path`, `audio_path`, `text`, `type` | ✅ **Required** |

### 2.4 Relative vs Absolute Path Usage

**Relative Paths**:

If your data uses relative paths, configure the training script as follows:

```bash
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
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

# Download MiniMax-H3 official weights
hf download MiniMax-AI/MiniMax-H3 --local-dir models/Diffusion_Transformer/MiniMax-H3
```

> 💡 The loader accepts either the converted diffusers layout above or an *original* MiniMax-H3 partition (e.g. `MiniMax-H3/FL2VA`); the original shards are converted on the fly while loading, with no intermediate copy on disk.

### 3.2 Quick Start (FSDP)

If you have downloaded the data as per **2.1 Quick Test Dataset** and the weights as per **3.1 Download Pretrained Model**, you can directly copy and run the quick start command.

FSDP is recommended for training MiniMax-H3. The transformer alone is about 62 GB in bfloat16 and the Qwen3-VL conditioner is another 62 GB, so the model weights must be sharded across GPUs — which FSDP (`FULL_SHARD`) does but DeepSpeed-Zero-2 does not.

```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/minimax_h3/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --trainable_modules "." \
  --t2v_ratio=0.25 \
  --train_mode="fl2v"
```

### 3.3 Common Training Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--pretrained_model_name_or_path` | Path to pretrained model | `models/Diffusion_Transformer/MiniMax-H3` |
| `--train_data_dir` | Training data directory | `datasets/X-Fun-Videos-Audios-Demo/` |
| `--train_data_meta` | Training data metadata file | `datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json` |
| `--train_batch_size` | Samples per batch | 1 |
| `--video_sample_size` | Maximum video resolution for training | 960 |
| `--token_sample_size` | Token length sampling size | 960 |
| `--video_sample_stride` | Frame sampling stride (MiniMax-H3 is 24 fps) | 1 |
| `--video_sample_n_frames` | Number of frames to sample, must follow the `17*n+5` form of the video VAE (duration stays between 5 and 15 seconds) | 124 |
| `--video_repeat` | Number of times each video is repeated per epoch | 1 |
| `--gradient_accumulation_steps` | Gradient accumulation steps (equivalent to larger batch) | 1 |
| `--dataloader_num_workers` | DataLoader subprocesses | 4 |
| `--num_train_epochs` | Number of training epochs | 100 |
| `--checkpointing_steps` | Save checkpoint every N steps | 500 |
| `--learning_rate` | Initial learning rate | 1e-5 |
| `--lr_scheduler` | Learning rate scheduler | `constant_with_warmup` |
| `--lr_warmup_steps` | Learning rate warmup steps | 100 |
| `--seed` | Random seed | 42 |
| `--output_dir` | Output directory | `output_dir_minimax_h3` |
| `--gradient_checkpointing` | Enable activation checkpointing | - |
| `--gradient_checkpointing_save_on_cpu` | Offload the activations saved for backward of the transformer blocks to CPU memory | - |
| `--mixed_precision` | Mixed precision: `fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW weight decay | 3e-2 |
| `--adam_epsilon` | AdamW epsilon value | 1e-10 |
| `--vae_mini_batch` | Mini-batch size for VAE encoding | 1 |
| `--max_grad_norm` | Gradient clipping threshold | 1.0 |
| `--random_hw_adapt` | Auto-scale videos to random size in range `[512, video_sample_size]` | - |
| `--training_with_video_token_length` | Train based on token length instead of fixed resolution | - |
| `--enable_bucket` | Enable bucket training: trains entire videos grouped by resolution without center cropping | - |
| `--uniform_sampling` | Uniform timestep sampling | - |
| `--low_vram` | Keep VAE and conditioner on CPU, move to GPU only while encoding | - |
| `--train_mode` | `t2v` (text only) or `fl2v` (first-frame keyframe conditioning, the keyframe taken from the training sample itself) | `fl2v` |
| `--t2v_ratio` | Under `--train_mode=fl2v`, the fraction of steps that drop the keyframe and train t2v instead, so one run keeps both conditionings. Must be in [0, 1] and only applies to fl2v; 0 trains fl2v only | 0.25 |
| `--resume_from_checkpoint` | Resume training from checkpoint path, use `"latest"` to auto-select latest | None |
| `--trainable_modules` | Trainable modules (`"."` means all modules) | `"."` |
| `--validation_steps` | Execute validation every N steps | 2000 |
| `--validation_epochs` | Execute validation every N epochs | 5 |
| `--validation_prompts` | Prompts used during validation | `"A man in a blue blazer..."` |


### 3.4 Training Validation

You can configure validation parameters to periodically generate test videos during training, allowing you to monitor training progress and model quality.

**Validation Parameters**:

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--validation_steps` | Execute validation every N steps | 2000 |
| `--validation_epochs` | Execute validation every N epochs | 5 |
| `--validation_prompts` | Prompt for validation video generation. Use multiple space-separated prompt strings | Space-separated prompt strings |

**Example**:

```bash
  --validation_steps=2000 \
  --validation_epochs=5 \
  --validation_prompts="A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks."
```

**Notes**:
- Validation videos will be saved to the `output_dir/sample/` directory
- For multi-prompt validation, use: `--validation_prompts "prompt1" "prompt2" "prompt3"`

### 3.5 Training with DeepSpeed-Zero-2

> ⚠️ **Warning**: DeepSpeed-Zero-2 only partitions optimizer states and gradients, **not the model weights**. The MiniMax-H3 transformer is about 62 GB, so each GPU still holds a full weight replica and this setup usually runs out of memory. Prefer FSDP (**3.2**) for MiniMax-H3; the command below is provided for reference only.

```sh
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/minimax_h3/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --trainable_modules "." \
  --t2v_ratio=0.25 \
  --train_mode="fl2v"
```

### 3.6 Training Without DeepSpeed or FSDP

**This approach is not recommended as it lacks VRAM-saving backends**. With MiniMax-H3's ~62 GB transformer and ~62 GB Qwen3-VL conditioner replicated on every GPU, it will almost certainly run out of memory. This is provided for reference only.

```sh
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/minimax_h3/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --trainable_modules "." \
  --t2v_ratio=0.25 \
  --train_mode="fl2v"
```

### 3.7 Multi-Machine Distributed Training

**Suitable for**: Ultra-large-scale datasets, faster training speed

#### 3.7.1 Environment Configuration

Assuming 2 machines with 8 GPUs each:

**Machine 0 (Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
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

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/minimax_h3/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --trainable_modules "." \
  --t2v_ratio=0.25 \
  --train_mode="fl2v"
```

**Machine 1 (Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
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

## 4. Inference Testing

### 4.1 Inference Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|------|------|-------|
| `GPU_memory_mode` | GPU memory mode, see table below for options | `model_cpu_offload` |
| `ulysses_degree` | Head dimension parallelization degree, 1 for single GPU | 1 |
| `ring_degree` | Sequence dimension parallelization degree, 1 for single GPU | 1 |
| `fsdp_dit` | Use FSDP for Transformer in multi-GPU inference to save VRAM | `False` |
| `fsdp_text_encoder` | Use FSDP for the Qwen3-VL text encoder in multi-GPU inference to save VRAM | `False` |
| `compile_dit` | Compile Transformer to accelerate inference (effective at fixed resolution) | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/MiniMax-H3` |
| `transformer_path` | Path to trained Transformer weights | `None` |
| `vae_path` | Path to trained VAE weights | `None` |
| `lora_path` | LoRA weights path | `None` |
| `sample_size` | Generated video resolution `[height, width]`; height/width must be multiples of 32. `None` uses MiniMax-H3's own 16:9 canvas (768x1344) | `[704, 1280]` |
| `video_length` | Number of frames to generate, snapped up to the next `17*n+5` the video VAE can decode (duration stays between 5 and 15 seconds) | 124 |
| `fps` | Frames per second (MiniMax-H3 generates at a fixed 24 fps) | 24 |
| `weight_dtype` | Model weight precision, use `torch.float16` for GPUs without bf16 support | `torch.bfloat16` |
| `prompt` | Positive prompt describing the content to generate | `"A red fox trotting..."` |
| `seed` | Random seed for reproducibility | 43 |
| `num_inference_steps` | Number of denoising steps, i.e. of model evaluations; the sigma grid it is built from holds one more point, the terminal 0 | 40 |
| `guidance_scale` | Guidance strength. The released checkpoint is guidance-distilled: keep it at 1 to run one forward pass per step with no CFG; a value above 1 enables classifier-free guidance with two passes | 1 |
| `flow_shift` | Exponential sigma shift of the video schedule, `None` keeps the one of the checkpoint (12.0) | `None` |
| `audio_flow_shift` | Exponential sigma shift of the audio schedule, `None` keeps the one of the checkpoint (3.0) | `None` |
| `lora_weight` | LoRA weight strength | 0.55 |
| `save_path` | Generated video save path | `samples/minimax-h3-videos-t2v` |

**GPU Memory Mode Description**:

| Mode | Description | VRAM Usage |
|------|------|---------|
| `model_full_load` | Load entire model to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Layer group offload between CPU/CUDA | Low |
| `sequential_cpu_offload` | Offload each layer individually (slowest) | Lowest |

> 💡 The transformer alone is 61.7 GB in bfloat16 and the Qwen3-VL conditioner is another 62.1 GB, so a single 80 GB card needs `model_cpu_offload` or `model_group_offload`.

### 4.2 Single GPU Inference

Run single GPU inference with:

```bash
python examples/minimax_h3/predict_t2v.py
```

Edit `examples/minimax_h3/predict_t2v.py` according to your needs. For first-time inference, focus on these parameters. For other parameters, see the Inference Parameters section above.

```python
# Choose based on your GPU VRAM
GPU_memory_mode = "model_cpu_offload"
# Your actual model path
model_name = "models/Diffusion_Transformer/MiniMax-H3"  
# Trained weights path, e.g. "output_dir_minimax_h3/checkpoint-xxx/transformer"
transformer_path = None  
# Write based on content to generate
prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"  
# ...
```

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser yunchang
```

#### Configure Parallel Strategy

Edit `examples/minimax_h3/predict_t2v.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelization
ring_degree = 1     # Sequence dimension parallelization
```

**Configuration Principles**:
- `ulysses_degree` must evenly divide the model's number of heads
- `ring_degree` splits on sequence dimension, affecting communication overhead; avoid using it when heads can be divided
- Multi-GPU runs through the xfuser sequence-parallel path and is **incompatible with the `*cpu_offload*` memory modes** (accelerate offload hooks own a single device); use `model_full_load` / `model_full_load_and_qfloat8` across GPUs there, with `fsdp_dit` / `fsdp_text_encoder` to save memory

**Example Configurations**:

| GPU Count | ulysses_degree | ring_degree | Description |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | Single GPU |
| 4 | 4 | 1 | Head parallelization |
| 8 | 2 | 4 | Hybrid parallelization |
| 8 | 8 | 1 | Head parallelization |

#### Run Multi-GPU Inference

```bash
torchrun --nproc_per_node=2 examples/minimax_h3/predict_t2v.py
```

## 5. Additional Resources

- **MiniMax-H3 Official GitHub**: https://github.com/MiniMax-AI/MiniMax-H3
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
