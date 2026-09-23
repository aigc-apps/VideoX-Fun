# Qwen-Image 2.1 Control (ControlNet-Union) Training Guide

This document provides a complete workflow for training a **ControlNet-Union** adapter on top of the frozen
**Qwen-Image 2.1** base transformer: environment setup, data preparation, distributed training, CFG
distillation, and inference testing.

A parallel chain of zero-initialized `control_blocks` produces a per-layer skip (`hints`) that is added back into
the frozen base blocks, so the adapter starts as an identity skip and learns control gradually. Only the control
modules are trained (`--trainable_modules "control"`).

The adapter is a **union** of control + inpaint: the conditioning tensor `control_context` packs
`[control_latents (64) | mask (1) | masked-image latents (64)] = 129` channels, so one adapter handles both
spatial control (depth / canny / pose / …) and inpainting.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Dataset Structure](#22-dataset-structure)
  - [2.3 metadata.json Format](#23-metadatajson-format)
  - [2.4 Relative vs Absolute Path Usage](#24-relative-vs-absolute-path-usage)
- [3. Control Training](#3-control-training)
  - [3.1 Download Pre-trained Model](#31-download-pre-trained-model)
  - [3.2 Quick Start (DeepSpeed-Zero-2)](#32-quick-start-deepspeed-zero-2)
  - [3.3 Common Training Parameters](#33-common-training-parameters)
  - [3.4 Training Validation](#34-training-validation)
  - [3.5 Training with FSDP](#35-training-with-fsdp)
  - [3.6 Other Backends](#36-other-backends)
  - [3.7 Multi-machine Distributed Training](#37-multi-machine-distributed-training)
  - [3.8 CFG Distillation](#38-cfg-distillation)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Inference Parameters](#41-inference-parameters)
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
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**Option 3: Using Docker**

When using Docker, please ensure that the GPU drivers and CUDA environment are correctly installed, then execute the following commands:

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

> **Qwen-Image 2.1 specific**: the text encoder is a **Qwen3-VL** model, so the environment needs a `transformers`
> build that ships the `qwen3_vl` architecture (newer than the base pin in `requirements.txt`). If
> `Qwen3VLForConditionalGeneration` / `Qwen3VLProcessor` import as `None`, your `transformers` is too old.
>
> The YOLO object-mask feature (see [2.3](#23-metadatajson-format)) needs `ultralytics`; `yolov8x-seg.pt`
> downloads automatically on first use.

---

## 2. Data Preparation

Control training uses `ImageVideoControlDataset`.

### 2.1 Quick Test Dataset

We provide a test dataset containing several training samples with corresponding control files.

```bash
# Download official example dataset
modelscope download --dataset PAI/X-Fun-Images-Controls-Demo --local_dir ./datasets/X-Fun-Images-Controls-Demo
```

### 2.2 Dataset Structure

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/                 # target images (what the model should generate)
│   │   ├── 📄 image001.jpg
│   │   └── 📄 ...
│   ├── 📂 control/               # paired control / condition images (pose, canny, depth, ...)
│   │   ├── 📄 image001.png
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json Format

The manifest is the standard image metadata JSON plus one extra `control_file_path` field that pairs each
**target** image with its **control** image.

```json
[
    {
      "file_path": "train/image001.jpg",
      "control_file_path": "control/image001.png",
      "text": "A young woman, studio lighting, high quality.",
      "width": 1024,
      "height": 1024,
      "type": "image"
    }
]
```

**Key field descriptions**:
- `file_path`: the **target** image (relative or absolute).
- `control_file_path`: the **control / condition** image (pose map, edge map, depth, gray sketch, …). It is loaded
  as RGB and resized/cropped with the **same** transform as the target, so they stay pixel-aligned.
- `text`: caption.
- `width` / `height`: recommended for bucket training; use `scripts/process_json_add_width_and_height.py` to add
  them to a JSON that lacks them.
- `type`: `"image"`.

> **You only supply the target + control images. You do NOT supply masks.** The inpaint mask is generated on the
> fly:
> - A random rectangular hole via `get_random_mask` in the collate.
> - Then, on a random ~70% subset of frames, an **irregular object-shaped mask** produced by a **YOLO-seg**
>   detector (`ObjectInstanceDetector`), with its edges randomly dilated / eroded / Gaussian-blurred. This mirrors
>   `scripts/qwenimage_fun/train_control.py` and gives the model realistic, object-shaped inpaint holes instead of
>   only rectangles.
> - The masked image fed to the union branch is always `target * (1 - mask)`.

> **RGBA note**: the 2.1 VAE reads RGBA. Training images are loaded as RGB and automatically composited over an
> opaque alpha channel before encoding, so you do not need to provide RGBA data.

### 2.4 Relative vs Absolute Path Usage

**Relative paths** (small, local dataset):
```bash
export DATASET_NAME="datasets/X-Fun-Images-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Controls-Demo/metadata_add_width_height.json"
```

**Absolute paths** (NAS / OSS / multi-machine shared data):
```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> If the dataset is stored on external storage or shared across machines, prefer absolute paths.

---

## 3. Control Training

### 3.1 Download Pre-trained Model

Point `MODEL_NAME` at a local **Qwen-Image 2.1** checkpoint directory. Its `transformer/` subfolder supplies the
frozen base weights; the control modules are zero-initialized on load.

```bash
mkdir -p models/Diffusion_Transformer
# Place your Qwen-Image 2.1 weights here, e.g.
# models/Diffusion_Transformer/Qwen-Image-2.1/{transformer,vae,text_encoder,...}
```

> **No released 2.1 ControlNet-Union checkpoint.** Unlike Qwen-Image 2512, there is currently no published
> `...-Fun-Controlnet-Union.safetensors` for 2.1, so training starts from scratch with the **zero-initialized**
> control branch. Consequently the launcher leaves `--transformer_path` out; only add it to resume or fine-tune a
> control checkpoint you have already trained (or produced with `scripts/*/extract_control_weights.py`).

### 3.2 Quick Start (DeepSpeed-Zero-2)

It is recommended to use DeepSpeed-Zero-2 or FSDP for training, which can save a significant amount of GPU memory.

After following **2.1 Quick Test Dataset** and **3.1 Download Pre-trained Model**, you can directly copy and run the following command:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/qwenimage21_fun/train_control.py \
  --config_path="config/qwenimage21/qwenimage21_control.yaml" \
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
  --output_dir="output_dir_qwen_image_21_control" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "control"
```

### 3.3 Common Training Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config_path` | **Required.** Builds the control transformer with `control_layers` / `control_in_dim` | `config/qwenimage21/qwenimage21_control.yaml` |
| `--pretrained_model_name_or_path` | Base Qwen-Image 2.1 model (frozen weights) | `models/Diffusion_Transformer/Qwen-Image-2.1` |
| `--train_data_dir` / `--train_data_meta` | Dataset root / manifest JSON | `""` / `/path/metadata.json` |
| `--trainable_modules` | `"control"` trains only `control_blocks.*` + `control_img_in.*`; base stays frozen | `"control"` |
| `--transformer_path` | **Omit for from-scratch** training; add only to resume/finetune a trained control branch | *(none)* |
| `--image_sample_size` | Max training resolution, auto bucketing | `1024` |
| `--train_batch_size` / `--gradient_accumulation_steps` | Per-device batch / accumulation | `1` / `1` |
| `--learning_rate` | Initial learning rate | `2e-05` |
| `--lr_scheduler` / `--lr_warmup_steps` | Scheduler / warmup | `constant_with_warmup` / `100` |
| `--checkpointing_steps` | Save a checkpoint every N steps | `50` |
| `--gradient_checkpointing` | Activation recomputation | flag |
| `--vae_mini_batch` | Mini-batch size for VAE encoding (control encodes 3 streams) | `1` |
| `--max_grad_norm` | Gradient clipping | `0.05` |
| `--enable_bucket` | Bucket training by resolution without cropping | flag |
| `--uniform_sampling` | Uniform timestep sampling | flag |
| `--low_vram` | Offload VAE / text encoder when idle to save memory | flag (optional) |

> **Memory**: each control step encodes **three** latent streams (target, control, masked image), so it is heavier
> than base training. Keep `--vae_mini_batch=1`; add `--low_vram` if you are tight on memory.

### 3.4 Training Validation

Configure validation during training to periodically render control previews:

```bash
  --validation_paths "asset/pose.jpg" \
  --validation_steps=50 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, ... solo, upper_body"
```

- `--validation_prompts` and `--validation_paths` must have **matching counts**; entry `i` of each pair is used
  together. The output resolution is derived from each control image's aspect ratio via
  `calculate_dimensions(image_sample_size^2, w/h)`.
- Validation triggers on either `--validation_steps` or `--validation_epochs`.
- Previews are written to `{output_dir}/sample/`. Because the 2.1 VAE decodes to **RGBA**, previews are saved as
  **`.png`** (JPEG cannot store alpha).
- `log_validation` is wrapped in `try/except`: a bad control path only logs `Eval error on rank N` and never
  crashes training. To confirm validation actually produced images, check `output_dir/sample/` **and** grep the log
  for `Eval error`.

### 3.5 Training with FSDP

If DeepSpeed-Zero-2 runs out of GPU memory, you can switch to FSDP for training. The launcher `scripts/qwenimage21_fun/train_control.sh` runs
exactly the command below; edit the paths at the top (`MODEL_NAME`, `DATASET_META_NAME`, …) and run it with `bash` if you prefer
(the wrap classes must match the control model's blocks):

```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap=BaseQwenImage21TransformerBlock,QwenImage21ControlTransformerBlock \
  --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
  scripts/qwenimage21_fun/train_control.py \
  --config_path="config/qwenimage21/qwenimage21_control.yaml" \
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
  --output_dir="output_dir_qwen_image_21_control" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "control"
```

### 3.6 Other Backends

#### 3.6.1 Training without DeepSpeed and FSDP

Using neither DeepSpeed nor FSDP may result in insufficient GPU memory; only recommended when GPU memory is
sufficient. Plain DDP also has to replicate the 2.1 base transformer on every GPU, so it is generally not
recommended:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/qwenimage21_fun/train_control.py \
  --config_path="config/qwenimage21/qwenimage21_control.yaml" \
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
  --output_dir="output_dir_qwen_image_21_control" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "control"
```

### 3.7 Multi-machine Distributed Training

**Suitable for**: Ultra-large-scale datasets, faster training speed

#### 3.7.1 Environment Configuration

When using multi-machine training, please set the following environment variables:

```bash
export MASTER_ADDR="your master address"
export MASTER_PORT=10086
export WORLD_SIZE=1 # The number of machines
export NUM_PROCESS=8 # The number of processes, such as WORLD_SIZE * 8
export RANK=0 # The rank of this machine

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK scripts/qwenimage21_fun/train_control.py \
  [other training parameters...]
```

#### 3.7.2 Multi-machine Training Considerations

- **Network Requirements**:
   - Recommended: RDMA/InfiniBand (high performance)
   - Without RDMA, add environment variables:
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **Data Synchronization**: All machines must be able to access the same data paths (NFS/shared storage)

### 3.8 CFG Distillation

`train_control_distill.py` / `train_control_distill.sh` are an **optional second stage**. They distill
classifier-free guidance (CFG) into the trained control branch, so that **inference needs no guidance scale**.

Algorithm (identical in spirit to `scripts/minimax_h3_fun` / `scripts/flux2_fun` control distillation):
- A **frozen teacher** (a second copy of the same control model, loaded with the **same** `--transformer_path` —
  your stage-1 trained control branch) runs two forward passes per step, on the prompt and on an **empty**
  negative prompt, both **with the control condition**. The two velocities combine into the CFG target:
  `target = uncond + (cond - uncond) * real_guidance_scale`.
- The trainable **student** (control branch) runs a single conditional forward and regresses onto that target
  (MSE in velocity space). Only `--trainable_modules "control"` is trained.

Run it after you have a trained control checkpoint:

```bash
# Set CONTROL_TRANSFORMER_PATH to the stage-1 checkpoint's
# output_dir_qwen_image_21_control/<ts>/checkpoint-<step>/diffusion_pytorch_model.safetensors
bash scripts/qwenimage21_fun/train_control_distill.sh
```

Distillation-specific parameters:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--transformer_path` | **Required**: the trained control branch both student and teacher load | `/root/diffusion_pytorch_model.safetensors` |
| `--real_guidance_scale` | CFG scale applied to the teacher to build the target | `3.5` |
| `--learning_rate` | Lower LR for distillation | `2e-06` |
| `--output_dir` | Separate output dir for the distilled adapter | `output_dir_qwen_image_21_control_distill` |

> The teacher is a separate, unsharded bf16 copy on each GPU (only the student is FSDP-sharded), which is
> memory-heavy. Add `--low_vram` to stream the teacher only for its two forward passes. A distilled student is run
> at inference with `guidance_scale = 1.0` (CFG is already baked into the weights).

---

## 4. Inference Testing

### 4.1 Inference Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `config_path` | Must match the trained adapter's config | `config/qwenimage21/qwenimage21_control.yaml` |
| `model_name` | Base Qwen-Image 2.1 path | `models/Diffusion_Transformer/Qwen-Image-2.1` |
| `transformer_path` | Trained control weights (`control_*` keys load with `strict=False`), or `None` for the base | `output_dir_qwen_image_21_control/.../diffusion_pytorch_model.safetensors` |
| `sampler_name` | Flow-matching sampler | `Flow` |
| `sample_size` | Output canvas `[height, width]` | `[1728, 992]` |
| `control_image` | Control condition image (`predict_t2i_control.py`) | `asset/pose.jpg` |
| `control_image_path` | Optional control condition image (`predict_i2i_inpaint.py`, defaults to `None`) | `asset/pose.jpg` |
| `control_context_scale` | Control-branch strength (the value the adapter was trained to consume) | `1.0` |
| `image_path` / `mask_path` | Inpaint source image / mask (only `predict_i2i_inpaint.py`, see 4.2) | `asset/pose.jpg` / `asset/mask.png` |
| `guidance_scale` | CFG strength. `1.0` for a CFG-distilled checkpoint | `1.0` |
| `weight_dtype` | Use `torch.float16` on GPUs without bf16 (v100, 2080Ti, …) | `torch.bfloat16` |
| `GPU_memory_mode` | GPU memory management mode, see table below | `model_group_offload` |
| `ulysses_degree` / `ring_degree` | Multi-GPU parallelism (see 4.3). `ring_degree` must stay `1` | `1` / `1` |
| `num_inference_steps` / `seed` | Sampling steps / seed | `40` / `43` |
| `save_path` | Output directory | `samples/qwenimage21-control-images` |

**GPU Memory Management Modes**:

| Mode | Description | Memory Usage |
|------|------|---------|
| `model_full_load` | Load entire model to GPU | Highest |
| `model_full_load_and_qfloat8` | Full load + FP8 quantization | High |
| `model_cpu_offload` | Offload model to CPU after use | Medium |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 quantization | Medium-Low |
| `model_group_offload` | Switch layer groups between CPU/CUDA | Low |
| `sequential_cpu_offload` | Layer-by-layer offload (slowest) | Lowest |

### 4.2 Single GPU Inference

#### Quick Start

```bash
python examples/qwenimage21_fun/predict_t2i_control.py
```

Edit the top-of-file constants to match your setup. The pipeline preprocesses and VAE-encodes `control_image`
(accepts a PIL image / a path), builds the 129-channel `control_context`, and injects the control skips:

```python
GPU_memory_mode     = "model_group_offload"
model_name          = "models/Diffusion_Transformer/Qwen-Image-2.1"
transformer_path    = "models/Personalized_Model/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"  # or your trained checkpoint's diffusion_pytorch_model.safetensors
control_image       = "asset/pose.jpg"
control_context_scale = 1.0
prompt              = "A young woman with long straight black hair ..."
sample_size         = [1728, 992]
num_inference_steps = 40
```

Results are saved to `samples/qwenimage21-control-images/*.png`.

> The KV cache is **disabled automatically** when `control_context` is present: the control skip depends on the
> per-step base joint stream, so a prefix cache would be invalid.

**Image Inpainting Inference**:

The union adapter also does inpainting. `predict_i2i_inpaint.py` feeds `image_path` + `mask_path` (and leaves
`control_image_path` as `None`), so only the inpaint half of the 129-channel context is used:

```bash
python examples/qwenimage21_fun/predict_i2i_inpaint.py
```

`mask_path` semantics: **white (`>= 0.5`) = repaint**, black = keep. You can supply a control image *and* the
inpaint pair together to use the full union.

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, faster inference

Qwen-Image 2.1 supports **Ulysses (head-parallel) sequence parallelism only**.

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/qwenimage21_fun/predict_t2i_control.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs
# For example, using 4 GPUs:
ulysses_degree = 4  # Head dimension parallelism
ring_degree = 1     # Sequence dimension parallelism, must stay 1
```

**Configuration Principles**:
- `ulysses_degree` must divide `num_attention_heads` (32): one of `1/2/4/8/16/32`.
- `ring_degree` **must stay 1** — ring attention rotates KV chunks and cannot express 2.1's block-causal mask or
  its prefix KV cache.

**Example Configurations**:

| GPU count | ulysses_degree | ring_degree |
|-----------|----------------|-------------|
| 1 | 1 | 1 |
| 4 | 4 | 1 |
| 8 | 8 | 1 |

#### Run Multi-GPU Inference

```bash
# Set ulysses_degree > 1, keep ring_degree = 1, and GPU count = ulysses_degree * ring_degree.
torchrun --nproc-per-node=4 examples/qwenimage21_fun/predict_t2i_control.py
```

---

## 5. Additional Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
- **Qwen-Image Official Repository**: https://github.com/QwenLM/Qwen-Image
