# MiniMax-H3-Image Training Guide

This document provides a complete workflow for full-parameter training of MiniMax-H3-Image, the single-stream packed-sequence image transformer of the MiniMax-H3 family: environment configuration, data preparation, full parameter training, and inference testing. LoRA training is covered in `scripts/minimax_h3_image/README_TRAIN_LORA.md`.

MiniMax-H3-Image generates and edits images with the same model: a request is packed into one sequence laid out as `[text | reference images | generated image]`, so `--train_mode=t2i` and `--train_mode=edit` differ only in whether the samples of the meta carry a `source_file_path`.

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
  - [3.5 Image Editing Training](#35-image-editing-training)
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
# Versions used to verify MiniMax-H3-Image (includes Qwen3-VL support).
pip install "transformers==4.57.3" "diffusers==0.32.2" "accelerate==1.7.0" "peft==0.17.0"
```

**Method 2: Manual Dependency Installation**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft==0.17.0" "accelerate==1.7.0" "gradio>=3.41.2" "diffusers==0.32.2" "transformers==4.57.3"
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

We provide two test datasets: one for text-to-image, one for image editing.

```bash
# Download official demo dataset
modelscope download --dataset PAI/X-Fun-Images-Demo --local_dir ./datasets/X-Fun-Images-Demo
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
│   ├── 📂 source/                     # Reference images, only for `--train_mode=edit`
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
    "text": "Make the jacket red",
    "width": 1328,
    "height": 1328,
    "source_file_path": "source/image002.png"
  }
]
```

**Absolute Path Format**:
```json
[
  {
    "file_path": "/absolute/path/to/images/sunset.jpg",
    "text": "A beautiful sunset over the ocean",
    "width": 1024,
    "height": 1024
  }
]
```

**Key Fields Description**:
- `file_path`: Target image path (relative or absolute)
- `text`: Instruction or description of the target image
- `source_file_path`: Reference image path. Read only under `--train_mode=edit`; a list is accepted when one request conditions on several images
- `width` / `height`: Image dimensions (**recommended** to provide for bucket training; if not provided, they will be automatically read during training, which may slow down training when data is stored on slow systems like OSS)
  - You can use `scripts/process_json_add_width_and_height.py` to add width and height fields to JSON files without these fields
  - Usage: `python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Images-Demo/metadata.json --output_file datasets/X-Fun-Images-Demo/metadata_add_width_height.json`

A reference image is encoded at its own resolution: the trainer rescales it so that its short edge is 2048, rounding both sides to a multiple of 32, and never crops it. Large references therefore cost sequence length, not canvas size, so keep the edit meta to a reasonable number of references per sample.

The `text` field is used as given and truncated to the release's 4096 prompt tokens, with a warning when that cuts it. The cap is on the caption alone: a reference's vision block sits ahead of it in the sequence and never costs the caption its length.

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
export DATASET_META_NAME="/absolute/path/to/metadata_add_width_height.json"
```

> 💡 **Recommendation**: If the dataset is small and stored locally, use relative paths. If the dataset is stored on external storage (e.g., NAS, OSS) or shared across multiple machines, use absolute paths.

---

## 3. Full Parameter Training

### 3.1 Download Pretrained Model

Training and inference read the released native `transformer/` by default, converting its tensors while loading and initializing the shared `proj_out` from `final_layer.video_out`; `transformer_diffusers/` is not read by default and only serves as a fallback when the native folder is absent, in which case the original single projection must be supplied as `<model_root>/original_output_head.safetensors`. The image VAE uses `vae_diffusers/`; the Qwen3-VL conditioner is the one MiniMax-H3 ships, so a video snapshot may be symlinked component by component.

```bash
# Download the official MiniMax-H3-Image weights
modelscope download --model MiniMax/MiniMax-H3-Image --local_dir models/Diffusion_Transformer/MiniMax-H3-Image
```

### 3.2 Quick Start (FSDP)

The 33B backbone does not fit on one GPU next to its frozen conditioner, so the recommended launcher shards it with FSDP. Each transformer block (`MiniMaxH3TransformerBlock`) becomes one sharding unit, which is also what lets the LoRA modules of `train_lora.py` shard with the layer they belong to.

```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3-Image"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3_image/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --train_batch_size=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_image" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --train_mode="t2i" \
  --trainable_modules "."
```

A copy of the launcher is provided as `scripts/minimax_h3_image/train.sh`.

**Batch size is fixed at 1**: the packed layout (text tokens, reference rows, generated rows) is per sample, so a larger batch would need a sample loop inside the pack. Use `--gradient_accumulation_steps` to widen an effective batch.

### 3.3 Common Training Parameters

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|-----|------|-------|
| `--pretrained_model_name_or_path` | Path to the MiniMax-H3-Image snapshot | `models/Diffusion_Transformer/MiniMax-H3-Image` |
| `--train_data_dir` | Training data directory | `datasets/X-Fun-Images-Demo/` |
| `--train_data_meta` | Training data metadata file | `datasets/X-Fun-Images-Demo/metadata_add_width_height.json` |
| `--train_mode` | `t2i` reads `file_path`; `edit` additionally reads `source_file_path` | `t2i` |
| `--image_sample_size` | Side of the training canvas, a multiple of 32; buckets scale both sides from it | 1024 |
| `--fix_sample_size H W` | Force one canvas instead of bucketing, rounded to multiples of 32 | - |
| `--min_generation_area` | Area floor of the validation samples only; defaults to `image_sample_size ** 2` | 4194304 |
| `--train_batch_size` | Kept at 1, see the note above | 1 |
| `--gradient_accumulation_steps` | Gradient accumulation steps (equivalent to larger batch) | 1 |
| `--sigma_shift` | Exponential shift applied to the sampled sigma. 1.0 matches the released image grid; a larger value moves training mass to the noisy end | 1.0 |
| `--uniform_sampling` | Uniform sigma draw instead of the logit-normal weighting | - |
| `--weighting_scheme` | Loss weighting: `sigma_sqrt`, `logit_normal`, `mode`, `cosmap`, `none` | `none` |
| `--enable_bucket` | Group samples by aspect ratio so nothing is cropped or padded | - |
| `--random_hw_adapt` | additionally rescale each batch to a random fraction of the canvas, an augmentation | - |
| `--low_vram` | Keep the VAE and the conditioner on CPU, move them per encode | - |
| `--gradient_checkpointing_save_on_cpu` | Store recomputed activations in host RAM; slower, saves the activation footprint | - |
| `--checkpointing_steps` | Save checkpoint every N steps | 500 |
| `--resume_from_checkpoint` | Resume training from checkpoint path, use `"latest"` to auto-select latest | None |
| `--trainable_modules` | Trainable modules (`"."` means all modules) | `"."` |
| `--use_ema` | Track an exponential moving average and use it for validation | - |
| `--validation_prompts` / `--validation_steps` / `--validation_epochs` | Periodic sample generation, see **3.4** | - |

Training minimizes the same rectified-flow objective the sampler integrates: the model input is `x_t = t * x0 + (1 - t) * noise` with `t = 1 - sigma` (`t = 1` is the clean image, the opposite of the Wan scripts), and the regression target is `x0 - noise`. Conditioning rows are noise-augmented and never supervised. All timesteps share the original single `proj_out`.

### 3.4 Training Validation

Validation is optional and not enabled by the default launchers. Append the validation parameters to periodically generate test images during training, allowing you to monitor training progress and model quality.

```bash
accelerate launch --mixed_precision="bf16" --use_fsdp ... scripts/minimax_h3_image/train.py \
  # ... (other training parameters)
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="a red cube on a white table, studio lighting." "a portrait of an astronaut"
```

**Notes**:
- Validation uses `MiniMaxH3ImagePipeline` with the single projection, defaulting to 40 steps and shift 12. Each prompt also requires one Qwen3-VL conditioner pass; the progress bar pauses while validation runs.
- Validation generates square images at `--image_sample_size` by default; to match portrait training data append `--fix_sample_size 1152 896` (height width), which pins the training canvas and the validation samples to the same non-square shape. Both training scripts also accept `--validation_num_inference_steps` and `--validation_flow_shift` to override the validation schedule.
- Validation images are written to `output_dir/sample/sample-{step}-rank{rank}-image-{i}.png`, and the first training batch to `output_dir/sanity_check/` so a glance confirms the data pipe.
- Under `--train_mode=edit`, pair each prompt with a reference through `--validation_image_paths`, one path per prompt in the same order.

### 3.5 Image Editing Training

Editing is the same trainer on a meta that carries `source_file_path`, at a larger canvas: the references enter the sequence as conditioning rows, and 10% of the samples drop their text so the unconditional branch the guidance distillation relies on stays alive.

```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3-Image"
export DATASET_NAME="datasets/X-Fun-Images-Edit-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Edit-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3_image/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --train_batch_size=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_image" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --train_mode="edit" \
  --trainable_modules "."

# Optional validation (3.4): pair each --validation_prompts entry with a reference
# --validation_image_paths="datasets/X-Fun-Images-Edit-Demo/source/souce-20260421_140343_099985_000/10_00015405.png"
```

The sequence a sample builds is `text tokens + vision tokens + reference latent rows + generated rows`, so the edit mode costs roughly one extra 4096-token block per reference. Keep `--train_batch_size=1` and raise `--gradient_accumulation_steps` if you widen the canvas.

### 3.6 Other Backends

#### 3.6.1 Training with DeepSpeed-Zero-2

DeepSpeed-Zero-2 keeps a full copy of the backbone on every GPU and shards only the optimizer state and the gradients, so it needs the VRAM of a resident 33B model per rank. Use it when the number of GPUs is small and the canvas is modest:

```bash
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/minimax_h3_image/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  # ... (the parameters of 3.2, without the FSDP flags)
```

#### 3.6.2 Training with DeepSpeed-Zero-3

Zero-3 shards the weights too. Checkpoints then come out as sharded state, so the run switches to `--save_state` on its own and the folder is gathered back with:

```bash
python scripts/zero_to_bf16.py output_dir/checkpoint-{your-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

Pass `--use_deepspeed --deepspeed_config_file config/zero_stage3_config.json` to `accelerate launch`; the trainer reads the stage from the config file and adapts the save hooks to it.

### 3.7 Multi-Machine Distributed Training

Same code, same data paths (NFS / shared storage) on every machine, one `accelerate` config per machine that differs only in the rank:

```bash
# Machine 0
accelerate launch --config_file multi_node_yaml/zero_stage2.yaml \
    --num_machines=2 --machine_rank=0 --main_process_ip=192.168.1.2 --main_process_port=12345 \
    scripts/minimax_h3_image/train.py ...

# Machine 1: the same command with --machine_rank=1
```

- **Firewall**: Make sure ports between machines are open, or you'll get connection errors
- **NCCL Environment**: With no RDMA across the machines, uncomment `NCCL_IB_DISABLE=1` and `NCCL_P2P_DISABLE=1`
- **Data Synchronization**: All machines must be able to access the same data paths (NFS/shared storage)

---

## 4. Inference Testing

### 4.1 Inference Parameter Parsing

The inference entries are `examples/minimax_h3_image/predict_t2i.py` and `examples/minimax_h3_image/predict_i2i_edit.py`, configured by editing the variables at their top.

**Key Parameter Descriptions**:

| Parameter | Description | Example Value |
|------|------|-------|
| `GPU_memory_mode` | VRAM management mode, see the table below | `model_group_offload` |
| `ulysses_degree` / `ring_degree` | Head / sequence parallelism degree, both 1 on a single GPU | 1 |
| `fsdp_dit` | Shard the transformer across GPUs | `True` |
| `fsdp_text_encoder` | Shard the Qwen3-VL conditioner too (~62 GB replicated otherwise) | `True` |
| `compile_dit` | Compile the transformer for a speedup at fixed resolution | `False` |
| `model_name` | Model path | `models/Diffusion_Transformer/MiniMax-H3-Image` |
| `transformer_path` | Trained transformer folder or file | `None` |
| `vae_path` | Trained VAE weights | `None` |
| `lora_path` | LoRA weights of a `train_lora.py` run | `None` |
| `lora_weight` | LoRA strength | 0.55 |
| `sample_size` | `[height, width]` of the returned image | `[1024, 1024]` |
| `min_generation_area` | Area floor: a smaller request is generated on a scaled-up canvas and resized back | `2048 * 2048` |
| `weight_dtype` | Use `torch.float16` on GPUs without bf16 (v100, 2080 Ti) | `torch.bfloat16` |
| `prompt` | Instruction or description | `"A red fox trotting..."` |
| `images` | Reference images, `predict_i2i_edit.py` only | `["/path/to/source.png"]` |
| `seed` | Random seed for reproducible results | 43 |
| `save_path` | Where to write the generated image | `samples/minimax-h3-image-t2i` |

The default single projection accepts `num_inference_steps` and `flow_shift` (40 and 12 by default). Guidance is distilled into the released backbone, so `guidance_scale` and `negative_prompt` are not supported. The native transformer already contains its single projection; when only the PDD diffusers checkpoint is available, place `original_output_head.safetensors` in the model root (see **3.1**).

**PDD (six-step inference and fine-tuning).** Parallel Decoding Distillation (PDD, [arXiv 2607.26004](https://arxiv.org/abs/2607.26004)) turns the flow model into a parallel decoder; the released Image checkpoint ships six PDD output heads on a fixed six-step schedule. Select them at inference with `output_head_type = "pdd"`, or `MiniMaxH3ImagePipeline.from_pretrained(model_root, output_head_type="pdd")`. In native-transformer mode the heads are read from `pdd_fused_heads.safetensors` and `pdd_config.json` at the model root; the converted checkpoint reads `pdd_proj_out`. PDD uses the fixed six-step grid and does not accept a custom `flow_shift`. Both Image training scripts also accept `--output_head_type=pdd` for ordinary flow-matching fine-tuning; LoRA training updates adapters while keeping the heads frozen. Neither script runs the repository's PDD teacher distillation. `train_lora.py` additionally accepts `--pdd_discrete_timestep` to train at the six released nodes.

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

Run from the repository root.

```bash
python examples/minimax_h3_image/predict_t2i.py
python examples/minimax_h3_image/predict_i2i_edit.py
```

Edit the variables at the top according to your needs. For first-time inference, focus on these.

```python
# Choose based on GPU VRAM
GPU_memory_mode = "model_group_offload"
# Model path
model_name = "models/Diffusion_Transformer/MiniMax-H3-Image"
# Full fine-tune weights, e.g. "output_dir_minimax_h3_image/checkpoint-500/transformer"
transformer_path = None
# LoRA weights, e.g. "output_dir_minimax_h3_image_lora/checkpoint-500.safetensors"
lora_path = None
# Write based on generation content
prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"
```

#### LoRA and Output Head

Both scripts default to `output_head_type = "single"` and use the repository's `merge_lora` / `unmerge_lora` functions when `lora_path` is set. Both the kohya and PEFT linear-layer formats exported by this repository are supported. Merging into BF16 base weights can round small LoRA updates, so inference can differ from validation that evaluates the adapter separately. Leave `lora_path = None` for no LoRA; a full-parameter transformer folder goes to `transformer_path`. Editing references are listed in `images` in `predict_i2i_edit.py`. Output defaults to 1024×1024 with a minimum generation area of 2048×2048 (`sample_size`, `min_generation_area`). A LoRA without head metadata only triggers a warning at inference; verify its provenance before using it. Changing the projection does not alter the rest of the backbone, and the 40-step single head is not guaranteed to converge or to beat the official six-step PDD.

### 4.3 Multi-GPU Parallel Inference

**Suitable for**: High-resolution generation, accelerated inference

#### Install Parallel Inference Dependencies

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### Configure Parallel Strategy

Edit `examples/minimax_h3_image/predict_t2i.py`:

```python
# Ensure ulysses_degree × ring_degree = number of GPUs
# For example, using 2 GPUs:
ulysses_degree = 2  # Head dimension parallelization
ring_degree = 1     # Sequence dimension parallelization
```

**Configuration Principles**:
- `ulysses_degree` must evenly divide the model's 56 attention heads
- `ring_degree` splits on sequence dimension, affecting communication overhead; avoid using it when heads can be divided

#### Run Multi-GPU Inference

```bash
torchrun --nproc-per-node=2 examples/minimax_h3_image/predict_t2i.py
```

## 5. Additional Resources

- **MiniMax-H3-Image LoRA training**: `scripts/minimax_h3_image/README_TRAIN_LORA.md`
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
- **Model Card**: https://modelscope.cn/models/MiniMax/MiniMax-H3-Image
