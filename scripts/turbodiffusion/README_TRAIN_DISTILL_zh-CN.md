# TurboDiffusion 蒸馏训练指南

本文档提供了将 Wan2.1 蒸馏为 TurboWan2.1 的完整工作流，包括环境配置、数据准备、分布式训练和推理测试。

> **注意**: TurboDiffusion 是一种知识蒸馏方法，可以将推理步数从 25-50 步减少到 4-8 步，同时保持视频生成质量。

---

## 目录
- [1. 环境配置](#1-环境配置)
- [2. 数据准备](#2-数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径使用](#24-相对路径与绝对路径使用)
- [3. 蒸馏训练](#3-蒸馏训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始（不使用 DeepSpeed/FSDP）](#32-快速开始不使用-deepspeedfsdp)
  - [3.3 常见训练参数](#33-常见训练参数)
  - [3.4 使用 DeepSpeed Zero-2 训练](#34-使用-deepspeed-zero-2-训练)
  - [3.5 使用 FSDP 训练](#35-使用-fsdp-训练)
  - [3.6 使用 DeepSpeed Zero-3 训练](#36-使用-deepspeed-zero-3-训练)
  - [3.7 多机分布式训练](#37-多机分布式训练)
- [4. 推理测试](#4-推理测试)
  - [4.1 推理参数](#41-推理参数)
  - [4.2 文本生成视频（T2V）推理](#42-文本生成视频t2v推理)
  - [4.3 图像生成视频（I2V）推理](#43-图像生成视频i2v推理)
  - [4.4 多 GPU 并行推理](#44-多-gpu-并行推理)
- [5. 其他资源](#5-其他资源)

---

## 1. 环境配置

**方法 1: 使用 requirements.txt**

```bash
pip install -r requirements.txt
```

**方法 2: 手动安装依赖**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl deepspeed==0.17.0 numpy==1.26.4
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

**方法 3: 使用 Docker**

使用 Docker 时，请确保您的机器已正确安装 GPU 驱动和 CUDA 环境，然后执行以下命令：

```
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# 进入容器
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. 数据准备

### 2.1 快速测试数据集

我们提供了一个包含多个训练样本的测试数据集。

```bash
# 下载官方演示数据集
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 2.2 数据集结构

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json 格式

**相对路径格式**（示例）：
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

**绝对路径格式**：
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

**关键字段说明**：
- `file_path`: 视频路径（相对路径或绝对路径）
- `text`: 视频描述（英文提示词）
- `type`: 数据类型，固定为 `"video"`
- `width` / `height`: 视频尺寸（**建议**提供以用于 bucket 训练。如果不提供，将在训练时自动读取，当数据存储在 OSS 等慢速系统时可能会降低训练速度）。
  - 您可以使用 `scripts/process_json_add_width_and_height.py` 为没有宽高字段的 JSON 文件提取宽高字段，支持图片和视频。
  - 用法：`python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Demo/metadata.json --output_file datasets/X-Fun-Videos-Demo/metadata_add_width_height.json`。

### 2.4 相对路径与绝对路径使用

**相对路径**：

如果您的数据使用相对路径，请在训练脚本中配置：

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

**绝对路径**：

如果您的数据使用绝对路径，请在训练脚本中配置：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **建议**：如果数据集较小且存储在本地，请使用相对路径。如果数据集存储在外部存储（如 NAS、OSS）或在多台机器间共享，请使用绝对路径。

---

## 3. 蒸馏训练

### 3.1 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Wan2.1 官方权重
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B
```

### 3.2 快速开始（不使用 DeepSpeed/FSDP）

按照 **2.1 快速测试数据集** 下载数据并按照 **3.1 下载预训练模型** 下载权重后，您可以直接复制并运行快速启动命令。

Wan 蒸馏不使用 DeepSpeed 和 FSDP 更适合 1.3B Wan，因为将其与 14B Wan 一起使用可能会导致 GPU 显存不足。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于没有 RDMA 的多节点
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/turbodiffusion/train_distill.py \
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
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_distill" \
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
  --train_mode="normal" \
  --trainable_modules "." \
  --low_vram
```

### 3.3 常见训练参数

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--config_path` | 配置文件路径 | `config/wan2.1/wan_civitai.yaml` |
| `--pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B/` |
| `--train_data_dir` | 训练数据目录 | `datasets/internal_datasets/` |
| `--train_data_meta` | 训练数据元数据文件 | `datasets/internal_datasets/metadata.json` |
| `--train_batch_size` | 每个 GPU 的批次大小 | 1 |
| `--image_sample_size` | 最大图像训练分辨率 | 640 |
| `--video_sample_size` | 最大视频训练分辨率 | 640 |
| `--token_sample_size` | Token 样本大小 | 640 |
| `--video_sample_stride` | 视频采样步长 | 2 |
| `--video_sample_n_frames` | 视频采样帧数 | 81 |
| `--gradient_accumulation_steps` | 梯度累积步数（有效增加批次） | 1 |
| `--dataloader_num_workers` | DataLoader 工作进程数 | 8 |
| `--num_train_epochs` | 训练轮数 | 100 |
| `--checkpointing_steps` | 每 N 步保存检查点 | 50 |
| `--learning_rate` | 初始学习率（生成器） | 2e-06 |
| `--learning_rate_critic` | 初始学习率（判别器） | 2e-07 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--seed` | 随机种子 | 42 |
| `--output_dir` | 输出目录 | `output_dir_distill` |
| `--gradient_checkpointing` | 启用梯度检查点 | - |
| `--mixed_precision` | 混合精度：`fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon | 1e-10 |
| `--vae_mini_batch` | VAE 编码小批次大小 | 1 |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--enable_bucket` | 启用 bucket 训练，不裁剪，按分辨率分组 | - |
| `--random_hw_adapt` | 自动将图像/视频缩放到 `[min_size, max_size]` 范围内的随机大小 | - |
| `--training_with_video_token_length` | 基于 token 长度训练，支持任意分辨率 | - |
| `--uniform_sampling` | 均匀时间步采样 | - |
| `--low_vram` | 低显存模式 | - |
| `--train_mode` | 训练模式：`normal`（标准）或 `i2v`（图像生成视频） | `normal` |
| `--resume_from_checkpoint` | 恢复训练路径，使用 `"latest"` 自动选择最新检查点 | None |
| `--validation_steps` | 每 N 步运行验证 | 2000 |
| `--validation_epochs` | 每 N 轮运行验证 | 5 |
| `--validation_prompts` | 用于视频生成验证的提示词 | `"A dog shaking head..."` |
| `--trainable_modules` | 可训练模块（`"."` 表示所有模块） | `"."` |

**蒸馏特有参数**：

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--denoising_step_indices_list` | 去噪步骤列表（蒸馏核心参数） | `1000 750 500 250` |
| `--real_guidance_scale` | 用于评分的真实 guidance scale | 6.0 |
| `--fake_guidance_scale` | 用于评分的虚拟 guidance scale | 0.0 |
| `--gen_update_interval` | 生成器更新间隔 | 5 |
| `--negative_prompt` | 用于蒸馏的负向提示词 | 中文负向提示词 |
| `--validation_paths` | I2V 模式的验证图像路径 | 图像路径列表 |
| `--train_sampling_steps` | 训练采样步数 | 1000 |

**样本大小配置指南**：
- `video_sample_size` 表示视频的分辨率大小；当 `random_hw_adapt` 为 True 时，表示视频和图像分辨率之间的最小值。
- `image_sample_size` 表示图像的分辨率大小；当 `random_hw_adapt` 为 True 时，表示视频和图像分辨率之间的最大值。
- `token_sample_size` 表示当 `training_with_video_token_length` 为 True 时，最大 token 长度对应的分辨率。
- 由于配置可能产生混淆，**如果您不需要任意分辨率进行微调**，建议将 `video_sample_size`、`image_sample_size` 和 `token_sample_size` 设置为相同的固定值，例如 **(320, 480, 512, 640, 960)**。
  - **全部设置为 320** 表示 **240P**。
  - **全部设置为 480** 表示 **320P**。
  - **全部设置为 640** 表示 **480P**。
  - **全部设置为 960** 表示 **720P**。

**Token 长度训练指南**：
- 当启用 `training_with_video_token_length` 时，模型基于 token 长度进行训练。
- 例如：分辨率为 512x512 且 49 帧的视频的 token 长度为 13,312，需要设置 `token_sample_size = 512`。
  - 在 512x512 分辨率下，视频帧数为 49 (~= 512 * 512 * 49 / 512 / 512)。
  - 在 768x768 分辨率下，视频帧数为 21 (~= 512 * 512 * 49 / 768 / 768)。
  - 在 1024x1024 分辨率下，视频帧数为 9 (~= 512 * 512 * 49 / 1024 / 1024)。
  - 这些分辨率与相应的帧数结合，使模型能够生成不同尺寸的视频。

**其他参数说明**：
- `enable_bucket` 用于启用 bucket 训练。启用后，模型不会在中心裁剪图像和视频，而是根据分辨率将它们分组到 bucket 中后对整个图像和视频进行训练。
- `random_frame_crop` 用于对视频帧进行随机裁剪，以模拟不同帧数的视频。
- `random_hw_adapt` 用于启用图像和视频的自动高度和宽度缩放。启用 `random_hw_adapt` 后，训练图像的高度和宽度将设置为 `image_sample_size` 作为最大值，`min(video_sample_size, 512)` 作为最小值。对于训练视频，高度和宽度将设置为 `image_sample_size` 作为最大值，`min(video_sample_size, 512)` 作为最小值。
  - 例如，启用 `random_hw_adapt` 后，设置 `video_sample_n_frames=49`、`video_sample_size=1024` 和 `image_sample_size=1024`，训练图像输入的分辨率为 `512x512` 到 `1024x1024`，训练视频输入的分辨率为 `512x512x49` 到 `1024x1024x49`。
  - 例如，启用 `random_hw_adapt` 后，设置 `video_sample_n_frames=49`、`video_sample_size=256` 和 `image_sample_size=1024`，训练图像输入的分辨率为 `256x256` 到 `1024x1024`，训练视频输入的分辨率为 `256x256x49`。
- `training_with_video_token_length` 指定根据 token 长度训练模型。对于训练图像和视频，高度和宽度将设置为 `image_sample_size` 作为最大值，`video_sample_size` 作为最小值。
  - 例如，启用 `training_with_video_token_length` 后，设置 `video_sample_n_frames=49`、`token_sample_size=1024`、`video_sample_size=256` 和 `image_sample_size=1024`，训练图像输入的分辨率为 `256x256` 到 `1024x1024`，训练视频输入的分辨率为 `256x256x49` 到 `1024x1024x49`。
  - 例如，启用 `training_with_video_token_length` 后，设置 `video_sample_n_frames=49`、`token_sample_size=512`、`video_sample_size=256` 和 `image_sample_size=1024`，训练图像输入的分辨率为 `256x256` 到 `1024x1024`，训练视频输入的分辨率为 `256x256x49` 到 `1024x1024x9`。
  - 分辨率为 512x512 且 49 帧的视频的 token 长度为 13,312。我们需要设置 `token_sample_size = 512`。
    - 在 512x512 分辨率下，视频帧数为 49 (~= 512 * 512 * 49 / 512 / 512)。
    - 在 768x768 分辨率下，视频帧数为 21 (~= 512 * 512 * 49 / 768 / 768)。
    - 在 1024x1024 分辨率下，视频帧数为 9 (~= 512 * 512 * 49 / 1024 / 1024)。
    - 这些分辨率与相应的长度结合，使模型能够生成不同尺寸的视频。
- `train_mode` 用于指定训练模式，可以是 normal 或 i2v。由于 Wan 使用 inpaint 模型来实现图像生成视频，因此默认设置为 inpaint 模式。如果您只想实现文本生成视频，可以删除此行，它将默认为文本生成视频模式。
- `resume_from_checkpoint` 用于设置是否应从先前的检查点恢复训练。使用路径或 `"latest"` 自动选择最后一个可用的检查点。

### 3.4 使用 DeepSpeed Zero-2 训练

使用 DeepSpeed Zero-2 的 Wan 适合训练 1.3B Wan 和低分辨率的 14B Wan，但在高分辨率下训练 14B Wan 仍可能导致 GPU 显存不足。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于没有 RDMA 的多节点
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/turbodiffusion/train_distill.py \
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
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_distill" \
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
  --train_mode="normal" \
  --trainable_modules "." \
  --low_vram
```

### 3.5 使用 FSDP 训练

使用 FSDP 的 Wan 适合高分辨率的 14B Wan。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于没有 RDMA 的多节点
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/turbodiffusion/train_distill.py \
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
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_distill" \
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
  --train_mode="normal" \
  --trainable_modules "." \
  --low_vram
```

### 3.6 其他后端

#### 3.6.1 使用DeepSpeed-Zero-3进行训练

目前不太推荐使用 DeepSpeed Zero-3。在本仓库中，使用 FSDP 出错更少且更稳定。

使用 DeepSpeed Zero-3 的 Wan 适合高分辨率的 14B Wan。训练后，您可以使用以下命令获取最终模型：
```bash
python scripts/zero_to_bf16.py output_dir/checkpoint-{our-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

训练 shell 命令如下：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于没有 RDMA 的多节点
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/turbodiffusion/train_distill.py \
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
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_distill" \
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
  --train_mode="normal" \
  --trainable_modules "." \
  --low_vram
```

#### 3.6.2 不使用 DeepSpeed 与 FSDP 训练

**该方案并不被推荐，因为没有显存节约后端，容易造成显存不足**。这里仅提供训练 Shell 用于参考训练。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于没有 RDMA 的多节点
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/turbodiffusion/train_distill.py \
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
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_distill" \
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
  --train_mode="normal" \
  --trainable_modules "." \
  --low_vram
```

### 3.7 多机分布式训练

**适用场景**：超大数据集、更快的训练速度

#### 3.7.1 环境配置

假设有 2 台机器，每台 8 个 GPU：

**机器 0（主节点）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
export MASTER_ADDR="192.168.1.100"  # 主节点 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 机器总数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 当前机器排名（0 或 1）
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于没有 RDMA 的多节点
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/turbodiffusion/train_distill.py \
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
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_distill" \
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
  --train_mode="normal" \
  --trainable_modules "." \
  --low_vram
```

**机器 1（工作节点）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
export MASTER_ADDR="192.168.1.100"  # 与主节点相同
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # 注意这是 1
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于没有 RDMA 的多节点
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# 使用与机器 0 相同的 accelerate launch 命令
```

#### 3.7.2 多机训练注意事项

- **网络要求**：
   - 建议使用 RDMA/InfiniBand（高性能）
   - 没有 RDMA 时，添加环境变量：
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **数据同步**：所有机器必须能够访问相同的数据路径（NFS/共享存储）

---

## 4. 推理测试

### 4.1 推理参数

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `GPU_memory_mode` | GPU 显存模式，见下表选项 | `sequential_cpu_offload` |
| `ulysses_degree` | 多 GPU 推理的 Ulysses 并行度 | 1 |
| `ring_degree` | 多 GPU 推理的 Ring 并行度 | 1 |
| `fsdp_dit` | 多 GPU 推理时对 Transformer 使用 FSDP 以节省显存 | `False` |
| `fsdp_text_encoder` | 多 GPU 推理时对文本编码器使用 FSDP | `True` |
| `compile_dit` | 编译 Transformer 以加快推理（固定分辨率时有效） | `False` |
| `model_name` | 模型路径 | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `sampler_name` | 采样器类型：`Flow`、`Flow_Unipc`、`Flow_DPM++` | `Flow` |
| `transformer_path` | 训练好的 Transformer 权重路径 | `models/Personalized_Model/TurboWan2.1-T2V-1.3B-480P.pth` |
| `vae_path` | 训练好的 VAE 权重路径 | `None` |
| `lora_path` | LoRA 权重路径 | `None` |
| `sample_size` | 生成视频分辨率 `[height, width]` | `[480, 832]` |
| `video_length` | 生成帧数 | `81` |
| `fps` | 每秒帧数 | `16` |
| `weight_dtype` | 模型权重精度，不支持 bf16 的 GPU 使用 `torch.float16` | `torch.bfloat16` |
| `validation_image_start` | I2V 模式的参考图像路径 | `"asset/1.png"` |
| `prompt` | 描述生成内容的正向提示词 | `"A stylish woman walks..."` |
| `negative_prompt` | 避免某些内容的负向提示词 | 中文负向提示词 |
| `guidance_scale` | 引导强度（蒸馏模型通常使用 1.0） | 1.0 |
| `seed` | 用于可重复性的随机种子 | 43 |
| `num_inference_steps` | 推理步数（蒸馏模型通常为 4） | 4 |
| `lora_weight` | LoRA 权重强度 | 0.55 |
| `save_path` | 保存生成视频的路径 | `samples/turbowan-videos-t2v` |

**GPU 显存模式说明**：

| 模式 | 说明 | 显存使用 |
|------|------|---------|
| `model_full_load` | 将整个模型加载到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 完整加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 使用后卸载模型到 CPU | 中等 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `sequential_cpu_offload` | 逐层顺序卸载（最慢） | 最低 |

### 4.2 文本生成视频（T2V）推理

运行单 GPU 推理：

```bash
python examples/turbodiffusion/predict_t2v_wan2.1.py
```

根据您的需要编辑 `examples/turbodiffusion/predict_t2v_wan2.1.py`。首次推理时，请关注以下关键参数。其他参数请参考上方的推理参数说明。

```python
# 根据 GPU 显存选择
GPU_memory_mode = "sequential_cpu_offload"
# 您的实际模型路径
model_name = "models/Diffusion_Transformer/Wan2.1-T2V-1.3B"  
# 训练好的权重路径，例如 "models/Personalized_Model/TurboWan2.1-T2V-1.3B-480P.pth"
transformer_path = "models/Personalized_Model/TurboWan2.1-T2V-1.3B-480P.pth"  
# 根据您的生成内容编写
prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage..."  
# 蒸馏模型通常使用 4 步
num_inference_steps = 4
# ...
```

### 4.3 图像生成视频（I2V）推理

运行单 GPU 推理：

```bash
python examples/turbodiffusion/predict_i2v_wan2.2.py
```

根据您的需要编辑 `examples/turbodiffusion/predict_i2v_wan2.2.py`。首次推理时，请关注以下关键参数。其他参数请参考上方的推理参数说明。

```python
# 根据 GPU 显存选择
GPU_memory_mode = "sequential_cpu_offload"
# 您的实际模型路径
model_name = "models/Diffusion_Transformer/Wan2.1-T2V-1.3B"  
# 训练好的权重路径
transformer_path = "models/Personalized_Model/TurboWan2.1-T2V-1.3B-480P.pth"  
# 参考图像路径
validation_image_start = "asset/1.png"
# 根据您的生成内容编写
prompt = "The dog is shaking head. The video is of high quality, and the view is very clear..."  
# 蒸馏模型通常使用 4 步
num_inference_steps = 4
# ...
```

### 4.4 多 GPU 并行推理

**适用场景**：高分辨率生成、加速推理

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

编辑 `examples/turbodiffusion/predict_t2v_wan2.1.py` 或 `examples/turbodiffusion/predict_i2v_wan2.2.py`：

```python
# 确保 ulysses_degree × ring_degree = GPU 数量
# 例如使用 2 张 GPU：
ulysses_degree = 2  # Head 维度并行
ring_degree = 1     # Sequence 维度并行
```

**配置原则**：
- `ulysses_degree` 必须能整除模型的头数
- `ring_degree` 在序列维度上拆分，会影响通信开销。当头可以均匀划分时，尽量避免使用它。

**配置示例**：

| GPU 数量 | ulysses_degree | ring_degree | 说明 |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | 单 GPU |
| 2 | 2 | 1 | 头并行 |
| 4 | 4 | 1 | 头并行 |
| 8 | 8 | 1 | 头并行 |
| 8 | 4 | 2 | 混合并行 |

#### 运行多 GPU 推理

```bash
torchrun --nproc-per-node=2 examples/turbodiffusion/predict_t2v_wan2.1.py
```

---

## 5. 其他资源

- **官方 GitHub**: https://github.com/aigc-apps/VideoX-Fun
- **TurboDiffusion 论文**: https://arxiv.org/abs/2411.19823
