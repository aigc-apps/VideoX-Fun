# Wan Latent Upsampler 训练指南

本文档提供 Wan Latent Upsampler（`WanLatentUpsamplerModel`）针对 Wan2.2 VAE 进行训练的完整流程，包括环境配置、数据准备、训练，以及将训练好的 checkpoint 用于推理。

> **说明**：Wan Latent Upsampler 是一个轻量的 3D 卷积模型，在 VAE 解码前对 Wan2.2 VAE latent 进行**空间上采样**（默认 `1.5x`，由 `rational_spatial_scale` 控制）。它让扩散模型在较低的 latent 分辨率下生成，再将 latent 上采样到更高分辨率，从而降低高分辨率视频生成的计算开销。与扩散训练不同，upsampler 采用**在成对低/高分辨率 latent 上的纯监督 MSE 回归**进行训练：
>
> ```
> x_hr (高分辨率视频, [-1, 1])
>   x_lr   = spatial_downsample(x_hr, scale)   # scale = rational_spatial_scale (1.5)
>   z_hr   = vae.encode(x_hr).mode()           # 冻结 VAE -> 目标 latent
>   z_lr   = vae.encode(x_lr).mode()           # 冻结 VAE -> 输入 latent
>   z_pred = upsampler(z_lr)
> loss = MSE(z_pred, z_hr)                     # latent 空间 MSE（默认）
> ```
>
> VAE 是**冻结**的；只训练 upsampler。低/高分辨率数据对由普通视频即时构造，因此不需要预先配对的数据。可选开关可加入退化、flow-matching 噪声或像素空间 loss（见 [3.5](#35-高级训练选项)）。

upsampler 面向 Wan2.2 2.2VAE latent 空间（48 通道，`16x` 空间压缩）。`--config_path` 必须包含 `latent_upsampler_kwargs` 块，它决定了 upsampler 架构与空间 scale：

| Latent | 完整 VAE | `--config_path` | 空间 scale | 推理使用方 |
|--------|----------|-----------------|------------|------------|
| 48ch | `AutoencoderKLWan3_8`（Wan2.2_VAE.pth） | `config/wan2.2/wan_civitai_t2v_2.2vae.yaml` | 1.5 | `Wan2.2-Fun-*-A14B-2.2VAE` predict 脚本 |

> `config/wan2.2/wan_civitai_i2v_2.2vae.yaml` 拥有相同的 `latent_upsampler_kwargs`，训练时可互换使用。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径使用方案](#24-相对路径与绝对路径使用方案)
- [三、Latent Upsampler 训练](#三latent-upsampler-训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始](#32-快速开始)
  - [3.3 训练常用参数解析](#33-训练常用参数解析)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 高级训练选项](#35-高级训练选项)
  - [3.6 训练技巧](#36-训练技巧)
  - [3.7 多机分布式训练](#37-多机分布式训练)
- [四、推理测试](#四推理测试)
  - [4.1 Checkpoint 目录结构](#41-checkpoint-目录结构)
  - [4.2 在 Predict 脚本中使用训练好的 Upsampler](#42-在-predict-脚本中使用训练好的-upsampler)
- [五、更多资源](#五更多资源)

---

## 一、环境配置

**方式 1：使用 requirements.txt**

```bash
pip install -r requirements.txt
```

**方式 2：手动安装依赖**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
```

> upsampler 是一个轻量的卷积模型（远小于 diffusion transformer），因此**普通数据并行即可** —— 不需要 DeepSpeed / FSDP（但脚本仍支持）。内存中唯一较大的模型是冻结的 Wan2.2 VAE；如果它与训练激活一起放不下，使用 `--low_vram` 让它在 encode/decode 步骤之间保持在 CPU 上。

---

## 二、数据准备

### 2.1 快速测试数据集

我们提供了一个测试的数据集，其中包含若干训练数据。

```bash
# 下载官方示例数据集
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

**相对路径格式**（示例格式）：
```json
[
  {
    "file_path": "train/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "type": "video",
    "width": 1024,
    "height": 1024
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
- `file_path`：视频路径（相对或绝对路径）
- `text`：视频描述（upsampler loss 不使用，仅为兼容 meta 格式保留）
- `type`：数据类型，固定为 `"video"`
- `width` / `height`：视频宽高（**最好提供**，用于分桶训练）。
  - 可以使用 `scripts/process_json_add_width_and_height.py` 文件对无 width 与 height 字段的 json 进行提取。

### 2.4 相对路径与绝对路径使用方案

**相对路径**：

```bash
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
```

**绝对路径**：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **建议**：如果数据集较小且存储在本地，推荐使用相对路径；如果数据集存储在外部存储（如 NAS、OSS）或多个机器共享存储，推荐使用绝对路径。

---

## 三、Latent Upsampler 训练

### 3.1 下载预训练模型

训练脚本只需要 **Wan2.2 VAE 权重**（作为冻结的 encoder/decoder），它随模型目录一起提供：

```bash
mkdir -p models/Diffusion_Transformer

# Wan2.2 TI2V-5B（48ch latent，包含 Wan2.2_VAE.pth）—— upsampler 训练仅使用其中冻结的 VAE
modelscope download --model Wan-AI/Wan2.2-TI2V-5B --local_dir models/Diffusion_Transformer/Wan2.2-TI2V-5B
```

> 推理时 upsampler 与 `Wan2.2-Fun-*-A14B-2.2VAE` 模型配合使用。它们共享**相同的 Wan2.2 VAE latent 空间**，因此针对 `Wan2.2-TI2V-5B` 的 VAE 训练的 upsampler 可直接用于这些模型。任何包含 `Wan2.2_VAE.pth` 的模型目录都可作为 `--pretrained_model_name_or_path`。

### 3.2 快速开始

按照 **2.1** 下载数据、**3.1** 下载预训练模型后，直接复制快速开始的启动指令进行启动。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.2_fun/train_upsampler.py \
  --config_path="config/wan2.2/wan_civitai_t2v_2.2vae.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=640 \
  --video_sample_stride=2 \
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
  --output_dir="output_dir_wan2.2_fun_upsampler" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --enable_bucket \
  --low_vram \
  --trainable_modules "."
```

> 这些超参数遵循标准的 Wan2.2 Fun 视频训练设置。`scripts/wan2.2_fun/train_upsampler.sh` 包含相同的命令。

### 3.3 训练常用参数解析

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--config_path` | 模型配置 yaml；必须包含 `latent_upsampler_kwargs`（决定 upsampler 架构与空间 scale） | `config/wan2.2/wan_civitai_t2v_2.2vae.yaml` |
| `--pretrained_model_name_or_path` | 包含 `Wan2.2_VAE.pth` 的模型目录（仅使用其中冻结的 VAE） | `models/Diffusion_Transformer/Wan2.2-TI2V-5B` |
| `--latent_upsampler_path` | 可选，用于热启动/继续训练的 upsampler 权重（文件或目录） | `None` |
| `--vae_path` | 可选，其他完整 VAE 权重路径 | `None` |
| `--train_data_dir` | 训练数据目录 | `datasets/X-Fun-Videos-Demo/` |
| `--train_data_meta` | 训练数据元文件 | `datasets/X-Fun-Videos-Demo/metadata.json` |
| `--train_batch_size` | 每批次样本数（单卡） | 1 |
| `--video_sample_size` | 目标**高分辨率**尺寸；低分辨率输入为 `high_res / rational_spatial_scale` | 640 |
| `--video_sample_stride` | 视频采样步幅 | 2 |
| `--video_sample_n_frames` | 采样帧数，**必须为 4k+1**（33、49、81……） | 81 |
| `--vae_mini_batch` | VAE 编码时的迷你批次大小 | 1 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | 50 |
| `--checkpoints_total_limit` | 最多保存的 checkpoint 数量 | `None` |
| `--learning_rate` | 初始学习率 | 2e-05 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--use_8bit_adam` / `--use_came` | 备选优化器 | - |
| `--use_ema` | 保留 upsampler 的 EMA 副本（用于验证与最终保存） | - |
| `--seed` | 随机种子 | 42 |
| `--output_dir` | 输出目录 | `output_dir_wan2.2_fun_upsampler` |
| `--gradient_checkpointing` | 对 upsampler 启用激活重计算（启用 `--enable_pixel_loss` 时也作用于 VAE decoder） | - |
| `--mixed_precision` | 混合精度：`fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon 值 | 1e-10 |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--enable_bucket` | 启用分桶训练，不裁剪图片/视频，按分辨率分组训练 | - |
| `--random_hw_adapt` | 自动缩放视频到一定范围内的随机尺寸 | - |
| `--low_vram` | 让冻结的 VAE 保持在 CPU，仅在 encode/decode 时移到 GPU | - |
| `--trainable_modules` | 可训练模块（`.` 表示所有模块） | `"."` |
| `--trainable_modules_low_learning_rate` | 使用 lr/2 的可训练模块 | `[]` |
| `--resume_from_checkpoint` | 恢复训练路径，使用 `"latest"` 自动选择最新 checkpoint | `None` |
| `--validation_steps` / `--validation_epochs` | 每 N 步 / 每 N 个 epoch 执行一次验证 | 2000 / 5 |
| `--validation_paths` | 验证视频路径（encode 低分辨率 -> 上采样 -> decode） | `"asset/inpaint_video.mp4"` |

**Sample Size 配置指南**：
- `video_sample_size` 是**高分辨率目标**。启用 `random_hw_adapt` 时，它表示最小分辨率，视频可能被放大到更大的分桶尺寸。
- 低分辨率输入会自动推导：`low_res = high_res / rational_spatial_scale`，并对齐到 VAE 空间压缩比（`16`），因此两种分辨率都能被 VAE 编码。
- `video_sample_n_frames` 必须满足 `4k+1`（如 33、49、81），因为 Wan2.2 VAE 是因果 `4x` 时间压缩器。

### 3.4 训练验证

你可以配置验证参数，在训练过程中定期对测试视频执行 **encode 低分辨率 -> 上采样 -> decode**，以便直观监控超分辨率质量。

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--validation_steps` | 每 N 步执行一次验证 | 2000 |
| `--validation_epochs` | 每 N 个 epoch 执行一次验证 | 5 |
| `--validation_paths` | 验证视频路径 | `"asset/inpaint_video.mp4"` |

```bash
  --validation_paths "asset/inpaint_video.mp4" \
  --validation_steps=2000 \
  --validation_epochs=5
```

**注意事项**：
- 每个验证视频会被缩放到由 `--video_sample_size` 推导的正方形目标尺寸，然后依次编码、下采样为低分辨率输入、经模型上采样并解码。
- 验证会为每个样本保存两个视频到 `output_dir/sample/`：`step{N}_val{i}_upsampled.mp4`（upsampler 结果）与 `step{N}_val{i}_lowres.mp4`（解码后的低分辨率输入，用于对比）。
- 启用 `--use_ema` 时，验证使用 EMA 权重运行。

### 3.5 高级训练选项

这些开关默认关闭；**3.2** 中的纯 latent-MSE 回归是推荐的起点。

**退化（修复 + 超分辨率）**：

在空间下采样**之前**对高分辨率帧施加随机退化，使 upsampler 在学习上采样的同时学习修复。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable_degradation` | 启用对高分辨率帧的随机退化 | 关闭 |
| `--degradation_ops_range MIN MAX` | 每个片段退化操作数量的范围 | `1 4` |

**加噪训练（对齐推理去噪步）**：

推理时 upsampler 可能作用于部分去噪的 latent。加噪训练为 latent 添加 flow-matching 噪声，使 upsampler 能在任意去噪步下工作。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable_noisy_training` | 训练时为 latent 添加 flow-matching 噪声 | 关闭 |
| `--noise_sample_mode` | `scheduler` 从真实调度中采样 sigma；`uniform` 在 `[sigma_min, sigma_max]` 内采样 | `scheduler` |
| `--noise_num_inference_steps` | 采样 sigma 时模拟的推理步数（scheduler 模式） | 50 |
| `--noise_step_range MIN MAX` | 限制采样到步数索引的子区间（scheduler 模式） | `None` |
| `--noise_sigma_max` / `--noise_sigma_min` | `uniform` 模式的 sigma 范围 | 0.25 / 0.0 |

**像素空间 loss**：

将预测的 latent 经过冻结的 VAE decoder 解码，在像素空间计算 MSE；梯度会穿过冻结的 decoder 反传到 upsampler。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable_pixel_loss` | 启用像素空间 MSE loss（替代 latent 空间 MSE） | 关闭 |
| `--pixel_loss_weight` | 像素空间 loss 的权重 | 1.0 |

> `--enable_pixel_loss` 会让 VAE 保持在 GPU 上（在 `--low_vram` 下也不卸载），并且需要 VAE decoder 的梯度，因此显存占用更高。可配合 `--gradient_checkpointing` 启用 VAE decoder 的激活重计算。

### 3.6 训练技巧

- **默认 loss**：针对高分辨率帧的冻结 VAE 编码结果的 latent 空间 MSE。VAE 永远不会被更新。
- **显存**：显存随 `video_sample_size` 与 `video_sample_n_frames` 增长，因为冻结的 VAE 会对全分辨率视频进行 encode/decode。如果显存不足，保留 `--low_vram` + `--gradient_checkpointing` 并减小 `--video_sample_size` 或 `--video_sample_n_frames`。
- **热启动**：`--latent_upsampler_path` 加载已有的 upsampler 权重。与当前架构形状不匹配的键（例如改变 `rational_spatial_scale` 后）会被自动跳过并重新初始化。
- **EMA**：`--use_ema` 保留用于验证与最终保存的 EMA 副本。DeepSpeed ZeRO-3 不支持该选项。
- **空间 scale**：由 config 中的 `latent_upsampler_kwargs.rational_spatial_scale` 决定；它**必须与推理 config 一致**，否则 upsampler 输出分辨率会出错。

### 3.7 多机分布式训练

**适合场景**：大规模数据集、需要更快的训练速度。

假设有 2 台机器，每台 8 张 GPU：

**机器 0（Master）**：
```bash
export MASTER_ADDR="192.168.1.100"  # Master 机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 机器总数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 当前机器 rank（0 或 1）
# 无 RDMA 时：
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK scripts/wan2.2_fun/train_upsampler.py \
  <与快速开始相同的训练参数>
```

**机器 1（Worker）**：使用相同的命令，并设置 `export RANK=1`。

**注意事项**：
- 无 RDMA 时添加 `NCCL_IB_DISABLE=1` 与 `NCCL_P2P_DISABLE=1`。
- 所有机器必须能够访问相同的数据/模型路径（NFS/共享存储）。

---

## 四、推理测试

### 4.1 Checkpoint 目录结构

每个 checkpoint 以 `output_dir/checkpoint-{step}/` 形式写出，包含：

```
📦 output_dir_wan2.2_fun_upsampler/
├── 📂 checkpoint-50/
│   ├── 📂 latent_upsampler/       # upsampler 权重 + config.json（save_pretrained 格式）
│   ├── 📂 latent_upsampler_ema/   # 仅当启用 --use_ema
│   └── 📄 sampler_pos_start.pkl
├── 📂 sample/                     # 验证视频
└── 📂 logs/                       # tensorboard
```

`latent_upsampler` 子目录是标准的 diffusers 目录 checkpoint，可被 `WanLatentUpsamplerModel.from_pretrained` 直接加载。

> 使用 DeepSpeed ZeRO-3 或 FSDP `FULL_SHARD` 时，upsampler 权重会被保存为 checkpoint 目录内的单个 `diffusion_pytorch_model.safetensors`，而不是 `latent_upsampler/` 子目录。

### 4.2 在 Predict 脚本中使用训练好的 Upsampler

Wan2.2 2.2VAE 的 predict 脚本支持 latent 上采样。设置 `enable_latent_upsample = True`，并将 `latent_upsampler_path` 指向你 checkpoint 的 `latent_upsampler` 子目录：

| 脚本 | 任务 |
|------|------|
| `examples/wan2.2_fun/predict_t2v_2.2vae.py` | 文生视频 |
| `examples/wan2.2_fun/predict_i2v_2.2vae.py` | 图生视频 |
| `examples/wan2.2_fun/predict_t2v_2.2vae_tae.py` | 文生视频 + TAE 快速解码 |
| `examples/wan2.2_fun/predict_i2v_2.2vae_tae.py` | 图生视频 + TAE 快速解码 |

```python
# 例如 examples/wan2.2_fun/predict_t2v_2.2vae_tae.py 中
enable_latent_upsample = True
latent_upsampler_path  = "output_dir_wan2.2_fun_upsampler/checkpoint-50/latent_upsampler"
```

如果 `latent_upsampler_path` 为 `None`，脚本会回退到 `model_name` 下的 `latent_upsampler` 子目录。推理时 pipeline 在基础分辨率下生成 latent，upsampler 在 VAE 解码前对其进行空间上采样（`1.5x`）。

---

## 五、更多资源

- **Wan2.2 Fun 全参数训练指南**：[README_TRAIN_zh-CN.md](README_TRAIN_zh-CN.md)
- **TAE（Tiny AutoEncoder）训练指南**：[../taehv/README_TRAIN_zh-CN.md](../taehv/README_TRAIN_zh-CN.md)
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
