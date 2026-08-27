# TAE（Tiny AutoEncoder）训练指南

本文档提供 TAE（Tiny AutoEncoder，`AutoencoderTinyWan`）针对完整 Wan VAE 进行蒸馏训练/微调的完整流程，包括环境配置、数据准备、训练，以及将训练好的 checkpoint 用于推理。

> **说明**：TAE（[madebyollin/taehv](https://github.com/madebyollin/taehv)）是一个约 20MB 的蒸馏 VAE，与完整尺寸的 Wan VAE **共享完全相同的 latent 空间**。它的解码速度比完整 VAE 快约 100 倍，重建质量略有下降，主要用于扩散采样过程中的快速预览/低显存解码。训练采用简单的重建蒸馏范式：
>
> ```
> x (video, [-1, 1])
>   -> teacher (完整 VAE, 冻结)   : z_full = teacher.encode(x).mode()
>   -> TAE encoder               : z_tae  = tae.encode(x).mode()
>   -> TAE decoder               : x_hat  = tae.decode(z_tae).sample
> loss = pixel L1(x_hat, x) + latent_loss_weight * MSE(z_tae, z_full)
> ```
>
> latent MSE 将 TAE 的 latent 锚定到原生 VAE 的 latent 空间，这正是 TAE latent 能与扩散模型 latent 互换的关键。

支持两个 TAE 家族（通过 `--config_path` 选择，它决定了 teacher VAE）：

| 家族 | Latent | Teacher 完整 VAE | `--config_path` | 适用模型 |
|------|--------|------------------|-----------------|----------|
| taew2_1 | 16ch, patch_size=1 | `AutoencoderKLWan`（Wan2.1_VAE.pth） | `config/wan2.1/wan_civitai.yaml` | Wan2.1、Wan2.2 14B |
| taew2_2 | 48ch, patch_size=2 | `AutoencoderKLWan3_8`（Wan2.2_VAE.pth） | `config/wan2.2/wan_civitai_5b.yaml` | Wan2.2 TI2V-5B / Fun-2.2VAE |

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径使用方案](#24-相对路径与绝对路径使用方案)
- [三、TAE 训练](#三tae-训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始](#32-快速开始)
  - [3.3 训练常用参数解析](#33-训练常用参数解析)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 训练技巧](#35-训练技巧)
  - [3.6 多机分布式训练](#36-多机分布式训练)
- [四、推理测试](#四推理测试)
  - [4.1 Checkpoint 目录结构](#41-checkpoint-目录结构)
  - [4.2 在 Predict 脚本中使用训练好的 TAE](#42-在-predict-脚本中使用训练好的-tae)
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

> TAE 本身只有约 20MB 权重，因此**普通数据并行即可**——不需要 DeepSpeed / FSDP（但脚本仍然支持）。显存中唯一的大模型是冻结的 teacher 完整 VAE（2.2 VAE 约 1.5GB）；如果它与训练激活放不下，请使用 `--low_vram`。

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
- `text`：视频描述（TAE 损失不使用，仅为 meta 格式兼容而保留）
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

## 三、TAE 训练

### 3.1 下载预训练模型

训练脚本只需要**完整 VAE 权重**（用作冻结的 teacher），它们随模型目录一起提供：

```bash
mkdir -p models/Diffusion_Transformer

# taew2_2 家族（Wan2.2 TI2V-5B，48ch latent，包含 Wan2.2_VAE.pth）
modelscope download --model Wan-AI/Wan2.2-TI2V-5B --local_dir models/Diffusion_Transformer/Wan2.2-TI2V-5B

# taew2_1 家族（Wan2.1，16ch latent，包含 Wan2.1_VAE.pth）
# modelscope download --model Wan-AI/Wan2.1-T2V-14B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-14B
```

可选：下载官方发布的 TAE 权重进行热启动（warm-start），而不是从头训练：

```bash
# 来自 https://github.com/madebyollin/taehv
wget https://github.com/madebyollin/taehv/raw/main/taew2_2.safetensors
# wget https://github.com/madebyollin/taehv/raw/main/taew2_1.safetensors
```

### 3.2 快速开始

**Wan2.2 TI2V-5B / Fun-2.2VAE（taew2_2）训练示例**：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
# 可选：用官方发布的 TAE 权重热启动，而不是从头训练。
export TAE_PATH="taew2_2.safetensors"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/taehv/train_taehv.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --tae_path=$TAE_PATH \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=33 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --latent_loss_weight=1.0 \
  --seed=42 \
  --output_dir="output_dir_taehv_w2.2" \
  --mixed_precision="bf16" \
  --adam_weight_decay=1e-2 \
  --adam_epsilon=1e-08 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --enable_bucket \
  --trainable_modules "." \
  --resume_from_checkpoint=latest
```

**Wan2.1 / Wan2.2 14B（taew2_1）训练示例**：

与上面相同，仅做如下修改：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-14B"
export TAE_PATH="taew2_1.safetensors"
# ...
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --output_dir="output_dir_taehv_w2.1" \
# ...
```

### 3.3 训练常用参数解析

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--config_path` | 模型配置 yaml；其 `vae_kwargs.vae_type` 决定 teacher 完整 VAE 家族 | `config/wan2.2/wan_civitai_5b.yaml` |
| `--pretrained_model_name_or_path` | 包含完整 VAE 权重的模型目录 | `models/Diffusion_Transformer/Wan2.2-TI2V-5B` |
| `--tae_path` | 可选的 TAE 热启动权重（文件或目录）。不填则从头训练 | `taew2_2.safetensors` |
| `--tae_arch_variant` | 从头训练时的 TAE decoder 变体：base（`None`）或 `super`（decoder 参数量约 2 倍） | `None` |
| `--vae_path` | 可选的完整 VAE（teacher）热加载权重路径 | `None` |
| `--freeze_tae_encoder` | 只训练 TAE decoder（decoder-only 蒸馏） | - |
| `--use_taehv_sequential` | 以 O(1) 显存的串行模式运行 TAE，而不是并行模式 | - |
| `--latent_loss_weight` | latent MSE（TAE latent 对齐完整 VAE latent）相对 pixel L1 的权重 | 1.0 |
| `--train_data_dir` | 训练数据目录 | `datasets/X-Fun-Videos-Demo/` |
| `--train_data_meta` | 训练数据元文件 | `datasets/X-Fun-Videos-Demo/metadata.json` |
| `--train_batch_size` | 每卡批次大小 | 1 |
| `--video_sample_size` | 训练分辨率 | 512 |
| `--video_sample_stride` | 视频采样步幅 | 1 |
| `--video_sample_n_frames` | 采样帧数，**必须为 4k+1**（33、49、81……） | 33 |
| `--vae_mini_batch` | teacher VAE 编码时的迷你批次大小 | 1 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 4 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | 500 |
| `--checkpoints_total_limit` | 最多保留的 checkpoint 数量 | `None` |
| `--learning_rate` | 初始学习率 | 1e-4 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--use_8bit_adam` / `--use_came` | 可选优化器 | - |
| `--use_ema` | 维护 TAE 的 EMA 副本（用于验证与最终保存） | - |
| `--seed` | 随机种子 | 42 |
| `--output_dir` | 输出目录 | `output_dir_taehv_w2.2` |
| `--mixed_precision` | 混合精度：`fp16/bf16` | `bf16` |
| `--max_grad_norm` | 梯度裁剪阈值 | 1.0 |
| `--enable_bucket` | 启用分桶训练，不裁剪视频，按分辨率分组训练 | - |
| `--random_hw_adapt` | 自动缩放视频到一定范围内的随机尺寸 | - |
| `--low_vram` | 将 teacher VAE 放在 CPU，仅在编码时搬上 GPU | - |
| `--trainable_modules` | 可训练模块（`"."` 表示所有模块） | `"."` |
| `--trainable_modules_low_learning_rate` | 以 lr/2 训练的模块 | `[]` |
| `--resume_from_checkpoint` | 恢复训练路径，使用 `"latest"` 自动选择最新 checkpoint | `latest` |
| `--validation_steps` / `--validation_epochs` | 每 N 步 / 每 N 个 epoch 执行一次验证 | 2000 / 5 |
| `--validation_paths` | 用于重建对比验证的视频路径 | `"asset/1.mp4"` |

**Sample Size 配置指南**：
- `video_sample_size` 表示训练分辨率；当启用 `random_hw_adapt` 时，表示分辨率的最小值。
- `video_sample_n_frames` 必须满足 `4k+1`（如 33、49、81），因为完整 VAE 与 TAE 都是因果 4 倍时间压缩。

### 3.4 训练验证

你可以配置验证参数，在训练过程中定期用 TAE 和完整 VAE 分别重建测试视频，直观监控重建质量。

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--validation_steps` | 每 N 步执行一次验证 | 2000 |
| `--validation_epochs` | 每 N 个 epoch 执行一次验证 | 5 |
| `--validation_paths` | 验证视频路径 | `"asset/1.mp4"` |

```bash
  --validation_paths "asset/1.mp4" \
  --validation_steps=2000 \
  --validation_epochs=5
```

**注意事项**：
- 验证会为每个样本在 `output_dir/sample/` 保存三个视频：`*_input.mp4`（缩放后的输入）、`*_taehv.mp4`（TAE 重建）、`*_fullvae.mp4`（完整 VAE 重建，供参考对比）。
- 启用 `--use_ema` 时，验证使用 EMA 权重。

### 3.5 训练技巧

- **热启动 vs 从头训练**：官方发布的 TAE 权重已经蒸馏得很好；做领域适配时通常直接从 `taew2_x.safetensors` 热启动、用小学习率（1e-5）微调即可。从头训练可以收敛，但需要多得多的数据/步数。
- **损失平衡**：`latent_loss_weight=1.0` 用于保持 TAE latent 与扩散 latent 空间对齐。设为 `0.0` 则是纯像素重建（如果你会在采样流程中使用 TAE latent，不推荐）。
- **Decoder-only 蒸馏**：加上 `--freeze_tae_encoder` 只提升解码质量。
- **显存**：teacher 完整 VAE 是主要的显存消耗者。使用 `--low_vram` 可以将其放在 CPU、仅在编码时搬上 GPU；也可以降低 `--video_sample_n_frames` / `--video_sample_size`。
- **串行 TAE**：`--use_taehv_sequential` 以速度换显存，激活显存与视频长度无关（O(1)）。
- **EMA**：推荐开启 `--use_ema` 作为最终交付物；最终保存的 `taehv` 目录里是 EMA 权重。

### 3.6 多机分布式训练

**适合场景**：大规模数据集、需要更快的训练速度

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

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK scripts/taehv/train_taehv.py \
  <与快速开始相同的训练参数>
```

**机器 1（Worker）**：使用相同命令，但 `export RANK=1`。

**注意事项**：
- 无 RDMA 时添加 `NCCL_IB_DISABLE=1` 与 `NCCL_P2P_DISABLE=1`。
- 所有机器必须能访问相同的数据/模型路径（NFS/共享存储）。

---

## 四、推理测试

### 4.1 Checkpoint 目录结构

每个 checkpoint 写入 `output_dir/checkpoint-{step}/`，结构如下：

```
📦 output_dir_taehv_w2.2/
├── 📂 checkpoint-500/
│   ├── 📂 taehv/            # TAE 权重 + config.json（save_pretrained 格式）
│   ├── 📂 taehv_ema/        # 仅当开启 --use_ema
│   └── 📄 sampler_pos_start.pkl
├── 📂 sample/               # 验证视频
└── 📂 logs/                 # tensorboard
```

`taehv` 子目录是标准的 diffusers 目录 checkpoint，可以被 `AutoencoderTinyWan.from_pretrained` 直接加载。

### 4.2 在 Predict 脚本中使用训练好的 TAE

将任意 TAE predict 脚本中的 `tae_path` 指向你的 checkpoint 的 `taehv` 子目录即可：

| 脚本 | 家族 |
|------|------|
| `examples/wan2.2/predict_ti2v_tae.py` | taew2_2 |
| `examples/wan2.2_fun/predict_t2v_2.2vae_tae.py` | taew2_2 |
| `examples/wan2.2_fun/predict_i2v_2.2vae_tae.py` | taew2_2 |
| `examples/wan2.1/predict_t2v_tae.py` | taew2_1 |
| `examples/wan2.1/predict_i2v_tae.py` | taew2_1 |

```python
# 例如 examples/wan2.2_fun/predict_t2v_2.2vae_tae.py 中
tae_path = "output_dir_taehv_w2.2/checkpoint-500/taehv"
```

训练好的 TAE 与官方发布的 TAE 完全互换：它保持与完整 VAE 相同的 latent 空间，因此可以像 `taew2_2.safetensors` 一样在扩散 pipeline 中用于快速预览解码。

---

## 五、更多资源

- **TAE 参考实现**：https://github.com/madebyollin/taehv
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
