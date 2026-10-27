# TAE Super（Tiny AutoEncoder）训练指南

本文档提供 Wan2.2 Super decoder（`AutoencoderTinyWan`）训练/微调的完整流程，
包括环境配置、数据准备、训练，以及将训练好的 checkpoint 用于推理。

> **说明**：训练使用冻结的官方 Wan2.2 encoder，只更新 Super decoder 约 19M 参数。
> Tiny encoder 冻结，不参与训练前向。
>
> ```text
> x (video, [-1, 1])
>   -> official Wan2.2 encoder (frozen) : z = vae.encode(x).latent_dist.mode()
>   -> Super decoder                  : x_hat = tae(z).sample
> loss = 0.1 L1 + 0.1 LPIPS + 0.02 DINO + 1.0 grid
> ```
>
> 损失在重建与输入 RGB 映射到 [0,1] 后计算，decoder 直接学习解码官方 Wan latent。

脚本支持 Wan2.2 5B Super 模型：

| 家族 | Latent | 官方完整 VAE | `--config_path` | 适用模型 |
|------|--------|--------------|-----------------|----------|
| taew2_2_super | 48ch, patch_size=2 | `AutoencoderKLWan3_8` | `config/wan2.2/wan_civitai_5b.yaml` | Wan2.2 TI2V-5B / Fun-2.2VAE |

**与 [train_taehv.py](train_taehv.py) 的差异**：

| 对比项 | `train_taehv.py` | `train_taehv_super.py` |
|--------|-----------------|-----------------------|
| 重建路径 | 小 encoder → 小 decoder，官方 encoder 提供 latent 监督 | 冻结官方 encoder → Super decoder |
| 更新参数 | 默认训练小 encoder 和 decoder，由 `--trainable_modules` 控制 | 仅 Super decoder；官方和小 encoder 均冻结 |
| 损失 | 像素 L1 + latent MSE | 像素 L1 + LPIPS + DINO + grid |

---

## 目录

- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径使用方案](#24-相对路径与绝对路径使用方案)
- [三、TAE Super 训练](#三tae-super-训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始](#32-快速开始)
  - [3.3 训练常用参数解析](#33-训练常用参数解析)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 训练技巧](#35-训练技巧)
  - [3.6 后续调优方向](#36-后续调优方向)
- [四、推理测试](#四推理测试)
  - [4.1 Checkpoint 目录结构](#41-checkpoint-目录结构)
  - [4.2 在 Predict 脚本中使用训练好的 TAE](#42-在-predict-脚本中使用训练好的-tae)
- [五、更多资源](#五更多资源)

---

## 一、环境配置

以下命令在 VideoX-Fun 仓库根目录执行，使用 pip 将依赖安装到当前 Python 环境。

**方式 1：使用 requirements.txt**

```bash
pip install -r requirements.txt
pip install lpips torchvision av modelscope opencv-python
```

**方式 2：手动安装依赖**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install lpips torchvision av modelscope opencv-python
```

LPIPS 的 AlexNet 主干和 DINOv2 在缺少缓存时自动下载，后续复用 PyTorch 缓存；
LPIPS 校准权重随安装包提供。离线加载 DINOv2 时，可使用 `--dino_repo` 和 `--dino_weights`。

> 脚本通过 Accelerate 支持单设备与普通 DDP。官方 Wan VAE 和冻结的 LPIPS/DINOv2
> 特征网络均保留在 GPU 上。

---

## 二、数据准备

### 2.1 快速测试数据集

我们提供了一个测试的数据集，其中包含若干训练数据。

```bash
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
- `width` / `height`：视频宽高（推荐保留）。Super 训练入口使用固定的缩放/裁剪画布。

### 2.4 相对路径与绝对路径使用方案

**相对路径**：

```bash
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
```

**绝对路径**：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **建议**：如果数据集较小且存储在本地，推荐使用相对路径；如果数据集存储在外部存储（如 NAS、OSS）或多个机器共享存储，推荐使用绝对路径。

---

## 三、TAE Super 训练

### 3.1 下载预训练模型

训练脚本需要随模型目录提供的官方 Wan2.2 VAE 权重。微调时同时下载发布版 Super 权重：

```bash
mkdir -p models/Diffusion_Transformer

modelscope download --model Wan-AI/Wan2.2-TI2V-5B --local_dir models/Diffusion_Transformer/Wan2.2-TI2V-5B

wget -O models/Diffusion_Transformer/taew2_2_super.pth \
  https://github.com/madebyollin/taehv/raw/main/taew2_2_super.pth
```

LPIPS 与 DINOv2 权重由损失实现单独加载。

### 3.2 快速开始

**Wan2.2 TI2V-5B / Fun-2.2VAE（taew2_2_super）训练示例**：

启动器采用 720x1280 画布、训练 25 帧、验证 81 帧、10,000 步和 EMA。
使用 `accelerate config` 配置 GPU/进程数；四卡、每卡 batch=1、累积=1 时有效 batch 为 4。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="${DATASET_NAME:-datasets/X-Fun-Videos-Demo}"
export DATASET_META_NAME="${DATASET_META_NAME:-$DATASET_NAME/metadata_add_width_height.json}"
export TAE_PATH="models/Diffusion_Transformer/taew2_2_super.pth"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/taehv/train_taehv_super.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --tae_path="$TAE_PATH" \
  --train_data_dir="$DATASET_NAME" \
  --train_data_meta="$DATASET_META_NAME" \
  --fix_sample_size 720 1280 \
  --video_sample_strides 1 2 3 \
  --video_sample_n_frames=25 \
  --validation_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --max_train_steps=10000 \
  --checkpointing_steps=1000 \
  --validation_steps=1000 \
  --validation_max_samples=0 \
  --learning_rate=5e-06 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=200 \
  --lr_min_ratio=0.2 \
  --l1_loss_weight=0.1 \
  --lpips_loss_weight=0.1 \
  --dino_loss_weight=0.02 \
  --grid_loss_weight=1.0 \
  --grid_period=16 \
  --feature_batch_size=13 \
  --use_ema \
  --ema_decay=0.999 \
  --selection_lpips_ratio=1.02 \
  --selection_grid_ratio=1.05 \
  --seed=42 \
  --output_dir="output_dir_taehv_w2.2_super" \
  --mixed_precision="bf16" \
  --adam_beta1=0.9 \
  --adam_beta2=0.99 \
  --adam_weight_decay=1e-2 \
  --adam_epsilon=1e-08 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0
```

相同参数已写入 `scripts/taehv/train_taehv_super.sh`，也可修改其中的训练参数后，
运行 `bash scripts/taehv/train_taehv_super.sh`。

**从头训练**：

```bash
accelerate launch --mixed_precision="bf16" scripts/taehv/train_taehv_super.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$DATASET_NAME" \
  --train_data_meta="$DATASET_META_NAME" \
  --from_scratch \
  --fix_sample_size 720 1280 \
  --video_sample_n_frames=25 \
  --video_sample_strides 1 2 3 \
  --max_train_steps=10000 \
  --learning_rate=5e-6 --lr_warmup_steps=200 --lr_min_ratio=0.2 \
  --use_ema --ema_decay=0.999 \
  --dataloader_num_workers=8 --checkpointing_steps=1000 --validation_steps=1000 \
  --output_dir="output_dir_taehv_w2.2_super_scratch"
```

使用 `--from_scratch` 时不要传 `--tae_path`。只训练 decoder，保存的 tiny encoder
仍是随机初始化权重。

**中断后恢复训练**：

```bash
accelerate launch --mixed_precision="bf16" scripts/taehv/train_taehv_super.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$DATASET_NAME" \
  --train_data_meta="$DATASET_META_NAME" \
  --fix_sample_size 720 1280 --video_sample_n_frames=25 \
  --video_sample_strides 1 2 3 --max_train_steps=10000 \
  --learning_rate=5e-6 --lr_warmup_steps=200 --lr_min_ratio=0.2 \
  --use_ema --ema_decay=0.999 \
  --dataloader_num_workers=8 --checkpointing_steps=1000 --validation_steps=1000 \
  --output_dir="output_dir_taehv_w2.2_super" \
  --resume_from_checkpoint=latest
```

恢复时保持创建 checkpoint 时的配置，使用过的验证索引也需继续提供。
`--max_train_steps` 表示总更新预算，应保留原 schedule 预算。Checkpoint 恢复
optimizer、scheduler、EMA、RNG、step 和 batch 游标；改变 workers 或采样设置可能改变
恢复后的样本顺序。新训练应使用新的输出目录。

### 3.3 训练常用参数解析

下表列出启动器的配置值，不代表所有参数的 Python CLI 默认值。

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--config_path` | 官方 Wan VAE 组件配置 | `config/wan2.2/wan_civitai_5b.yaml` |
| `--pretrained_model_name_or_path` | 包含官方 VAE 权重的模型目录 | `models/Diffusion_Transformer/Wan2.2-TI2V-5B` |
| `--tae_path` | Super 权重文件或 Diffusers 目录；不传则随机初始化 | `models/Diffusion_Transformer/taew2_2_super.pth` |
| `--from_scratch` | 随机初始化 Super decoder；不能与加载权重或恢复组合 | - |
| `--train_data_dir` / `--train_data_meta` | 视频根目录与 JSON 元数据数组 | `datasets/X-Fun-Videos-Demo` |
| `--fix_sample_size` | 固定 `[高, 宽]`，覆盖正方形 `--video_sample_size` | `720 1280` |
| `--video_sample_strides` | 每个训练片段随机选择一个采样 stride | `1 2 3` |
| `--video_sample_n_frames` | 训练片段帧数，必须为 4k+1 | 25 |
| `--train_batch_size` / `--gradient_accumulation_steps` | 每卡 batch / 梯度累积 | 1 / 1 |
| `--vae_mini_batch` | 官方 VAE 编码 mini-batch | 1 |
| `--dataloader_num_workers` | DataLoader workers | 8 |
| `--max_train_steps` | 总优化器更新次数，覆盖 `--num_train_epochs` | 10000 |
| `--checkpointing_steps` / `--checkpoints_total_limit` | 保存间隔 / checkpoint 数量上限 | 1000 / `None` |
| `--learning_rate` / `--lr_scheduler` | 峰值学习率 / 调度器 | `5e-6` / `cosine` |
| `--lr_warmup_steps` / `--lr_min_ratio` | 预热步数 / cosine 终点与峰值之比 | 200 / 0.2 |
| `--l1_loss_weight` / `--lpips_loss_weight` | 像素 L1 / LPIPS 损失权重 | 0.1 / 0.1 |
| `--dino_loss_weight` / `--grid_loss_weight` | DINO / 周期格纹损失权重 | 0.02 / 1.0 |
| `--feature_batch_size` / `--feature_checkpointing` | 感知网络帧 batch / 激活 checkpointing | 13 / 关闭 |
| `--use_ema` / `--ema_decay` | 维护用于验证和选模的 decoder EMA | 开启 / 0.999 |
| `--seed` / `--mixed_precision` | 随机种子 / 混合精度 | 42 / `bf16` |
| `--output_dir` | 输出目录 | `output_dir_taehv_w2.2_super` |
| `--resume_from_checkpoint` | 恢复 checkpoint 目录或选择最新断点 | `latest` |

**Sample Size 配置指南**：

- `--fix_sample_size 720 1280` 等比例缩放覆盖画布后中心裁剪；未指定时，`--video_sample_size` 控制正方形画布。
- 25 帧、stride=3 至少需要 73 个源视频帧；81 帧验证至少需要 81 帧。短视频会使 batch 按最短视频截为 4k+1 帧，实际长度记录在训练日志的 `frames` 字段。
- 训练随机选择时间起点，验证固定从第 0 帧开始、stride=1。解码失败时沿用原有随机换视频重试逻辑。

**损失配置指南**：

- 冻结官方 encoder 提供 latent，只优化 Super decoder。
- LPIPS 使用 AlexNet，DINOv2 使用 ViT-B/14 中间 patch 特征；特征网络冻结，但仍将重建分支的梯度传回 decoder。
- Grid loss 按空间相位统计残差均值，并去掉每帧每通道的 DC；应结合局部原生裁剪判断伪影。

### 3.4 训练验证

在训练命令中通过 `--validation_data_meta` 提供独立 JSON 数组，或将该参数加入启动器。
相对视频路径基于 `--train_data_dir` 解析；验证数据位于其他根目录时，可使用绝对视频路径。

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--validation_data_meta` | 独立验证集元数据数组 | 自己的验证索引 |
| `--validation_steps` | 每 N 次优化器更新执行验证 | 1000 |
| `--validation_n_frames` | 验证片段帧数，4k+1 | 81 |
| `--validation_max_samples` | 验证片段数量上限，0 表示全部 | 0 |

初始化、周期节点和最后一步均执行验证，报告 PSNR、SSIM、LPIPS 和 grid，并将首个
样本的输入/重建预览保存为 `sample/step-N.mp4`。指标使用视频编码前的浮点输出，
先对片段内帧平均，再对片段等权平均。

开启 EMA 时验证和最佳权重选择使用 EMA，否则使用 raw。选模在相对首次验证的
LPIPS/grid 比例约束下最大化 PSNR，由 `--selection_lpips_ratio`、`--selection_grid_ratio`
控制。初始权重也是候选。最佳推理权重写入 `best/taehv`，可恢复 checkpoint 保留 raw
训练权重。

可使用官方 Demo 进行短流程检查：

```bash
accelerate launch --mixed_precision="bf16" scripts/taehv/train_taehv_super.py \
  --pretrained_model_name_or_path="$MODEL_NAME" --tae_path="$TAE_PATH" \
  --train_data_dir="datasets/X-Fun-Videos-Demo" \
  --train_data_meta="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json" \
  --validation_data_meta="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json" \
  --fix_sample_size 64 96 --video_sample_n_frames=5 --validation_n_frames=5 \
  --video_sample_strides 1 --max_train_steps=2 --lr_warmup_steps=0 \
  --checkpointing_steps=1 --validation_steps=1 --validation_max_samples=1 \
  --dataloader_num_workers=0 --use_ema --ema_decay=0.999 \
  --output_dir="output_dir_taehv_w2.2_super_smoke"
```

短测中训练和验证共用 Demo 只用于检查执行；正式实验应按视频来源隔离，评估 720p
质量时应使用原生分辨率数据。

### 3.5 训练技巧

- **热启动与从头训练**：从 `taew2_2_super.pth` 或已有 Super 目录微调；使用 `--from_scratch` 训练新 decoder。初始化、数据和预算都会影响结果。
- **损失平衡**：先使用启动器中的四项损失配置，同时检查重建指标和原生分辨率裁剪。
- **显存**：按需降低训练帧数、画布尺寸或特征 batch，或开启 `--feature_checkpointing`。感知网络虽然冻结，预测分支仍需要激活显存。
- **EMA**：启动器默认开启 EMA。`best/taehv` 为选出的推理权重，`checkpoint-N/taehv` 为恢复训练使用的 raw 权重。
- **分布式训练**：使用 Accelerate 单设备或普通 DDP 配置。有效 batch 为设备数、每卡 batch 和梯度累积的乘积。

### 3.6 后续调优方向

- 尝试更长的时间上下文，同时检查视频起始段和后续帧，并计入额外计算开销。
- 探索像素 MSE 或官方 decoder 输出监督，平衡像素精度、感知质量和 teacher 计算成本。
- 通过受控对照调整 LPIPS/grid 权重，结合细纹理、文字和局部格纹检查平均指标。

以上为可探索的扩展方向，当前训练入口使用前述四项损失配置。

---

## 四、推理测试

### 4.1 Checkpoint 目录结构

可恢复 checkpoint 写入 `output_dir/checkpoint-{step}/`：

```text
output_dir_taehv_w2.2_super/
  args-step-N.yaml          # 实际运行参数记录
  train.jsonl              # 总损失、四项损失、LR 和实际帧数
  validation.jsonl         # 提供验证索引时保存指标与选模结果
  best/taehv/              # 提供验证索引时保存最佳推理权重
  checkpoint-1000/
    taehv/                 # raw Super 权重与 config.json
    optimizer.bin
    custom_checkpoint_*.pkl # scheduler 和可选 EMA 状态
    random_states_*.pkl
    trainer_state.json
  sample/                  # 验证视频
  logs/                    # TensorBoard 日志
```

`taehv` 目录使用 Diffusers `save_pretrained` 格式，可由
`AutoencoderTinyWan.from_pretrained` 加载。`args-step-N.yaml` 记录本次调用参数，
训练参数直接在 shell 启动器中配置。

### 4.2 在 Predict 脚本中使用训练好的 TAE

将兼容的 Wan2.2 5B TAE predict 脚本中的 `tae_path` 指向选出的 `best/taehv` 目录：

| 脚本 | 家族 |
|------|------|
| `examples/wan2.2/predict_ti2v_tae.py` | taew2_2 |
| `examples/wan2.2_fun/predict_t2v_2.2vae_tae.py` | taew2_2 |
| `examples/wan2.2_fun/predict_i2v_2.2vae_tae.py` | taew2_2 |

```python
# 例如 examples/wan2.2_fun/predict_t2v_2.2vae_tae.py 中
tae_path = "output_dir_taehv_w2.2_super/best/taehv"
```

训练好的 decoder 接收官方 Wan2.2 5B latent。从头训练的 checkpoint 没有训练好的
小 encoder；此类 checkpoint 用于图像/视频条件生成时，应保留官方 Wan encoder。
