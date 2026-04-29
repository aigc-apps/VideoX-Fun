# Wan2.2 Fun LoRA 微调训练指南

本文档提供 Wan2.2 Fun（视频修复）LoRA 微调训练的完整流程，包括环境配置、数据准备、多种分布式训练策略和推理测试。

> **说明**：Wan2.2 Fun 是一个基于 Wan2.2 架构的视频修复模型，支持视频修复任务。Wan2.2采用双Transformer架构（高噪声/低噪声模型），5B版本使用单Transformer架构。本指南涵盖 Wan2.2 Fun 的 LoRA 微调训练流程，支持 A14B 和 5B 两种模型变体。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径使用方案](#24-相对路径与绝对路径使用方案)
- [三、LoRA 训练](#三lora-训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始（DeepSpeed-Zero-2）](#32-快速开始deepspeed-zero-2)
  - [3.3 LoRA 专用参数解析](#33-lora-专用参数解析)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 使用 FSDP 训练](#35-使用-fsdp-训练)
  - [3.6 其他后端](#36-其他后端)
  - [3.7 多机分布式训练](#37-多机分布式训练)
- [四、推理测试](#四推理测试)
  - [4.1 推理参数解析](#41-推理参数解析)
  - [4.2 视频修复推理](#42-视频修复推理)
  - [4.3 多卡并行推理](#43-多卡并行推理)
- [五、更多资源](#五更多资源)

---

## 一、环境配置

**方式 1：使用requirements.txt**

```bash
pip install -r requirements.txt
```

**方式 2：手动安装依赖**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl deepspeed==0.17.0 numpy==1.26.4
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

**方式 3：使用docker**

使用docker的情况下，请保证机器中已经正确安装显卡驱动与CUDA环境，然后以此执行以下命令：

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

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
- `file_path`：视频路径（相对或绝对路径）
- `text`：视频描述（英文提示词）
- `type`：数据类型，固定为 `"video"`
- `width` / `height`：视频宽高（**最好提供**，用于分桶训练，如果不提供则自动在训练时读取，当数据存储在如oss这样的速度较慢的系统上时，可能会影响训练速度）。
  - 可以使用`scripts/process_json_add_width_and_height.py`文件对无width与height字段的json进行提取，支持处理图片与视频。
  - 使用方案为`python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Demo/metadata.json --output_file datasets/X-Fun-Videos-Demo/metadata_add_width_height.json`。

### 2.4 相对路径与绝对路径使用方案

**相对路径**：

如果数据的路径为相对路径，则在训练脚本中设置：

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

**绝对路径**：

如果数据的路径为绝对路径，则在训练脚本中设置：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **建议**：如果数据集较小且存储在本地，推荐使用相对路径；如果数据集存储在外部存储（如 NAS、OSS）或多个机器共享存储，推荐使用绝对路径。

---

## 三、LoRA 训练

### 3.1 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Wan2.2 Fun 官方权重
# A14B 模型（双Transformer架构）
modelscope download --model PAI/Wan2.2-Fun-A14B-InP --local_dir models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP
# 或 5B 模型（单Transformer架构）
# modelscope download --model PAI/Wan2.2-Fun-5B-InP --local_dir models/Diffusion_Transformer/Wan2.2-Fun-5B-InP
```

### 3.2 快速开始（DeepSpeed-Zero-2）

如果按照 **2.1 快速测试数据集下载数据** 与 **3.1 下载预训练模型下载权重**后，直接复制快速开始的启动指令进行启动。

推荐使用 DeepSpeed-Zero-2 与 FSDP 方案进行训练。这里以 DeepSpeed-Zero-2 为例配置 shell 文件。

**DeepSpeed-Zero-2 与 FSDP 对比**：
- 两者都是分布式训练策略，帮助降低GPU显存占用
- DeepSpeed-Zero-2：优化器状态和梯度在GPU间分片
- FSDP（完全分片数据并行）：模型权重、优化器状态和梯度全部分片
- **如果使用多卡且 DeepSpeed-Zero-2 显存不足**，切换使用 FSDP 以获得更好的显存效率

> **关于 train_lora.sh**：本目录中的 `train_lora.sh` 脚本提供了一个**不带 DeepSpeed 或 FSDP** 的基础训练模板。适用于：
> - 单GPU训练
> - 快速测试和调试
> - 根据特定需求进行自定义修改
> 
> 对于多GPU的生产环境训练，**请使用下方的 DeepSpeed-Zero-2 或 FSDP 命令**以获得更好的性能和显存效率。

**Wan2.2 Fun LoRA 训练示例（DeepSpeed-Zero-2）**：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.2_fun/train_lora.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
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
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_fun_lora" \
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
  --boundary_type="low" \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --train_mode="inpaint" \
  --low_vram
```

> **说明**：本目录中的 `train_lora.sh` 脚本提供了不带 DeepSpeed 的基础训练模板。对于更好的多GPU训练性能和显存效率，请使用上方的 DeepSpeed-Zero-2 命令。

### 3.3 LoRA 专用参数解析

**Wan2.2 双Transformer架构说明**：

Wan2.2采用了创新的双Transformer架构：
- **低噪声模型（Low Noise Model）**：负责处理低噪声阶段（接近最终输出）
- **高噪声模型（High Noise Model）**：负责处理高噪声阶段（初始生成阶段）
- **边界类型（boundary_type）**：
  - `low`：训练低噪声模型，高噪声模型使用预训练权重（推荐用于T2V/I2V LoRA微调）
  - `high`：训练高噪声模型，低噪声模型使用预训练权重
  - `full`：单模型训练（用于TI2V-5B等单Transformer模型）

**LoRA 关键参数说明**：

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--config_path` | 配置文件路径 | `config/wan2.2/wan_civitai_i2v.yaml` |
| `--pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP` |
| `--train_data_dir` | 训练数据目录 | `datasets/X-Fun-Videos-Demo/` |
| `--train_data_meta` | 训练数据元文件 | `datasets/X-Fun-Videos-Demo/metadata_add_width_height.json` |
| `--train_batch_size` | 每批次样本数 | 1 |
| `--image_sample_size` | 图像最大训练分辨率 | 640 |
| `--video_sample_size` | 视频最大训练分辨率 | 640 |
| `--token_sample_size` | Token 采样尺寸 | 640 |
| `--video_sample_stride` | 视频采样步幅 | 2 |
| `--video_sample_n_frames` | 视频采样帧数 | 81 |
| `--gradient_accumulation_steps` | 梯度累积步数（等效增大 batch） | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | 50 |
| `--learning_rate` | 初始学习率（LoRA 推荐值） | 1e-04 |
| `--lr_scheduler` | 学习率调度器：`linear`、`cosine`、`cosine_with_restarts`、`polynomial`、`constant`、`constant_with_warmup` | `constant` |
| `--lr_warmup_steps` | 学习率预热步数 | 500 |
| `--seed` | 随机种子（可复现训练） | 42 |
| `--output_dir` | 输出目录 | `output_dir_wan2.2_fun_lora` |
| `--gradient_checkpointing` | 激活重计算以节省显存 | - |
| `--mixed_precision` | 混合精度：`no`、`fp16`、`bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon 值 | 1e-10 |
| `--vae_mini_batch` | VAE 编码时的迷你批次大小 | 1 |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--enable_bucket` | 启用分桶训练，不裁剪图片/视频，按分辨率分组训练 | - |
| `--random_hw_adapt` | 自动缩放图片/视频到 `[min_size, max_size]` 范围内的随机尺寸 | - |
| `--training_with_video_token_length` | 根据 token 长度训练，支持任意分辨率 | - |
| `--uniform_sampling` | 均匀采样 timestep（推荐启用） | - |
| `--low_vram` | 低显存模式，提高显存效率 | - |
| `--boundary_type` | Wan2.2双Transformer边界类型：`low`（训练低噪声模型）、`high`（训练高噪声模型）、`full`（训练单模型如TI2V-5B） | `low` |
| `--train_mode` | 训练模式：`normal`（普通T2V）、`inpaint`（视频修复） | `inpaint` |
| `--resume_from_checkpoint` | 恢复训练路径，使用 `"latest"` 自动选择最新 checkpoint | None |
| `--rank` | LoRA 更新矩阵的维度（rank 越大表达能力越强，但显存占用越高） | 64 |
| `--network_alpha` | LoRA 更新矩阵的缩放系数（通常设置为 rank 的一半或相同） | 32 |
| `--target_name` | 应用 LoRA 的组件/模块，用逗号分隔（如 `q,k,v,ffn.0,ffn.2`） | `q,k,v,ffn.0,ffn.2` |
| `--lora_skip_name` | LoRA训练中跳过的组件，用逗号分隔 | None |
| `--use_peft_lora` | 使用 PEFT 模块添加 LoRA（更节省显存） | - |
| `--validation_steps` | 每 N 步执行一次验证 | 2000 |
| `--validation_epochs` | 每 N 个epoch执行一次验证 | 5 |
| `--validation_prompts` | 验证视频生成的提示词 | `"一只棕色的狗摇着头..."` |
| `--validation_paths` | 验证 I2V 的参考图像路径（仅 inpaint 模式） | `"asset/1.png"` |
| `--use_deepspeed` | 启用 DeepSpeed 分布式训练 | - |
| `--use_fsdp` | 启用 FSDP 分布式训练 | - |
| `--use_8bit_adam` | 使用 8-bit Adam 优化器节省显存 | - |
| `--use_came` | 使用 CAME 优化器 | - |
| `--multi_stream` | 使用 CUDA 多流提升性能 | - |
| `--snr_loss` | 使用 SNR 损失函数 | - |
| `--weighting_scheme` | Timestep 加权方案：`sigma_sqrt`、`logit_normal`、`mode`、`cosmap`、`none` | `none` |
| `--motion_sub_loss` | 启用运动子损失以提升时序一致性 | - |
| `--motion_sub_loss_ratio` | 运动子损失比例 | 0.25 |

**Sample Size 配置指南**：
- `video_sample_size` 表示视频的分辨率大小；当 `random_hw_adapt` 为 True 时，表示视频和图像分辨率的最小值。
- `image_sample_size` 表示图像的分辨率大小；当 `random_hw_adapt` 为 True 时，表示视频和图像分辨率的最大值。
- `token_sample_size` 表示当 `training_with_video_token_length` 为 True 时，最大 token 长度对应的分辨率。
- 由于配置可能产生混淆，**如果你不需要任意分辨率进行 finetuning**，建议将 `video_sample_size`、`image_sample_size` 和 `token_sample_size` 设置为相同的固定值，例如 **(320, 480, 512, 640, 960)**。
  - **全部设置为 320** 代表 **240P**。
  - **全部设置为 480** 代表 **320P**。
  - **全部设置为 640** 代表 **480P**。
  - **全部设置为 960** 代表 **720P**。

**Token Length 训练说明**：
- 当启用 `training_with_video_token_length` 时，模型根据 token 长度进行训练。
- 例如：512x512 分辨率、49 帧的视频，其 token 长度为 13,312，需要设置 `token_sample_size = 512`。
  - 在 512x512 分辨率下，视频帧数为 49 (~= 512 * 512 * 49 / 512 / 512)。
  - 在 768x768 分辨率下，视频帧数为 21 (~= 512 * 512 * 49 / 768 / 768)。
  - 在 1024x1024 分辨率下，视频帧数为 9 (~= 512 * 512 * 49 / 1024 / 1024)。
  - 这些分辨率与对应帧数的组合，使模型能够生成不同尺寸的视频。

### 3.4 训练验证

你可以配置验证参数，在训练过程中定期生成测试视频，以便监控训练进度和模型质量。

**验证参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--validation_steps` | 每 N 步执行一次验证 | 2000 |
| `--validation_epochs` | 每 N 个epoch执行一次验证 | 5 |
| `--validation_prompts` | 验证视频生成的提示词 | None |
| `--validation_paths` | 验证 I2V 的参考图像路径（仅 inpaint 模式） | None |

**验证示例**（inpaint 模式）：

```bash
  --validation_paths "asset/1.png" \
  --validation_steps=100 \
  --validation_epochs=100 \
  --validation_prompts="A brown dog shaking its head, sitting on a light-colored sofa in a cozy room. Behind the dog, there's a framed painting on a shelf, surrounded by pink flowers. The soft, warm lighting in the room creates a comfortable atmosphere."
```

**注意事项**：
- 验证视频会保存到 `output_dir` 目录中
- 多提示词验证格式：`--validation_prompts "prompt1" "prompt2" "prompt3"`
- Wan2.2 Fun 的验证会根据 `boundary_type` 自动选择使用单Transformer或双Transformer
- 当 `train_mode="inpaint"` 时，验证使用修复模式，参考图像作为起始帧
- 对于 T2V 验证（不带参考图像），设置 `train_mode="normal"` 并省略 `--validation_paths`

### 3.5 使用 FSDP 训练

**如果使用多卡且使用DeepSpeed-Zero-2的情况下显存不足**，可以切换使用FSDP进行训练。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.2_fun/train_lora.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
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
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_fun_lora" \
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
  --boundary_type="low" \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --low_vram \
  --train_mode="inpaint"
```

> **说明**：在本仓库中，FSDP 比 DeepSpeed-Zero-3 更稳定且出错更少。当 DeepSpeed-Zero-2 在多GPU情况下遇到显存问题时，请使用 FSDP。

### 3.6 其他后端

#### 3.6.1 使用DeepSpeed-Zero-3进行训练

目前不太推荐使用 DeepSpeed Zero-3。在本仓库中，使用 FSDP 出错更少且更稳定。

DeepSpeed Zero-3 适合高分辨率的 14B Wan。训练后，您可以使用以下命令获取最终模型：
```bash
python scripts/zero_to_bf16.py output_dir/checkpoint-{our-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

训练 shell 命令如下：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/wan2.2_fun/train_lora.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
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
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_fun_lora" \
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
  --boundary_type="low" \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --low_vram \
  --train_mode="inpaint"
```

#### 3.6.2 不使用 DeepSpeed 与 FSDP 训练

**该方案并不被推荐，因为没有显存节约后端，容易造成显存不足**。这里仅提供训练Shell用于参考训练。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.2_fun/train_lora.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
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
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_fun_lora" \
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
  --boundary_type="low" \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --low_vram \
  --train_mode="inpaint"
```

> **说明**：这与 `train_lora.sh` 脚本类似，但使用了正确的数据集路径。`train_lora.sh` 脚本可以用作单GPU训练的起点。

### 3.7 多机分布式训练

**适合场景**：超大规模数据集、需要更快的训练速度

#### 3.7.1 环境配置

假设有 2 台机器，每台 8 张 GPU：

**机器 0（Master）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master 机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 机器总数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 当前机器 rank（0 或 1）
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.2_fun/train_lora.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
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
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_fun_lora" \
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
  --boundary_type="low" \
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --low_vram \
  --train_mode="inpaint"
```

**机器 1（Worker）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # 与 Master 相同
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # 注意这里是 1
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

# 使用与机器 0 相同的 accelerate launch 命令
```

#### 3.7.2 多机训练注意事项

- **网络要求**：
   - 推荐 RDMA/InfiniBand（高性能）
   - 无 RDMA 时添加环境变量：
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **数据同步**：所有机器必须能够访问相同的数据路径（NFS/共享存储）

---

## 四、推理测试

### 4.1 推理参数解析

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `GPU_memory_mode` | 显存管理模式，可选值见下表 | `model_group_offload` |
| `ulysses_degree` | Head 维度并行度，单卡时为 1 | 1 |
| `ring_degree` | Sequence 维度并行度，单卡时为 1 | 1 |
| `fsdp_dit` | 多卡推理时对 Transformer 使用 FSDP 节省显存 | `False` |
| `fsdp_text_encoder` | 多卡推理时对文本编码器使用 FSDP | `True` |
| `compile_dit` | 编译 Transformer 加速推理（固定分辨率下有效） | `False` |
| `model_name` | 模型路径 | `models/Diffusion_Transformer/Wan2.2-T2V-A14B` |
| `sampler_name` | 采样器类型：`Flow`、`Flow_Unipc`、`Flow_DPM++` | `Flow_Unipc` |
| `transformer_path` | 加载训练好的低噪声 Transformer 权重路径 | `None` |
| `transformer_high_path` | 加载训练好的高噪声 Transformer 权重路径（仅双Transformer模型） | `None` |
| `vae_path` | 加载训练好的 VAE 权重路径 | `None` |
| `lora_path` | 低噪声模型 LoRA 权重路径 | `None` |
| `lora_high_path` | 高噪声模型 LoRA 权重路径（仅双Transformer模型） | `None` |
| `sample_size` | 生成视频分辨率 `[高度, 宽度]` | `[480, 832]` 或 `[832, 480]` |
| `video_length` | 生成视频帧数 | `81` |
| `fps` | 每秒帧数 | `16` |
| `weight_dtype` | 模型权重精度，不支持 bf16 的显卡使用 `torch.float16` | `torch.bfloat16` |
| `validation_image_start` | 图生视频的参考图像路径（I2V 模式） | `"asset/1.png"` |
| `validation_image_end` | 结束帧的参考图像路径（可选） | `None` |
| `validation_mask` | 修复的掩码路径（inpaint 模式） | `None` |
| `prompt` | 正向提示词，描述生成内容 | `"一只棕色的狗摇着头..."` |
| `negative_prompt` | 负向提示词，避免生成的内容 | `"低分辨率，低画质..."` |
| `guidance_scale` | 引导强度 | 6.0 |
| `seed` | 随机种子，用于复现结果 | 43 |
| `num_inference_steps` | 推理步数 | 50 |
| `lora_weight` | 低噪声模型 LoRA 权重强度 | 0.55 |
| `lora_high_weight` | 高噪声模型 LoRA 权重强度（仅双Transformer模型） | 0.55 |
| `save_path` | 生成视频保存路径 | `samples/wan-videos-i2v` 或 `samples/wan-videos-t2v` |

**显存管理模式说明**：

| 模式 | 说明 | 显存占用 |
|------|------|---------|
| `model_full_load` | 整个模型加载到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 全量加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 使用后将模型卸载到 CPU | 中等 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `model_group_offload` | 层组在 CPU/CUDA 间切换 | 低 |
| `sequential_cpu_offload` | 逐层卸载（速度最慢） | 最低 |

### 4.2 视频修复推理

#### 4.2.1 推理脚本选择

Wan2.2 Fun 提供多种推理脚本，请根据您的模型版本和任务类型选择：

| 脚本 | 模型版本 | 架构 | 主要用途 |
|------|---------|------|---------|
| `predict_i2v.py` | A14B | 双Transformer | 图生视频/视频修复（I2V/Inpaint） |
| `predict_t2v.py` | A14B | 双Transformer | 文生视频（T2V） |
| `predict_i2v_5b.py` | 5B | 单Transformer | 图生视频/视频修复（I2V/Inpaint） |
| `predict_t2v_5b.py` | 5B | 单Transformer | 文生视频（T2V） |

> **说明**：
> - A14B 模型使用双Transformer架构（低噪声+高噪声模型），需要配置 `transformer_path` 和 `transformer_high_path`
> - 5B 模型使用单Transformer架构，只需配置 `transformer_path`，`transformer_high_path` 保持为 `None`

#### 4.2.2 A14B模型推理（双Transformer）

单卡推理运行如下命令：

```bash
python examples/wan2.2_fun/predict_i2v.py
```

根据需求修改 `examples/wan2.2_fun/predict_i2v.py`，初次推理重点关注如下参数，如果对其他参数感兴趣，请查看上方的推理参数解析。

```python
# 根据显卡显存选择
GPU_memory_mode = "sequential_cpu_offload"
# 根据实际模型路径
model_name = "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"  
# 训练好的低噪声权重路径
transformer_path = None  
# 训练好的高噪声权重路径
transformer_high_path = None  
# 训练好的权重路径，如 "output_dir_wan2.2_fun_lora/checkpoint-xxx/diffusion_pytorch_model.safetensors"
lora_path = None
lora_high_path = None
# 修复的起始图像
validation_image_start = "asset/1.png"
# 修复的掩码（可选，如果不提供将自动生成）
validation_mask = None
# 根据生成内容编写
prompt = "A brown dog shaking its head, sitting on a light-colored sofa in a cozy room. Behind the dog, there's a framed painting on a shelf, surrounded by pink flowers. The soft, warm lighting in the room creates a comfortable atmosphere."  
# ...
```

> **说明**：Wan2.2 Fun 主要用于视频修复任务。使用 `predict_i2v.py` 进行 I2V 和修复工作流。当提供或生成掩码时，模型将自动处理修复任务。

#### 4.2.3 5B模型推理（单Transformer）

单卡推理运行如下命令：

```bash
python examples/wan2.2_fun/predict_i2v_5b.py
```

根据需求修改 `examples/wan2.2_fun/predict_i2v_5b.py`，重点关注如下参数：

```python
# 根据显卡显存选择
GPU_memory_mode = "sequential_cpu_offload"
# 5B模型路径
model_name = "models/Diffusion_Transformer/Wan2.2-Fun-5B-InP/"  
# 训练好的权重路径（5B为单Transformer，只需设置transformer_path）
transformer_path = None  
# 5B模型不使用高噪声Transformer，保持为None
transformer_high_path = None  
# 训练好的权重路径，如 "output_dir_wan2.2_fun_lora/checkpoint-xxx/diffusion_pytorch_model.safetensors"
lora_path = None
# 5B模型不使用高噪声LoRA，保持为None
lora_high_path = None
# 修复的起始图像
validation_image_start = "asset/1.png"
validation_image_end = None
# 根据生成内容编写
prompt = "A brown dog licking its tongue, sitting on a light-colored sofa in a cozy room. Behind the dog, there's a framed painting on a shelf, surrounded by pink flowers. The soft, warm lighting in the room creates a comfortable atmosphere."  
# ...
```

> **说明**：
> - 5B 模型采用单Transformer架构，配置更简单，显存占用更低
> - 训练时如果使用 `boundary_type="full"`，推理时只需加载 `transformer_path`，无需设置 `transformer_high_path`
> - LoRA 训练时也只需设置 `lora_path`，`lora_high_path` 保持为 `None`

### 4.3 多卡并行推理

**适合场景**：高分辨率生成、加速推理

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

编辑 `examples/wan2.2_fun/predict_i2v.py`：

```python
# 确保 ulysses_degree × ring_degree = 使用的 GPU 数
# 例如使用 2 张 GPU：
ulysses_degree = 2  # Head 维度并行
ring_degree = 1     # Sequence 维度并行
```

**配置原则**：
- `ulysses_degree` 必须能整除模型的 head 数
- `ring_degree` 是在 sequence 维度切分，会影响通信开销，在 head 能整除的情况下尽量不要用

**配置示例**：

| GPU 数量 | ulysses_degree | ring_degree | 说明 |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | 单 GPU |
| 4 | 4 | 1 | Head 并行 |
| 8 | 8 | 1 | Head 并行 |
| 8 | 4 | 2 | 混合并行 |

#### 运行多卡推理

```bash
torchrun --nproc-per-node=2 examples/wan2.2_fun/predict_i2v.py
```

---

## 五、更多资源

- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
