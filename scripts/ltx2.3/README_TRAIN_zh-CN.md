# LTX-2.3 全参数训练指南

本文档提供 LTX-2.3（Lightricks 文生视频模型）全参数训练的完整工作流，包括环境配置、数据准备、分布式训练和推理测试。

> **注意**：LTX-2.3 是一个音视频生成视频模型，可以同时生成视频和对应的音频。训练数据需要同时包含视频和音频文件。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径使用](#24-相对路径与绝对路径使用)
- [三、全量参数训练](#三全量参数训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始（DeepSpeed-Zero-2）](#32-快速开始deepspeed-zero-2)
  - [3.3 常用训练参数](#33-常用训练参数)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 使用 FSDP 训练](#35-使用-fsdp-训练)
  - [3.6 不使用 DeepSpeed 或 FSDP 训练](#36-不使用-deepspeed-或-fsdp-训练)
  - [3.7 多机分布式训练](#37-多机分布式训练)
- [四、推理测试](#四推理测试)
  - [4.1 推理参数](#41-推理参数)
  - [4.2 单 GPU 推理](#42-单-gpu-推理)
  - [4.3 多 GPU 并行推理](#43-多-gpu-并行推理)
- [五、更多资源](#五更多资源)

---

## 一、环境配置

**方式一：使用 requirements.txt**

```bash
pip install -r requirements.txt
```

**方式二：手动安装依赖**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl deepspeed==0.17.0 numpy==1.26.4
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

**方式三：使用 Docker**

使用 Docker 时，请先确保本机已正确安装 GPU 驱动和 CUDA 环境，然后执行以下命令：

```
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# 进入镜像
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 二、数据准备

### 2.1 快速测试数据集

我们提供了一个包含数个音视频训练样本的测试数据集。

```bash
# 下载官方示例数据集
modelscope download --dataset PAI/X-Fun-Videos-Audios-Demo --local_dir ./datasets/X-Fun-Videos-Audios-Demo
```

### 2.2 数据集结构

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

### 2.3 metadata.json 格式

> ⚠️ **重要**：LTX-2.3 是音视频生成模型，与普通视频训练不同，**必须在 metadata.json 中提供 `audio_path` 字段**。

**相对路径格式**（示例）：
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

**绝对路径格式**：
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

**关键字段说明**：
- `file_path`：视频文件路径（相对或绝对）
- `audio_path`：音频文件路径（**LTX-2.3 特有且必须提供**，与普通视频训练的主要区别）
  - 音频文件通常为 `.wav` 格式
  - 路径应与 `file_path` 对应，如 `train/video001.mp4` 对应 `wav/audio001.wav`
- `text`：视频描述（英文提示词）
- `type`：数据类型，固定为 `"video"`
- `width` / `height`：视频尺寸（**建议**提供以启用 bucket 训练；若不提供，训练时会自动读取，但当数据存储在 OSS 等较慢的系统中时可能会拖慢训练速度）
  - 可使用 `scripts/process_json_add_width_and_height.py` 为没有宽高字段的 JSON 文件添加宽高字段，支持图片和视频
  - 使用方法：`python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Audios-Demo/metadata.json --output_file datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json`

**LTX-2.3 与普通视频训练的数据集对比**：

| 模型类型 | 必需字段 | 音频字段 |
|---------|---------|---------|
| 普通视频（WAN、CogVideoX 等） | `file_path`, `text`, `type` | ❌ 不需要 |
| **LTX-2.3（音视频生成）** | `file_path`, `audio_path`, `text`, `type` | ✅ **必须提供** |

### 2.4 相对路径与绝对路径使用

**相对路径**：

如果你的数据使用的是相对路径，训练脚本中请这样配置：

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

**绝对路径**：

如果你的数据使用的是绝对路径，训练脚本中请这样配置：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **建议**：如果数据集较小且存放在本地，请使用相对路径。如果数据集存放在外部存储（如 NAS、OSS）或多机共享，请使用绝对路径。

---

## 三、全参数训练

### 3.1 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 LTX-2.3 官方权重
hf download dg845/LTX-2.3-Diffusers --local-dir models/Diffusion_Transformer/LTX-2.3-Diffusers
```

### 3.2 快速开始（DeepSpeed-Zero-2）

如果你已按 **2.1 快速测试数据集** 下载了数据，按 **3.1 下载预训练模型** 下载了权重，你可以直接复制运行快速开始命令。

推荐使用 DeepSpeed-Zero-2 或 FSDP 进行训练。这里以 DeepSpeed-Zero-2 为例。

DeepSpeed-Zero-2 与 FSDP 的区别在于模型权重是否分片。**如果多卡使用 DeepSpeed-Zero-2 显存不够**，可切换为 FSDP。

```bash
export MODEL_NAME="models/Diffusion_Transformer/LTX-2.3-Diffusers"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/ltx2/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=121 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_ltx2.3" \
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
  --trainable_modules "."
```

### 3.3 常用训练参数

**核心参数说明**：

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/LTX-2.3-Diffusers` |
| `--train_data_dir` | 训练数据目录 | `datasets/internal_datasets/` |
| `--train_data_meta` | 训练数据元数据文件 | `datasets/internal_datasets/metadata.json` |
| `--train_batch_size` | 每批训练的样本数 | 1 |
| `--image_sample_size` | 最大训练分辨率，自动 bucket | 640 |
| `--video_sample_size` | 视频最大训练分辨率 | 640 |
| `--token_sample_size` | Token 长度采样大小 | 640 |
| `--video_sample_stride` | 帧采样步长 | 1 |
| `--video_sample_n_frames` | 采样帧数 | 81 |
| `--video_repeat` | 每个视频在每个 epoch 重复次数 | 1 |
| `--gradient_accumulation_steps` | 梯度累积步数（等效于更大 batch） | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练轮数 | 100 |
| `--checkpointing_steps` | 每 N 步保存检查点 | 50 |
| `--learning_rate` | 初始学习率 | 2e-05 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--seed` | 随机种子 | 42 |
| `--output_dir` | 输出目录 | `output_dir_ltx2.3` |
| `--gradient_checkpointing` | 启用激活检查点 | - |
| `--mixed_precision` | 混合精度：`fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon 值 | 1e-10 |
| `--vae_mini_batch` | VAE 编码的 mini-batch 大小 | 1 |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--random_hw_adapt` | 自动将视频缩放到 `[512, video_sample_size]` 范围内的随机大小 | - |
| `--training_with_video_token_length` | 按 token 长度训练，而非固定分辨率 | - |
| `--enable_bucket` | 启用 bucket 训练：不进行中心裁剪，而是按分辨率分组训练完整视频 | - |
| `--uniform_sampling` | 均匀时间步采样 | - |
| `--low_vram` | 启用低显存优化 | - |
| `--resume_from_checkpoint` | 从检查点路径恢复训练，使用 `"latest"` 自动选择最新检查点 | None |
| `--trainable_modules` | 可训练模块（`"."` 表示所有模块） | `"."` |
| `--validation_steps` | 每 N 步执行一次验证 | 100 |
| `--validation_epochs` | 每 N 个epoch执行一次验证 | 500 |
| `--validation_prompts` | 验证视频生成的提示词 | `"A man in a blue blazer..."` |


### 3.4 训练验证

你可以配置验证参数，在训练过程中定期生成测试视频，以便监控训练进度和模型质量。

**验证参数说明**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--validation_steps` | 每 N 步执行一次验证 | 100 |
| `--validation_epochs` | 每 N 个epoch执行一次验证 | 500 |
| `--validation_prompts` | 验证视频生成的提示词，可用空格分隔多个提示词 | 多个空格分隔的提示词 |

**示例**：

```bash
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks. At one point, he says, \"Hi.\""
```

**注意事项**：
- 验证视频会保存到 `output_dir` 目录中
- 多提示词验证格式：`--validation_prompts "prompt1" "prompt2" "prompt3"`

### 3.5 使用 FSDP 训练

**如果多卡使用 DeepSpeed-Zero-2 显存不够**，可切换为 FSDP。

```sh
export MODEL_NAME="models/Diffusion_Transformer/LTX-2.3-Diffusers"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap LTX2VideoTransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/ltx2/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=121 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_ltx2.3" \
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
  --trainable_modules "."
```

### 3.6 不使用 DeepSpeed 或 FSDP 训练

**不推荐此方式，因为缺少显存优化后端，容易显存溢出**。仅供参考。

```sh
export MODEL_NAME="models/Diffusion_Transformer/LTX-2.3-Diffusers"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/ltx2/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=121 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_ltx2.3" \
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
  --trainable_modules "."
```

### 3.7 多机分布式训练

**适用场景**：超大规模数据集，更快训练速度

#### 3.7.1 环境配置

假设 2 台机器，每台 8 张 GPU：

**机器 0（Master）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/LTX-2.3-Diffusers"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master 机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 总机器数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 当前机器 rank（0 或 1）
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/ltx2/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=121 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_ltx2.3" \
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
  --trainable_modules "."
```

**机器 1（Worker）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/LTX-2.3-Diffusers"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # 与 Master 相同
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # 注意这里是 1
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# 使用与机器 0 相同的 accelerate launch 命令
```

#### 3.7.2 多机训练注意事项

- **网络要求**：
   - 推荐使用 RDMA/InfiniBand（高性能）
   - 无 RDMA 时，需要添加环境变量：
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **数据同步**：所有机器必须能够访问相同的数据路径（NFS/共享存储）

## 四、推理测试

### 4.1 推理参数

**核心参数说明**：

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `GPU_memory_mode` | GPU 显存模式，见下表选项 | `model_group_offload` |
| `ulysses_degree` | Head 维度并行度，单 GPU 为 1 | 1 |
| `ring_degree` | Sequence 维度并行度，单 GPU 为 1 | 1 |
| `fsdp_dit` | 多 GPU 推理时对 Transformer 使用 FSDP 节省显存 | `False` |
| `fsdp_text_encoder` | 多 GPU 推理时对文本编码器使用 FSDP | `False` |
| `compile_dit` | 编译 Transformer 加速推理（固定分辨率有效） | `False` |
| `model_name` | 模型路径 | `models/Diffusion_Transformer/LTX-2.3-Diffusers` |
| `sampler_name` | 采样器类型：`Flow`、`Flow_Unipc`、`Flow_DPM++` | `Flow` |
| `transformer_path` | 训练后的 Transformer 权重路径 | `None` |
| `vae_path` | 训练后的 VAE 权重路径 | `None` |
| `lora_path` | LoRA 权重路径 | `None` |
| `sample_size` | 生成视频分辨率 `[height, width]` | `[512, 768]` |
| `video_length` | 生成帧数 | 121 |
| `fps` | 每秒帧数 | 24 |
| `weight_dtype` | 模型权重精度，无 bf16 的显卡使用 `torch.float16` | `torch.bfloat16` |
| `prompt` | 正向提示词，描述要生成的内容 | `"A brown dog barks..."` |
| `negative_prompt` | 负向提示词，描述要避免的内容 | `"worst quality, inconsistent motion..."` |
| `guidance_scale` | 引导强度 | 3.0 |
| `audio_guidance_scale` | 音频引导强度 | 7.0 |
| `stg_scale` | 视频时空引导（STG）强度 | 1.0 |
| `audio_stg_scale` | 音频 STG 强度 | 1.0 |
| `modality_scale` | 视频模态隔离引导强度 | 3.0 |
| `audio_modality_scale` | 音频模态隔离引导强度 | 3.0 |
| `guidance_rescale` | 视频引导重缩放因子 | 0.7 |
| `audio_guidance_rescale` | 音频引导重缩放因子 | 0.7 |
| `spatio_temporal_guidance_blocks` | STG 块索引 | [28] |
| `seed` | 随机种子，保证可重复性 | 43 |
| `num_inference_steps` | 推理步数 | 50 |
| `lora_weight` | LoRA 权重强度 | 0.55 |
| `save_path` | 生成视频保存路径 | `samples/ltx2-videos-t2v` |
| `audio_sample_rate` | 音频采样率（从 vocoder 配置读取） | 24000 |

**GPU 显存模式说明**：

| 模式 | 说明 | 显存占用 |
|------|------|---------|
| `model_full_load` | 将整个模型加载到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 全量加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 使用后模型卸载到 CPU | 中 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `model_group_offload` | 层组在 CPU/CUDA 间传输 | 低 |
| `sequential_cpu_offload` | 每层使用后卸载到 CPU（最慢） | 最低 |

### 4.2 单 GPU 推理

运行单卡推理：

```bash
python examples/ltx2.3/predict_t2v.py
```

根据需求编辑 `examples/ltx2.3/predict_t2v.py`。首次推理请重点修改以下参数，其他参数见上方推理参数说明。

```python
# 根据显卡显存选择
GPU_memory_mode = "model_group_offload"
# 你的实际模型路径
model_name = "models/Diffusion_Transformer/LTX-2.3-Diffusers"  
# 训练后的权重路径，如 "output_dir_ltx2.3/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = None  
# 根据要生成的内容编写
prompt = "A brown dog barks on a sofa, sitting on a light-colored couch in a cozy room"  
# ...
```

### 4.3 多 GPU 并行推理

**适用场景**：高分辨率生成、加速推理

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

编辑 `examples/ltx2.3/predict_t2v.py`：

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
| 8 | 2 | 4 | 混合并行 |
| 8 | 8 | 1 | Head 并行 |

#### 运行多 GPU 推理

```bash
torchrun --nproc-per-node=2 examples/ltx2.3/predict_t2v.py
```

## 五、更多资源

- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
