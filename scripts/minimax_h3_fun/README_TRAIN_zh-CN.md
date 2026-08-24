# MiniMax-H3 控制模型训练指南

本文档提供 MiniMax-H3 控制模型（VACE 风格的可控视频生成）训练的完整工作流，包括环境配置、数据准备、多种分布式训练策略、CFG 蒸馏和推理测试。

> **注意**：MiniMax-H3 是一个音视频联合生成模型，可同时生成视频和对应音频。控制训练在基础 packed-sequence transformer 之上添加了一个零初始化的控制旁路分支（`MiniMaxH3ControlTransformer3DModel`）：配对的管控视频（pose / depth / canny 等）经过与目标视频相同的 patchify 处理后，通过零初始化的逐层 skip 注入主干，因此刚初始化的模型在数值上与基础 MiniMax-H3 模型完全一致，只需训练控制分支参数（`--trainable_modules control`）。训练保留 `scripts/minimax_h3/train.py` 的视频 + 音频联合 flow-matching 损失，因此训练数据需要同时包含视频、管控视频**和**音频文件。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径使用](#24-相对路径与绝对路径使用)
- [三、控制分支训练](#三控制分支训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 控制分支 YAML 配置](#32-控制分支-yaml-配置)
  - [3.3 快速开始（FSDP）](#33-快速开始fsdp)
  - [3.4 控制训练专用参数参考](#34-控制训练专用参数参考)
  - [3.5 训练验证](#35-训练验证)
  - [3.6 使用 FSDP2 训练](#36-使用-fsdp2-训练)
  - [3.7 其他后端](#37-其他后端)
    - [3.7.1 使用 DeepSpeed-Zero-2 训练](#371-使用-deepspeed-zero-2-训练)
    - [3.7.2 不使用 DeepSpeed 或 FSDP 训练](#372-不使用-deepspeed-或-fsdp-训练)
  - [3.8 多机分布式训练](#38-多机分布式训练)
  - [3.9 控制分支 CFG 蒸馏](#39-控制分支-cfg-蒸馏)
  - [3.10 提取控制分支权重](#310-提取控制分支权重)
- [四、推理测试](#四推理测试)
  - [4.1 推理参数参考](#41-推理参数参考)
  - [4.2 V2V Control 推理](#42-v2v-control-推理)
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
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**方式三：使用 Docker**

使用 Docker 时，请确保机器上的 GPU 驱动和 CUDA 环境已正确安装，然后执行以下命令：

```
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# 进入镜像
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 二、数据准备

### 2.1 快速测试数据集

我们提供了一个包含管控信号的测试数据集，其中包含若干训练数据。

```bash
# 下载官方示例数据集（含管控信号）
modelscope download --dataset PAI/X-Fun-Videos-Controls-Demo --local_dir ./datasets/X-Fun-Videos-Controls-Demo
```

下载后数据集包含以下 metadata 文件：
- `metadata.json`：基本格式（仅包含管控视频路径）
- `metadata_add_width_height.json`：含宽高信息
- `metadata_add_width_height_add_wav.json`：含宽高信息 + 音频路径（推荐用于 MiniMax-H3 控制训练）

> 💡 示例数据集的管控信号位于 `canny/`，配对音频位于 `wav/`（由 `metadata_add_width_height_add_wav.json` 引用）；不含 `audio_path` 的 metadata 同样可用，此时训练数据集会直接从视频容器中解码音频轨道。

### 2.2 数据集结构

控制训练数据集需要原始视频、与之对应的管控信号视频（如 pose 视频、depth 视频、canny 边缘视频等），**以及配对的音频轨道**（MiniMax-H3 保留视频 + 音频联合训练损失）。

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/                    # 原始训练视频
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   ├── 📂 control/                  # 管控信号视频（如 pose / depth / canny）
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   ├── 📂 wav/                      # 配对音频轨道
│   │   ├── 📄 audio001.wav
│   │   ├── 📄 audio002.wav
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

> **说明**：
> - `train/` 目录存放原始视频
> - `control/`（或 `pose/`、`depth/`、`canny/` 等）目录存放与原始视频一一对应的管控信号视频。目录名可自定义，只要 `metadata.json` 中的 `control_file_path` 正确指向即可
> - `wav/` 目录存放配对音频，训练时会被重采样到音频 VAE 的采样率（32 kHz）

### 2.3 metadata.json 格式

> ⚠️ **重要**：与普通视频训练不同，MiniMax-H3 控制训练要求 `metadata.json` 中**包含 `control_file_path` 字段** —— `VideoSpeechControlDataset` 正是在常规字段之上读取该字段。`audio_path` 字段可选；缺省时直接从视频容器中解码音频轨道。

**相对路径格式**（示例）：
```json
[
  {
    "file_path": "train/video001.mp4",
    "control_file_path": "control/video001.mp4",
    "audio_path": "wav/audio001.wav",
    "text": "A brown dog barks on a sofa, sitting on a light-colored couch in a cozy room",
    "type": "video",
    "width": 768,
    "height": 512
  }
]
```

**绝对路径格式**：
```json
[
  {
    "file_path": "/mnt/data/videos/dog.mp4",
    "control_file_path": "/mnt/data/control/dog.mp4",
    "audio_path": "/mnt/data/wavs/dog.wav",
    "text": "A brown dog barks on a sofa",
    "type": "video",
    "width": 768,
    "height": 512
  }
]
```

**关键字段说明**：
- `file_path`：原始视频路径（相对或绝对）
- `control_file_path`：管控信号视频路径（**控制训练必需**）
- `audio_path`：音频文件路径（**MiniMax-H3 特有，可选**）。音频通常为 `.wav` 格式，路径应与 `file_path` 对应。缺省时直接从视频容器中解码音频轨道
- `text`：视频描述（英文提示词）
- `type`：数据类型，固定为 `"video"`
- `width` / `height`：视频尺寸（**建议提供**，用于 bucket 训练。若不提供，训练时会自动读取，当数据存放在 OSS 等较慢的存储系统上时可能影响训练速度）。
  - 可使用 `scripts/process_json_add_width_and_height.py` 为缺少这两个字段的 JSON 文件提取宽高，支持图片和视频。

### 2.4 相对路径与绝对路径使用

**相对路径**：

若数据使用相对路径，在训练脚本中设置：

```bash
export DATASET_NAME="datasets/my_dataset/"
export DATASET_META_NAME="datasets/my_dataset/metadata.json"
```

**绝对路径**：

若数据使用绝对路径，在训练脚本中设置：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **建议**：小型本地数据集使用相对路径；存放在外部存储（NAS、OSS）或多机共享存储上的数据集使用绝对路径。

---

## 三、控制分支训练

### 3.1 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 MiniMax-H3 官方权重
hf download MiniMax-AI/MiniMax-H3 --local-dir models/Diffusion_Transformer/MiniMax-H3

# 下载预训练控制分支（Controlnet-Union）权重
modelscope download --model PAI/MiniMax-H3-Fun-Controlnet-Union --local_dir models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union
```

> 💡 加载器既接受上述转换后的 diffusers 布局，也接受*原始* MiniMax-H3 分区（如 `MiniMax-H3/FL2VA`）；原始分片在加载时即时转换，不落中间文件。控制分支**不包含**在基础 MiniMax-H3 权重中：`from_pretrained` 会自动补全——每个控制块从其挂载的主干块初始化、`control_proj_in` 从 `proj_in` 初始化，且 `before_proj` / `after_proj` 置零，因此刚加载的模型在数值上与基础 MiniMax-H3 模型完全一致。若需带管控信号训练或推理，请通过 `--transformer_path`（训练热启动）或 `transformer_path`（推理）加载发布的控制分支 checkpoint `models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union/MiniMax-H3-Fun-Controlnet-Union.safetensors`。

### 3.2 控制分支 YAML 配置

控制分支的结构由通过 `--config_path` 传入的 YAML 配置驱动，其 `transformer_additional_kwargs` 在训练和推理时以相同方式展开传入 `from_pretrained`：

```yaml
# config/minimax_h3/minimax_h3_control.yaml（inpaint 变体）
format: diffusers
pipeline: minimax-h3
transformer_additional_kwargs:
    control_blocks_places: [0, 10, 20, 30, 40]
    control_in_dim: 49
```

| 键 | 说明 | 默认值 |
|----|------|--------|
| `control_blocks_places` | 零初始化控制块挂载的 transformer 层索引列表 | - |
| `control_in_dim` | 控制行携带的通道数。`49` = control latent（24）+ 可见性图（1）+ 被 mask 的视频 latent（24），与 `--enable_inpaint` 搭配使用；不带 mask 的分支使用 `24`（`config/minimax_h3/minimax_h3_control_only.yaml`） | - |
| `control_apply_audio` | 控制 skip 是否作用于音频行。设为 `False` 时每个 skip 注入前先将音频行置零，控制视频只引导视频行，音轨保持基座模型的生成路径 | `True` |

> ⚠️ **重要**：`control_in_dim` 必须由 YAML 显式指定，代码中不做任何推断。`control_in_dim=49` 仅在启用 `--enable_inpaint` 时有效；不开启该 flag 时使用 `control_in_dim=24`。用一种布局训练的 checkpoint 无法加载到另一种布局构建的模型中（`control_proj_in.weight` 形状不匹配），且推理必须加载与训练**相同**的 YAML 配置。

### 3.3 快速开始（FSDP）

推荐使用 FSDP 训练 MiniMax-H3 控制模型：仅 transformer 在 bfloat16 下就约 62 GB，Qwen3-VL conditioner 另有约 62 GB，模型权重必须在多卡间分片——这正是 FSDP（`FULL_SHARD`）能做而 DeepSpeed-Zero-2 不能做的。仅控制分支可训练（`--trainable_modules control`），其余部分保持冻结。

```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境。
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" \
    --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap BaseMiniMaxH3TransformerBlock,MiniMaxH3ControlTransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
      scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=311 \
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
  --output_dir="output_dir_minimax_h3_control_inpaint" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --enable_inpaint \
  --low_vram \
  --validation_paths "asset/pose.mp4" \
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body" \
  --uniform_sampling \
  --trainable_modules "control" \
  --resume_from_checkpoint=latest
```

> 💡 启动脚本 `scripts/minimax_h3_fun/train_control.sh` 可作为起点。`VIDEOX_OFFLOAD_VACE_LATENTS=True` 会将编码后的 control/video latent 在步间卸载到 CPU。

### 3.4 控制训练专用参数参考

**控制训练关键参数**：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--config_path` | 控制分支 YAML 配置（固定 `control_blocks_places` / `control_in_dim`，见 3.2） | `config/minimax_h3/minimax_h3_control.yaml` |
| `--enable_inpaint` | 在控制分支中与管控视频一起送入随机 inpaint mask：控制行在 `in_channels` 之上额外携带可见性图 + 被 mask 的视频 latent。YAML 必须固定与之匹配的加宽 `control_in_dim`（49） | - |
| `--trainable_modules` | 可训练模块（`control` 只训练控制旁路分支，为默认训练配方） | `"control"` |
| `--trainable_modules_low_learning_rate` | 使用更低学习率训练的模块 | `[]` |
| `--transformer_path` | 从其他 checkpoint 加载 transformer 权重（如热启动） | None |
| `--pretrained_model_name_or_path` | MiniMax-H3 预训练模型路径 | `models/Diffusion_Transformer/MiniMax-H3` |
| `--train_data_dir` | 训练数据目录（meta 使用绝对路径时置空） | `""` |
| `--train_data_meta` | 训练数据元信息文件 | `/mnt/data/datasets/my_dataset/metadata.json` |
| `--train_batch_size` | 每卡 batch size | 1 |
| `--video_sample_size` | 视频最大训练分辨率 | 960 |
| `--token_sample_size` | 开启 `training_with_video_token_length` 时最大 token 长度对应的分辨率 | 960 |
| `--video_sample_stride` | 抽帧步长（MiniMax-H3 为 24 fps） | 1 |
| `--video_sample_n_frames` | 采样帧数，需满足视频 VAE 的 `17*n+5` 形式（时长保持在 5～15 秒之间） | 311 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练轮数 | 100 |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | 50 |
| `--learning_rate` | 初始学习率（控制分支训练推荐值） | 2e-05 |
| `--lr_scheduler` | 学习率调度器：`linear`、`cosine`、`cosine_with_restarts`、`polynomial`、`constant`、`constant_with_warmup` | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率 warmup 步数 | 100 |
| `--seed` | 随机种子，用于复现训练 | 42 |
| `--output_dir` | 输出目录 | `output_dir_minimax_h3_control_inpaint` |
| `--gradient_checkpointing` | 开启梯度检查点以节省显存 | - |
| `--gradient_checkpointing_save_on_cpu` | 将 transformer 块为反向传播保存的激活卸载到 CPU 内存 | - |
| `--mixed_precision` | 混合精度：`no`、`fp16`、`bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon 值 | 1e-10 |
| `--vae_mini_batch` | VAE 编码 mini batch 大小 | 1 |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--enable_bucket` | 开启 bucket 训练，不裁剪视频，按分辨率分组训练 | - |
| `--random_hw_adapt` | 将视频自动缩放到 `[min_size, max_size]` 范围内的随机尺寸 | - |
| `--training_with_video_token_length` | 按 token 长度训练，支持任意分辨率 | - |
| `--uniform_sampling` | 均匀时间步采样（推荐） | - |
| `--low_vram` | VAE 与 conditioner 常驻 CPU，仅在编码时搬上 GPU | - |
| `--offload_every_step` | 每步之间将 transformer 在 CPU 间搬运（适用于远小于 62 GB 的显卡） | - |
| `--resume_from_checkpoint` | 断点续训路径，使用 `"latest"` 自动选择最新 checkpoint | `latest` |
| `--validation_steps` | 每 N 步执行一次验证 | 100 |
| `--validation_epochs` | 每 N 个 epoch 执行一次验证 | 500 |
| `--validation_prompts` | 验证用提示词 | `"1girl, ..."` |
| `--validation_paths` | 验证用管控视频路径 | `"asset/pose.mp4"` |
| `--validation_sampling_steps` | 验证采样循环的去噪步数 | 50 |
| `--use_8bit_adam` | 使用 8-bit Adam 优化器以节省显存 | - |

**训练配方说明**：
- 10% 的 batch 会将控制 latent 置零，保持无条件路径可训练（CFG）
- 音频流保留 `scripts/minimax_h3/train.py` 的视频 + 音频联合 flow-matching 损失
- checkpoint 以 diffusers 布局序列化：`<output_dir>/checkpoint-x/transformer/diffusion_pytorch_model.safetensors` 加 `config.json`，推理脚本通过 `from_pretrained(..., subfolder="transformer")` 加载

### 3.5 训练验证

可以配置验证参数，在训练过程中周期性生成测试视频，以监控训练进度和模型质量。

**验证参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--validation_steps` | 每 N 步执行一次验证 | 2000 |
| `--validation_epochs` | 每 N 个 epoch 执行一次验证 | 5 |
| `--validation_prompts` | 验证视频生成所用提示词 | None |
| `--validation_paths` | 验证用管控视频路径 | None |
| `--validation_sampling_steps` | 验证采样循环的去噪步数 | 50 |

**验证示例**：

```bash
  --validation_paths "asset/pose.mp4" \
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body"
```

**说明**：
- 验证视频保存在 `output_dir/sample/` 目录
- 多提示词格式：`--validation_prompts "prompt1" "prompt2" "prompt3"`
- `validation_paths` 应与 `validation_prompts` 一一对应，指向管控视频文件

### 3.6 使用 FSDP2 训练

同一个 `train_control.py` 在 accelerate launch 命令上加 `--fsdp_version 2` 即可以 FSDP2 运行。选择前需要了解：

- `--fsdp_sharding_strategy`、`--fsdp_backward_prefetch` 和 `--fsdp_use_orig_params` 仅适用于 FSDP1；FSDP2 下的 reshard 行为由 `--fsdp_reshard_after_forward` 控制，且必须是布尔值
- FSDP2 通过 unit 前向输入上的 autograd 函数对完全冻结的 unit 做 reshard，因此冻结的主干在反向传播过程中即可释放，不需要 FSDP1 所需的各种变通手段
- accelerate 的 FSDP2 路径在开启混合精度时会将模型上转为 float32，冻结的基座因此保留 fp32 master weights：分片后的参数显存约为 FSDP1 运行的两倍
- `fully_shard` 在分片前会将 CPU 上的模型整体搬上 GPU，因此 60GB+ 的模型需要 `--fsdp_cpu_ram_efficient_loading True`（meta 初始化 + rank-0 广播）以避免每卡显存 OOM

```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" \
    --use_fsdp --fsdp_version 2 \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap BaseMiniMaxH3TransformerBlock,MiniMaxH3ControlTransformerBlock \
    --fsdp_reshard_after_forward true \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_cpu_ram_efficient_loading True \
      scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  ... # train_control.py 其余参数与快速开始一致
```

### 3.7 其他后端

#### 3.7.1 使用 DeepSpeed-Zero-2 训练

> ⚠️ **警告**：DeepSpeed-Zero-2 只分片优化器状态和梯度，**不分片模型权重**。MiniMax-H3 的 transformer 约 62 GB，每张卡仍需保留一份完整权重副本，该配置通常会显存溢出。MiniMax-H3 控制训练请优先使用 FSDP（**3.3**）；以下命令仅供参考。

```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  ... # train_control.py 其余参数与快速开始一致
```

#### 3.7.2 不使用 DeepSpeed 或 FSDP 训练

**不推荐此方式**：缺少省显存后端，极易显存溢出。MiniMax-H3 约 62 GB 的 transformer 和约 62 GB 的 Qwen3-VL conditioner 会在每张卡上完整复制，几乎必然 OOM。以下仅供参考。

```bash
accelerate launch --mixed_precision="bf16" scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  ... # train_control.py 其余参数与快速开始一致
```

### 3.8 多机分布式训练

**适用场景**：超大规模数据集、更快的训练速度

#### 3.8.1 环境配置

假设 2 台机器，每台 8 张 GPU：

**机器 0（Master）**：
```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
export MASTER_ADDR="192.168.1.100"  # Master 机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 总机器数
export NUM_PROCESS=16                # 总进程数 = 机器数 x 8
export RANK=0                        # 当前机器 rank（0 或 1）
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境。
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK \
    --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap BaseMiniMaxH3TransformerBlock,MiniMaxH3ControlTransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
      scripts/minimax_h3_fun/train_control.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  ... # train_control.py 其余参数与快速开始一致
```

**机器 1（Worker）**：
```bash
export RANK=1  # 注意这里是 1
# 其余环境变量与机器 0 相同

# 使用与机器 0 相同的 accelerate launch 命令
```

#### 3.8.2 多机训练注意事项

- **网络要求**：
   - 推荐 RDMA/InfiniBand（高性能）
   - 无 RDMA 时添加环境变量：
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **数据同步**：所有机器必须能访问相同的数据路径（NFS/共享存储）

### 3.9 控制分支 CFG 蒸馏

`scripts/minimax_h3_fun/train_control_distill.py` 将 classifier-free guidance 蒸馏进控制分支（参照 `scripts/flux2_fun/train_control_distill.py`）：冻结的 teacher 控制 transformer 副本每步执行两次前向——一次用提示词、一次用空负面提示词——两次预测按 `--real_guidance_scale` 组合出 teacher 分数，可训练的 student 对视频和音频行以 MSE 损失回归该目标。student 与 teacher 加载同一份训练好的控制分支，因此将 `--transformer_path` 指向它即可——既可以是 `train_control.py` 保存的整份 transformer safetensors，也可以是 **3.10** 提取的 control-only 文件（加载时对缺失键使用 `strict=False`）。

```bash
export VIDEOX_OFFLOAD_VACE_LATENTS=True
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height_add_wav.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" \
    --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap BaseMiniMaxH3TransformerBlock,MiniMaxH3ControlTransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
      scripts/minimax_h3_fun/train_control_distill.py \
  --config_path="config/minimax_h3/minimax_h3_control.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=311 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_control_distill" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --enable_inpaint \
  --uniform_sampling \
  --trainable_modules "control" \
  --transformer_path="output_dir_minimax_h3_control_inpaint/checkpoint-xxx/transformer/diffusion_pytorch_model_control.safetensors" \
  --real_guidance_scale=3.5 \
  --resume_from_checkpoint=latest
```

**蒸馏关键参数**：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--config_path` | 与源 checkpoint 训练时相同的控制分支 YAML（见 3.2） | `config/minimax_h3/minimax_h3_control.yaml` |
| `--transformer_path` | student 与冻结 teacher 共同加载的训练好的控制分支 checkpoint；既接受整份 transformer safetensors，也接受 **3.10** 的 control-only 文件 | `output_dir_minimax_h3_control_inpaint/checkpoint-xxx/transformer/diffusion_pytorch_model_control.safetensors` |
| `--enable_inpaint` | 保持与源 checkpoint 一致的 flag：蒸馏时 student 与 teacher 走相同的 mask 通道输入 | - |
| `--real_guidance_scale` | teacher 两次前向的 CFG 强度 | 3.5 |
| `--learning_rate` | 推荐使用比控制训练更低的学习率 | 2e-06 |

> 💡 启动脚本 `scripts/minimax_h3_fun/train_control_distill.sh` 可作为起点。发布的 MiniMax-H3 checkpoint 已经过 guidance 蒸馏，因此 student 侧不接受 guidance 输入；guidance 仅通过 teacher 的两次 CFG 前向进入。控制训练中的 `--low_vram` 在这里同样支持，但冻结 teacher 会因此在每步进出 GPU，所以启动脚本默认不开启。

### 3.10 提取控制分支权重

`train_control.py` 保存的是完整 transformer（主干分支 + 控制分支）。若只需将控制分支（`control_blocks.*` + `control_proj_in.*`）导出为独立 safetensors 文件：

```bash
python scripts/minimax_h3_fun/extract_control_weights.py \
  --model_path output_dir_minimax_h3_control_inpaint/checkpoint-1000/transformer/diffusion_pytorch_model.safetensors \
  --output_path output_dir_minimax_h3_control_inpaint/checkpoint-1000/transformer/diffusion_pytorch_model_control.safetensors
```

> 💡 `--model_path` 也可直接指向 `<checkpoint>/transformer` 目录（会自动合并所有分片，并将其 `config.json` 中的 `control_blocks_places` / `control_in_dim` 作为 safetensors 元数据写入）。

提取出的文件可通过 `MiniMaxH3ControlTransformer3DModel.materialize_missing_control_params(...)` 重新应用到新的基础模型上，也可以直接通过 `train_control.py` / `train_control_distill.py` 的 `--transformer_path` 以及推理脚本的 `transformer_path` 加载（对缺失键使用 `strict=False`）。

---

## 四、推理测试

### 4.1 推理参数参考

**关键参数**：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `GPU_memory_mode` | GPU 显存管理模式，见下表 | `model_group_offload` |
| `ulysses_degree` | 头维度并行度，单卡为 1 | 1 |
| `ring_degree` | 序列维度并行度，单卡为 1 | 1 |
| `fsdp_dit` | 多卡推理时对 Transformer（主干 + 控制块）使用 FSDP | `False` |
| `fsdp_text_encoder` | 对 Qwen3-VL 文本编码器（约 62 GB）使用 FSDP；包裹内层 `text_encoder.model` 的 decoder 层 | `True` |
| `compile_dit` | 编译 Transformer 以加速推理（固定分辨率下有效） | `False` |
| `model_name` | MiniMax-H3 基座模型路径 | `models/Diffusion_Transformer/MiniMax-H3` |
| `config_path` | 控制分支 YAML，**必须与 `train_control.py` 训练时使用的一致**（inpaint checkpoint 对应 `control_in_dim` 49）。置 `None` 会构建默认 24 通道分支，无法加载 inpaint checkpoint | `config/minimax_h3/minimax_h3_control.yaml` |
| `transformer_path` | 控制分支 checkpoint 路径：发布的 `MiniMax-H3-Fun-Controlnet-Union` 权重或训练得到的控制分支 checkpoint。仅接受 `.safetensors` 文件（整份 transformer 或 **3.10** 的 control-only 文件均可，对缺失键使用 `strict=False`），不接受 checkpoint 的 `transformer` 目录 | `models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union/MiniMax-H3-Fun-Controlnet-Union.safetensors` |
| `vae_path` | 训练好的 VAE 权重路径 | `None` |
| `lora_path` | LoRA 权重路径 | `None` |
| `sample_size` | 生成视频分辨率 `[height, width]`，宽高必须是 32 的倍数。控制推理按训练时的 resize + crop 几何将管控视频适配到该画布，因此**不能为 None** | `[704, 1280]` |
| `video_length` | 生成长度的上限，受视频 VAE 的 `17*n+5` 网格约束（时长不超过 15 秒）：实际生成长度跟随管控视频的真实长度，向下对齐到最大的合法 `17*n+5`，短管控视频不会被补帧 | 243 |
| `fps` | 帧率（MiniMax-H3 固定以 24 fps 生成） | 24 |
| `control_context_scale` | 每个控制 skip 加到主干前的缩放系数。0.0 关闭控制分支；小于 1.0 减弱控制引导强度 | 1.00 |
| `weight_dtype` | 模型权重精度，不支持 bf16 的 GPU 请使用 `torch.float16` | `torch.bfloat16` |
| `control_video` | 管控信号视频路径 | `"asset/pose.mp4"` |
| `inpaint_video` | mask 背后的源视频（仅 `--enable_inpaint` 训练的 checkpoint 读取）；需提供 `inpaint_video_mask` | `None` |
| `inpaint_video_mask` | 标记待重绘区域的 mask 视频，以 0.5 二值化——白色 = 重绘、黑色 = 保留（仅 inpaint checkpoint） | `None` |
| `prompt` | 描述生成内容的正向提示词 | `"视频中，一位年轻女性站在阳光洒满的沙滩上..."` |
| `negative_prompt` | 负向提示词，仅在 `guidance_scale` > 1 时使用 | `"色调艳丽，过曝，静态..."` |
| `guidance_scale` | 引导强度。发布的 checkpoint 已经过 guidance 蒸馏：保持为 1 时每步只跑一次前向、不做 CFG——`train_control_distill.py` 的蒸馏 checkpoint 已将 teacher 的 CFG 目标烘进权重，大于 1 会二次施加 guidance、降低输出质量 | 1.0 |
| `num_inference_steps` | 去噪步数 | 40 |
| `flow_shift` | 视频调度表的指数 sigma shift，`None` 保留 checkpoint 自带值（12.0） | `None` |
| `audio_flow_shift` | 音频调度表的指数 sigma shift，`None` 保留 checkpoint 自带值（3.0） | `None` |
| `seed` | 随机种子，用于复现 | 43 |
| `lora_weight` | LoRA 权重强度 | 0.55 |
| `save_path` | 生成视频保存路径 | `samples/minimax-h3-videos-v2v-control` |

**GPU 显存管理模式**：

| 模式 | 说明 | 显存占用 |
|------|------|----------|
| `model_full_load` | 整个模型加载到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 全量加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 用完后将模型卸载到 CPU | 中 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `model_group_offload` | 层组在 CPU/CUDA 间切换 | 低 |
| `sequential_cpu_offload` | 逐层卸载（最慢） | 最低 |

> 💡 仅 transformer 在 bfloat16 下就有 61.7 GB，Qwen3-VL conditioner 另有 62.1 GB，单张 80 GB 卡需要 `model_cpu_offload` 或 `model_group_offload`。

### 4.2 V2V Control 推理

运行单卡推理：

```bash
python examples/minimax_h3_fun/predict_v2v_control.py
```

按需修改 `examples/minimax_h3_fun/predict_v2v_control.py`。首次推理重点关注以下参数，其余参数见上文推理参数参考。

```python
# 根据 GPU 显存选择
GPU_memory_mode = "model_group_offload"
# 设置为你的实际模型路径
model_name = "models/Diffusion_Transformer/MiniMax-H3"
# 控制分支布局：必须与 train_control.py 训练时使用的 yaml 一致
#（--enable_inpaint 的 checkpoint 用 minimax_h3_control.yaml，否则用 minimax_h3_control_only.yaml）
config_path = "config/minimax_h3/minimax_h3_control.yaml"
# 控制分支 checkpoint（.safetensors 文件）：发布的 MiniMax-H3-Fun-Controlnet-Union 权重，
# 或 train_control.py 训练得到的 checkpoint（extract_control_weights.py 的 control-only 文件也可加载）
transformer_path = "models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union/MiniMax-H3-Fun-Controlnet-Union.safetensors"
# 管控信号视频（如 pose 视频）；控制推理按训练时的 resize + crop 几何
# 将其适配到 sample_size 画布
control_video = "asset/pose.mp4"
# 生成视频分辨率 [height, width]，32 的倍数，此处不能为 None
sample_size = [704, 1280]
# 控制引导强度，1.00 表示每个控制 skip 全强度注入
control_context_scale = 1.00
# inpaint 输入，仅 --enable_inpaint 训练的 checkpoint 读取
inpaint_video = None
inpaint_video_mask = None
# 按生成内容填写
prompt = "..."
# ...
```

**说明**：
- 控制分支不包含在基础 MiniMax-H3 权重中：需将 `transformer_path` 指向发布的 `MiniMax-H3-Fun-Controlnet-Union` checkpoint（或 `train_control.py` 训练得到的 checkpoint），管控视频才会生效；只给 `model_name` 不给 `transformer_path` 时，旁路分支以恒等初始化（`after_proj` 为零），管控视频不起作用
- 生成长度跟随管控视频的真实长度，向下对齐到视频 VAE 可解码的最大 `17*n+5`，并以 `video_length` 为上限——短管控视频不会被补帧（不足 5 帧时提升到 5 帧）。使用 inpaint checkpoint 但未提供 inpaint 输入时，pipeline 将 mask 通道补零，运行退化为纯生成；不带 mask 的 checkpoint 则会直接拒绝 inpaint 输入
- mask 视频使用前会以 0.5 二值化，灰度 mask 可直接使用（白色 = 重绘、黑色 = 保留）
- 生成结果带音频：视频通过 `save_videos_with_audio_grid` 保存

### 4.3 多 GPU 并行推理

**适用场景**：高分辨率生成、加速推理

#### 安装并行推理依赖

```bash
pip install xfuser yunchang
```

#### 配置并行策略

编辑 `examples/minimax_h3_fun/predict_v2v_control.py`：

```python
# 确保 ulysses_degree x ring_degree = 使用的 GPU 数
# 例如使用 2 张 GPU：
ulysses_degree = 2  # 头维度并行
ring_degree = 1     # 序列维度并行
```

**配置原则**：
- `ulysses_degree` 必须能整除模型的 head 数
- `ring_degree` 沿序列维度切分，影响通信开销；head 能整除时尽量不用
- 多卡走 xfuser 序列并行路径，**与 `*cpu_offload*` 显存模式不兼容**（accelerate 的 offload hook 绑定单一设备）；多卡时请使用 `model_full_load` / `model_full_load_and_qfloat8`，并用 `fsdp_dit` / `fsdp_text_encoder` 节省显存

**配置示例**：

| GPU 数 | ulysses_degree | ring_degree | 说明 |
|--------|----------------|-------------|------|
| 1 | 1 | 1 | 单卡 |
| 4 | 4 | 1 | 头并行 |
| 8 | 8 | 1 | 头并行 |
| 8 | 2 | 4 | 混合并行 |

#### 运行多卡推理

```bash
torchrun --nproc_per_node=2 examples/minimax_h3_fun/predict_v2v_control.py
```

---

## 五、更多资源

- **MiniMax-H3 官方 GitHub**：https://github.com/MiniMax-AI/MiniMax-H3
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
