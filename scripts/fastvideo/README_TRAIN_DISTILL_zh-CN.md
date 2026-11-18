# FastVideo VSA 蒸馏训练指南

本文档提供了将 Wan2.1 蒸馏为 FastWan2.1（采用 VSA，Video Sparse Attention）的完整工作流，包括环境配置、数据准备、分布式训练和推理测试。

> **注意**: FastVideo VSA 将块稀疏视频注意力（由可学习门控为每个 query 保留约 20% 的 key 块）与 DMD 少步蒸馏结合，使学生模型在约 3 步、无 CFG 的条件下以稀疏注意力推理。训练配方、数据管线与分布式后端与 TurboDiffusion 蒸馏指南完全一致，仅稀疏学生模型类不同。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径使用方案](#24-相对路径与绝对路径使用方案)
- [三、蒸馏训练](#三蒸馏训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始（DeepSpeed-Zero-2）](#32-快速开始deepspeed-zero-2)
  - [3.3 常见训练参数](#33-常见训练参数)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 使用 FSDP 训练](#35-使用-fsdp-训练)
  - [3.6 其他后端](#36-其他后端)
  - [3.7 多机分布式训练](#37-多机分布式训练)
- [四、推理测试](#四推理测试)
  - [4.1 推理参数解析](#41-推理参数解析)
  - [4.2 文本生成视频（T2V）推理](#42-文本生成视频t2v推理)
  - [4.3 仅支持单卡](#43-仅支持单卡)
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

> 💡 VSA 块稀疏注意力运行在 Triton 内核上；Triton 随 PyTorch（>=2.1）在 Linux 上一并提供，无需额外安装。

**方式三：使用 Docker**

使用 Docker 时，请确保机器上已正确安装 GPU 驱动与 CUDA 环境，然后执行以下命令：

```
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# 进入容器
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 二、数据准备

### 2.1 快速测试数据集

我们提供了一个包含若干训练样本的测试数据集。

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
- `file_path`：视频路径（相对或绝对路径）
- `text`：视频描述（英文 prompt）
- `type`：数据类型，固定为 `"video"`
- `width` / `height`：视频尺寸（**建议**提供以便分桶训练。若不提供，训练时会自动读取，当数据存放在 OSS 等较慢系统上时会拖慢训练）。
  - 可使用 `scripts/process_json_add_width_and_height.py` 为缺少宽高字段的 JSON 文件提取 width 和 height，同时支持图片与视频。
  - 用法：`python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Demo/metadata.json --output_file datasets/X-Fun-Videos-Demo/metadata_add_width_height.json`。

### 2.4 相对路径与绝对路径使用方案

**相对路径**：

若数据使用相对路径，在训练脚本中配置：

```bash
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
```

**绝对路径**：

若数据使用绝对路径，在训练脚本中配置：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata_add_width_height.json"
```

> 💡 **建议**：数据集较小且存放在本地时使用相对路径；数据集存放在外部存储（如 NAS、OSS）或跨多台机器共享时使用绝对路径。

---

## 三、蒸馏训练

蒸馏是一次 DMD（distribution matching distillation）训练，包含三个网络：稀疏的 **generator**（FastWan 学生，`FastVideoWanTransformer3DModel`）、稀疏的 **fake_score** critic（同一个类），以及冻结的 **dense** `real_score` 教师（`WanTransformer3DModel`）。学生与 critic 携带 VSA 块稀疏自注意力，教师保持全注意力。

### 3.1 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Wan2.1 官方权重
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B
```

### 3.2 快速开始（DeepSpeed-Zero-2）

按照 **2.1 快速测试数据集** 下载数据、按照 **3.1 下载预训练模型** 下载权重后，即可直接复制运行快速开始命令。

推荐使用 DeepSpeed-Zero-2 或 FSDP 训练，这里以 DeepSpeed-Zero-2 为例。

DeepSpeed-Zero-2 与 FSDP 的区别在于模型权重是否分片。**若使用 DeepSpeed-Zero-2 多卡训练时显存不足**，可切换到 FSDP 训练。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/fastvideo/train_distill_wan2.1.py \
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
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_fastvideo_wan2.1_1.3b" \
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

| 参数 | 描述 | 示例值 |
|-----|------|-------|
| `--config_path` | 配置文件路径 | `config/wan2.1/wan_civitai.yaml` |
| `--pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B/` |
| `--train_data_dir` | 训练数据目录 | `datasets/X-Fun-Videos-Demo/` |
| `--train_data_meta` | 训练数据元信息文件 | `datasets/X-Fun-Videos-Demo/metadata_add_width_height.json` |
| `--train_batch_size` | 每张卡的 batch size | 1 |
| `--image_sample_size` | 图片训练最大分辨率 | 640 |
| `--video_sample_size` | 视频训练最大分辨率 | 640 |
| `--token_sample_size` | Token 采样尺寸 | 640 |
| `--video_sample_stride` | 视频采样步长 | 2 |
| `--video_sample_n_frames` | 视频采样帧数 | 81 |
| `--gradient_accumulation_steps` | 梯度累积步数（等效增大 batch） | 1 |
| `--dataloader_num_workers` | DataLoader 工作进程数 | 8 |
| `--num_train_epochs` | 训练轮数 | 100 |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | 50 |
| `--learning_rate` | 初始学习率（generator） | 2e-06 |
| `--learning_rate_critic` | 初始学习率（critic） | 2e-06 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率 warmup 步数 | 100 |
| `--seed` | 随机种子 | 42 |
| `--output_dir` | 输出目录 | `output_dir_fastvideo_wan2.1_1.3b` |
| `--gradient_checkpointing` | 启用梯度检查点 | - |
| `--mixed_precision` | 混合精度：`fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon | 1e-10 |
| `--vae_mini_batch` | VAE 编码 mini-batch 大小 | 1 |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--enable_bucket` | 启用分桶训练，不裁剪，按分辨率分组 | - |
| `--random_hw_adapt` | 将图片/视频自动缩放到 `[min_size, max_size]` 范围内的随机尺寸 | - |
| `--training_with_video_token_length` | 基于 token 长度训练，支持任意分辨率 | - |
| `--uniform_sampling` | 均匀时间步采样 | - |
| `--low_vram` | 低显存模式 | - |
| `--train_mode` | 训练模式：`normal`（标准）或 `i2v`（图生视频） | `normal` |
| `--resume_from_checkpoint` | 续训路径，使用 `"latest"` 自动选择最新 checkpoint | None |
| `--validation_steps` | 每 N 步运行一次验证 | 2000 |
| `--validation_epochs` | 每 N 轮运行一次验证 | 5 |
| `--validation_prompts` | 用于视频生成验证的 prompt | `"A dog shaking head..."` |
| `--trainable_modules` | 可训练模块（`"."` 表示全部模块） | `"."` |

**蒸馏专用参数**：

| 参数 | 描述 | 示例值 |
|-----|------|-------|
| `--denoising_step_indices_list` | 去噪步列表（蒸馏核心参数） | `1000 750 500 250` |
| `--real_guidance_scale` | 打分用的 real guidance scale | 6.0 |
| `--fake_guidance_scale` | 打分用的 fake guidance scale | 0.0 |
| `--gen_update_interval` | generator 更新间隔 | 5 |
| `--negative_prompt` | 蒸馏用的负向 prompt | 中文负向 prompt |
| `--validation_paths` | I2V 模式的验证图片路径 | 图片路径列表 |
| `--train_sampling_steps` | 训练采样步数 | 1000 |

**VSA 稀疏度**：

- 学生/critic 的自注意力为 VSA 块稀疏。保留比 `student_topk` 在 `FastVideoWanTransformer3DModel` 内默认为 `0.20`（即 80% 稀疏度）；它是模型类默认值，而非命令行旗标。
- 可学习的压缩门控 `to_gate_compress` 以零初始化创建，并在 `--trainable_modules "."` 下自动变为可训练（其参数名与其它模块一样包含点号）。
- `real_score` 教师保持 dense（`WanTransformer3DModel`）且冻结，因此稀疏性是被蒸馏进学生，而非从教师拷贝。
- 推理时的 `student_topk` 须与 checkpoint 训练时的取值一致；修改它会重建注意力算子并重置已训练好的门控。

**采样尺寸配置指南**：
- `video_sample_size` 表示视频的分辨率尺寸；当 `random_hw_adapt` 为 True 时，表示视频与图片分辨率之间的最小值。
- `image_sample_size` 表示图片的分辨率尺寸；当 `random_hw_adapt` 为 True 时，表示视频与图片分辨率之间的最大值。
- `token_sample_size` 表示 `training_with_video_token_length` 为 True 时最大 token 长度对应的分辨率。
- 由于配置容易混淆，**若微调不需要任意分辨率**，建议将 `video_sample_size`、`image_sample_size`、`token_sample_size` 设为同一固定值，例如 **(320, 480, 512, 640, 960)**。
  - **全部设为 320** 代表 **240P**。
  - **全部设为 480** 代表 **320P**。
  - **全部设为 640** 代表 **480P**。
  - **全部设为 960** 代表 **720P**。

**其它参数说明**：
- `enable_bucket` 用于启用分桶训练。启用后模型不在中心裁剪图片与视频，而是按分辨率分桶后对完整图片与视频进行训练。
- `random_hw_adapt` 用于对图片与视频自动高宽缩放。启用后，训练图片的高宽以 `image_sample_size` 为最大、`min(video_sample_size, 512)` 为最小。
- `training_with_video_token_length` 指定按 token 长度训练。训练图片与视频的高宽以 `image_sample_size` 为最大、`video_sample_size` 为最小。
- `train_mode` 用于指定训练模式，可为 normal 或 i2v。若只需文本生成视频，保持 `normal`。
- `resume_from_checkpoint` 用于设置从之前的 checkpoint 续训。使用路径或 `"latest"` 自动选择最近可用的 checkpoint。

### 3.4 训练验证

你可以配置验证参数，在训练期间周期性生成测试视频，以便监控训练进度与模型质量。

**验证参数说明**：

| 参数 | 描述 | 推荐值 |
|------|------|--------|
| `--validation_steps` | 每 N 步运行一次验证 | 2000 |
| `--validation_epochs` | 每 N 轮运行一次验证 | 5 |
| `--validation_prompts` | 用于视频生成验证的 prompt | 英文 prompt |
| `--validation_paths` | I2V 模式的验证图片路径（仅 i2v 模式） | `"asset/1.png"` |

**Normal 模式示例**（T2V 验证）：

```bash
  --validation_steps=2000 \
  --validation_epochs=5 \
  --validation_prompts="A dog shaking head. The video is of high quality, and the view is very clear."
```

**说明**：
- 验证视频会保存到 `output_dir` 目录。
- 多 prompt 验证格式：`--validation_prompts "prompt1" "prompt2" "prompt3"`。
- 蒸馏模型验证使用 `denoising_step_indices_list` 定义的少步、并以无 CFG（`guidance_scale=1.0`）运行，与推理保持一致。

### 3.5 使用 FSDP 训练

**若使用 DeepSpeed-Zero-2 多卡训练时显存不足**，可切换到 FSDP 训练。`scripts/fastvideo/train_distill_wan2.1.sh` 是一个开箱即用的 FSDP launcher（它以 `FULL_SHARD` 包装 `WanAttentionBlock`）。

```sh
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/fastvideo/train_distill_wan2.1.py \
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
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_fastvideo_wan2.1_1.3b" \
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

#### 3.6.1 使用 DeepSpeed-Zero-3 进行训练

目前不强烈推荐 DeepSpeed Zero-3。在本仓库中，使用 FSDP 报错更少、更稳定。

DeepSpeed Zero-3 适合高分辨率下的 14B Wan。训练后可用以下命令得到最终模型：
```bash
python scripts/zero_to_bf16.py output_dir/checkpoint-{our-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

训练 shell 命令如下：
```sh
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/fastvideo/train_distill_wan2.1.py \
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
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_fastvideo_wan2.1_1.3b" \
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

**不推荐这种方式，因为没有省显存后端时极易导致显存不足**。这里仅作为训练参考 shell 提供。

```sh
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/fastvideo/train_distill_wan2.1.py \
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
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_fastvideo_wan2.1_1.3b" \
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

**适用于**：超大数据集、更快的训练速度

#### 3.7.1 环境配置

假设有 2 台机器，每台 8 张 GPU：

**机器 0（Master）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master 机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 机器总数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 当前机器 rank（0 或 1）
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/fastvideo/train_distill_wan2.1.py \
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
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_fastvideo_wan2.1_1.3b" \
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

**机器 1（Worker）**：
```bash
export MASTER_ADDR="192.168.1.100"  # 与 Master 相同
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # 注意这里是 1
NCCL_DEBUG=INFO

# 使用与机器 0 相同的 accelerate launch 命令
```

#### 3.7.2 多机训练注意事项

- **网络要求**：
   - 推荐 RDMA/InfiniBand（高性能）
   - 没有 RDMA 时，添加环境变量：
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```
- **数据同步**：所有机器必须能访问相同的数据路径（NFS/共享存储）

---

## 四、推理测试

### 4.1 推理参数解析

**关键参数说明**（取自 `examples/fastvideo/predict_t2v_wan2.1.py`）：

| 参数 | 描述 | 示例值 |
|------|------|-------|
| `GPU_memory_mode` | GPU 显存模式，选项见下表 | `model_cpu_offload` |
| `ulysses_degree` | Ulysses 并行度（VSA 仅单卡，保持 1） | 1 |
| `ring_degree` | Ring 并行度（VSA 仅单卡，保持 1） | 1 |
| `config_path` | 配置文件路径 | `config/wan2.1/wan_civitai.yaml` |
| `model_name` | 基座模型路径（config + dense 主干 + VAE + T5） | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `student_topk` | VSA 保留比；须与训练时取值一致 | `0.20` |
| `sampler_name` | 采样器类型：`Flow`、`Flow_Unipc`、`Flow_DPM++` | `Flow_Unipc` |
| `shift` | 噪声调度 shift（官方 FastWan flow_shift） | `3.0` |
| `transformer_path` | FastWan 学生 Transformer 权重路径 | 训练 checkpoint / 官方 FastWan |
| `vae_path` | 训练好的 VAE 权重路径 | `None` |
| `lora_path` | LoRA 权重路径 | `None` |
| `sample_size` | 生成视频分辨率 `[height, width]` | `[480, 848]` |
| `video_length` | 生成帧数 | `81` |
| `fps` | 每秒帧数 | `16` |
| `weight_dtype` | 模型权重精度，不支持 bf16 的 GPU 用 `torch.float16` | `torch.bfloat16` |
| `prompt` | 描述生成内容的正向 prompt | `"A cinematic aerial shot..."` |
| `negative_prompt` | 用于规避特定内容的负向 prompt | 中文负向 prompt |
| `guidance_scale` | 引导强度（DMD 学生无 CFG） | 1.0 |
| `seed` | 复现用随机种子 | 1024 |
| `num_inference_steps` | 推理步数（FastWan 用 3 步） | 3 |
| `save_path` | 生成视频保存路径 | `samples/fastvideo-wan2.1-official` |

**GPU 显存模式说明**：

| 模式 | 描述 | 显存占用 |
|------|------|---------|
| `model_full_load` | 整个模型加载到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 全量加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 用后将模型卸载到 CPU | 中 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `model_group_offload` | 层组在 CPU/CUDA 间切换 | 低 |
| `sequential_cpu_offload` | 逐层卸载（最慢） | 最低 |

### 4.2 文本生成视频（T2V）推理

运行单卡推理：

```bash
python examples/fastvideo/predict_t2v_wan2.1.py
```

根据需求编辑 `examples/fastvideo/predict_t2v_wan2.1.py`。首次推理重点关注以下关键参数，其它参数参见上文推理参数说明。

```python
# 根据 GPU 显存选择
GPU_memory_mode = "model_cpu_offload"
# 基座模型路径（config + dense 主干 + VAE + T5）
model_name = "models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
# VSA 保留比；须与 checkpoint 训练时取值一致
student_topk = 0.20
# FastWan 学生权重路径（你训练的 checkpoint，或官方 FastWan 发布权重）
transformer_path = "models/Personalized_Model/FastWan2.1-T2V-1.3B.pth"
# FastWan 采样器 + shift（无 CFG 少步学生）
sampler_name = "Flow_Unipc"
shift = 3.0
guidance_scale = 1.0
num_inference_steps = 3
# 根据你的生成内容编写
prompt = "A cinematic aerial shot of a snow-covered mountain village at dawn..."
# ...
```

> 💡 官方 `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` checkpoint 采用 diffusers 键名；`examples/fastvideo/predict_t2v_wan2.1.py` 会在加载前将其 remap 到 VideoX-Fun civitai 布局（包含 VSA `to_gate_compress` 门控）。本训练指南产出的 checkpoint 已是原生布局。

### 4.3 仅支持单卡

VSA 块稀疏内核没有序列并行路径，因此 FastWan 学生只能在**单卡**上运行。请保持 `ulysses_degree = ring_degree = 1`；否则调用 `enable_multi_gpus_inference()` 会 raise。要在单卡上容纳更大分辨率，请通过 `GPU_memory_mode`（如 `model_cpu_offload` 或 `sequential_cpu_offload`）降低显存占用，而不是跨卡拆分。

---

## 五、更多资源

- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
- **FastVideo GitHub**：https://github.com/hao-ai-lab/FastVideo
- **VSA 论文（Video Sparse Attention）**：https://arxiv.org/abs/2505.13389
