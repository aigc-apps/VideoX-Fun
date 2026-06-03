# Wan2.1 Self-Forcing ODE 回归预训练指南

本文档介绍 Wan2.1 Self-Forcing 的 **ODE 回归预训练** 完整流程，涵盖环境配置、ODE 轨迹对生成、ODE 回归训练。

> **什么是 ODE 回归训练？**
>
> ODE 回归是 Self-Forcing 蒸馏的 **预训练阶段**，整体流程分两步：
>
> 1. **第一步 — 生成 ODE 对**（`generate_ode_pairs.py`）：使用 **双向教师模型** Wan2.1-T2V-1.3B，对一组文本提示词执行完整的多步 CFG 去噪，将 ODE 轨迹上的中间 latent 与编码后的 prompt embedding 一起保存为 `.safetensors` 文件。
> 2. **第二步 — ODE 回归训练**（`train_ode.py`）：加载第一步生成的 ODE 对，训练一个 **因果生成器**，在轨迹上随机抽样多个噪声等级，预测干净的终点 `x0`。训练得到的权重（通常保存为 `ode_init.pt`）作为 **Self-Forcing 蒸馏阶段**（`train_distill.py`，参见 [README_TRAIN.md](./README_TRAIN.md)）的强初始化。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、下载预训练模型](#二下载预训练模型)
- [三、第一步 — 生成 ODE 轨迹对](#三第一步--生成-ode-轨迹对)
  - [3.1 下载提示词文件](#31-下载提示词文件)
  - [3.2 运行 ODE 对生成](#32-运行-ode-对生成)
  - [3.3 输出格式](#33-输出格式)
  - [3.4 生成参数说明](#34-生成参数说明)
  - [3.5 多卡生成](#35-多卡生成)
- [四、第二步 — ODE 回归训练](#四第二步--ode-回归训练)
  - [4.1 快速开始](#41-快速开始)
  - [4.2 训练常用参数](#42-训练常用参数)
  - [4.3 使用 DeepSpeed-Zero-2 / FSDP 训练](#43-使用-deepspeed-zero-2--fsdp-训练)
  - [4.4 多机分布式训练](#44-多机分布式训练)
- [五、使用训练好的 ODE 权重](#五使用训练好的-ode-权重)
- [六、更多资源](#六更多资源)

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
pip install yunchang xfuser modelscope openpyxl deepspeed==0.17.0 numpy==1.26.4
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

**方式 3：使用 docker**

使用 docker 时，请确保机器中已正确安装显卡驱动与 CUDA 环境，然后依次执行以下命令：

```bash
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# 进入容器
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 二、下载预训练模型

ODE 生成阶段使用 **双向教师模型** Wan2.1-T2V-1.3B 进行去噪；ODE 训练阶段同样以该基础模型来初始化 **因果生成器**。

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Wan2.1 T2V 基础模型（生成时作为教师，训练时作为初始化）
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B
```

---

## 三、第一步 — 生成 ODE 轨迹对

该步骤使用双向教师模型对每条提示词执行 **48 步 CFG 去噪**，并将得到的 ODE 轨迹与对应的 prompt embedding 一起保存为 `.safetensors` 文件。所有提示词处理完成后，会自动生成一个 `outputs.json` 标注文件，供后续训练阶段使用。

### 3.1 下载提示词文件

推荐使用 Self-Forcing 官方提供的提示词列表：

```bash
mkdir -p datasets

# 从 Self-Forcing 官方仓库下载 vidprom_filtered_extended.txt
hf download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir datasets/
# 最终路径：datasets/vidprom_filtered_extended.txt
```

也可以使用任意纯文本文件，每行一条提示词。

### 3.2 运行 ODE 对生成

直接复用启动脚本 [scripts/wan2.1_self_forcing/generate_ode_pairs.sh](./generate_ode_pairs.sh)：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.1_self_forcing/generate_ode_pairs.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --video_sample_n_frames=81 \
  --height=480 \
  --width=832 \
  --guidance_scale=6.0 \
  --shift=8.0 \
  --num_inference_steps=48 \
  --caption_path="datasets/vidprom_filtered_extended.txt" \
  --output_folder="datasets/ode_pairs_output" \
  --sample_every_n_prompts=50
```

或者直接执行 shell 脚本：

```bash
bash scripts/wan2.1_self_forcing/generate_ode_pairs.sh
```

### 3.3 输出格式

生成完成后，`--output_folder` 中包含以下内容：

```
📦 datasets/ode_pairs_output/
├── 📄 00000.safetensors    # 单条提示词对应的 ODE 轨迹与 prompt embedding
├── 📄 00001.safetensors
├── 📄 ...
├── 📂 sample/              # 可选预览视频（当 sample_every_n_prompts > 0 时）
│   └── 📄 00000_clean.mp4
└── 📄 outputs.json         # 由 train_ode.py 读取的标注文件
```

每个 `.safetensors` 文件包含以下字段：

| 字段 | 形状 | 说明 |
|------|------|------|
| `latents` | `[5, C, F, H, W]` | 对 48 步 ODE 轨迹的稀疏 5 点采样：索引 `[0, 12, 24, 36, -1]`（初始噪声 → 3 个中间点 → 干净终点） |
| `prompt_embeds` | `[512, D]` | 经 padding 的 T5 prompt embedding（最大长度 512） |
| `prompt_attention_mask` | `[512]` | prompt embedding 的注意力掩码 |

自动生成的 `outputs.json` 与标准 `metadata.json` 格式一致：

```json
[
  { "file_path": "datasets/ode_pairs_output/00000.safetensors" },
  { "file_path": "datasets/ode_pairs_output/00001.safetensors" }
]
```

### 3.4 生成参数说明

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `--pretrained_model_name_or_path` | Wan2.1-T2V-1.3B 教师模型路径 | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `--config_path` | 模型配置 YAML | `config/wan2.1/wan_civitai.yaml` |
| `--caption_path` | 每行一条提示词的纯文本文件 | `datasets/vidprom_filtered_extended.txt` |
| `--output_folder` | `.safetensors` 与 `outputs.json` 的输出目录 | `datasets/ode_pairs_output` |
| `--guidance_scale` | 教师模型使用的 CFG 引导强度 | 6.0 |
| `--num_inference_steps` | 教师去噪步数（必须 ≥ 37，因为代码采样的索引为 `[0,12,24,36,-1]`） | 48 |
| `--shift` | `FlowMatchEulerDiscreteScheduler` 的 shift 值（**必须与训练阶段一致**） | 8.0 |
| `--video_sample_n_frames` | 生成视频的像素帧数 | 81 |
| `--height` / `--width` | 视频分辨率（像素） | 480 / 832 |
| `--negative_prompt` | CFG 使用的负向提示词 | （默认中文负向提示词） |
| `--sample_every_n_prompts` | 每 N 条提示词解码并保存一次预览 MP4（0 表示关闭） | 50 |
| `--mixed_precision` | `no` / `fp16` / `bf16` | `bf16` |

> ⚠️ **生成与训练阶段必须使用相同的 `--shift` 值**，提供的脚本均默认为 `8.0`。

### 3.5 多卡生成

`generate_ode_pairs.py` 基于 `accelerate` 实现，每个 rank 自动按 `prompt_index = index * world_size + rank` 交替处理提示词，并自动跳过已存在的文件，因此天然 **可断点续跑、可多卡并行**：

```bash
# 8 卡生成
accelerate launch --multi_gpu --num_processes=8 --mixed_precision="bf16" \
    scripts/wan2.1_self_forcing/generate_ode_pairs.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --config_path="config/wan2.1/wan_civitai.yaml" \
    --caption_path="datasets/vidprom_filtered_extended.txt" \
    --output_folder="datasets/ode_pairs_output" \
    --num_inference_steps=48 --guidance_scale=6.0 --shift=8.0 \
    --height=480 --width=832 --video_sample_n_frames=81
```

最终 `outputs.json` 仅由主进程写入。

---

## 四、第二步 — ODE 回归训练

第一步完成、`datasets/ode_pairs_output/outputs.json` 生成后，即可训练因果生成器（`WanTransformer3DModel_SelfForcing`）来回归 ODE 轨迹。

每个训练样本上，训练脚本会：
1. 从一个 `.safetensors` 文件中加载稀疏的 5 点轨迹与 prompt embedding；
2. 按 **块**（每 `--num_frame_per_block` 帧共享同一时间步）随机选取一个轨迹点，将带噪 latent 与逐帧时间步送入因果生成器；
3. 将生成器输出的 flow 转换为 `x0` 预测，与轨迹的 **干净终点** 计算 MSE 损失。

### 4.1 快速开始

直接复用启动脚本 [scripts/wan2.1_self_forcing/train_ode.sh](./train_ode.sh)：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME=""
export ODE_DATA_META="datasets/ode_pairs_output/outputs.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.1_self_forcing/train_ode.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$ODE_DATA_META \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_self_forcing_ode_regression" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.05 \
  --num_frame_per_block=3 \
  --train_sampling_steps=1000 \
  --denoising_step_indices_list 1000 750 500 250 \
  --shift=8.0 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "."
```

或者直接执行：

```bash
bash scripts/wan2.1_self_forcing/train_ode.sh
```

> 💡 因为 ODE 轨迹与 prompt embedding 已在第一步预先计算完毕，**ODE 训练阶段不会再调用 VAE / 文本编码器**，训练速度快、显存占用低。当 `outputs.json` 中已经使用绝对路径时，`train_data_dir` 可以留空。

### 4.2 训练常用参数

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `--pretrained_model_name_or_path` | 用于初始化因果生成器的基础模型 | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `--config_path` | 模型配置 YAML | `config/wan2.1/wan_civitai.yaml` |
| `--train_data_dir` | 拼接到 `file_path` 之前的可选根目录；若 `outputs.json` 已使用绝对路径可留空 | `""` |
| `--train_data_meta` | 第一步生成的标注 JSON | `datasets/ode_pairs_output/outputs.json` |
| `--train_batch_size` | 每卡 batch size | 1 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存一次 checkpoint | 500 |
| `--learning_rate` | 初始学习率 | 2e-06 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--seed` | 随机种子 | 42 |
| `--output_dir` | 输出目录 | `output_dir_wan2.1_self_forcing_ode_regression` |
| `--gradient_checkpointing` | 启用激活重计算 | - |
| `--mixed_precision` | `fp16` / `bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon | 1e-10 |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--trainable_modules` | 可训练模块（`"."` 表示全量） | `"."` |
| `--resume_from_checkpoint` | 恢复训练路径或 `"latest"` | `latest` |

**ODE 特有参数**（除非清楚后果，否则需与第一步保持一致）：

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `--train_sampling_steps` | 调度器总时间步数，从中按 `denoising_step_indices_list` 抽样 | 1000 |
| `--denoising_step_indices_list` | ODE 回归使用的离散时间步索引（与第一步抽样的 5 个稀疏轨迹点对应） | `1000 750 500 250` |
| `--shift` | `FlowMatchEulerDiscreteScheduler` 的 shift —— **必须与第一步生成时使用的 `--shift` 一致** | 8.0 |
| `--num_frame_per_block` | 每个因果块包含的帧数（同一块内的帧共享同一时间步） | 3 |
| `--independent_first_frame` | 第一帧是否独立（`[1, N, N, ...]` 块模式） | - |
| `--context_noise` | 上下文噪声等级（与下游 Self-Forcing 蒸馏配置匹配） | 0 |

**验证参数（可选）**：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--validation_steps` | 每 N 步执行一次验证 | 2000 |
| `--validation_epochs` | 每 N 个 epoch 执行一次验证 | 5 |
| `--validation_prompts` | 验证视频生成使用的提示词 | 英文提示词 |
| `--video_sample_size` | 验证采样尺寸 | 640 |
| `--video_sample_n_frames` | 验证生成的视频帧数 | 81 |
| `--fix_sample_size` | 验证使用的固定 `[高度, 宽度]` | `480 832` |

### 4.3 使用 DeepSpeed-Zero-2 / FSDP 训练

多卡训练支持与蒸馏阶段相同的显存节约后端。

**DeepSpeed-Zero-2**（推荐默认）：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME=""
export ODE_DATA_META="datasets/ode_pairs_output/outputs.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_self_forcing/train_ode.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$ODE_DATA_META \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_self_forcing_ode_regression" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.05 \
  --num_frame_per_block=3 \
  --train_sampling_steps=1000 \
  --denoising_step_indices_list 1000 750 500 250 \
  --shift=8.0 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "."
```

**FSDP**（DeepSpeed-Zero-2 显存不足时使用）：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME=""
export ODE_DATA_META="datasets/ode_pairs_output/outputs.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.1_self_forcing/train_ode.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$ODE_DATA_META \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_self_forcing_ode_regression" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.05 \
  --num_frame_per_block=3 \
  --train_sampling_steps=1000 \
  --denoising_step_indices_list 1000 750 500 250 \
  --shift=8.0 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "."
```

### 4.4 多机分布式训练

假设 2 台机器、每台 8 卡：

**机器 0（Master）**：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME=""
export ODE_DATA_META="datasets/ode_pairs_output/outputs.json"
export MASTER_ADDR="192.168.1.100"  # 主节点 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 机器总数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 本机 rank（0 或 1）
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_self_forcing/train_ode.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$ODE_DATA_META \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_self_forcing_ode_regression" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.05 \
  --num_frame_per_block=3 \
  --train_sampling_steps=1000 \
  --denoising_step_indices_list 1000 750 500 250 \
  --shift=8.0 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "."
```

**机器 1（Worker）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME=""
export ODE_DATA_META="datasets/ode_pairs_output/outputs.json"
export MASTER_ADDR="192.168.1.100"  # 与 Master 相同
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # 注意此处为 1
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# 与机器 0 使用完全相同的 accelerate launch 命令
```

**注意事项**：
- 优先使用 RDMA / InfiniBand。无 RDMA 时需设置 `NCCL_IB_DISABLE=1` 与 `NCCL_P2P_DISABLE=1`。
- 所有机器必须共享同一份 `outputs.json` 与对应的 `.safetensors` 文件（NFS / 共享存储）。

---

## 五、使用训练好的 ODE 权重

`output_dir_wan2.1_self_forcing_ode_regression/checkpoint-{N}/` 中保存的 ODE-init 权重作为 Self-Forcing 蒸馏的初始化。在 `train_distill.py` 中通过 `--ode_transformer_path` 指定即可：

```bash
# 例：保存的权重文件（如 diffusion_pytorch_model.safetensors）
--ode_transformer_path="output_dir_wan2.1_self_forcing_ode_regression/checkpoint-{N}/diffusion_pytorch_model.safetensors"
```

官方发布的对应权重为 `models/Diffusion_Transformer/Self-Forcing/checkpoints/ode_init.pt`。完整的蒸馏流程参见 [README_TRAIN.md](./README_TRAIN.md)。

---

## 六、更多资源

- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
