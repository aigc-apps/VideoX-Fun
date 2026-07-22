# Wan2.1 Causal-Forcing Stage 3: 分布匹配蒸馏 (DMD) 训练指南

本文档提供了将 Wan2.1 进行 **Causal-Forcing Stage 3 — 分布匹配蒸馏 (DMD)** 的完整工作流。

> **什么是分布匹配蒸馏？**
>
> DMD 是 Causal-Forcing 流水线的**第三阶段（最终阶段）**，使用大规模（14B）teacher 将 CCD 模型进一步蒸馏为 **少步因果自回归生成器**：
>
> 1. **Stage 1 — AR Diffusion** (`train_ar_diffusion.py`)：在干净视频 latent 上以 teacher forcing 训练模型，产出强 AR 基座。
> 2. **Stage 2 — 因果一致性蒸馏 (CCD)** (`train_causal_consistency_distill.py`)：使用 EMA teacher + CFG 将多步 AR 模型蒸馏为**每块一步**的一致性模型。
> 3. **Stage 3 — 分布匹配蒸馏 (DMD)** (`train_causal_dmd.py`)：使用 **14B real-score teacher** 的分布匹配进一步蒸馏为 **少步**生成器。
>
> 本文档仅覆盖 **Stage 3**。Stage 2 请参见 [README_TRAIN_CAUSAL_CONSISTENCY_DISTILL_zh-CN.md](./README_TRAIN_CAUSAL_CONSISTENCY_DISTILL_zh-CN.md)，Stage 1 请参见 [README_TRAIN_AR_DIFFUSION_zh-CN.md](./README_TRAIN_AR_DIFFUSION_zh-CN.md)。

---

## 目录
- [一、前置条件](#一前置条件)
- [二、环境配置](#二环境配置)
- [三、下载预训练模型](#三下载预训练模型)
- [四、数据准备](#四数据准备)
  - [4.1 快速测试数据集](#41-快速测试数据集)
  - [4.2 数据集结构](#42-数据集结构)
  - [4.3 metadata.json 格式](#43-metadatajson-格式)
- [五、训练](#五训练)
  - [5.1 快速开始](#51-快速开始)
  - [5.2 关键参数](#52-关键参数)
  - [5.3 DMD 特有参数](#53-dmd-特有参数)
- [六、使用训练好的 Checkpoint](#六使用训练好的-checkpoint)
- [七、更多资源](#七更多资源)

---

## 一、前置条件

Stage 3 需要：

1. **Stage 2 CCD checkpoint**：用于初始化生成器（和判别器）。
2. **Wan2.1-T2V-14B** 模型：作为 DMD real-score teacher。

```bash
# 示例：Stage 2 CCD 训练的 checkpoint
export STAGE2_CKPT="output_dir_wan2.1_causal_forcing_ccd/checkpoint-5000/transformer/diffusion_pytorch_model.safetensors"
```

如何产出该 checkpoint 请参见 [README_TRAIN_CAUSAL_CONSISTENCY_DISTILL_zh-CN.md](./README_TRAIN_CAUSAL_CONSISTENCY_DISTILL_zh-CN.md)。

---

## 二、环境配置

**方式 1：使用 requirements.txt**

```bash
pip install -r requirements.txt
```

**方式 2：手动安装依赖**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**方式 3：使用 Docker**

```bash
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# 进入容器
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 三、下载预训练模型

DMD 需要**两个**预训练模型：

- **Wan2.1-T2V-1.3B**：生成器/判别器的基础模型。
- **Wan2.1-T2V-14B**：非因果的 real-score teacher，用于计算真实分布得分。

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Wan2.1 T2V 1.3B（student 基础模型）
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B

# 下载 Wan2.1 T2V 14B（DMD real-score teacher）
modelscope download --model Wan-AI/Wan2.1-T2V-14B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-14B
```

Stage 2 CCD checkpoint（生成器/判别器初始化）通过 `--ode_transformer_path` 加载。

---

## 四、数据准备

DMD 使用 **TextDataset**（`train_mode="normal"`）— 只需要提示词，不需要视频数据，因为生成器通过自回归 rollout 自行创建训练样本。但仍需提供包含提示词的 `metadata.json`。

### 4.1 快速测试数据集

```bash
# 下载官方示例数据集
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 4.2 数据集结构

与 Stage 1/2 相同。详见 [README_TRAIN_CAUSAL_CONSISTENCY_DISTILL_zh-CN.md](./README_TRAIN_CAUSAL_CONSISTENCY_DISTILL_zh-CN.md#42-数据集结构)。

### 4.3 metadata.json 格式

DMD 仅使用每条记录的 `"text"` 字段。当 `--train_mode="normal"`（TextDataset 模式）时，`file_path` 和其他字段会被忽略。

```json
[
  {
    "text": "A beautiful sunset over the ocean, golden hour lighting"
  },
  {
    "text": "A person walking through a forest, cinematic view"
  }
]
```

> **说明**：你可以复用任何视频 metadata.json — DMD 只关心 `text` 字段。

---

## 五、训练

### 5.1 快速开始

可直接使用的启动脚本为 [train_causal_dmd.sh](./train_causal_dmd.sh)：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export REAL_SCORE_MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export STAGE2_CKPT="output_dir_wan2.1_causal_forcing_ccd/checkpoint-5000/transformer/diffusion_pytorch_model.safetensors"

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/wan2.1_causal_forcing/train_causal_dmd.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --real_score_pretrained_model_name_or_path=$REAL_SCORE_MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --ode_transformer_path=$STAGE2_CKPT \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --fix_sample_size 480 832 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=200 \
  --learning_rate=2.0e-06 \
  --learning_rate_critic=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_causal_forcing_dmd" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=0.0 \
  --adam_beta1=0.0 \
  --adam_beta2=0.999 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=10.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --num_frame_per_block=3 \
  --use_kv_cache_training \
  --denoising_step_indices_list 1000 667 334 1 \
  --real_guidance_scale=6.0 \
  --randomize_step_indices \
  --fake_guidance_scale=0.0 \
  --gen_update_interval=5 \
  --trainable_modules "."
```

或直接运行：

```bash
bash scripts/wan2.1_causal_forcing/train_causal_dmd.sh
```

### 5.2 关键参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--pretrained_model_name_or_path` | 基础模型（1.3B），用于生成器/判别器初始化 | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B/` |
| `--real_score_pretrained_model_name_or_path` | 14B real-score teacher | `models/Diffusion_Transformer/Wan2.1-T2V-14B` |
| `--config_path` | 模型配置 YAML | `config/wan2.1/wan_civitai.yaml` |
| `--train_data_dir` | 数据根目录 | `""` |
| `--train_data_meta` | `metadata.json` 路径（仅含提示词） | `datasets/X-Fun-Videos-Demo/metadata.json` |
| `--ode_transformer_path` | Stage 2 CCD checkpoint（生成器/判别器初始化） | `$STAGE2_CKPT` |
| `--train_batch_size` | 每 GPU batch 大小 | 1 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | 200 |
| `--learning_rate` | 生成器学习率 | 2e-06 |
| `--learning_rate_critic` | 判别器学习率 | 2e-06 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--output_dir` | 输出目录 | `output_dir_wan2.1_causal_forcing_dmd` |
| `--gradient_checkpointing` | 激活重计算 | - |
| `--mixed_precision` | 混合精度：`fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 0.0 |
| `--adam_beta1` | AdamW beta1（DMD 使用 0.0） | 0.0 |
| `--adam_beta2` | AdamW beta2 | 0.999 |
| `--adam_epsilon` | AdamW epsilon | 1e-10 |
| `--max_grad_norm` | 梯度裁剪阈值 | 10.0 |
| `--trainable_modules` | 可训练模块（`"."` = 全部） | `"."` |
| `--low_vram` | 低显存模式（卸载 VAE/文本编码器） | - |

**视频采样参数**：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--image_sample_size` | 图像采样尺寸 | 640 |
| `--video_sample_size` | 视频采样尺寸 | 640 |
| `--token_sample_size` | Token 采样尺寸 | 640 |
| `--fix_sample_size` | 固定输出 `[高度, 宽度]` | `480 832` |
| `--video_sample_stride` | 帧采样步幅 | 2 |
| `--video_sample_n_frames` | 视频帧数 | 81 |
| `--random_hw_adapt` | 启用随机分辨率适配 | - |
| `--training_with_video_token_length` | 启用基于 token 长度的训练 | - |
| `--enable_bucket` | 启用宽高比分桶采样 | - |
| `--vae_mini_batch` | VAE 编码迷你批次大小（设为 1 避免 OOM） | 1 |

### 5.3 DMD 特有参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--denoising_step_indices_list` | 去噪步骤索引（DMD 核心参数）。示例为 4 步；解析器默认 `[1000, 500]` = 2 步 | `1000 667 334 1` |
| `--real_guidance_scale` | real-score（14B teacher）的 CFG scale | 6.0 |
| `--randomize_step_indices` | 训练时随机化去噪步骤索引 | - |
| `--fake_guidance_scale` | fake-score（生成器）的 CFG scale。0.0 = 无 CFG | 0.0 |
| `--gen_update_interval` | 生成器更新间隔（每 N 步判别器更新后更新 1 次生成器） | 5 |
| `--num_frame_per_block` | 每个因果块的帧数。`3` = chunkwise（默认），`1` = 逐帧 | 3 |
| `--use_kv_cache_training` | 使用 KV 缓存逐块训练（匹配原始 Self-Forcing） | - |
| `--independent_first_frame` | 第一帧是否独立（`[1, N, N, ...]` 模式，适用于 I2V） | - |
| `--context_noise` | KV 缓存更新的上下文噪声级别 | 0 |
| `--use_teacher_forcing` | 启用 teacher forcing（将 clean_x 传给 transformer） | - |
| `--teacher_forcing_prob` | 每步应用 teacher forcing 的概率（1.0 = 始终） | 1.0 |
| `--train_mode` | 训练模式：`normal`（TextDataset，仅提示词）或 `i2v` | `normal` |
| `--resume_from_checkpoint` | 恢复训练。使用 `"latest"` 自动选择最新 checkpoint | `"latest"` |
---

## 六、使用训练好的 Checkpoint

Stage 3 DMD checkpoint 是 Causal-Forcing 流水线的**最终模型**，直接用于推理：

```python
# 在 examples/wan2.1_causal_forcing/predict_t2v.py 中
transformer_path = "output_dir_wan2.1_causal_forcing_dmd/checkpoint-{N}/diffusion_pytorch_model.safetensors"

# DMD Stage 3 推理配置
guidance_scale      = 1.0    # CFG 已烘焦到蒸馏权重中
num_inference_steps = 4      # 4 步 DMD
stochastic_sampling = True
num_frame_per_block = 3      # 分块生成
```

或运行：

```bash
python examples/wan2.1_causal_forcing/predict_t2v.py
```

> **阶段选择器**：`predict_t2v.py` 中包含阶段选择器部分，提供不同阶段的预设配置：
> - **Stage 1 (AR Diffusion)**：`guidance_scale=3.0`、`num_inference_steps=50`、`stochastic_sampling=False`
> - **Stage 2 (CCD)**：`guidance_scale=1.0`、`num_inference_steps=4`、`stochastic_sampling=True`
> - **Stage 3 (DMD)**：`guidance_scale=1.0`、`num_inference_steps=4`、`stochastic_sampling=True`

---

## 七、更多资源

- **Causal-Forcing 论文**：https://github.com/thu-ml/Causal-Forcing
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
