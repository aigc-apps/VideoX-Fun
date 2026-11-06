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
  - [5.4 可选 One-Forcing GAN 与精度控制](#54-可选-one-forcing-gan-与精度控制)
- [六、使用训练好的 Checkpoint](#六使用训练好的-checkpoint)
- [七、更多资源](#七更多资源)

---

## 一、前置条件

Stage 3 需要：

1. **Stage 2 CCD checkpoint**：用于初始化生成器（和判别器）。
2. **Wan2.1-T2V-14B** 模型：作为 DMD real-score teacher。

```bash
# Example: Stage 2 CCD checkpoint
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
# Pull the image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# Enter the container
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 三、下载预训练模型

DMD 需要**两个**预训练模型：

- **Wan2.1-T2V-1.3B**：生成器/判别器的基础模型。
- **Wan2.1-T2V-14B**：非因果的 real-score teacher，用于计算真实分布得分。

```bash
# Create the model directory
mkdir -p models/Diffusion_Transformer

# Download the Wan2.1 T2V 1.3B student base model
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B

# Download the Wan2.1 T2V 14B real-score teacher
modelscope download --model Wan-AI/Wan2.1-T2V-14B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-14B
```

Stage 2 CCD checkpoint（生成器/判别器初始化）通过 `--ode_transformer_path` 加载。

---

## 四、数据准备

当 `train_mode="normal"`、未启用 teacher forcing 且 GAN/R1/R2 权重均为零时，DMD 使用 **TextDataset**：只需要提示词，生成器通过自回归 rollout 自行创建训练样本。任一 GAN/R1/R2 权重为正时会切换为 **ImageVideoDataset**，需要带描述的真实视频，详见 5.4。

### 4.1 快速测试数据集

```bash
# Download the official demo dataset
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 4.2 数据集结构

与 Stage 1/2 相同。详见 [README_TRAIN_CAUSAL_CONSISTENCY_DISTILL_zh-CN.md](./README_TRAIN_CAUSAL_CONSISTENCY_DISTILL_zh-CN.md#42-数据集结构)。

### 4.3 metadata.json 格式

仅提示词的 TextDataset 模式只读取每条记录的 `"text"` 字段。GAN/R1/R2 或 teacher forcing 需要与 Stage 1/2 相同的视频元数据，包括 `file_path`、`type="video"` 和 `text`。

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

> **说明**：可以复用现有视频元数据。仅提示词模式只读取 `text`，GAN 训练还会读取视频文件。

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
  --learning_rate_critic=4.0e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_causal_forcing_dmd" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --fsdp_reduce_fp32 \
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
  --use_ema \
  --denoising_step_indices_list 1000 750 500 250 \
  --real_guidance_scale=4.0 \
  --score_timestep_clamp \
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
| `--train_data_meta` | 元数据路径：普通 DMD 使用提示词，GAN/R1/R2 使用带描述的视频 | `datasets/X-Fun-Videos-Demo/metadata.json` |
| `--ode_transformer_path` | Stage 2 CCD checkpoint（生成器/判别器初始化） | `$STAGE2_CKPT` |
| `--train_batch_size` | 每 GPU batch 大小 | 1 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | 200 |
| `--learning_rate` | 生成器学习率 | 2e-06 |
| `--learning_rate_critic` | 判别器学习率 | 4.0e-07 |
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
| `--use_ema` | 维护生成器权重的 EMA 副本 | - |

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
| `--denoising_step_indices_list` | 去噪步骤索引（DMD 核心参数）。示例为 4 步；解析器默认 `[1000, 500]` = 2 步 | `1000 750 500 250` |
| `--real_guidance_scale` | real-score（14B teacher）的 CFG scale | 4.0 |
| `--randomize_step_indices` | 训练时随机化去噪步骤索引 | - |
| `--fake_guidance_scale` | fake-score（生成器）的 CFG scale。0.0 = 无 CFG | 0.0 |
| `--gen_update_interval` | 生成器更新间隔（每 N 步判别器更新后更新 1 次生成器） | 5 |
| `--num_frame_per_block` | 每个因果块的帧数。`3` = chunkwise（默认），`1` = 逐帧 | 3 |
| `--use_kv_cache_training` | 使用 KV 缓存逐块训练（匹配原始 Self-Forcing） | - |
| `--independent_first_frame` | 第一帧是否独立（`[1, N, N, ...]` 模式，适用于 I2V） | - |
| `--context_noise` | KV 缓存更新的上下文噪声级别 | 0 |
| `--use_teacher_forcing` | 启用 teacher forcing（将 clean_x 传给 transformer） | - |
| `--teacher_forcing_prob` | 每步应用 teacher forcing 的概率（1.0 = 始终） | 1.0 |
| `--train_mode` | `normal`（未启用 teacher forcing 或 GAN/R1/R2 时仅提示词）或 `i2v` | `normal` |
| `--resume_from_checkpoint` | 恢复训练。使用 `"latest"` 自动选择最新 checkpoint | `"latest"` |
---

### 5.4 可选 One-Forcing GAN 与精度控制

这些选项复用 `scripts/wan2.1_self_forcing/train_distill.py` 中的 [One-Forcing](https://github.com/Aurora-edu/One-Forcing) 机制。GAN/R1/R2 权重默认均为 **0**，两个精度开关在 argparse 中默认均**关闭**。AR diffusion、CCD 和 DMD 的 launcher 均启用 `--fsdp_reduce_fp32`；GAN/R1/R2 仍关闭，DMD 的 launcher 额外启用 `--score_timestep_clamp`。AR diffusion 和 CCD 不添加 GAN 损失。

判别器共享 **fake-score backbone**，每个选定特征层对应一个可学习 register token。判别器头由 critic optimizer 更新，不加入 generator optimizer。生成器损失为 `DMD + 加权 GAN`，critic 损失为 `denoising + 加权 GAN + R1 + R2`。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--gan_g_weight` | 非饱和生成器 GAN 损失权重 | `0.0` |
| `--gan_d_weight` | 判别器 GAN 损失权重 | `0.0` |
| `--gan_feature_layers` | 判别器头使用的 fake-score block 索引 | `21 29` |
| `--gan_block_ffn_dim` | 判别器交叉注意力 FFN 维度 | `2048` |
| `--gan_block_num_heads` | 判别器头数，须整除 fake-score 隐藏维度 | `12` |
| `--relativistic_discriminator` | 使用配对真实/生成样本 logit 差值 | 关闭 |
| `--r1_weight` | 噪声真实 latent 上的 APT 形式有限差分正则权重 | `0.0` |
| `--r1_sigma` | R1 扰动标准差 | `0.01` |
| `--r2_weight` | 噪声生成 latent 上的归一化有限差分正则权重 | `0.0` |
| `--r2_sigma` | R2 扰动标准差 | `0.01` |
| `--fsdp_reduce_fp32` | FSDP 使用 FP32 梯度归约，参数仍以低精度聚合 | 关闭（launcher 已启用） |
| `--score_timestep_clamp` | 将 DMD、生成器 GAN 和 critic 时间步限制到 `[0.02, 0.98] * num_train_timesteps` | 关闭（DMD launcher 已启用） |

设 logits 为 `D_fake` 和 `D_real`，普通 GAN 的生成器损失为 `softplus(-D_fake)`，判别器损失为 `softplus(-D_real) + softplus(D_fake)`。相对论模式分别使用 `softplus(-(D_fake - D_real))` 和 `softplus(-(D_real - D_fake))`。

R1 保持源实现的 APT 形式：`r1_weight * mean((D_real_perturbed - D_real)^2)`，**不除以 sigma**。R2 为 `0.5 * r2_weight * mean(((D_fake_perturbed - D_fake) / r2_sigma)^2)`。二者都是有限差分正则，不是需要 double backward 的输入梯度惩罚。真实/生成样本共用时间步和噪声，扰动噪声独立抽样。

开启 GAN 时，向 `train_causal_dmd.py` 传入正的生成器/判别器权重，并按需配置 R1/R2。要求：
- 使用带描述的真实视频，不能只提供提示词 JSON。元数据采用 Stage 1/2 视频格式，帧数须满足 `score_num_frames` 与因果块几何。
- `--gradient_accumulation_steps=1`；GAN 分支不支持 DeepSpeed。
- 特征层索引必须存在。默认 `21 29` 与 `12` 个头适用于 Wan2.1-T2V-1.3B，其他 backbone 可能需要调整。
- 仅开启 `gan_g_weight` 不会提供判别器的分类学习目标。须明确选择判别器权重，非零权重不保证质量提升。

可训练 fake-score 与判别器头的 master weights 保持 **FP32**，前向按配置使用 autocast/FSDP 混合精度。Logits、softplus 损失及有限差分归约使用 FP32。`--fsdp_reduce_fp32` 只改变梯度通信精度，不改变前向 dtype 或模型结构。时间步 clamp 只作用于 score 训练，不改变生成器 rollout 或 validation schedule。

恢复 GAN checkpoint 时应使用相同的 GAN/判别器头配置。判别器头随 fake-score 保存，包括 FP32 判别器头张量，并在加载前重建。导出的 generator 不包含该头，推理无需判别器或新增 GAN 参数。

---

## 六、使用训练好的 Checkpoint

Stage 3 DMD checkpoint 是 Causal-Forcing 流水线的**最终模型**，直接用于推理：

```python
# In examples/wan2.1_causal_forcing/predict_t2v.py
transformer_path = "output_dir_wan2.1_causal_forcing_dmd/checkpoint-{N}/diffusion_pytorch_model.safetensors"

# DMD Stage 3 inference config
guidance_scale      = 1.0    # CFG is baked into distilled weights
num_inference_steps = 4      # 4-step DMD
stochastic_sampling = True
num_frame_per_block = 3      # Chunk-wise generation
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
- **One-Forcing（可选 GAN 机制）**：https://github.com/Aurora-edu/One-Forcing
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
