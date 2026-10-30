# Wan2.1 One-Forcing：稳定的单步自回归视频生成

本文档提供在 Wan2.1-T2V-1.3B 上进行 **One-Forcing** 的完整工作流，该配方移植自 [Aurora-edu/One-Forcing](https://github.com/Aurora-edu/One-Forcing)。

> **什么是 One-Forcing？**
>
> One-Forcing（*Towards Stable One-Step Autoregressive Video Generation*）= **Causal-Forcing Stage 3 — 分布匹配蒸馏 (DMD)** *加上*一个**共享噪声 latent 的对抗判别器（critic）**。热启动生成器由 Causal-Forcing 的前两个阶段产出，One-Forcing 再将其打磨为**稳定的单步、逐帧自回归**视频模型：
>
> 1. **Stage 1 — AR Diffusion**（`scripts/wan2.1_causal_forcing/train_ar_diffusion.py`）：在干净视频 latent 上以 **teacher forcing** 训练模型，学习按块（或按帧）因果自回归地去噪，产出强 AR 基座。参见 [README_TRAIN_AR_DIFFUSION_zh-CN.md](../wan2.1_causal_forcing/README_TRAIN_AR_DIFFUSION_zh-CN.md)。
> 2. **Stage 2 — 因果一致性蒸馏 (CCD)**（`scripts/wan2.1_causal_forcing/train_causal_consistency_distill.py`）：使用 EMA teacher + CFG 将多步 AR 模型蒸馏为**每块一步**的一致性模型，其产出 checkpoint 即 One-Forcing 的**热启动生成器**。参见 [README_TRAIN_CAUSAL_CONSISTENCY_DISTILL_zh-CN.md](../wan2.1_causal_forcing/README_TRAIN_CAUSAL_CONSISTENCY_DISTILL_zh-CN.md)。
> 3. **Stage 3 — One-Forcing**（`train_one_forcing.py`，即本文档），在 DMD 之上增加两个机制：
>    - **DMD 核心**：生成器通过 14B real-score teacher 与 1.3B fake-score critic 匹配真实分布（`train_causal_dmd.py`）。
>    - **对抗 critic**：判别器头挂在 **fake-score backbone** 上（每个选定特征层对应一个 register token）。DMD 去噪损失与对抗损失作用在**同一个噪声生成 latent** 上，critic 对真实/生成样本使用**同一份共享噪声与时间步**。
>    - **单步**：`denoising_step_list = [1000]` —— 每帧在单个去噪步内生成（即 One-Forcing 中的 “One”）。
>
> 你也可以不自己训练 Stage 1/2，而是直接使用 One-Forcing 作者发布的**官方 ODE 初始化 checkpoint** 作为热启动权重（见第三节）。
>
> **训练器** `scripts/wan2.1_one_forcing/train_one_forcing.py` 实现了 One-Forcing 的生成器/critic GAN 损失、相对论（relativistic）选项以及有限差分 R1/R2 正则。底层 DMD 阶段请参见 [README_TRAIN_CAUSAL_DMD_DISTILL_zh-CN.md](../wan2.1_causal_forcing/README_TRAIN_CAUSAL_DMD_DISTILL_zh-CN.md)。

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
  - [5.2 配方参数](#52-配方参数)
  - [5.3 GAN 前置要求](#53-gan-前置要求)
  - [5.4 可选正则（相对论 / R1 / R2）](#54-可选正则相对论--r1--r2)
- [六、推理](#六推理)
  - [6.1 快速开始](#61-快速开始)
  - [6.2 单步与 4 步选择器](#62-单步与-4-步选择器)
  - [6.3 首帧调度](#63-首帧调度)
- [七、Checkpoint 往返（训练→推理）](#七checkpoint-往返训练推理)
- [八、更多资源](#八更多资源)

---

## 一、前置条件

One-Forcing 是在已蒸馏生成器之上的最后打磨，因此需要：

1. **热启动生成器 checkpoint** —— 一个已蒸馏/ODE 生成器，由 DMD + 对抗 critic 进一步打磨（例如 **Causal-Forcing Stage-2 CCD** checkpoint，或第三节的官方 ODE 初始化 checkpoint）。
2. **Wan2.1-T2V-14B** 模型，作为 DMD real-score teacher。
3. **带描述的真实视频** —— 对抗 critic 需要干净的真实 latent，因此仅有提示词的 JSON **不够用**（这一点与普通 DMD 不同）。

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

One-Forcing 需要**两个**预训练模型：

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

One-Forcing 官方 checkpoint 托管在 HuggingFace（`JiaqiFeng/OneForcing`）：

```bash
# 训练用热启动生成器（官方 ODE 初始化 checkpoint），通过 --ode_transformer_path 加载。
hf download JiaqiFeng/OneForcing checkpoints/framewise/causal_ode.pt --local-dir models/Diffusion_Transformer/One-Forcing

# 推理用已训练 One-Forcing 生成器（见第六节）。
hf download JiaqiFeng/OneForcing checkpoints/one_forcing.pt --local-dir models/Diffusion_Transformer/One-Forcing
```

如果你自己训练了 Causal-Forcing Stage 1/2，也可以将 `--ode_transformer_path` 指向你产出的 Stage-2 CCD checkpoint。

---

## 四、数据准备

One-Forcing 默认开启 GAN 分支（`--gan_g_weight=0.03 --gan_d_weight=0.03`），因此数据加载始终使用 **ImageVideoDataset**：对抗 critic 需要**带描述的真实视频**来构建干净 latent，仅有提示词的 JSON **不够用**（这一点与普通 DMD 不同）。

### 4.1 快速测试数据集

```bash
# Download the official demo dataset
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 4.2 数据集结构

与 Causal-Forcing Stage 1/2 相同。详见 [README_TRAIN_CAUSAL_CONSISTENCY_DISTILL_zh-CN.md](../wan2.1_causal_forcing/README_TRAIN_CAUSAL_CONSISTENCY_DISTILL_zh-CN.md#42-数据集结构)。

### 4.3 metadata.json 格式

每条记录需要 `file_path`、`type="video"` 与 `text`；启动脚本使用 `metadata_add_width_height.json` 变体，额外携带 `width`/`height`，分桶训练无需现场解码媒体文件：

```json
[
  {
    "file_path": "train/00000000.mp4",
    "text": "A young woman gently turns her head to the right, tilting it slightly as she offers a soft greeting...",
    "type": "video",
    "width": 832,
    "height": 480
  }
]
```

> **说明**：每条视频须有足够的帧数以满足 `--video_sample_stride=1` 与 `--video_sample_n_frames=81`：当源视频产出的 latent 帧数少于 `score_num_frames=21` 时，GAN 分支会中止（见 5.3）。

---

## 五、训练

### 5.1 快速开始

可直接使用的启动脚本为 [train_one_forcing.sh](./train_one_forcing.sh)。它调用 One-Forcing 训练器（`train_one_forcing.py`）并开启 GAN 配方：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export REAL_SCORE_MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export STAGE2_CKPT="models/Diffusion_Transformer/One-Forcing/checkpoints/framewise/causal_ode.pt"

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/wan2.1_one_forcing/train_one_forcing.py \
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
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --max_train_steps=200 \
  --checkpointing_steps=100 \
  --learning_rate=1.0e-05 \
  --learning_rate_critic=1.0e-05 \
  --lr_scheduler="constant" \
  --seed=42 \
  --output_dir="output_dir_wan2.1_one_forcing_framewise_1step" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --fsdp_reduce_fp32 \
  --adam_weight_decay=0.01 \
  --adam_beta1=0.0 \
  --adam_beta2=0.999 \
  --adam_epsilon=1e-08 \
  --vae_mini_batch=1 \
  --max_grad_norm=10.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --num_frame_per_block=1 \
  --use_kv_cache_training \
  --use_ema \
  --ema_decay=0.99 \
  --denoising_step_indices_list 1000 \
  --real_guidance_scale=6.0 \
  --score_timestep_clamp \
  --fake_guidance_scale=0.0 \
  --gen_update_interval=5 \
  --gan_g_weight=0.03 \
  --gan_d_weight=0.03 \
  --gan_feature_layers 21 29 \
  --gan_block_ffn_dim=2048 \
  --gan_block_num_heads=12 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "."
```

或直接运行：

```bash
bash scripts/wan2.1_one_forcing/train_one_forcing.sh
```

### 5.2 配方参数

One-Forcing 配方设置以下 launcher 参数：

| 参数 | Launcher 参数 | 取值 |
|--------------------|---------------|------|
| `denoising_step_list: [1000]` | `--denoising_step_indices_list` | `1000`（单步） |
| `num_frame_per_block: 1` | `--num_frame_per_block` | `1`（逐帧） |
| `num_training_frames: 21` / `score_num_frames: 21` | `--video_sample_n_frames=81` @ 480×832 | 21 个 latent 帧 |
| `guidance_scale: 5.0` | `--real_guidance_scale` | `6.0`（见下方 CFG 说明） |
| `lr: 1e-5` / `lr_critic: 1e-5` | `--learning_rate` / `--learning_rate_critic` | `1.0e-05` |
| `beta1: 0.0`、`beta2: 0.999`、`weight_decay: 0.01` | `--adam_beta1`、`--adam_beta2`、`--adam_weight_decay` | `0.0`、`0.999`、`0.01` |
| `ema_weight: 0.99` | `--use_ema`、`--ema_decay` | 开启、`0.99` |
| `max_steps: 200` | `--max_train_steps` | `200` |
| `dfake_gen_update_ratio: 5` | `--gen_update_interval` | `5` |
| `gan_feature_layers: [21, 29]` | `--gan_feature_layers` | `21 29` |
| `gan_block_ffn_dim: 2048` | `--gan_block_ffn_dim` | `2048` |
| `gan_block_num_heads: 12` | `--gan_block_num_heads` | `12` |
| `gan_g_weight: 0.03`、`gan_d_weight: 0.03` | `--gan_g_weight`、`--gan_d_weight` | `0.03`、`0.03` |
| `relativistic_discriminator: false` | `--relativistic_discriminator` | 不传（关闭） |
| `r1_weight: 0.0`、`r2_weight: 0.0` | `--r1_weight`、`--r2_weight` | 不传（关闭） |
| `context_noise: 0` | `--context_noise` | `0`（默认） |

> **CFG 约定**：本仓库的引导计算为 `cond + scale * (cond - uncond)`，本配方取 `--real_guidance_scale=6.0`。

与 DMD 阶段共用的通用训练/视频采样参数，参见 [README_TRAIN_CAUSAL_DMD_DISTILL_zh-CN.md](../wan2.1_causal_forcing/README_TRAIN_CAUSAL_DMD_DISTILL_zh-CN.md#52-关键参数) 5.2 节。

### 5.3 GAN 前置要求

GAN 分支在 `train_one_forcing.py` 启动时会被校验，要求：

- **带描述的真实视频**（`type="video"` + `file_path` + `text`），不能只提供提示词 JSON。任一 GAN/R1/R2 权重为正时，数据加载会从 `TextDataset` 切换为 `ImageVideoDataset`。
- **`--gradient_accumulation_steps=1`** 且**不使用 DeepSpeed** —— 否则训练器抛出 `ValueError`。
- **`--video_sample_stride=1`**：当 `stride=2` 时，81 帧源仅得到约 10 个 latent 帧，低于 `score_num_frames=21` 而中止。
- fake-score backbone 上**有效的特征层索引**。默认 `21 29`（层）与 `12`（头）适用于 Wan2.1-T2V-1.3B；头数须整除 fake-score 隐藏维度。

可训练的 fake-score 与判别器头 master weights 保持 **FP32**；前向按配置使用 autocast/FSDP 混合精度，logits/softplus/有限差分归约在 FP32 下计算。`--fsdp_reduce_fp32` 只改变梯度通信精度。

### 5.4 可选正则（相对论 / R1 / R2）

这些正则默认**关闭**。若要实验，向 launcher 追加：

```bash
# Relativistic discriminator: use paired real/fake logit differences
--relativistic_discriminator

# Finite-difference gradient penalties (APT form, not double-backward)
--r1_weight=1.0 --r1_sigma=0.1     # on noisy real latents:   r1_weight * mean((D_real' - D_real)^2)
--r2_weight=0.001 --r2_sigma=0.01  # on noisy gen latents: 0.5 * r2_weight * mean(((D_fake' - D_fake)/sigma)^2)
```

设 logits 为 `D_fake` 与 `D_real`，普通 GAN 损失为 `softplus(-D_fake)`（生成器）与 `softplus(-D_real) + softplus(D_fake)`（判别器）；相对论模式改用 logit 差值。真实/生成样本共用时间步与噪声，扰动噪声独立抽样。

---

## 六、推理

### 6.1 快速开始

推理入口为 [examples/wan2.1_one_forcing/predict_t2v.py](../../examples/wan2.1_one_forcing/predict_t2v.py)。它加载 **1.3B 学生模型**（14B 仅在训练时作为 DMD teacher），运行逐帧单步生成。默认使用 **One-Forcing 官方 checkpoint**：

```bash
hf download JiaqiFeng/OneForcing checkpoints/one_forcing.pt --local-dir models/Diffusion_Transformer/One-Forcing
```

```python
# examples/wan2.1_one_forcing/predict_t2v.py
model_name          = "models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
transformer_path    = "models/Diffusion_Transformer/One-Forcing/checkpoints/one_forcing.pt"
use_ema             = False   # only applies to trainer-output checkpoint dirs
stochastic_sampling = True    # DMD-distilled student
guidance_scale      = 1.0     # CFG is baked into the distilled weights
num_inference_steps = 1       # One-Forcing one-step generation
num_frame_per_block = 1       # frame-wise
sample_size         = [480, 832]
video_length        = 81
```

运行：

```bash
python examples/wan2.1_one_forcing/predict_t2v.py
```

`transformer_path` 既可以是官方 checkpoint 文件、纯 `.pt`/`.safetensors` 文件，也可以是训练输出的 **checkpoint 目录**（其 live/EMA 权重会被自动解析）。GAN 判别器头是 fake_score 的训练专用子模块，因此推理**无需任何判别器接线**。

### 6.2 单步与 4 步选择器

当 `stochastic_sampling=True` 时，`num_inference_steps` 决定 warped 时间步调度：

| `num_inference_steps` | 去噪时间步 | 适用场景 |
|-----------------------|------------|----------|
| `1`（One-Forcing 默认） | `[1000]` | 最快；逐帧单步生成 |
| `4` | `[1000, 750, 500, 250]` | 更稳定、更慢 |

### 6.3 首帧调度

共享的 `WanSelfForcingPipeline` 对**所有块使用同一份去噪调度**，因此此处不单独为首帧设置调度（例如首帧 `[1000, 750, 500, 250]`、后续帧 `[1000]`）；该 pipeline 被多个 example 共用，有意保持不变。

---

## 七、Checkpoint 往返（训练→推理）

FSDP 训练器将每个 `checkpoint-{step}/` 按如下布局写出，因此训练产物可直接喂给推理，无需改动 key：

| 组件 | 保存位置 | 是否含 `gan_head`？ |
|------|----------|---------------------|
| **生成器**（live） | `checkpoint-{step}/diffusion_pytorch_model.safetensors`（根目录，经 FSDP hook） | 否 |
| **生成器**（EMA） | `checkpoint-{step}/transformer_ema/` | 否 |
| **Fake-score + 判别器头** | `checkpoint-{step}/fake_score/` | 是（FP32 判别器头张量） |

- **生成器从不包含 `gan_head`** 键（判别器头挂在 fake-score 上），因此 `predict_t2v.py` 可用 `strict=False` 干净加载。
- 推理脚本中的 `_resolve_transformer_path` 依次解析：`transformer_ema/`、`ema_transformer/`（兼容别名）、`transformer/`，最后到根目录 safetensors。
- **恢复** GAN 训练时须使用相同的 GAN/判别器头配置：加载前会根据 CLI 参数重建判别器头，并从 `fake_score/` 还原。

---

## 八、更多资源

- **One-Forcing（源配方）**：https://github.com/Aurora-edu/One-Forcing
- **Causal-Forcing（DMD 阶段）**：https://github.com/thu-ml/Causal-Forcing
- **Stage 1（AR Diffusion）训练指南**：[README_TRAIN_AR_DIFFUSION_zh-CN.md](../wan2.1_causal_forcing/README_TRAIN_AR_DIFFUSION_zh-CN.md)
- **Stage 2（CCD）训练指南**：[README_TRAIN_CAUSAL_CONSISTENCY_DISTILL_zh-CN.md](../wan2.1_causal_forcing/README_TRAIN_CAUSAL_CONSISTENCY_DISTILL_zh-CN.md)
- **底层 DMD 训练器指南**：[README_TRAIN_CAUSAL_DMD_DISTILL_zh-CN.md](../wan2.1_causal_forcing/README_TRAIN_CAUSAL_DMD_DISTILL_zh-CN.md)
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
