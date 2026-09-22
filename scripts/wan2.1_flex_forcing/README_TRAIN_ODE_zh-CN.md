# Wan2.1 Flex-Forcing 阶段 1：ODE 回归预训练指南

本文档介绍 Wan2.1 Flex-Forcing **阶段 1——ODE 回归**（[arXiv 2607.03509](https://arxiv.org/abs/2607.03509)）的完整流程，涵盖环境配置、ODE 轨迹对生成、ODE 回归训练。

> **什么是 ODE 回归训练？**
>
> ODE 回归是 Flex-Forcing 两阶段流程的 **预训练阶段**，整体流程分两步：
>
> 1. **第一步 — 生成 ODE 对**（`generate_ode_pairs.py`）：使用 **双向教师模型** Wan2.1-T2V-1.3B，对一组文本提示词执行完整的多步 CFG 去噪，将 ODE 轨迹上的中间 latent 与编码后的 prompt embedding 一起保存为 `.safetensors` 文件。
> 2. **第二步 — ODE 回归训练**（`train_ode.py`）：加载第一步生成的 ODE 对，训练一个 **灵活分块生成器**，在轨迹上随机抽样多个噪声等级，预测干净的终点 `x0`；每次迭代都会抽取一个帧划分（九成是全新一轮的随机划分，一成钉成 `--num_frame_per_block` 的均匀块）。训练得到的权重作为 **阶段 2 DMD 蒸馏**（`train_distill.py`，参见 [README_TRAIN_FLEX_DMD_DISTILL_zh-CN.md](./README_TRAIN_FLEX_DMD_DISTILL_zh-CN.md)）的少步初始化。
>
> Flex-Forcing 本身把 Self-Forcing 的标量 `num_frame_per_block` 换成**帧轴上的一个划分** `a = (a_0, ..., a_K)`：chunk（分块）内部是双向注意力，chunk 之间是自回归。`[1] * F` 退化为纯自回归，`[F]` 退化为纯双向注意力，中间的一切形态都由同一个模型覆盖。随机分块正是在本阶段首次引入的。

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
  - [4.3 Flex-Forcing 专属参数](#43-flex-forcing-专属参数)
  - [4.4 使用 DeepSpeed-Zero-2 / FSDP 训练](#44-使用-deepspeed-zero-2--fsdp-训练)
  - [4.5 多机分布式训练](#45-多机分布式训练)
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
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**方式 3：使用 docker**

使用 docker 时，请确保机器中已正确安装显卡驱动与 CUDA 环境，然后依次执行以下命令：

```bash
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# 进入容器
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

> 灵活注意力掩码由 **FlexAttention**（`torch.nn.attention.flex_attention`）构建。这个依赖继承自 Self-Forcing 骨干，不是本目录新增的——无需额外安装，但实际要求 `torch>=2.5`。

---

## 二、下载预训练模型

阶段 1 只需要**一个**预训练模型 **Wan2.1-T2V-1.3B**：它既是被训练的学生，也是第一步中提供 ODE 轨迹的双向教师。

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Wan2.1 T2V 1.3B（生成时作为教师，训练时作为初始化）
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B
```

然后把 `train_ode.sh` 里的 `MODEL_NAME` 指向该 1.3B 目录。

> 用 `WanTransformer3DModel_FlexForcing` 加载普通 Self-Forcing / CausVid checkpoint 时，会报告 `flex_kproj.*` 键**缺失**。这是预期行为：K-Projection（论文 §3.3，一个对 cache 做噪声级对齐的投影层）是恒等初始化的，没有它的老 checkpoint 在训练之前的行为与恒等映射完全一致。

---

## 三、第一步 — 生成 ODE 轨迹对

轨迹是**教师模型**的属性，与分块方式无关，所以这一步与 Self-Forcing 流程完全共用：使用双向教师模型对每条提示词执行 **48 步 CFG 去噪**，并将得到的 ODE 轨迹与对应的 prompt embedding 一起保存为 `.safetensors` 文件。所有提示词处理完成后，会自动生成一个 `outputs.json` 标注文件，供后续训练阶段使用。

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

直接复用启动脚本 [scripts/wan2.1_self_forcing/generate_ode_pairs.sh](../wan2.1_self_forcing/generate_ode_pairs.sh)：

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
| `--num_inference_steps` | 教师去噪步数（必须 ≥ 37，因为代码抽样的索引为 `[0,12,24,36,-1]`） | 48 |
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

第一步完成、`datasets/ode_pairs_output/outputs.json` 生成后，即可训练灵活分块生成器（`WanTransformer3DModel_FlexForcing`）来回归 ODE 轨迹。

每个训练样本上，训练脚本会：
1. 从一个 `.safetensors` 文件中加载稀疏的 5 点轨迹与 prompt embedding；
2. 抽取帧划分，并给**每个 chunk 配一个统一的时间步**（`get_timestep_for_ode_flexible`）——这正是"chunk 内部双向"的来源；其中 90% 的迭代抽取一个**全新的随机划分**（chunk 大小在 `--flex_chunk_min`..`--flex_chunk_max` 之间），另外 `UNIFORM_BLOCK_PROB = 0.1` 钉成 `--num_frame_per_block` 的均匀划分——21 个 latent 帧下随机抽取碰到 `[3]*7` 的概率只有 `1/183708`，而 `log_validation` 渲染的正是它；
3. 将带噪 latent 与逐帧时间步送入生成器，把输出的 flow 转换为 `x0` 预测，与轨迹的**干净终点**计算 MSE 损失。

### 4.1 快速开始

直接复用启动脚本 [train_ode.sh](./train_ode.sh)：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME=""
export ODE_DATA_META="datasets/ode_pairs_output/outputs.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/wan2.1_flex_forcing/train_ode.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$ODE_DATA_META \
  --fix_sample_size 432 832 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_ode_regression" \
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about." \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.05 \
  --train_sampling_steps=1000 \
  --denoising_step_indices_list 1000 750 500 250 \
  --shift=8.0 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --num_frame_per_block=3
```

或者直接执行：

```bash
bash scripts/wan2.1_flex_forcing/train_ode.sh
```

> 💡 因为 ODE 轨迹与 prompt embedding 已在第一步预先计算完毕，**ODE 训练阶段不会再调用 VAE / 文本编码器**，训练速度快、显存占用低。当 `outputs.json` 中已经使用绝对路径时，`train_data_dir` 可以留空。

> 阶段 1 带 `--flex_forcing` 与 `--flex_chunk_min/max`，但**不带** `--flex_pyramid_levels`：金字塔是**多步**调度，而 ODE 阶段对每个 block 只做一次前向来回归 `x0`，没有可以逐级细化的调度。金字塔时间步在阶段 2 才出现。

输出目录：`output_dir_wan2.1_flex_forcing_ode_regression/`。

### 4.2 训练常用参数

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `--pretrained_model_name_or_path` | 用于初始化生成器的基础模型（1.3B） | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `--config_path` | 模型配置 YAML | `config/wan2.1/wan_civitai.yaml` |
| `--train_data_dir` | 拼接到 `file_path` 之前的可选根目录；若 `outputs.json` 已使用绝对路径可留空 | `""` |
| `--train_data_meta` | 第一步生成的标注 JSON | `datasets/ode_pairs_output/outputs.json` |
| `--fix_sample_size` | 固定 `H W`；论文的 5 秒片段是 432×832 | `432 832` |
| `--train_batch_size` | 每卡 batch size | 1 |
| `--gradient_accumulation_steps` | 8 卡 × 8 = 论文的 batch 64 | 8 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存一次 checkpoint | 500 |
| `--learning_rate` | 初始学习率 | 2.0e-06 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--seed` | 随机种子 | 42 |
| `--output_dir` | 输出目录 | `output_dir_wan2.1_flex_forcing_ode_regression` |
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
| `--train_sampling_steps` | `denoising_step_indices_list` 所索引的调度器网格大小 | 1000 |
| `--denoising_step_indices_list` | ODE 回归使用的离散时间步索引（与第一步抽样的 5 个稀疏轨迹点对应）。直接读调度器自带的网格，而不是 `set_timesteps()` 重建的那张，所以 `shift=8.0` 下映射为 t = `[1000, 960, 888.89, 727.27]` —— 与 `stochastic_sampling_timesteps` 喂给校验和推理的值逐位一致。重建的那张最多偏 `5.44`，因为它的网格末端 `sigma_min` 不同。 | `1000 750 500 250` |
| `--shift` | `FlowMatchEulerDiscreteScheduler` 的 shift —— **必须与第一步生成时使用的 `--shift` 一致** | 8.0 |
| `--num_frame_per_block` | 继承路径的均匀 fallback；`UNIFORM_BLOCK_PROB = 0.1` 的迭代会把划分钉成它，使该布局是真正被训练而不只是理论上可达；也是 `log_validation` 采样验证时用的固定划分，保证 checkpoint 之间可比较 | 3 |
| `--independent_first_frame` | 第一帧是否独立（`[1, N, N, ...]` 块模式） | - |
| `--context_noise` | 上下文噪声等级（与下游蒸馏配置匹配） | 0 |

**验证参数（可选）**：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--validation_steps` | 每 N 步执行一次验证 | 100 |
| `--validation_epochs` | 每 N 个 epoch 执行一次验证 | 500 |
| `--validation_prompts` | 验证视频生成使用的提示词 | 英文提示词 |

### 4.3 Flex-Forcing 专属参数

| 参数 | 论文 | 说明 | 默认值 |
|-----------|-------|-------------|---------|
| `--flex_forcing` | §3.1 | 实例化 `WanTransformer3DModel_FlexForcing` 并用 `WanFlexForcingPipeline` 校验。关闭 = 走继承的原路径，行为不变。 | 关 |
| `--flex_chunk_min` | §3.1 | 每次迭代抽取的最小 chunk 大小。`1` 合法，会让单帧（严格因果）chunk 出现在片段中间；`2` 是论文取值，也能避免采样器把尾部退化成 1 帧的碎块。 | 2 |
| `--flex_chunk_max` | §3.1 | 每次迭代抽取的最大 chunk 大小。设为与 `--flex_chunk_min` 相等即固定一种布局。与阶段 2 不同，这里没有粗粒度混合机制，所以把这个值抬到 latent 帧数是阶段 1 **唯一**能把“整段作为一个完全双向 chunk”回归出来的途径（21 个 latent 帧时约占 5% 的抽取）；低于帧数则该布局永远不会被抽到。 | 10 |

> §3.3 的 K-Projection 同样**不是 launcher 参数**：变体由模型配置（`transformer_additional_kwargs.flex_kproj_mode`，默认 `diag_rank1`）给出，要消融就在那里改成 `none`。它是恒等初始化的，所以阶段 1 直接带着它一起训，和其他层同速、跟 `--learning_rate`。

> 每张卡必须训练同一种布局，否则 FlexAttention 掩码与其派生的 `num_frame_per_block` 在 SP/FSDP 组之间会不一致。trainer 在使用抽取的划分前会先经 `broadcast_chunk_sizes` 广播对齐。

### 4.4 使用 DeepSpeed-Zero-2 / FSDP 训练

多卡训练支持与 Self-Forcing 各阶段相同的显存节约后端。上面的快速开始已经使用 **FSDP**；如需改用 **DeepSpeed-Zero-2**，只需替换启动前缀：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME=""
export ODE_DATA_META="datasets/ode_pairs_output/outputs.json"
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_flex_forcing/train_ode.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$ODE_DATA_META \
  --fix_sample_size 432 832 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_ode_regression" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.05 \
  --train_sampling_steps=1000 \
  --denoising_step_indices_list 1000 750 500 250 \
  --shift=8.0 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --num_frame_per_block=3
```

### 4.5 多机分布式训练

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

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_flex_forcing/train_ode.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$ODE_DATA_META \
  --fix_sample_size 432 832 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_ode_regression" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.05 \
  --train_sampling_steps=1000 \
  --denoising_step_indices_list 1000 750 500 250 \
  --shift=8.0 \
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --num_frame_per_block=3
```

**机器 1（Worker）**：使用与机器 0 完全相同的命令，仅把 `RANK` 改为 `1`。

**注意事项**：
- 优先使用 RDMA / InfiniBand。无 RDMA 时需设置 `NCCL_IB_DISABLE=1` 与 `NCCL_P2P_DISABLE=1`。
- 所有机器必须共享同一份 `outputs.json` 与对应的 `.safetensors` 文件（NFS / 共享存储）。

---

## 五、使用训练好的 ODE 权重

阶段 1 的产出是阶段 2 的少步初始化，不是最终模型。把 checkpoint 路径填进 [train_distill.sh](./train_distill.sh) 的 `--ode_transformer_path`：

```bash
# 例：保存的权重文件（如 diffusion_pytorch_model.safetensors）
--ode_transformer_path="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

然后按 [README_TRAIN_FLEX_DMD_DISTILL_zh-CN.md](./README_TRAIN_FLEX_DMD_DISTILL_zh-CN.md) 继续。

---

## 六、更多资源

- **Flex-Forcing 论文**：https://arxiv.org/abs/2607.03509
- **阶段 2——DMD 蒸馏**：[README_TRAIN_FLEX_DMD_DISTILL_zh-CN.md](./README_TRAIN_FLEX_DMD_DISTILL_zh-CN.md)
- **共用的 ODE 对生成**：[README_TRAIN_ODE_zh-CN.md](../wan2.1_self_forcing/README_TRAIN_ODE_zh-CN.md)
- **共享分块 helper**：`videox_fun/utils/flex_chunking.py`
- **模型 / pipeline**：`videox_fun/models/wan_transformer3d_flex_forcing.py`、`videox_fun/pipeline/pipeline_wan_flex_forcing.py`
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
