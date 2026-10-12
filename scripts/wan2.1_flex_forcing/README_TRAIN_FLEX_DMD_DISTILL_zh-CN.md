# Wan2.1 Flex-Forcing 阶段 2：DMD 蒸馏训练指南

本文档给出在 Wan2.1-T2V-1.3B 上运行 **Flex-Forcing 阶段 2——分布匹配蒸馏（DMD）**（[arXiv 2607.03509](https://arxiv.org/abs/2607.03509)）的完整流程。

> **Flex-Forcing 是什么？**
>
> Self-Forcing 用一个标量 `num_frame_per_block` 固定整个模型的因果结构，所以一个 checkpoint 要么是分块因果的、要么是双向的，不能两者兼得。Flex-Forcing 把这个标量换成**帧轴上的一个划分** `a = (a_0, ..., a_K)`：chunk（分块）内部是双向注意力，chunk 之间是自回归。`[1] * F` 退化为纯自回归，`[F]` 退化为纯双向注意力，中间的一切形态都由同一个模型覆盖。
>
> 三个想法支撑起这一点，每个都对应下文的一个参数：
>
> 1. **§3.1 灵活帧分块** —— 划分本身是一等输入，而不是构建期常量。
> 2. **§3.2 金字塔时间步分块** —— 完成步 `t+1` 后，步 `t` 的划分是在**保留已有边界的前提下插入新边界**细化的。步 `t` 的结果先在整个原 chunk 上缓冲，随后每个子 chunk 在其所需 KV 可用后逐个自回归续写。先粗规划、后细刻画。
> 3. **§3.3 K-Projection** —— 干净上下文的 key 与时间步 `t` 的含噪 query 处在不同的空间，因此用一个恒等初始化、以时间步为条件的投影把 cache 映射进当前含噪 latent 空间。它即时施加、从不改写 cache，并与生成器一起训练。
>
> Flex-Forcing 流程分两个阶段：
>
> 1. **阶段 1——ODE 回归**（`train_ode.py`）：对教师模型的 ODE 轨迹做回归，让模型先成为一个合格的小步数生成器。
> 2. **阶段 2——DMD 蒸馏**（`train_distill.py`，即本 README）：用 **Wan2.1-T2V-14B** 真实分数教师蒸馏出 **4 步**生成器，并加入金字塔时间步调度。
>
> 本 README 只覆盖**阶段 2**。阶段 1 见 [README_TRAIN_ODE_zh-CN.md](./README_TRAIN_ODE_zh-CN.md)。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 metadata.json 格式](#22-metadatajson-格式)
- [三、蒸馏训练](#三蒸馏训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始（DeepSpeed-Zero-2）](#32-快速开始deepspeed-zero-2)
  - [3.3 训练常用参数解析](#33-训练常用参数解析)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 使用 FSDP 训练](#35-使用-fsdp-训练)
  - [3.6 其他后端](#36-其他后端)
  - [3.7 多机分布式训练](#37-多机分布式训练)
- [四、推理测试](#四推理测试)
  - [4.1 推理参数解析](#41-推理参数解析)
  - [4.2 文生视频（T2V）推理](#42-文生视频t2v推理)
  - [4.3 多卡并行推理](#43-多卡并行推理)
  - [4.4 任意序视频编辑](#44-任意序视频编辑)
- [五、更多资源](#五更多资源)

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

> 两个阶段的注意力掩码都用 **FlexAttention**（`torch.nn.attention.flex_attention`）构建。这个依赖继承自 Self-Forcing 骨干，不是本目录新增的——无需额外安装，但实际要求 `torch>=2.5`。

---

## 二、数据准备

### 2.1 快速测试数据集

我们提供了一个测试的数据集，其中包含若干训练数据。

```bash
# 下载官方示例数据集
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 2.2 metadata.json 格式

DMD 使用 **TextDataset**（`--train_mode="normal"`）——生成器通过 rollout 自己制造训练样本，所以每个条目只读 `"text"` 字段，任何视频元数据都能当 prompt 列表：

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

`train_distill.sh` 默认指向 `datasets/internal_datasets/metadata.json`。真实训练请换成论文使用的 **VidProM extended** prompt 集。

---

## 三、蒸馏训练

### 3.1 下载预训练模型

阶段 2 需要**两个**预训练模型：

- **Wan2.1-T2V-1.3B**：生成器 / critic 的底座模型。
- **Wan2.1-T2V-14B**：DMD 用于真实分布分数的非因果真实分数教师。

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Wan2.1 T2V 1.3B（学生底座模型）
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B

# 下载 Wan2.1 T2V 14B（DMD 真实分数教师）
modelscope download --model Wan-AI/Wan2.1-T2V-14B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-14B
```

阶段 2 还需要一个**阶段 1 ODE checkpoint**，用于初始化生成器（与 critic）：

```bash
# 示例：来自 ODE 回归的阶段 1 checkpoint
export STAGE1_CKPT="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

阶段 1 的产出方式见 [README_TRAIN_ODE_zh-CN.md](./README_TRAIN_ODE_zh-CN.md)。然后把 `train_distill.sh` 里的 `MODEL_NAME` 指向该 1.3B 目录。

> 用 `WanTransformer3DModel_FlexForcing` 加载普通 Self-Forcing / CausVid checkpoint 时，会报告 `flex_kproj.*` 键**缺失**。这是预期行为：§3.3 的 K-Projection 是恒等初始化的，没有它的老 checkpoint 在训练之前的行为与恒等映射完全一致。

### 3.2 快速开始（DeepSpeed-Zero-2）

如果按照 **2.1 快速测试数据集** 与 **3.1 下载预训练模型** 准备好数据与权重后，直接复制快速开始的启动指令进行启动。

推荐使用DeepSpeed-Zero-2与FSDP方案进行训练。这里使用DeepSpeed-Zero-2为例配置shell文件。

本文中DeepSpeed-Zero-2与FSDP的差别在于是否对模型权重进行分片，**如果使用多卡且使用DeepSpeed-Zero-2的情况下显存不足**，可以切换使用FSDP进行训练。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_flex_forcing/train_distill.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --fix_sample_size 480 832 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --score_num_frames=21 \
  --video_repeat=1 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_distill" \
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
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --flex_pyramid_levels=4 \
  --flex_min_num_frame_per_block=1 \
  --flex_self_generated_context \
  --denoising_step_indices_list 1000 750 500 250 \
  --randomize_step_indices \
  --flow_euler_rollout \
  --use_kv_cache_training \
  --num_frame_per_block=3 \
  --ode_transformer_path="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

launcher 钉死的是：81 个像素帧 = **21 个 latent 帧**（一段 5 秒片段）、分辨率 **480×832**、**4 步**模型（`--denoising_step_indices_list 1000 750 500 250`，并由 `--randomize_step_indices` 每次迭代抖动）、`--flex_pyramid_levels=4` —— 在 21 个 latent 帧上正好复现论文的阶梯 `[[21], [11, 10], [6, 5, 5, 5], [3, 3, 3, 2, 3, 2, 3, 2]]` —— 随机档的划分抽取范围 **2..10**（该档只占 10% 的迭代；另有 60% 以 §3.2 的整段粗粒度布局起步，见下）、`--flow_euler_rollout`、`diag_rank1` K-Projection、batch 8（8 卡 × `--gradient_accumulation_steps=1`）。

输出目录：`output_dir_wan2.1_flex_forcing_distill/`。

### 3.3 训练常用参数解析

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|-----------|-------------|---------------|
| `--config_path` | 组件子路径 / `transformer_additional_kwargs` | `config/wan2.1/wan_civitai.yaml` |
| `--pretrained_model_name_or_path` | 底座模型目录（1.3B） | `$MODEL_NAME` |
| `--train_data_meta` | prompt JSON（只读 `"text"` 字段） | `$DATASET_META_NAME` |
| `--ode_transformer_path` | 用于生成器 / critic 初始化的阶段 1 ODE checkpoint | `$STAGE1_CKPT` |
| `--fix_sample_size` | 固定训练片段的 `H W` | `480 832` |
| `--video_sample_n_frames` | 像素帧数；81 → 21 个 latent 帧 | 81 |
| `--score_num_frames` | 真实分数教师看到的 latent 帧数 | 21 |
| `--train_batch_size` | 单卡 batch 大小 | 1 |
| `--gradient_accumulation_steps` | 8 卡 × 1 = batch 8 | 1 |
| `--dataloader_num_workers` | DataLoader 工作进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数（被 `--max_train_steps` 覆盖） | 100 |
| `--max_train_steps` | 硬性步数上限；论文用 600 | 600 |
| `--checkpointing_steps` | 每 N 步保存一次 checkpoint | 50 |
| `--learning_rate` | 生成器学习率 | `2e-06` |
| `--learning_rate_critic` | critic 学习率 | `2e-06` |
| `--lr_scheduler` | 学习率调度器类型 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--real_guidance_scale` | 真实分数（14B 教师）的 CFG scale | 4.5 |
| `--fake_guidance_scale` | fake 分数（生成器）的 CFG scale。0.0 = 不做 CFG | 0.0 |
| `--gen_update_interval` | 生成器每 N 个 critic 步更新一次 | 5 |
| `--denoising_step_indices_list` | 去噪步索引（DMD 核心参数）。经 `timesteps[train_sampling_steps - idx]` 映射，所以 `shift=5.0` 下 `1000 750 500 250` 变成 t = `[1000, 937.5, 833.33, 625]` —— 与 `stochastic_sampling_timesteps(4, shift)` 喂给 pipeline 的值逐位一致。 | `1000 750 500 250` |
| `--randomize_step_indices` | 每次迭代对除首个索引以外的所有索引做对称抖动，幅度为相邻间隔的 `--index_jitter_ratio`。期望调度不变，单调性由构造保证，所以推理用的固定 t 始终落在每一步的训练包络内。 | 关 |
| `--index_jitter_ratio` | 抖动预算（相邻间隔的比例）。在 `1000 750 500 250` 且取 0.3 时，20000 次抽取给出 idx `[675,825] / [425,575] / [175,325]`，即 t `[912.2,959.3] / [787.0,871.2] / [514.7,706.5]`；推理点 937.5 / 833.33 / 625 全部落在其中。 | 0.3 |
| `--flow_euler_rollout` | 生成器自滚用 flow 空间里的确定性 Euler ODE 步推进（`x_next = x_t + (sigma_next - sigma_t) * v`，fp32），而不是转成 x0 再用新噪声重加噪。代数上这**就等于**用同一个隐含噪声重加噪 —— 在精确算术下与之完全相同，而重采新噪声是一次独立抽样 —— 因此它消掉了 rollout 的随机性，与正常的 flow-matching 推理一致。最后一步仍然转 x0，因为 DMD 目标定义在 x0 上。critic 重滚生成器，跟随同一开关。 | 关 |
| `--output_dir` | 输出目录 | `output_dir_wan2.1_flex_forcing_distill` |
| `--gradient_checkpointing` | 用算力换显存 | - |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--trainable_modules` | 可训练组的子串匹配（`"."` = 全部） | `"."` |
| `--train_mode` | `normal`（TextDataset，纯 prompt） | `normal` |
| `--resume_from_checkpoint` | `"latest"` 自动选择 | `"latest"` |

**Flex-Forcing 专属参数**：

| 参数 | 论文 | 说明 | 默认值 |
|-----------|-------|-------------|---------|
| `--flex_forcing` | §3.1 | 实例化 `WanTransformer3DModel_FlexForcing` 并用 `WanFlexForcingPipeline` 校验。关闭 = 走继承的原路径，行为不变。 | 关 |
| `--flex_chunk_min` | §5 | **random 档**（占 10% 迭代，与 `--flex_pyramid_levels` 无关）抽取的最小 chunk 大小；其余三档各自钉定自己的布局、平分剩下的 90%。金字塔下 `1` 合法但不增加覆盖：2 帧 chunk 细化后本来就会产生 1 帧叶子，而它会让单帧 chunk 出现在片段中间。 | 2 |
| `--flex_chunk_max` | §5 | **random 档**抽取的最大 chunk 大小。设为与 `--flex_chunk_min` 相等即把该档钉成一种固定布局。**必须小于 latent 帧数**：“整段一个 chunk”本身就是独立的一档，把这个值抬到帧数只是把 random 档的抽取次数花在一个你已经有的布局上，反而稀释其他所有布局（21 个 latent 帧下穷举计数：`max=10` 有 6441 种不同划分、`max=21` 有 6765 种，但 `max=21` 会让单是 `[21]` 就占该档 1/19 的抽取，并把 `P([3]*7)` 从 1/183708 压到 1/1106560）。 | 10 |
| `--flex_pyramid_levels` | §3.2 | `1` = 每次迭代一个划分。`>1` = 把它嵌套成由粗到细的阶梯，每个去噪步一层；两条训练路径都会展开。阶梯本身**仅阶段 2 有**（阶段 1 每 chunk 一次前向回归 x0，没有阶梯可搭），但阶段 1 **仍然要带同一个值**：它决定共享的 level-0 混合里 coarse 档是否存在。 | 1 |
| `--flex_min_num_frame_per_block` | §3.2 | 细化终止的块大小（每块二分直到不超过它）；`1` 表示叶子级完全因果。 | 1 |
| `--flex_self_generated_context` | §3.3 | 把模型自己上一步的 `x0` 预测作为上下文喂给 K-Projection，因此在纯 prompt 训练下它会进入自动求导图。代价是注意力序列翻倍；从去噪第 1 步起生效。在 `--use_kv_cache_training` 下它不起作用，但**并不是因为 K-Projection 空闲** —— 那条路径已经把自己的干净上下文写进 KV cache，不需要额外的 `clean_x` 一半；Π 本身在那里**是**被调用的。 | 关 |
| `--num_frame_per_block` | — | 继承路径（未开 Flex）的均匀 fallback；开了 Flex 后它还是 level 0 四档混合中 uniform 那一档（该档占 30% 的迭代、内部再三等分，其一把 level 0 钉成它的均匀划分，故 `[nfpb x k]` 单独占 10%），且 `--flex_pyramid_levels=1` 时 `log_validation` 就用它采样。 | 3 |
| `--independent_first_frame` | — | 在划分前插入一个长度为 1 的 chunk（`[1, N, N, ...]`）；划分采样器会遵守它。 | 关 |

---

### 3.4 训练验证

训练脚本支持在训练过程中定期生成验证视频，无需单独启动推理任务，即可直观观察模型当前的生成能力。验证相关参数说明如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--validation_prompts` | `None` | 验证用的提示词，多个提示词以 `#` 分隔。未提供时使用 `sample_questions.txt` 或内置示例提示词 |
| `--validation_epochs` | 5 | 每多少个 epoch 执行一次验证 |
| `--validation_steps` | 2000 | 每多少个 step 执行一次验证 |

验证示例（T2V 蒸馏）：

```bash
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_flex_forcing/train_distill.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_mode="normal" \
  --validation_prompts="A cat walking on the grass in slow motion#A woman smiling at the camera" \
  --validation_steps=500 \
  --validation_epochs=1 \
  ...训练参数...
```

> 注意事项：
> - 验证仅支持 `--train_mode="normal"`（T2V 蒸馏）模式，其他训练模式（如 source / diffusion-refine）下不会执行验证。
> - 验证使用蒸馏后的去噪步数调度（`--denoising_step_indices_list`），与推理时的设置保持一致。
> - 多卡环境下验证会在 rank 0 单独执行。
> - 验证视频会保存到 `output_dir` 的 `sample` 子目录中。
> - 当 `--flex_pyramid_levels > 1` 时，验证采用 `chunk_spec=None` + `denoise_mode="pyramid"`，与推理脚本的金字塔模式一致。

---

### 3.5 使用 FSDP 训练

**如果使用多卡且使用 DeepSpeed-Zero-2 的情况下显存不足**，可以切换使用 FSDP 进行训练。现成的启动脚本是 [train_distill.sh](./train_distill.sh)：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/wan2.1_flex_forcing/train_distill.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --fix_sample_size 480 832 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --score_num_frames=21 \
  --video_repeat=1 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_distill" \
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
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --flex_pyramid_levels=4 \
  --flex_min_num_frame_per_block=1 \
  --flex_self_generated_context \
  --denoising_step_indices_list 1000 750 500 250 \
  --randomize_step_indices \
  --flow_euler_rollout \
  --use_kv_cache_training \
  --num_frame_per_block=3 \
  --ode_transformer_path="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

或者直接：

```bash
bash scripts/wan2.1_flex_forcing/train_distill.sh
```

---

### 3.6 其他后端

#### 3.6.1 使用 DeepSpeed-Zero-3 进行训练

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_flex_forcing/train_distill.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --fix_sample_size 480 832 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --score_num_frames=21 \
  --video_repeat=1 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_distill" \
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
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --flex_pyramid_levels=4 \
  --flex_min_num_frame_per_block=1 \
  --flex_self_generated_context \
  --denoising_step_indices_list 1000 750 500 250 \
  --randomize_step_indices \
  --flow_euler_rollout \
  --use_kv_cache_training \
  --num_frame_per_block=3 \
  --ode_transformer_path="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

DeepSpeed-Zero-3 训练完成之后，得到 `checkpoint-xxx` 状态的保存权重，我们需要将权重文件合并为单个，可以使用提供的`scripts/zero_to_bf16.py`脚本。

```shell
python utils/zero_to_bf16.py checkpoint-950 pytorch_model.bin --auto"detect"
```

#### 3.6.2 不使用 DeepSpeed 与 FSDP 训练

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.1_flex_forcing/train_distill.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --fix_sample_size 480 832 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --score_num_frames=21 \
  --video_repeat=1 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_distill" \
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
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --flex_pyramid_levels=4 \
  --flex_min_num_frame_per_block=1 \
  --flex_self_generated_context \
  --denoising_step_indices_list 1000 750 500 250 \
  --randomize_step_indices \
  --flow_euler_rollout \
  --use_kv_cache_training \
  --num_frame_per_block=3 \
  --ode_transformer_path="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

---

### 3.7 多机分布式训练

多机分布式训练代码与单机一致，只需要额外设置`MASTER_ADDR`、`MASTER_PORT`、`RANK`、`WORLD_SIZE`。我们提供了多机训练脚本作为案例展示。

#### 3.7.1 环境配置

**主机 0：**

```shell
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export MASTER_ADDR="192.168.1.100"  # Master machine IP
export MASTER_PORT=14536
export WORLD_SIZE=2     # Total number of machines
export NUM_PROCESS=16   # Total number of processes = number of machines × GPUs per machine
export RANK=0           # Machine rank, 0/1/2/3...
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_flex_forcing/train_distill.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --fix_sample_size 480 832 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --score_num_frames=21 \
  --video_repeat=1 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_distill" \
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
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --flex_forcing \
  --flex_chunk_min=2 \
  --flex_chunk_max=10 \
  --flex_pyramid_levels=4 \
  --flex_min_num_frame_per_block=1 \
  --flex_self_generated_context \
  --denoising_step_indices_list 1000 750 500 250 \
  --randomize_step_indices \
  --flow_euler_rollout \
  --use_kv_cache_training \
  --num_frame_per_block=3 \
  --ode_transformer_path="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

**主机 1：**

```shell
# Use the same environment variables as Machine 0, only modify RANK
export RANK=1

# Use the same accelerate launch command as Machine 0
```

#### 3.7.2 多机训练注意事项

> Note:
> - `MASTER_ADDR`: IP address of the master node (machine 0).
> - `MASTER_PORT`: Port number used for communication between machines.
> - `WORLD_SIZE`: Total number of machines.
> - `NUM_PROCESS`: Total number of processes across all machines (number of machines × GPUs per machine).
> - `RANK`: Rank of the current machine (starting from 0).
> - Ensure all machines have the same environment configuration and model files.
> - Verify network connectivity between machines, especially the reachability of `MASTER_ADDR` and `MASTER_PORT`.
> - Use `NCCL_DEBUG=INFO` to troubleshoot issues in multi-machine training.
> - If there is no RDMA environment, set `export NCCL_IB_DISABLE=1` and `export NCCL_P2P_DISABLE=1` to avoid communication issues.

---

## 四、推理测试

### 4.1 推理参数解析

推理参数 `GPU_memory_mode` 用于控制模型的加载方式：

| GPU_memory_mode | 说明 |
|-----------------|----------|
| `model_full_load` | 模型全量加载到GPU显存。速度快，显存占用高。 |
| `model_cpu_offload` | 模型加载到CPU，运行时逐层交换到显存。显存占用中等，速度较慢。 |
| `model_cpu_offload_and_qfloat8` | 模型加载到CPU，逐层转换到FLOAT8，降低GPU内存使用。显存占用低，速度中等。 |
| `model_tiling` | 使用tiling分块计算。降低VRAM使用。 |
| `sequential_cpu_offload` | 逐层卸载到CPU，支持量化加载。显存占用最低，速度最慢。 |
| `model_full_load_and_qfloat8` | 模型全量加载到显存，转换到FLOAT8，降低GPU内存使用。显存占用较高，速度快。 |

推理常用参数说明如下：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--GPU_memory_mode` | 显存管理模式 | `model_full_load` |
| `--ulysses_degree` / `--ring_degree` | 序列并行度 | 1 / 1 |
| `--fsdp_dit` / `--fsdp_text_encoder` | DiT / 文本编码器是否使用 FSDP 分片 | False / True |
| `--compile_dit` | 启用 `torch.compile` 加速 | False |
| `--model_name` | 模型路径 | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `--sampler_name` / `--shift` | 采样器类型 / Flow shift 值 | `Flow` / 5 |
| `--transformer_path` | 阶段 2 checkpoint 路径 | `output_dir_wan2.1_flex_forcing_distill/checkpoint-1000/diffusion_pytorch_model.safetensors` |
| `--sample_size` | 输出视频分辨率 `H W` | `[432, 832]` |
| `--video_length` / `--fps` | 输出视频帧数 / 帧率 | 81 / 16 |
| `--denoise_mode` | 去噪模式 | `pyramid` |
| `--min_num_frame_per_block` / `--num_frame_per_block` | 每 block 最小帧数 / 帧数 | 1 / 3 |
| `--local_attn_size` / `--sink_size` | 局部注意力窗口（-1 = 全局）/ Sink token 数量 | -1 / 0 |
| `--independent_first_frame` | 首帧是否独立 | False |
| `--context_noise` | 上下文噪声水平 | 0.0 |
| `--weight_dtype` | 权重精度 | `torch.bfloat16` |
| `--guidance_scale` | CFG 引导系数 | 1.0 |
| `--num_inference_steps` / `--seed` | 推理去噪步数 / 随机种子 | 4 / 43 |
| `--lora_path` / `--lora_weight` | 可选 LoRA | `None` / 0.55 |
| `--save_path` | 输出路径 | `samples/wan-videos-flex-forcing-t2v` |

### 4.2 文生视频（T2V）推理

阶段 2 的 checkpoint 就是最终模型。把推理入口里的 `transformer_path` 指向它：

```bash
# 3.1 灵活分块 / 3.2 金字塔
python examples/wan2.1_flex_forcing/predict_t2v.py
```

```python
# 在 examples/wan2.1_flex_forcing/predict_t2v.py 里
transformer_path = "output_dir_wan2.1_flex_forcing_distill/checkpoint-1000/diffusion_pytorch_model.safetensors"

num_inference_steps = 4          # 与 --denoising_step_indices_list 对齐
video_length        = 81         # 21 个 latent 帧 = 一段 5 秒片段
sample_size         = [432, 832]

# --- 3.1 / 3.2 推理旋钮 --------------------------------------------------
# chunk_spec 接受 None / int / "11-10" / "uniform:3" / "ar" / "bidir"。
# denoise_mode 的 "fixed" 全程保持单一分区；"pyramid" 每个去噪步降一级，
# 深度自动取 num_inference_steps，不需手工同步。传整数则钉成截断金字塔：
# 在 21 个 latent 帧上，4 复现论文的 [[21], [11, 10], [6, 5, 5, 5], [3, 3, 3, 2, 3, 2, 3, 2]]。
chunk_spec          = None
denoise_mode        = "pyramid"
min_num_frame_per_block = 1

# 3.3 K-Projection：无需设置 —— 模型自己就会建 Pi（diag_rank1）并在每次调用中应用。
```

`predict_t2v.py` 启动时会打印 `PAPER_CHUNK_CONFIGS`——§3.1 评测的十五种布局，从 `[21]`（完全双向）经 `[11, 10]`、`[7, 7, 7]` 一直到 `[3, 2, 2, 2, 2, 2, 2, 2, 2, 2]`。把其中任何一个以连字符分隔的字符串喂给 `chunk_spec`。

### 4.3 多卡并行推理

**适合场景**：高分辨率生成、加速推理

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

编辑 `examples/wan2.1_flex_forcing/predict_t2v.py`：

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
torchrun --nproc-per-node=2 examples/wan2.1_flex_forcing/predict_t2v.py
```

### 4.4 任意序视频编辑

```bash
# 4.2 任意序、任意时间步编辑
python examples/wan2.1_flex_forcing/predict_t2v_edit.py
```

编辑方面，`predict_t2v_edit.py` **只编辑、不生成**，并把重新生成限制在**细化**时间步（`edit_steps = 1`），规划时间步保持不动——这正是中间片段能以来自未来的干净上下文为条件的原因：

```python
input_video_path    = "samples/wan-videos-flex-forcing-t2v/00000001.mp4"  # 要编辑的片段，必填
edit_span           = (8, 15)    # *latent* 帧的半开区间
edit_steps          = 1          # 只做低层细化
num_frame_per_block = 7          # 干净上下文的提交粒度；None = 整段一趟双向提交
```

这里的 `num_frame_per_block` 与 Self-Forcing 那个同名参数是一回事：干净上下文按这么多帧一块、依时间序提交。区别只有两点——尾块吸收余数所以不要求整除（继承的 rollout 会断言整除），且 transformer 上的块宽取 `max(它, 编辑区间宽度)`。

任意序编辑需要完整 KV cache：两个入口都设 `local_attn_size = -1`。任何逐步换划分的 ladder（`denoise_mode="pyramid"` 或 `"full_then_blocks"`）受同一条约束，并额外要求 `stochastic_sampling = True`——两级之间的缓冲步本身就是调度自己的重加噪。任何一项不满足时 pipeline 会直接报错，而不是默默降级。

---

## 五、更多资源

- **Flex-Forcing 论文**：https://arxiv.org/abs/2607.03509
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
