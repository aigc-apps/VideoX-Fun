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
> 训练时每次迭代都抽取**全新的随机划分**（chunk 大小 2–10），这正是让同一套权重覆盖整个"因果↔双向"谱系的原因。论文 §4.2 随后把同一套机制复用为**任意序编辑**。
>
> Flex-Forcing 流程分两个阶段：
>
> 1. **阶段 1——ODE 回归**（`train_ode.py`）：对教师模型的 ODE 轨迹做回归，让模型先成为一个合格的小步数生成器。随机分块在这一阶段就首次出现。
> 2. **阶段 2——DMD 蒸馏**（`train_distill.py`，即本 README）：用 **Wan2.1-T2V-14B** 真实分数教师蒸馏出 **4 步**生成器，并加入金字塔时间步调度。
>
> 本 README 只覆盖**阶段 2**。阶段 1 见 [README_TRAIN_ODE_zh-CN.md](./README_TRAIN_ODE_zh-CN.md)。

---

## 目录
- [1. 前置条件](#1-前置条件)
- [2. 环境安装](#2-环境安装)
- [3. 下载预训练模型](#3-下载预训练模型)
- [4. 准备训练数据](#4-准备训练数据)
- [5. 训练](#5-训练)
  - [5.1 快速开始](#51-快速开始)
  - [5.2 关键参数](#52-关键参数)
  - [5.3 Flex-Forcing 专属参数](#53-flex-forcing-专属参数)
- [6. 使用训练得到的 checkpoint](#6-使用训练得到的-checkpoint)
- [7. 更多资源](#7-更多资源)

---

## 1. 前置条件

阶段 2 需要：

1. 一个**阶段 1 ODE checkpoint**，用于初始化生成器（与 critic）。
2. 一个 **Wan2.1-T2V-14B** 模型，充当 DMD 的真实分数教师。

```bash
# 示例：来自 ODE 回归的阶段 1 checkpoint
export STAGE1_CKPT="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

该 checkpoint 的产出方式见 [README_TRAIN_ODE_zh-CN.md](./README_TRAIN_ODE_zh-CN.md)。

> §3.2 的金字塔是**按步主导**（step-major，`for step: for sub-chunk:`）的，而 KV-cache rollout 是**按块主导**（block-major，`for block: for step:`）。KV-cache 循环无法表达金字塔阶梯，所以当 `--flex_pyramid_levels > 1` 与 `--use_kv_cache_training` 同时出现时 `train_distill.py` 会直接报错。请走 launcher 默认的块掩码路径，或保持 `--flex_pyramid_levels 1` 只训练灵活划分。

> 术语说明：DMD（Distribution Matching Distillation，分布匹配蒸馏）里，"真实分数"教师指在真实视频分布上打分的教师模型（这里用 14B），它给学生指明真实数据分布的梯度方向；相对地，"fake 分数"来自学生自己——critic（评论家）就是学习这份 fake 分数的网络。

---

## 2. 环境安装

**方法一：使用 requirements.txt**

```bash
pip install -r requirements.txt
```

**方法二：手动安装**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**方法三：使用 Docker**

```bash
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# 进入容器
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

> 两个阶段的注意力掩码都用 **FlexAttention**（`torch.nn.attention.flex_attention`）构建。这个依赖继承自 Self-Forcing 骨干，不是本目录新增的——无需额外安装，但实际要求 `torch>=2.5`。

---

## 3. 下载预训练模型

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

然后把 `train_distill.sh` 里的 `MODEL_NAME` 指向该 1.3B 目录。

> 用 `WanTransformer3DModel_FlexForcing` 加载普通 Self-Forcing / CausVid checkpoint 时，会报告 `flex_kproj.*` 键**缺失**。这是预期行为：§3.3 的 K-Projection 是恒等初始化的，没有它的老 checkpoint 在训练之前的行为与恒等映射完全一致。

---

## 4. 准备训练数据

DMD 使用 **TextDataset**（`--train_mode="normal"` 且 `--train_data_dir` 留空）——生成器通过 rollout 自己制造训练样本，所以每个条目只读 `"text"` 字段，任何视频元数据都能当 prompt 列表：

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

`train_distill.sh` 默认指向 `datasets/X-Fun-Videos-Demo/metadata_add_width_height.json`，冒烟跑不需要额外下载。真实训练请换成论文使用的 **VidProM extended** prompt 集。

---

## 5. 训练

### 5.1 快速开始

现成的启动脚本是 [train_distill.sh](./train_distill.sh)：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME=""
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"

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
  --token_sample_size=640 \
  --fix_sample_size 432 832 \
  --video_sample_n_frames=81 \
  --score_num_frames=21 \
  --video_repeat=1 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --max_train_steps=600 \
  --checkpointing_steps=50 \
  --learning_rate=2e-06 \
  --learning_rate_critic=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_flex_forcing_distill_4step_pyramid4_flow_euler_randomize_kproj_diag_rank1" \
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
  --num_frame_per_block=3 \
  --ode_transformer_path="output_dir_wan2.1_flex_forcing_ode_regression/checkpoint-1000/diffusion_pytorch_model.safetensors"
```

或者直接：

```bash
bash scripts/wan2.1_flex_forcing/train_distill.sh
```

launcher 钉死的是：81 个像素帧 = **21 个 latent 帧**（一段 5 秒片段）、分辨率 **432×832**、**4 步**模型（`--denoising_step_indices_list 1000 750 500 250`，并由 `--randomize_step_indices` 每次迭代抖动）、`--flex_pyramid_levels=4` —— 在 21 个 latent 帧上正好复现论文的阶梯 `[[21], [11, 10], [6, 5, 5, 5], [3, 3, 3, 2, 3, 2, 3, 2]]` —— 划分抽取范围 **2..10**（另有半数的迭代以 §3.2 的整段粗粒度布局起步，见下）、`--flow_euler_rollout`、`diag_rank1` K-Projection、batch 64（8 卡 × `--gradient_accumulation_steps=8`）、`--max_train_steps=600`。

输出目录：`output_dir_wan2.1_flex_forcing_distill_4step_pyramid4_flow_euler_randomize_kproj_diag_rank1/`。

### 5.2 关键参数

| 参数 | 说明 | 示例值 |
|-----------|-------------|---------------|
| `--config_path` | 组件子路径 / `transformer_additional_kwargs` | `config/wan2.1/wan_civitai.yaml` |
| `--pretrained_model_name_or_path` | 底座模型目录（1.3B） | `$MODEL_NAME` |
| `--train_data_meta` | prompt JSON（只读 `"text"` 字段） | `$DATASET_META_NAME` |
| `--ode_transformer_path` | 用于生成器 / critic 初始化的阶段 1 ODE checkpoint | `$STAGE1_CKPT` |
| `--fix_sample_size` | 固定 `H W`；论文的 5 秒片段是 432×832 | `432 832` |
| `--video_sample_n_frames` | 像素帧数；81 → 21 个 latent 帧 | 81 |
| `--score_num_frames` | 真实分数教师看到的 latent 帧数 | 21 |
| `--train_batch_size` | 单卡 batch 大小 | 1 |
| `--gradient_accumulation_steps` | 8 卡 × 8 = 论文的 batch 64 | 8 |
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
| `--flow_euler_rollout` | 生成器自滚用 flow 空间里的确定性 Euler ODE 步推进（`x_next = x_t + (sigma_next - sigma_t) * v`，fp32），而不是转成 x0 再用新噪声重加噪。代数上这**就等于**用同一个隐含噪声重加噪 —— fp64 下实测相差 `8.9e-16`，而重采新噪声最大差 `6.4` —— 因此它消掉了 rollout 的随机性，与正常的 flow-matching 推理一致。最后一步仍然转 x0，因为 DMD 目标定义在 x0 上。critic 重滚生成器，跟随同一开关。 | 关 |
| `--output_dir` | 输出目录 | `output_dir_wan2.1_flex_forcing_distill_4step_pyramid4_flow_euler_randomize_kproj_diag_rank1` |
| `--gradient_checkpointing` | 用算力换显存 | - |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--trainable_modules` | 可训练组的子串匹配（`"."` = 全部） | `"."` |
| `--train_mode` | `normal`（TextDataset，纯 prompt） | `normal` |
| `--resume_from_checkpoint` | `"latest"` 自动选择 | `"latest"` |

### 5.3 Flex-Forcing 专属参数

| 参数 | 论文 | 说明 | 默认值 |
|-----------|-------|-------------|---------|
| `--flex_forcing` | §3.1 | 实例化 `WanTransformer3DModel_FlexForcing` 并用 `WanFlexForcingPipeline` 校验。关闭 = 走继承的原路径，行为不变。 | 关 |
| `--flex_chunk_min` | §3.1 | 每次迭代抽取的最小 chunk 大小。金字塔下 `1` 合法但不增加覆盖：2 帧 chunk 细化后本来就会产生 1 帧叶子，而它会让单帧 chunk 出现在片段中间。 | 2 |
| `--flex_chunk_max` | §3.1 | 每次迭代抽取的最大 chunk 大小。设为与 `--flex_chunk_min` 相等即固定一种布局。**必须小于 latent 帧数**：“整段一个 chunk”这个布局已经由下面的混合机制覆盖，把这个值抬到帧数只是把抽取次数花在一个你已经有的布局上，反而稀释 §3.1（21 个 latent 帧时，`max=21` 只剩 3649 种不同划分，而 `max=10` 有 4882 种）。 | 10 |
| `--flex_pyramid_levels` | §3.2 | `1` = 每次迭代一个划分。`>1` = 把它嵌套成由粗到细的阶梯，每个去噪步一层。要求走块掩码路径。**仅阶段 2 有。** | 1 |
| `--flex_min_num_frame_per_block` | §3.2 | 细化终止的块大小（每块二分直到不超过它）；`1` 表示叶子级完全因果。 | 1 |
| `--flex_self_generated_context` | §3.3 | 把模型自己上一步的 `x0` 预测喂给 K-Projection 当上下文，使其在纯 prompt 训练下也能进入 autograd 图。代价是注意力序列翻倍；从去噪步 1 起生效。 | 关 |
| `--num_frame_per_block` | — | 继承路径（未开 Flex）的均匀 fallback；开了 Flex 后它还是 level 0 三档混合中的一档（`UNIFORM_BLOCK_PROB = 0.1` 的迭代把 level 0 钉成它的均匀划分），且 `--flex_pyramid_levels=1` 时 `log_validation` 就用它采样 —— 见下面两条 bullet。 | 3 |
| `--independent_first_frame` | — | 在划分前插入一个长度为 1 的 chunk（`[1, N, N, ...]`）；划分采样器会遵守它。 | 关 |

有七个交互值得明说：

- **level 0 是三档混合，一次抽样定档。** 推理时 `denoise_mode="pyramid"` 的 level 0 是整段一个 chunk（全双向的全局规划），只从 2..10 抽划分会让那一步从未被训练；而 `--num_frame_per_block` 那个均匀布局光靠随机抽样又几乎碰不到（21 个 latent 帧下 `[3]*7` 的概率是 `1/183708`，20000 次迭代期望 0.11 次）。所以 `train_distill.py` 用一个均匀抽样把 level 0 分成三档，两个常量都故意不做成 CLI 参数：`COARSE_GLOBAL_PROB = 0.5` 整段一个 chunk、`UNIFORM_BLOCK_PROB = 0.1` 钉成 `--num_frame_per_block` 的均匀划分、剩下 `0.4` 是 §3.1 的随机划分；三档都细化成同一条阶梯。实测 200000 次（21 个 latent 帧、`--flex_pyramid_levels=4`）：整段 `49.999%`、`[3]*7` `10.034%`、随机 `39.968%`，共 **4078** 种不同的 level-0 布局，推理那条阶梯 `[[21], [11, 10], [6, 5, 5, 5], [3, 3, 3, 2, 3, 2, 3, 2]]` 占 `49.999%`。`--flex_pyramid_levels=1` 时整段那一档为空，比例变成 `10% / 90%`，与论文描述一致。峰值显存由最密的那种情形（level 0 = 整段）决定，请据此预留。
- 阶梯会被**截断到训练的去噪步数**。2 步调度下配 4 级阶梯永远到不了最后两级，所以 `sample_flex_partitions` 会丢弃到不了的那部分，而不是默默把它们当作已训练。请保持 `len(--denoising_step_indices_list) >= --flex_pyramid_levels`；launcher 里的 4 个索引与 `--flex_pyramid_levels=4` 就是故意对齐的。反过来也浪费算力：步数多于级数时，多出来的那些步里 `install_flex_partition` 会被钉在最细的一级上。
- **`--flow_euler_rollout` 会把预测留在 flow 空间，所以 §3.3 的上下文需要自己那份 x0。** `--flex_self_generated_context` 喂给 K-Projection 的是上一步的**干净**预测；在 Euler rollout 下那个变量是速度，所以两个 rollout 循环都会另外转换一份 detached 副本给上下文用，而 rollout 状态本身仍按 Euler 推进。生成器与 critic 的转换必须完全一致，否则 critic 评的就不是正在被训的那个 rollout。
- **`--randomize_step_indices` 不会破坏金字塔。** 抖动动的是**时间步**而不是划分，而 `sample_flex_partitions` 只看 `len(denoising_step_list)`，抖动不会改变它。阶梯的每一级仍然对得上自己的去噪步。
- **`log_validation` 渲染的就是训练渲染的那个布局。** `--flex_pyramid_levels > 1` 时它传 `chunk_spec=None` + `denoise_mode="pyramid"`，阶梯与 `predict_t2v.py` 的完全相同，实测占 `49.999%` 的迭代。它原本把 `chunk_spec` 钉成 `--num_frame_per_block`，那会嵌套出 `[[3]*7, [2,1]*7, [1]*21]` —— 在 4 步调度下只有 **3 级**（细化到全 1 就撞上固定点），而当时那条阶梯 trainer 在 20000 次迭代里抽出 **0 次**，所以那些校验视频根本说明不了正在训练的模型。`--flex_pyramid_levels=1` 时结论反过来：整段那一档为空、`[21]` 一次也抽不到，所以校验用 `--num_frame_per_block` 的均匀划分 —— 它是唯一既跨 checkpoint 固定、又真被训练的一档（`10.066%` 的迭代，而最高频的随机划分只有约 `1.3%`）。顺带说明那条 3 级阶梯：它现在有 `10.034%` 的覆盖率，第 4 步由 `install_flex_partition` 的索引 clamp（`min(step_index, len(partitions) - 1)`）复用最细一级 `[1]*21`，不会越界。
- §3.3 **没有 launcher 参数**：变体由 config yaml 里的 `transformer_additional_kwargs.flex_kproj_mode` 决定（默认 `diag_rank1`；要消融就在那里改成 `none`，想更省就改 `diag`）。它的参数故意取名 `flex_kproj.proj_out.*` —— 正是这个名字让 `initialize_missing_parameters()` 在 checkpoint 早于 §3.3 时将其置零，从而在 `low_cpu_mem_usage=True` 下保住恒等初始化；除此之外它和其他层一样训，跟 `--learning_rate`。只要构建出的模型带 Pi，trainer 就强制其 `requires_grad`，所以收窄 `--trainable_modules` 也不会把 Pi 悄悄冻在恒等点。
- 每张卡必须训练同一种布局，否则 FlexAttention 掩码与其派生的 `num_frame_per_block` 在 SP/FSDP 组之间会不一致。trainer 在使用抽取的划分前会先经 `broadcast_chunk_sizes` 广播对齐；`randomize_denoising_step_indices` 同样从 rank 0 广播它抖动后的索引，道理一样。

---

## 6. 使用训练得到的 checkpoint

阶段 2 的 checkpoint 就是最终模型。把推理入口里的 `transformer_path` 指向它：

```bash
# 3.1 灵活分块 / 3.2 金字塔
python examples/wan2.1_flex_forcing/predict_t2v.py

# 4.2 任意序、任意时间步编辑
python examples/wan2.1_flex_forcing/predict_t2v_edit.py
```

```python
# 在 examples/wan2.1_flex_forcing/predict_t2v.py 里
transformer_path = "output_dir_wan2.1_flex_forcing_distill_4step_pyramid4_flow_euler_randomize_kproj_diag_rank1/checkpoint-600/diffusion_pytorch_model.safetensors"

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

编辑方面，`predict_t2v_edit.py` **只编辑、不生成**，并把重新生成限制在**细化**时间步（`edit_steps = 1`），规划时间步保持不动——这正是中间片段能以来自未来的干净上下文为条件的原因：

```python
input_video_path    = "samples/wan-videos-flex-forcing-t2v/00000001.mp4"  # 要编辑的片段，必填
edit_span           = (8, 15)    # *latent* 帧的半开区间
edit_steps          = 1          # 只做低层细化
num_frame_per_block = 7          # 干净上下文的提交粒度；None = 整段一趟双向提交
```

这里的 `num_frame_per_block` 与 Self-Forcing 那个同名参数是一回事：干净上下文按这么多帧一块、依时间序提交。区别只有两点——尾块吸收余数所以不要求整除（继承的 rollout 会断言整除），且 transformer 上的块宽取 `max(它, 编辑区间宽度)`。

任意序编辑需要完整 KV cache：两个入口都设 `local_attn_size = -1`。金字塔还需要 `stochastic_sampling = True`——两级之间的缓冲步本身就是调度自己的重加噪。任何一项不满足时 pipeline 会直接报错，而不是默默降级。

---

## 7. 更多资源

- **Flex-Forcing 论文**：https://arxiv.org/abs/2607.03509
- **阶段 1**：[README_TRAIN_ODE_zh-CN.md](./README_TRAIN_ODE_zh-CN.md)
- **共享分块 helper**：`videox_fun/utils/flex_chunking.py` — `normalize_chunk_spec`、`sample_flexible_chunks`、`refine_partition`、`build_pyramid_partitions`、`validate_nested_partitions`、`broadcast_chunk_sizes`
- **模型 / pipeline**：`videox_fun/models/wan_transformer3d_flex_forcing.py`、`videox_fun/pipeline/pipeline_wan_flex_forcing.py`
- **上游阶段**：[README_TRAIN_ODE.md](../wan2.1_self_forcing/README_TRAIN_ODE.md)、[README_TRAIN_DISTILL.md](../wan2.1_self_forcing/README_TRAIN_DISTILL.md)
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
