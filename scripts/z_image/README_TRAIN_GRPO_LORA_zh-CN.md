# Z-Image GRPO LoRA 训练指南

本文档提供 Z-Image GRPO（Group Relative Policy Optimization）LoRA 强化微调的完整流程，包括环境配置、Prompt 数据准备、Reward 模型配置、多种分布式训练策略和推理测试。

> **说明**：Z-Image 有两个模型变体：`Z-Image`（标准版）和 `Z-Image-Turbo`（快速推理版）。本指南默认使用 `Z-Image`，如需使用 `Z-Image-Turbo`，替换对应的模型路径即可。

**与监督式 LoRA 微调的区别**：GRPO 不需要任何图片真值，只需要一批提示词（prompt），由当前模型自行采样图像，再用 Reward 模型对采样结果打分，以组内相对优势作为策略梯度信号更新 LoRA 权重。相比反向传播 Reward 的方案（`train_reward_lora.py`），GRPO 不要求 Reward 模型可微，也不存在梯度穿透 Reward 模型带来的显存与对抗补丁问题。

| 项目 | LoRA 微调（`train_lora.py`） | GRPO LoRA 训练（本文档） |
|--|--|--|
| 训练数据 | 图片 + 提示词 | **仅需提示词** |
| 监督信号 | 去噪重建误差 | Reward 模型打分经组内标准化后的优势值 |
| 是否需要真值图片 | 需要 | 不需要 |
| Reward 模型是否需要可微 | - | 不需要 |
| 典型用途 | 学习特定风格/概念/主体 | 提升人类偏好对齐（美学、图文一致性） |

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 Prompt 数据格式](#22-prompt-数据格式)
  - [2.3 相对路径与绝对路径使用方案](#23-相对路径与绝对路径使用方案)
  - [2.4 Prompt 池设计建议](#24-prompt-池设计建议)
- [三、GRPO 训练](#三grpo-训练)
  - [3.1 下载预训练模型与 Reward 模型](#31-下载预训练模型与-reward-模型)
  - [3.2 快速开始（DeepSpeed-Zero-2）](#32-快速开始deepspeed-zero-2)
  - [3.3 GRPO 专用参数解析](#33-grpo-专用参数解析)
  - [3.4 Reward 模型配置](#34-reward-模型配置)
  - [3.5 训练验证与指标监控](#35-训练验证与指标监控)
  - [3.6 使用 FSDP 训练](#36-使用-fsdp-训练)
  - [3.7 其他后端](#37-其他后端)
  - [3.8 多机分布式训练](#38-多机分布式训练)
- [四、推理测试](#四推理测试)
  - [4.1 推理参数解析](#41-推理参数解析)
  - [4.2 单卡推理](#42-单卡推理)
  - [4.3 多卡并行推理](#43-多卡并行推理)
- [五、更多资源](#五更多资源)

---

## 一、环境配置

**方式 1：使用requirements.txt**

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

**GRPO 额外依赖**

Reward 模型的依赖取决于所选的 `--reward_fn`：

```bash
pip install hpsv2
```

- `MPSReward` 首次运行会自动从 PAI OSS 下载 `MPS_overall.pth` 到 `torch.hub` 缓存目录。
- `PickScoreReward` 与 `AestheticReward` 首次运行会自动从 Hugging Face 下载对应的 CLIP/SigLIP 底座。
- 若训练机器无法联网，请提前把权重下载到本地并在 `--reward_fn_kwargs` 中填入本地路径，详见 [3.4 Reward 模型配置](#34-reward-模型配置)。

**方式 3：使用docker**

使用docker的情况下，请保证机器中已经正确安装显卡驱动与CUDA环境，然后以此执行以下命令：

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 二、数据准备

### 2.1 快速测试数据集

我们提供了一个测试的数据集，其中包含若干训练数据。GRPO 训练只会读取其中的 `text` 字段，图片文件不会被加载。

```bash
# 下载官方示例数据集
modelscope download --dataset PAI/X-Fun-Images-Demo --local_dir ./datasets/X-Fun-Images-Demo
```

### 2.2 Prompt 数据格式

GRPO 使用纯文本数据集（`TextDataset`），标注文件是一个 JSON 数组，**唯一必需的字段是 `text`**：

```json
[
  {
    "text": "1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body"
  },
  {
    "text": "A beautiful sunset over the ocean, golden hour lighting"
  }
]
```

**字段说明**：
- `text`：提示词，GRPO 会用它采样图像并交给 Reward 模型打分，**唯一必需字段**
- `file_path` / `width` / `height` / `type`：都会被忽略，因此可以直接复用监督式 LoRA 训练用的 `datasets/X-Fun-Images-Demo/metadata.json` 作为 prompt 池
- 不需要运行 `scripts/process_json_add_width_and_height.py`，也不需要准备任何图片

**生成分辨率**：由于没有真值图片，采样分辨率由训练参数决定，默认为 `--image_sample_size` 对应的正方形。如需非正方形输出，请使用 `--fix_sample_size 高度 宽度` 固定分辨率，或加上 `--random_hw_adapt` 让每个 batch 随机选择一种宽高比。

### 2.3 相对路径与绝对路径使用方案

**相对路径**：

如果数据的路径为相对路径，则在训练脚本中设置：

```bash
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata.json"
```

**绝对路径**：

如果数据的路径为绝对路径，则在训练脚本中设置：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/prompts.json"
```

### 2.4 Prompt 池设计建议

GRPO 的学习信号来自「同一个 prompt 的多张采样图之间的相对好坏」，因此 prompt 池的规模与多样性直接决定训练效果：

- **prompt 数量**：建议不少于 `num_batches_per_epoch × train_batch_size ÷ num_image_per_prompt` 的数量级，否则同一个 epoch 内会反复出现同一批 prompt。以示例配置（`16 × 1 ÷ 16 = 1`）为例，一个更新周期只覆盖 1 个 prompt，若 prompt 池过小，模型很容易过拟合到少数几条提示词上。
- **监控 `zero_std_ratio`**：该指标表示「组内 Reward 标准差为 0 的 prompt 占比」。当占比持续偏高时，说明同一 prompt 的采样结果没有差异（要么模型已经收敛到该 prompt，要么 Reward 模型区分不开），优势信号会退化，此时应扩充 prompt 池或更换/组合 Reward 模型。
- **提示词风格**：Reward 模型（尤其是 HPS 系列）对训练数据风格的分布敏感，prompt 池应尽量覆盖你希望提升的目标分布，而不是照抄验证用的提示词。

---

## 三、GRPO 训练

### 3.1 下载预训练模型与 Reward 模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer
mkdir -p models/Diffusion_Transformer/HPSv3

# 下载 Z-Image 官方权重
modelscope download --model Tongyi-MAI/Z-Image --local_dir models/Diffusion_Transformer/Z-Image

# （可选）下载 Z-Image-Turbo 快速推理版
modelscope download --model Tongyi-MAI/Z-Image-Turbo --local_dir models/Diffusion_Transformer/Z-Image-Turbo

# 下载示例使用的 Reward 模型：HPSv3 打分权重 + 其 Qwen2-VL 底座（仅 HuggingFace 提供）
huggingface-cli download MizzenAI/HPSv3 --local-dir models/Diffusion_Transformer/HPSv3
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir models/Diffusion_Transformer/Qwen2-VL-7B-Instruct
```

> **说明**：`HPSv3Reward` 的 `checkpoint_path` 留空时会自动通过 `hf_hub_download("MizzenAI/HPSv3", "HPSv3.safetensors")` 联网下载；`Qwen2-VL-7B-Instruct` 底座同理会走在线下载。离线环境请显式指定本地路径。

### 3.2 快速开始（DeepSpeed-Zero-2）

如果按照 **2.1 快速测试数据集下载数据** 与 **3.1 下载预训练模型与 Reward 模型下载权重**后，直接复制快速开始的启动指令进行启动。

推荐使用 DeepSpeed-Zero-2 与 FSDP 方案进行训练。这里使用 DeepSpeed-Zero-2 为例配置 shell 文件。

本文中 DeepSpeed-Zero-2 与 FSDP 的差别在于是否对模型权重进行分片，**如果使用多卡且使用 DeepSpeed-Zero-2 的情况下显存不足**，可以切换使用 FSDP 进行训练。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Z-Image"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/z_image/train_grpo_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1328 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=10 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_z_image_grpo_lora" \
  --validation_steps=10 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1 \
  --enable_bucket \
  --uniform_sampling \
  --rank=128 \
  --network_alpha=64 \
  --target_name="to_q,to_k,to_v,feed_forward.w1,feed_forward.w2,feed_forward.w3" \
  --noise_level 1.2 \
  --grpo_num_steps=20 \
  --grpo_cfg_scale=6 \
  --sde_window_size 5 \
  --sde_window_range 0 10 \
  --num_image_per_prompt=16 \
  --num_batches_per_epoch=16 \
  --reward_fn="HPSv3Reward" \
  --reward_fn_kwargs='{"HPSv3Reward": {"checkpoint_path": "models/Diffusion_Transformer/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Diffusion_Transformer/Qwen2-VL-7B-Instruct"}}' \
  --multi_reward_weights='{"HPSv3Reward": 1}'
```

> **说明**：示例中未出现但常被调节的还有 `--clip_range`（默认 `1e-5`）、`--adv_clip_max`（默认 `5.0`）、`--grpo_beta`（默认 `0.0`，即不加 KL 约束）、`--use_peft_lora`、`--low_vram`，可按需追加，含义见 [3.3 GRPO 专用参数解析](#33-grpo-专用参数解析)。

### 3.3 GRPO 专用参数解析

**GRPO 核心参数**：

| 参数 | 说明 | 默认值 | 示例值 |
|-----|------|--------|-------|
| `--grpo_num_steps` | 采样阶段的去噪步数 | 20 | 20 |
| `--grpo_cfg_scale` | 采样与训练重算 log-prob 时使用的 CFG 强度；设为 ≤ 1.0 时脚本会跳过 CFG（前向只跑条件分支），可显著降低 Turbo 模型的开销 | 4.5 | 6 |
| `--noise_level` | SDE 窗口内每一步注入的噪声强度，越大探索性越强、Reward 区分度越高，但偏离基座分布也越远 | 1.2 | 1.2 |
| `--sde_window_size` | 参与策略训练的时间步数量，设为 0 表示使用全部步（`grpo_num_steps - 1`） | 2 | 5 |
| `--sde_window_range` | SDE 窗口起点的候选区间 `[start, end]`，需满足 `end - sde_window_size >= start`（有断言校验） | `0 5` | `0 10` |
| `--clip_range` | PPO 裁剪范围，控制单次更新的信任域大小；调大会增强Reward提升速度，也更容易崩坏基座 | 1e-5 | 1e-5 |
| `--adv_clip_max` | 优势值截断上下限 `±adv_clip_max`，抑制离群样本主导梯度 | 5.0 | 5.0 |
| `--grpo_beta` | KL 正则系数，`0.0` 表示完全不加 KL 约束（参考模型仍会被加载但不参与前向）；出现 Reward Hacking 时可调大 | 0.0 | 0.0 |
| `--num_image_per_prompt` | 每个 prompt 的采样数量，即 GRPO 的组大小，决定优势估计的方差 | 16 | 16 |
| `--num_batches_per_epoch` | 累积多少个采样 batch 后触发一次策略更新；所有 batch 由同一份模型权重采出后合并计算优势 | 16 | 16 |
| `--reward_fn` | Reward 模型类名，多个模型用逗号分隔，如 `HPSv3Reward,MPSReward` | `MPSReward` | `HPSv3Reward` |
| `--reward_fn_kwargs` | JSON 字符串形式的 Reward 模型构造参数，详见 [3.4 Reward 模型配置](#34-reward-模型配置) | None | 见示例 |
| `--multi_reward_weights` | 多 Reward 时各 Reward 优势的组合权重（JSON），不填则等权，填写后会自动归一化 | None | `'{"HPSv3Reward": 1}'` |
| `--per_prompt_stat_tracking` | 按 prompt 分组统计均值/标准差做优势归一化（组内相对比较）。为开关型参数且默认开启；当 `num_image_per_prompt=1` 时脚本会自动关闭 | True | - |
| `--global_std` | 归一化分母使用全局 std 而非组内 std（开关型，默认开启），组内全部结果一致时仍可保留梯度 | True | - |

**LoRA 与通用训练参数**：

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/Z-Image` |
| `--train_data_meta` | Prompt 标注 JSON 路径 | `datasets/X-Fun-Images-Demo/metadata.json` |
| `--train_batch_size` | 每批次样本数 | 1 |
| `--image_sample_size` | 采样分辨率（正方形）；配合 `--fix_sample_size`/`--random_hw_adapt` 可改变输出宽高 | 1328 |
| `--fix_sample_size` | 固定采样分辨率 `[高度, 宽度]`，设置后会关闭 `random_hw_adapt` | None |
| `--random_hw_adapt` | 每个 batch 随机选择一种宽高比采样 | - |
| `--gradient_accumulation_steps` | 梯度累积系数，会放大策略更新的累积窗口 | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数（每个 epoch 遍历一遍 prompt 池） | 100 |
| `--checkpointing_steps` | 每 N 个更新步保存 checkpoint | 10 |
| `--learning_rate` | 初始学习率 | 1e-04 |
| `--max_grad_norm` | 梯度裁剪阈值（GRPO 建议使用默认 1.0，监督式 LoRA 常用的 0.05 会明显抑制策略梯度幅度） | 1 |
| `--seed` | 随机种子；各 rank 使用相同 prompt 序列、不同噪声 | 42 |
| `--output_dir` | 输出目录 | `output_dir_z_image_grpo_lora` |
| `--gradient_checkpointing` | 激活重计算 | - |
| `--mixed_precision` | 混合精度：`fp16/bf16` | `bf16` |
| `--rank` | LoRA 更新矩阵的维度（rank 越大表达能力越强，但显存占用越高） | 128 |
| `--network_alpha` | LoRA 更新矩阵的缩放系数（通常设置为 rank 的一半或相同） | 64 |
| `--target_name` | 应用 LoRA 的组件/模块，用逗号分隔 | `to_q,to_k,to_v,feed_forward.w1,feed_forward.w2,feed_forward.w3` |
| `--use_peft_lora` | 使用 PEFT 模块添加 LoRA（更省显存），会额外产出 ComfyUI 兼容权重 | - |
| `--low_vram` | 将参考模型/文本编码器常驻 CPU，仅在需要时搬回 GPU | - |
| `--resume_from_checkpoint` | 恢复训练路径，使用 `"latest"` 自动选择最新 checkpoint | None |
| `--validation_steps` / `--validation_epochs` / `--validation_prompts` | 验证触发频率与验证提示词 | 见 [3.5](#35-训练验证与指标监控) |

### 3.4 Reward 模型配置

`--reward_fn_kwargs` 是 JSON 字符串。单 Reward 时既可以写成扁平结构，也可以写成 `{"Reward名": {...}}` 嵌套结构；多 Reward 时必须使用嵌套结构。

**图像类 Reward（适用于 Z-Image）的构造参数**：

| Reward | 关键 kwargs | 说明 |
|--------|-------------|------|
| `HPSv3Reward` | `checkpoint_path`、`model_name_or_path` | 基于 Qwen2-VL-7B 的偏好模型；`checkpoint_path` 留空则自动从 `MizzenAI/HPSv3` 下载 `HPSv3.safetensors`，`model_name_or_path` 指向本地 `Qwen2-VL-7B-Instruct` |
| `HPSReward` | `model_path`、`version` | HPS v2 / v2.1，`version` 取 `"v2.0"` 或 `"v2.1"`，需额外安装 `hpsv2` |
| `PickScoreReward` | `model_path`、`processor_name_or_path` | 默认 `yuvalkirstain/PickScore_v1` + `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`，联网可自动下载 |
| `MPSReward` | `model_path`、`processor_name_or_path` | 默认自动下载 `MPS_overall.pth`；`model_path` 可指定本地 `.pth` |
| `AestheticReward` | `encoder_path`、`predictor_path`、`version` | `version` 取 `"v2"`（需 `clip-vit-large-patch14` + 美学 MLP 权重）或 `"v2.5"`（需 `siglip-so400m-patch14-384`）；纯美学打分，不使用 prompt |

> **注意**：`max_reward` 与 `loss_scale` 只影响反向传播式 Reward 训练的损失定义，GRPO 只调用 `get_reward` 拿原始分值再自己做组内标准化，因此这两个参数对 GRPO 训练结果无影响。`HPSv3Reward` / `VideoAlignReward` 的 `differentiable` 同理无需开启。`VideoAlignReward` 是视频类 Reward，不适用于本脚本。

**单 Reward 示例**：

```bash
  --reward_fn="HPSv3Reward" \
  --reward_fn_kwargs='{"HPSv3Reward": {"checkpoint_path": "models/Diffusion_Transformer/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Diffusion_Transformer/Qwen2-VL-7B-Instruct"}}' \
  --multi_reward_weights='{"HPSv3Reward": 1}'
```

**多 Reward 组合示例**（对每个 Reward 分别计算优势，再按归一化后的权重求和）：

```bash
  --reward_fn="HPSv3Reward,MPSReward" \
  --reward_fn_kwargs='{"HPSv3Reward": {"checkpoint_path": "models/Diffusion_Transformer/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Diffusion_Transformer/Qwen2-VL-7B-Instruct"}, "MPSReward": {}}' \
  --multi_reward_weights='{"HPSv3Reward": 0.7, "MPSReward": 0.3}'
```

多 Reward 时 TensorBoard 会额外记录 `{Reward名}_reward_mean`、`{Reward名}_advantage_mean`、`{Reward名}_weight` 等指标，便于观察两个模型是否出现互相拉扯。

### 3.5 训练验证与指标监控

你可以配置验证参数，在训练过程中定期生成测试图像，以便监控训练进度和模型质量。

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--validation_steps` | 每 N 步执行一次验证 | 100 |
| `--validation_epochs` | 每 N 个epoch执行一次验证 | 100 |
| `--validation_prompts` | 验证图像生成的提示词,可用空格分隔多个提示词 | 多个空格分隔的提示词 |

验证采样固定使用 `ZImagePipeline`：非 Turbo 模型为 `guidance_scale=4.5, num_inference_steps=25`，模型路径含 `Turbo` 时为 `guidance_scale=0, num_inference_steps=8`。

### 3.6 使用 FSDP 训练

**如果使用多卡且使用 DeepSpeed-Zero-2 的情况下显存不足**，可以切换使用 FSDP 进行训练。

> ✅ **推荐**：FSDP 在当前仓库中经过充分测试，错误更少、更稳定。

```sh
export MODEL_NAME="models/Diffusion_Transformer/Z-Image"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=ZImageTransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/z_image/train_grpo_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1328 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=10 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_z_image_grpo_lora" \
  --validation_steps=10 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1 \
  --enable_bucket \
  --uniform_sampling \
  --rank=128 \
  --network_alpha=64 \
  --target_name="to_q,to_k,to_v,feed_forward.w1,feed_forward.w2,feed_forward.w3" \
  --noise_level 1.2 \
  --grpo_num_steps=20 \
  --grpo_cfg_scale=6 \
  --sde_window_size 5 \
  --sde_window_range 0 10 \
  --num_image_per_prompt=16 \
  --num_batches_per_epoch=16 \
  --reward_fn="HPSv3Reward" \
  --reward_fn_kwargs='{"HPSv3Reward": {"checkpoint_path": "models/Diffusion_Transformer/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Diffusion_Transformer/Qwen2-VL-7B-Instruct"}}' \
  --multi_reward_weights='{"HPSv3Reward": 1}'
```

### 3.7 其他后端

**不使用 DeepSpeed 与 FSDP 的训练方式并不被推荐**，因为没有显存节约后端，容易造成显存不足（这也是 `scripts/z_image/train_grpo_lora.sh` 中的原始启动方式，单卡可直接使用）。这里仅提供训练 Shell 用于参考训练。

```sh
export MODEL_NAME="models/Diffusion_Transformer/Z-Image"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/z_image/train_grpo_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1328 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=10 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_z_image_grpo_lora" \
  --validation_steps=10 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1 \
  --enable_bucket \
  --uniform_sampling \
  --rank=128 \
  --network_alpha=64 \
  --target_name="to_q,to_k,to_v,feed_forward.w1,feed_forward.w2,feed_forward.w3" \
  --noise_level 1.2 \
  --grpo_num_steps=20 \
  --grpo_cfg_scale=6 \
  --sde_window_size 5 \
  --sde_window_range 0 10 \
  --num_image_per_prompt=16 \
  --num_batches_per_epoch=16 \
  --reward_fn="HPSv3Reward" \
  --reward_fn_kwargs='{"HPSv3Reward": {"checkpoint_path": "models/Diffusion_Transformer/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Diffusion_Transformer/Qwen2-VL-7B-Instruct"}}' \
  --multi_reward_weights='{"HPSv3Reward": 1}'
```

**DeepSpeed-Zero-3**：与 DeepSpeed-Zero-2 的差别仅在于启动参数换成 `config/zero_stage3_config.json`。Zero-3 与 FSDP-3 会自动把 `--save_state` 置为 True，此时 checkpoint 是包含优化器状态的目录 `output_dir/checkpoint-{global_step}/`，恢复训练使用 `--resume_from_checkpoint="latest"`。

### 3.8 多机分布式训练

**适合场景**：超大规模 prompt 池、需要更快的采样吞吐

#### 3.8.1 环境配置

假设有 2 台机器，每台 8 张 GPU：

**机器 0（Master）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Z-Image"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata.json"
export MASTER_ADDR="192.168.1.100"  # Master 机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 机器总数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 当前机器 rank（0 或 1）
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/z_image/train_grpo_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1328 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=10 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_z_image_grpo_lora" \
  --validation_steps=10 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1 \
  --enable_bucket \
  --uniform_sampling \
  --rank=128 \
  --network_alpha=64 \
  --target_name="to_q,to_k,to_v,feed_forward.w1,feed_forward.w2,feed_forward.w3" \
  --noise_level 1.2 \
  --grpo_num_steps=20 \
  --grpo_cfg_scale=6 \
  --sde_window_size 5 \
  --sde_window_range 0 10 \
  --num_image_per_prompt=16 \
  --num_batches_per_epoch=16 \
  --reward_fn="HPSv3Reward" \
  --reward_fn_kwargs='{"HPSv3Reward": {"checkpoint_path": "models/Diffusion_Transformer/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Diffusion_Transformer/Qwen2-VL-7B-Instruct"}}' \
  --multi_reward_weights='{"HPSv3Reward": 1}'
```

**机器 1（Worker）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Z-Image"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata.json"
export MASTER_ADDR="192.168.1.100"  # 与 Master 相同
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # 注意这里是 1
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

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

- **数据同步**：所有机器必须能够访问相同的数据路径（NFS/共享存储），包括 Reward 模型权重路径。

---

## 四、推理测试

### 4.1 推理参数解析

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `GPU_memory_mode` | 显存管理模式，可选值见下表 | `model_cpu_offload` |
| `ulysses_degree` | Head 维度并行度，单卡时为 1 | 1 |
| `ring_degree` | Sequence 维度并行度，单卡时为 1 | 1 |
| `fsdp_dit` | 多卡推理时对 Transformer 使用 FSDP 节省显存 | `False` |
| `fsdp_text_encoder` | 多卡推理时对文本编码器使用 FSDP | `False` |
| `compile_dit` | 编译 Transformer 加速推理（固定分辨率下有效） | `False` |
| `model_name` | 模型路径 | `models/Diffusion_Transformer/Z-Image` |
| `sampler_name` | 采样器类型：`Flow`、`Flow_Unipc`、`Flow_DPM++` | `Flow` |
| `transformer_path` | 加载训练好的 Transformer 权重路径 | `None` |
| `vae_path` | 加载训练好的 VAE 权重路径 | `None` |
| `lora_path` | LoRA 权重路径 | `None` |
| `sample_size` | 生成图像分辨率 `[高度, 宽度]` | `[1728, 992]` |
| `weight_dtype` | 模型权重精度，不支持 bf16 的显卡使用 `torch.float16` | `torch.bfloat16` |
| `prompt` | 正向提示词，描述生成内容 | `"1girl, black_hair, ..."` |
| `negative_prompt` | 负向提示词，避免生成的内容 | `"低分辨率，低画质..."` |
| `guidance_scale` | 引导强度，Turbo 模型建议设为 0.0 | 4.0 / 0.0 |
| `seed` | 随机种子，用于复现结果 | 43 |
| `num_inference_steps` | 推理步数，Turbo 模型可大幅减少 | 25 / 9 |
| `lora_weight` | LoRA 权重强度 | 0.55 |
| `save_path` | 生成图像保存路径 | `samples/z-image-t2i` |

**显存管理模式说明**：

| 模式 | 说明 | 显存占用 |
|------|------|---------|
| `model_full_load` | 整个模型加载到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 全量加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 使用后将模型卸载到 CPU | 中等 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `model_group_offload` | 层组在 CPU/CUDA 间切换 | 低 |
| `sequential_cpu_offload` | 逐层卸载（速度最慢） | 最低 |

> **与 LoRA 微调的差异**：GRPO 训练产出的权重直接命名为 `output_dir/checkpoint-{global_step}.safetensors`（不是 `checkpoint-{global_step}/` 目录），推理时直接指向该 `.safetensors` 文件即可。加载逻辑同时兼容 PEFT 命名与 ComfyUI（kohya）命名两种格式，两种文件都可使用。

### 4.2 单卡推理

#### Z-Image（标准版）

单卡推理运行如下命令：

```bash
python examples/z_image/predict_t2i.py
```

根据需求修改编辑 `examples/z_image/predict_t2i.py`，初次推理重点关注如下参数，如果对其他参数感兴趣，请查看上方的推理参数解析。

```python
# 根据显卡显存选择
GPU_memory_mode = "model_cpu_offload"
# 根据实际模型路径
model_name = "models/Diffusion_Transformer/Z-Image"  
# GRPO LoRA 权重路径，如 "output_dir_z_image_grpo_lora/checkpoint-100.safetensors"
lora_path = None
# LoRA 权重强度，GRPO 权重建议从 1.0 开始，配合 0.5/0.75 观察是否过拟合 Reward
lora_weight = 0.55
# 根据生成内容编写
prompt = "一位年轻女子站在阳光明媚的海岸线上，白裙在轻拂的海风中微微飘动。"  
# ...
```

> 💡 **CFG 一致性提示**：GRPO 是在 `--grpo_cfg_scale` 对应的引导强度下采集轨迹并优化策略的，推理时 `guidance_scale` 若与该值差距较大，收益可能不明显。建议先用接近的训练值验证效果。

#### Z-Image-Turbo（快速版）

单卡推理运行如下命令：

```bash
python examples/z_image/predict_turbo_t2i.py
```

根据需求修改编辑 `examples/z_image/predict_turbo_t2i.py`，参数设置方式与标准版一致，仅需把 `model_name` 换成 `models/Diffusion_Transformer/Z-Image-Turbo`。使用 Turbo 模型训练时，建议同时把 `--grpo_cfg_scale` 设为 `1.0` 以下以跳过 CFG。

### 4.3 多卡并行推理

**适合场景**：高分辨率生成、加速推理

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

编辑 `examples/z_image/predict_t2i.py`：

```python
# 确保 ulysses_degree × ring_degree = GPU 数量
# 例如使用 2 张 GPU：
ulysses_degree = 2  # Head 维度并行
ring_degree = 1     # Sequence 维度并行
```

**配置原则**：
- `ulysses_degree` 必须能整除模型的head数。
- `ring_degree` 会在sequence上切分，影响通信开销，在head数能切分的时候尽量不用。

**示例配置**：

| GPU 数量 | ulysses_degree | ring_degree | 说明 |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | 单卡 |
| 4 | 4 | 1 | Head 并行 |
| 8 | 8 | 1 | Head 并行 |
| 8 | 4 | 2 | 混合并行 |

#### 运行多卡推理

```bash
torchrun --nproc-per-node=2 examples/z_image/predict_t2i.py
```

## 五、更多资源

- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
- **监督式 LoRA 训练**：[README_TRAIN_LORA_zh-CN.md](./README_TRAIN_LORA_zh-CN.md)
- **全参数训练**：[README_TRAIN_zh-CN.md](./README_TRAIN_zh-CN.md)
