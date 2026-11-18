# Wan2.2 Fun Reward LoRA 训练指南

本文档提供 Wan2.2 Fun **Reward 反向传播（Reward Backpropagation）** 微调的完整流程，即用 LoRA 优化生成视频，使其更符合人类偏好。内容涵盖环境配置、数据准备、Reward 专用训练参数（单/多 reward、可微 VAE 解码、backprop 策略）、训练验证与推理测试。

> **说明**：Wan2.2 Fun 是一个基于 Wan2.2 架构的视频修复模型。Wan2.2 采用双 Transformer 架构（高噪声/低噪声模型），5B 版本使用单 Transformer 架构。Reward LoRA 训练会跑一段可微的 rollout（一条简短的去噪轨迹），并把 reward 模型的 loss 反传到可训练的专家上，因此比标准 LoRA 微调更耗显存。本指南支持 A14B 与 5B 两种变体。

我们采用 Reward Backpropagation 技术 <sup>[1](#ref1) [2](#ref2)</sup> 来优化 [Wan2.2-Fun](https://github.com/aigc-apps/VideoX-Fun) 生成的视频，使其更好地对齐人类偏好。我们随[训练脚本](https://github.com/aigc-apps/VideoX-Fun/blob/main/scripts/wan2.2_fun/train_reward_lora.py)一并提供以下预训练模型（即 LoRA）。你可以用这些 LoRA 作为插件增强对应的基座模型，也可以训练自己的 reward LoRA。

| 名称 | 基座模型 | Reward 模型 | Hugging Face | 说明 |
|--|--|--|--|--|
| Wan2.2-Fun-A14B-InP-high-noise-HPS2.1.safetensors | [Wan2.2-Fun-A14B-InP（high noise）](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP/tree/main/high_noise_model) | [HPS v2.1](https://github.com/tgxs002/HPSv2) | [🤗链接](https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-high-noise-HPS2.1.safetensors) | 面向 Wan2.2-Fun-A14B-InP（high noise）的官方 HPS v2.1 reward LoRA（`rank=128`、`network_alpha=64`）。 |
| Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors | [Wan2.2-Fun-A14B-InP（low noise）](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP/tree/main/low_noise_model) | [HPS v2.1](https://github.com/tgxs002/HPSv2) | [🤗链接](https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors) | 面向 Wan2.2-Fun-A14B-InP（low noise）的官方 HPS v2.1 reward LoRA（`rank=128`、`network_alpha=64`）。 |
| Wan2.2-Fun-A14B-InP-high-noise-MPS.safetensors | [Wan2.2-Fun-A14B-InP（high noise）](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP/tree/main/high_noise_model) | [MPS](https://github.com/Kwai-Kolors/MPS) | [🤗链接](https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-high-noise-MPS.safetensors) | 面向 Wan2.2-Fun-A14B-InP（high noise）的官方 MPS reward LoRA（`rank=128`、`network_alpha=64`）。 |
| Wan2.2-Fun-A14B-InP-low-noise-MPS.safetensors | [Wan2.2-Fun-A14B-InP（low noise）](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP/tree/main/low_noise_model) | [MPS](https://github.com/Kwai-Kolors/MPS) | [🤗链接](https://huggingface.co/alibaba-pai/Wan2.1-Fun-Reward-LoRAs/resolve/main/Wan2.1-Fun-14B-InP-MPS.safetensors) | 面向 Wan2.2-Fun-A14B-InP（low noise）的官方 MPS reward LoRA（`rank=128`、`network_alpha=64`）。 |

> [!NOTE]
> 我们发现，low-noise 模型的 MPS reward LoRA 收敛速度明显慢于其他模型，且效果可能不理想。因此对于 low-noise 模型，我们推荐使用 HPS v2.1 reward LoRA。

## 演示
请参考[此处](https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs#demo)。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 Reward 模型权重](#22-reward-模型权重)
  - [2.3 数据集结构与 metadata.json](#23-数据集结构与-metadatajson)
- [三、Reward LoRA 训练](#三reward-lora-训练)
  - [3.1 下载预训练基座模型](#31-下载预训练基座模型)
  - [3.2 快速开始（FSDP）](#32-快速开始fsdp)
  - [3.3 Reward 专用参数解析](#33-reward-专用参数解析)
  - [3.4 单 Reward 与多 Reward 训练](#34-单-reward-与多-reward-训练)
  - [3.5 可微解码与 `--use_1steps_vae`](#35-可微解码与---use_1steps_vae)
  - [3.6 训练验证](#36-训练验证)
- [四、推理测试](#四推理测试)
- [五、局限性](#五局限性)
- [六、更多资源](#六更多资源)
- [参考文献](#参考文献)

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

> 部分视频 reward 模型（如 `VideoAlignReward`、`HPSv3Reward`）基于 Qwen2-VL 骨干，需要 `transformers==4.51.3`。使用这些 reward 时请安装对应版本。

**方式 3：使用 docker**

使用 docker 的情况下，请保证机器中已经正确安装显卡驱动与 CUDA 环境，然后以此执行以下命令：

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 二、数据准备

### 2.1 快速测试数据集

Reward LoRA 训练从文本 prompt 出发跑可微 rollout，因此标准的 T2V reward 路径只需一个**纯 prompt 数据集**即可。对于 I2V/inpaint 路径，则使用视频数据集来提供参考条件。

```bash
# 下载官方示例数据集
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 2.2 Reward 模型权重

不同 reward 的骨干/checkpoint 来源不同：

- **`HPSReward`（HPS v2.1）与 `MPSReward`**：打分头很小，会在**首次运行时自动下载**到 torch hub 缓存（`~/.cache/torch/hub`）；其中 **HPS 还需给 `hpsv2` 内置的 open_clip 补两个特殊配置文件**（见下），MPS 无需额外准备；
- **`VideoAlignReward`（VideoReward）与 `HPSv3Reward`**：基于 Qwen2-VL 骨干，体积较大，**建议提前下载到本地**，再用 `--reward_fn_kwargs` 指向本地路径。

各 reward 的依赖与来源汇总如下（`--reward_fn` 取值见 [3.3](#33-reward-专用参数解析)）：

| Reward | 骨干 / 依赖 | 打分头 / checkpoint | 手动下载 |
|--|--|--|--|
| `HPSReward`（`version="v2.1"`） | open_clip `ViT-H-14`，需 `pip install hpsv2` + `factory.py` 补丁与 BPE 词表 | `HPS_v2.1_compressed.pt`（自动下载） | 否，可用 `model_path` 覆盖 |
| `MPSReward` | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`（config/tokenizer 自动下载），MPS 代码已内置于 `videox_fun/reward/MPS/` | `MPS_overall.pth`（自动下载） | 否，可用 `model_path` 覆盖 |
| `VideoAlignReward` | `Qwen/Qwen2-VL-2B-Instruct` | `KlingTeam/VideoReward`（含 `model_config.json` 与权重） | **是**，`model_path` 必填 |
| `HPSv3Reward` | `Qwen/Qwen2-VL-7B-Instruct` | `MizzenAI/HPSv3` 的 `HPSv3.safetensors` | 建议，缺省时自动从 HF 下载 |

**HPS v2.1（`HPSReward`）**——除了 `pip install hpsv2`，还要给 `hpsv2` 内置的 open_clip 补两个特殊配置文件：`create_model_and_transforms("ViT-H-14", ...)` 用到了 HPS 分支新增的参数，需要打过补丁的 `factory.py`；`get_tokenizer("ViT-H-14")` 需要 BPE 词表 `bpe_simple_vocab_16e6.txt.gz`。缺任一都会在加载时报错。`ViT-H-14` 骨干与 `HPS_v2.1_compressed.pt` 打分头会在首次运行时自动下载到 torch hub 缓存：

```bash
pip install hpsv2
# 覆盖 hpsv2 内置 open_clip 的 factory.py 补丁，并补上 BPE 词表
site_packages=$(python -c "import site; print(site.getsitepackages()[0])")
wget -O $site_packages/hpsv2/src/open_clip/factory.py https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/package/patches/hpsv2_src_open_clip_factory_patches.py
wget -O $site_packages/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz https://github.com/tgxs002/HPSv2/raw/refs/heads/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz
```

```bash
  --reward_fn="HPSReward" \
  --reward_fn_kwargs='{"version": "v2.1"}'
```

**MPS（`MPSReward`）**——MPS 代码已内置于 `videox_fun/reward/MPS/`，无需打补丁；`MPS_overall.pth` 打分头会自动从 OSS 下载，`laion/CLIP-ViT-H-14-laion2B-s32B-b79K` 的 config 与 tokenizer 会自动从 HuggingFace 下载（若无法访问 HF，可用 `processor_name_or_path` 指向本地镜像）：

```bash
  --reward_fn="MPSReward" \
  --reward_fn_kwargs='{"max_reward": 1, "loss_scale": 1}'
```

**VideoReward（`VideoAlignReward`）**——需 Qwen2-VL-2B 骨干 + VideoReward 打分头目录（内含 `model_config.json` 与权重），二者都要提前下载，且 `model_path` 必填：

```bash
modelscope download --model Qwen/Qwen2-VL-2B-Instruct --local_dir models/Personalized_Model/Qwen2-VL-2B-Instruct
huggingface-cli download KlingTeam/VideoReward --local-dir models/Personalized_Model/VideoReward
```

```bash
  --reward_fn="VideoAlignReward" \
  --reward_fn_kwargs='{"model_path": "models/Personalized_Model/VideoReward/", "model_name_or_path": "models/Personalized_Model/Qwen2-VL-2B-Instruct", "reward_dim": "TA+MQ", "fps": 16, "use_norm": true, "differentiable": true, "max_reward": 2.0}'
```

**HPSv3（`HPSv3Reward`）**——需 Qwen2-VL-7B 骨干 + `HPSv3.safetensors`；不指定 `checkpoint_path` 时会自动从 HF `MizzenAI/HPSv3` 下载，建议提前落盘：

```bash
modelscope download --model Qwen/Qwen2-VL-7B-Instruct --local_dir models/Personalized_Model/Qwen2-VL-7B-Instruct
modelscope download --model MizzenAI/HPSv3 --local_dir models/Personalized_Model/HPSv3
```

```bash
  --reward_fn="HPSv3Reward" \
  --reward_fn_kwargs='{"checkpoint_path": "models/Personalized_Model/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Personalized_Model/Qwen2-VL-7B-Instruct", "differentiable": true, "max_reward": 12}'
```

> 使用 `--backprop` 时，`VideoAlignReward` / `HPSv3Reward` 必须设置 `"differentiable": true`，否则其 `@torch.no_grad()` 路径会切断梯度链（见 [3.3](#33-reward-专用参数解析)）。

### 2.3 数据集结构与 metadata.json

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

**相对路径格式**（示例）：
```json
[
  {
    "file_path": "train/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "type": "video",
    "width": 1024,
    "height": 1024
  }
]
```

**关键字段说明**：
- `file_path`：视频路径（相对路径或绝对路径）。
- `text`：视频描述（英文 prompt）。
- `type`：数据类型，固定为 `"video"`。
- `width` / `height`：视频宽高（**强烈建议提供**，用于分桶训练）。可用 `scripts/process_json_add_width_and_height.py` 提取这两个字段。

**相对路径与绝对路径**：
- 相对路径：`export DATASET_NAME="datasets/X-Fun-Videos-Demo/"` 与 `export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"`。
- 绝对路径：`export DATASET_NAME=""` 与 `export DATASET_META_NAME="/mnt/data/metadata_add_width_height.json"`。

---

## 三、Reward LoRA 训练

### 3.1 下载预训练基座模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Wan2.2 Fun 官方权重
# A14B 模型（双 Transformer 架构）
modelscope download --model PAI/Wan2.2-Fun-A14B-InP --local_dir models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP
# 或 5B 模型（单 Transformer 架构）
# modelscope download --model PAI/Wan2.2-Fun-5B-InP --local_dir models/Diffusion_Transformer/Wan2.2-Fun-5B-InP
```

### 3.2 快速开始（FSDP）

Reward LoRA 训练会把完整的去噪 rollout 与 reward 模型的前向计算图保持在显存中，因此推荐使用 FSDP（`FULL_SHARD`）。当 `--boundary_type="high"` 时，*另一个*（冻结的）专家在训练期已被分片并常驻 GPU，验证阶段会直接复用它，而不会从磁盘再加载一份副本。

**Wan2.2 Fun Reward LoRA 训练示例（单一 HPS v2.1 reward，FSDP）**：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 与 NCCL_P2P_DISABLE=1 用于没有 RDMA 的多机环境。
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
  --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" \
  --fsdp_cpu_ram_efficient_loading False \
  scripts/wan2.2_fun/train_reward_lora.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_fun_reward_lora" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --enable_bucket \
  --boundary_type="low" \
  --rank=128 \
  --network_alpha=64 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --num_inference_steps=25 \
  --num_decoded_latents=3 \
  --reward_fn="HPSReward" \
  --reward_fn_kwargs='{"version": "v2.1"}' \
  --backprop \
  --backprop_strategy="tail" \
  --backprop_num_steps=1 \
  --train_mode="inpaint" \
  --low_vram
```

> 本目录下的 `train_reward_lora.sh` 提供了一份可直接运行的模板。请根据你的环境调整 `MODEL_NAME`、数据集路径、`--reward_fn` 与 `--boundary_type`。

### 3.3 Reward 专用参数解析

除标准 LoRA 参数外（见 [README_TRAIN_LORA_zh-CN.md](./README_TRAIN_LORA_zh-CN.md)），reward 训练脚本还新增以下参数：

| 参数 | 说明 | 示例值 |
|-----------|-------------|---------------|
| `--reward_fn` | Reward 模型类名，可用**逗号分隔的列表**启用多 reward 训练；各 reward 的可微 loss 以加权和合并。可选：`AestheticReward`、`HPSReward`、`PickScoreReward`、`MPSReward`、`HPSv3Reward`、`VideoAlignReward`。 | `"HPSv3Reward,VideoAlignReward"` |
| `--reward_fn_kwargs` | Reward 模型的关键词参数。单 reward 用**扁平 JSON dict**；多 reward 用**以 reward 名为 key 的嵌套 JSON dict**。 | `'{"VideoAlignReward": {"max_reward": 2.0}, "HPSv3Reward": {"max_reward": 12}}'` |
| `--multi_reward_weights` | 多 reward 各 loss 权重的 JSON dict。会归一化到和为 1；多 reward 时若省略则使用等权重；单 reward 时忽略。 | `'{"VideoAlignReward": 0.5, "HPSv3Reward": 0.5}'` |
| `--num_inference_steps` | 可微 rollout 与验证中的去噪步数。 | 25 |
| `--num_decoded_latents` | 解码回像素以供 reward 评估的潜帧数。分块因果 VAE 受梯度检查点限制，除非设置 `--use_1steps_vae`，否则要求 `<= 3`。 | 3 |
| `--num_reward_frames` | 随机选取该数量的解码帧用于 reward 评估。`None` = 使用全部。仅对**图像** reward 生效（见下）。 | 2 |
| `--use_1steps_vae` | 使用单段式 Wan2.2 VAE（`AutoencoderKLWan3_8_1Steps`）做可微解码；数值等价，一次前向解码整个 latent，干净地按解码块应用梯度检查点，并解除 `<= 3` 的限制。仅影响 `AutoencoderKLWan3_8`（2.2 VAE）路径。 | - |
| `--backprop` | 启用 reward 反向传播训练模式。 | - |
| `--backprop_strategy` | rollout 中哪些步保留梯度图：`last`、`tail`、`uniform`、`random`。 | `tail` |
| `--backprop_num_steps` | backprop 步数，仅当 `--backprop_strategy` 为 `tail`/`uniform`/`random` 时生效。 | 1 |
| `--first_frame_mse_weight` | inpaint 模式下首帧 latent MSE 辅助 loss 的权重。 | 1.0 |
| `--force_t2v` | 当 `train_mode` 为 `"inpaint"`（I2V）或 `"ti2v"` 时，按纯 T2V 训练（条件置零、使用纯 prompt 数据集）。 | - |

**图像 reward 与视频 reward（`is_image_reward`）**：
- **图像 reward**（`AestheticReward`、`HPSReward`、`PickScoreReward`、`MPSReward`、`HPSv3Reward`）逐帧打分，因此帧抽稀（`--num_reward_frames`）是安全的。
- **视频 reward**（`VideoAlignReward`，`is_image_reward=False`）需要完整时序序列来评判运动质量；若压缩帧会把视频退化为近乎静态的输入并破坏运动维度。因此这类 reward 保留全部解码帧（数量受 `--num_decoded_latents` 约束）。
- 在 inpaint 模式下，所有 reward 在评估前都会丢弃首帧（`--force_t2v` 时除外）。

> **重要**：使用 `--backprop` 时，`VideoAlignReward` / `HPSv3Reward` 这类视频 reward 必须在 `--reward_fn_kwargs` 中设置 `"differentiable": true`，否则其 `@torch.no_grad()` 路径会彻底切断梯度链。

### 3.4 单 Reward 与多 Reward 训练

单 reward 使用 `--reward_fn="HPSReward"` 配一个扁平的 `--reward_fn_kwargs`。若要组合多个目标，传入逗号分隔的列表、嵌套的 kwargs dict，以及（可选的）权重。每个 reward 的 loss 与 reward 值会分开记录（如 `train/loss_HPSv3Reward`、`train/reward_VideoAlignReward`），而实际优化的 `loss = Σ wᵢ · lossᵢ`。

**多 reward 示例（HPSv3 + VideoAlign，T2V）**：

```bash
  --num_decoded_latents=13 \
  --num_reward_frames=2 \
  --use_1steps_vae \
  --reward_fn="VideoAlignReward,HPSv3Reward" \
  --reward_fn_kwargs='{"VideoAlignReward": {"model_path": "models/Personalized_Model/VideoReward/", "model_name_or_path": "models/Personalized_Model/Qwen2-VL-2B-Instruct", "fps": 16, "reward_dim": "TA+MQ", "differentiable": true, "use_norm": true, "max_reward": 2.0, "loss_scale": 1.0}, "HPSv3Reward": {"checkpoint_path": "models/Personalized_Model/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Personalized_Model/Qwen2-VL-7B-Instruct", "differentiable": true, "max_reward": 12}}' \
  --multi_reward_weights='{"VideoAlignReward": 0.5, "HPSv3Reward": 0.5}' \
  --backprop \
  --backprop_strategy="tail" \
  --backprop_num_steps=1
```

### 3.5 可微解码与 `--use_1steps_vae`

默认的分块因果 VAE（`AutoencoderKLWan3_8`）按块应用梯度检查点，从而把可微解码限制在**前 3 个潜帧**（`--num_decoded_latents <= 3`）。不设置 `--use_1steps_vae` 而使用 `--num_decoded_latents > 3` 会抛出 `ValueError`。

启用 `--use_1steps_vae` 可切换到单段式变体，它在一次前向中解码整个 latent、按解码块干净地应用梯度检查点，并解除该限制——当视频 reward 需要更多时序上下文时很有用。它数值等价，且仅影响 `AutoencoderKLWan3_8`（2.2 VAE）路径。

### 3.6 训练验证

你可以配置验证参数，在训练过程中周期性生成测试视频，以监控对齐进度。在 FSDP 下，`log_validation` 里的 pipeline 前向是一次跨 rank 的集合通信，因此**所有进程都必须进入验证调用**（它刻意**没有**包在 `is_main_process` 里）；可训练专家已带上 LoRA，冻结专家则被复用而非重新加载。

| 参数 | 说明 | 默认值 |
|-----------|-------------|---------------|
| `--validation_steps` | 每 N 步运行一次验证 | 2000 |
| `--validation_epochs` | 每 N 个 epoch 运行一次验证 | 5 |
| `--validation_prompts` | 用于验证视频生成的 prompt | None |
| `--validation_paths` | I2V 验证的参考图路径（仅 inpaint 模式） | None |

**验证示例**：

```bash
  --validation_prompts="A brown dog shaking its head, sitting on a light-colored sofa in a cozy room." \
  --validation_steps=50 \
  --validation_epochs=500
```

> 验证视频会保存到 `output_dir` 目录。若做 T2V 验证（无参考图），请设置 `train_mode="normal"`（或加上 `--force_t2v`）并省略 `--validation_paths`。

---

## 四、推理测试

训练好的 reward LoRA 就是一个普通 LoRA，加载方式与 [README_TRAIN_LORA_zh-CN.md](./README_TRAIN_LORA_zh-CN.md) 完全一致。对于 A14B 双 Transformer 模型，在 [examples/wan2.2_fun/predict_t2v.py](https://github.com/aigc-apps/VideoX-Fun/blob/main/examples/wan2.2_fun/predict_t2v.py)（或 `predict_i2v.py`）中，为 low-noise reward LoRA 设置 `lora_path` / `lora_weight`，为 high-noise reward LoRA 设置 `lora_high_path` / `lora_high_weight`：

```python
# 训练好的 reward LoRA 权重路径，例如
# "output_dir_wan2.2_fun_reward_lora/checkpoint-xxx/diffusion_pytorch_model.safetensors"
lora_path = None
lora_high_path = None
lora_weight = 0.55
lora_high_weight = 0.55
```

单卡推理运行：

```bash
python examples/wan2.2_fun/predict_t2v.py
```

显存管理模式与多卡并行推理的详细说明，请参见 [README_TRAIN_LORA_zh-CN.md](./README_TRAIN_LORA_zh-CN.md) 的推理测试章节。

---

## 五、局限性

1. 训练到一定程度后，reward 持续上升，但生成视频的质量不再进一步提升。模型会取巧地学到一些捷径（在背景里添加伪影，即对抗补丁）来抬高 reward。
2. 目前仍缺乏适合视频生成的偏好模型。直接使用图像偏好模型无法沿时序维度评估偏好（如运动性与一致性），并且会降低生成视频的运动性。可以仅用解码视频的第一帧来计算 reward 来部分缓解，但影响依然存在——这也正是对于运动敏感的对齐，推荐使用视频 reward（如 `VideoAlignReward`）并配合 `--use_1steps_vae` 与更大的 `--num_decoded_latents` 的原因。

---

## 六、更多资源

- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
- **标准 LoRA 训练**：[README_TRAIN_LORA_zh-CN.md](./README_TRAIN_LORA_zh-CN.md)
- **英文版**：[README_TRAIN_REWARD.md](./README_TRAIN_REWARD.md)

---

## 参考文献
<ol>
  <li id="ref1">Clark, Kevin, et al. "Directly fine-tuning diffusion models on differentiable rewards.". In ICLR 2024.</li>
  <li id="ref2">Prabhudesai, Mihir, et al. "Aligning text-to-image diffusion models with reward backpropagation." arXiv preprint arXiv:2310.03739 (2023).</li>
</ol>
