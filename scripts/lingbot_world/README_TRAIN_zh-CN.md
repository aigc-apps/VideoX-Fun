# LingBot-World I2V 训练指南

本文档提供 LingBot-World 相机可控 I2V 模型（基于 Wan2.2）的完整训练流程，包括环境配置、数据准备、分布式训练和推理测试。

> **说明**：LingBot-World 是一个基于 Wan2.2 I2V 双 Transformer 骨干的相机位姿可控图生视频模型。给定一张参考图、一段文本提示，**以及一条逐帧的相机轨迹**（`poses.npy` + `intrinsics.npy`），模型生成一段视点沿轨迹运动的视频。训练脚本是 `scripts/wan2.2/train.py` 的超集：加上 `--enable_camera_control` 后，模型类切换为 `WanTransformer3DModel_LingbotWorld`，数据集切换为 `LingbotImageVideoDataset`，并在 forward 前构造相机条件传入 transformer。其它一切（双 Transformer boundary、FSDP、bucket 采样、EMA…）与 Wan2.2 完全一致。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 数据集结构](#21-数据集结构)
  - [2.2 metadata.json 格式](#22-metadatajson-格式)
  - [2.3 相机轨迹文件](#23-相机轨迹文件)
  - [2.4 Intrinsics 约定](#24-intrinsics-约定)
  - [2.5 相对路径与绝对路径使用方案](#25-相对路径与绝对路径使用方案)
- [三、全量参数训练](#三全量参数训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始（FSDP）](#32-快速开始fsdp)
  - [3.3 训练常用参数解析](#33-训练常用参数解析)
  - [3.4 可训练模块选择](#34-可训练模块选择)
  - [3.5 Boundary Type（高噪声 / 低噪声）](#35-boundary-type高噪声--低噪声)
  - [3.6 训练流程原理](#36-训练流程原理)
- [四、推理测试](#四推理测试)
  - [4.1 加载训练好的权重](#41-加载训练好的权重)
  - [4.2 图生视频（I2V）推理](#42-图生视频i2v推理)
- [五、更多资源](#五更多资源)

---

## 一、环境配置

**方式 1：使用 requirements.txt**

```bash
pip install -r requirements.txt
```

**方式 2：手动安装依赖**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image scipy
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

> **说明**：`scipy` 是必需的，LingBot 相机工具使用 `Slerp` / `interp1d` 进行位姿插值。

---

## 二、数据准备

### 2.1 数据集结构

LingBot-World 训练要求每个视频样本都配套一个**相机轨迹目录**，其中包含 `poses.npy` 和 `intrinsics.npy`。

```
📦 datasets/
├── 📂 lingbot_world/
│   ├── 📂 videos/
│   │   ├── 📄 clip_000001.mp4
│   │   ├── 📄 clip_000002.mp4
│   │   └── 📄 ...
│   ├── 📂 actions/
│   │   ├── 📂 clip_000001/
│   │   │   ├── 📄 poses.npy
│   │   │   └── 📄 intrinsics.npy
│   │   ├── 📂 clip_000002/
│   │   │   ├── 📄 poses.npy
│   │   │   └── 📄 intrinsics.npy
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

仓库内 `asset/lingbot_demo/`（推理脚本 `examples/lingbot_world/predict_i2v.py` 使用）就是一份可用的样例目录，可直接对照。

也可以直接从 ModelScope 下载现成的演示数据集（7 个 832x480 视频 + 配套相机轨迹，可直接用于训练流程冒烟测试）：

```bash
modelscope download --dataset PAI/X-Fun-Videos-Lingbot-Demo --local_dir datasets/X-Fun-Videos-Lingbot-Demo
```

### 2.2 metadata.json 格式

标注文件在标准 Wan2.2 字段的基础上新增一个字段：`action_path`。

```json
[
  {
    "file_path": "videos/clip_000001.mp4",
    "action_path": "actions/clip_000001",
    "text": "视频展示了一次穿越奇幻丛林的飞行……",
    "type": "video",
    "width": 832,
    "height": 480
  },
  {
    "file_path": "videos/clip_000002.mp4",
    "action_path": "actions/clip_000002",
    "text": "第一人称无人机镜头，向一座石头城堡下降……",
    "type": "video",
    "width": 832,
    "height": 480
  }
]
```

**关键字段说明**：
- `file_path`：视频路径（相对或绝对路径）。
- `action_path`：存放 `poses.npy` 与 `intrinsics.npy` 的目录。相对路径基于 `--action_data_root` 解析（未设置时回退到 `--train_data_dir`）。
- `text`：视频描述（英文提示词）。
- `type`：数据类型，固定为 `"video"`。图像样本或缺失 `action_path` 的视频样本会被相机注入路径静默跳过，退化为普通 Wan2.2 forward。
- `width` / `height`：视频宽高（**最好提供**，用于分桶训练）。
  - 可以使用 `scripts/process_json_add_width_and_height.py` 对无 width / height 字段的 json 进行提取。

### 2.3 相机轨迹文件

两个文件都是普通 NumPy 数组，格式与推理脚本 `examples/lingbot_world/predict_i2v.py` 完全一致：

- `poses.npy`：`float32`，形状 `[F, 4, 4]`。每帧相机→世界矩阵，采用 **OpenCV 坐标系**（x 右、y 下、z 前）。`F` 至少要不小于 `--video_sample_n_frames`，且 `poses[i]` 必须对应视频的第 `i` 帧。
- `intrinsics.npy`：`float32`，形状 `[F, 4]` 或 `[4]`。每行 `(fx, fy, cx, cy)` 单位为像素，标定分辨率见 [2.4](#24-intrinsics-约定)。与推理保持一致，只使用第一行。

**帧采样对齐**：
- `ImageVideoDataset` 会按 `min(video_sample_n_frames, video_length // video_sample_stride)` 从视频中截取连续片段，随机偏移开头。
- `LingbotImageVideoDataset` 用**相同**的帧索引切片 `poses.npy`（`c2ws[batch_index]`），保证相机轨迹与所采 RGB 片段始终对齐。
- 训练辅助函数 `prepare_lingbot_dit_cond_dict_from_c2ws` 再把这些位姿插值到 `lat_f = (F - 1) // 4 + 1` 个隐空间帧，并生成 plücker embedding token 网格 —— 与推理路径完全一致。

如果某个样本的 `poses.npy` 长度不够 `int(batch_index.max()) + 1`，该样本的相机轨迹会被丢弃，transformer 走无相机路径。

### 2.4 Intrinsics 约定

`intrinsics.npy` 使用的是**原始拍摄分辨率**的内参，而不是训练分辨率。每个带相机控制的视频条目**必须**在标注文件里通过 `intrinsics_org_height` / `intrinsics_org_width` 逐样本声明该标定分辨率：

```json
{
  "file_path": "videos/xxx.mp4",
  "action_path": "actions/xxx",
  "type": "video",
  "intrinsics_org_height": 480,
  "intrinsics_org_width": 832
}
```

训练时内参会由 `videox_fun/data/utils.py` 中的 `get_Ks_transformed` 按当前 bucket 尺寸自动缩放。**不要**手动修改 `intrinsics.npy` 里的数值，只需把这两个字段设为你标定时的分辨率即可。由于是逐样本声明，不同标定分辨率的数据集可以自由混用。

> **必填。** 每个带相机控制的样本都必须成对提供这两个字段。带相机轨迹却缺少它们的样本会在**数据加载时被跳过**（数据集会自动重采另一条），不会进入训练步骤。（旧的 `--intrinsics_org_height/width` CLI 参数已移除。）

### 2.5 相对路径与绝对路径使用方案

**相对路径**：

如果数据的路径为相对路径，则在训练脚本中设置：

```bash
export DATASET_NAME="datasets/lingbot_world/"
export DATASET_META_NAME="datasets/lingbot_world/metadata.json"
```

`metadata.json` 中的 `action_path` 会基于 `--action_data_root`（默认 `$DATASET_NAME`）解析。

**绝对路径**：

如果数据的路径为绝对路径，则在训练脚本中设置：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/lingbot_world/metadata.json"
```

如果相机轨迹与视频不在同一根目录下，可以额外通过 `--action_data_root` 指定。

> 💡 **建议**：如果数据集较小且存储在本地，推荐使用相对路径；如果数据集存储在外部存储（如 NAS、OSS）或多个机器共享存储，推荐使用绝对路径。

---

## 三、全量参数训练

### 3.1 下载预训练模型

推荐从已发布的 LingBot-World 权重出发（与推理保持一致）：

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# LingBot-World 相机可控基础权重（目录结构与 Wan2.2-I2V-A14B 一致）
modelscope download --model your-org/lingbot-world-base-cam --local_dir models/Diffusion_Transformer/lingbot-world-base-cam
```

也可以从原始 Wan2.2-I2V-A14B 出发。此时 LingBot 特有的层（`cam_injector_*`、`cam_scale_layer`、`cam_shift_layer`、`patch_embedding_wancamctrl`、`c2ws_hidden_states_layer{1,2}`）会随机初始化，**必须**加入 `--trainable_modules`（详见 [3.4](#34-可训练模块选择)）。

配置文件仍然是 `config/wan2.2/wan_civitai_i2v.yaml`：LingBot-World 复用了 Wan2.2 I2V 的 VAE / 文本编码器 / 双 transformer 拓扑。

### 3.2 快速开始（FSDP）

在按 **2.1 数据集结构** 准备好数据（或直接下载演示数据集 `PAI/X-Fun-Videos-Lingbot-Demo`）、按 **3.1 下载预训练模型** 下载好权重后，直接复制以下启动指令即可运行。

LingBot-World 推荐使用 FSDP 训练。因为 LingBot 用 `LingbotWorldWanAttentionBlock` 替换了 `WanAttentionBlock`（增加了 4 个 `cam_*` linear），FSDP 的 transformer 包装类需要设成 LingBot 的 block，以保证相机注入参数与本 block 的注意力 / FFN 处于同一个 FSDP 单元。仓库自带的 `scripts/lingbot_world/train.sh` 用的是普通 `accelerate launch`（未开 FSDP），训练**高噪声**分支、分辨率 640 —— 显存不够时可改用下方的 FSDP 参数。

**LingBot-World I2V 训练示例**（低噪声分支）：

```sh
export MODEL_NAME="models/Diffusion_Transformer/lingbot-world-base-cam"
export DATASET_NAME="datasets/X-Fun-Videos-Lingbot-Demo"
export DATASET_META_NAME="datasets/X-Fun-Videos-Lingbot-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=LingbotWorldWanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/lingbot_world/train.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=832 \
  --video_sample_size=832 \
  --token_sample_size=832 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_lingbot_world_i2v_low" \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --boundary_type="low" \
  --train_mode="i2v" \
  --enable_camera_control \
  --trainable_modules "cam_injector_layer1" "cam_injector_layer2" "cam_scale_layer" "cam_shift_layer" "patch_embedding_wancamctrl" "c2ws_hidden_states_layer1" "c2ws_hidden_states_layer2" \
  --resume_from_checkpoint=latest
```

### 3.3 训练常用参数解析

LingBot-World 继承了 `scripts/wan2.2/train.py` 的所有参数，此处只列出 LingBot 新增的参数 —— 其它参数请参考 [scripts/wan2.2/README_TRAIN_zh-CN.md](../wan2.2/README_TRAIN_zh-CN.md#33-训练常用参数解析)。

| 参数 | 说明 | 示例值 |
|-----|------|--------|
| `--enable_camera_control` | 总开关。开启后：切换到 `WanTransformer3DModel_LingbotWorld` + `LingbotImageVideoDataset`，并在 forward 前构造 `dit_cond_dict`。 | - |
| `--action_data_root` | 相对 `action_path` 的根目录。未设置时回退到 `--train_data_dir`。 | `None` |
| `intrinsics_org_height` / `intrinsics_org_width`（标注字段） | `intrinsics.npy` 的逐样本标定分辨率（像素）。在**标注文件**里声明，**非** CLI 参数。每个带相机控制的样本必填。 | 480 / 832 |
| `--control_type` | LingBot-World 控制类型。目前只支持 `cam`（6 维 plücker）。 | `cam` |

**说明**：
- `train_batch_size=1` 是当前推荐（且经过验证的）配置。相机轨迹长度可能因样本而异，batch 字典里以 python 列表形式携带，训练循环消费的是 index 0。如需更大的等效 batch，请调大 `--gradient_accumulation_steps`。
- 不开启 `--enable_camera_control` 时，`scripts/lingbot_world/train.py` 的行为与 `scripts/wan2.2/train.py` 完全一致。

### 3.4 可训练模块选择

`--trainable_modules` 通过子串匹配参数名。两种常见配方：

**（a）只训练 LingBot 相机注入层** —— 从官方 LingBot 权重继续微调，或者从 Wan2.2 I2V 冷启动 LingBot 层时都推荐：

```
--trainable_modules \
    "cam_injector_layer1" "cam_injector_layer2" \
    "cam_scale_layer" "cam_shift_layer" \
    "patch_embedding_wancamctrl" \
    "c2ws_hidden_states_layer1" "c2ws_hidden_states_layer2"
```

**（b）全量微调** —— 所有参数均可训练：

```
--trainable_modules "."
```

建议配合 FSDP 使用以节省显存。

### 3.5 Boundary Type（高噪声 / 低噪声）

Wan2.2 采用双 Transformer 结构：**高噪声** transformer 生成粗结构，**低噪声** transformer 细化细节，在 `boundary`（默认 0.900）处切分。

- `--boundary_type=low`：训练低噪声分支，timestep 采样于 `[0, boundary * T]`。
- `--boundary_type=high`（`train.sh` 自带的配置）：训练高噪声分支，timestep 采样于 `[boundary * T, T]`。
- `--boundary_type=full`：训练单个 transformer 覆盖 `[0, T]`（LingBot 一般不用）。

要覆盖完整流水线，分别用 `--boundary_type=low` 和 `--boundary_type=high` 各跑一次。推理时通过 `transformer_path`（低）和 `transformer_high_path`（高）分别喂给推理脚本，或者把 `model_name` 指向按 `config/wan2.2/wan_civitai_i2v.yaml` 布局保存两个子目录的目录。

### 3.6 训练流程原理

相对于原生 Wan2.2 I2V trainer，训练脚本只在以下四处做了改动：

1. **模型类切换** —— `scripts/lingbot_world/train.py` 在 `--enable_camera_control` 打开时选择 `WanTransformer3DModel_LingbotWorld`。该类继承自 `Wan2_2Transformer3DModel`，追加了 `patch_embedding_wancamctrl` / `c2ws_hidden_states_layer{1,2}`（全局 plücker → hidden state 映射），以及 `LingbotWorldWanAttentionBlock` 内的 `cam_injector_layer{1,2}` / `cam_scale_layer` / `cam_shift_layer`。
2. **数据集扩展** —— `LingbotImageVideoDataset` 是 `ImageVideoDataset` 的直接子类，从样本的 `action_path` 读入 `poses.npy` / `intrinsics.npy`，用**相同**的帧索引切片位姿数组，并将结果写入 `sample["action_c2ws"]` / `sample["action_intrinsics"]`。
3. **相机条件构造** —— 训练循环中对每个含轨迹的样本调用 `prepare_lingbot_dit_cond_dict_from_c2ws(...)`。这个函数镜像了推理路径中 `prepare_lingbot_dit_cond_dict` 的所有步骤：位姿插值到 `lat_f = (frame_num - 1) // vae_temporal_ratio + 1` 帧、逐帧相对位姿、按 bucket 尺寸缩放内参、生成 plücker embedding、并打包为 `[1, C, lat_f, lat_h, lat_w]` 张量。
4. **Forward** —— 生成的 `dit_cond_dict` 通过 `transformer3d(..., dit_cond_dict=dit_cond_dict)` 传入模型。`WanTransformer3DModel_LingbotWorld.forward` 把 plücker 张量投影为每个 token 的相机隐状态，广播给每个 `LingbotWorldWanAttentionBlock`，后者在自注意力和交叉注意力之间执行 `(1 + cam_scale) * x + cam_shift` 调制。

其它一切（flow-matching loss、梯度裁剪、EMA、FSDP sharded state dict 存储）与 `scripts/wan2.2/train.py` 完全一致。

---

## 四、推理测试

### 4.1 加载训练好的权重

将 [examples/lingbot_world/predict_i2v.py](../../examples/lingbot_world/predict_i2v.py) 指向你的 checkpoint。两种方式：

**方式 A** —— 按 `wan_civitai_i2v.yaml` 中的子路径把两支 transformer 存到同一个 `model_name` 目录：

```
model_name/
├── low_noise_model/            # boundary_type=low 训练的产物
├── high_noise_model/           # boundary_type=high 训练的产物
├── Wan2.1_VAE.pth
├── models_t5_umt5-xxl-enc-bf16.pth
└── ...
```

然后在推理脚本里把 `model_name` 设为 `"/path/to/your/trained/lingbot-world"`。

**方式 B** —— 在推理脚本里单独覆盖两个 transformer 路径：

```python
transformer_path      = "output_dir_lingbot_world_i2v_low/checkpoint-5000/diffusion_pytorch_model.safetensors"
transformer_high_path = "output_dir_lingbot_world_i2v_high/checkpoint-5000/diffusion_pytorch_model.safetensors"
```

### 4.2 图生视频（I2V）推理

运行下面的命令进行单卡推理：

```bash
python examples/lingbot_world/predict_i2v.py
```

按需求编辑 `examples/lingbot_world/predict_i2v.py`。初次推理时重点关注以下参数：

```python
# 根据显存选择
GPU_memory_mode = "model_cpu_offload"
# 根据实际模型路径填写
model_name = "models/Diffusion_Transformer/lingbot-world-base-cam"
# 训练得到的低噪声权重路径，例如 "output_dir_lingbot_world_i2v_low/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = None
# 训练得到的高噪声权重路径
transformer_high_path = None
# 参考图（图生视频起始帧）
validation_image_start = "asset/lingbot_demo/image.jpg"
# 相机轨迹目录，包含 poses.npy 和 intrinsics.npy（与训练 action_path 相同格式）
action_path = "asset/lingbot_demo"
# 根据要生成的内容填写提示词
prompt = "The video presents a soaring journey through a fantasy jungle..."
# ...
```

`GPU_memory_mode` 选项、多卡并行推理配置等请参考 [scripts/wan2.2/README_TRAIN_zh-CN.md#四推理测试](../wan2.2/README_TRAIN_zh-CN.md#四推理测试) —— LingBot 推理脚本复用了标准的 `Wan2_2I2VPipeline`，所有相关 flag 语义一致。

---

## 五、更多资源

- **基础 Wan2.2 训练文档**：[scripts/wan2.2/README_TRAIN_zh-CN.md](../wan2.2/README_TRAIN_zh-CN.md)
- **推理参考**：[examples/lingbot_world/predict_i2v.py](../../examples/lingbot_world/predict_i2v.py)
- **相机 / plücker 工具函数**：[videox_fun/data/utils.py](../../videox_fun/data/utils.py) —— `prepare_lingbot_dit_cond_dict{,_from_c2ws}`
- **模型代码**：[videox_fun/models/wan_transformer3d_lingbot_world.py](../../videox_fun/models/wan_transformer3d_lingbot_world.py)
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
