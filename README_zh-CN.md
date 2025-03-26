# VideoX-Fun

😊 Welcome!

CogVideoX-Fun:
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/alibaba-pai/CogVideoX-Fun-5b)

Wan-Fun:
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/alibaba-pai/Wan2.1-Fun-1.3B-InP)

[English](./README.md) | 简体中文 | [日本語](./README_ja-JP.md)

# 目录
- [目录](#目录)
- [简介](#简介)
- [快速启动](#快速启动)
- [视频作品](#视频作品)
- [如何使用](#如何使用)
- [模型地址](#模型地址)
- [参考文献](#参考文献)
- [许可证](#许可证)

# 简介
VideoX-Fun是一个视频生成的pipeline，可用于生成AI图片与视频、训练Diffusion Transformer的基线模型与Lora模型，我们支持从已经训练好的基线模型直接进行预测，生成不同分辨率，不同秒数、不同FPS的视频，也支持用户训练自己的基线模型与Lora模型，进行一定的风格变换。

我们会逐渐支持从不同平台快速启动，请参阅 [快速启动](#快速启动)。

新特性：
- 更新Wan2.1-Fun-V1.0版本：支持14B与1.3B模型的I2V和Control模型，支持首尾图预测。[2025.03.26]
- 更新CogVideoX-Fun-V1.5版本：上传I2V模型与相关训练预测代码。[2024.12.16]
- 奖励Lora支持：通过奖励反向传播技术训练Lora，以优化生成的视频，使其更好地与人类偏好保持一致，[更多信息](scripts/README_TRAIN_REWARD.md)。新版本的控制模型，支持不同的控制条件，如Canny、Depth、Pose、MLSD等。[2024.11.21]
- diffusers支持：CogVideoX-Fun Control现在在diffusers中得到了支持。感谢 [a-r-r-o-w](https://github.com/a-r-r-o-w)在这个 [PR](https://github.com/huggingface/diffusers/pull/9671)中贡献了支持。查看[文档](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox)以了解更多信息。[2024.10.16]
- 更新CogVideoX-Fun-V1.1版本：重新训练i2v模型，添加Noise，使得视频的运动幅度更大。上传控制模型训练代码与Control模型。[2024.09.29]
- 更新CogVideoX-Fun-V1.0版本：创建代码！现在支持 Windows 和 Linux。支持2b与5b最大256x256x49到1024x1024x49的任意分辨率的视频生成。[2024.09.18]

功能概览：
- [数据预处理](#data-preprocess)
- [训练DiT](#dit-train)
- [模型生成](#video-gen)

我们的ui界面如下:
![ui](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/ui.jpg)

# 快速启动
### 1. 云使用: AliyunDSW/Docker
#### a. 通过阿里云 DSW
DSW 有免费 GPU 时间，用户可申请一次，申请后3个月内有效。

阿里云在[Freetier](https://free.aliyun.com/?product=9602825&crowd=enterprise&spm=5176.28055625.J_5831864660.1.e939154aRgha4e&scm=20140722.M_9974135.P_110.MO_1806-ID_9974135-MID_9974135-CID_30683-ST_8512-V_1)提供免费GPU时间，获取并在阿里云PAI-DSW中使用，5分钟内即可启动CogVideoX-Fun。

[![DSW Notebook](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/dsw.png)](https://gallery.pai-ml.com/#/preview/deepLearning/cv/cogvideox_fun)

#### b. 通过ComfyUI
我们的ComfyUI界面如下，具体查看[ComfyUI README](comfyui/README.md)。
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/cogvideoxfunv1_workflow_i2v.jpg)

#### c. 通过docker
使用docker的情况下，请保证机器中已经正确安装显卡驱动与CUDA环境，然后以此执行以下命令：

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# clone code
git clone https://github.com/aigc-apps/CogVideoX-Fun.git

# enter CogVideoX-Fun's dir
cd CogVideoX-Fun

# download weights
mkdir models/Diffusion_Transformer
mkdir models/Personalized_Model

# Please use the hugginface link or modelscope link to download the model.
# CogVideoX-Fun
# https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP
# https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-InP

# Wan
# https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-InP
# https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-InP
```

### 2. 本地安装: 环境检查/下载/安装
#### a. 环境检查
我们已验证该库可在以下环境中执行：

Windows 的详细信息：
- 操作系统 Windows 10
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU： Nvidia-3060 12G & Nvidia-3090 24G

Linux 的详细信息：
- 操作系统 Ubuntu 20.04, CentOS
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU：Nvidia-V100 16G & Nvidia-A10 24G & Nvidia-A100 40G & Nvidia-A100 80G

我们需要大约 60GB 的可用磁盘空间，请检查！

#### b. 权重放置
我们最好将[权重](#model-zoo)按照指定路径进行放置：

```
📦 models/
├── 📂 Diffusion_Transformer/
│   ├── 📂 CogVideoX-Fun-V1.1-2b-InP/
│   ├── 📂 CogVideoX-Fun-V1.1-5b-InP/
│   ├── 📂 Wan2.1-Fun-14B-InP
│   └── 📂 Wan2.1-Fun-1.3B-InP/
├── 📂 Personalized_Model/
│   └── your trained trainformer model / your trained lora model (for UI load)
```

# 视频作品

### Wan2.1-Fun-14B-InP && Wan2.1-Fun-1.3B-InP

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/bd72a276-e60e-4b5d-86c1-d0f67e7425b9" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/cb7aef09-52c2-4973-80b4-b2fb63425044" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/4e10d491-f1cf-4b08-a7c5-1e01e5418140" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/f7e363a9-be09-4b72-bccf-cce9c9ebeb9b" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/28f3e720-8acc-4f22-a5d0-ec1c571e9466" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fb6e4cb9-270d-47cd-8501-caf8f3e91b5c" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/989a4644-e33b-4f0c-b68e-2ff6ba37ac7e" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9c604fa7-8657-49d1-8066-b5bb198b28b6" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### Wan2.1-Fun-14B-Control && Wan2.1-Fun-1.3B-Control

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/f35602c4-9f0a-4105-9762-1e3a88abbac6" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/8b0f0e87-f1be-4915-bb35-2d53c852333e" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/972012c1-772b-427a-bce6-ba8b39edcfad" width="100%" controls autoplay loop></video>
     </td>
  <tr>
</table>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/53002ce2-dd18-4d4f-8135-b6f68364cabd" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fce43c0b-81fa-4ab2-9ca7-78d786f520e6" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/b208b92c-5add-4ece-a200-3dbbe47b93c3" width="100%" controls autoplay loop></video>
     </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/c6c5d557-9772-483e-ae47-863d8a26db4a" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/af617971-597c-4be4-beb5-f9e8aaca2d14" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/8411151e-f491-4264-8368-7fc3c5a6992b" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### CogVideoX-Fun-V1.1-5B

Resolution-1024

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/34e7ec8f-293e-4655-bb14-5e1ee476f788" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7809c64f-eb8c-48a9-8bdc-ca9261fd5434" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/8e76aaa4-c602-44ac-bcb4-8b24b72c386c" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/19dba894-7c35-4f25-b15c-384167ab3b03" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>


Resolution-768

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/0bc339b9-455b-44fd-8917-80272d702737" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/70a043b9-6721-4bd9-be47-78b7ec5c27e9" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/d5dd6c09-14f3-40f8-8b6d-91e26519b8ac" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9327e8bc-4f17-46b0-b50d-38c250a9483a" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

Resolution-512

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ef407030-8062-454d-aba3-131c21e6b58c" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7610f49e-38b6-4214-aa48-723ae4d1b07e" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/1fff0567-1e15-415c-941e-53ee8ae2c841" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/bcec48da-b91b-43a0-9d50-cf026e00fa4f" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### CogVideoX-Fun-V1.1-5B-Control

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/53002ce2-dd18-4d4f-8135-b6f68364cabd" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fce43c0b-81fa-4ab2-9ca7-78d786f520e6" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/b208b92c-5add-4ece-a200-3dbbe47b93c3" width="100%" controls autoplay loop></video>
     </td>
  <tr>
      <td>
          A young woman with beautiful clear eyes and blonde hair, wearing white clothes and twisting her body, with the camera focused on her face. High quality, masterpiece, best quality, high resolution, ultra-fine, dreamlike.
      </td>
      <td>
          A young woman with beautiful clear eyes and blonde hair, wearing white clothes and twisting her body, with the camera focused on her face. High quality, masterpiece, best quality, high resolution, ultra-fine, dreamlike.
      </td>
       <td>
          A young bear.
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ea908454-684b-4d60-b562-3db229a250a9" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/ffb7c6fc-8b69-453b-8aad-70dfae3899b9" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/d3f757a3-3551-4dcb-9372-7a61469813f5" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

# 如何使用

<h3 id="video-gen">1. 生成 </h3>

#### a、显存节省方案
由于Wan2.1的参数非常大，我们需要考虑显存节省方案，以节省显存适应消费级显卡。我们给每个预测文件都提供了GPU_memory_mode，可以在model_cpu_offload，model_cpu_offload_and_qfloat8，sequential_cpu_offload中进行选择。该方案同样适用于CogVideoX-Fun的生成。

- model_cpu_offload代表整个模型在使用后会进入cpu，可以节省部分显存。
- model_cpu_offload_and_qfloat8代表整个模型在使用后会进入cpu，并且对transformer模型进行了float8的量化，可以节省更多的显存。
- sequential_cpu_offload代表模型的每一层在使用后会进入cpu，速度较慢，节省大量显存。

qfloat8会部分降低模型的性能，但可以节省更多的显存。如果显存足够，推荐使用model_cpu_offload。

#### b、通过comfyui
具体查看[ComfyUI README](comfyui/README.md)。

#### c、运行python文件
- 步骤1：下载对应[权重](#model-zoo)放入models文件夹。
- 步骤2：根据不同的权重与预测目标使用不同的文件进行预测。当前该库支持CogVideoX-Fun、Wan2.1和Wan2.1-Fun，在examples文件夹下用文件夹名以区分，不同模型支持的功能不同，请视具体情况予以区分。以CogVideoX-Fun为例。
  - 文生视频：
    - 使用examples/cogvideox_fun/predict_t2v.py文件中修改prompt、neg_prompt、guidance_scale和seed。
    - 而后运行examples/cogvideox_fun/predict_t2v.py文件，等待生成结果，结果保存在samples/cogvideox-fun-videos文件夹中。
  - 图生视频：
    - 使用examples/cogvideox_fun/predict_i2v.py文件中修改validation_image_start、validation_image_end、prompt、neg_prompt、guidance_scale和seed。
    - validation_image_start是视频的开始图片，validation_image_end是视频的结尾图片。
    - 而后运行examples/cogvideox_fun/predict_i2v.py文件，等待生成结果，结果保存在samples/cogvideox-fun-videos_i2v文件夹中。
  - 视频生视频：
    - 使用examples/cogvideox_fun/predict_v2v.py文件中修改validation_video、validation_image_end、prompt、neg_prompt、guidance_scale和seed。
    - validation_video是视频生视频的参考视频。您可以使用以下视频运行演示：[演示视频](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/play_guitar.mp4)
    - 而后运行examples/cogvideox_fun/predict_v2v.py文件，等待生成结果，结果保存在samples/cogvideox-fun-videos_v2v文件夹中。
  - 普通控制生视频（Canny、Pose、Depth等）：
    - 使用examples/cogvideox_fun/predict_v2v_control.py文件中修改control_video、validation_image_end、prompt、neg_prompt、guidance_scale和seed。
    - control_video是控制生视频的控制视频，是使用Canny、Pose、Depth等算子提取后的视频。您可以使用以下视频运行演示：[演示视频](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1.1/pose.mp4)
    - 而后运行examples/cogvideox_fun/predict_v2v_control.py文件，等待生成结果，结果保存在samples/cogvideox-fun-videos_v2v_control文件夹中。
- 步骤3：如果想结合自己训练的其他backbone与Lora，则看情况修改examples/{model_name}/predict_t2v.py中的examples/{model_name}/predict_i2v.py和lora_path。

#### d、通过ui界面

webui支持文生视频、图生视频、视频生视频和普通控制生视频（Canny、Pose、Depth等）。当前该库支持CogVideoX-Fun、Wan2.1和Wan2.1-Fun，在examples文件夹下用文件夹名以区分，不同模型支持的功能不同，请视具体情况予以区分。以CogVideoX-Fun为例。

- 步骤1：下载对应[权重](#model-zoo)放入models文件夹。
- 步骤2：运行examples/cogvideox_fun/app.py文件，进入gradio页面。
- 步骤3：根据页面选择生成模型，填入prompt、neg_prompt、guidance_scale和seed等，点击生成，等待生成结果，结果保存在sample文件夹中。

### 2. 模型训练
一个完整的模型训练链路应该包括数据预处理和Video DiT训练。不同模型的训练流程类似，数据格式也类似：

<h4 id="data-preprocess">a.数据预处理</h4>
我们给出了一个简单的demo通过图片数据训练lora模型，详情可以查看[wiki](https://github.com/aigc-apps/CogVideoX-Fun/wiki/Training-Lora)。

一个完整的长视频切分、清洗、描述的数据预处理链路可以参考video caption部分的[README](cogvideox/video_caption/README.md)进行。

如果期望训练一个文生图视频的生成模型，您需要以这种格式排列数据集。
```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

json_of_internal_datasets.json是一个标准的json文件。json中的file_path可以被设置为相对路径，如下所示：
```json
[
    {
      "file_path": "train/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "train/00000002.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```

你也可以将路径设置为绝对路径：
```json
[
    {
      "file_path": "/mnt/data/videos/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "/mnt/data/train/00000001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```
<h4 id="dit-train">b. Video DiT训练 </h4>

如果数据预处理时，数据的格式为相对路径，则进入scripts/{model_name}/train.sh进行如下设置。
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"
```

如果数据的格式为绝对路径，则进入scripts/train.sh进行如下设置。
```
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/json_of_internal_datasets.json"
```

最后运行scripts/train.sh。
```sh
sh scripts/train.sh
```

关于一些参数的设置细节：
Wan2.1-Fun可以查看[Readme Train](scripts/wan2.1_fun/README_TRAIN.md)与[Readme Lora](scripts/wan2.1_fun/README_TRAIN_LORA.md)。
Wan2.1可以查看[Readme Train](scripts/wan2.1/README_TRAIN.md)与[Readme Lora](scripts/wan2.1/README_TRAIN_LORA.md)。
CogVideoX-Fun可以查看[Readme Train](scripts/cogvideox_fun/README_TRAIN.md)与[Readme Lora](scripts/cogvideox_fun/README_TRAIN_LORA.md)。


# 模型地址
## 1. Wan2.1-Fun

V1.0:
| 名称 | 存储空间 | Hugging Face | Model Scope | 描述 |
|--|--|--|--|--|
| Wan2.1-Fun-1.3B-InP | 19.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP) | Wan2.1-Fun-1.3B文图生视频权重，以多分辨率训练，支持首尾图预测。 |
| Wan2.1-Fun-14B-InP | 47.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-InP) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-InP) | Wan2.1-Fun-14B文图生视频权重，以多分辨率训练，支持首尾图预测。 |
| Wan2.1-Fun-1.3B-Control | 19.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-Control) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control)| Wan2.1-Fun-1.3B视频控制权重，支持不同的控制条件，如Canny、Depth、Pose、MLSD等，同时支持使用轨迹控制。支持多分辨率（512，768，1024）的视频预测，支持多分辨率（512，768，1024）的视频预测，以81帧、每秒16帧进行训练，支持多语言预测 |
| Wan2.1-Fun-14B-Control | 47.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-Control) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-Control)| Wan2.1-Fun-14B视频控制权重，支持不同的控制条件，如Canny、Depth、Pose、MLSD等，同时支持使用轨迹控制。支持多分辨率（512，768，1024）的视频预测，支持多分辨率（512，768，1024）的视频预测，以81帧、每秒16帧进行训练，支持多语言预测 |

## 2. Wan2.1

| 名称  | Hugging Face | Model Scope | 描述 |
|--|--|--|--|
| Wan2.1-T2V-1.3B | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) | 万象2.1-1.3B文生视频权重 |
| Wan2.1-T2V-14B | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B) | 万象2.1-14B文生视频权重 |
| Wan2.1-I2V-14B-480P | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-InP) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P) | 万象2.1-14B-480P图生视频权重 |
| Wan2.1-I2V-14B-720P| [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-InP) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P) | 万象2.1-14B-720P图生视频权重 |

## 3. CogVideoX-Fun

V1.5:

| 名称 | 存储空间 | Hugging Face | Model Scope | 描述 |
|--|--|--|--|--|
| CogVideoX-Fun-V1.5-5b-InP |  20.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.5-5b-InP) | 官方的图生视频权重。支持多分辨率（512，768，1024）的视频预测，以85帧、每秒8帧进行训练 |
| CogVideoX-Fun-V1.5-Reward-LoRAs | - | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.5-Reward-LoRAs) | 官方的奖励反向传播技术模型，优化CogVideoX-Fun-V1.5生成的视频，使其更好地符合人类偏好。 |


V1.1:

| 名称 | 存储空间 | Hugging Face | Model Scope | 描述 |
|--|--|--|--|--|
| CogVideoX-Fun-V1.1-2b-InP | 13.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-InP) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-InP) | 官方的图生视频权重。支持多分辨率（512，768，1024，1280）的视频预测，以49帧、每秒8帧进行训练 |
| CogVideoX-Fun-V1.1-5b-InP | 20.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-InP) | 官方的图生视频权重。添加了Noise，运动幅度相比于V1.0更大。支持多分辨率（512，768，1024，1280）的视频预测，以49帧、每秒8帧进行训练 |
| CogVideoX-Fun-V1.1-2b-Pose | 13.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-Pose) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-Pose) | 官方的姿态控制生视频权重。支持多分辨率（512，768，1024，1280）的视频预测，以49帧、每秒8帧进行训练 |
| CogVideoX-Fun-V1.1-2b-Control | 13.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-Control) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-Control) | 官方的控制生视频权重。支持多分辨率（512，768，1024，1280）的视频预测，以49帧、每秒8帧进行训练。支持不同的控制条件，如Canny、Depth、Pose、MLSD等 |
| CogVideoX-Fun-V1.1-5b-Pose | 20.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-Pose) | 官方的姿态控制生视频权重。支持多分辨率（512，768，1024，1280）的视频预测，以49帧、每秒8帧进行训练 |
| CogVideoX-Fun-V1.1-5b-Control | 20.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-Control) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-Control) | 官方的控制生视频权重。支持多分辨率（512，768，1024，1280）的视频预测，以49帧、每秒8帧进行训练。支持不同的控制条件，如Canny、Depth、Pose、MLSD等 |
| CogVideoX-Fun-V1.1-Reward-LoRAs | - | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-Reward-LoRAs) | 官方的奖励反向传播技术模型，优化CogVideoX-Fun-V1.1生成的视频，使其更好地符合人类偏好。 |

<details>
  <summary>(Obsolete) V1.0:</summary>

| 名称 | 存储空间 | Hugging Face | Model Scope | 描述 |
|--|--|--|--|--|
| CogVideoX-Fun-2b-InP | 13.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-2b-InP) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-2b-InP) | 官方的图生视频权重。支持多分辨率（512，768，1024，1280）的视频预测，以49帧、每秒8帧进行训练 |
| CogVideoX-Fun-5b-InP | 20.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-5b-InP) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-5b-InP) | 官方的图生视频权重。支持多分辨率（512，768，1024，1280）的视频预测，以49帧、每秒8帧进行训练 |
</details>

# 参考文献
- CogVideo: https://github.com/THUDM/CogVideo/
- EasyAnimate: https://github.com/aigc-apps/EasyAnimate
- Wan2.1: https://github.com/Wan-Video/Wan2.1/

# 许可证
本项目采用 [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).

CogVideoX-2B 模型 (包括其对应的Transformers模块，VAE模块) 根据 [Apache 2.0 协议](LICENSE) 许可证发布。

CogVideoX-5B 模型（Transformer 模块）在[CogVideoX许可证](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE)下发布.
