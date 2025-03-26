# VideoX-Fun

😊 Welcome!

CogVideoX-Fun:
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/alibaba-pai/CogVideoX-Fun-5b)

Wan-Fun:
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/alibaba-pai/Wan2.1-Fun-1.3B-InP)

English | [简体中文](./README_zh-CN.md) | [日本語](./README_ja-JP.md)

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Video Result](#video-result)
- [How to use](#how-to-use)
- [Model zoo](#model-zoo)
- [Reference](#reference)
- [License](#license)

# Introduction
VideoX-Fun is a video generation pipeline that can be used to generate AI images and videos, as well as to train baseline and Lora models for Diffusion Transformer. We support direct prediction from pre-trained baseline models to generate videos with different resolutions, durations, and FPS. Additionally, we also support users in training their own baseline and Lora models to perform specific style transformations.

We will support quick pull-ups from different platforms, refer to [Quick Start](#quick-start).

What's New:
- Update Wan2.1-Fun-V1.0: Support I2V and Control models for 14B and 1.3B models, with support for start and end frame prediction. [2025.03.26]
- Update CogVideoX-Fun-V1.5: Upload I2V model and related training/prediction code. [2024.12.16]
- Reward Lora Support: Train Lora using reward backpropagation techniques to optimize generated videos, making them better aligned with human preferences. [More Information](scripts/README_TRAIN_REWARD.md). New version of the control model supports various control conditions such as Canny, Depth, Pose, MLSD, etc. [2024.11.21]
- Diffusers Support: CogVideoX-Fun Control is now supported in diffusers. Thanks to [a-r-r-o-w](https://github.com/a-r-r-o-w) for contributing support in this [PR](https://github.com/huggingface/diffusers/pull/9671). Check out the [documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox) for more details. [2024.10.16]
- Update CogVideoX-Fun-V1.1: Retrain i2v model, add Noise to increase the motion amplitude of the video. Upload control model training code and Control model. [2024.09.29]
- Update CogVideoX-Fun-V1.0: Initial code release! Now supports Windows and Linux. Supports video generation at arbitrary resolutions from 256x256x49 to 1024x1024x49 for 2B and 5B models. [2024.09.18]

Function：
- [Data Preprocessing](#data-preprocess)
- [Train DiT](#dit-train)
- [Video Generation](#video-gen)

Our UI interface is as follows:
![ui](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/ui.jpg)

# Quick Start
### 1. Cloud usage: AliyunDSW/Docker
#### a. From AliyunDSW
DSW has free GPU time, which can be applied once by a user and is valid for 3 months after applying.

Aliyun provide free GPU time in [Freetier](https://free.aliyun.com/?product=9602825&crowd=enterprise&spm=5176.28055625.J_5831864660.1.e939154aRgha4e&scm=20140722.M_9974135.P_110.MO_1806-ID_9974135-MID_9974135-CID_30683-ST_8512-V_1), get it and use in Aliyun PAI-DSW to start CogVideoX-Fun within 5min!

[![DSW Notebook](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/dsw.png)](https://gallery.pai-ml.com/#/preview/deepLearning/cv/cogvideox_fun)

#### b. From ComfyUI
Our ComfyUI is as follows, please refer to [ComfyUI README](comfyui/README.md) for details.
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/cogvideoxfunv1_workflow_i2v.jpg)

#### c. From docker
If you are using docker, please make sure that the graphics card driver and CUDA environment have been installed correctly in your machine.

Then execute the following commands in this way:

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

### 2. Local install: Environment Check/Downloading/Installation
#### a. Environment Check
We have verified this repo execution on the following environment:

The detailed of Windows:
- OS: Windows 10
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU： Nvidia-3060 12G & Nvidia-3090 24G

The detailed of Linux:
- OS: Ubuntu 20.04, CentOS
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU：Nvidia-V100 16G & Nvidia-A10 24G & Nvidia-A100 40G & Nvidia-A100 80G

We need about 60GB available on disk (for saving weights), please check!

#### b. Weights
We'd better place the [weights](#model-zoo) along the specified path:

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

# Video Result

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

# How to Use

<h3 id="video-gen">1. Generation</h3>

#### a. GPU Memory Optimization
Since Wan2.1 has a very large number of parameters, we need to consider memory optimization strategies to adapt to consumer-grade GPUs. We provide `GPU_memory_mode` for each prediction file, allowing you to choose between `model_cpu_offload`, `model_cpu_offload_and_qfloat8`, and `sequential_cpu_offload`. This solution is also applicable to CogVideoX-Fun generation.

- `model_cpu_offload`: The entire model is moved to the CPU after use, saving some GPU memory.
- `model_cpu_offload_and_qfloat8`: The entire model is moved to the CPU after use, and the transformer model is quantized to float8, saving more GPU memory.
- `sequential_cpu_offload`: Each layer of the model is moved to the CPU after use. It is slower but saves a significant amount of GPU memory.

`qfloat8` may slightly reduce model performance but saves more GPU memory. If you have sufficient GPU memory, it is recommended to use `model_cpu_offload`.

#### b. Using ComfyUI
For details, refer to [ComfyUI README](comfyui/README.md).

#### c. Running Python Files
- **Step 1**: Download the corresponding [weights](#model-zoo) and place them in the `models` folder.
- **Step 2**: Use different files for prediction based on the weights and prediction goals. This library currently supports CogVideoX-Fun, Wan2.1, and Wan2.1-Fun. Different models are distinguished by folder names under the `examples` folder, and their supported features vary. Use them accordingly. Below is an example using CogVideoX-Fun:
  - **Text-to-Video**:
    - Modify `prompt`, `neg_prompt`, `guidance_scale`, and `seed` in the file `examples/cogvideox_fun/predict_t2v.py`.
    - Run the file `examples/cogvideox_fun/predict_t2v.py` and wait for the results. The generated videos will be saved in the folder `samples/cogvideox-fun-videos`.
  - **Image-to-Video**:
    - Modify `validation_image_start`, `validation_image_end`, `prompt`, `neg_prompt`, `guidance_scale`, and `seed` in the file `examples/cogvideox_fun/predict_i2v.py`.
    - `validation_image_start` is the starting image of the video, and `validation_image_end` is the ending image of the video.
    - Run the file `examples/cogvideox_fun/predict_i2v.py` and wait for the results. The generated videos will be saved in the folder `samples/cogvideox-fun-videos_i2v`.
  - **Video-to-Video**:
    - Modify `validation_video`, `validation_image_end`, `prompt`, `neg_prompt`, `guidance_scale`, and `seed` in the file `examples/cogvideox_fun/predict_v2v.py`.
    - `validation_video` is the reference video for video-to-video generation. You can use the following demo video: [Demo Video](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/play_guitar.mp4).
    - Run the file `examples/cogvideox_fun/predict_v2v.py` and wait for the results. The generated videos will be saved in the folder `samples/cogvideox-fun-videos_v2v`.
  - **Controlled Video Generation (Canny, Pose, Depth, etc.)**:
    - Modify `control_video`, `validation_image_end`, `prompt`, `neg_prompt`, `guidance_scale`, and `seed` in the file `examples/cogvideox_fun/predict_v2v_control.py`.
    - `control_video` is the control video extracted using operators such as Canny, Pose, or Depth. You can use the following demo video: [Demo Video](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1.1/pose.mp4).
    - Run the file `examples/cogvideox_fun/predict_v2v_control.py` and wait for the results. The generated videos will be saved in the folder `samples/cogvideox-fun-videos_v2v_control`.
- **Step 3**: If you want to integrate other backbones or Loras trained by yourself, modify `lora_path` and relevant paths in `examples/{model_name}/predict_t2v.py` or `examples/{model_name}/predict_i2v.py` as needed.

#### d. Using the Web UI
The web UI supports text-to-video, image-to-video, video-to-video, and controlled video generation (Canny, Pose, Depth, etc.). This library currently supports CogVideoX-Fun, Wan2.1, and Wan2.1-Fun. Different models are distinguished by folder names under the `examples` folder, and their supported features vary. Use them accordingly. Below is an example using CogVideoX-Fun:

- **Step 1**: Download the corresponding [weights](#model-zoo) and place them in the `models` folder.
- **Step 2**: Run the file `examples/cogvideox_fun/app.py` to access the Gradio interface.
- **Step 3**: Select the generation model on the page, fill in `prompt`, `neg_prompt`, `guidance_scale`, and `seed`, click "Generate," and wait for the results. The generated videos will be saved in the `sample` folder.

### 2. Model Training
A complete model training pipeline should include data preprocessing and Video DiT training. The training process for different models is similar, and the data formats are also similar:

<h4 id="data-preprocess">a. data preprocessing</h4>

We have provided a simple demo of training the Lora model through image data, which can be found in the [wiki](https://github.com/aigc-apps/CogVideoX-Fun/wiki/Training-Lora) for details.

A complete data preprocessing link for long video segmentation, cleaning, and description can refer to [README](cogvideox/video_caption/README.md) in the video captions section. 

If you want to train a text to image and video generation model. You need to arrange the dataset in this format.

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

The json_of_internal_datasets.json is a standard JSON file. The file_path in the json can to be set as relative path, as shown in below:
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

You can also set the path as absolute path as follow:
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

<h4 id="dit-train">b. Video DiT training </h4>
 
If the data format is relative path during data preprocessing, please set ```scripts/{model_name}/train.sh``` as follow.
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"
```

If the data format is absolute path during data preprocessing, please set ```scripts/train.sh``` as follow.
```
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/json_of_internal_datasets.json"
```

Then, we run scripts/train.sh.
```sh
sh scripts/train.sh
```

For details on some parameter settings:
Wan2.1-Fun can be found in [Readme Train](scripts/wan2.1_fun/README_TRAIN.md) and [Readme Lora](scripts/wan2.1_fun/README_TRAIN_LORA.md).
Wan2.1 can be found in [Readme Train](scripts/wan2.1/README_TRAIN.md) and [Readme Lora](scripts/wan2.1/README_TRAIN_LORA.md).
CogVideoX-Fun can be found in [Readme Train](scripts/cogvideox_fun/README_TRAIN.md) and [Readme Lora](scripts/cogvideox_fun/README_TRAIN_LORA.md).


# Model zoo

## 1. Wan2.1-Fun

V1.0:
| Name | Storage Space | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| Wan2.1-Fun-1.3B-InP | 19.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP) | Wan2.1-Fun-1.3B text-to-video weights, trained at multiple resolutions, supporting start and end frame prediction. |
| Wan2.1-Fun-14B-InP | 47.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-InP) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-InP) | Wan2.1-Fun-14B text-to-video weights, trained at multiple resolutions, supporting start and end frame prediction. |
| Wan2.1-Fun-1.3B-Control | 19.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-Control) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control) | Wan2.1-Fun-1.3B video control weights, supporting various control conditions such as Canny, Depth, Pose, MLSD, etc., and trajectory control. Supports multi-resolution (512, 768, 1024) video prediction at 81 frames, trained at 16 frames per second, with multilingual prediction support. |
| Wan2.1-Fun-14B-Control | 47.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-Control) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-Control) | Wan2.1-Fun-14B video control weights, supporting various control conditions such as Canny, Depth, Pose, MLSD, etc., and trajectory control. Supports multi-resolution (512, 768, 1024) video prediction at 81 frames, trained at 16 frames per second, with multilingual prediction support. |

## 2. Wan2.1

| Name  | Hugging Face | Model Scope | Description |
|--|--|--|--|
| Wan2.1-T2V-1.3B | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) | Wanxiang 2.1-1.3B text-to-video weights |
| Wan2.1-T2V-14B | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B) | Wanxiang 2.1-14B text-to-video weights |
| Wan2.1-I2V-14B-480P | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-InP) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P) | Wanxiang 2.1-14B-480P image-to-video weights |
| Wan2.1-I2V-14B-720P| [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-InP) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P) | Wanxiang 2.1-14B-720P image-to-video weights |

## 3. CogVideoX-Fun

V1.5:

| Name | Storage Space | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| CogVideoX-Fun-V1.5-5b-InP |  20.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.5-5b-InP) | Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024) and has been trained on 85 frames at a rate of 8 frames per second. |
| CogVideoX-Fun-V1.5-Reward-LoRAs | - | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.5-Reward-LoRAs) | The official reward backpropagation technology model optimizes the videos generated by CogVideoX-Fun-V1.5 to better match human preferences. ｜

V1.1:

| Name | Storage Space | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| CogVideoX-Fun-V1.1-2b-InP | 13.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-InP) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-InP) | Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second. |
| CogVideoX-Fun-V1.1-5b-InP | 20.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-InP) | Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second. Noise has been added to the reference image, and the amplitude of motion is greater compared to V1.0. |
| CogVideoX-Fun-V1.1-2b-Pose | 13.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-Pose) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-Pose) | Our official pose-control video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second.|
| CogVideoX-Fun-V1.1-2b-Control | 13.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-Control) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-Control) | Our official control video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second. Supporting various control conditions such as Canny, Depth, Pose, MLSD, etc.|
| CogVideoX-Fun-V1.1-5b-Pose | 20.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-Pose) | Our official pose-control video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second.|
| CogVideoX-Fun-V1.1-5b-Control | 20.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-Control) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-Control) | Our official control video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second. Supporting various control conditions such as Canny, Depth, Pose, MLSD, etc.|
| CogVideoX-Fun-V1.1-Reward-LoRAs | - | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-Reward-LoRAs) | The official reward backpropagation technology model optimizes the videos generated by CogVideoX-Fun-V1.1 to better match human preferences. ｜

<details>
  <summary>(Obsolete) V1.0:</summary>

| Name | Storage Space | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| CogVideoX-Fun-2b-InP | 13.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-2b-InP) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-2b-InP) | Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second. |
| CogVideoX-Fun-5b-InP | 20.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-5b-InP)| [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-5b-InP)| Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second. |
</details>

# Reference
- CogVideo: https://github.com/THUDM/CogVideo/
- EasyAnimate: https://github.com/aigc-apps/EasyAnimate
- Wan2.1: https://github.com/Wan-Video/Wan2.1/

# License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).

The CogVideoX-2B model (including its corresponding Transformers module and VAE module) is released under the [Apache 2.0 License](LICENSE).

The CogVideoX-5B model (Transformers module) is released under the [CogVideoX LICENSE](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE).
