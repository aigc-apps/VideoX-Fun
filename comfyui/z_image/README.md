# Z-Image-Turbo Model Setup Guide

## a. Model Links and Storage Locations

**Chunked loading is recommended** as it better aligns with ComfyUI's standard workflow.

### 1. Chunked Loading Weights (Recommended)

For chunked loading, it is recommended to directly download the Z-Image weights provided by ComfyUI official. Please organize the files according to the following directory structure:

**Core Model Files:**

| Component | File Name | 
|-----------|-----------| 
| Text Encoder | [`qwen_3_4b.safetensors`](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) |
| Diffusion Model | [`z_image_turbo_bf16.safetensors`](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors) | 
| VAE | [`ae.safetensors`](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors) | 
| tokenizer(Qwen3-4B) | [`tokenizer`](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/tokenizer) | 

**ControlNet Model Files:**

| Name | Hugging Face | Model Scope | Description |
|------|--------------|-------------|-------------|
| Z-Image-Turbo-Fun-Controlnet-Union | [🤗Link](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union) | [😄Link](https://modelscope.cn/models/PAI/Z-Image-Turbo-Fun-Controlnet-Union) | ControlNet weights for Z-Image-Turbo, supporting multiple control conditions including Canny, Depth, Pose, MLSD, etc. |
| Z-Image-Turbo-Fun-Controlnet-Union-2.1 | [🤗Link](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | [😄Link](https://modelscope.cn/models/PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | Upgraded ControlNet weights for Z-Image-Turbo with additions at more layers and longer training time, supporting multiple control conditions including Canny, Depth, Pose, MLSD, etc. |

**Storage Location:**

```
📂 ComfyUI/
├── 📂 models/
│ ├── 📂 text_encoders/
│ │ └── qwen_3_4b.safetensors
│ ├── 📂 diffusion_models/
│ │ └── z_image_turbo_bf16.safetensors
│ ├── 📂 vae/
│ │ └── ae.safetensors
│ ├── 📂 Fun_Models/
│ │ └── Qwen3-4B/
│ └── 📂 model_patches/
│   └── Z-Image-Turbo-Fun-Controlnet-Union.safetensors
```

### 2. Preprocessing Weights (Optional)

If you want to use the control preprocessing nodes, you can download the preprocessing weights to `ComfyUI/custom_nodes/Fun_Models/Third_Party/`.

**Required Files:**

| File Name | Download Link | Purpose |
|-----------|---------------|---------|
| `yolox_l.onnx` | [Download](https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx) | YOLO Detection Model |
| `dw-ll_ucoco_384.onnx` | [Download](https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx) | DWPose Pose Estimation Model |
| `ZoeD_M12_N.pt` | [Download](https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt) | ZoeDepth Depth Estimation Model |

**Storage Location:**

```
📂 ComfyUI/
├── 📂 models/
│ └── 📂 Fun_Models/
│   └── 📂 Third_Party
│       ├── yolox_l.onnx
│       ├── dw-ll_ucoco_384.onnx
│       └── ZoeD_M12_N.pt
```

### 3. Full Model Loading (Optional)

If you prefer full model loading, you can directly download the diffusers weights.

**Required Files:**

| Name | Hugging Face | Model Scope | Description |
|------|--------------|-------------|-------------|
| Z-Image-Turbo | [🤗Link](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | [😄Link](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) | Official full weights for Z-Image-Turbo |

For full model loading, use the diffusers version of Z-Image Turbo and place the model in `ComfyUI/models/Fun_Models/`.

**Storage Location:**

```
📂 ComfyUI/
├── 📂 models/
│ └── 📂 Fun_Models/
│   └── 📂 Z-Image-Turbo
```

## b. ComfyUI Json Workflows

### 1. Chunked Loading (Recommended)

[Z Image Turbo Text to Image](v1/z_image_chunked_loading_workflow_t2i.json)

[Z Image Turbo Text to Image and Control](v1/z_image_chunked_loading_workflow_t2i_control.json)

[Z Image Turbo Text to Image and Control with Pose Detect](v1/z_image_chunked_loading_workflow_t2i_control_pose_process.json)

[Z Image Turbo Text to Image and Control with Depth Detect](v1/z_image_chunked_loading_workflow_t2i_control_depth_process.json)

[Z Image Turbo Text to Image and Control with Canny Detect](v1/z_image_chunked_loading_workflow_t2i_control_canny_process.json)

[Z Image Turbo Image to Image with Inpaint](v1/z_image_chunked_loading_workflow_i2i_inpaint.json)

### 2. Full Model Loading (Optional)

[Z Image Turbo Text to Image](v1/z_image_workflow_t2i.json)

[Z Image Turbo Text to Image and Control](v1/z_image_workflow_t2i_control.json)
