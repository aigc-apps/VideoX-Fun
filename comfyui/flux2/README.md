# FLUX.2-dev Model Setup Guide

## a. Model Links and Storage Locations

**Chunked loading is recommended** as it better aligns with ComfyUI's standard workflow.

### 1. Chunked Loading Weights (Recommended)

For chunked loading, it is recommended to directly download the FLUX.2-dev weights provided by ComfyUI official. Please organize the files according to the following directory structure:

**Core Model Files:**

| Component | File Name | 
|-----------|-----------| 
| Text Encoder | [`mistral_3_small_flux2_bf16.safetensors`](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors) |
| Diffusion Model | [`flux2_dev_fp8mixed.safetensors`](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors) | 
| VAE | [`flux2-vae.safetensors`](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors) | 
| tokenizer | [`tokenizer`](https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main/tokenizer) | 

**ControlNet Model Files:**

| Name | Storage | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| FLUX.2-dev-Fun-Controlnet-Union | - | [🤗Link](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union) | [😄Link](https://modelscope.cn/models/PAI/FLUX.2-dev-Fun-Controlnet-Union) | ControlNet weights for FLUX.2-dev, supporting multiple control conditions such as Canny, Depth, Pose, MLSD, Scribble, etc. |

**Storage Location:**

```
📂 ComfyUI/
├── 📂 models/
│ ├── 📂 text_encoders/
│ │ └── mistral_3_small_flux2_bf16.safetensors
│ ├── 📂 diffusion_models/
│ │ └── flux2_dev_fp8mixed.safetensors
│ ├── 📂 vae/
│ │ └── flux2-vae.safetensors
│ ├── 📂 Fun_Models/
│ │ └── flux2_tokenizer/
│ └── 📂 model_patches/
│   └── FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors
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

| Name | Storage | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| FLUX.2-dev | [🤗Link](https://huggingface.co/black-forest-labs/FLUX.2-dev) | [😄Link](https://modelscope.cn/models/black-forest-labs/FLUX.2-dev) | Official FLUX.2-dev weights |

For full model loading, use the diffusers version of FLUX.2-dev Turbo and place the model in `ComfyUI/models/Fun_Models/`.

**Storage Location:**

```
📂 ComfyUI/
├── 📂 models/
│ └── 📂 Fun_Models/
|   └── 📂 FLUX.2-dev/
```

## b. ComfyUI Json Workflows

### 1. Chunked Loading (Recommended)

[FLUX.2-dev Text to Image](v1/flux2_chunked_loading_workflow_t2i.json)

[FLUX.2-dev Text to Image Control](v1/flux2_chunked_loading_workflow_t2i_control.json)

[FLUX.2-dev Text to Image Inpaint](v1/flux2_chunked_loading_workflow_t2i_inpaint.json)

### 2. Full Model Loading (Optional)

[FLUX.2-dev Text to Image](v1/flux2_workflow_t2i.json)

[FLUX.2-dev Text to Image Control](v1/flux2_workflow_t2i_control.json)

[FLUX.2-dev Text to Image Inpaint](v1/flux2_workflow_t2i_inpaint.json)