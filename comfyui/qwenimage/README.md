# Qwen-Image Model Setup Guide

## a. Model Links and Storage Locations

**Chunked loading is recommended** as it better aligns with ComfyUI's standard workflow.

### 1. Chunked Loading Weights (Recommended)

For chunked loading, it is recommended to directly download the Z-Image weights provided by ComfyUI official. Please organize the files according to the following directory structure:

**Core Model Files:**

| Component | File Name | 
|-----------|-----------| 
| Text Encoder | [`qwen_2.5_vl_7b_fp8_scaled.safetensors`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) |
| Diffusion Model | [`qwen_image_fp8_e4m3fn.safetensors`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors) | 
| VAE | [`qwen_image_vae.safetensors`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors) | 
| tokenizer | [`tokenizer`](https://huggingface.co/Qwen/Qwen-Image-Edit/tree/main/tokenizer) | 
| processor | [`processor`](https://huggingface.co/Qwen/Qwen-Image-Edit/tree/main/processor) | 

**ControlNet Model Files:**

| Name | Storage | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| Qwen-Image-2512-Fun-Controlnet-Union | - | [🤗Link](https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union) | [😄Link](https://modelscope.cn/models/PAI/Qwen-Image-2512-Fun-Controlnet-Union) | ControlNet weights for Qwen-Image-2512, supporting multiple control conditions such as Canny, Depth, Pose, MLSD, Scribble, etc. |

**Storage Location:**

```
📂 ComfyUI/
├── 📂 models/
│ ├── 📂 text_encoders/
│ │ └── qwen_2.5_vl_7b_fp8_scaled.safetensors
│ ├── 📂 diffusion_models/
│ │ └── qwen_image_fp8_e4m3fn.safetensors`
│ ├── 📂 vae/
│ │ └── qwen_image_vae.safetensors
│ ├── 📂 Fun_Models/
│ │ ├── qwen2_tokenizer/
│ │ └── qwen2_processor/
│ └── 📂 model_patches/
│   └── Qwen-Image-2512-Fun-Controlnet-Union.safetensors
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
| Qwen-Image | [🤗Link](https://huggingface.co/Qwen/Qwen-Image) | [😄Link](https://modelscope.cn/models/Qwen/Qwen-Image) | Official Qwen-Image weights |
| Qwen-Image-2512 | [🤗Link](https://huggingface.co/Qwen/Qwen-Image-2512) | [😄Link](https://modelscope.cn/models/Qwen/Qwen-Image-2512) | Official Qwen-Image weights |
| Qwen-Image-Edit | [🤗Link](https://huggingface.co/Qwen/Qwen-Image-Edit) | [😄Link](https://modelscope.cn/models/Qwen/Qwen-Image-Edit) | Official Qwen-Image-Edit weights |
| Qwen-Image-Edit-2509 | [🤗Link](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) | [😄Link](https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2509) | Official Qwen-Image-Edit-2509 weights |
| Qwen-Image-Edit-2511 | [🤗Link](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | [😄Link](https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2511) | Official Qwen-Image-Edit-2511 weights |

For full model loading, use the diffusers version of Qwen-Image Turbo and place the model in `ComfyUI/models/Fun_Models/`.

**Storage Location:**

```
📂 ComfyUI/
├── 📂 models/
│ └── 📂 Fun_Models/
|   ├── 📂 Qwen-Image-Edit/
|   └── 📂 Qwen-Image/
```

## b. ComfyUI Json Workflows

### 1. Chunked Loading (Recommended)

[Qwen-Image Text to Image](v1/qwenimage_chunked_loading_workflow_t2i.json)

[Qwen-Image Text to Image Control](v1/qwenimage_chunked_loading_workflow_t2i_control.json)

[Qwen-Image Text to Image Inpaint](v1/qwenimage_chunked_loading_workflow_t2i_inpaint.json)

[Qwen-Image Edit](v1/qwenimage_chunked_loading_workflow_edit.json)

[Qwen-Image Edit 2509](v1/qwenimage_chunked_loading_workflow_edit_2509.json)

[Qwen-Image Edit 2511](v1/qwenimage_chunked_loading_workflow_edit_2511.json)

### 2. Full Model Loading (Optional)

[Qwen-Image Text to Image](v1/qwenimage_workflow_t2i.json)

[Qwen-Image Text to Image Control](v1/qwenimage_workflow_t2i_control.json)

[Qwen-Image Text to Image Inpaint](v1/qwenimage_workflow_t2i_inpaint.json)

[Qwen-Image Edit](v1/qwenimage_workflow_edit.json)

[Qwen-Image Edit 2509](v1/qwenimage_workflow_edit_2509.json)

[Qwen-Image Edit 2511](v1/qwenimage_workflow_edit_2511.json)