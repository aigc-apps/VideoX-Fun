# Qwen-Image Model Setup Guide

## a. Model Links and Storage Locations

**Required Files:**

| Name | Storage | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| Qwen-Image | [🤗Link](https://huggingface.co/Qwen/Qwen-Image) | [😄Link](https://modelscope.cn/models/Qwen/Qwen-Image) | Official Qwen-Image weights |
| Qwen-Image-Edit | [🤗Link](https://huggingface.co/Qwen/Qwen-Image-Edit) | [😄Link](https://modelscope.cn/models/Qwen/Qwen-Image-Edit) | Official Qwen-Image-Edit weights |
| Qwen-Image-Edit-2509 | [🤗Link](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) | [😄Link](https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2509) | Official Qwen-Image-Edit-2509 weights |

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

[Qwen-Image Edit](v1/qwenimage_chunked_loading_workflow_edit.json)

### 2. Full Model Loading (Optional)

[Qwen-Image Text to Image](v1/qwenimage_workflow_t2i.json)

[Qwen-Image Edit](v1/qwenimage_workflow_edit.json)