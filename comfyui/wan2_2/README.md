# Wan2.2 Model Setup Guide

## a. Model Links and Storage Locations

**Required Files:**

| Name | Hugging Face | Model Scope | Description |
|--|--|--|--|
| Wan2.2-TI2V-5B | [🤗Link](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B) | Wan2.2-5B Text-to-Video Weights |
| Wan2.2-T2V-14B | [🤗Link](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B) | Wan2.2-14B Text-to-Video Weights |
| Wan2.2-I2V-A14B | [🤗Link](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B) | Wan2.2-I2V-A14B Image-to-Video Weights |

**Storage Location:**

```
📂 ComfyUI/
├── 📂 models/
│ └── 📂 Fun_Models/
|   ├── 📂 Wan2.2-TI2V-5B/
|   ├── 📂 Wan2.2-T2V-14B/
|   └── 📂 Wan2.2-I2V-A14B/
```

## b. ComfyUI Json Workflows

### 1. Chunked Loading (Recommended)

[Wan2.2 Image to Video](v1/wan2.2_chunked_loading_workflow_i2v.json)

[Wan2.2 Text to Video](v1/wan2.2_chunked_loading_workflow_t2v.json)

[Wan2.2-5B Image to Video](v1/wan2.2_chunked_loading_workflow_i2v_5b.json)

[Wan2.2-5B Text to Video](v1/wan2.2_chunked_loading_workflow_t2v_5b.json)

### 2. Full Model Loading (Optional)

[Wan2.2 Image to Video](v1/wan2.2_workflow_i2v.json)

[Wan2.2 Text to Video](v1/wan2.2_workflow_t2v.json)

[Wan2.2-5B Image to Video](v1/wan2.2_workflow_i2v_5b.json)

[Wan2.2-5B Text to Video](v1/wan2.2_workflow_t2v_5b.json)