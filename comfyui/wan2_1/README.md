# Wan2.1 Model Setup Guide

## a. Model Links and Storage Locations

**Required Files:**

| Name  | Hugging Face | Model Scope | Description |
|--|--|--|--|
| Wan2.1-T2V-1.3B | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) | Wan 2.1-1.3B text-to-video weights |
| Wan2.1-T2V-14B | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B) | Wan 2.1-14B text-to-video weights |
| Wan2.1-I2V-14B-480P | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-InP) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P) | Wan 2.1-14B-480P image-to-video weights |
| Wan2.1-I2V-14B-720P| [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-InP) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P) | Wan 2.1-14B-720P image-to-video weights |

**Storage Location:**

```
📂 ComfyUI/
├── 📂 models/
│ └── 📂 Fun_Models/
|   ├── 📂 Wan2.1-T2V-1.3B/
|   ├── 📂 Wan2.1-T2V-14B/
|   ├── 📂 Wan2.1-I2V-14B-480P/
│   └── 📂 Wan2.1-I2V-14B-720P/
```

## b. Node types
- **LoadWanModel**
    - Loads the Wan Model.
- **LoadWanLora**
    - Write the prompt for Wan-Fun model
- **WanI2VSampler**
    - Wan-Fun Sampler for Image to Video 
- **WanT2VSampler**
    - Wan-Fun Sampler for Text to Video

## c. ComfyUI Json Workflows

### i. Image to video generation
[Download link](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/wan/asset/v1.0/wan2.1_workflow_i2v.json) for wan-fun.

Our ui is shown as follow:
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/wan/asset/v1.0/wan2.1_workflow_i2v.jpg)

You can run the demo using following photo:
![demo image](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/firework.png)

### ii. Text to video generation
[Download link](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/wan/asset/v1.0/wan2.1_workflow_t2v.json) for wan-fun.

![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/wan/asset/v1.0/wan2.1_workflow_t2v.jpg)
