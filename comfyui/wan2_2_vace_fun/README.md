# Wan2.2-VACE-Fun Model Setup Guide

## a. Model Links and Storage Locations

**Required Files:**

| Name | Storage Size | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| Wan2.2-VACE-Fun-A14B | 64.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B) | [😄Link](https://modelscope.cn/models/PAI/Wan2.2-VACE-Fun-A14B) | Control weights for Wan2.2 trained using the VACE scheme (based on the base model Wan2.2-T2V-A14B), supporting various control conditions such as Canny, Depth, Pose, MLSD, trajectory control, etc. It supports video generation by specifying the subject. It supports multi-resolution (512, 768, 1024) video prediction, and is trained with 81 frames at 16 FPS. It also supports multi-language prediction. |

**Storage Location:**

```
📂 ComfyUI/
├── 📂 models/
│ └── 📂 Fun_Models/
|   └── 📂 Wan2.2-VACE-Fun-A14B/
```

## b. ComfyUI Json Workflows

### 1. Chunked Loading (Recommended)

[Wan2.2-VACE-Fun Image to Video](v1/wan2.2_vace_fun_chunked_loading_workflow_i2v.json)

[Wan2.2-VACE-Fun Text to Video and Control](v1/wan2.2_vace_fun_chunked_loading_workflow_v2v_control.json)

[Wan2.2-VACE-Fun Text to Video and Control with Reference](v1/wan2.2_vace_fun_chunked_loading_workflow_v2v_control_ref.json)

[Wan2.2-VACE-Fun Subjects to Video](v1/wan2.2_vace_fun_chunked_loading_workflow_subjects.json)

[Wan2.2-VACE-Fun Subjects to Video with 4 steps](v1/wan2.2_vace_fun_chunked_loading_workflow_subjects_4steps_lora.json)

### 2. Full Model Loading (Optional)

[Wan2.2-VACE-Fun Image to Video](v1/wan2.2_vace_fun_workflow_i2v.json)

[Wan2.2-VACE-Fun Text to Video and Control](v1/wan2.2_vace_fun_workflow_v2v_control.json)

[Wan2.2-VACE-Fun Text to Video and Control with Reference](v1/wan2.2_vace_fun_workflow_v2v_control_ref.json)

[Wan2.2-VACE-Fun Subjects to Video](v1/wan2.2_vace_fun_chunked_loading_workflow_subjects.json)