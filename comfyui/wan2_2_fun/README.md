# Wan2.2-Fun Model Setup Guide

## a. Model Links and Storage Locations

**Required Files:**

| Name | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| Wan2.2-Fun-A14B-InP | 64.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP) | [😄Link](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-InP) | Wan2.2-Fun-14B text-to-video generation weights, trained at multiple resolutions, supports start-end image prediction. |
| Wan2.2-Fun-A14B-Control | 64.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control) | [😄Link](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control)| Wan2.2-Fun-14B video control weights, supporting various control conditions such as Canny, Depth, Pose, MLSD, etc., and trajectory control. Supports multi-resolution (512, 768, 1024) video prediction at 81 frames, trained at 16 frames per second, with multilingual prediction support. |
| Wan2.2-Fun-A14B-Control-Camera | 64.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control-Camera) | [😄Link](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control-Camera)| Wan2.2-Fun-14B camera lens control weights. Supports multi-resolution (512, 768, 1024) video prediction, trained with 81 frames at 16 FPS, supports multilingual prediction. |
| Wan2.2-Fun-5B-InP | 23.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-InP) | [😄Link](https://modelscope.cn/models/PAI/Wan2.2-Fun-5B-InP) | Wan2.2-Fun-5B text-to-video weights trained at 121 frames, 24 FPS, supporting first/last frame prediction. |
| Wan2.2-Fun-5B-Control | 23.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control) | [😄Link](https://modelscope.cn/models/PAI/Wan2.2-Fun-5B-Control)| Wan2.2-Fun-5B video control weights, supporting control conditions like Canny, Depth, Pose, MLSD, and trajectory control. Trained at 121 frames, 24 FPS, with multilingual prediction support. |
| Wan2.2-Fun-5B-Control-Camera | 23.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control-Camera) | [😄Link](https://modelscope.cn/models/PAI/Wan2.2-Fun-5B-Control-Camera)| Wan2.2-Fun-5B camera lens control weights. Trained at 121 frames, 24 FPS, with multilingual prediction support. |

**Storage Location:**

```
📂 ComfyUI/
├── 📂 models/
│ └── 📂 Fun_Models/
|   ├── 📂 Wan2.2-Fun-A14B-InP/
|   ├── 📂 Wan2.2-Fun-A14B-Control/
|   └── 📂 Wan2.2-Fun-A14B-Control-Camera/
```

## b. ComfyUI Json Workflows

### 1. Chunked Loading (Recommended)

[Wan2.2 Image to Video](v1/wan2.2_fun_chunked_loading_workflow_i2v.json)

[Wan2.2 Text to Video and Control](v1/wan2.2_fun_chunked_loading_workflow_v2v_control.json)

[Wan2.2 Text to Video and Control with Pose Detect](v1/wan2.2_fun_chunked_loading_workflow_v2v_control_pose_ref.json)

[Wan2.2 Image to Video and Camera Control](v1/wan2.2_fun_chunked_loading_workflow_control_camera.json)

### 2. Full Model Loading (Optional)

[Wan2.2 Text to Video](v1/wan2.2_fun_workflow_t2v.json)

[Wan2.2 Image to Video](v1/wan2.2_fun_workflow_i2v.json)

[Wan2.2 Text to Video and Control](v1/wan2.2_fun_workflow_v2v_control.json)

[Wan2.2 Image to Video and Camera Control](v1/wan2.2_fun_workflow_control_camera.json)

[Wan2.2 Text to Video and Control with Pose Detect](v1/wan2.2_fun_workflow_v2v_control_pose_ref.json)

[Wan2.2 Text to Video and Control with Depth Detect](v1/wan2.2_fun_workflow_v2v_control_depth.json)

[Wan2.2 Text to Video and Control with Canny Detect](v1/wan2.2_fun_workflow_v2v_control_canny.json)