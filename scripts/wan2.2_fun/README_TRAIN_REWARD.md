# Wan2.2-Fun-Reward-LoRAs
## Introduction
We explore the Reward Backpropagation technique <sup>[1](#ref1) [2](#ref2)</sup> to optimized the generated videos by [Wan2.2-Fun](https://github.com/aigc-apps/VideoX-Fun) for better alignment with human preferences.
We provide the following pre-trained models (i.e. LoRAs) along with [the training script](https://github.com/aigc-apps/VideoX-Fun/blob/main/scripts/wan2.2_fun/train_reward_lora.py). You can use these LoRAs to enhance the corresponding base model as a plug-in or train your own reward LoRA.

For more details, please refer to our [GitHub repo](https://github.com/aigc-apps/VideoX-Fun).

| Name | Base Model | Reward Model | Hugging Face | Description |
|--|--|--|--|--|
| Wan2.2-Fun-A14B-InP-high-noise-HPS2.1.safetensors | [Wan2.2-Fun-A14B-InP (high noise)](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP/tree/main/high_noise_model) | [HPS v2.1](https://github.com/tgxs002/HPSv2) | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-high-noise-HPS2.1.safetensors) | Official HPS v2.1 reward LoRA (`rank=128` and `network_alpha=64`) for Wan2.2-Fun-A14B-InP (high noise). It is trained with a batch size of 8 for 5,000 steps.|
| Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors | [Wan2.2-Fun-A14B-InP (low noise)](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP/tree/main/low_noise_model) | [MPS](https://github.com/Kwai-Kolors/MPS) | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors) | Official HPS v2.1 reward LoRA (`rank=128` and `network_alpha=64`) for Wan2.2-Fun-A14B-InP (low noise). It is trained with a batch size of 8 for 2,700 steps.|
| Wan2.2-Fun-A14B-InP-high-noise-MPS.safetensors | [Wan2.2-Fun-A14B-InP (high noise)](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP/tree/main/high_noise_model) | [HPS v2.1](https://github.com/tgxs002/HPSv2) | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-high-noise-MPS.safetensors) | Official MPS reward LoRA (`rank=128` and `network_alpha=64`) for Wan2.2-Fun-A14B-InP (high noise). It is trained with a batch size of 8 for 5,000 steps.|
| Wan2.2-Fun-A14B-InP-low-noise-MPS.safetensors | [Wan2.2-Fun-A14B-InP (low noise)](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP/tree/main/low_noise_model) | [MPS](https://github.com/Kwai-Kolors/MPS) | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-Reward-LoRAs/resolve/main/Wan2.1-Fun-14B-InP-MPS.safetensors) | Official MPS reward LoRA (`rank=128` and `network_alpha=64`) for Wan2.2-Fun-A14B-InP (low noise). It is trained with a batch size of 8 for xxx steps.|

> [!NOTE]
> We found that, MPS reward LoRA for the low-noise model converges significantly more slowly than on the other models, and may not deliver satisfactory results. Therefore, for the low-noise model, we recommend using HPSv2.1 reward LoRA.

## Demo
Please refer to [here](https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs#demo).

## Quick Start
Set `lora_path` along with `lora_weight` for the low noise reward LoRA, while specifying `lora_high_path` and `lora_high_weight` for high noise reward LoRA in [examples/wan2.2_fun/predict_t2v.py](https://github.com/aigc-apps/VideoX-Fun/blob/main/examples/wan2.1_fun/predict_t2v.py).

## Training
The training code is based on [train_lora.py](./train_lora.py). We provide a shell script to train the HPS v2.1 reward LoRA for the low noise model of Wan2.2-Fun-A14B-InP, which can be trained on a single 8*A100 node with 80GB VRAM. To train reward LoRA for the high noise model, Deepspeed Zero3 with CPU offload is required.

Please refer to [Setup](https://github.com/aigc-apps/VideoX-Fun/blob/main/scripts/cogvideox_fun/README_TRAIN_REWARD.md#setup) and [Important Args](https://github.com/aigc-apps/VideoX-Fun/blob/main/scripts/cogvideox_fun/README_TRAIN_REWARD.md#important-args) before training.


## Limitations
1. We observe after training to a certain extent, the reward continues to increase, but the quality of the generated videos does not further improve. 
   The model trickly learns some shortcuts (by adding artifacts in the background, i.e., adversarial patches) to increase the reward.
2. Currently, there is still a lack of suitable preference models for video generation. Directly using image preference models cannot 
   evaluate preferences along the temporal dimension (such as dynamism and consistency). Further more, We find using image preference models leads to a decrease 
   in the dynamism of generated videos. Although this can be mitigated by computing the reward using only the first frame of the decoded video, the impact still persists.

## Reference
<ol>
  <li id="ref1">Clark, Kevin, et al. "Directly fine-tuning diffusion models on differentiable rewards.". In ICLR 2024.</li>
  <li id="ref2">Prabhudesai, Mihir, et al. "Aligning text-to-image diffusion models with reward backpropagation." arXiv preprint arXiv:2310.03739 (2023).</li>
</ol>
