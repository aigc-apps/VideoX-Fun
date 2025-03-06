# Enhance CogVideoX-Fun with Reward Backpropagation (Preference Optimization)
We explore the Reward Backpropagation technique <sup>[1](#ref1) [2](#ref2)</sup> to optimized the generated videos by [CogVideoX-Fun-V1.1](https://github.com/aigc-apps/CogVideoX-Fun) for better alignment with human preferences.
We provide pre-trained models (i.e. LoRAs) along with the training script. You can use these LoRAs to enhance the corresponding base model as a plug-in or train your own reward LoRA.

- [Enhance CogVideoX-Fun with Reward Backpropagation (Preference Optimization)](#enhance-cogvideox-fun-with-reward-backpropagation-preference-optimization)
  - [Demo](#demo)
    - [CogVideoX-Fun-V1.1-5B](#cogvideox-fun-v11-5b)
    - [CogVideoX-Fun-V1.1-2B](#cogvideox-fun-v11-2b)
  - [Model Zoo](#model-zoo)
  - [Inference](#inference)
  - [Training](#training)
    - [Setup](#setup)
    - [Important Args](#important-args)
  - [Limitations](#limitations)
  - [References](#references)


## Demo
### CogVideoX-Fun-V1.1-5B

<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
    <thead>
        <tr>
            <th style="text-align: center;" width="10%">Prompt</sup></th>
            <th style="text-align: center;" width="30%">CogVideoX-Fun-V1.1-5B</th>
            <th style="text-align: center;" width="30%">CogVideoX-Fun-V1.1-5B <br> HPSv2.1 Reward LoRA</th>
            <th style="text-align: center;" width="30%">CogVideoX-Fun-V1.1-5B <br> MPS Reward LoRA</th>
        </tr>
    </thead>
    <tr>
        <td>
            Pig with wings flying above a diamond mountain
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/6682f507-4ca2-45e9-9d76-86e2d709efb3" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/ec9219a2-96b3-44dd-b918-8176b2beb3b0" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/a75c6a6a-0b69-4448-afc0-fda3c7955ba0" width="100%" controls autoplay loop></video>
        </td>
    </tr>
    <tr>
        <td>
            A dog runs through a field while a cat climbs a tree
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/0392d632-2ec3-46b4-8867-0da1db577b6d" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/7d8c729d-6afb-408e-b812-67c40c3aaa96" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/dcd1343c-7435-4558-b602-9c0fa08cbd59" width="100%" controls autoplay loop></video>
        </td>
    </tr>
    <tr>
        <td>
            Crystal cake shimmering beside a metal apple
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/af0df8e0-1edb-4e2c-9a87-70df2b564aef" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/59b840f7-d33c-4972-8024-11a097f1c419" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/4a1d0af0-54e3-455c-9930-0789e2346fa0" width="100%" controls autoplay loop></video>
        </td>
    </tr>
    <tr>
        <td>
            Elderly artist with a white beard painting on a white canvas
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/99e44f9d-c770-48ce-8cc5-69fe36d757bc" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/9c106677-e4cb-4970-a1a2-a013fa6ce903" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/0a7b57ab-36a8-4fb6-bcfa-75e3878c55b7" width="100%" controls autoplay loop></video>
        </td>
    </tr>
</table>

### CogVideoX-Fun-V1.1-2B

<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
    <thead>
        <tr>
            <th style="text-align: center;" width="10%">Prompt</th>
            <th style="text-align: center;" width="30%">CogVideoX-Fun-V1.1-2B</th>
            <th style="text-align: center;" width="30%">CogVideoX-Fun-V1.1-2B <br> HPSv2.1 Reward LoRA</th>
            <th style="text-align: center;" width="30%">CogVideoX-Fun-V1.1-2B <br> MPS Reward LoRA</th>
        </tr>
    </thead>
    <tr>
        <td>
            A blue car drives past a white picket fence on a sunny day
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/274b0873-4fbd-4afa-94c0-22b23168f0a1" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/730f2ba3-4c54-44ce-ad5b-4eeca7ae844e" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/1b8eb777-0f17-46ef-9e7e-c8be7636e157" width="100%" controls autoplay loop></video>
        </td>
    </tr>
    <tr>
        <td>
            Blue jay swooping near a red maple tree
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/a14778d2-38ea-42c3-89a2-18164c48f3cf" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/90af433f-ab01-4341-9977-c675041d76d0" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/dafe8bf6-77ac-4934-8c9c-61c25088f80b" width="100%" controls autoplay loop></video>
        </td>
    </tr>
    <tr>
        <td>
          Yellow curtains swaying near a blue sofa
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/e8a445a4-781b-4b3f-899b-2cc24201f247" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/318cfb00-8bd1-407f-aaee-8d4220573b82" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/6b90e8a4-1754-42f4-b454-73510ed0701d" width="100%" controls autoplay loop></video>
        </td>
    </tr>
    <tr>
        <td>
            White tractor plowing near a green farmhouse
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/42d35282-e964-4c8b-aae9-a1592178493a" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/c9704bd4-d88d-41a1-8e5b-b7980df57a4a" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/7a785b34-4a5d-4491-9e03-c40cf953a1dc" width="100%" controls autoplay loop></video>
        </td>
    </tr>
</table>

> [!NOTE]
> The above test prompts are from <a href="https://github.com/KaiyueSun98/T2V-CompBench">T2V-CompBench</a>. All videos are generated with lora weight 0.7.

## Model Zoo
| Name | Base Model | Reward Model | Hugging Face | Description |
|--|--|--|--|--|
| CogVideoX-Fun-V1.1-5b-InP-HPS2.1.safetensors | CogVideoX-Fun-V1.1-5b | [HPS v2.1](https://github.com/tgxs002/HPSv2) | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs/resolve/main/CogVideoX-Fun-V1.1-5b-InP-HPS2.1.safetensors) | Official HPS v2.1 reward LoRA (`rank=128` and `network_alpha=64`) for CogVideoX-Fun-V1.1-5b-InP. It is trained with a batch size of 8 for 1,500 steps.|
| CogVideoX-Fun-V1.1-2b-InP-HPS2.1.safetensors | CogVideoX-Fun-V1.1-2b | [HPS v2.1](https://github.com/tgxs002/HPSv2) | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs/resolve/main/CogVideoX-Fun-V1.1-2b-InP-HPS2.1.safetensors) | Official HPS v2.1 reward LoRA (`rank=128` and `network_alpha=64`) for CogVideoX-Fun-V1.1-2b-InP. It is trained with a batch size of 8 for 3,000 steps.|
| CogVideoX-Fun-V1.1-5b-InP-MPS.safetensors | CogVideoX-Fun-V1.1-5b | [MPS](https://github.com/Kwai-Kolors/MPS) | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs/resolve/main/CogVideoX-Fun-V1.1-5b-InP-MPS.safetensors) | Official MPS reward LoRA (`rank=128` and `network_alpha=64`) for CogVideoX-Fun-V1.1-5b-InP. It is trained with a batch size of 8 for 5,500 steps.|
| CogVideoX-Fun-V1.1-2b-InP-MPS.safetensors | CogVideoX-Fun-V1.1-2b | [MPS](https://github.com/Kwai-Kolors/MPS) | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs/resolve/main/CogVideoX-Fun-V1.1-2b-InP-MPS.safetensors) | Official MPS reward LoRA (`rank=128` and `network_alpha=64`) for CogVideoX-Fun-V1.1-2b-InP. It is trained with a batch size of 8 for 16,000 steps.|

## Inference
We provide an example inference code to run CogVideoX-Fun-V1.1-5b-InP with its HPS2.1 reward LoRA.

```python
import torch
from diffusers import CogVideoXDDIMScheduler

from cogvideox.models.transformer3d import CogVideoXTransformer3DModel
from cogvideox.pipeline.pipeline_cogvideox_inpaint import CogVideoX_Fun_Pipeline_Inpaint
from cogvideox.utils.lora_utils import merge_lora
from cogvideox.utils.utils import get_image_to_video_latent, save_videos_grid

model_path = "alibaba-pai/CogVideoX-Fun-V1.1-5b-InP"
lora_path = "alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs/CogVideoX-Fun-V1.1-5b-InP-HPS2.1.safetensors"
lora_weight = 0.7

prompt = "Pig with wings flying above a diamond mountain"
sample_size = [512, 512]
video_length = 49

transformer = CogVideoXTransformer3DModel.from_pretrained_2d(model_path, subfolder="transformer").to(torch.bfloat16)
scheduler = CogVideoXDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
pipeline = CogVideoX_Fun_Pipeline_Inpaint.from_pretrained(
    model_path, transformer=transformer, scheduler=scheduler, torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()
pipeline = merge_lora(pipeline, lora_path, lora_weight)

generator = torch.Generator(device="cuda").manual_seed(42)
input_video, input_video_mask, _ = get_image_to_video_latent(None, None, video_length=video_length, sample_size=sample_size)
sample = pipeline(
    prompt,
    num_frames = video_length,
    negative_prompt = "bad detailed",
    height = sample_size[0],
    width = sample_size[1],
    generator = generator,
    guidance_scale = 7.0,
    num_inference_steps = 50,
    video = input_video,
    mask_video = input_video_mask,
).videos

save_videos_grid(sample, "samples/output.mp4", fps=8)
```

## Training
The [training code](./train_reward_lora.py) is based on [train_lora.py](./train_lora.py).
We provide [a shell script](./train_reward_lora.sh) to train the HPS v2.1 reward LoRA for CogVideoX-Fun-V1.1-2b-InP, 
which can be trained on a single A10 with 24GB VRAM. To further reduce the VRAM requirement, please read [Important Args](#important-args).

### Setup
Please read the [quick-start](https://github.com/aigc-apps/CogVideoX-Fun/blob/main/README.md#quick-start) section to setup the CogVideoX-Fun environment.
**If you're playing with HPS reward model**, please run the following script to install the dependencies:
```bash
# For HPS reward model only
pip install hpsv2
site_packages=$(python -c "import site; print(site.getsitepackages()[0])")
wget -O $site_packages/hpsv2/src/open_clip/factory.py https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/package/patches/hpsv2_src_open_clip_factory_patches.py
wget -O $site_packages/hpsv2/src/open_clip/ https://github.com/tgxs002/HPSv2/raw/refs/heads/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz
```

> [!NOTE]
> Since some models will be downloaded automatically from HuggingFace, Please run `HF_ENDPOINT=https://hf-mirror.com sh scripts/train_reward_lora.sh` if you cannot access to huggingface.com.

### Important Args
+ `rank`: The size of LoRA model. The higher the LoRA rank, the more parameters it has, and the more it can learn (including some unnecessary information).
Bt default, we set the rank to 128. You can lower this value to reduce training GPU memory and the LoRA file size.
+ `network_alpha`: A scaling factor changes how the LoRA affect the base model weight. In general, it can be set to half of the `rank`.
+ `prompt_path`: The path to the prompt file (in txt format, each line is a prompt) for sampling training videos. 
We randomly selected 701 prompts from [MovieGenBench](https://github.com/facebookresearch/MovieGenBench/blob/main/benchmark/MovieGenVideoBench.txt).
+ `train_sample_height` and `train_sample_width`: The resolution of the sampled training videos. We found 
training at a 256x256 resolution can generalize to any other resolution. Reducing the resolution can save GPU memory 
during training, but it is recommended that the resolution should be equal to or greater than the image input resolution of the reward model. 
Due to the resize and crop preprocessing operations, we suggest using a 1:1 aspect ratio.
+ `reward_fn` and `reward_fn_kwargs`: The reward model name and its keyword arguments. All supported reward models 
(Aesthetic Predictor [v2](https://github.com/christophschuhmann/improved-aesthetic-predictor)/[v2.5](https://github.com/discus0434/aesthetic-predictor-v2-5), 
[HPS](https://github.com/tgxs002/HPSv2) v2/v2.1, [PickScore](https://github.com/yuvalkirstain/PickScore) and [MPS](https://github.com/Kwai-Kolors/MPS)) 
can be found in [reward_fn.py](../cogvideox/reward/reward_fn.py). 
You can also customize your own reward model (e.g., combining aesthetic predictor with HPS).
+ `num_decoded_latents` and `num_sampled_frames`: The number of decoded latents (for VAE) and sampled frames (for the reward model). 
Since CogVideoX-Fun adopts the 3D casual VAE, we found decoding only the first latent to obtain the first frame for computing the reward 
not only reduces training memory usage but also prevents excessive reward optimization and maintains the dynamics of generated videos.

## Limitations
1. We observe after training to a certain extent, the reward continues to increase, but the quality of the generated videos does not further improve. 
   The model trickly learns some shortcuts (by adding artifacts in the background, i.e., adversarial patches) to increase the reward.
2. Currently, there is still a lack of suitable preference models for video generation. Directly using image preference models cannot 
   evaluate preferences along the temporal dimension (such as dynamism and consistency). Further more, We find using image preference models leads to a decrease 
   in the dynamism of generated videos. Although this can be mitigated by computing the reward using only the first frame of the decoded video, the impact still persists.

## References
<ol>
  <li id="ref1">Clark, Kevin, et al. "Directly fine-tuning diffusion models on differentiable rewards.". In ICLR 2024.</li>
  <li id="ref2">Prabhudesai, Mihir, et al. "Aligning text-to-image diffusion models with reward backpropagation." arXiv preprint arXiv:2310.03739 (2023).</li>
</ol>