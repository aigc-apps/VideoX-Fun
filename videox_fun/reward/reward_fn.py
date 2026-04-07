import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torchvision.transforms as transforms
from einops import rearrange
from torchvision.datasets.utils import download_url

# All reward models.
__all__ = ["AestheticReward", "HPSReward", "PickScoreReward", "MPSReward", "HPSv3Reward", "VideoAlignReward"]


class BaseReward(ABC):
    """An base class for reward models. A custom Reward class must implement two functions below.
    """
    def __init__(self):
        """Define your reward model and image transformations (optional) here.
        """
        pass

    @abstractmethod
    def __call__(self, batch_frames: torch.Tensor, batch_prompt: Optional[list[str]]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given batch frames with shape `[B, C, T, H, W]` extracted from a list of videos and a list of prompts 
        (optional) correspondingly, return the loss and reward computed by your reward model (reduction by mean).
        """
        pass

    @abstractmethod
    def get_reward(self, batch_frames: torch.Tensor, batch_prompt: Optional[list[str]] = None) -> torch.Tensor:
        """Return per-sample rewards of shape [B], without any reduction across batch dimension."""
        pass


class AestheticReward(BaseReward):
    """Aesthetic Predictor [V2](https://github.com/christophschuhmann/improved-aesthetic-predictor) 
    and [V2.5](https://github.com/discus0434/aesthetic-predictor-v2-5) reward model.
    """
    def __init__(
        self,
        encoder_path="openai/clip-vit-large-patch14",
        predictor_path=None,
        version="v2",
        device="cpu",
        dtype=torch.float16,
        max_reward=10,
        loss_scale=0.1,
    ):
        from .aesthetic_v2_5_predictor import convert_v2_5_from_siglip
        from .aesthetic_v2_predictor import ImprovedAestheticPredictor

        self.encoder_path = encoder_path
        self.predictor_path = predictor_path
        self.version = version
        self.device = device
        self.dtype = dtype
        self.max_reward = max_reward
        self.loss_scale = loss_scale

        if self.version != "v2" and self.version != "v2.5":
            raise ValueError("Only v2 and v2.5 are supported.")
        if self.version == "v2":
            assert "clip-vit-large-patch14" in encoder_path.lower()
            self.model = ImprovedAestheticPredictor(encoder_path=self.encoder_path, predictor_path=self.predictor_path)
            # https://huggingface.co/openai/clip-vit-large-patch14/blob/main/preprocessor_config.json
            self.transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        elif self.version == "v2.5":
            assert "siglip-so400m-patch14-384" in encoder_path.lower()
            self.model, _ = convert_v2_5_from_siglip(encoder_model_name=self.encoder_path)
            # https://huggingface.co/google/siglip-so400m-patch14-384/blob/main/preprocessor_config.json
            self.transform = transforms.Compose([
                transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.requires_grad_(False)
    

    def __call__(self, batch_frames: torch.Tensor, batch_prompt: Optional[list[str]]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = self.get_reward(batch_frames, batch_prompt)
        if self.max_reward is None:
            loss_per_sample = (-1 * rewards) * self.loss_scale
        else:
            loss_per_sample = torch.abs(rewards - self.max_reward) * self.loss_scale
        return loss_per_sample.mean(), rewards.mean()

    def get_reward(self, batch_frames: torch.Tensor, batch_prompt: Optional[list[str]] = None) -> torch.Tensor:
        batch_frames = rearrange(batch_frames, "b c t h w -> t b c h w")
        total_rewards = []
        for frames in batch_frames:
            pixel_values = torch.stack([self.transform(frame) for frame in frames])
            pixel_values = pixel_values.to(self.device, dtype=self.dtype)
            if self.version == "v2":
                reward = self.model(pixel_values)
            elif self.version == "v2.5":
                reward = self.model(pixel_values).logits.squeeze(-1)
            total_rewards.append(reward)
        rewards = torch.stack(total_rewards, dim=0).mean(dim=0)
        return rewards


class HPSReward(BaseReward):
    """[HPS](https://github.com/tgxs002/HPSv2) v2 and v2.1 reward model.
    """
    def __init__(
        self,
        model_path=None,
        version="v2.0",
        device="cpu",
        dtype=torch.float16,
        max_reward=1,
        loss_scale=1,
    ):
        from hpsv2.src.open_clip import (create_model_and_transforms,
                                         get_tokenizer)

        self.model_path = model_path
        self.version = version
        self.device = device
        self.dtype = dtype
        self.max_reward = max_reward
        self.loss_scale = loss_scale

        self.model, _, _ = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision=self.dtype,
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        self.tokenizer = get_tokenizer("ViT-H-14")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        if version == "v2.0":
            url = "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Third_Party/HPS_v2_compressed.pt"
            filename = "HPS_v2_compressed.pt"
            md5 = "fd9180de357abf01fdb4eaad64631db4"
        elif version == "v2.1":
            url = "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Third_Party/HPS_v2.1_compressed.pt"
            filename = "HPS_v2.1_compressed.pt"
            md5 = "4067542e34ba2553a738c5ac6c1d75c0"
        else:
            raise ValueError("Only v2.0 and v2.1 are supported.")
        if self.model_path is None or not os.path.exists(self.model_path):
            download_url(url, torch.hub.get_dir(), md5=md5)
            model_path = os.path.join(torch.hub.get_dir(), filename)

        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.requires_grad_(False)
        self.model.eval()
    
    def __call__(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = self.get_reward(batch_frames, batch_prompt)
        if self.max_reward is None:
            loss_per_sample = (-1 * rewards) * self.loss_scale
        else:
            loss_per_sample = torch.abs(rewards - self.max_reward) * self.loss_scale
        return loss_per_sample.mean(), rewards.mean()

    def get_reward(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> torch.Tensor:
        assert batch_frames.shape[0] == len(batch_prompt)
        batch_frames = rearrange(batch_frames, "b c t h w -> t b c h w")
        total_rewards = []
        text_inputs = self.tokenizer(batch_prompt).to(device=self.device)

        for frames in batch_frames:
            image_inputs = torch.stack([self.transform(frame) for frame in frames])
            image_inputs = image_inputs.to(device=self.device, dtype=self.dtype)
            outputs = self.model(image_inputs, text_inputs)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            logits = image_features @ text_features.T
            reward = torch.diagonal(logits)
            total_rewards.append(reward)
        
        rewards = torch.stack(total_rewards, dim=0).mean(dim=0)
        return rewards


class PickScoreReward(BaseReward):
    """[PickScore](https://github.com/yuvalkirstain/PickScore) reward model.
    """
    def __init__(
        self,
        model_path="yuvalkirstain/PickScore_v1",
        device="cpu",
        dtype=torch.float16,
        max_reward=1,
        loss_scale=1,
    ):
        from transformers import AutoModel, AutoProcessor

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_reward = max_reward
        self.loss_scale = loss_scale

        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=self.dtype)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=self.dtype).eval().to(device)
        self.model.requires_grad_(False)
        self.model.eval()
     
    def __call__(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = self.get_reward(batch_frames, batch_prompt)
        if self.max_reward is None:
            loss_per_sample = (-1 * rewards) * self.loss_scale
        else:
            loss_per_sample = torch.abs(rewards - self.max_reward) * self.loss_scale
        return loss_per_sample.mean(), rewards.mean()

    def get_reward(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> torch.Tensor:
        assert batch_frames.shape[0] == len(batch_prompt)
        batch_frames = rearrange(batch_frames, "b c t h w -> t b c h w")
        total_rewards = []

        text_inputs = self.processor(
            text=batch_prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        for frames in batch_frames:
            image_inputs = torch.stack([self.transform(frame) for frame in frames])
            image_inputs = image_inputs.to(device=self.device, dtype=self.dtype)
            image_features = self.model.get_image_features(pixel_values=image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            image_features = image_features / torch.norm(image_features, dim=-1, keepdim=True)
            text_features = text_features / torch.norm(text_features, dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            reward = torch.diagonal(logits)
            total_rewards.append(reward)

        rewards = torch.stack(total_rewards, dim=0).mean(dim=0)
        return rewards


class MPSReward(BaseReward):
    """[MPS](https://github.com/Kwai-Kolors/MPS) reward model.
    """
    def __init__(
        self,
        model_path=None,
        device="cpu",
        dtype=torch.float16,
        max_reward=1,
        loss_scale=1,
    ):
        from transformers import AutoConfig, AutoTokenizer

        from .MPS.trainer.models.clip_model import CLIPModel

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things."
        self.max_reward = max_reward
        self.loss_scale = loss_scale

        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        url = "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Third_Party/MPS_overall.pth"
        filename = "MPS_overall.pth"
        md5 = "1491cbbbd20565747fe07e7572e2ac56"
        if self.model_path is None or not os.path.exists(self.model_path):
            download_url(url, torch.hub.get_dir(), md5=md5)
            model_path = os.path.join(torch.hub.get_dir(), filename)

        self.tokenizer = AutoTokenizer.from_pretrained(processor_name_or_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(processor_name_or_path)
        self.model = CLIPModel(config)
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.requires_grad_(False)
        self.model.eval()
    
    def _tokenize(self, caption):
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return input_ids
    
    def __call__(
        self,
        batch_frames: torch.Tensor,
        batch_prompt: list[str],
        batch_condition: Optional[list[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = self.get_reward(batch_frames, batch_prompt, batch_condition)
        if self.max_reward is None:
            loss_per_sample = (-1 * rewards) * self.loss_scale
        else:
            loss_per_sample = torch.abs(rewards - self.max_reward) * self.loss_scale
        return loss_per_sample.mean(), rewards.mean()

    def get_reward(
        self,
        batch_frames: torch.Tensor,
        batch_prompt: list[str],
        batch_condition: Optional[list[str]] = None
    ) -> torch.Tensor:
        if batch_condition is None:
            batch_condition = [self.condition] * len(batch_prompt)
        assert batch_frames.shape[0] == len(batch_prompt)
        batch_frames = rearrange(batch_frames, "b c t h w -> t b c h w")
        total_rewards = []

        text_inputs = self._tokenize(batch_prompt).to(self.device)
        condition_inputs = self._tokenize(batch_condition).to(self.device)

        for frames in batch_frames:
            image_inputs = torch.stack([self.transform(frame) for frame in frames])
            image_inputs = image_inputs.to(device=self.device, dtype=self.dtype)
            text_features, image_features = self.model(text_inputs, image_inputs, condition_inputs)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            reward = torch.diagonal(logits)
            total_rewards.append(reward)
        
        rewards = torch.stack(total_rewards, dim=0).mean(dim=0)
        return rewards


class HPSv3Reward(BaseReward):
    """[HPSv3](https://github.com/tgxs002/HPSv2) v3 reward model based on Qwen2-VL.
    """
    def __init__(
        self,
        config_path=None,
        checkpoint_path=None,
        device="cpu",
        dtype=torch.float16,
        max_reward=1,
        loss_scale=1,
    ):
        from .hpsv3_predictor import HPSv3RewardInferencer

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype = dtype
        self.max_reward = max_reward
        self.loss_scale = loss_scale

        self.inferencer = HPSv3RewardInferencer(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

    def __call__(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = self.get_reward(batch_frames, batch_prompt)
        print(rewards)
        if self.max_reward is None:
            loss_per_sample = (-1 * rewards) * self.loss_scale
        else:
            loss_per_sample = torch.abs(rewards - self.max_reward) * self.loss_scale
        return loss_per_sample.mean(), rewards.mean()

    @torch.no_grad()
    def get_reward(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> torch.Tensor:
        assert len(batch_frames) == len(batch_prompt)
        batch_frames = rearrange(batch_frames, "b c t h w -> t b c h w")
        total_rewards = []

        for frames in batch_frames:
            # Convert tensor frames to PIL images for HPSv3
            from PIL import Image
            pil_images = []
            for frame in frames:
                # frame shape: [C, H, W], value range [0, 1]
                frame_np = (frame.float().cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
                pil_images.append(Image.fromarray(frame_np))
            
            # Get rewards from HPSv3
            rewards_output = self.inferencer.reward(pil_images, batch_prompt)
            # Extract mu values (first element of each reward tuple)
            reward = torch.stack([r[0] for r in rewards_output])
            total_rewards.append(reward)

        rewards = torch.stack(total_rewards, dim=0).mean(dim=0)
        return rewards


class VideoAlignReward(BaseReward):
    def __init__(
        self,
        model_path=None,
        device="cpu",
        dtype=torch.float16,
        max_reward=1,
        loss_scale=1,
        reward_dim="Overall",
        fps=24,
        num_frames=None,
        use_norm=True,
    ):
        from .video_align_predictor import VideoVLMRewardInference

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_reward = max_reward
        self.loss_scale = loss_scale
        self.reward_dim = reward_dim  # "VQ", "MQ", "TA", or "Overall"
        self.fps = fps
        self.num_frames = num_frames
        self.use_norm = use_norm

        self.inferencer = VideoVLMRewardInference(
            load_from_pretrained=self.model_path,
            device=self.device,
            dtype=self.dtype,
        )

    def _save_frames_to_temp_video(self, frames: torch.Tensor, fps: float = 8.0) -> str:
        """Save tensor frames to a temporary video file in memory (tmpfs).
        
        Args:
            frames: Tensor of shape [T, C, H, W] with values in [0, 1]
            fps: Frames per second for the output video
            
        Returns:
            Path to the temporary video file
        """
        import tempfile
        import os
        import av
        
        # Use /dev/shm (tmpfs, RAM-based) to avoid disk IO, fallback to tempdir
        shm_dir = "/dev/shm"
        if os.path.exists(shm_dir) and os.access(shm_dir, os.W_OK):
            temp_dir = shm_dir
        else:
            temp_dir = tempfile.gettempdir()
        
        # Generate unique filename based on frame content hash
        frame_hash = hash((frames.shape, frames.sum().item(), frames[0].sum().item(), frames[-1].sum().item()))
        temp_video_path = os.path.join(temp_dir, f"videovlm_reward_temp_{os.getpid()}_{frame_hash}.mp4")
        
        # Convert frames to numpy: [T, C, H, W] -> [T, H, W, C]
        frames_np = (frames.float().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype('uint8')
        
        # Write video using PyAV with high quality settings
        t, h, w, c = frames_np.shape
        container = av.open(temp_video_path, mode='w')
        stream = container.add_stream('libx264', rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = 'yuv444p'  # Higher fidelity than yuv420p
        stream.options = {'crf': '10', 'preset': 'fast'}  # Low CRF = high quality
        
        for frame_data in frames_np:
            frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        
        return temp_video_path

    def __call__(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = self.get_reward(batch_frames, batch_prompt)
        if self.max_reward is None:
            loss_per_sample = (-1 * rewards) * self.loss_scale
        else:
            loss_per_sample = torch.abs(rewards - self.max_reward) * self.loss_scale
        return loss_per_sample.mean(), rewards.mean()

    @torch.no_grad()
    def get_reward(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> torch.Tensor:
        assert len(batch_frames) == len(batch_prompt)        
        total_rewards = []
        temp_video_paths = []
        
        try:
            for frames in batch_frames:
                # Save frames to temp video
                frames = rearrange(frames, "c t h w -> t c h w")
                temp_video_path = self._save_frames_to_temp_video(frames, fps=self.fps)
                temp_video_paths.append(temp_video_path)

            # Get rewards from VideoVLMRewardInference
            rewards_output = self.inferencer.reward(
                video_paths=temp_video_paths,
                prompts=batch_prompt,
                num_frames=self.num_frames,
                use_norm=self.use_norm,
            )

            for reward_dict in rewards_output:
                reward_value = reward_dict[self.reward_dim]
                total_rewards.append(torch.tensor(reward_value, device=self.device, dtype=self.dtype))
            
            rewards = torch.stack(total_rewards, dim=0)
        
        finally:
            # Clean up temporary video files
            for temp_path in temp_video_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        return rewards


if __name__ == "__main__":
    import numpy as np
    from decord import VideoReader

    video_path_list = ["your_video_path_1.mp4", "your_video_path_2.mp4"]
    prompt_list = ["your_prompt_1", "your_prompt_2"]
    num_sampled_frames = 8

    to_tensor = transforms.ToTensor()

    sampled_frames_list = []
    for video_path in video_path_list:
        vr = VideoReader(video_path)
        sampled_frame_indices = np.linspace(0, len(vr), num_sampled_frames, endpoint=False, dtype=int)
        sampled_frames = vr.get_batch(sampled_frame_indices).asnumpy()
        sampled_frames = torch.stack([to_tensor(frame) for frame in sampled_frames])
        sampled_frames_list.append(sampled_frames)
    sampled_frames = torch.stack(sampled_frames_list)
    sampled_frames = rearrange(sampled_frames, "b t c h w -> b c t h w")

    aesthetic_reward_v2 = AestheticReward(device="cuda", dtype=torch.bfloat16)
    print(f"aesthetic_reward_v2: {aesthetic_reward_v2(sampled_frames)}")

    aesthetic_reward_v2_5 = AestheticReward(
        encoder_path="google/siglip-so400m-patch14-384", version="v2.5", device="cuda", dtype=torch.bfloat16
    )
    print(f"aesthetic_reward_v2_5: {aesthetic_reward_v2_5(sampled_frames)}")

    hps_reward_v2 = HPSReward(device="cuda", dtype=torch.bfloat16)
    print(f"hps_reward_v2: {hps_reward_v2(sampled_frames, prompt_list)}")

    hps_reward_v2_1 = HPSReward(version="v2.1", device="cuda", dtype=torch.bfloat16)
    print(f"hps_reward_v2_1: {hps_reward_v2_1(sampled_frames, prompt_list)}")

    pick_score = PickScoreReward(device="cuda", dtype=torch.bfloat16)
    print(f"pick_score_reward: {pick_score(sampled_frames, prompt_list)}")

    mps_score = MPSReward(device="cuda", dtype=torch.bfloat16)
    print(f"mps_reward: {mps_score(sampled_frames, prompt_list)}")