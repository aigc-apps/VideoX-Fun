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
    """An base class for reward models. A custom Reward Class must implement two functions below.
    """
    # Whether this reward operates on individual frames (image-level) rather than full videos.
    # Image-based rewards (e.g. HPS, MPS, Aesthetic) need frame sampling before scoring.
    is_image_reward = False

    def __init__(self):
        """Define your reward model and image transformations (optional) here.
        """
        pass

    def to(self, device):
        """Move the reward model to the specified device.

        Supports two common patterns:
        - self.model (AestheticReward, HPSReward, PickScoreReward, MPSReward)
        - self.inferencer.model (HPSv3Reward, VideoAlignReward)

        Subclasses with non-standard model storage should override this method.
        """
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
            self.model.to(device)
        elif hasattr(self, 'inferencer') and hasattr(self.inferencer, 'model'):
            self.inferencer.model.to(device)
            if hasattr(self.inferencer, 'device'):
                self.inferencer.device = device
        self.device = device
        return self

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
    is_image_reward = True

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
            self.model, _ = convert_v2_5_from_siglip(
                predictor_name_or_path=self.predictor_path,
                encoder_model_name=self.encoder_path,
            )
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
    is_image_reward = True

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
    is_image_reward = True

    def __init__(
        self,
        model_path="yuvalkirstain/PickScore_v1",
        processor_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
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
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path, torch_dtype=self.dtype)
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
    is_image_reward = True

    def __init__(
        self,
        model_path=None,
        processor_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
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

        processor_name_or_path = processor_name_or_path
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
    is_image_reward = True

    def __init__(
        self,
        checkpoint_path=None,
        model_name_or_path=None,
        device="cpu",
        dtype=torch.float16,
        max_reward=1,
        loss_scale=1,
        differentiable=False,
    ):
        from .hpsv3_predictor import HPSv3RewardInferencer

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype = dtype
        self.max_reward = max_reward
        self.loss_scale = loss_scale
        self.differentiable = differentiable

        self.inferencer = HPSv3RewardInferencer(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            dtype=self.dtype,
            model_name_or_path=model_name_or_path,
        )

        # Freeze reward model parameters when using differentiable mode.
        # The forward pass still builds the computation graph for input gradients.
        if self.differentiable:
            self.inferencer.model.requires_grad_(False)

    def __call__(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.differentiable:
            rewards = self.get_reward_differentiable(batch_frames, batch_prompt)
        else:
            rewards = self.get_reward(batch_frames, batch_prompt)
        if self.max_reward is None:
            loss_per_sample = (-1 * rewards) * self.loss_scale
        else:
            loss_per_sample = torch.abs(rewards - self.max_reward) * self.loss_scale
        return loss_per_sample.mean(), rewards.mean()

    def get_reward_differentiable(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> torch.Tensor:
        """Differentiable reward computation that preserves grad_fn.

        Gradients flow from the returned scalar rewards back through the reward
        model to the input batch_frames.

        Args:
            batch_frames: [B, C, T, H, W] tensor in [0, 1].
            batch_prompt: List of B text prompts.
        Returns:
            torch.Tensor: [B] scalar rewards (mu) with grad_fn.
        """
        assert len(batch_frames) == len(batch_prompt)
        batch_frames = rearrange(batch_frames, "b c t h w -> t b c h w")
        total_rewards = []

        for frames in batch_frames:
            # frames: [B, C, H, W] in [0, 1]
            image_tensors = [frame for frame in frames]
            logits = self.inferencer.reward_differentiable(image_tensors, batch_prompt)
            # logits: [B, output_dim], extract mu (index 0)
            reward = logits[:, 0]
            total_rewards.append(reward)

        rewards = torch.stack(total_rewards, dim=0).mean(dim=0)
        return rewards

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
    is_image_reward = False
    
    def __init__(
        self,
        model_path=None,
        model_name_or_path=None,
        device="cpu",
        dtype=torch.float16,
        max_reward=1,
        loss_scale=1,
        reward_dim="Overall",
        fps=16,
        num_frames=None,
        use_norm=True,
        return_all_dims=False,
        use_legacy_video_io=True,
        differentiable=False,
    ):
        from .video_align_predictor import VideoVLMRewardInference

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_reward = max_reward
        self.loss_scale = loss_scale
        self.reward_dim = reward_dim  # Which dimension(s) to extract as the scalar reward.
        #   - "VQ"     : Visual Quality (clearness, resolution, brightness, color)
        #   - "MQ"     : Motion Quality (consistency, smoothness, completeness)
        #   - "TA"     : Text-to-Video Alignment (prompt-content & motion match)
        #   - "Overall": Overall Performance = VQ + MQ + TA (sum of the three)
        #   - Combinations like "VQ+MQ", "VQ+TA", "MQ+TA" are also supported,
        #     which sum the specified dimensions.
        self.fps = fps
        self.num_frames = num_frames
        self.use_norm = use_norm
        self.return_all_dims = return_all_dims  # Return all dimensions instead of single reward_dim
        self.use_legacy_video_io = use_legacy_video_io  # If True, save to temp video then read back (old path)
        self.differentiable = differentiable  # If True, use differentiable path for backprop

        self.inferencer = VideoVLMRewardInference(
            load_from_pretrained=self.model_path,
            device=self.device,
            dtype=self.dtype,
            model_name_or_path=model_name_or_path,
        )

        # Freeze reward model parameters when using differentiable mode.
        # The forward pass still builds the computation graph for input gradients.
        if self.differentiable:
            self.inferencer.model.requires_grad_(False)

    def _save_frames_to_temp_video(self, frames: torch.Tensor, fps: float = 8.0) -> str:
        """Save tensor frames to a temporary video file with lossless encoding.
        
        Args:
            frames: Tensor of shape [T, C, H, W] with values in [0, 1]
            fps: Frames per second for the output video
            
        Returns:
            Path to the temporary video file
        """
        import os
        import tempfile

        import av

        # Use /dev/shm (tmpfs, RAM-based) to avoid disk IO, fallback to tempdir
        shm_dir = "/dev/shm"
        if os.path.exists(shm_dir) and os.access(shm_dir, os.W_OK):
            temp_dir = shm_dir
        else:
            temp_dir = tempfile.gettempdir()
        
        # Generate unique filename based on frame content hash
        import hashlib

        # Use multiple frames' bytes for robust hashing
        frame_data = frames.float().cpu().numpy().tobytes()
        frame_hash = hashlib.md5(frame_data[:10000] + frame_data[-10000:]).hexdigest()[:16]
        temp_video_path = os.path.join(temp_dir, f"videovlm_reward_temp_{os.getpid()}_{frame_hash}.mp4")
        
        # Convert frames to numpy: [T, C, H, W] -> [T, H, W, C]
        frames_np = (frames.float().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype('uint8')
        
        # Write video using PyAV with LOSSLESS encoding to avoid fluctuation
        t, h, w, c = frames_np.shape
        container = av.open(temp_video_path, mode='w')
        # Use libx264 with CRF=0 for truly lossless encoding
        stream = container.add_stream('libx264', rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = 'yuv444p'
        stream.options = {'crf': '0', 'preset': 'ultrafast'}  # CRF=0 is lossless
        
        for frame_data in frames_np:
            frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        
        return temp_video_path

    def __call__(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.differentiable:
            if self.return_all_dims:
                rewards_dict = self.get_reward_all_dims_differentiable(batch_frames, batch_prompt)
                rewards = rewards_dict['Overall']
            else:
                rewards = self.get_reward_differentiable(batch_frames, batch_prompt)
        else:
            if self.return_all_dims:
                rewards_dict = self.get_reward_all_dims(batch_frames, batch_prompt)
                rewards = rewards_dict['Overall']
            else:
                rewards = self.get_reward(batch_frames, batch_prompt)
        
        if self.max_reward is None:
            loss_per_sample = (-1 * rewards) * self.loss_scale
        else:
            loss_per_sample = torch.abs(rewards - self.max_reward) * self.loss_scale
        return loss_per_sample.mean(), rewards.mean()

    def _get_rewards_legacy(self, batch_frames, batch_prompt):
        """Legacy path: save tensors to temp video files, then read back via inferencer.reward()."""
        temp_video_paths = []
        try:
            for frames in batch_frames:
                frames = rearrange(frames, "c t h w -> t c h w")
                temp_video_path = self._save_frames_to_temp_video(frames, fps=self.fps)
                temp_video_paths.append(temp_video_path)

            rewards_output = self.inferencer.reward(
                video_paths=temp_video_paths,
                prompts=batch_prompt,
                num_frames=self.num_frames,
                use_norm=self.use_norm,
            )
        finally:
            for temp_path in temp_video_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        return rewards_output

    def _get_rewards_direct(self, batch_frames, batch_prompt):
        """Direct tensor path: pass tensors to inferencer without file I/O."""
        video_tensors = [rearrange(frames, "c t h w -> t c h w") for frames in batch_frames]
        rewards_output = self.inferencer.reward_from_tensors(
            video_tensors=video_tensors,
            prompts=batch_prompt,
            num_frames=self.num_frames,
            video_fps=self.fps,
            use_norm=self.use_norm,
        )
        return rewards_output

    def _norm_logits(self, logits):
        """Apply per-dimension normalization to raw logits tensor (differentiable).

        Args:
            logits: torch.Tensor of shape [B, 3] with columns [VQ, MQ, TA].
        Returns:
            Normalized logits tensor of the same shape, with grad_fn preserved.
        """
        if self.inferencer.inference_config is None:
            return logits
        # Cast to float32 for precision parity with the non-differentiable path,
        # which normalizes in float64 after .item(). .float() preserves grad_fn.
        logits = logits.float()
        cfg = self.inferencer.inference_config
        mean = torch.tensor(
            [cfg['VQ_mean'], cfg['MQ_mean'], cfg['TA_mean']],
            device=logits.device, dtype=logits.dtype,
        )
        std = torch.tensor(
            [cfg['VQ_std'], cfg['MQ_std'], cfg['TA_std']],
            device=logits.device, dtype=logits.dtype,
        )
        return (logits - mean) / std

    def get_reward_differentiable(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> torch.Tensor:
        """Differentiable reward computation that preserves grad_fn.

        No torch.no_grad() context, no .item() calls, no torch.tensor() wrapping.
        Gradients flow from the returned scalar rewards back through the reward
        model to the input batch_frames.

        Args:
            batch_frames: [B, C, T, H, W] tensor in [0, 1].
            batch_prompt: List of B text prompts.
        Returns:
            torch.Tensor: [B] scalar rewards with grad_fn.
        """
        assert len(batch_frames) == len(batch_prompt)
        video_tensors = [rearrange(frames, "c t h w -> t c h w") for frames in batch_frames]
        logits = self.inferencer.reward_from_tensors_differentiable(
            video_tensors=video_tensors,
            prompts=batch_prompt,
            num_frames=self.num_frames,
            video_fps=self.fps,
        )  # [B, 3] with columns [VQ, MQ, TA]

        if self.use_norm:
            logits = self._norm_logits(logits)

        # Select the reward dimension(s).
        # Supports single dim ("VQ"), "Overall", or combinations like "VQ+MQ".
        dim_map = {"VQ": 0, "MQ": 1, "TA": 2}
        if self.reward_dim == "Overall":
            rewards = logits.sum(dim=-1)  # [B]
        elif "+" in self.reward_dim:
            dims = [d.strip() for d in self.reward_dim.split("+")]
            indices = [dim_map[d] for d in dims if d in dim_map]
            if len(indices) == 0:
                raise ValueError(f"Unknown reward_dim combination: {self.reward_dim}")
            rewards = logits[:, indices].sum(dim=-1)  # [B]
        elif self.reward_dim in dim_map:
            rewards = logits[:, dim_map[self.reward_dim]]  # [B]
        else:
            raise ValueError(f"Unknown reward_dim: {self.reward_dim}")

        return rewards

    def get_reward_all_dims_differentiable(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> dict:
        """Differentiable version of get_reward_all_dims.

        Returns:
            dict: 'VQ', 'MQ', 'TA', 'Overall' keys, each a [B] tensor with grad_fn.
        """
        assert len(batch_frames) == len(batch_prompt)
        video_tensors = [rearrange(frames, "c t h w -> t c h w") for frames in batch_frames]
        logits = self.inferencer.reward_from_tensors_differentiable(
            video_tensors=video_tensors,
            prompts=batch_prompt,
            num_frames=self.num_frames,
            video_fps=self.fps,
        )  # [B, 3]

        if self.use_norm:
            logits = self._norm_logits(logits)

        return {
            'VQ': logits[:, 0],
            'MQ': logits[:, 1],
            'TA': logits[:, 2],
            'Overall': logits.sum(dim=-1),
        }

    @torch.no_grad()
    def get_reward(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> torch.Tensor:
        assert len(batch_frames) == len(batch_prompt)        
        total_rewards = []

        if self.use_legacy_video_io:
            rewards_output = self._get_rewards_legacy(batch_frames, batch_prompt)
        else:
            rewards_output = self._get_rewards_direct(batch_frames, batch_prompt)

        # Support single dim, "Overall", or combinations like "VQ+MQ"
        if "+" in self.reward_dim:
            dims = [d.strip() for d in self.reward_dim.split("+")]
        else:
            dims = [self.reward_dim]

        for reward_dict in rewards_output:
            if "+" in self.reward_dim:
                reward_value = sum(reward_dict[d] for d in dims)
            else:
                reward_value = reward_dict[self.reward_dim]
            total_rewards.append(torch.tensor(reward_value, device=self.device, dtype=self.dtype))

        rewards = torch.stack(total_rewards, dim=0)
        return rewards

    @torch.no_grad()
    def get_reward_all_dims(self, batch_frames: torch.Tensor, batch_prompt: list[str]) -> dict:
        """Get rewards for all dimensions (VQ, MQ, TA, Overall).
        
        Returns:
            dict: Dictionary with keys 'VQ', 'MQ', 'TA', 'Overall', each containing a tensor of rewards.
        """
        assert len(batch_frames) == len(batch_prompt)        
        all_rewards = {'VQ': [], 'MQ': [], 'TA': [], 'Overall': []}

        if self.use_legacy_video_io:
            rewards_output = self._get_rewards_legacy(batch_frames, batch_prompt)
        else:
            rewards_output = self._get_rewards_direct(batch_frames, batch_prompt)

        for reward_dict in rewards_output:
            for dim in ['VQ', 'MQ', 'TA', 'Overall']:
                reward_value = reward_dict[dim]
                all_rewards[dim].append(torch.tensor(reward_value, device=self.device, dtype=self.dtype))

        # Stack all dimensions
        result = {}
        for dim in ['VQ', 'MQ', 'TA', 'Overall']:
            result[dim] = torch.stack(all_rewards[dim], dim=0)

        return result

if __name__ == "__main__":
    import numpy as np
    try:
        from decord import VideoReader
    except ImportError:
        from videox_fun.data.utils import AVVideoReader as VideoReader

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