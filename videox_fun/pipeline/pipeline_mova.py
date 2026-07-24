# Modified from hhttps://github.com/OpenMOSS/MOVA/blob/main/mova/diffusion/pipelines/pipeline_mova.py
import copy
import html
import re
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import ftfy
import numpy as np
import torch
from diffusers.configuration_utils import register_to_config
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from tqdm import tqdm

from ..models import (AutoencoderKLMOVAAudio, AutoencoderKLWan, AutoTokenizer,
                      MOVAModel, UMT5EncoderModel)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


@dataclass
class MOVAPipelineOutput(BaseOutput):
    r"""
    Output class for LTX pipelines.

    Args:
        videos (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
        audio (`torch.Tensor`, `np.ndarray`):
            TODO
    """

    videos: torch.Tensor
    audio: torch.Tensor


class MOVAPipeline(DiffusionPipeline):
    r"""
    Pipeline for image-to-video and audio generation.

    Reference: MOVA (Multi-modal Open Video-Audio generation)

    Args:
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode video frames to and from latent representations.
        audio_vae ([`AutoencoderKLMOVAAudio`]):
            Variational Auto-Encoder (VAE) Model to encode and decode audio to and from latent representations.
        text_encoder ([`UMT5EncoderModel`]):
            Text encoder to encode prompts into text embeddings.
        tokenizer (`AutoTokenizer`):
            Tokenizer for text encoding.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `mova_model` to denoise the encoded latents.
        transformer:
            Video DiT (low-noise) model.
        transformer_2:
            Video DiT 2 (high-noise) model.
        transformer_audio:
            Audio DiT model.
        dual_tower_bridge:
            Dual tower bridge for cross-modal interaction.
    """

    model_cpu_offload_seq = "text_encoder->dual_tower_bridge->transformer_2->transformer_audio->transformer->vae->audio_vae"
    _optional_components = ["transformer_audio", "dual_tower_bridge"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    vae: AutoencoderKLWan
    audio_vae: AutoencoderKLMOVAAudio
    text_encoder: UMT5EncoderModel
    tokenizer: AutoTokenizer
    scheduler: Any
    transformer: Any
    transformer_2: Any
    transformer_audio: Any
    dual_tower_bridge: Any

    @register_to_config
    def __init__(
        self,
        vae: AutoencoderKLWan,
        audio_vae: AutoencoderKLMOVAAudio,
        text_encoder: UMT5EncoderModel,
        tokenizer: AutoTokenizer,
        scheduler: Any,
        transformer: Any,
        transformer_2: Any,
        transformer_audio: Any,
        dual_tower_bridge: Any,
        audio_vae_type: str = "dac", # type: Literal["oobleck", "dac"]
    ):
        super().__init__()

        # Build MOVAModel helper internally
        mova_model = MOVAModel(
            transformer=transformer,
            transformer_2=transformer_2,
            transformer_audio=transformer_audio,
            dual_tower_bridge=dual_tower_bridge,
        )

        self.register_modules(
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
            transformer_2=transformer_2,
            transformer_audio=transformer_audio,
            dual_tower_bridge=dual_tower_bridge,
        )

        self.register_to_config(
            audio_vae_type=audio_vae_type,
        )

        self.audio_vae_type = audio_vae_type
        
        # Store mova_model as internal helper (not registered as module)
        self.mova_model = mova_model

        # build video vae
        self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio
        self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        
        # build audio vae
        self.audio_vae_scale_factor = int(self.audio_vae.hop_length)
        self.audio_sample_rate = self.audio_vae.sample_rate


    def normalize_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # Normalize latents with config stats (diffusers Wan convention)
        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        inv_std = (1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype)).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        latents = (latents - mean) * inv_std
        return latents
    

    def denormalize_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # Denormalize latents with config stats (diffusers Wan convention)
        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        latents = latents * std + mean
        return latents


    def check_inputs(
        self,
        height,
        width,
        num_frames,
    ):
        target_division_factor = self.vae_scale_factor_spatial * 2
        if height % target_division_factor != 0 or width % target_division_factor != 0:
            raise ValueError(f"`height` and `width` have to be divisible by {target_division_factor} but are {height} and {width}.")
        
        if num_frames % self.vae_scale_factor_temporal != 1:
            raise ValueError(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal} but is {num_frames - 1}."
            )
    
    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]

        if last_image is None:
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        # latent_condition = self.normalize_video_latents(latent_condition)

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)
    
    def prepare_audio_latents(
        self,
        audio: Optional[torch.Tensor],
        batch_size: int,
        num_channels: int,
        num_samples: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        latent_t = (num_samples - 1) // self.audio_vae_scale_factor + 1
        shape = (batch_size, num_channels, latent_t)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents
    
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput = None,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] | None = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        height: int = 360,
        width: int = 640,
        num_frames: int = 193,
        frame_rate: float = 24.0,
        num_inference_steps: int = 40,
        guidance_scale: float = 5.0,
        output_type: str = "pil",
        return_dict: bool = True,
        boundary: float = 0.9,
    ):
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            height=height,
            width=width,
            num_frames=num_frames,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False
        self._current_timestep = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        # 3. Prepare text embeddings
        prompt_embeds = self._get_t5_prompt_embeds(prompt)
        if self.do_classifier_free_guidance:
            negative_prompt_embeds = self._get_t5_prompt_embeds(negative_prompt)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        scheduler_support_audio = hasattr(self.scheduler, "get_pairs")
        if scheduler_support_audio:
            audio_scheduler = self.scheduler
            paired_timesteps = self.scheduler.get_pairs()
        else:
            audio_scheduler = copy.deepcopy(self.scheduler)
            paired_timesteps = torch.stack([self.scheduler.timesteps, self.scheduler.timesteps], dim=1)

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        latents, condition = self.prepare_latents(
            image,
            batch_size,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator=generator,
            latents=None,
            last_image=None,
        )

        audio_num_samples = int(self.audio_sample_rate * num_frames / frame_rate)
        audio_latents = self.prepare_audio_latents(
            None,
            batch_size,
            self.audio_vae.latent_dim,
            audio_num_samples,
            torch.float32,
            device,
            generator=generator,
            latents=None,
        )

        # --------------------------------------------------
        # diffusion steps
        # --------------------------------------------------
        total_steps = paired_timesteps.shape[0]
        boundary_timestep = boundary * self.scheduler.config.num_train_timesteps
        
        for idx_step in tqdm(range(total_steps)):
            timestep, audio_timestep = paired_timesteps[idx_step]

            # Switch DiT based on timestep (Wan2.2 convention: transformer_2 = high-noise, transformer = low-noise)
            # Large t (>= boundary) -> high noise -> transformer_2 -> use_low_noise_dit=False
            # Small t (< boundary) -> low noise -> transformer -> use_low_noise_dit=True
            use_low_noise_dit = timestep.item() < boundary_timestep
            
            latent_model_input = torch.cat([latents, condition], dim=1)
            # timestep
            timestep = timestep.unsqueeze(0).to(device=device, dtype=torch.float32)
            audio_timestep = audio_timestep.unsqueeze(0).to(device=device, dtype=torch.float32)

            # Forward through MOVA model
            with nullcontext():
                visual_noise_pred_posi, audio_noise_pred_posi = self.mova_model(
                    visual_latents=latent_model_input,
                    audio_latents=audio_latents,
                    context=prompt_embeds,
                    timestep=timestep,
                    audio_timestep=audio_timestep,
                    frame_rate=frame_rate,
                    use_low_noise_dit=use_low_noise_dit,
                )
            
            if guidance_scale == 1.0:
                visual_noise_pred = visual_noise_pred_posi.float()
                audio_noise_pred = audio_noise_pred_posi.float()
            else:
                visual_noise_pred_nega, audio_noise_pred_nega = self.mova_model(
                    visual_latents=latent_model_input,
                    audio_latents=audio_latents,
                    context=negative_prompt_embeds,
                    timestep=timestep,
                    audio_timestep=audio_timestep,
                    frame_rate=frame_rate,
                    use_low_noise_dit=use_low_noise_dit,
                )
                visual_noise_pred_nega = visual_noise_pred_nega.float()
                audio_noise_pred_nega = audio_noise_pred_nega.float()

                visual_noise_pred = visual_noise_pred_nega + guidance_scale * (visual_noise_pred_posi - visual_noise_pred_nega)
                audio_noise_pred = audio_noise_pred_nega + guidance_scale * (audio_noise_pred_posi - audio_noise_pred_nega)

            if scheduler_support_audio:
                next_timestep = paired_timesteps[idx_step + 1, 0] if idx_step + 1 < total_steps else None
                next_audio_timestep = paired_timesteps[idx_step + 1, 1] if idx_step + 1 < total_steps else None
                latents = self.scheduler.step_from_to(
                    visual_noise_pred,
                    timestep,
                    next_timestep,
                    latents,
                )
                audio_latents = audio_scheduler.step_from_to(
                    audio_noise_pred,
                    audio_timestep,
                    next_audio_timestep,
                    audio_latents,
                )
            else:
                latents = self.scheduler.step(visual_noise_pred, timestep, latents, return_dict=False)[0]
                audio_latents = audio_scheduler.step(audio_noise_pred, audio_timestep, audio_latents, return_dict=False)[0]
        
        if output_type == "latent":
            video = latents
            audio = audio_latents
        else:
            # decode video
            # video_latents = self.denormalize_video_latents(latents)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video = self.vae.decode(latents).sample
            video = self.video_processor.postprocess_video(video, output_type="pt").float().cpu().permute(0, 2, 1, 3, 4)

            # decode audio
            with torch.autocast("cuda", dtype=torch.float32):
                audio = self.audio_vae.decode(audio_latents).float().cpu()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, audio)

        return MOVAPipelineOutput(videos=video, audio=audio)
