# Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx2/pipeline_ltx2_latent_upsample.py
# Copyright 2025 The VideoX-Fun Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging
from diffusers.video_processor import VideoProcessor

from ..models import AutoencoderKLWan
from ..models.wan_latent_upsampler import WanLatentUpsamplerModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class WanLatentUpsamplePipelineOutput(BaseOutput):
    frames: torch.Tensor


class WanLatentUpsamplePipeline(DiffusionPipeline):
    """Pipeline to spatially upsample Wan VAE latents using a trained latent upsampler model."""

    model_cpu_offload_seq = "vae->latent_upsampler"

    def __init__(
        self,
        vae: AutoencoderKLWan,
        latent_upsampler: WanLatentUpsamplerModel,
    ) -> None:
        super().__init__()

        self.register_modules(vae=vae, latent_upsampler=latent_upsampler)

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 4
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)

    def prepare_latents(
        self,
        video=None,
        batch_size=1,
        num_frames=81,
        height=480,
        width=720,
        dtype=None,
        device=None,
        generator=None,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        # Encode video pixels to latents via VAE
        video = video.to(device=device, dtype=self.vae.dtype)
        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            init_latents = [
                self.vae.encode(video[i].unsqueeze(0)).sample for i in range(batch_size)
            ]
        else:
            init_latents = [self.vae.encode(vid.unsqueeze(0)).sample for vid in video]

        init_latents = torch.cat(init_latents, dim=0).to(dtype)
        return init_latents

    def adain_filter_latent(self, latents, reference_latents, factor=1.0):
        """
        Applies Adaptive Instance Normalization (AdaIN) to a latent tensor based on statistics from a reference latent
        tensor.

        Args:
            latents (`torch.Tensor`):
                Input latents to normalize
            reference_latents (`torch.Tensor`):
                The reference latents providing style statistics.
            factor (`float`):
                Blending factor between original and transformed latent. Range: -10.0 to 10.0, Default: 1.0

        Returns:
            torch.Tensor: The transformed latent tensor
        """
        result = latents.clone()

        for i in range(latents.size(0)):
            for c in range(latents.size(1)):
                r_sd, r_mean = torch.std_mean(reference_latents[i, c], dim=None)
                i_sd, i_mean = torch.std_mean(result[i, c], dim=None)

                result[i, c] = ((result[i, c] - i_mean) / i_sd) * r_sd + r_mean

        result = torch.lerp(latents, result, factor)
        return result

    def tone_map_latents(self, latents, compression):
        """
        Applies a non-linear tone-mapping function to latent values to reduce their dynamic range.

        Args:
            latents : torch.Tensor
                Input latent tensor.
            compression : float
                Compression strength in the range [0, 1].

        Returns:
            torch.Tensor: The tone-mapped latent tensor.
        """
        scale_factor = compression * 0.75
        abs_latents = torch.abs(latents)

        sigmoid_term = torch.sigmoid(4.0 * scale_factor * (abs_latents - 1.0))
        scales = 1.0 - 0.8 * scale_factor * sigmoid_term

        filtered = latents * scales
        return filtered

    def check_inputs(self, video, latents, tone_map_compression_ratio):
        if video is not None and latents is not None:
            raise ValueError("Only one of `video` or `latents` can be provided.")
        if video is None and latents is None:
            raise ValueError("One of `video` or `latents` has to be provided.")

        if not (0 <= tone_map_compression_ratio <= 1):
            raise ValueError("`tone_map_compression_ratio` must be in the range [0, 1]")

    @torch.no_grad()
    def __call__(
        self,
        video=None,
        height=480,
        width=720,
        num_frames=81,
        latents=None,
        adain_factor=0.0,
        tone_map_compression_ratio=0.0,
        generator=None,
        output_type="pt",
        return_dict=True,
    ):
        """
        Function invoked when calling the pipeline for latent upsampling.

        Args:
            video (`torch.Tensor`, *optional*):
                The video tensor of shape `[B, C, F, H, W]` in pixel space ([-1, 1]).
                If not supplied, `latents` should be supplied.
            height (`int`, *optional*, defaults to `480`):
                The height in pixels of the input video.
            width (`int`, *optional*, defaults to `720`):
                The width in pixels of the input video.
            num_frames (`int`, *optional*, defaults to `81`):
                The number of frames in the input video.
            latents (`torch.Tensor`, *optional*):
                Pre-generated video latents of shape `[B, C, F, H, W]`.
            adain_factor (`float`, *optional*, defaults to `0.0`):
                Adaptive Instance Normalization blending factor.
            tone_map_compression_ratio (`float`, *optional*, defaults to `0.0`):
                Compression strength for tone mapping in [0, 1].
            generator (`torch.Generator`, *optional*):
                A torch generator for reproducibility.
            output_type (`str`, *optional*, defaults to `"pt"`):
                Output format. "pt" for pixel tensor [0,1], "latent" for raw latents.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a dataclass or a tuple.

        Returns:
            `WanLatentUpsamplePipelineOutput` or `tuple`.
        """

        self.check_inputs(
            video=video,
            latents=latents,
            tone_map_compression_ratio=tone_map_compression_ratio,
        )

        if video is not None:
            batch_size = video.shape[0]
        else:
            batch_size = latents.shape[0]
        device = self._execution_device

        latents = self.prepare_latents(
            video=video,
            batch_size=batch_size,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        latents = latents.to(self.latent_upsampler.dtype)
        latents_upsampled = self.latent_upsampler(latents)

        if adain_factor > 0.0:
            latents = self.adain_filter_latent(latents_upsampled, latents, adain_factor)
        else:
            latents = latents_upsampled

        if tone_map_compression_ratio > 0.0:
            latents = self.tone_map_latents(latents, tone_map_compression_ratio)

        if output_type == "latent":
            video = latents
        else:
            video = self.vae.decode(latents.to(self.vae.dtype)).sample
            video = (video / 2 + 0.5).clamp(0, 1)
            video = video.cpu().float()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanLatentUpsamplePipelineOutput(frames=video)
