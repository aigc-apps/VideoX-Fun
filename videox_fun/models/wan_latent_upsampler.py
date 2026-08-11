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

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version

RATIONAL_RESAMPLER_SCALE_MAPPING = {
    0.75: (3, 4),
    1.5: (3, 2),
    2.0: (2, 1),
    4.0: (4, 1),
}


class ResBlock(torch.nn.Module):
    def __init__(self, channels: int, mid_channels: Optional[int] = None, dims: int = 3):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        Conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d

        self.conv1 = Conv(channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = torch.nn.GroupNorm(32, mid_channels)
        self.conv2 = Conv(mid_channels, channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(32, channels)
        self.activation = torch.nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.activation(hidden_states + residual)
        return hidden_states


class PixelShuffleND(torch.nn.Module):
    def __init__(self, dims, upscale_factors=(2, 2, 2)):
        super().__init__()

        self.dims = dims
        self.upscale_factors = upscale_factors

        if dims not in [1, 2, 3]:
            raise ValueError("dims must be 1, 2, or 3")

    def forward(self, x):
        if self.dims == 3:
            # spatiotemporal: b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)
            return (
                x.unflatten(1, (-1, *self.upscale_factors[:3]))
                .permute(0, 1, 5, 2, 6, 3, 7, 4)
                .flatten(6, 7)
                .flatten(4, 5)
                .flatten(2, 3)
            )
        elif self.dims == 2:
            # spatial: b (c p1 p2) h w -> b c (h p1) (w p2)
            return (
                x.unflatten(1, (-1, *self.upscale_factors[:2])).permute(0, 1, 4, 2, 5, 3).flatten(4, 5).flatten(2, 3)
            )
        elif self.dims == 1:
            # temporal: b (c p1) f h w -> b c (f p1) h w
            return x.unflatten(1, (-1, *self.upscale_factors[:1])).permute(0, 1, 3, 2, 4, 5).flatten(2, 3)


class BlurDownsample(torch.nn.Module):
    """
    Anti-aliased spatial downsampling by integer stride using a fixed separable binomial kernel.
    Works for dims=2 or dims=3 (per-frame).
    """

    def __init__(self, dims: int, stride: int, kernel_size: int = 5) -> None:
        super().__init__()

        if dims not in (2, 3):
            raise ValueError(f"`dims` must be either 2 or 3 but is {dims}")
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError(f"`kernel_size` must be an odd number >= 3 but is {kernel_size}")

        self.dims = dims
        self.stride = stride
        self.kernel_size = kernel_size

        k = torch.tensor([math.comb(kernel_size - 1, k) for k in range(kernel_size)])
        k2d = k[:, None] @ k[None, :]
        k2d = (k2d / k2d.sum()).float()
        self.register_buffer("kernel", k2d[None, None, :, :])

    def _conv2d_chunked(self, x: torch.Tensor, weight: torch.Tensor, c: int) -> torch.Tensor:
        """Run grouped conv2d in chunks to avoid 32-bit indexing overflow on large tensors."""
        max_elements = 2**31 - 1
        elements_per_sample = x.shape[1] * x.shape[2] * x.shape[3]
        chunk_size = max(1, max_elements // elements_per_sample)

        if chunk_size >= x.shape[0]:
            return F.conv2d(x, weight=weight, bias=None, stride=self.stride, padding=self.kernel_size // 2, groups=c)

        chunks = []
        for i in range(0, x.shape[0], chunk_size):
            chunks.append(
                F.conv2d(x[i:i + chunk_size], weight=weight, bias=None, stride=self.stride, padding=self.kernel_size // 2, groups=c)
            )
        return torch.cat(chunks, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x

        if self.dims == 2:
            c = x.shape[1]
            weight = self.kernel.expand(c, 1, self.kernel_size, self.kernel_size)
            x = self._conv2d_chunked(x, weight, c)
        else:
            b, c, f, _, _ = x.shape
            x = x.transpose(1, 2).flatten(0, 1)  # [B, C, F, H, W] --> [B * F, C, H, W]

            weight = self.kernel.expand(c, 1, self.kernel_size, self.kernel_size)
            x = self._conv2d_chunked(x, weight, c)

            h2, w2 = x.shape[-2:]
            x = x.unflatten(0, (b, f)).reshape(b, -1, f, h2, w2)  # [B * F, C, H, W] --> [B, C, F, H, W]
        return x


class SpatialRationalResampler(torch.nn.Module):
    """
    Scales the spatial size of the input by a rational number `scale`.
    Uses Conv2d + PixelShuffle upsampling + BlurDownsample anti-aliasing.
    """

    def __init__(self, mid_channels: int = 256, scale: float = 2.0):
        super().__init__()
        self.scale = float(scale)
        num_denom = RATIONAL_RESAMPLER_SCALE_MAPPING.get(scale, None)
        if num_denom is None:
            raise ValueError(
                f"The supplied `scale` {scale} is not supported; supported scales are {list(RATIONAL_RESAMPLER_SCALE_MAPPING.keys())}"
            )
        self.num, self.den = num_denom

        self.conv = torch.nn.Conv2d(mid_channels, (self.num**2) * mid_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffleND(2, upscale_factors=(self.num, self.num))
        self.blur_down = BlurDownsample(dims=2, stride=self.den)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected x shape: [B * F, C, H, W]
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.blur_down(x)
        return x


class WanLatentUpsamplerModel(ModelMixin, ConfigMixin):
    """
    Model to spatially upsample Wan VAE latents.

    Args:
        in_channels (`int`, defaults to `16`):
            Number of channels in the input latent (Wan2.1=16, Wan2.2=48)
        mid_channels (`int`, defaults to `256`):
            Number of channels in the middle layers
        num_blocks_per_stage (`int`, defaults to `4`):
            Number of ResBlocks to use in each stage (pre/post upsampling)
        dims (`int`, defaults to `3`):
            Number of dimensions for convolutions (2 or 3)
        spatial_upsample (`bool`, defaults to `True`):
            Whether to spatially upsample the latent
        temporal_upsample (`bool`, defaults to `False`):
            Whether to temporally upsample the latent
        rational_spatial_scale (`float`, defaults to `2.0`):
            Spatial upsampling scale factor
        use_rational_resampler (`bool`, defaults to `True`):
            Whether to use rational resampler (PixelShuffle + BlurDownsample)
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        rational_spatial_scale: float = 2.0,
        use_rational_resampler: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.dims = dims
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample

        ConvNd = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d

        self.initial_conv = ConvNd(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = torch.nn.GroupNorm(32, mid_channels)
        self.initial_activation = torch.nn.SiLU()

        self.res_blocks = torch.nn.ModuleList([ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)])

        if spatial_upsample and temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(3),
            )
        elif spatial_upsample:
            if use_rational_resampler:
                self.upsampler = SpatialRationalResampler(mid_channels=mid_channels, scale=rational_spatial_scale)
            else:
                self.upsampler = torch.nn.Sequential(
                    torch.nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
                    PixelShuffleND(2),
                )
        elif temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(1),
            )
        else:
            raise ValueError("Either spatial_upsample or temporal_upsample must be True")

        self.post_upsample_res_blocks = torch.nn.ModuleList(
            [ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )

        self.final_conv = ConvNd(mid_channels, in_channels, kernel_size=3, padding=1)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            self.gradient_checkpointing = kwargs["value"]
        elif "enable" in kwargs:
            self.gradient_checkpointing = kwargs["enable"]
        else:
            self.gradient_checkpointing = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # Prepare checkpointing utilities
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

        if self.dims == 2:
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
            hidden_states = self.initial_conv(hidden_states)
            hidden_states = self.initial_norm(hidden_states)
            hidden_states = self.initial_activation(hidden_states)

            for block in self.res_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block), hidden_states, **ckpt_kwargs)
                else:
                    hidden_states = block(hidden_states)

            hidden_states = self.upsampler(hidden_states)

            for block in self.post_upsample_res_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block), hidden_states, **ckpt_kwargs)
                else:
                    hidden_states = block(hidden_states)

            hidden_states = self.final_conv(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
        else:
            hidden_states = self.initial_conv(hidden_states)
            hidden_states = self.initial_norm(hidden_states)
            hidden_states = self.initial_activation(hidden_states)

            for block in self.res_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block), hidden_states, **ckpt_kwargs)
                else:
                    hidden_states = block(hidden_states)

            if self.temporal_upsample:
                hidden_states = self.upsampler(hidden_states)
                hidden_states = hidden_states[:, :, 1:, :, :]
            else:
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
                hidden_states = self.upsampler(hidden_states)
                hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

            for block in self.post_upsample_res_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block), hidden_states, **ckpt_kwargs)
                else:
                    hidden_states = block(hidden_states)

            hidden_states = self.final_conv(hidden_states)

        return hidden_states
