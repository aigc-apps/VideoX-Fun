# Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_image.py
# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

"""MiniMax-H3-Image autoencoder.

MiniMax-H3-Image reuses MiniMax-H3's video VAE *encoder* — the causal 3D CNN simply runs over a single frame, which
its causal temporal convolutions map onto a single latent frame — and replaces the 3D ViT decoder with a smaller,
image-only one: 24 layers instead of 36, 16 heads instead of 32, a `(h, w)` rotary grid instead of `(t, h, w)`, and a
patch projection that expands each latent pixel into a `16 x 16` pixel block instead of a `4 x 16 x 16` voxel block.
Everything that is not decoder-shape — the encoder stack, the ViT block, the attention processor, the rotary
embedding, the mixed-precision loader — is reused from `minimax_h3_vae`.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.vae import (DecoderOutput,
                                               DiagonalGaussianDistribution)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version, logging
from diffusers.utils.accelerate_utils import apply_forward_hook

from .minimax_h3_conversion import MiniMaxH3MixedPrecisionLoaderMixin
from .minimax_h3_vae import (MiniMaxH3VideoEncoder3d,
                             MiniMaxH3VideoRotaryPosEmbed,
                             MiniMaxH3VideoTransformerBlock)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MiniMaxH3ImageViTDecoder2d(nn.Module):
    r"""
    Non-causal 2D ViT decoder. Every latent pixel becomes one token; `num_register_tokens` learned register tokens
    plus a single all-zero token are appended (all at position `0`), attended over with full self-attention, and
    dropped again before the patch projection expands each token into a `patch_size x patch_size` pixel block. The
    rotary embedding carries `(h, w)` coordinates only.
    """

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        num_layers: int = 24,
        num_attention_heads: int = 16,
        attention_head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.5,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        # Two axes, not three: `int(attention_head_dim * rope_dim_ratio)` is 32 here, which the 3-axis layout cannot
        # divide, and a released image decoder carries no time coordinate at all.
        self.rope = MiniMaxH3VideoRotaryPosEmbed(
            int(attention_head_dim * rope_dim_ratio), theta=rope_theta, num_axes=2
        )
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoTransformerBlock(
                    dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    ffn_mult=ffn_mult,
                    eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size * patch_size)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = hidden_states.shape

        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, num_channels)
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]

        register_tokens = self.register_tokens.expand(batch_size, -1, -1).to(hidden_states.dtype)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
            for size in (height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 1)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 2))
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)
        rotary_emb = self.rope(position_ids)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block, hidden_states, rotary_emb, **ckpt_kwargs
                )
            else:
                hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[:, :num_patches, :]

        patch_size = self.patch_size
        hidden_states = hidden_states.view(batch_size, height, width, self.out_channels, patch_size, patch_size)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5).contiguous()
        return hidden_states.reshape(batch_size, self.out_channels, height * patch_size, width * patch_size)


class AutoencoderKLMiniMaxH3Image(MiniMaxH3MixedPrecisionLoaderMixin, ModelMixin, ConfigMixin):
    r"""
    A VAE model with a causal 3D CNN encoder and a non-causal 2D ViT decoder, used in
    [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI). The encoder is the MiniMax-H3 video encoder run over a
    single frame; the decoder is an image-only ViT.

    Latents are normalized with per-channel `latents_mean` / `latents_std` rather than a `scaling_factor`; a pipeline
    encodes with `(latent - latents_mean) / latents_std` and decodes with `latent * latents_std + latents_mean`.

    The pixel convention is the video VAE's: ImageNet-normalized RGB over a `[0, 1]` base range, not the usual
    `[-1, 1]`. `encode` expects `(pixel - imagenet_mean) / imagenet_std` and `decode` returns values in that same
    space, so a pipeline has to apply `sample * imagenet_std + imagenet_mean` (mean `(0.485, 0.456, 0.406)`, std
    `(0.229, 0.224, 0.225)`) and clamp to `[0, 1]` before postprocessing.

    Unlike most autoencoders, spatial tiling is **on by default**: MiniMax-H3-Image was released with tiling enabled
    for both encoding and decoding, and the released images are the blended-tile ones, so disabling tiling changes
    the output. Use `enable_tiling` to change the tile geometry, `disable_tiling` to turn it off. The tiling itself is
    purely spatial here — there is no temporal chunking, no `clip_length` and no `token_drop` for a single frame.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxH3VideoResnetBlock3d", "MiniMaxH3VideoTransformerBlock"]
    _repeated_blocks = ["MiniMaxH3VideoTransformerBlock"]
    _skip_layerwise_casting_patterns = ["norm"]
    # The released checkpoint is float32 and, as in the video VAE, the whole model is pinned so a pipeline-level
    # `torch_dtype=torch.bfloat16` cannot round the encoder or the decoder block stack: the reused
    # `MiniMaxH3VideoTransformerBlock` keeps its residual stream at the activation dtype, which is only consistent
    # while the decoder is a single dtype.
    _keep_in_fp32_modules = ["encoder", "decoder", "quant_conv", "post_quant_conv"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 24,
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: Tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: Tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
        decoder_num_layers: int = 24,
        decoder_num_attention_heads: int = 16,
        decoder_attention_head_dim: int = 64,
        decoder_num_register_tokens: int = 4,
        decoder_ffn_mult: int = 4,
        decoder_rope_theta: float = 100.0,
        decoder_rope_dim_ratio: float = 0.5,
        decoder_norm_eps: float = 1e-5,
        latents_mean: Tuple[float, ...] = (0.0,) * 24,
        latents_std: Tuple[float, ...] = (1.0,) * 24,
    ) -> None:
        super().__init__()

        self.spatial_compression_ratio = math.prod(spatial_downsample_factors)

        # The video encoder, run over one frame: `temporal_downsample_factors` still multiply to 4, but the causal
        # convolutions map a single pixel frame onto a single latent frame.
        self.encoder = MiniMaxH3VideoEncoder3d(
            in_channels=in_channels,
            out_channels=2 * latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            spatial_downsample_factors=spatial_downsample_factors,
            temporal_downsample_factors=temporal_downsample_factors,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        # The decoder is 2D, so its input projection is a `1x1` conv over `(height, width)` rather than over frames.
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.decoder = MiniMaxH3ImageViTDecoder2d(
            in_channels=latent_channels,
            out_channels=out_channels,
            patch_size=self.spatial_compression_ratio,
            num_layers=decoder_num_layers,
            num_attention_heads=decoder_num_attention_heads,
            attention_head_dim=decoder_attention_head_dim,
            num_register_tokens=decoder_num_register_tokens,
            ffn_mult=decoder_ffn_mult,
            rope_theta=decoder_rope_theta,
            rope_dim_ratio=decoder_rope_dim_ratio,
            norm_eps=decoder_norm_eps,
        )

        # When decoding a batch of latents at a time, one can save memory by slicing across the batch dimension to
        # perform decoding of a single latent at a time.
        self.use_slicing = False

        # Large images are encoded and decoded in overlapping tiles that are linearly blended back together.
        # MiniMax-H3-Image ships with tiling enabled.
        self.use_tiling = True

        # The tile size in pixel space, and the minimum overlap between two neighbouring tiles. The actual overlaps are
        # widened (in multiples of `spatial_compression_ratio`) so that every tile boundary stays latent-aligned.
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64

    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            value = kwargs["value"]
        elif "enable" in kwargs:
            value = kwargs["enable"]
        else:
            raise ValueError("Invalid set gradient checkpointing")
        self.decoder.gradient_checkpointing = value
        for down_block in self.encoder.down_blocks:
            down_block.gradient_checkpointing = value

    def enable_slicing(self) -> None:
        r"""Decode one image of a batch at a time, which lowers the memory requirement."""
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""Decode a whole batch at once again."""
        self.use_slicing = False

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_min_overlap_height: Optional[int] = None,
        tile_sample_min_overlap_width: Optional[int] = None,
    ) -> None:
        r"""
        Enable tiled VAE encoding/decoding. The VAE splits the image into tiles, encodes or decodes each tile
        separately and linearly blends the overlaps back together, which lowers the memory requirement and allows
        processing larger images.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_overlap_height = tile_sample_min_overlap_height or self.tile_sample_min_overlap_height
        self.tile_sample_min_overlap_width = tile_sample_min_overlap_width or self.tile_sample_min_overlap_width

    def disable_tiling(self) -> None:
        r"""Disable tiled VAE encoding/decoding."""
        self.use_tiling = False

    def _split_tiles(
        self, length: int, tile_size: int, min_overlap: int
    ) -> Tuple[List[int], List[int], List[int]]:
        r"""
        Lay `tile_size`-wide tiles over `length` pixels. The number of tiles is the smallest one whose union can cover
        `length` while keeping every overlap at least `min_overlap`; the slack is then distributed round-robin over the
        overlaps in whole `spatial_compression_ratio` steps so that every tile boundary stays latent-aligned.
        """
        if tile_size >= length:
            return [0], [length], []

        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1

        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // self.spatial_compression_ratio):
            overlaps[i % (num_tiles - 1)] += self.spatial_compression_ratio

        tile_start_indices = [0]
        for i in range(num_tiles - 1):
            tile_start_indices.append(tile_start_indices[-1] + tile_size - overlaps[i])
        return tile_start_indices, [tile_size] * num_tiles, overlaps

    def _blend(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).view(shape)
        weight_b = (positions / blend_extent).view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

        if blend_extent == b.shape[dim]:
            return blended
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)

    def _stitch_tiles(
        self,
        tiles: List[List[torch.Tensor]],
        height_overlaps: List[int],
        width_overlaps: List[int],
    ) -> torch.Tensor:
        result_rows = []
        for i, row in enumerate(tiles):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # The encoder is a causal video encoder; an image is a single frame, which maps to a single latent frame, so
        # the latent time axis is squeezed straight off again.
        if not self.use_tiling:
            return self.quant_conv(self.encoder(x[:, :, None]))[:, :, -1]

        height, width = x.shape[-2], x.shape[-1]
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )

        rows = []
        for i_pos, i_len in zip(y_indices, y_lengths):
            row = []
            for j_pos, j_len in zip(x_indices, x_lengths):
                tile = x[..., i_pos: i_pos + i_len, j_pos: j_pos + j_len]
                row.append(self.quant_conv(self.encoder(tile[:, :, None]))[:, :, -1])
            rows.append(row)

        latent_y_overlaps = [overlap // self.spatial_compression_ratio for overlap in y_overlaps]
        latent_x_overlaps = [overlap // self.spatial_compression_ratio for overlap in x_overlaps]
        return self._stitch_tiles(rows, latent_y_overlaps, latent_x_overlaps)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))

        # Tiles are laid out in pixel space and then mapped back onto the latent grid.
        height = z.shape[-2] * self.spatial_compression_ratio
        width = z.shape[-1] * self.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )

        ratio = self.spatial_compression_ratio
        rows = []
        for i_pos, i_len in zip(y_indices, y_lengths):
            row = []
            for j_pos, j_len in zip(x_indices, x_lengths):
                tile = z[
                    ...,
                    i_pos // ratio: i_pos // ratio + i_len // ratio,
                    j_pos // ratio: j_pos // ratio + j_len // ratio,
                ]
                row.append(self.decoder(self.post_quant_conv(tile)))
            rows.append(row)

        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        r"""
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`):
                Input batch of images, shape `(batch_size, in_channels, height, width)`, in the VAE's
                ImageNet-normalized pixel space.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return an [`~models.autoencoders.autoencoder_kl.AutoencoderKLOutput`] instead of a plain
                tuple.

        Returns:
            The latent distribution of the encoded images, over latents of shape `(batch_size, latent_channels,
            height // spatial_compression_ratio, width // spatial_compression_ratio)`. Note that MiniMax-H3-Image
            normalizes the sampled latents with `latents_mean` / `latents_std` afterwards.
        """
        # The encoder is pinned to float32 by `_keep_in_fp32_modules`; align lower-precision pixels with the weights.
        x = x.to(self.encoder.conv_in.weight.dtype)
        if self.use_slicing and x.shape[0] > 1:
            moments = torch.cat([self._encode(x_slice) for x_slice in x.split(1)])
        else:
            moments = self._encode(x)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        r"""
        Decode a batch of latents.

        Args:
            z (`torch.Tensor`):
                Input batch of latents, shape `(batch_size, latent_channels, height, width)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                The decoded images, shape `(batch_size, out_channels, height * spatial_compression_ratio, width *
                spatial_compression_ratio)`, in the VAE's ImageNet-normalized pixel space.
        """
        # The decoder is mixed precision, so align with the first module that consumes the latents.
        z = z.to(self.post_quant_conv.weight.dtype)
        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat([self._decode(z_slice) for z_slice in z.split(1)])
        else:
            decoded = self._decode(z)
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        r"""
        Encode then decode a batch of images.

        Args:
            sample (`torch.Tensor`):
                Input batch of images, shape `(batch_size, in_channels, height, width)`.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample the posterior instead of taking its mode.
            generator (`torch.Generator`, *optional*):
                Generator used when `sample_posterior=True`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                The round-tripped images, shape `(batch_size, out_channels, height, width)`.
        """
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(z, return_dict=return_dict)
