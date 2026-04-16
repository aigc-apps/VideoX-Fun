# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import glob
import json
import math
import os
import types
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version, logging
from torch import nn

from ..dist import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    usp_attn_forward, xFuserLongContextAttention)
from ..utils import cfg_skip
from .attention_utils import attention
from .cache_utils import TeaCache
from .wan_camera_adapter import SimpleAdapter


def sinusoidal_embedding_1d(dim, position):
    """
    Generate 1D sinusoidal positional embeddings.
    
    Args:
        dim: Embedding dimension (must be even)
        position: Position indices tensor
    
    Returns:
        Sinusoidal embedding tensor with shape (len(position), dim)
    """
    # Validate dimension is even and split into half
    assert dim % 2 == 0
    half = dim // 2
    # Convert position to float64 for numerical precision
    position = position.type(torch.float64)

    # Compute sinusoidal frequencies: position * 10000^(-i/dim/2)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    # Concatenate cosine and sine components
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    """
    Compute RoPE frequency parameters for a given sequence length.
    
    Args:
        max_seq_len: Maximum sequence length
        dim: Dimension of the embedding (must be even)
        theta: Base frequency scaling factor
    
    Returns:
        Complex frequency tensor with shape (max_seq_len, dim/2)
    """
    assert dim % 2 == 0
    # Compute inverse frequency scaling: 1/theta^(2i/dim)
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    # Convert to complex form: exp(i * freq)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# Modified from https://github.com/thu-ml/RIFLEx/blob/main/riflex_utils.py
@amp.autocast(enabled=False)
def get_1d_rotary_pos_embed_riflex(
    pos: Union[np.ndarray, int],
    dim: int,
    theta: float = 10000.0,
    use_real=False,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
    L_test_scale: Optional[int] = None,
):
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    # Convert position to tensor if needed
    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    # Compute base frequencies: 1/theta^(2i/dim)
    freqs = 1.0 / torch.pow(theta,
        torch.arange(0, dim, 2).to(torch.float64).div(dim))

    # === Riflex modification start ===
    # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===
    if L_test_scale is not None:
        freqs[k-1] = freqs[k-1] / L_test_scale

    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # Lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Complex64     # [S, D/2]
        return freqs_cis


@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs):
    """
    Apply rotary positional embedding to input tensor.
    
    Args:
        x: Input tensor with shape (batch, seq_len, n_channels, c*2)
        grid_sizes: Grid dimensions (f, h, w) for each sample
        freqs: Precomputed frequency parameters
    
    Returns:
        Tensor with RoPE applied
    """
    dtype = x.dtype
    n, c = x.size(2), x.size(3) // 2

    # Split frequencies into temporal, height, and width components
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # Process each sample in the batch
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # Reshape and convert to complex numbers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        # Broadcast frequencies to match spatial dimensions
        freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
        dim=-1).reshape(seq_len, 1, -1)

        # Apply rotation: x * exp(i*freq)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # Concatenate with padding tokens (if any)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # Append to collection
        output.append(x_i)
    return torch.stack(output).to(dtype)


def rope_apply_qk(q, k, grid_sizes, freqs):
    """
    Apply RoPE to both query and key tensors.
    
    Args:
        q: Query tensor
        k: Key tensor
        grid_sizes: Grid dimensions for spatial positioning
        freqs: Precomputed frequency parameters
    
    Returns:
        Tuple of (q, k) with RoPE applied
    """
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)
    return q, k


class WanRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    """Layer Normalization with dtype preservation."""

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x).type_as(x)

    def _norm(self, x):
        """Compute Layer Normalization with dtype handling."""
        if self.weight is not None:
            x = super().forward(x.to(self.weight.dtype))
            return x
        else:
            return super().forward(x)


class WanSelfAttention(nn.Module):
    """Wan self-attention mechanism with RoPE and optional windowed attention."""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # Query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q, k = rope_apply_qk(q, k, grid_sizes, freqs)

        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v=v.to(dtype),
            k_lens=seq_lens,
            window_size=self.window_size
        )

        # Output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    """Text-to-video cross-attention layer."""

    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        
        # Compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        
        # Compute attention
        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )

        # Output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    """Image-to-video cross-attention layer with separate image context processing."""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # Compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        img_x = attention(
            q.to(dtype), 
            k_img.to(dtype), 
            v_img.to(dtype), 
            k_lens=None
        )
        # Compute text cross-attention
        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )

        # Output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    """Generic cross-attention layer."""
    
    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        
        # Compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        
        # Compute attention
        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )

        # Output
        x = x.flatten(2)
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
    'cross_attn': WanCrossAttention,
}


class WanAttentionBlock(nn.Module):
    """Wan transformer block with self-attention, cross-attention, and FFN."""

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # Modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dtype=torch.bfloat16,
        t=0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C] or [B, L, 6, C] for modulation
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            context(Tensor): Shape [B, L_context, C]
            context_lens(Tensor): Shape [B]
        """
        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [e.squeeze(2) for e in e]
        else:        
            e = (self.modulation + e).chunk(6, dim=1)

        # Self-attention with modulation
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)

        y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype, t=t)
        x = x + y * e[2]

        # Cross-attention and FFN with modulation
        def cross_attn_ffn(x, context, context_lens, e):
            # Cross-attention: attend to text context
            x = x + self.cross_attn(self.norm3(x).to(x.dtype), context, context_lens, dtype, t=t)

            # FFN with modulation
            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)
            
            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):
    """Output head for reconstructing video from transformer features."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # Layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # Modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        if e.dim() > 2:
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            e = [e.squeeze(2) for e in e]
        else:
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        
        x = self.head(
            (self.norm(x) * (1 + e[1]) + e[0]).to(x.dtype)
        )
        return x


class MLPProj(torch.nn.Module):
    """MLP projection network for embedding transformation."""

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens



class WanTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        cross_attn_type=None,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to True):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            in_channels (`int`, *optional*, defaults to 16):
                Alias for in_dim (diffusers compatibility)
            hidden_size (`int`, *optional*, defaults to 2048):
                Alias for dim (diffusers compatibility)
            add_control_adapter (`bool`, *optional*, defaults to False):
                Enable camera control adapter
            in_dim_control_adapter (`int`, *optional*, defaults to 24):
                Input channels for control adapter
            downscale_factor_control_adapter (`int`, *optional*, defaults to 8):
                Downscale factor for control adapter
            add_ref_conv (`bool`, *optional*, defaults to False):
                Enable reference frame convolution
            in_dim_ref_conv (`int`, *optional*, defaults to 16):
                Input channels for reference convolution
            cross_attn_type (`str`, *optional*, defaults to None):
                Cross-attention type, auto-determined from model_type if None
        """

        super().__init__()

        # assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # Blocks
        if cross_attn_type is None:
            cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        # Head
        self.head = Head(dim, out_dim, patch_size, eps)

        # Buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        self.dim = dim
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1
        )

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:], downscale_factor=downscale_factor_control_adapter)
        else:
            self.control_adapter = None

        if add_ref_conv:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None

        self.teacache = None
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None
        self.gradient_checkpointing = False
        self.all_gather = None
        self.sp_world_size = 1
        self.sp_world_rank = 0
        self.init_weights()

    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            self.gradient_checkpointing = kwargs["value"]
            if hasattr(self, "motioner") and hasattr(self.motioner, "gradient_checkpointing"):
                self.motioner.gradient_checkpointing = kwargs["value"]
        elif "enable" in kwargs:
            self.gradient_checkpointing = kwargs["enable"]
            if hasattr(self, "motioner") and hasattr(self.motioner, "gradient_checkpointing"):
                self.motioner.gradient_checkpointing = kwargs["enable"]
        else:
            raise ValueError("Invalid set gradient checkpointing")

    def enable_teacache(
        self,
        coefficients,
        num_steps: int,
        rel_l1_thresh: float,
        num_skip_start_steps: int = 0,
        offload: bool = True,
    ):
        self.teacache = TeaCache(
            coefficients, num_steps, rel_l1_thresh=rel_l1_thresh, num_skip_start_steps=num_skip_start_steps, offload=offload
        )

    def share_teacache(
        self,
        transformer = None,
    ):
        self.teacache = transformer.teacache

    def disable_teacache(self):
        self.teacache = None

    def enable_cfg_skip(self, cfg_skip_ratio, num_steps):
        if cfg_skip_ratio != 0:
            self.cfg_skip_ratio = cfg_skip_ratio
            self.current_steps = 0
            self.num_inference_steps = num_steps
        else:
            self.cfg_skip_ratio = None
            self.current_steps = 0
            self.num_inference_steps = None

    def share_cfg_skip(
        self,
        transformer = None,
    ):
        self.cfg_skip_ratio = transformer.cfg_skip_ratio
        self.current_steps = transformer.current_steps
        self.num_inference_steps = transformer.num_inference_steps

    def disable_cfg_skip(self):
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None

    def enable_riflex(
        self,
        k = 6,
        L_test = 66,
        L_test_scale = 4.886,
    ):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                get_1d_rotary_pos_embed_riflex(1024, self.d - 4 * (self.d // 6), use_real=False, k=k, L_test=L_test, L_test_scale=L_test_scale),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def disable_riflex(self):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                rope_params(1024, self.d - 4 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def enable_multi_gpus_inference(self,):
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather

        # For normal model.
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn)

        # For vace model.
        if hasattr(self, 'vace_blocks'):
            for block in self.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)

    @cfg_skip()
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        full_ref=None,
        subject_ref=None,
        cond_flag=True,
    ):
        r"""
        Forward pass through the diffusion model
        
        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B] or [B, seq_len] for per-token timesteps
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode, shape [B, 257, dim]
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            y_camera (Tensor, *optional*):
                Camera control features for camera-controlled generation
            full_ref (Tensor, *optional*):
                Full reference frame for fun control mode, shape [B, C, F, H, W]
            subject_ref (Tensor, *optional*):
                Subject reference frames for phantom mode, shape [B, C, F_ref, H, W]
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether this is conditional or unconditional forward pass
        
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # Wan2.2 I2V doesn't need a clip embeddings (handled separately if provided)
        # Get device and dtype
        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        # Concatenate condition video to input (for I2V)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        
        # Patch embedding: convert video to sequence of patches
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        
        # Add control adapter features (for camera control)
        if self.control_adapter is not None and y_camera is not None:
            y_camera = self.control_adapter(y_camera)
            x = [u + v for u, v in zip(x, y_camera)]

        # Get grid_sizes
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        
        # Flatten x
        x = [u.flatten(2).transpose(1, 2) for u in x]

        # Add full reference frame (for fun control)
        if self.ref_conv is not None and full_ref is not None:
            full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += full_ref.size(1)
            x = [torch.concat([_full_ref.unsqueeze(0), u], dim=1) for _full_ref, u in zip(full_ref, x)]
            if t.dim() != 1 and t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                last_elements = t[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                t = torch.cat([padding, t], dim=1)

        # Add reference for phantom
        if subject_ref is not None:
            subject_ref_frames = subject_ref.size(2)
            subject_ref = self.patch_embedding(subject_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + subject_ref_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += subject_ref.size(1)
            x = [torch.concat([u, _subject_ref.unsqueeze(0)], dim=1) for _subject_ref, u in zip(subject_ref, x)]
            if t.dim() != 1 and t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                last_elements = t[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                t = torch.cat([t, padding], dim=1)
        
        # Padding for multi-gpu inference
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # Time embeddings with sinusoidal encoding
        if t.dim() != 1:
            if t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                last_elements = t[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                t = torch.cat([t, padding], dim=1)
            bt = t.size(0)
            ft = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, ft).unflatten(0, (bt, seq_len)).float()).to(dtype)
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        else:
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float()).to(dtype)
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # Context: text embeddings (padded to fixed length)
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # Shape: [B, 257, dim]
            context = torch.concat([context_clip, context], dim=1)

        # Context Parallel: split input across GPUs
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            if t.dim() != 1:
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
                e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]
        
        # TeaCache: skip computation when change is small
        if self.teacache is not None:
            if cond_flag:
                if t.dim() != 1:
                    modulated_inp = e0[:, -1, :]
                else:
                    modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    self.should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        self.should_calc = False
                    else:
                        self.should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = self.should_calc
            else:
                self.should_calc = self.teacache.should_calc
        
        # Prepare checkpointing utilities
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
        
        # Main transformer loop
        if self.teacache is not None:
            if not self.should_calc:
                # Skip: use cached residual
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            dtype,
                            t,
                            **ckpt_kwargs,
                        )
                    else:
                        # Arguments
                        kwargs = dict(
                            e=e0,
                            seq_lens=seq_lens,
                            grid_sizes=grid_sizes,
                            freqs=self.freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype,
                            t=t  
                        )
                        x = block(x, **kwargs)
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        e0,
                        seq_lens,
                        grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        dtype,
                        t,
                        **ckpt_kwargs,
                    )
                else:
                    # Arguments
                    kwargs = dict(
                        e=e0,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=self.freqs,
                        context=context,
                        context_lens=context_lens,
                        dtype=dtype,
                        t=t  
                    )
                    x = block(x, **kwargs)

        # Head: project to output space
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x, e, **ckpt_kwargs)
        else:
            x = self.head(x, e)

        # Context Parallel: gather results from all GPUs
        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # Remove full reference frame from output
        if self.ref_conv is not None and full_ref is not None:
            full_ref_length = full_ref.size(1)
            x = x[:, full_ref_length:]
            grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # Remove subject reference frames from output
        if subject_ref is not None:
            subject_ref_length = subject_ref.size(1)
            x = x[:, :-subject_ref_length]
            grid_sizes = torch.stack([torch.tensor([u[0] - subject_ref_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # Unpatchify: reconstruct video from patches
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)

        # Increment teacache counter and reset if completed
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x


    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # Basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # Init output layer
        nn.init.zeros_(self.head.head.weight)

    @staticmethod
    def _convert_from_wan_model_config(config: dict) -> dict:
        """
        Convert WanModel config to WanTransformer3DModel config.
        This enables loading WanModel checkpoints with WanTransformer3DModel.
        """
        new_config = config.copy()
        
        # Convert has_image_input -> model_type
        if 'has_image_input' in new_config:
            new_config['model_type'] = 'i2v' if new_config.pop('has_image_input') else 't2v'
        
        # Convert has_ref_conv -> add_ref_conv
        if 'has_ref_conv' in new_config:
            new_config['add_ref_conv'] = new_config.pop('has_ref_conv')
        
        # Set WanTransformer3DModel specific defaults
        new_config.setdefault('cross_attn_type', 'cross_attn')
        new_config.setdefault('qk_norm', True)
        new_config.setdefault('cross_attn_norm', True)
        new_config.setdefault('text_len', 512)
        new_config.setdefault('window_size', (-1, -1))
        
        # Compatibility fields
        if 'in_dim' in new_config:
            new_config['in_channels'] = new_config['in_dim']
        if 'dim' in new_config:
            new_config['hidden_size'] = new_config['dim']
        
        # Remove WanModel-specific keys that WanTransformer3DModel doesn't use
        keys_to_remove = [
            'has_image_pos_emb', 'require_clip_embedding', 'require_vae_embedding',
            'seperated_timestep', 'fuse_vae_embedding_in_latents', '_class_name',
            '_diffusers_version', '_name_or_path'
        ]
        for key in keys_to_remove:
            new_config.pop(key, None)
        
        return new_config

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        # Auto-convert WanModel config to WanTransformer3DModel config
        if 'has_image_input' in config:
            print("Detected diffsynth config, converting to WanTransformer3DModel config...")
            config = cls._convert_from_wan_model_config(config)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                if diffusers_version >= "0.33.0":
                    from diffusers.models.model_loading_utils import \
                        load_model_dict_into_meta
                else:
                    from diffusers.models.modeling_utils import \
                        load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate
                
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file, safe_open
                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    print(model_files_safetensors)
                    for _model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(_model_file_safetensors)
                        for key in _state_dict:
                            state_dict[key] = _state_dict[key]

                if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
                    model.state_dict()['patch_embedding.weight'][:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight'][:, :model.state_dict()['patch_embedding.weight'].size()[1], :, :]
                    model.state_dict()['patch_embedding.weight'][:, state_dict['patch_embedding.weight'].size()[1]:, :, :] = 0
                    state_dict['patch_embedding.weight'] = model.state_dict()['patch_embedding.weight']

                filtered_state_dict = {}
                for key in state_dict:
                    if key in model.state_dict() and model.state_dict()[key].size() == state_dict[key].size():
                        filtered_state_dict[key] = state_dict[key]
                    else:
                        print(f"Skipping key '{key}' due to size mismatch or absence in model.")
                        
                model_keys = set(model.state_dict().keys())
                loaded_keys = set(filtered_state_dict.keys())
                missing_keys = model_keys - loaded_keys

                def initialize_missing_parameters(missing_keys, model_state_dict, torch_dtype=None):
                    initialized_dict = {}
                    
                    with torch.no_grad():
                        for key in missing_keys:
                            param_shape = model_state_dict[key].shape
                            param_dtype = torch_dtype if torch_dtype is not None else model_state_dict[key].dtype
                            if 'weight' in key:
                                if any(norm_type in key for norm_type in ['norm', 'ln_', 'layer_norm', 'group_norm', 'batch_norm']):
                                    initialized_dict[key] = torch.ones(param_shape, dtype=param_dtype)
                                elif 'embedding' in key or 'embed' in key:
                                    initialized_dict[key] = torch.randn(param_shape, dtype=param_dtype) * 0.02
                                elif 'head' in key or 'output' in key or 'proj_out' in key:
                                    initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                                elif len(param_shape) >= 2:
                                    initialized_dict[key] = torch.empty(param_shape, dtype=param_dtype)
                                    nn.init.xavier_uniform_(initialized_dict[key])
                                else:
                                    initialized_dict[key] = torch.randn(param_shape, dtype=param_dtype) * 0.02
                            elif 'bias' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                            elif 'running_mean' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                            elif 'running_var' in key:
                                initialized_dict[key] = torch.ones(param_shape, dtype=param_dtype)
                            elif 'num_batches_tracked' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=torch.long)
                            else:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                            
                    return initialized_dict

                if missing_keys:
                    print(f"Missing keys will be initialized: {sorted(missing_keys)}")
                    initialized_params = initialize_missing_parameters(
                        missing_keys, 
                        model.state_dict(), 
                        torch_dtype
                    )
                    filtered_state_dict.update(initialized_params)

                if diffusers_version >= "0.33.0":
                    # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
                    # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
                    load_model_dict_into_meta(
                        model,
                        filtered_state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(filtered_state_dict)
                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        filtered_state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        print(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )
                
                return model
            except Exception as e:
                print(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )
        
        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]
        
        if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
            model.state_dict()['patch_embedding.weight'][:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight'][:, :model.state_dict()['patch_embedding.weight'].size()[1], :, :]
            model.state_dict()['patch_embedding.weight'][:, state_dict['patch_embedding.weight'].size()[1]:, :, :] = 0
            state_dict['patch_embedding.weight'] = model.state_dict()['patch_embedding.weight']
        
        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
                
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)
        
        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")
        
        model = model.to(torch_dtype)
        return model


class Wan2_2Transformer3DModel(WanTransformer3DModel):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True
    
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,
        in_dim_ref_conv=16,
    ):
        r"""
        Initialize the diffusion model backbone.
        
        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to True):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            in_channels (`int`, *optional*, defaults to 16):
                Alias for in_dim (diffusers compatibility)
            hidden_size (`int`, *optional*, defaults to 2048):
                Alias for dim (diffusers compatibility)
            add_control_adapter (`bool`, *optional*, defaults to False):
                Enable camera control adapter
            in_dim_control_adapter (`int`, *optional*, defaults to 24):
                Input channels for control adapter
            downscale_factor_control_adapter (`int`, *optional*, defaults to 8):
                Downscale factor for control adapter
            add_ref_conv (`bool`, *optional*, defaults to False):
                Enable reference frame convolution
            in_dim_ref_conv (`int`, *optional*, defaults to 16):
                Input channels for reference convolution
        """
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            in_channels=in_channels,
            hidden_size=hidden_size,
            add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            downscale_factor_control_adapter=downscale_factor_control_adapter,
            add_ref_conv=add_ref_conv,
            in_dim_ref_conv=in_dim_ref_conv,
            cross_attn_type="cross_attn"
        )
        
        if hasattr(self, "img_emb"):
            del self.img_emb
