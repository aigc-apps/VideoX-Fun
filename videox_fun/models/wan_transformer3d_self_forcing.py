# Modified from https://github.com/guandeh17/Self-Forcing/blob/main/wan/modules/causal_model.py
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
import torch._dynamo as dynamo
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version, logging
from torch import nn
from torch.nn.attention.flex_attention import (BlockMask, create_block_mask,
                                               flex_attention)

from ..dist import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    usp_attn_self_forcing_forward, xFuserLongContextAttention)
from .attention_utils import attention
from .wan_transformer3d import (MLPProj, WanLayerNorm, WanRMSNorm,
                                WanTransformer3DModel, rope_apply, rope_params,
                                sinusoidal_embedding_1d)

if dynamo.config.cache_size_limit < 128:
    dynamo.config.cache_size_limit = 128

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


@amp.autocast(enabled=False)
@torch.compiler.disable()
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """
    Apply causal rotary positional embedding with frame offset support.
    
    This function applies RoPE with a starting frame offset, enabling causal
    inference where different frames can have different positional indices.
    
    Args:
        x: Input tensor with shape (batch, seq_len, n_channels, c*2)
        grid_sizes: Grid dimensions (f, h, w) for each sample
        freqs: Precomputed frequency parameters
        start_frame: Starting frame index for causal positioning
    
    Returns:
        Tensor with causal RoPE applied
    """
    n, c = x.size(2), x.size(3) // 2

    # Split freqs into temporal, height, and width components
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # Process each sample in the batch
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # Reshape and convert to complex numbers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        # Broadcast frequencies with start_frame offset for temporal dimension
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
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
    return torch.stack(output).type_as(x)


class CasualWanSelfAttention(nn.Module):
    """Wan self-attention mechanism with RoPE and optional windowed attention."""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 local_attn_size=-1,
                 sink_size=0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

        # Layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None, 
        dtype=torch.bfloat16, 
        t=0
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask): Block mask for flex attention
            kv_cache: KV cache for causal self-attention
            current_start: Current starting position in token sequence
            cache_start: Cache starting position
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # Query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        if kv_cache is None:
            # Check if this is teacher forcing training (sequence length is doubled)
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                # Split into clean and noisy parts for teacher forcing
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # Apply same RoPE to both clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                # Pad to 128 multiple for flex attention
                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                # Apply flex attention with block mask
                if padded_length != 0:
                    x = flex_attention(
                        query=padded_roped_query.transpose(2, 1),
                        key=padded_roped_key.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask
                    )[:, :, :-padded_length].transpose(2, 1)
                else:
                    x = flex_attention(
                        query=padded_roped_query.transpose(2, 1),
                        key=padded_roped_key.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask
                    ).transpose(2, 1)
            else:
                # Standard inference without teacher forcing
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                # Pad to 128 multiple for flex attention
                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                # Apply flex attention with block mask
                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            # Causal inference with KV cache
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            # Apply causal RoPE with frame offset
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            
            # Compute attention with local window
            if self.local_attn_size == -1:
                max_attention_size = local_end_index
            else:
                max_attention_size = self.local_attn_size * frame_seqlen
            x = attention(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - max_attention_size):local_end_index]
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x


class CasualWanT2VCrossAttention(CasualWanSelfAttention):
    """Text-to-video cross-attention layer."""

    def forward(self, x, context, context_lens, crossattn_cache=None, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # Compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            # Use cached key/value if available
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # Compute attention
        x = attention(q, k, v, k_lens=context_lens)

        # Output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(CasualWanSelfAttention):
    """Image-to-video cross-attention layer with separate image context processing."""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 local_attn_size=-1,
                 sink_size=0):
        super().__init__(dim, num_heads, window_size, qk_norm, eps, local_attn_size, sink_size)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, crossattn_cache=None, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        # Split context into image and text parts
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # Compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        # Image cross-attention
        img_x = attention(q, k_img, v_img, k_lens=None)
        # Text cross-attention
        x = attention(q, k, v, k_lens=context_lens)

        # Output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanCrossAttention(CasualWanSelfAttention):
    """Generic cross-attention layer."""
    
    def forward(self, x, context, context_lens, crossattn_cache=None, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # Compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        # Compute attention
        x = attention(q.to(dtype), k.to(dtype), v.to(dtype), k_lens=context_lens)
        # Output
        x = x.flatten(2)
        x = self.o(x.to(dtype))
        return x


# Define local cross-attention classes mapping
WAN_SELF_FORCING_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': CasualWanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
    'cross_attn': WanCrossAttention,
}


class CasualWanAttentionBlock(nn.Module):
    """Wan transformer block with self-attention, cross-attention, and FFN."""

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 local_attn_size=-1,
                 sink_size=0):
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
        self.self_attn = CasualWanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps, local_attn_size, sink_size)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_SELF_FORCING_CROSSATTENTION_CLASSES[cross_attn_type](dim,
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
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        block_mask=None,
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
            kv_cache: KV cache for causal self-attention
            crossattn_cache: Cross-attention cache
            current_start: Current starting position in token sequence
            cache_start: Cache starting position
            block_mask: Block mask for flex attention
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # Self-attention with modulation
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        # Residual connection with modulation
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # Cross-attention and FFN function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):
    """
    Causal head with per-frame modulation for Self-Forcing inference.
    
    Unlike the base Head class which expects [B, C] timestep embeddings,
    CausalHead expects [B, F, 1, C] per-frame timestep embeddings and applies
    modulation independently to each frame before the final projection.
    """

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
            e(Tensor): Shape [B, F, 1, C]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]

        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        # Apply modulation per frame and project to output
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class WanTransformer3DModel_SelfForcing(WanTransformer3DModel):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """
    # _no_split_modules = ['CasualWanAttentionBlock']
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
        
        # Self-Forcing causal inference parameters
        local_attn_size=-1,
        sink_size=0,
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
            local_attn_size (`int`, *optional*, defaults to -1):
                Local attention window size (-1 for global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Sink token size for local attention
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
            cross_attn_type=cross_attn_type
        )
        # Blocks
        if cross_attn_type is None:
            cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CasualWanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                window_size, qk_norm, cross_attn_norm, eps, local_attn_size, sink_size
            )
            for _ in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        # Head
        self.head = CausalHead(dim, out_dim, patch_size, eps)
        
        # Self-forcing causal inference state
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.block_mask = None
        self.num_frame_per_block = 1
        self.independent_first_frame = False

        # Other parameters
        self.gradient_checkpointing = False
        self.all_gather = None
        self.sp_world_size = 1
        self.sp_world_rank = 0
        self.init_weights()

    def enable_multi_gpus_inference(self):
        """Enable multi-GPU inference with sequence parallelism for KV cache mode."""
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather
        
        # Replace self_attn forward method with USP version
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_self_forcing_forward, block.self_attn)

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

    def create_block_mask_for_training(
        self,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
        independent_first_frame: bool = False,
        device: torch.device | str = "cpu"
    ):
        """
        Create block-wise causal mask for Self-Forcing training.
        
        This creates a mask where each block can only attend to previous blocks,
        implementing causal self-attention without KV cache (using flex attention).
        
        Args:
            num_frames: Number of frames in the video
            frame_seqlen: Sequence length per frame (H * W / patch_size^2)
            num_frame_per_block: Number of frames per causal block
            independent_first_frame: If True, first frame is independent [1, N, N, ...]
            device: Device to create the mask on
        """
        total_length = num_frames * frame_seqlen
        
        # Right padding to multiple of 128 for flex attention
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        
        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        
        if not independent_first_frame:
            # Standard block pattern: [N, N, N, ...]
            frame_indices = torch.arange(
                start=0,
                end=total_length,
                step=frame_seqlen * num_frame_per_block,
                device=device
            )
            
            for tmp in frame_indices:
                ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block
        else:
            # Independent first frame pattern: [1, N, N, ...]
            # First frame
            ends[:frame_seqlen] = frame_seqlen
            # Remaining blocks
            frame_indices = torch.arange(
                start=frame_seqlen,
                end=total_length,
                step=frame_seqlen * num_frame_per_block,
                device=device
            )
            for tmp in frame_indices:
                ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block
        
        def attention_mask(b, h, q_idx, kv_idx):
            if self.local_attn_size == -1:
                # Global block-wise causal: can attend to all previous blocks
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                # Local attention: limited window
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - self.local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
        
        from torch.nn.attention.flex_attention import create_block_mask
        
        self.block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=True,
            device=device
        )
        
        # Store parameters for future reference
        self.num_frame_per_block = num_frame_per_block
        self.independent_first_frame = independent_first_frame

    def create_teacher_forcing_mask(
        self,
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1
    ) -> BlockMask:
        """
        Create block-wise teacher forcing mask for Self-Forcing training.
        
        This creates a mask where:
        - Clean frames (first half): causal attention within clean sequence
        - Noisy frames (second half): attend to all preceding clean frames + causal within noisy
        
        Sequence layout: [clean_frame_1, clean_frame_2, ..., noisy_frame_1, noisy_frame_2, ...]
        
        Args:
            device: Device to create the mask on
            num_frames: Number of frames in the video
            frame_seqlen: Sequence length per frame (H * W / patch_size^2)
            num_frame_per_block: Number of frames per causal block
            
        Returns:
            BlockMask for flex attention
        """
        total_length = num_frames * frame_seqlen * 2  # Clean + noisy
        
        # Right padding to multiple of 128 for flex attention
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        
        clean_ends = num_frames * frame_seqlen
        
        # For clean context frames: [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        
        # For noisy frames: need two intervals [context_start, context_end] + [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        
        # Block-wise causal mask
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        
        # Clean frames: causal attention
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size
        
        # Noisy frames: start positions
        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size
        
        # Noisy frames mask configuration
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # Attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # Attend to context tokens in previous blocks
            noise_context_ends[start:end] = block_index * attention_block_size
        
        def attention_mask(b, h, q_idx, kv_idx):
            # Clean frames mask
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # Noisy frames mask: attend to clean + self
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)
            
            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask
    
        self.block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=True,
            device=device
        )

        # Store parameters for future reference
        self.num_frame_per_block = num_frame_per_block

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        clean_x=None,
        aug_t=None,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # Params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        # Padding for multi-gpu inference
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # Concatenate clean features for teacher forcing
        if clean_x is not None:
            # Handle teacher forcing: concatenate clean and noisy features
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]
            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat(clean_x)
            x = torch.cat([clean_x, x], dim=1)

        # Manage block mask for training (kv_cache is None during training).
        # We must recreate the mask when switching between normal training
        # (causal mask) and teacher forcing (clean+noisy mask), because the
        # two modes expect different sequence lengths.
        num_frames_actual = None
        if kv_cache is None:
            num_frames_actual = grid_sizes[0, 0].item()
            frame_seqlen_actual = grid_sizes[0, 1].item() * grid_sizes[0, 2].item()
            
            if clean_x is not None:
                expected_mask_len = num_frames_actual * frame_seqlen_actual * 2
                is_teacher_forcing_mask = True
            else:
                expected_mask_len = num_frames_actual * frame_seqlen_actual
                is_teacher_forcing_mask = False
            
            if (self.block_mask is None or
                getattr(self, '_block_mask_expected_len', None) != expected_mask_len or
                getattr(self, '_block_mask_is_teacher_forcing', None) != is_teacher_forcing_mask):
                
                if is_teacher_forcing_mask:
                    if self.independent_first_frame:
                        raise NotImplementedError("Teacher forcing with independent first frame is not supported")
                    self.create_teacher_forcing_mask(
                        num_frames=num_frames_actual,
                        frame_seqlen=frame_seqlen_actual,
                        num_frame_per_block=self.num_frame_per_block,
                        device=device,
                    )
                else:
                    self.create_block_mask_for_training(
                        num_frames=num_frames_actual,
                        frame_seqlen=frame_seqlen_actual,
                        num_frame_per_block=self.num_frame_per_block,
                        independent_first_frame=self.independent_first_frame,
                        device=device
                    )
                self._block_mask_expected_len = expected_mask_len
                self._block_mask_is_teacher_forcing = is_teacher_forcing_mask

        # Time embeddings
        # Ensure t is 2D [B, num_frames] to align with inference behavior.
        # Training passes 1D t=[B] while inference passes 2D t=[B, F].
        # Without this, e0 shape differs and CasualWanAttentionBlock groups
        # tokens by modulation dim (6) instead of actual frames (F).
        if t.dim() == 1:
            num_frames_actual = grid_sizes[0, 0].item()
            t = t.unsqueeze(1).expand(-1, num_frames_actual)
        
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        
        # Handle teacher forcing: concatenate clean and noisy time embeddings
        if clean_x is not None:
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            if aug_t.dim() == 1:
                aug_t = aug_t.unsqueeze(1).expand(-1, num_frames_actual)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # context: text embeddings (padded to fixed length)
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

        # Arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index] if kv_cache else None,
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index] if kv_cache else None,
                        "crossattn_cache": crossattn_cache[block_index] if crossattn_cache else None,
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)
        
        # Remove clean part for teacher forcing output
        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # Context Parallel: gather results from all GPUs
        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # Head: project to output space
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # Unpatchify: reconstruct video from patches
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)