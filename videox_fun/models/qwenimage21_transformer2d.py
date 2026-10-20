# Modified from https://github.com/huggingface/diffusers/blob/cp-support-qwenimage2.1/src/diffusers/models/transformers/transformer_qwenimage21.py
# Copyright 2026 Qwen-Image Team, The HuggingFace Team. All rights reserved.
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
#
# NOTE: This is a VideoX-Fun port of the diffusers `QwenImage21Transformer2DModel`. It is adapted to the
# repository conventions (diffusers 0.32.x): attention routes through `videox_fun.models.attention_utils.attention`
# instead of `dispatch_attention_fn`, and the flex-attention / context-parallel paths of the upstream file are
# dropped. The block-causal prefill (exact multi-pass SDPA) and the prefix KV cache are preserved so results match
# the reference `QwenImage21AttnProcessor`. A repo-style `from_pretrained` (config.json + dict_mapping + shape-filtered
# low_cpu_mem_usage loading + missing-key init) mirrors `qwenimage_transformer2d.py`.

import glob
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph

from .attention_utils import attention
from ..dist import (QwenImage21MultiGPUsAttnProcessor,
                    get_sequence_parallel_rank,
                    get_sequence_parallel_world_size,
                    sequence_parallel_all_gather)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Each vision-language image slot represents a 2×2 group of latent tokens.
_IMG_TOKENS_PER_SLOT = 4


class QwenImage21KVLayerCache:
    """Per-layer KV cache for text and condition-image prefix tokens.

    Stores K and V projections (post-RoPE) for the prefix extracted during the first denoising step. Tensor format:
    ``(batch_size, num_prefix_tokens, num_heads, head_dim)``.
    """

    def __init__(self):
        self.k = None
        self.v = None

    def store(self, k: torch.Tensor, v: torch.Tensor):
        self.k = k
        self.v = v

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            raise RuntimeError("KV cache has not been populated yet.")
        return self.k, self.v


class QwenImage21KVCache:
    """Container for all transformer blocks' prefix KV caches."""

    def __init__(self, num_layers: int):
        self.layer_caches = [QwenImage21KVLayerCache() for _ in range(num_layers)]

    def get_layer(self, layer_idx: int) -> QwenImage21KVLayerCache:
        return self.layer_caches[layer_idx]


# Copied from diffusers.models.transformers.transformer_qwenimage.apply_rotary_emb_qwen
def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    Args:
        x (`torch.Tensor`): Query or key tensor to apply rotary embeddings. [B, S, H, D]
        freqs_cis (`tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class QwenImage21TemporalTimesteps(nn.Module):
    r"""Sinusoidal timestep embedding. `cos` occupies the first half of the channels and `sin` the second."""

    def __init__(self, timestep_dim: int, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor

        half = timestep_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = self.time_factor * timestep.float()
        args = timestep[:, None] * self.freqs[None].to(timestep.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.timestep_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(timestep.dtype)


class QwenImage21TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = QwenImage21TemporalTimesteps(timestep_dim=256)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, sample_proj_bias=False
        )

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))


class QwenImage21ZeroCenterRMSNorm(nn.Module):
    r"""
    RMSNorm whose learnable weight is stored zero-centered: the effective scale is `weight + 1`, computed in fp32.
    Checkpoints therefore store `scale - 1`.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        rrms = torch.rsqrt(torch.mean(hidden_states**2, dim=-1, keepdim=True) + self.eps)
        return (hidden_states * rrms * (self.weight.float() + 1)).to(input_dtype)


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.text_norm(hidden_states)
        hidden_states = self.in_layer(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.out_layer(hidden_states)


class QwenImage21SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out(self.activation_fn(self.gate_layer(hidden_states)) * self.proj(hidden_states))


class QwenImage21AdaLayerNormContinuous(nn.Module):
    r"""
    Final adaptive norm. Scale only — this variant emits no shift, so `linear` maps to `embedding_dim` rather than
    `2 * embedding_dim`.
    """

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=False, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_embedding: torch.Tensor,
        target_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = self.linear(self.silu(conditioning_embedding).to(hidden_states.dtype))
        scale = _select_modulation_rows(scale, target_token_mask)
        return self.norm(hidden_states) * (1 + scale)


def _select_modulation_rows(params: torch.Tensor, target_token_mask: Optional[torch.Tensor]) -> torch.Tensor:
    r"""
    Broadcast per-sample modulation `params` over the token axis.

    With `causal_condition`, `params` holds `batch_size + 1` rows: rows `[0, batch_size)` come from the real timestep
    and the trailing row from `t = 0`. Text and condition-image tokens take the `t = 0` row, target-image tokens take
    their own sample's row.

    Args:
        params (`torch.Tensor`): `(batch_size, dim)` without `causal_condition`, else `(batch_size + 1, dim)`.
        target_token_mask (`torch.Tensor`, *optional*): `(seq_len,)` bool, `True` at target-image positions. `None`
            disables the split and every token uses its own sample's row.
    """
    if target_token_mask is None:
        return params.unsqueeze(1)
    real, zero = params[:-1].unsqueeze(1), params[-1:].unsqueeze(0)
    return torch.where(target_token_mask.view(1, -1, 1), real, zero)


def _qwenimage21_prefix_segments(image_ids: torch.Tensor, prefix_len: int) -> List[Tuple[int, int, bool]]:
    """Split the prefix into `(start, end, is_text)` runs of equal `image_ids`.

    This is the block-causal structure in the form [`QwenImage21AttnProcessor`] consumes it. It only depends on
    `image_ids` and `prefix_len`, so the model derives it once per forward rather than in every processor call —
    `tolist()` is a device sync, and there is one processor call per layer.
    """
    prefix_ids = image_ids[:prefix_len].tolist()
    segments = []
    start = 0
    for index in range(1, prefix_len + 1):
        if index == prefix_len or prefix_ids[index] != prefix_ids[start]:
            segments.append((start, index, prefix_ids[start] < 0))
            start = index
    return segments


def _qwenimage21_project_qkv(
    attn: "QwenImage21Attention",
    hidden_states: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QKV projection, QK norm and RoPE. Returns ``[B, S, H, D]`` tensors; no KV-cache handling.

    Split out of `_qwenimage21_prepare_qkv` so the sequence-parallel processor can run an all-to-all
    between RoPE (sequence-split) and the cache (head-split).
    """
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))

    query = attn.norm_q(query).to(value.dtype)
    key = attn.norm_k(key).to(value.dtype)

    if rotary_emb is not None:
        query = apply_rotary_emb_qwen(query, rotary_emb, use_real=False)
        key = apply_rotary_emb_qwen(key, rotary_emb, use_real=False)

    return query, key, value


def _qwenimage21_apply_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_cache: Optional[QwenImage21KVLayerCache],
    kv_cache_mode: Optional[str],
    cache_write_slice: Optional[slice],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prefix KV-cache bookkeeping: store the prefix on ``extract``, prepend it on ``cached``."""
    if layer_cache is not None:
        if kv_cache_mode == "extract" and cache_write_slice is not None:
            # `clone()`, not `contiguous()`: at batch size 1 the prefix slice already counts as contiguous
            # (size-1 dims are ignored), so `contiguous()` returns the same view and the cache would pin the
            # whole prefill K/V for every step of the denoising loop.
            layer_cache.store(
                key[:, cache_write_slice].clone(),
                value[:, cache_write_slice].clone(),
            )
        elif kv_cache_mode == "cached":
            cached_k, cached_v = layer_cache.get()
            key = torch.cat([cached_k, key], dim=1)
            value = torch.cat([cached_v, value], dim=1)
    return key, value


def _qwenimage21_prepare_qkv(
    attn: "QwenImage21Attention",
    hidden_states: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
    layer_cache: Optional[QwenImage21KVLayerCache],
    kv_cache_mode: Optional[str],
    cache_write_slice: Optional[slice],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Shared QKV projection, norm, RoPE and KV-cache bookkeeping for the attention processor."""
    query, key, value = _qwenimage21_project_qkv(attn, hidden_states, rotary_emb)
    key, value = _qwenimage21_apply_cache(key, value, layer_cache, kv_cache_mode, cache_write_slice)
    seq_len_q = query.shape[1]
    return query, key, value, seq_len_q


def _qwenimage21_block_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_len_q: int,
    attention_mask: Optional[Any],
    segments: Optional[List[Tuple[int, int, bool]]],
    key_valid: Optional[torch.Tensor],
) -> torch.Tensor:
    """Exact block-causal attention shared by the single-GPU and sequence-parallel processors.

    ``segments is None`` -> decode: full attention over ``[cached prefix, target]``. Otherwise prefill:
    every prefix segment attends to the keys ``[0, end)`` (text segments also get a causal triangle over
    their own keys), then the target attends to everything. Returns ``[B, S_q, H, D]``.
    """
    if segments is None:
        hidden_states = attention(
            query,
            key,
            value,
            dropout_p=0.0,
            attn_mask=attention_mask,
        )
    else:
        prefix_len = segments[-1][1] if segments else 0
        outputs = []
        for start, end, is_text in segments:
            seg_mask = None
            if is_text:
                seg_len = end - start
                seg_mask = torch.cat(
                    [
                        torch.ones(seg_len, start, dtype=torch.bool, device=query.device),
                        torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool, device=query.device)),
                    ],
                    dim=1,
                )[None, None]
            if key_valid is not None:
                seg_key_valid = key_valid[:, None, None, :end]
                seg_mask = seg_key_valid if seg_mask is None else (seg_mask & seg_key_valid)
            outputs.append(
                attention(
                    query[:, start:end],
                    key[:, :end],
                    value[:, :end],
                    dropout_p=0.0,
                    attn_mask=seg_mask,
                )
            )
        outputs.append(
            attention(
                query[:, prefix_len:],
                key,
                value,
                dropout_p=0.0,
                attn_mask=None if key_valid is None else key_valid[:, None, None, :],
            )
        )
        hidden_states = torch.cat(outputs, dim=1)

    return hidden_states[:, :seq_len_q]


class QwenImage21AttnProcessor:
    r"""
    Attention processor for Qwen-Image 2.1 that needs neither `flex_attention` nor a compiled model.

    The prefill decomposes the block-causal mask into one attention call per prefix segment plus one for the target
    image, which is exact. The segment boundaries are computed once per forward by the model and passed in as
    `segments`. Attention is routed through the repository's unified `videox_fun.models.attention_utils.attention`
    helper, which dispatches to FlashAttention / SDPA depending on the backend and mask.
    """

    def __call__(
        self,
        attn: "QwenImage21Attention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[Any] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        layer_cache: Optional[QwenImage21KVLayerCache] = None,
        kv_cache_mode: Optional[str] = None,
        cache_write_slice: Optional[slice] = None,
        segments: Optional[List[Tuple[int, int, bool]]] = None,
        key_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value, seq_len_q = _qwenimage21_prepare_qkv(
            attn,
            hidden_states,
            rotary_emb,
            layer_cache,
            kv_cache_mode,
            cache_write_slice,
        )

        hidden_states = _qwenimage21_block_causal_attention(
            query, key, value, seq_len_q, attention_mask, segments, key_valid
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class QwenImage21Attention(nn.Module):
    r"""
    Attention module for [`QwenImage21TransformerBlock`]. Projection layout matches the legacy
    `diffusers.models.attention_processor.Attention` so Qwen-Image 2.x checkpoints load into it unchanged.
    """

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-6, processor: Optional[Any] = None):
        super().__init__()
        self.heads = heads
        self.inner_dim = heads * dim_head
        # 2.1 has no biases anywhere.
        self.use_bias = False

        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(0.0)])
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.processor = processor if processor is not None else QwenImage21AttnProcessor()

    def set_processor(self, processor: Any) -> None:
        self.processor = processor

    def get_processor(self) -> Any:
        return self.processor

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.processor(self, hidden_states, **kwargs)


@maybe_allow_in_graph
class QwenImage21TransformerBlock(nn.Module):
    r"""
    Single-stream block. Modulation is not learned per block — the parent model computes one shared `modulation`
    tensor and every block slices its own scales and gates out of it.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: int = 3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenImage21Attention(dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenImage21SwiGLUFeedForward(hidden_size=dim, mlp_hidden_size=dim * mlp_ratio)

    def _modulate(
        self,
        hidden_states: torch.Tensor,
        mod_params: torch.Tensor,
        target_token_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, gate = mod_params.chunk(2, dim=-1)
        scale = _select_modulation_rows(scale, target_token_mask)
        gate = _select_modulation_rows(gate, target_token_mask)
        return hidden_states * (1 + scale), gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        rotary_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[Any] = None,
        target_token_mask: Optional[torch.Tensor] = None,
        layer_cache: Optional[QwenImage21KVLayerCache] = None,
        kv_cache_mode: Optional[str] = None,
        cache_write_slice: Optional[slice] = None,
        segments: Optional[List[Tuple[int, int, bool]]] = None,
        key_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mod1, mod2 = modulation.chunk(2, dim=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), mod1, target_token_mask)
        attn_output = self.attn(
            hidden_states=img_modulated,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            layer_cache=layer_cache,
            kv_cache_mode=kv_cache_mode,
            cache_write_slice=cache_write_slice,
            segments=segments,
            key_valid=key_valid,
        )
        hidden_states = hidden_states + img_gate1.tanh() * attn_output

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), mod2, target_token_mask)
        hidden_states = hidden_states + img_gate2.tanh() * self.img_mlp(img_modulated2)

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class QwenImage21Rope(nn.Module):
    r"""
    3-axis (frame, height, width) rotary embedding over the joint text/image sequence.

    Text tokens advance a shared position on all three axes. Every image block freezes the frame axis at the position
    reached by the preceding text and lays its tokens out on a height/width grid centred on zero, so a block's spatial
    positions do not depend on where it sits in the sequence.
    """

    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

        pos_index = torch.arange(8192)
        neg_index = torch.arange(1024).flip(0) * -1 - 1
        self.freqs = [
            torch.cat([self.rope_params(pos_index, dim, theta), self.rope_params(neg_index, dim, theta)], dim=0)
            for dim in axes_dim
        ]

    def rope_params(self, index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self, img_shapes: List[Tuple[int, int, int]], image_pad_mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        self.freqs = [freq.to(device) for freq in self.freqs]

        frame_index, height_index, width_index = [], [], []
        image_height_index, image_width_index = [], []
        cursor, position = 0, 0
        total_len = image_pad_mask.shape[-1]
        is_image_token = image_pad_mask.tolist()

        for _, height, width in img_shapes:
            block_start = is_image_token.index(True, cursor)
            text_len = block_start - cursor
            frame_index.extend(range(position, position + text_len))
            position += text_len

            cursor = block_start + height * width
            frame_index.extend([position] * (height * width))
            position += max(height, width)

            image_height_index.extend([h for h in range(-(height - height // 2), height // 2) for _ in range(width)])
            image_width_index.extend([w for _ in range(height) for w in range(-(width - width // 2), width // 2)])

        if cursor < total_len:
            frame_index.extend(range(position, position + total_len - cursor))

        frame_index = torch.tensor(frame_index, dtype=torch.long, device=device)
        height_index = frame_index.clone()
        width_index = frame_index.clone()
        height_index[image_pad_mask] = torch.tensor(image_height_index, dtype=torch.long, device=device)
        width_index[image_pad_mask] = torch.tensor(image_width_index, dtype=torch.long, device=device)

        return torch.cat([self.freqs[0][frame_index], self.freqs[1][height_index], self.freqs[2][width_index]], dim=-1)


class QwenImage21Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    The single-stream Transformer used by Qwen-Image 2.1.

    Text and image latents share one sequence: condition-image tokens are substituted into the text stream at the
    positions the vision-language encoder reserved for them, and the target image's tokens are appended. A single
    shared `modulation` projection feeds every block, so blocks hold no modulation parameters of their own.

    Two behaviours distinguish 2.1:

    - **Block-causal attention** — attention follows `(q_idx >= kv_idx) or same_image_block`, so the sequence is causal
      while each image block stays internally bidirectional. This port uses an exact multi-pass SDPA prefill that
      processes each block separately (the upstream flex-attention single-pass path is not used here).
    - `causal_condition` — text and condition-image tokens are modulated from `t = 0` instead of the sampled timestep,
      which also makes their activations timestep-independent and so cacheable across denoising steps.

    Args:
        patch_size (`int`, defaults to `1`): Side length of the latent patch folded into the channel dim. 2.1 consumes
            latents unpatched.
        in_channels (`int`, defaults to `64`): Latent channels of the input.
        out_channels (`int`, *optional*, defaults to `64`): Latent channels of the output. Falls back to `in_channels`.
        num_layers (`int`, defaults to `32`): Number of single-stream blocks.
        attention_head_dim (`int`, defaults to `128`): Channels per attention head.
        num_attention_heads (`int`, defaults to `32`): Number of attention heads.
        context_in_dim (`int`, defaults to `4096`): Channel dim of `encoder_hidden_states`.
        mlp_ratio (`int`, defaults to `3`): Feed-forward expansion factor.
        axes_dims_rope (`tuple[int]`, defaults to `(16, 56, 56)`): Rotary dims for the frame, height and width axes.
        eps (`float`, defaults to `1e-6`): Epsilon for the norm layers.
        causal_condition (`bool`, defaults to `True`): Modulate text and condition-image tokens from `t = 0`. Required
            for KV caching.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImage21TransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["QwenImage21TransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = 64,
        num_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        context_in_dim: int = 4096,
        mlp_ratio: int = 3,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        eps: float = 1e-6,
        causal_condition: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenImage21Rope(theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = QwenImage21TimestepProjEmbeddings(embedding_dim=self.inner_dim)
        self.txt_in = QwenImage21TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels * patch_size * patch_size, self.inner_dim, bias=False)

        # One shared modulation for every block: [mod1.scale, mod1.gate, mod2.scale, mod2.gate].
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = QwenImage21AdaLayerNormContinuous(self.inner_dim, self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        self.gradient_checkpointing = False
        # Ulysses sequence-parallel state, set by `enable_multi_gpus_inference` (1 = single GPU).
        self.sp_world_size = 1
        self.sp_world_rank = 0

    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            self.gradient_checkpointing = kwargs["value"]
        elif "enable" in kwargs:
            self.gradient_checkpointing = kwargs["enable"]
        else:
            raise ValueError("Invalid set gradient checkpointing")

    def enable_multi_gpus_inference(self):
        """Enable Ulysses (head-parallel) sequence-parallel inference.

        Only Ulysses is supported, so ``ring_degree`` must be 1: ring attention rotates KV chunks and cannot
        express 2.1's block-causal mask or its prefix KV cache, whereas Ulysses gathers the full sequence per
        head-subset and leaves both unchanged. ``num_attention_heads`` (32) must be divisible by the
        sequence-parallel world size, i.e. ``ulysses_degree`` in {1, 2, 4, 8, 16, 32}.
        """
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        if self.sp_world_size > 1 and self.config.num_attention_heads % self.sp_world_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.config.num_attention_heads}) must be divisible by the "
                f"sequence-parallel world size ({self.sp_world_size}). Use ulysses_degree in "
                f"{{1, 2, 4, 8, 16, 32}} (ring_degree must stay 1)."
            )
        processor = QwenImage21MultiGPUsAttnProcessor()
        for module in self.modules():
            if isinstance(module, QwenImage21Attention):
                module.set_processor(processor)

    @staticmethod
    def build_token_metadata(
        image_pad_mask: torch.Tensor, img_shapes: List[Tuple[int, int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Label every token of the joint sequence with the image block it belongs to.

        Block boundaries come from the token counts in `img_shapes`, not from runs of `True` in `image_pad_mask`: two
        condition images that happen to sit next to each other with no text between them form one run but must stay
        separate blocks, otherwise they would attend to each other bidirectionally.

        Args:
            image_pad_mask (`torch.Tensor`): `(seq_len,)` bool, `True` at image-token positions.
            img_shapes (`list[tuple[int, int, int]]`): Per-image `(frame, height, width)` in latent tokens, condition
                images first and the target image last.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: `image_ids` `(seq_len,)` with `-1` at text positions and a unique id
            per image block, and `target_token_mask` `(seq_len,)` marking the target image's tokens.
        """
        image_positions = image_pad_mask.nonzero(as_tuple=True)[0]
        block_lengths = [math.prod(shape) for shape in img_shapes]
        if sum(block_lengths) != image_positions.numel():
            raise ValueError(
                f"img_shapes accounts for {sum(block_lengths)} image tokens but image_pad_mask marks "
                f"{image_positions.numel()}."
            )

        image_ids = torch.full_like(image_pad_mask, -1, dtype=torch.long)
        block_ids = torch.repeat_interleave(
            torch.arange(len(block_lengths), device=image_pad_mask.device),
            torch.tensor(block_lengths, device=image_pad_mask.device),
        )
        image_ids[image_positions] = block_ids

        target_token_mask = torch.zeros_like(image_pad_mask)
        target_token_mask[image_positions[-block_lengths[-1]:]] = True
        return image_ids, target_token_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: List[List[Tuple[int, int, int]]],
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        kv_cache: Optional[QwenImage21KVCache] = None,
        kv_cache_mode: Optional[str] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Packed latents, condition images first and the target image last.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, context_in_dim)`):
                Text embeddings from the vision-language encoder.
            timestep (`torch.Tensor`): Current denoising step, scaled to `[0, 1]`.
            img_shapes (`list[list[tuple[int, int, int]]]`): Per-sample list of `(frame, height, width)` in latent
                tokens, condition images first and the target image last. All samples must share a layout.
            img_mask (`torch.Tensor` of shape `(batch_size, vlm_sequence_length)`): `True` at the vision-language
                encoder's image slots, each standing for a `2x2` group of latent tokens.
            encoder_hidden_states_mask (`torch.Tensor`, *optional*): `(batch_size, text_sequence_length)` bool marking
                valid text tokens. Padded positions are excluded from attention.
            kv_cache (`QwenImage21KVCache`, *optional*): Cache container. Pass together with `kv_cache_mode` to enable
                prefix KV caching.
            kv_cache_mode (`str`, *optional*): `"extract"` to prefill the cache (first denoising step), `"cached"` to
                decode from it (later steps). Requires `causal_condition=True`.
            attention_kwargs (`dict`, *optional*): Reserved for parity with the diffusers API; unused in this port.
            return_dict (`bool`, *optional*, defaults to `True`): Whether to return a [`Transformer2DModelOutput`].

        Returns:
            [`Transformer2DModelOutput`] or `tuple`.
        """
        batch_size = hidden_states.shape[0]
        if kv_cache is not None and not self.config.causal_condition:
            raise ValueError(
                "kv_cache requires `causal_condition=True`. The cache is only valid because text and condition-image "
                "tokens modulate from t=0, which makes their activations independent of the denoising step."
            )
        if kv_cache is not None and kv_cache_mode not in ("extract", "cached"):
            raise ValueError(
                f"kv_cache_mode must be 'extract' or 'cached' when kv_cache is provided, got {kv_cache_mode!r}."
            )
        if kv_cache is None and kv_cache_mode is not None:
            raise ValueError(f"kv_cache_mode is {kv_cache_mode!r} but no kv_cache was passed to hold the prefix.")

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # Each vision-language image slot stands for 2x2 latent tokens, so expand those positions four-fold and drop
        # the actual latents into them. Samples share a layout, hence the single row.
        repeats = torch.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)

        target_tokens = math.prod(img_shapes[0][-1])
        joint_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                encoder_hidden_states.new_zeros(batch_size, target_tokens // 4, encoder_hidden_states.shape[2]),
            ],
            dim=1,
        )
        joint_hidden_states = joint_hidden_states.repeat_interleave(repeats, dim=1)
        joint_hidden_states[:, image_pad_mask] = hidden_states

        rotary_emb = self.pos_embed(img_shapes[0], image_pad_mask, device=hidden_states.device)
        image_ids, target_token_mask = self.build_token_metadata(image_pad_mask, img_shapes[0])

        timestep = timestep.to(hidden_states.dtype)
        if self.config.causal_condition:
            # Extra t=0 row; text and condition-image tokens modulate from it. `modulation_mask` selects which row
            # each token reads, and is `None` when every token shares the sampled timestep.
            timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
            modulation_mask = target_token_mask
        else:
            modulation_mask = None
        temb = self.time_text_embed(timestep, hidden_states)
        modulation = self.modulation(temb)

        # Right-padded prompt positions must never be attended to, on any path. Text positions of the joint sequence
        # line up, in order, with the non-image positions of the vision-language sequence — the two are interleaved,
        # so the mask cannot be sliced off as a prefix.
        joint_key_valid = None
        if encoder_hidden_states_mask is not None:
            joint_key_valid = torch.ones(
                batch_size, image_pad_mask.shape[0], dtype=torch.bool, device=hidden_states.device
            )
            text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
            vlm_text_positions = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
            joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[:, vlm_text_positions]

        prefix_len = int((~target_token_mask).sum())

        if kv_cache_mode == "cached":
            # decode: only the target image's queries are recomputed. The block-causal mask degenerates to full
            # attention for target rows (they see the entire prefix + their own block), so only the padding mask is
            # needed.
            joint_hidden_states = joint_hidden_states[:, prefix_len:]
            rotary_emb = rotary_emb[prefix_len:]
            modulation_mask = modulation_mask[prefix_len:]
            attention_mask = None if joint_key_valid is None else joint_key_valid[:, None, None, :]
            cache_write_slice = None
            block_segments, block_key_valid = None, None
        else:
            # prefill: the whole joint sequence. The block-causal structure is handed to the processor as per-segment
            # boundaries.
            attention_mask = None
            block_segments = _qwenimage21_prefix_segments(image_ids, prefix_len)
            cache_write_slice = slice(0, prefix_len) if kv_cache_mode == "extract" else None
            block_key_valid = joint_key_valid

        # Ulysses sequence parallel: pad the active sequence (the whole joint sequence when prefilling, the
        # target tokens when decoding) to a multiple of the SP world size, mask the padding as keys, then keep
        # only this rank's contiguous slice. `block_segments` / `attention_mask` / `block_key_valid` stay
        # full-sequence: the SP attention processor gathers the full sequence (per head-subset) before applying
        # them. `modulation` and `temb` are per-sample and are never split.
        sp_size = self.sp_world_size
        sp_pad_len = 0
        sp_active_len = joint_hidden_states.shape[1]
        if sp_size > 1:
            sp_padded_len = math.ceil(sp_active_len / sp_size) * sp_size
            sp_pad_len = sp_padded_len - sp_active_len
            if sp_pad_len > 0:
                joint_hidden_states = F.pad(joint_hidden_states, (0, 0, 0, sp_pad_len))
                rotary_emb = torch.cat(
                    [rotary_emb, rotary_emb.new_zeros(sp_pad_len, rotary_emb.shape[-1])], dim=0
                )
                if modulation_mask is not None:
                    modulation_mask = torch.cat(
                        [modulation_mask, modulation_mask.new_zeros(sp_pad_len, dtype=torch.bool)], dim=0
                    )
                # Padding tokens must never be attended to: extend the key-validity mask (creating one when the
                # prompt itself had no padding) with `sp_pad_len` False entries at the end of the active sequence.
                if kv_cache_mode == "cached":
                    if joint_key_valid is None:
                        joint_key_valid = torch.ones(
                            batch_size, prefix_len + sp_active_len, dtype=torch.bool,
                            device=joint_hidden_states.device,
                        )
                    joint_key_valid = torch.cat(
                        [joint_key_valid, joint_key_valid.new_zeros(batch_size, sp_pad_len, dtype=torch.bool)],
                        dim=1,
                    )
                    attention_mask = joint_key_valid[:, None, None, :]
                else:
                    if block_key_valid is None:
                        block_key_valid = torch.ones(
                            batch_size, sp_active_len, dtype=torch.bool, device=joint_hidden_states.device
                        )
                    block_key_valid = torch.cat(
                        [block_key_valid, block_key_valid.new_zeros(batch_size, sp_pad_len, dtype=torch.bool)],
                        dim=1,
                    )
            sp_local = sp_padded_len // sp_size
            sp_lo = self.sp_world_rank * sp_local
            joint_hidden_states = joint_hidden_states[:, sp_lo:sp_lo + sp_local]
            rotary_emb = rotary_emb[sp_lo:sp_lo + sp_local]
            if modulation_mask is not None:
                modulation_mask = modulation_mask[sp_lo:sp_lo + sp_local]

        for index_block, block in enumerate(self.transformer_blocks):
            layer_cache = kv_cache.get_layer(index_block) if kv_cache is not None else None
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # diffusers 0.32.2 has no `ModelMixin._gradient_checkpointing_func` (that is a newer API the upstream
                # 2.1 file assumes), so call `torch.utils.checkpoint` directly like the sibling 2.0 / z_image models.
                # The positional order below matches `QwenImage21TransformerBlock.forward`; `use_reentrant=False` is
                # required because several args (layer_cache / kv_cache_mode / cache_write_slice) are non-tensors.
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                joint_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    joint_hidden_states,
                    modulation,
                    rotary_emb,
                    attention_mask,
                    modulation_mask,
                    layer_cache,
                    kv_cache_mode,
                    cache_write_slice,
                    block_segments,
                    block_key_valid,
                    use_reentrant=False,
                )
            else:
                joint_hidden_states = block(
                    hidden_states=joint_hidden_states,
                    modulation=modulation,
                    rotary_emb=rotary_emb,
                    attention_mask=attention_mask,
                    target_token_mask=modulation_mask,
                    layer_cache=layer_cache,
                    kv_cache_mode=kv_cache_mode,
                    cache_write_slice=cache_write_slice,
                    segments=block_segments,
                    key_valid=block_key_valid,
                )

        joint_hidden_states = self.norm_out(joint_hidden_states, temb, modulation_mask)
        output = self.proj_out(joint_hidden_states)
        if sp_size > 1:
            # Gather the per-rank sequence slices back into the full (padded) sequence, then drop the padding so
            # the output matches the single-GPU shape (whole joint sequence when prefilling, target when decoding).
            output = sequence_parallel_all_gather(output, dim=1)
            if sp_pad_len > 0:
                output = output[:, :sp_active_len]

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs=None,
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        transformer_additional_kwargs = {} if transformer_additional_kwargs is None else dict(transformer_additional_kwargs)
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded QwenImage21 transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            dict_mapping = transformer_additional_kwargs.pop("dict_mapping")
            for key in dict_mapping:
                transformer_additional_kwargs[dict_mapping[key]] = config[key]

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                from packaging import version as pkg_version
                if pkg_version.parse(diffusers_version) >= pkg_version.parse("0.33.0"):
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
                    state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
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
                if len(state_dict) == 0:
                    raise FileNotFoundError(f"No weights found in {pretrained_model_path}")

                model_state_dict = model.state_dict()
                filtered_state_dict = {}
                for key in state_dict:
                    if key in model_state_dict and model_state_dict[key].size() == state_dict[key].size():
                        filtered_state_dict[key] = state_dict[key]
                    else:
                        print(f"Skipping key '{key}' due to size mismatch or absence in model.")

                model_keys = set(model_state_dict.keys())
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
                            else:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)

                    return initialized_dict

                if missing_keys:
                    print(f"Missing keys will be initialized: {sorted(missing_keys)}")
                    initialized_params = initialize_missing_parameters(
                        missing_keys,
                        model_state_dict,
                        torch_dtype
                    )
                    filtered_state_dict.update(initialized_params)

                if pkg_version.parse(diffusers_version) >= pkg_version.parse("0.33.0"):
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
                import traceback
                traceback.print_exc()
                print(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )

        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
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
        if len(state_dict) == 0:
            raise FileNotFoundError(f"No weights found in {pretrained_model_path}")

        model_state_dict = model.state_dict()
        tmp_state_dict = {}
        for key in state_dict:
            if key in model_state_dict.keys() and model_state_dict[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")

        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)

        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        model = model.to(torch_dtype)
        return model
