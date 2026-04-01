# Modified from https://github.com/MeiGen-AI/InfiniteTalk/blob/main/wan/modules/multitalk_model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import types
from functools import lru_cache
from typing import Any, Dict

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import is_torch_version
from einops import rearrange, repeat

from ..dist import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    usp_attn_infinitetalk_forward)
from .attention_utils import attention
from .wan_transformer3d import (WanI2VCrossAttention, WanLayerNorm, WanRMSNorm,
                                WanTransformer3DModel, rope_apply, rope_params,
                                sinusoidal_embedding_1d)


def normalize_and_scale(column, source_range, target_range, epsilon=1e-8):
    source_min, source_max = source_range
    new_min, new_max = target_range
    normalized = (column - source_min) / (source_max - source_min + epsilon)
    scaled = normalized * (new_max - new_min) + new_min
    return scaled


def split_token_counts_and_frame_ids(T, token_frame, world_size, rank):
    S = T * token_frame
    # compute split sizes per rank
    base = S // world_size
    rem = S % world_size
    split_sizes = torch.full((world_size,), base, dtype=torch.long)
    split_sizes[:rem] += 1

    start = split_sizes[:rank].sum()
    end = start + split_sizes[rank]

    # vectorized mapping: global index -> frame id
    idx = torch.arange(start, end, dtype=torch.long)
    frame_ids = idx // token_frame.to(idx.device)

    # unique counts
    unique_frames, counts = torch.unique(frame_ids, return_counts=True)

    # return as Python list (optional)
    return counts.tolist(), unique_frames.tolist()


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@torch.compile
def calculate_x_ref_attn_map(visual_q, ref_k, ref_target_masks, mode='mean', attn_bias=None):
    ref_k = ref_k.to(visual_q.dtype).to(visual_q.device)
    scale = 1.0 / visual_q.shape[-1] ** 0.5
    visual_q = visual_q * scale
    visual_q = visual_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    attn = visual_q @ ref_k.transpose(-2, -1)
    if attn_bias is not None:
        attn = attn + attn_bias
    x_ref_attn_map_source = attn.softmax(-1)
    x_ref_attn_maps = []
    ref_target_masks = ref_target_masks.to(visual_q.dtype)
    x_ref_attn_map_source = x_ref_attn_map_source.to(visual_q.dtype)
    for class_idx, ref_target_mask in enumerate(ref_target_masks):
        ref_target_mask = ref_target_mask[None, None, None, ...]
        x_ref_attnmap = x_ref_attn_map_source * ref_target_mask
        x_ref_attnmap = x_ref_attnmap.sum(-1) / ref_target_mask.sum()
        x_ref_attnmap = x_ref_attnmap.permute(0, 2, 1)
        if mode == 'mean':
            x_ref_attnmap = x_ref_attnmap.mean(-1)
        elif mode == 'max':
            x_ref_attnmap = x_ref_attnmap.max(-1)
        x_ref_attn_maps.append(x_ref_attnmap)
    del attn
    del x_ref_attn_map_source
    return torch.concat(x_ref_attn_maps, dim=0)


def get_attn_map_with_target(visual_q, ref_k, shape, ref_target_masks=None, split_num=2, enable_sp=False):
    N_t, N_h, N_w = shape
    x_seqlens = N_h * N_w
    ref_k = ref_k[:, :x_seqlens]
    _, seq_lens, heads, _ = visual_q.shape
    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = torch.zeros(class_num, seq_lens).to(visual_q.device).to(visual_q.dtype)
    split_chunk = heads // split_num
    for i in range(split_num):
        x_ref_attn_maps_perhead = calculate_x_ref_attn_map(
            visual_q[:, :, i*split_chunk:(i+1)*split_chunk, :],
            ref_k[:, :, i*split_chunk:(i+1)*split_chunk, :],
            ref_target_masks
        )
        x_ref_attn_maps += x_ref_attn_maps_perhead
    return x_ref_attn_maps / split_num


class RotaryPositionalEmbedding1D(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.base = 10000

    @lru_cache(maxsize=32)
    def precompute_freqs_cis_1d(self, pos_indices):
        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim))
        freqs = freqs.to(pos_indices.device)
        freqs = torch.einsum("..., f -> ... f", pos_indices.float(), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def forward(self, x, pos_indices):
        freqs_cis = self.precompute_freqs_cis_1d(pos_indices)
        x_ = x.float()
        freqs_cis = freqs_cis.float().to(x.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, 'n d -> 1 1 n d'), rearrange(sin, 'n d -> 1 1 n d')
        x_ = (x_ * cos) + (rotate_half(x_) * sin)
        return x_.type_as(x)


class SingleStreamAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer: nn.Module,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qk_norm = qk_norm

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)
        self.add_q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.add_k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor, shape=None, enable_sp=False, kv_seq=None) -> torch.Tensor:
        N_t, N_h, N_w = shape
        if not enable_sp:
            x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)
        B, N, C = x.shape
        q = self.q_linear(x)
        q_shape = (B, N, self.num_heads, self.head_dim)
        q = q.view(q_shape).permute((0, 2, 1, 3))
        if self.qk_norm:
            q = self.q_norm(q)
        _, N_a, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv_shape = (B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 3, 1, 4))
        encoder_k, encoder_v = encoder_kv.unbind(0)
        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)
        q = rearrange(q, "B H M K -> B M H K")
        encoder_k = rearrange(encoder_k, "B H M K -> B M H K")
        encoder_v = rearrange(encoder_v, "B H M K -> B M H K")
        # Use attention from attention_utils
        x = attention(q=q, k=encoder_k, v=encoder_v, attention_type="FLASH_ATTENTION")
        x = rearrange(x, "B M H K -> B H M K")
        x_output_shape = (B, N, C)
        x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        if not enable_sp:
            x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)
        return x


class SingleStreamMutiAttention(SingleStreamAttention):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer: nn.Module,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
        class_range: int = 24,
        class_interval: int = 4,
    ) -> None:
        super().__init__(
            dim=dim,
            encoder_hidden_states_dim=encoder_hidden_states_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            eps=eps,
        )
        self.class_interval = class_interval
        self.class_range = class_range
        self.rope_h1 = (0, self.class_interval)
        self.rope_h2 = (self.class_range - self.class_interval, self.class_range)
        self.rope_bak = int(self.class_range // 2)
        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor, shape=None, x_ref_attn_map=None, human_num=None) -> torch.Tensor:
        encoder_hidden_states = encoder_hidden_states.squeeze(0)
        if human_num == 1:
            return super().forward(x, encoder_hidden_states, shape)
        N_t, _, _ = shape
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)
        B, N, C = x.shape
        q = self.q_linear(x)
        q_shape = (B, N, self.num_heads, self.head_dim)
        q = q.view(q_shape).permute((0, 2, 1, 3))
        if self.qk_norm:
            q = self.q_norm(q)
        max_values = x_ref_attn_map.max(1).values[:, None, None]
        min_values = x_ref_attn_map.min(1).values[:, None, None]
        max_min_values = torch.cat([max_values, min_values], dim=2)
        human1_max_value, human1_min_value = max_min_values[0, :, 0].max(), max_min_values[0, :, 1].min()
        human2_max_value, human2_min_value = max_min_values[1, :, 0].max(), max_min_values[1, :, 1].min()
        human1 = normalize_and_scale(x_ref_attn_map[0], (human1_min_value, human1_max_value), (self.rope_h1[0], self.rope_h1[1]))
        human2 = normalize_and_scale(x_ref_attn_map[1], (human2_min_value, human2_max_value), (self.rope_h2[0], self.rope_h2[1]))
        back = torch.full((x_ref_attn_map.size(1),), self.rope_bak, dtype=human1.dtype).to(human1.device)
        max_indices = x_ref_attn_map.argmax(dim=0)
        normalized_map = torch.stack([human1, human2, back], dim=1)
        normalized_pos = normalized_map[range(x_ref_attn_map.size(1)), max_indices]
        q = rearrange(q, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        q = self.rope_1d(q, normalized_pos)
        q = rearrange(q, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)
        _, N_a, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv_shape = (B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 3, 1, 4))
        encoder_k, encoder_v = encoder_kv.unbind(0)
        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)
        per_frame = torch.zeros(N_a, dtype=encoder_k.dtype).to(encoder_k.device)
        per_frame[:per_frame.size(0)//2] = (self.rope_h1[0] + self.rope_h1[1]) / 2
        per_frame[per_frame.size(0)//2:] = (self.rope_h2[0] + self.rope_h2[1]) / 2
        encoder_pos = torch.concat([per_frame]*N_t, dim=0)
        encoder_k = rearrange(encoder_k, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        encoder_k = self.rope_1d(encoder_k, encoder_pos)
        encoder_k = rearrange(encoder_k, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)
        q = rearrange(q, "B H M K -> B M H K")
        encoder_k = rearrange(encoder_k, "B H M K -> B M H K")
        encoder_v = rearrange(encoder_v, "B H M K -> B M H K")
        # Use attention from attention_utils
        x = attention(q=q, k=encoder_k, v=encoder_v, attention_type="FLASH_ATTENTION")
        x = rearrange(x, "B M H K -> B H M K")
        x_output_shape = (B, N, C)
        x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)
        return x


class WanSelfAttention(nn.Module):

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

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def _get_attn_map_with_target(self, visual_q, ref_k, shape, ref_target_masks=None, split_num=2):
        """Helper method for computing attention map, used by xfuser."""
        return get_attn_map_with_target(visual_q, ref_k, shape, ref_target_masks, split_num)

    def forward(self, x, seq_lens, grid_sizes, freqs, ref_target_masks=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        q, k, v = qkv_fn(x)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        x = attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size
        ).type_as(x)

        # output
        x = x.flatten(2)
        x = self.o(x)
        with torch.no_grad():
            x_ref_attn_map = get_attn_map_with_target(q.type_as(x), k.type_as(x), grid_sizes[0], 
                                                    ref_target_masks=ref_target_masks)

        return x, x_ref_attn_map


class WanAttentionBlock(nn.Module):
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 output_dim=768,
                 context_tokens=32,
                 norm_input_visual=True,
                 class_range=24,
                 class_interval=4):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Multi-GPU sequence parallel attributes
        self.sp_world_size = 1
        self.sp_world_rank = 0
        self.all_gather = None

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanI2VCrossAttention(dim,
                                                num_heads,
                                                (-1, -1),
                                                qk_norm,
                                                eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # init audio module
        self.audio_cross_attn = SingleStreamMutiAttention(
            dim=dim,
            encoder_hidden_states_dim=output_dim, 
            num_heads=num_heads,
            qk_norm=False,
            qkv_bias=True,
            eps=eps,
            norm_layer=WanRMSNorm,
            class_range=class_range,
            class_interval=class_interval
        )
        self.norm_x = WanLayerNorm(dim, eps, elementwise_affine=True)  if norm_input_visual else nn.Identity()
        

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        audio_embedding=None,
        ref_target_masks=None,
        human_num=None,
    ):
        dtype = x.dtype
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation.to(e.device) + e).chunk(6, dim=1)

        # self-attention
        y, x_ref_attn_map = self.self_attn(
            (self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x), seq_lens, grid_sizes,
            freqs, ref_target_masks=ref_target_masks)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]
        
        x = x.to(dtype)

        # cross-attention of text
        x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype=dtype)

        # cross attn of audio
        # For multi-GPU: audio_cross_attn expects full sequence, so all_gather x first
        if self.sp_world_size > 1 and self.all_gather is not None:
            # All gather x to get full sequence for audio cross attention
            x_full = self.all_gather(x, dim=1)
            x_a_full = self.audio_cross_attn(self.norm_x(x_full), encoder_hidden_states=audio_embedding,
                                             shape=grid_sizes[0], x_ref_attn_map=x_ref_attn_map, human_num=human_num)
            # Chunk result back to local rank
            x_a = torch.chunk(x_a_full, self.sp_world_size, dim=1)[self.sp_world_rank]
        else:
            x_a = self.audio_cross_attn(self.norm_x(x), encoder_hidden_states=audio_embedding,
                                            shape=grid_sizes[0], x_ref_attn_map=x_ref_attn_map, human_num=human_num)
        x = x + x_a

        y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).to(dtype))
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]

        x = x.to(dtype)
        return x


class AudioProjModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        seq_len=5,
        seq_len_vf=12,
        blocks=12,  
        channels=768, 
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        norm_output_audio=False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels  
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.reshape(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.reshape(batch_size_vf, window_size_vf * blocks_vf * channels_vf)

        # first projection
        audio_embeds = torch.relu(self.proj1(audio_embeds)) 
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf)) 
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1) 
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c*N_t, C_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

        context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c*N_t, self.context_tokens, self.output_dim)

        # normalization and reshape
        with amp.autocast(dtype=torch.float32):
            context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens


class InfiniteTalkTransformer3DModel(WanTransformer3DModel):
    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(
        self,
        model_type='i2v',
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
        # audio params
        audio_window=5,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        vae_scale=4, # vae timedownsample scale

        norm_input_visual=True,
        norm_output_audio=True,
        weight_init=True,
        # custom block params
        class_range=24,
        class_interval=4,
    ):
        # Call parent class initialization
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
            cross_attn_type='i2v_cross_attn',
        )

        assert model_type == 'i2v', 'MultiTalk model requires your model_type is i2v.'

        # Audio-specific parameters
        self.norm_output_audio = norm_output_audio
        self.audio_window = audio_window
        self.intermediate_dim = intermediate_dim
        self.vae_scale = vae_scale
        
        # Replace blocks with audio-enhanced version
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type='i2v_cross_attn', dim=dim, ffn_dim=ffn_dim, 
                              num_heads=num_heads, window_size=window_size, qk_norm=qk_norm, 
                              cross_attn_norm=cross_attn_norm, eps=eps, 
                              output_dim=output_dim, context_tokens=context_tokens,
                              norm_input_visual=norm_input_visual,
                              class_range=class_range, class_interval=class_interval)
            for _ in range(num_layers)
        ])
        
        # Initialize audio adapter
        # blocks should match the number of hidden layers from audio encoder
        # Wav2Vec2 base has 12 transformer layers, so blocks=12
        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=audio_window+vae_scale-1,
            blocks=12,  # Match Wav2Vec2 hidden layers
            channels=768,
            intermediate_dim=intermediate_dim,
            output_dim=output_dim,
            context_tokens=context_tokens,
            norm_output_audio=norm_output_audio,
        )

        # Initialize weights
        if weight_init:
            self.init_weights()

    def enable_multi_gpus_inference(self):
        """Enable multi-GPU inference using sequence parallel."""
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather

        # Replace self_attn forward with xfuser version for all blocks
        # and pass sp parameters to each block for audio_cross_attn
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_infinitetalk_forward, block.self_attn)
            # Pass sp parameters to block for audio_cross_attn multi-GPU support
            block.sp_world_size = self.sp_world_size
            block.sp_world_rank = self.sp_world_rank
            block.all_gather = self.all_gather

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        audio=None,
        ref_target_masks=None,
        cond_flag=True,
    ):
        assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)
        x[0] = x[0].to(context[0].dtype)

        # Get size
        _, T, H, W = x[0].shape
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        # Align seq_len for multi-GPU
        if hasattr(self, 'sp_world_size') and self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            # e0 = e0.to(dtype)
            # e = e.to(dtype)

        # text embedding
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # clip embedding
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea) 
            context = torch.concat([context_clip, context], dim=1).to(x.dtype)
        
        audio_cond = audio.to(device=x.device, dtype=x.dtype)
        first_frame_audio_emb_s = audio_cond[:, :1, ...] 
        latter_frame_audio_emb = audio_cond[:, 1:, ...] 
        latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale) 
        middle_index = self.audio_window // 2
        latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...] 
        latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
        latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...] 
        latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
        latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...] 
        latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
        latter_frame_audio_emb_s = torch.concat([latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2) 
        audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s) 
        human_num = len(audio_embedding)
        audio_embedding = torch.concat(audio_embedding.split(1), dim=2).to(x.dtype)

        # convert ref_target_masks to token_ref_target_masks
        if ref_target_masks is not None:
            ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32) 
            token_ref_target_masks = nn.functional.interpolate(ref_target_masks, size=(N_h, N_w), mode='nearest') 
            token_ref_target_masks = token_ref_target_masks.squeeze(0)
            token_ref_target_masks = (token_ref_target_masks > 0)
            token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1) 
            token_ref_target_masks = token_ref_target_masks.to(x.dtype)

        # Context Parallel
        if hasattr(self, 'sp_world_size') and self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]

        # TeaCache
        if self.teacache is not None:
            if cond_flag:
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

        # Arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            audio_embedding=audio_embedding,
            ref_target_masks=token_ref_target_masks,
            human_num=human_num,
        )
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()
                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)
                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, e0, seq_lens, grid_sizes, self.freqs,
                            context, context_lens, audio_embedding,
                            token_ref_target_masks, human_num,
                            **ckpt_kwargs,
                        )
                    else:
                        x = block(x, **kwargs)
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, e0, seq_lens, grid_sizes, self.freqs,
                        context, context_lens, audio_embedding,
                        token_ref_target_masks, human_num,
                        **ckpt_kwargs,
                    )
                else:
                    x = block(x, **kwargs)

        # head
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x, e, **ckpt_kwargs)
        else:
            x = self.head(x, e)

        # Context Parallel all_gather
        if hasattr(self, 'sp_world_size') and self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x