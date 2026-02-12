# Modified from https://github.com/meituan-longcat/LongCat-Video/blob/main/longcat_video/modules/avatar/longcat_video_dit_avatar.py
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version
from einops import rearrange, repeat

from .attention_utils import attention
from .longcatvideo_transformer3d import (CaptionEmbedder, FeedForwardSwiGLU,
                                         FinalLayer_FP32, LayerNorm_FP32,
                                         MultiHeadCrossAttention, PatchEmbed3D,
                                         RMSNorm_FP32,
                                         RotaryPositionalEmbedding,
                                         TimestepEmbedder, broadcat,
                                         modulate_fp32, rotate_half)


def normalize_and_scale(column, source_range, target_range, epsilon=1e-8):
    source_min, source_max = source_range
    new_min, new_max = target_range 
    normalized = (column - source_min) / (source_max - source_min + epsilon)
    scaled = normalized * (new_max - new_min) + new_min
    return scaled


class RotaryPositionalEmbedding1D(nn.Module):

    def __init__(self,
                 head_dim
                 ):
        """Rotary positional embedding for 1D
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
        """
        super().__init__()
        self.head_dim = head_dim
        self.base = 10000

    def precompute_freqs_cis_1d(self, pos_indices):

        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim))

        freqs = freqs.to(pos_indices.device)
        freqs = torch.einsum("..., f -> ... f", pos_indices.float(), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        return freqs

    def forward(self, x, pos_indices):
        """1D RoPE.

        Args:
            query (torch.tensor): [B, head, seq, head_dim]
            pos_indices (torch.tensor): [seq,]
        Returns:
            query with the same shape as input.
        """
        freqs_cis = self.precompute_freqs_cis_1d(pos_indices)

        x_ = x.float()

        freqs_cis = freqs_cis.float().to(x.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, 'n d -> 1 1 n d'), rearrange(sin, 'n d -> 1 1 n d')
        x_ = (x_ * cos) + (rotate_half(x_) * sin)

        return x_.type_as(x)


# https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/rotary_positional_embedding.py
class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding for 3D spatial-temporal data
    Reference: https://blog.eleuther.ai/rotary-embeddings/
    Paper: https://arxiv.org/pdf/2104.09864.pdf
    """
    def __init__(
        self,
        head_dim,
        cp_split_hw=None
    ):
        """
        Args:
            head_dim: Dimension of each attention head (must be divisible by 8 for 3D RoPE)
            cp_split_hw: Context parallel split for height and width dimensions
        """
        super().__init__()
        self.head_dim = head_dim
        assert self.head_dim % 8 == 0, 'Dim must be a multiply of 8 for 3D RoPE.'
        self.cp_split_hw = cp_split_hw
        # We take the assumption that the longest side of grid will not larger than 512, i.e, 512 * 8 = 4098 input pixels
        self.base = 10000
        self.freqs_dict = {}

    def register_grid_size(self, grid_size, key_name, frame_index=None, num_ref_latents=None):
        
        if key_name not in self.freqs_dict:
            self.freqs_dict.update({
                key_name: self.precompute_freqs_cis_3d(grid_size, frame_index, num_ref_latents)
            })

    def precompute_freqs_cis_3d(self, grid_size, frame_index=None, num_ref_latents=None):
        """
        Precompute frequency embeddings for 3D grid (time, height, width)
        
        Args:
            grid_size: Tuple of (num_frames, height, width)
            frame_index: Optional reference frame index for video continuation
            num_ref_latents: Optional number of reference latents
            
        Returns:
            freqs: Precomputed frequency tensor of shape (T*H*W, D)
        """
        num_frames, height, width = grid_size     
        dim_t = self.head_dim - 4 * (self.head_dim // 6)
        dim_h = 2 * (self.head_dim // 6)
        dim_w = 2 * (self.head_dim // 6)
        freqs_t = 1.0 / (self.base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (self.base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (self.base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
        if frame_index is not None and num_ref_latents is not None:
            grid_t = torch.concat([torch.tensor([frame_index], dtype=torch.float32), torch.arange(0, num_frames-num_ref_latents, dtype=torch.float32)], dim=0)
        else:
            grid_t = np.linspace(0, num_frames, num_frames, endpoint=False, dtype=np.float32)
            grid_t = torch.from_numpy(grid_t).float()
        grid_h = np.linspace(0, height, height, endpoint=False, dtype=np.float32)
        grid_w = np.linspace(0, width, width, endpoint=False, dtype=np.float32)
        grid_h = torch.from_numpy(grid_h).float()
        grid_w = torch.from_numpy(grid_w).float()
        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
        # (T H W D)
        freqs = rearrange(freqs, "T H W D -> (T H W) D")

        return freqs

    def forward(self, q, k, grid_size, frame_index=None, num_ref_latents=None):
        """
        Apply 3D RoPE to query and key tensors
        
        Args:
            q: Query tensor [B, head, seq, head_dim]
            k: Key tensor [B, head, seq, head_dim]
            grid_size: Tuple of (num_frames, height, width)
            frame_index: Optional reference frame index
            num_ref_latents: Optional number of reference latents
            
        Returns:
            Tuple of (q, k) with rotary positional embeddings applied
        """
        key_name = '.'.join([str(i) for i in grid_size]) + f"-{str(frame_index)}-{str(num_ref_latents)}"
        if key_name not in self.freqs_dict:
            self.register_grid_size(grid_size, key_name, frame_index, num_ref_latents)

        freqs_cis = self.freqs_dict[key_name].to(q.device)
        q_, k_ = q.float(), k.float()
        freqs_cis = freqs_cis.float().to(q.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, 'n d -> 1 1 n d'), rearrange(sin, 'n d -> 1 1 n d')
        q_ = (q_ * cos) + (rotate_half(q_) * sin)
        k_ = (k_ * cos) + (rotate_half(k_) * sin)

        return q_.type_as(q), k_.type_as(k)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw: Optional[List[int]] = None
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers
        self.enable_bsa = enable_bsa
        self.bsa_params = bsa_params
        self.cp_split_hw = cp_split_hw

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

        self.rope_3d = RotaryPositionalEmbedding(
            self.head_dim,
            cp_split_hw=cp_split_hw
        )

    def _process_attn(self, q, k, v, shape):
        q = rearrange(q, "B H S D -> B S H D")
        k = rearrange(k, "B H S D -> B S H D")
        v = rearrange(v, "B H S D -> B S H D")
        x = attention(q, k, v)
        x = rearrange(x, "B S H D -> B H S D")
        return x

    def forward(self, x: torch.Tensor, shape=None, num_cond_latents=None, return_kv=False, num_ref_latents=None, ref_img_index=None, mask_frame_range=None, ref_target_masks=None) -> torch.Tensor:
        """
        """
        B, N, C = x.shape
        qkv = self.qkv(x)

        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape).permute((2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if return_kv:
            k_cache, v_cache = k.clone(), v.clone()

        q, k = self.rope_3d(q, k, shape, ref_img_index, num_ref_latents)

        N_t, N_h, N_w = shape
        # cond mode
        if num_cond_latents is not None and num_cond_latents == 1:
            # image to video
            num_cond_latents_thw = num_cond_latents * (N // N_t)
            # process the condition tokens
            q_cond = q[:, :, :num_cond_latents_thw].contiguous()
            k_cond = k[:, :, :num_cond_latents_thw].contiguous()
            v_cond = v[:, :, :num_cond_latents_thw].contiguous()
            x_cond = self._process_attn(q_cond, k_cond, v_cond, shape)
            # process the noise tokens
            q_noise = q[:, :, num_cond_latents_thw:].contiguous()
            x_noise = self._process_attn(q_noise, k, v, shape)
            # merge x_cond and x_noise
            x = torch.cat([x_cond, x_noise], dim=2).contiguous()
        elif num_cond_latents is not None and num_cond_latents > 1:
            # video continuation
            assert num_ref_latents is not None and ref_img_index is not None, f"No specified insertion position for reference frame"
            num_ref_latents_thw = (N // N_t)
            num_cond_latents_thw = num_cond_latents * (N // N_t)
            # process the condition tokens
            q_ref = q[:, :, :num_ref_latents_thw].contiguous()
            k_ref = k[:, :, :num_ref_latents_thw].contiguous()
            v_ref = v[:, :, :num_ref_latents_thw].contiguous()
            q_cond = q[:, :, num_ref_latents_thw:num_cond_latents_thw].contiguous()
            k_cond = k[:, :, num_ref_latents_thw:num_cond_latents_thw].contiguous()
            v_cond = v[:, :, num_ref_latents_thw:num_cond_latents_thw].contiguous()
            x_ref = self._process_attn(q_ref, k_ref, v_ref, shape)
            x_cond = self._process_attn(q_cond, k_cond, v_cond, shape)
            if num_cond_latents == N_t:
                x = torch.cat([x_ref, x_cond], dim=2).contiguous()
            else:
                # process the noise tokens
                q_noise = q[:, :, num_cond_latents_thw:].contiguous()
                
                start_noise, end_noise, num_noisy_frames = 0, 0, N_t - num_cond_latents
                if mask_frame_range is not None and mask_frame_range > 0:
                    start_noise = ref_img_index - mask_frame_range - num_cond_latents + num_ref_latents
                    end_noise   = ref_img_index + mask_frame_range - num_cond_latents + num_ref_latents + 1

                if start_noise >= 0 and end_noise > start_noise and end_noise <= num_noisy_frames:
                    # remove attention with the reference image in the target range, preventing repeated actions.
                    _enable_bsa = self.enable_bsa
                    self.enable_bsa = False # close bsa to prevent the temporal dimension from being divisible by bsa chunks
                    
                    start_pos = start_noise * (N // N_t)
                    end_pos   = end_noise * (N // N_t)
                    q_noise_front = q_noise[:, :, :start_pos].contiguous()
                    q_noise_maskref = q_noise[:, :, start_pos:end_pos].contiguous()
                    q_noise_back = q_noise[:, :, end_pos:].contiguous()
                    k_non_ref = k[:, :, num_ref_latents_thw:].contiguous()
                    v_non_ref = v[:, :, num_ref_latents_thw:].contiguous()
                    x_noise_front = self._process_attn(q_noise_front, k, v, shape) # q_front has attention with ref + cond + noisy
                    x_noise_back = self._process_attn(q_noise_back, k, v, shape) # q_back has attention with ref + cond + noisy
                    x_noise_maskref = self._process_attn(q_noise_maskref, k_non_ref, v_non_ref, shape) # q_mask has attention with cond+noisy
                    x_noise = torch.cat([x_noise_front, x_noise_maskref, x_noise_back], dim=2).contiguous()
                    self.enable_bsa = _enable_bsa # recover bsa state
                else:
                    x_noise = self._process_attn(q_noise, k, v, shape)
                # merge x_cond and x_noise
                x = torch.cat([x_ref, x_cond, x_noise], dim=2).contiguous()

        else:
            # text to video
            x = self._process_attn(q, k, v, shape)

        x_output_shape = (B, N, C)
        x = x.transpose(1, 2) 
        x = x.reshape(x_output_shape)
        x = self.proj(x)

        # calculate attention mask for the given area in reference image
        x_ref_attn_map = None
        if ref_target_masks is not None:
            assert num_cond_latents is not None and num_cond_latents > 0, f"currently, multitalk only supports image to video or video continuation"
            x_ref_attn_map = get_attn_map_with_target(q.permute(0, 2, 1, 3)[:, num_cond_latents_thw:].type_as(x), k.permute(0, 2, 1, 3).type_as(x), shape, ref_target_masks=ref_target_masks, cp_split_hw=self.cp_split_hw)

        if return_kv:
            return x, (k_cache, v_cache), x_ref_attn_map
        else:
            return x, x_ref_attn_map

    def forward_with_kv_cache(self, x: torch.Tensor, shape=None, num_cond_latents=None, kv_cache=None, num_ref_latents=None, ref_img_index=None, mask_frame_range=None, ref_target_masks=None) -> torch.Tensor:
        """
        """
        B, N, C = x.shape
        qkv = self.qkv(x)
        
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape).permute((2, 0, 3, 1, 4)) 
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        N_t, N_h, N_w = shape
        k_cache, v_cache = kv_cache
        assert k_cache.shape[0] == v_cache.shape[0] and k_cache.shape[0] in [1, B]
        if k_cache.shape[0] == 1:
            k_cache = k_cache.repeat(B, 1, 1, 1)
            v_cache = v_cache.repeat(B, 1, 1, 1)
        
        if num_cond_latents is not None and num_cond_latents > 0:
            k_full = torch.cat([k_cache, k], dim=2).contiguous()
            v_full = torch.cat([v_cache, v], dim=2).contiguous()
            q_padding = torch.cat([torch.empty_like(k_cache), q], dim=2).contiguous()
            q_padding, k_full = self.rope_3d(q_padding, k_full, (N_t + num_cond_latents, N_h, N_w), ref_img_index, num_ref_latents)
            q = q_padding[:, :, -N:].contiguous()
        
        start_noise, end_noise, num_noisy_frames = 0, 0, N_t
        if mask_frame_range is not None and mask_frame_range > 0:
            start_noise = ref_img_index - mask_frame_range - num_cond_latents + num_ref_latents 
            end_noise   = ref_img_index + mask_frame_range - num_cond_latents + num_ref_latents + 1 
        
        if start_noise >= 0 and end_noise > start_noise and end_noise <= num_noisy_frames:
            # remove attention with the reference image in the target range, preventing repeated actions.
            _enable_bsa = self.enable_bsa
            self.enable_bsa = False # close bsa to prevent the temporal dimension from being divisible by bsa chunks
            
            num_ref_latents_thw = (N // N_t)
            start_pos = start_noise * (N // N_t)
            end_pos   = end_noise * (N // N_t)
            q_noise_front = q[:, :, :start_pos].contiguous()
            q_noise_maskref = q[:, :, start_pos:end_pos].contiguous()
            q_noise_back = q[:, :, end_pos:].contiguous()
            k_non_ref = k_full[:, :, num_ref_latents_thw:].contiguous()
            v_non_ref = v_full[:, :, num_ref_latents_thw:].contiguous()
            x_noise_front = self._process_attn(q_noise_front, k_full, v_full, shape) # q_front --> ref+cond+noisy
            x_noise_back = self._process_attn(q_noise_back, k_full, v_full, shape) # q_back --> ref+cond+noisy
            x_noise_maskref = self._process_attn(q_noise_maskref, k_non_ref, v_non_ref, shape) # q_mask --> cond+noisy
            x = torch.cat([x_noise_front, x_noise_maskref, x_noise_back], dim=2).contiguous()
            self.enable_bsa = _enable_bsa # recover bsa state
        else:
            x = self._process_attn(q, k_full, v_full, shape)
        
        x_output_shape = (B, N, C)
        x = x.transpose(1, 2)
        x = x.reshape(x_output_shape) 
        x = self.proj(x)

        # calculate attention mask for the given area in reference image
        x_ref_attn_map = None
        if ref_target_masks is not None:
            assert num_cond_latents is not None and num_cond_latents > 0, f"currently, multitalk only supports image to video or video continuation"
            x_ref_attn_map = get_attn_map_with_target(q.permute(0, 2, 1, 3).type_as(x), k_full.permute(0, 2, 1, 3).type_as(x), shape, ref_target_masks=ref_target_masks, cp_split_hw=self.cp_split_hw)

        return x, x_ref_attn_map


class SingleStreamAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
        class_range: int = 24,
        class_interval: int = 4,
        cp_split_hw: Optional[List[int]] = None,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.cp_split_hw = cp_split_hw
        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=eps) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=eps) if qk_norm else nn.Identity()

        # multitalk related params
        self.class_interval = class_interval
        self.class_range = class_range
        self.rope_h1  = (0, self.class_interval)
        self.rope_h2  = (self.class_range - self.class_interval, self.class_range)
        self.rope_bak = int(self.class_range // 2)
        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)

    def _process_cross_attn(self, x, cond, frames_num=None, x_ref_attn_map=None):

        N_t = frames_num
        out_dtype = x.dtype
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)

        # get q for hidden_state
        B, N, C = x.shape
        q = self.q_linear(x)
        q_shape = (B, N, self.num_heads, self.head_dim)
        q = q.view(q_shape).permute((0, 2, 1, 3)) # [B, H, N, D]
        q = self.q_norm(q)

        # multitalk with rope1d pe
        if x_ref_attn_map is not None:
            max_values = x_ref_attn_map.max(1).values[:, None, None] 
            min_values = x_ref_attn_map.min(1).values[:, None, None] 
            max_min_values = torch.cat([max_values, min_values], dim=2) 
            if self.cp_split_hw is not None and self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
                max_min_values = context_parallel_util.gather_cp(max_min_values, 1)
            human1_max_value, human1_min_value = max_min_values[0, :, 0].max(), max_min_values[0, :, 1].min()
            human2_max_value, human2_min_value = max_min_values[1, :, 0].max(), max_min_values[1, :, 1].min()

            human1 = normalize_and_scale(x_ref_attn_map[0], (human1_min_value, human1_max_value), (self.rope_h1[0], self.rope_h1[1]))
            human2 = normalize_and_scale(x_ref_attn_map[1], (human2_min_value, human2_max_value), (self.rope_h2[0], self.rope_h2[1]))
            back   = torch.full((x_ref_attn_map.size(1),), self.rope_bak, dtype=human1.dtype).to(human1.device)
            max_indices = x_ref_attn_map.argmax(dim=0)
            normalized_map = torch.stack([human1, human2, back], dim=1)
            normalized_pos = normalized_map[range(x_ref_attn_map.size(1)), max_indices] 

            q = rearrange(q, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
            q = self.rope_1d(q, normalized_pos)
            q = rearrange(q, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)
        
        # get kv from encoder_hidden_states
        _, N_a, _ = cond.shape
        encoder_kv = self.kv_linear(cond)
        encoder_kv_shape = (B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 3, 1, 4))

        encoder_k, encoder_v = encoder_kv.unbind(0)
        encoder_k = self.k_norm(encoder_k)

        # multitalk with rope1d pe
        if x_ref_attn_map is not None:
            per_frame = torch.zeros(N_a, dtype=encoder_k.dtype).to(encoder_k.device)
            per_frame[:per_frame.size(0)//2] = (self.rope_h1[0] + self.rope_h1[1]) / 2
            per_frame[per_frame.size(0)//2:] = (self.rope_h2[0] + self.rope_h2[1]) / 2
            encoder_pos = torch.concat([per_frame]*N_t, dim=0)
            encoder_k = rearrange(encoder_k, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
            encoder_k = self.rope_1d(encoder_k, encoder_pos)
            encoder_k = rearrange(encoder_k, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)

        # Input tensors must be in format ``[B, M, H, K]``, where B is the batch size, M \
        # the sequence length, H the number of heads, and K the embeding size per head
        q = rearrange(q, "B H S D -> B S H D")
        encoder_k = rearrange(encoder_k, "B H S D -> B S H D")
        encoder_v = rearrange(encoder_v, "B H S D -> B S H D")
        x = attention(q, encoder_k, encoder_v, attention_type="FLASH_ATTENTION")
        x = rearrange(x, "B S H D -> B H S D")

        # linear transform
        x_output_shape = (B, N, C)
        x = x.transpose(1, 2) 
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)

        # reshape x to origin shape
        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)

        return x.type(out_dtype)

    def forward(self, x, cond, shape=None, num_cond_latents=None, x_ref_attn_map=None, human_num=None):

        B, N, C = x.shape
        if (num_cond_latents is None or num_cond_latents == 0): 
            # text to video
            output = self._process_cross_attn(x, cond, shape[0], x_ref_attn_map)
            return None, output
        elif num_cond_latents is not None and num_cond_latents > 0:
            # image to video or video continuation
            assert shape is not None, "SHOULD pass in the shape"
            num_cond_latents_thw = num_cond_latents * (N // shape[0])
            x_noise = x[:, num_cond_latents_thw:]
            cond = rearrange(cond, "(B N_t) M C -> B N_t M C", B=B)
            cond = cond[:, num_cond_latents:] 
            cond = rearrange(cond, "B N_t M C -> (B N_t) M C")
            frames_num = shape[0] - num_cond_latents
            if human_num is not None and human_num == 2:
                # multitalk mode
                output_noise = self._process_cross_attn(x_noise, cond, frames_num, x_ref_attn_map)
            else:
                # singletalk mode
                output_noise = self._process_cross_attn(x_noise, cond, frames_num)
            output_cond = torch.zeros((B, num_cond_latents_thw, C), dtype=output_noise.dtype, device=output_noise.device)
            return output_cond, output_noise
        else:
            raise NotImplementedError


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
        norm_output_audio=True,
        enable_compile=False,
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
        self.flops = 0.0
        self.enable_compile = enable_compile

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, window_size_vf * blocks_vf * channels_vf)

        # first projection
        B1, _ = audio_embeds.shape
        audio_embeds = torch.relu(self.proj1(audio_embeds)) 
        if not self.enable_compile:
            self.flops += B1 * self.input_dim * self.intermediate_dim * 2

        B1_vf, _ = audio_embeds_vf.shape
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))
        if not self.enable_compile:
            self.flops += B1_vf * self.input_dim_vf * self.intermediate_dim * 2 

        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1) 
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c*N_t, C_a)

        # second projection
        B2, _ = audio_embeds_c.shape
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))
        if not self.enable_compile:
            self.flops += B2 * self.intermediate_dim * self.intermediate_dim * 2 

        # third projection
        B3, _ = audio_embeds_c.shape
        context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c*N_t, self.context_tokens, self.output_dim)
        if not self.enable_compile:
            self.flops += B3 * self.intermediate_dim * (self.context_tokens * self.output_dim) * 2 

        # normalization and reshape
        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens


class LongCatAvatarSingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params=None,
        cp_split_hw=None,
        # avatar config
        output_dim=768,
        audio_prenorm=True,
        class_range=24,
        class_interval=4
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # scale and gate modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True)
        )
        self.audio_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_tembed_dim, 3 * hidden_size, bias=True)
        )

        self.mod_norm_attn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn  = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)

        self.pre_video_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)
        self.pre_audio_crs_attn_norm = LayerNorm_FP32(output_dim, eps=1e-6, elementwise_affine=True) if audio_prenorm else nn.Identity()
        
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
            enable_bsa=enable_bsa,
            bsa_params=bsa_params,
            cp_split_hw=cp_split_hw
        )
        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers
        )

        self.audio_cross_attn = SingleStreamAttention(
                dim=hidden_size,
                encoder_hidden_states_dim=output_dim,
                num_heads=num_heads,
                qk_norm=True,
                qkv_bias=True,
                class_range=class_range,
                class_interval=class_interval,
                cp_split_hw=cp_split_hw,
                enable_flashattn3=enable_flashattn3,
                enable_flashattn2=enable_flashattn2,
                enable_xformers=enable_xformers
            )

        self.ffn = FeedForwardSwiGLU(dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio))

    def forward(
        self, 
        x, 
        y, 
        t, 
        y_seqlen, 
        latent_shape, 
        num_cond_latents=None, 
        return_kv=False, 
        kv_cache=None, 
        skip_crs_attn=False,
        # avatar related params
        num_ref_latents=None,
        audio_hidden_states=None, 
        ref_img_index=None,
        mask_frame_range=None,
        token_ref_target_masks=None,
        human_num=None,
    ):
        """
            x: [B, N, C]
            y: [1, N_valid_tokens, C]
            t: [B, T, C_t]
            y_seqlen: [B]; type of a list
            latent_shape: latent shape of a single item
        """
        x_dtype = x.dtype

        B, N, C = x.shape
        T, _, _ = latent_shape # S != T*H*W in case of CP split on H*W.

        # compute modulation params in fp32
        with amp.autocast(device_type="cuda", dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1) # [B, T, 1, C]

        # self attn with modulation
        x_m = modulate_fp32(self.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa).view(B, N, C)

        if kv_cache is not None:
            kv_cache = (kv_cache[0].to(x.device), kv_cache[1].to(x.device))
            attn_outputs = self.attn.forward_with_kv_cache(x_m, shape=latent_shape, num_cond_latents=num_cond_latents, kv_cache=kv_cache, num_ref_latents=num_ref_latents, \
                                                            ref_img_index=ref_img_index, mask_frame_range=mask_frame_range, ref_target_masks=token_ref_target_masks)
        else:
            attn_outputs = self.attn(x_m, shape=latent_shape, num_cond_latents=num_cond_latents, return_kv=return_kv, num_ref_latents=num_ref_latents, \
                                                            ref_img_index=ref_img_index, mask_frame_range=mask_frame_range, ref_target_masks=token_ref_target_masks)
        
        if return_kv:
            x_s, kv_cache, x_ref_attn_map = attn_outputs
        else:
            x_s, x_ref_attn_map = attn_outputs

        with amp.autocast(device_type="cuda", dtype=torch.float32):
            x = x + (gate_msa * x_s.view(B, -1, N//T, C)).view(B, -1, C) # [B, N, C]
        x = x.to(x_dtype)

        # text cross attn
        if not skip_crs_attn:
            if kv_cache is not None:
                num_cond_latents = None
            x = x + self.cross_attn(self.pre_crs_attn_norm(x), y, y_seqlen, num_cond_latents=num_cond_latents, shape=latent_shape)
        
        # audio cross attn
        if not skip_crs_attn:
            if kv_cache is not None:
                num_cond_latents = 0

            with amp.autocast(device_type="cuda", dtype=torch.float32):  
                audio_shift_mca, audio_scale_mca, audio_gate_mca = \
                        self.audio_adaLN_modulation(t[:, num_cond_latents:]).unsqueeze(2).chunk(3, dim=-1) # [B, T, 1, C]

            audio_output_cond, audio_output_noise = self.audio_cross_attn(self.pre_video_crs_attn_norm(x), self.pre_audio_crs_attn_norm(audio_hidden_states), \
                                                                            shape=latent_shape, num_cond_latents=num_cond_latents, x_ref_attn_map=x_ref_attn_map, human_num=human_num)

            with amp.autocast(device_type="cuda", dtype=torch.float32):  
                audio_output_noise = modulate_fp32(self.mod_norm_attn, audio_output_noise.view(B, T-num_cond_latents, -1, C), audio_shift_mca, audio_scale_mca).view(B, -1, C)
                audio_add_x = (audio_gate_mca * audio_output_noise.view(B, T-num_cond_latents, -1, C)).view(B, -1, C) # [B, N, C]
                if audio_output_cond is not None:
                    audio_add_x = torch.cat([audio_output_cond, audio_add_x], dim=1).contiguous()
            x = x + audio_add_x
            x = x.to(x_dtype)

        # ffn with modulation
        x_m = modulate_fp32(self.mod_norm_ffn, x.view(B, -1, N//T, C), shift_mlp, scale_mlp).view(B, -1, C)
        x_s = self.ffn(x_m)
        with amp.autocast(device_type="cuda", dtype=torch.float32):
            x = x + (gate_mlp * x_s.view(B, -1, N//T, C)).view(B, -1, C) # [B, N, C]
        x = x.to(x_dtype)

        if return_kv:
            return x, kv_cache
        else:
            return x


class LongCatVideoAvatarTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    # _no_split_modules = ['LongCatAvatarSingleStreamBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 4096,
        depth: int = 48,
        num_heads: int = 32,
        caption_channels: int = 4096,
        mlp_ratio: int = 4,
        adaln_tembed_dim: int = 512,
        frequency_embedding_size: int = 256,
        # default params
        patch_size: Tuple[int] = (1, 2, 2),
        # attention config
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw: Optional[List[int]] = None,
        text_tokens_zero_pad: bool = False,
        # avatar config
        audio_window: int = 5,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 32,
        vae_scale: int = 4, 
        audio_prenorm: bool = False,
        class_range: int = 24,
        class_interval: int = 4
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cp_split_hw = cp_split_hw
        self.vae_scale = vae_scale
        self.audio_window = audio_window

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(t_embed_dim=adaln_tembed_dim, frequency_embedding_size=frequency_embedding_size)
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                LongCatAvatarSingleStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    adaln_tembed_dim=adaln_tembed_dim,
                    enable_flashattn3=enable_flashattn3,
                    enable_flashattn2=enable_flashattn2,
                    enable_xformers=enable_xformers,
                    enable_bsa=enable_bsa,
                    bsa_params=bsa_params,
                    cp_split_hw=cp_split_hw,
                    output_dim=output_dim,
                    audio_prenorm=audio_prenorm,
                    class_range=class_range,
                    class_interval=class_interval
                )
                for i in range(depth)
            ]
        )

        self.audio_proj = AudioProjModel(
                    seq_len=audio_window,
                    seq_len_vf=audio_window+vae_scale-1,
                    intermediate_dim=intermediate_dim,
                    output_dim=output_dim,
                    context_tokens=context_tokens
                )

        self.final_layer = FinalLayer_FP32(
            hidden_size,
            np.prod(self.patch_size),
            out_channels,
            adaln_tembed_dim,
        )

        self.gradient_checkpointing = False
        self.text_tokens_zero_pad = text_tokens_zero_pad

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

        def _gradient_checkpointing_func(module, *args):
            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            return torch.utils.checkpoint.checkpoint(
                module.__call__,
                *args,
                **ckpt_kwargs,
            )
        self._gradient_checkpointing_func = _gradient_checkpointing_func

    def enable_bsa(self,):
        for block in self.blocks:
            block.attn.enable_bsa = True
    
    def disable_bsa(self,):
        for block in self.blocks:
            block.attn.enable_bsa = False    

    def forward(
        self, 
        hidden_states, 
        timestep, 
        encoder_hidden_states, 
        encoder_attention_mask=None, 
        num_cond_latents=0,
        return_kv=False, 
        kv_cache_dict={},
        skip_crs_attn=False, 
        offload_kv_cache=False,
        # avatar related params
        audio_embs=None,
        num_ref_latents=None,
        ref_img_index=None, 
        mask_frame_range=None,
        ref_target_masks=None
    ):

        B, _, T, H, W = hidden_states.shape

        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        assert self.patch_size[0]==1, "Currently, 3D x_embedder should not compress the temporal dimension."

        # Expand the shape of timestep from [B] to [B, T]
        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t) # [B, T]
        timestep[:, :num_cond_latents] = 0

        # Hidden_States Process
        hidden_states = self.x_embedder(hidden_states)  # [B, N, C]

        # Timestep Process
        with amp.autocast(device_type="cuda", dtype=torch.float32):
            t = self.t_embedder(timestep.float().flatten(), dtype=torch.float32).reshape(B, N_t, -1)  # [B, T, C_t]

        # Encoder_Hidden_States Process
        encoder_hidden_states = self.y_embedder(encoder_hidden_states)  # [B, 1, N_token, C]

        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[:, None, :, None]
            encoder_attention_mask = (encoder_attention_mask * 0 + 1).to(encoder_attention_mask.dtype)

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            encoder_hidden_states = encoder_hidden_states.squeeze(1).masked_select(encoder_attention_mask.unsqueeze(-1) != 0).view(1, -1, hidden_states.shape[-1]) # [1, N_valid_tokens, C]
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist() # [B]
        else:
            y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.squeeze(1).view(1, -1, hidden_states.shape[-1])

        # Audio token Process
        audio_cond = audio_embs.to(device=hidden_states.device, dtype=hidden_states.dtype)
        first_frame_audio_emb_s = audio_cond[:, :1, ...] # [B, 1, W, S, C_a]

        latter_frame_audio_emb = audio_cond[:, 1:, ...] # [B, T-1, W, S, C_a]
        latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale)
        middle_index = self.audio_window // 2
        latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...]
        latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")

        latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
        latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")

        latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...]
        latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_frame_audio_emb_s = torch.concat([latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2) # [B, (T-1)//vae_scale, W-1+vae_scale, S, C_a]
        audio_hidden_states = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s) # B T 32 768
        
        if num_ref_latents is not None and num_ref_latents > 0:
            audio_start_ref = audio_hidden_states[:, [0], :, :] # padding
            audio_hidden_states = torch.cat([audio_start_ref, audio_hidden_states], dim=1).contiguous()
        audio_hidden_states = audio_hidden_states[:, -N_t:]

        human_num = None
        if ref_target_masks is not None:
            # multitalk
            human_num = len(audio_hidden_states)
            audio_hidden_states = torch.concat(audio_hidden_states.split(1), dim=2) # B T 32 768 --> # 1 T B*32 768
            audio_hidden_states = audio_hidden_states.squeeze(0)
        else:
            audio_hidden_states = rearrange(audio_hidden_states, "b t n c -> (b t) n c")
        
        # Convert ref_target_masks to token_ref_target_masks
        token_ref_target_masks = None
        if ref_target_masks is not None:
            ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32) # [1, B, H, W]; cast for interpolation
            token_ref_target_masks = nn.functional.interpolate(ref_target_masks, size=(N_h, N_w), mode='nearest') # [1, B, N_h, N_w]
            token_ref_target_masks = token_ref_target_masks.squeeze(0) # [B, N_h, N_w]
            token_ref_target_masks = (token_ref_target_masks > 0)
            token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1) # [B, N_h, N_w] --> [B, N_h * N_w]
            token_ref_target_masks = token_ref_target_masks.to(dtype)

        # Blocks
        kv_cache_dict_ret = {}
        for i, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                block_outputs = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, t, y_seqlens,
                    (N_t, N_h, N_w), num_cond_latents, return_kv, kv_cache_dict.get(i, None), skip_crs_attn, num_ref_latents, audio_hidden_states, ref_img_index, mask_frame_range, token_ref_target_masks, human_num
                )
            else:
                block_outputs = block(
                    hidden_states, encoder_hidden_states, t, y_seqlens,
                    (N_t, N_h, N_w), num_cond_latents, return_kv, kv_cache_dict.get(i, None), skip_crs_attn, num_ref_latents, audio_hidden_states, ref_img_index, mask_frame_range, token_ref_target_masks, human_num
                )
            
            if return_kv:
                hidden_states, kv_cache = block_outputs
                if offload_kv_cache:
                    kv_cache_dict_ret[i] = (kv_cache[0].cpu(), kv_cache[1].cpu())
                else:
                    kv_cache_dict_ret[i] = (kv_cache[0].contiguous(), kv_cache[1].contiguous())
            else:
                hidden_states = block_outputs

        hidden_states = self.final_layer(hidden_states, t, (N_t, N_h, N_w))  # [B, N, C=T_p*H_p*W_p*C_out]

        hidden_states = self.unpatchify(hidden_states, N_t, N_h, N_w)  # [B, C_out, H, W]

        # Cast to float32 for better accuracy
        hidden_states = hidden_states.to(torch.float32)

        if return_kv:
            return hidden_states, kv_cache_dict_ret
        else:
            return hidden_states
    

    def unpatchify(self, x, N_t, N_h, N_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x