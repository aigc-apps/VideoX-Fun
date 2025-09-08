# Modified from https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/s2v/motioner.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import importlib.metadata
import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models import ModelMixin
from diffusers.models.attention import AdaLayerNorm
from diffusers.utils import BaseOutput, is_torch_version, logging
from einops import rearrange, repeat

from .attention_utils import attention
from .wan_transformer3d import WanAttentionBlock, WanCrossAttention


def rope_precompute(x, grid_sizes, freqs, start=None):
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2

    # split freqs
    if type(freqs) is list:
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = torch.view_as_complex(x.detach().reshape(b, s, n, -1,
                                                      2).to(torch.float64))
    seq_bucket = [0]
    if not type(grid_sizes) is list:
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not type(g) is list:
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    factor_f, factor_h, factor_w = (t_f / seq_f).item(), (
                        t_h / seq_h).item(), (t_w / seq_w).item()
                    # Generate a list of seq_f integers starting from f_o and ending at math.ceil(factor_f * seq_f.item() + f_o.item())
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1,
                                            seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(),
                                            (-t_f - f_o).item() + 1,
                                            seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1,
                                        seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1,
                                        seq_w).astype(int).tolist()

                    assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                    freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][
                        f_sam].conj()
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1)

                    freqs_i = torch.cat([
                        freqs_0.expand(seq_f, seq_h, seq_w, -1),
                        freqs[1][h_sam].view(1, seq_h, 1, -1).expand(
                            seq_f, seq_h, seq_w, -1),
                        freqs[2][w_sam].view(1, 1, seq_w, -1).expand(
                            seq_f, seq_h, seq_w, -1),
                    ],
                                        dim=-1).reshape(seq_len, 1, -1)
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                # apply rotary embedding
                output[i, seq_bucket[-1]:seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs, start=None):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    if type(freqs) is list:
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    output = x.clone()
    seq_bucket = [0]
    if not type(grid_sizes) is list:
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not type(g) is list:
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    factor_f, factor_h, factor_w = (t_f / seq_f).item(), (
                        t_h / seq_h).item(), (t_w / seq_w).item()

                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1,
                                            seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(),
                                            (-t_f - f_o).item() + 1,
                                            seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1,
                                        seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1,
                                        seq_w).astype(int).tolist()

                    assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                    freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][
                        f_sam].conj()
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1)

                    freqs_i = torch.cat([
                        freqs_0.expand(seq_f, seq_h, seq_w, -1),
                        freqs[1][h_sam].view(1, seq_h, 1, -1).expand(
                            seq_f, seq_h, seq_w, -1),
                        freqs[2][w_sam].view(1, 1, seq_w, -1).expand(
                            seq_f, seq_h, seq_w, -1),
                    ],
                                        dim=-1).reshape(seq_len, 1, -1)
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                # apply rotary embedding
                # precompute multipliers
                x_i = torch.view_as_complex(
                    x[i, seq_bucket[-1]:seq_bucket[-1] + seq_len].to(
                        torch.float64).reshape(seq_len, n, -1, 2))
                x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
                output[i, seq_bucket[-1]:seq_bucket[-1] + seq_len] = x_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output.float()



class CausalConv1d(nn.Module):

    def __init__(self,
                 chan_in,
                 chan_out,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 pad_mode='replicate',
                 **kwargs):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (kernel_size - 1, 0)  # T
        self.time_causal_padding = padding

        self.conv = nn.Conv1d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class MotionEncoder_tc(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 num_heads=int,
                 need_global=True,
                 dtype=None,
                 device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.num_heads = num_heads
        self.need_global = need_global
        self.conv1_local = CausalConv1d(
            in_dim, hidden_dim // 4 * num_heads, 3, stride=1)
        if need_global:
            self.conv1_global = CausalConv1d(
                in_dim, hidden_dim // 4, 3, stride=1)
        self.norm1 = nn.LayerNorm(
            hidden_dim // 4,
            elementwise_affine=False,
            eps=1e-6,
            **factory_kwargs)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)

        if need_global:
            self.final_linear = nn.Linear(hidden_dim, hidden_dim,
                                          **factory_kwargs)

        self.norm1 = nn.LayerNorm(
            hidden_dim // 4,
            elementwise_affine=False,
            eps=1e-6,
            **factory_kwargs)

        self.norm2 = nn.LayerNorm(
            hidden_dim // 2,
            elementwise_affine=False,
            eps=1e-6,
            **factory_kwargs)

        self.norm3 = nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        x_ori = x.clone()
        b, c, t = x.shape
        x = self.conv1_local(x)
        x = rearrange(x, 'b (n c) t -> (b n) t c', n=self.num_heads)
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm3(x)
        x = self.act(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)
        padding = self.padding_tokens.repeat(b, x.shape[1], 1, 1)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        if not self.need_global:
            return x_local

        x = self.conv1_global(x_ori)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm3(x)
        x = self.act(x)
        x = self.final_linear(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)

        return x, x_local


class CausalAudioEncoder(nn.Module):

    def __init__(self,
                 dim=5120,
                 num_layers=25,
                 out_dim=2048,
                 video_rate=8,
                 num_token=4,
                 need_global=False):
        super().__init__()
        self.encoder = MotionEncoder_tc(
            in_dim=dim,
            hidden_dim=out_dim,
            num_heads=num_token,
            need_global=need_global)
        weight = torch.ones((1, num_layers, 1, 1)) * 0.01

        self.weights = torch.nn.Parameter(weight)
        self.act = torch.nn.SiLU()

    def forward(self, features):
        with amp.autocast(dtype=torch.float32):
            # features B * num_layers * dim * video_length
            weights = self.act(self.weights)
            weights_sum = weights.sum(dim=1, keepdims=True)
            weighted_feat = ((features * weights) / weights_sum).sum(
                dim=1)  # b dim f
            weighted_feat = weighted_feat.permute(0, 2, 1)  # b f dim
            res = self.encoder(weighted_feat)  # b f n dim

        return res  # b f n dim


class AudioCrossAttention(WanCrossAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        # compute attention
        x = attention(q.to(dtype), k.to(dtype), v.to(dtype), k_lens=context_lens, attention_type="FLASH_ATTENTION")
        # output
        x = x.flatten(2)
        x = self.o(x.to(dtype))
        return x


class AudioInjector_WAN(nn.Module):

    def __init__(self,
                 all_modules,
                 all_modules_names,
                 dim=2048,
                 num_heads=32,
                 inject_layer=[0, 27],
                 root_net=None,
                 enable_adain=False,
                 adain_dim=2048,
                 need_adain_ont=False):
        super().__init__()
        num_injector_layers = len(inject_layer)
        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, WanAttentionBlock):
                for inject_id in inject_layer:
                    if f'transformer_blocks.{inject_id}' in mod_name:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1

        self.injector = nn.ModuleList([
            AudioCrossAttention(
                dim=dim,
                num_heads=num_heads,
                qk_norm=True,
            ) for _ in range(audio_injector_id)
        ])
        self.injector_pre_norm_feat = nn.ModuleList([
            nn.LayerNorm(
                dim,
                elementwise_affine=False,
                eps=1e-6,
            ) for _ in range(audio_injector_id)
        ])
        self.injector_pre_norm_vec = nn.ModuleList([
            nn.LayerNorm(
                dim,
                elementwise_affine=False,
                eps=1e-6,
            ) for _ in range(audio_injector_id)
        ])
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList([
                AdaLayerNorm(
                    output_dim=dim * 2, embedding_dim=adain_dim, chunk_dim=1)
                for _ in range(audio_injector_id)
            ])
            if need_adain_ont:
                self.injector_adain_output_layers = nn.ModuleList(
                    [nn.Linear(dim, dim) for _ in range(audio_injector_id)])


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class LayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class SelfAttention(nn.Module):

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
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class SwinSelfAttention(SelfAttention):

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        assert b == 1, 'Only support batch_size 1'

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        T, H, W = grid_sizes[0].tolist()

        q = rearrange(q, 'b (t h w) n d -> (b t) (h w) n d', t=T, h=H, w=W)
        k = rearrange(k, 'b (t h w) n d -> (b t) (h w) n d', t=T, h=H, w=W)
        v = rearrange(v, 'b (t h w) n d -> (b t) (h w) n d', t=T, h=H, w=W)

        ref_q = q[-1:]
        q = q[:-1]

        ref_k = repeat(
            k[-1:], "1 s n d -> t s n d", t=k.shape[0] - 1)  # t hw n d
        k = k[:-1]
        k = torch.cat([k[:1], k, k[-1:]])
        k = torch.cat([k[1:-1], k[2:], k[:-2], ref_k], dim=1)  # (bt) (3hw) n d

        ref_v = repeat(v[-1:], "1 s n d -> t s n d", t=v.shape[0] - 1)
        v = v[:-1]
        v = torch.cat([v[:1], v, v[-1:]])
        v = torch.cat([v[1:-1], v[2:], v[:-2], ref_v], dim=1)

        # q: b (t h w) n d
        # k: b (t h w) n d
        out = attention(
            q=q,
            k=k,
            v=v,
            # k_lens=torch.tensor([k.shape[1]] * k.shape[0], device=x.device, dtype=torch.long),
            window_size=self.window_size)
        out = torch.cat([out, ref_v[:1]], axis=0)
        out = rearrange(out, '(b t) (h w) n d -> b (t h w) n d', t=T, h=H, w=W)
        x = out

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


#Fix the reference frame RoPE to 1,H,W.
#Set the current frame RoPE to 1.
#Set the previous frame RoPE to 0.
class CasualSelfAttention(SelfAttention):

    def forward(self, x, seq_lens, grid_sizes, freqs):
        shifting = 3
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        assert b == 1, 'Only support batch_size 1'

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        T, H, W = grid_sizes[0].tolist()

        q = rearrange(q, 'b (t h w) n d -> (b t) (h w) n d', t=T, h=H, w=W)
        k = rearrange(k, 'b (t h w) n d -> (b t) (h w) n d', t=T, h=H, w=W)
        v = rearrange(v, 'b (t h w) n d -> (b t) (h w) n d', t=T, h=H, w=W)

        ref_q = q[-1:]
        q = q[:-1]

        grid_sizes = torch.tensor([[1, H, W]] * q.shape[0], dtype=torch.long)
        start = [[shifting, 0, 0]] * q.shape[0]
        q = rope_apply(q, grid_sizes, freqs, start=start)

        ref_k = k[-1:]
        grid_sizes = torch.tensor([[1, H, W]], dtype=torch.long)
        # start = [[shifting, H, W]]

        start = [[shifting + 10, 0, 0]]
        ref_k = rope_apply(ref_k, grid_sizes, freqs, start)
        ref_k = repeat(
            ref_k, "1 s n d -> t s n d", t=k.shape[0] - 1)  # t hw n d

        k = k[:-1]
        k = torch.cat([*([k[:1]] * shifting), k])
        cat_k = []
        for i in range(shifting):
            cat_k.append(k[i:i - shifting])
        cat_k.append(k[shifting:])
        k = torch.cat(cat_k, dim=1)  # (bt) (3hw) n d

        grid_sizes = torch.tensor(
            [[shifting + 1, H, W]] * q.shape[0], dtype=torch.long)
        k = rope_apply(k, grid_sizes, freqs)
        k = torch.cat([k, ref_k], dim=1)

        ref_v = repeat(v[-1:], "1 s n d -> t s n d", t=q.shape[0])  # t hw n d
        v = v[:-1]
        v = torch.cat([*([v[:1]] * shifting), v])
        cat_v = []
        for i in range(shifting):
            cat_v.append(v[i:i - shifting])
        cat_v.append(v[shifting:])
        v = torch.cat(cat_v, dim=1)  # (bt) (3hw) n d
        v = torch.cat([v, ref_v], dim=1)

        # q: b (t h w) n d
        # k: b (t h w) n d
        outs = []
        for i in range(q.shape[0]):
            out = attention(
                q=q[i:i + 1],
                k=k[i:i + 1],
                v=v[i:i + 1],
                window_size=self.window_size)
            outs.append(out)
        out = torch.cat(outs, dim=0)
        out = torch.cat([out, ref_v[:1]], axis=0)
        out = rearrange(out, '(b t) (h w) n d -> b (t h w) n d', t=T, h=H, w=W)
        x = out

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class MotionerAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 self_attn_block="SelfAttention"):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = LayerNorm(dim, eps)
        if self_attn_block == "SelfAttention":
            self.self_attn = SelfAttention(dim, num_heads, window_size, qk_norm,
                                           eps)
        elif self_attn_block == "SwinSelfAttention":
            self.self_attn = SwinSelfAttention(dim, num_heads, window_size,
                                               qk_norm, eps)
        elif self_attn_block == "CasualSelfAttention":
            self.self_attn = CasualSelfAttention(dim, num_heads, window_size,
                                                 qk_norm, eps)

        self.norm2 = LayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
    ):
        # self-attention
        y = self.self_attn(self.norm1(x).float(), seq_lens, grid_sizes, freqs)
        x = x + y
        y = self.ffn(self.norm2(x).float())
        x = x + y
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = self.head(self.norm(x))
        return x


class MotionerTransformers(nn.Module, PeftAdapterMixin):

    def __init__(
        self,
        patch_size=(1, 2, 2),
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        self_attn_block="SelfAttention",
        motion_token_num=1024,
        enable_tsm=False,
        motion_stride=4,
        expand_ratio=2,
        trainable_token_pos_emb=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.enable_tsm = enable_tsm
        self.motion_stride = motion_stride
        self.expand_ratio = expand_ratio
        self.sample_c = self.patch_size[0]

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # blocks
        self.blocks = nn.ModuleList([
            MotionerAttentionBlock(
                dim,
                ffn_dim,
                num_heads,
                window_size,
                qk_norm,
                cross_attn_norm,
                eps,
                self_attn_block=self_attn_block) for _ in range(num_layers)
        ])

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        self.gradient_checkpointing = False

        self.motion_side_len = int(math.sqrt(motion_token_num))
        assert self.motion_side_len**2 == motion_token_num
        self.token = nn.Parameter(
            torch.zeros(1, motion_token_num, dim).contiguous())

        self.trainable_token_pos_emb = trainable_token_pos_emb
        if trainable_token_pos_emb:
            x = torch.zeros([1, motion_token_num, num_heads, d])
            x[..., ::2] = 1

            gride_sizes = [[
                torch.tensor([0, 0, 0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([1, self.motion_side_len,
                              self.motion_side_len]).unsqueeze(0).repeat(1, 1),
                torch.tensor([1, self.motion_side_len,
                              self.motion_side_len]).unsqueeze(0).repeat(1, 1),
            ]]
            token_freqs = rope_apply(x, gride_sizes, self.freqs)
            token_freqs = token_freqs[0, :, 0].reshape(motion_token_num, -1, 2)
            token_freqs = token_freqs * 0.01
            self.token_freqs = torch.nn.Parameter(token_freqs)

    def after_patch_embedding(self, x):
        return x

    def forward(
        self,
        x,
    ):
        """
        x:              A list of videos each with shape [C, T, H, W].
        t:              [B].
        context:        A list of text embeddings each with shape [L, C].
        """
        # params
        motion_frames = x[0].shape[1]
        device = self.patch_embedding.weight.device
        freqs = self.freqs
        if freqs.device != device:
            freqs = freqs.to(device)

        if self.trainable_token_pos_emb:
            with amp.autocast(dtype=torch.float64):
                token_freqs = self.token_freqs.to(torch.float64)
                token_freqs = token_freqs / token_freqs.norm(
                    dim=-1, keepdim=True)
                freqs = [freqs, torch.view_as_complex(token_freqs)]

        if self.enable_tsm:
            sample_idx = [
                sample_indices(
                    u.shape[1],
                    stride=self.motion_stride,
                    expand_ratio=self.expand_ratio,
                    c=self.sample_c) for u in x
            ]
            x = [
                torch.flip(torch.flip(u, [1])[:, idx], [1])
                for idx, u in zip(sample_idx, x)
            ]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        x = self.after_patch_embedding(x)

        seq_f, seq_h, seq_w = x[0].shape[-3:]
        batch_size = len(x)
        if not self.enable_tsm:
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            grid_sizes = [[
                torch.zeros_like(grid_sizes), grid_sizes, grid_sizes
            ]]
            seq_f = 0
        else:
            grid_sizes = []
            for idx in sample_idx[0][::-1][::self.sample_c]:
                tsm_frame_grid_sizes = [[
                    torch.tensor([idx, 0,
                                  0]).unsqueeze(0).repeat(batch_size, 1),
                    torch.tensor([idx + 1, seq_h,
                                  seq_w]).unsqueeze(0).repeat(batch_size, 1),
                    torch.tensor([1, seq_h,
                                  seq_w]).unsqueeze(0).repeat(batch_size, 1),
                ]]
                grid_sizes += tsm_frame_grid_sizes
            seq_f = sample_idx[0][-1] + 1

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat([u for u in x])

        batch_size = len(x)

        token_grid_sizes = [[
            torch.tensor([seq_f, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
            torch.tensor(
                [seq_f + 1, self.motion_side_len,
                 self.motion_side_len]).unsqueeze(0).repeat(batch_size, 1),
            torch.tensor(
                [1 if not self.trainable_token_pos_emb else -1, seq_h,
                 seq_w]).unsqueeze(0).repeat(batch_size, 1),
        ]  # 第三行代表rope emb的想要覆盖到的范围
                           ]

        grid_sizes = grid_sizes + token_grid_sizes
        token_unpatch_grid_sizes = torch.stack([
            torch.tensor([1, 32, 32], dtype=torch.long)
            for b in range(batch_size)
        ])
        token_len = self.token.shape[1]
        token = self.token.clone().repeat(x.shape[0], 1, 1).contiguous()
        seq_lens = seq_lens + torch.tensor([t.size(0) for t in token],
                                           dtype=torch.long)
        x = torch.cat([x, token], dim=1)
        # arguments
        kwargs = dict(
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )

        # grad ckpt args
        def create_custom_forward(module, return_dict=None):

            def custom_forward(*inputs, **kwargs):
                if return_dict is not None:
                    return module(*inputs, **kwargs, return_dict=return_dict)
                else:
                    return module(*inputs, **kwargs)

            return custom_forward

        ckpt_kwargs: Dict[str, Any] = ({
            "use_reentrant": False
        } if is_torch_version(">=", "1.11.0") else {})

        for idx, block in enumerate(self.blocks):
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    **ckpt_kwargs,
                )
            else:
                x = block(x, **kwargs)
        # head
        out = x[:, -token_len:]
        return out

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))


class FramePackMotioner(nn.Module):

    def __init__(
            self,
            inner_dim=1024,
            num_heads=16,  # Used to indicate the number of heads in the backbone network; unrelated to this module's design
            zip_frame_buckets=[
                1, 2, 16
            ],  # Three numbers representing the number of frames sampled for patch operations from the nearest to the farthest frames
            drop_mode="drop",  # If not "drop", it will use "padd", meaning padding instead of deletion
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = nn.Conv3d(
            16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(
            16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(
            16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.zip_frame_buckets = torch.tensor(
            zip_frame_buckets, dtype=torch.long)

        self.inner_dim = inner_dim
        self.num_heads = num_heads

        assert (inner_dim %
                num_heads) == 0 and (inner_dim // num_heads) % 2 == 0
        d = inner_dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)
        self.drop_mode = drop_mode

    def forward(self, motion_latents, add_last_motion=2):
        motion_frames = motion_latents[0].shape[1]
        mot = []
        mot_remb = []
        for m in motion_latents:
            lat_height, lat_width = m.shape[2], m.shape[3]
            padd_lat = torch.zeros(16, self.zip_frame_buckets.sum(), lat_height,
                                   lat_width).to(
                                       device=m.device, dtype=m.dtype)
            overlap_frame = min(padd_lat.shape[1], m.shape[1])
            if overlap_frame > 0:
                padd_lat[:, -overlap_frame:] = m[:, -overlap_frame:]

            if add_last_motion < 2 and self.drop_mode != "drop":
                zero_end_frame = self.zip_frame_buckets[:self.zip_frame_buckets.
                                                        __len__() -
                                                        add_last_motion -
                                                        1].sum()
                padd_lat[:, -zero_end_frame:] = 0

            padd_lat = padd_lat.unsqueeze(0)
            clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[:, :, -self.zip_frame_buckets.sum(
            ):, :, :].split(
                list(self.zip_frame_buckets)[::-1], dim=2)  # 16, 2 ,1

            # patchfy
            clean_latents_post = self.proj(clean_latents_post).flatten(
                2).transpose(1, 2)
            clean_latents_2x = self.proj_2x(clean_latents_2x).flatten(
                2).transpose(1, 2)
            clean_latents_4x = self.proj_4x(clean_latents_4x).flatten(
                2).transpose(1, 2)

            if add_last_motion < 2 and self.drop_mode == "drop":
                clean_latents_post = clean_latents_post[:, :
                                                        0] if add_last_motion < 2 else clean_latents_post
                clean_latents_2x = clean_latents_2x[:, :
                                                    0] if add_last_motion < 1 else clean_latents_2x

            motion_lat = torch.cat(
                [clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

            # rope
            start_time_id = -(self.zip_frame_buckets[:1].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[0]
            grid_sizes = [] if add_last_motion < 2 and self.drop_mode == "drop" else \
                        [
                            [torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                            torch.tensor([end_time_id, lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                            torch.tensor([self.zip_frame_buckets[0], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), ]
                        ]

            start_time_id = -(self.zip_frame_buckets[:2].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[1] // 2
            grid_sizes_2x = [] if add_last_motion < 1 and self.drop_mode == "drop" else \
            [
                [torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([end_time_id, lat_height // 4, lat_width // 4]).unsqueeze(0).repeat(1, 1),
                torch.tensor([self.zip_frame_buckets[1], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), ]
            ]

            start_time_id = -(self.zip_frame_buckets[:3].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[2] // 4
            grid_sizes_4x = [[
                torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([end_time_id, lat_height // 8,
                              lat_width // 8]).unsqueeze(0).repeat(1, 1),
                torch.tensor([
                    self.zip_frame_buckets[2], lat_height // 2, lat_width // 2
                ]).unsqueeze(0).repeat(1, 1),
            ]]

            grid_sizes = grid_sizes + grid_sizes_2x + grid_sizes_4x

            motion_rope_emb = rope_precompute(
                motion_lat.detach().view(1, motion_lat.shape[1], self.num_heads,
                                         self.inner_dim // self.num_heads),
                grid_sizes,
                self.freqs,
                start=None)

            mot.append(motion_lat)
            mot_remb.append(motion_rope_emb)
        return mot, mot_remb


def sample_indices(N, stride, expand_ratio, c):
    indices = []
    current_start = 0

    while current_start < N:
        bucket_width = int(stride * (expand_ratio**(len(indices) / stride)))

        interval = int(bucket_width / stride * c)
        current_end = min(N, current_start + bucket_width)
        bucket_samples = []
        for i in range(current_end - 1, current_start - 1, -interval):
            for near in range(c):
                bucket_samples.append(i - near)

        indices += bucket_samples[::-1]
        current_start += bucket_width

    return indices