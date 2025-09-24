# Modified from https://github.com/ali-vilab/VACE/blob/main/vace/models/wan/wan_vace.py
# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import os
import math
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.utils import is_torch_version

from .wan_transformer3d import (WanAttentionBlock, WanTransformer3DModel,
                                sinusoidal_embedding_1d)
from ..utils import cfg_skip


VIDEOX_OFFLOAD_VACE_LATENTS = os.environ.get("VIDEOX_OFFLOAD_VACE_LATENTS", False)

class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        if VIDEOX_OFFLOAD_VACE_LATENTS:
            c = c.to(x.device)

        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)

        if VIDEOX_OFFLOAD_VACE_LATENTS:
            c_skip = c_skip.to("cpu")
            c = c.to("cpu")

        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c
    
    
class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            if VIDEOX_OFFLOAD_VACE_LATENTS:
                x = x + hints[self.block_id].to(x.device) * context_scale
            else:
                x = x + hints[self.block_id] * context_scale
        return x
    
    
class VaceWanTransformer3DModel(WanTransformer3DModel):
    @register_to_config
    def __init__(self,
                 vace_layers=None,
                 vace_in_dim=None,
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
                 eps=1e-6):
        model_type = "t2v"   # TODO: Hard code for both preview and official versions.
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim,
                         num_heads, num_layers, window_size, qk_norm, cross_attn_norm, eps)

        self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # blocks
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                  self.cross_attn_norm, self.eps,
                                  block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None)
            for i in range(self.num_layers)
        ])

        # vace blocks
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                     self.cross_attn_norm, self.eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward_vace(
        self,
        x,
        vace_context,
        seq_len,
        kwargs
    ):
        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])
        # Context Parallel
        if self.sp_world_size > 1:
            c = torch.chunk(c, self.sp_world_size, dim=1)[self.sp_world_rank]

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)
        
        for block in self.vace_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module, **static_kwargs):
                    def custom_forward(*inputs):
                        return module(*inputs, **static_kwargs)
                    return custom_forward
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, **new_kwargs),
                    c,
                    **ckpt_kwargs,
                )
            else:
                c = block(c, **new_kwargs)
        hints = torch.unbind(c)[:-1]
        return hints

    @cfg_skip()
    def forward(
        self,
        x,
        t,
        vace_context,
        context,
        seq_len,
        vace_context_scale=1.0,
        clip_fea=None,
        y=None,
        cond_flag=True
    ):
        r"""
        Forward pass through the diffusion model

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
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params
        dtype = x.dtype
        device = self.patch_embedding.weight.device
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        # if y is not None:
        #     x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
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
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            dtype=dtype,
            t=t)
        hints = self.forward_vace(x, vace_context, seq_len, kwargs)

        kwargs['hints'] = hints
        kwargs['context_scale'] = vace_context_scale

        # TeaCache
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
        
        # TeaCache
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        def create_custom_forward(module, **static_kwargs):
                            def custom_forward(*inputs):
                                return module(*inputs, **static_kwargs)
                            return custom_forward
                        extra_kwargs = {
                            'e': e0,
                            'seq_lens': seq_lens,
                            'grid_sizes': grid_sizes,
                            'freqs': self.freqs,
                            'context': context,
                            'context_lens': context_lens,
                            'dtype': dtype,
                            't': t,
                        }

                        ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block, **extra_kwargs),
                            x,
                            hints,
                            vace_context_scale,
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
                    def create_custom_forward(module, **static_kwargs):
                        def custom_forward(*inputs):
                            return module(*inputs, **static_kwargs)
                        return custom_forward
                    extra_kwargs = {
                        'e': e0,
                        'seq_lens': seq_lens,
                        'grid_sizes': grid_sizes,
                        'freqs': self.freqs,
                        'context': context,
                        'context_lens': context_lens,
                        'dtype': dtype,
                        't': t,
                    }

                    ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block, **extra_kwargs),
                        x,
                        hints,
                        vace_context_scale,
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

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x