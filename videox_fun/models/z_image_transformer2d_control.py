# Modified from https://github.com/ali-vilab/VACE/blob/main/vace/models/wan/wan_vace.py
# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (USE_PEFT_BACKEND, is_torch_version,
                             scale_lora_layers, unscale_lora_layers)
import glob
import inspect
import json
import os
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.utils import (USE_PEFT_BACKEND, is_torch_version, logging,
                             scale_lora_layers, unscale_lora_layers)
from .z_image_transformer2d import (ZImageTransformer2DModel, FinalLayer,
                                      ZImageTransformerBlock)


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class ZImageControlTransformerBlock(ZImageTransformerBlock):
    def __init__(
        self, 
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        block_id=0
    ):
        super().__init__(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation)
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

        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c
    
    
class BaseZImageTransformerBlock(ZImageTransformerBlock):
    def __init__(
        self, 
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        block_id=0
    ):
        super().__init__(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation)
        self.block_id = block_id

    def forward(self, hidden_states, hints=None, context_scale=1.0, **kwargs):
        hidden_states = super().forward(hidden_states, **kwargs)
        if self.block_id is not None:
            hidden_states = hidden_states + hints[self.block_id] * context_scale
        return hidden_states
    
class ZImageControlTransformer2DModel(ZImageTransformer2DModel):
    @register_to_config
    def __init__(
        self,
        control_layers_places=None,
        control_in_dim=None,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
    ):
        super().__init__(
            all_patch_size=all_patch_size,
            all_f_patch_size=all_f_patch_size,
            in_channels=in_channels,
            dim=dim,
            n_layers=n_layers,
            n_refiner_layers=n_refiner_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            cap_feat_dim=cap_feat_dim,
            rope_theta=rope_theta,
            t_scale=t_scale,
            axes_dims=axes_dims,
            axes_lens=axes_lens,
        )

        self.control_layers_places = [i for i in range(0, self.num_layers, 2)] if control_layers_places is None else control_layers_places
        self.control_in_dim = self.in_dim if control_in_dim is None else control_in_dim

        assert 0 in self.control_layers_places
        self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers_places)}

        # blocks
        del self.layers
        self.layers = nn.ModuleList(
            [
                BaseZImageTransformerBlock(
                    i, 
                    dim, 
                    n_heads, 
                    n_kv_heads, 
                    norm_eps, 
                    qk_norm,
                    block_id=self.control_layers_mapping[i] if i in self.control_layers_places else None
                )
                for i in range(n_layers)
            ]
        )

        # control blocks
        self.control_layers = nn.ModuleList(
            [
                ZImageControlTransformerBlock(
                    i, 
                    dim, 
                    n_heads, 
                    n_kv_heads, 
                    norm_eps, 
                    qk_norm,
                    block_id=i
                )
                for i in self.control_layers_places
            ]
        )

        # control patch embeddings
        all_x_embedder = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * self.control_in_dim, dim, bias=True)
            print(f_patch_size * patch_size * patch_size * self.control_in_dim, dim)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

        self.control_all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.control_noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

    def forward_control(
        self,
        x,
        cap_feats,
        control_context,
        kwargs,
        t=None,
        patch_size=2,
        f_patch_size=1,
    ):
        # embeddings
        bsz = len(control_context)
        device = control_context[0].device
        (
            control_context,
            x_size,
            x_pos_ids,
            x_inner_pad_mask,
        ) = self.patchify(control_context, patch_size, f_patch_size, cap_feats[0].size(0))

        # control_context embed & refine
        x_item_seqlens = [len(_) for _ in control_context]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        control_context = torch.cat(control_context, dim=0)
        control_context = self.control_all_x_embedder[f"{patch_size}-{f_patch_size}"](control_context)

        # Match t_embedder output dtype to control_context for layerwise casting compatibility
        adaln_input = t.type_as(control_context)
        control_context[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        control_context = list(control_context.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        control_context = pad_sequence(control_context, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        # Context Parallel
        if self.sp_world_size > 1:
            control_context = torch.chunk(control_context, self.sp_world_size, dim=1)[self.sp_world_rank]

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.control_noise_refiner:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                control_context = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    control_context, x_attn_mask, x_freqs_cis, adaln_input,
                    **ckpt_kwargs,
                )
        else:
            for layer in self.control_noise_refiner:
                control_context = layer(control_context, x_attn_mask, x_freqs_cis, adaln_input)

        # unified
        cap_item_seqlens = [len(_) for _ in cap_feats]
        control_context_unified = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            control_context_unified.append(torch.cat([control_context[i][:x_len], cap_feats[i][:cap_len]]))
        control_context_unified = pad_sequence(control_context_unified, batch_first=True, padding_value=0.0)
        c = control_context_unified

        # Context Parallel
        if self.sp_world_size > 1:
            c = torch.chunk(c, self.sp_world_size, dim=1)[self.sp_world_rank]

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)
        
        for layer in self.control_layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module, **static_kwargs):
                    def custom_forward(*inputs):
                        return module(*inputs, **static_kwargs)
                    return custom_forward
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer, **new_kwargs),
                    c,
                    **ckpt_kwargs,
                )
            else:
                c = layer(c, **new_kwargs)
 
        hints = torch.unbind(c)[:-1]
        return hints


    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        control_context=None,
        control_context_scale=1.0,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        # Match t_embedder output dtype to x for layerwise casting compatibility
        adaln_input = t.type_as(x)
        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.noise_refiner:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x, x_attn_mask, x_freqs_cis, adaln_input,
                    **ckpt_kwargs,
                )
        else:
            for layer in self.noise_refiner:
                x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

        # cap embed & refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.context_refiner:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                cap_feats = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    cap_feats, 
                    cap_attn_mask, 
                    cap_freqs_cis,
                    **ckpt_kwargs,
                )
        else:
            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

        # unified
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        # Arguments
        kwargs = dict(
            attn_mask=unified_attn_mask,
            freqs_cis=unified_freqs_cis, 
            adaln_input=adaln_input,
        )
        hints = self.forward_control(
            unified, cap_feats, control_context, kwargs, t=t, patch_size=patch_size, f_patch_size=f_patch_size,
        )

        for layer in self.layers:
            # Arguments
            kwargs = dict(
                attn_mask=unified_attn_mask,
                freqs_cis=unified_freqs_cis, 
                adaln_input=adaln_input,
                hints=hints,
                context_scale=control_context_scale
            )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module, **static_kwargs):
                    def custom_forward(*inputs):
                        return module(*inputs, **static_kwargs)
                    return custom_forward

                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                unified = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer, **kwargs),
                    unified,
                    **ckpt_kwargs,
                )
            else:
                unified = layer(unified, **kwargs)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)
        x = torch.stack(x)
        return x, {}