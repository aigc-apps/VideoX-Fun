# Modified from https://github.com/ali-vilab/VACE/blob/main/control/models/wan/wan_control.py
# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import inspect
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import (TimestepEmbedding, Timesteps,
                                         apply_rotary_emb,
                                         get_1d_rotary_pos_embed)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import (USE_PEFT_BACKEND, is_torch_npu_available,
                             is_torch_version, logging, scale_lora_layers,
                             unscale_lora_layers)

from .flux2_transformer2d import (Flux2SingleTransformerBlock,
                                  Flux2Transformer2DModel,
                                  Flux2TransformerBlock)


class Flux2ControlTransformerBlock(Flux2TransformerBlock):
    def __init__(
        self, 
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
        block_id=0
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, mlp_ratio, eps, bias)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        encoder_hidden_states, c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return encoder_hidden_states, c
    
    
class BaseFlux2TransformerBlock(Flux2TransformerBlock):
    def __init__(
        self, 
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
        block_id=0
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, mlp_ratio, eps, bias)
        self.block_id = block_id

    def forward(self, hidden_states, hints=None, context_scale=1.0, **kwargs):
        encoder_hidden_states, hidden_states = super().forward(hidden_states, **kwargs)
        if self.block_id is not None:
            hidden_states = hidden_states + hints[self.block_id] * context_scale
        return encoder_hidden_states, hidden_states


class Flux2ControlTransformer2DModel(Flux2Transformer2DModel):
    @register_to_config
    def __init__(
        self,
        control_layers=None,
        control_in_dim=None,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: Optional[int] = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
    ):
        super().__init__(
            patch_size, in_channels, out_channels, num_layers, num_single_layers, attention_head_dim, 
            num_attention_heads, joint_attention_dim, timestep_guidance_channels, mlp_ratio, axes_dims_rope, 
            rope_theta, eps
        )

        self.control_layers = [i for i in range(0, self.num_layers, 2)] if control_layers is None else control_layers
        self.control_in_dim = self.in_dim if control_in_dim is None else control_in_dim

        assert 0 in self.control_layers
        self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers)}

        # blocks
        del self.transformer_blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BaseFlux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    block_id=self.control_layers_mapping[i] if i in self.control_layers else None
                )
                for i in range(num_layers)
            ]
        )

        # control blocks
        self.control_transformer_blocks = nn.ModuleList(
            [
                Flux2ControlTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    block_id=i
                )
                for i in self.control_layers
            ]
        )

        # control patch embeddings
        self.control_img_in = nn.Linear(self.control_in_dim, self.inner_dim)

    def forward_control(
        self,
        x,
        control_context,
        kwargs
    ):
        # embeddings
        c = self.control_img_in(control_context)
        # Context Parallel
        if self.sp_world_size > 1:
            c = torch.chunk(c, self.sp_world_size, dim=1)[self.sp_world_rank]

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)
        
        for block in self.control_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module, **static_kwargs):
                    def custom_forward(*inputs):
                        return module(*inputs, **static_kwargs)
                    return custom_forward
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, **new_kwargs),
                    c,
                    **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, c = block(c, **new_kwargs)
            new_kwargs["encoder_hidden_states"] = encoder_hidden_states
 
        hints = torch.unbind(c)[:-1]
        return hints

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        control_context=None,
        control_context_scale=1.0,
        return_dict: bool = True,
    ):
        num_txt_tokens = encoder_hidden_states.shape[1]

        # 1. Calculate timestep embedding and modulation parameters
        timestep = timestep.to(hidden_states.dtype) * 1000
        guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)[0]

        # 2. Input projection for image (hidden_states) and conditioning text (encoder_hidden_states)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 3. Calculate RoPE embeddings from image and text tokens
        # NOTE: the below logic means that we can't support batched inference with images of different resolutions or
        # text prompts of differents lengths. Is this a use case we want to support?
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        if is_torch_npu_available():
            freqs_cos_image, freqs_sin_image = self.pos_embed(img_ids.cpu())
            image_rotary_emb = (freqs_cos_image.npu(), freqs_sin_image.npu())
            freqs_cos_text, freqs_sin_text = self.pos_embed(txt_ids.cpu())
            text_rotary_emb = (freqs_cos_text.npu(), freqs_sin_text.npu())
        else:
            image_rotary_emb = self.pos_embed(img_ids)
            text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # Arguments
        kwargs = dict(
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_params_img=double_stream_mod_img,
            temb_mod_params_txt=double_stream_mod_txt,
            image_rotary_emb=concat_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        hints = self.forward_control(
            hidden_states, control_context, kwargs
        )

        for index_block, block in enumerate(self.transformer_blocks):
            # Arguments
            kwargs = dict(
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                hints=hints,
                context_scale=control_context_scale
            )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module, **static_kwargs):
                    def custom_forward(*inputs):
                        return module(*inputs, **static_kwargs)
                    return custom_forward

                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, **kwargs),
                    hidden_states,
                    **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(hidden_states, **kwargs)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    single_stream_mod,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_params=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        # 6. Output layers
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
