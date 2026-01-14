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

from .qwenimage_transformer2d import (QwenImageTransformer2DModel,
                                      QwenImageTransformerBlock)


class QwenImageControlTransformerBlock(QwenImageTransformerBlock):
    def __init__(
        self, 
        dim: int, num_attention_heads: int, attention_head_dim: int, 
        qk_norm: str = "rms_norm", eps: float = 1e-6, 
        zero_cond_t: bool = False, block_id=0
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, qk_norm, eps, zero_cond_t)
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

        encoder_hidden_states, c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return encoder_hidden_states, c
    
    
class BaseQwenImageTransformerBlock(QwenImageTransformerBlock):
    def __init__(
        self, 
        dim: int, num_attention_heads: int, attention_head_dim: int, 
        qk_norm: str = "rms_norm", eps: float = 1e-6, 
        zero_cond_t: bool = False, block_id=0
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, qk_norm, eps, zero_cond_t)
        self.block_id = block_id

    def forward(self, hidden_states, hints=None, context_scale=1.0, **kwargs):
        encoder_hidden_states, hidden_states = super().forward(hidden_states, **kwargs)
        if self.block_id is not None:
            hidden_states = hidden_states + hints[self.block_id] * context_scale
        return encoder_hidden_states, hidden_states
    
class QwenImageControlTransformer2DModel(QwenImageTransformer2DModel):
    @register_to_config
    def __init__(
        self,
        control_layers=None,
        control_in_dim=None,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
    ):
        super().__init__(
            patch_size, in_channels, out_channels, num_layers, attention_head_dim, 
            num_attention_heads, joint_attention_dim, guidance_embeds, axes_dims_rope,
            zero_cond_t, use_additional_t_cond, use_layer3d_rope
        )

        self.control_layers = [i for i in range(0, self.num_layers, 2)] if control_layers is None else control_layers
        self.control_in_dim = self.in_dim if control_in_dim is None else control_in_dim

        assert 0 in self.control_layers
        self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers)}

        # blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BaseQwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=zero_cond_t,
                    block_id=self.control_layers_mapping[i] if i in self.control_layers else None
                )
                for i in range(num_layers)
            ]
        )

        # control blocks
        self.control_blocks = nn.ModuleList(
            [
                QwenImageControlTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=zero_cond_t,
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
        
        for block in self.control_blocks:
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
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        additional_t_cond=None,
        cond_flag: bool=True,
        control_context=None,
        control_context_scale=1.0,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = torch.stack(encoder_hidden_states)
            encoder_hidden_states_mask = torch.stack(encoder_hidden_states_mask)

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        # Context Parallel
        if self.sp_world_size > 1:
            hidden_states = torch.chunk(hidden_states, self.sp_world_size, dim=1)[self.sp_world_rank]
            if image_rotary_emb is not None:
                image_rotary_emb = (
                    torch.chunk(image_rotary_emb[0], self.sp_world_size, dim=0)[self.sp_world_rank],
                    image_rotary_emb[1]
                )

        # TeaCache
        if self.teacache is not None:
            if cond_flag:
                inp = hidden_states.clone()
                temb_ = temb.clone()
                encoder_hidden_states_ = encoder_hidden_states.clone()

                img_mod_params_ = self.transformer_blocks[0].img_mod(temb_)
                img_mod1_, img_mod2_ = img_mod_params_.chunk(2, dim=-1) 
                img_normed_ = self.transformer_blocks[0].img_norm1(inp)
                modulated_inp, img_gate1_ = self.transformer_blocks[0]._modulate(img_normed_, img_mod1_)

                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    self.should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)

                    if torch.distributed.is_initialized():
                        if not isinstance(self.teacache.accumulated_rel_l1_distance, torch.Tensor):
                            accumulated_distance_tensor = torch.tensor(
                                self.teacache.accumulated_rel_l1_distance, 
                                device=hidden_states.device,
                                dtype=torch.float32
                            )
                        else:
                            accumulated_distance_tensor = self.teacache.accumulated_rel_l1_distance.clone()
                        
                        torch.distributed.broadcast(accumulated_distance_tensor, src=0)
                        self.teacache.accumulated_rel_l1_distance = accumulated_distance_tensor.item()

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
                hidden_states = hidden_states + previous_residual.to(hidden_states.device)[-hidden_states.size()[0]:,]
            else:
                ori_hidden_states = hidden_states.clone().cpu() if self.teacache.offload else hidden_states.clone()

                # Arguments
                kwargs = dict(
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                    modulate_index=modulate_index,
                )
                hints = self.forward_control(
                    hidden_states, control_context, kwargs
                )
                # 4. Transformer blocks
                for index_block, block in enumerate(self.transformer_blocks):
                    # Arguments
                    kwargs = dict(
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                        modulate_index=modulate_index,
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
                if cond_flag:
                    self.teacache.previous_residual_cond = hidden_states.cpu() - ori_hidden_states if self.teacache.offload else hidden_states - ori_hidden_states
                else:
                    self.teacache.previous_residual_uncond = hidden_states.cpu() - ori_hidden_states if self.teacache.offload else hidden_states - ori_hidden_states
                del ori_hidden_states

        else:
            # Arguments
            kwargs = dict(
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
                modulate_index=modulate_index,
            )
            hints = self.forward_control(
                hidden_states, control_context, kwargs
            )
            for index_block, block in enumerate(self.transformer_blocks):
                # Arguments
                kwargs = dict(
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                    modulate_index=modulate_index,
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

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if self.sp_world_size > 1:
            output = self.all_gather(output, dim=1)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return output