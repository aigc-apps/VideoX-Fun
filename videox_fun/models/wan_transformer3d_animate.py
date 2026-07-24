# Modified from https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/model_animate.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math
import types
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version, logging
from einops import rearrange

from .attention_utils import attention
from .wan_animate_adapter import FaceAdapter, FaceEncoder
from .wan_animate_motion_encoder import Generator
from .wan_transformer3d import (Head, MLPProj, WanAttentionBlock, WanLayerNorm,
                                WanRMSNorm, WanSelfAttention,
                                WanTransformer3DModel, rope_apply,
                                sinusoidal_embedding_1d)
from ..utils import cfg_skip


class Wan2_2Transformer3DModel_Animate(WanTransformer3DModel):
    """Wan Transformer 3D model for face-driven video animation."""
    
    # _no_split_modules = ['WanAnimateAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=36,
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=40,
        num_layers=40,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        motion_encoder_dim=512,
        use_img_emb=True
    ):
        r"""
        Initialize the Animate diffusion model backbone.
        
        Args:
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 36):
                Input video channels
            dim (`int`, *optional*, defaults to 5120):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 13824):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels
            num_heads (`int`, *optional*, defaults to 40):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 40):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to True):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            motion_encoder_dim (`int`, *optional*, defaults to 512):
                Dimension for motion encoder input
            use_img_emb (`bool`, *optional*, defaults to True):
                Whether to use image embeddings
        """
        model_type = "i2v"   # TODO: Hard code for both preview and official versions.
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
        )

        self.motion_encoder_dim = motion_encoder_dim
        self.use_img_emb = use_img_emb

        # Pose patch embedding for latent inputs
        self.pose_patch_embedding = nn.Conv3d(
            16, dim, kernel_size=patch_size, stride=patch_size
        )

        # Initialize weights
        self.init_weights()

        # Motion encoder for face pixel values
        self.motion_encoder = Generator(size=512, style_dim=512, motion_dim=20)
        
        # Face adapter for injecting face features
        self.face_adapter = FaceAdapter(
            heads_num=self.num_heads,
            hidden_dim=self.dim,
            num_adapter_layers=self.num_layers // 5,
        )

        # Face encoder for motion vector processing
        self.face_encoder = FaceEncoder(
            in_dim=motion_encoder_dim,
            hidden_dim=self.dim,
            num_heads=4,
        )

    def after_patch_embedding(self, x: List[torch.Tensor], pose_latents, face_pixel_values):
        r"""
        Process pose latents and face pixel values after patch embedding.
        
        Args:
            x: List of patch embedded tensors
            pose_latents: Pose latent representations
            face_pixel_values: Face pixel values for motion encoding
        
        Returns:
            Tuple of (modified x, motion vectors)
        """
        # Add pose latents to all frames except the first
        pose_latents = [self.pose_patch_embedding(u.unsqueeze(0)) for u in pose_latents]
        for x_, pose_latents_ in zip(x, pose_latents):
            x_[:, :, 1:] += pose_latents_
        
        # Process face pixel values through motion encoder
        b, c, T, h, w = face_pixel_values.shape
        face_pixel_values = rearrange(face_pixel_values, "b c t h w -> (b t) c h w")

        # Prepare checkpointing utilities
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

        # Batch processing with gradient checkpointing support
        encode_bs = 8
        face_pixel_values_tmp = []
        for i in range(math.ceil(face_pixel_values.shape[0]/encode_bs)):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                face_pixel_values_tmp.append(
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.motion_encoder.get_motion),
                        face_pixel_values[i*encode_bs:(i+1)*encode_bs],
                        **ckpt_kwargs
                    )
                )
            else:
                face_pixel_values_tmp.append(self.motion_encoder.get_motion(face_pixel_values[i*encode_bs:(i+1)*encode_bs]))
        motion_vec = torch.cat(face_pixel_values_tmp)
        
        # Reshape and encode motion vectors
        motion_vec = rearrange(motion_vec, "(b t) c -> b t c", t=T)
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            motion_vec = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.face_encoder),
                motion_vec,
                **ckpt_kwargs
            )
        else:
            motion_vec = self.face_encoder(motion_vec)

        # Add padding for first frame
        B, L, H, C = motion_vec.shape
        pad_face = torch.zeros(B, 1, H, C).type_as(motion_vec)
        motion_vec = torch.cat([pad_face, motion_vec], dim=1)
        return x, motion_vec

    def after_transformer_block(self, block_idx, x, motion_vec, motion_masks=None):
        r"""
        Apply face adapter after transformer block at specified intervals.
        
        Args:
            block_idx: Current block index
            x: Input tensor from transformer block
            motion_vec: Motion vector tensor
            motion_masks: Optional motion masks
        
        Returns:
            Output tensor with face adapter features added
        """
        # Apply face adapter every 5 blocks
        if block_idx % 5 == 0:
            use_context_parallel = self.sp_world_size > 1
            adapter_args = [x, motion_vec, motion_masks, use_context_parallel, self.all_gather, self.sp_world_size, self.sp_world_rank]
            residual_out = self.face_adapter.fuser_blocks[block_idx // 5](*adapter_args)
            x = residual_out + x
        return x

    @cfg_skip()
    def forward(
        self,
        x,
        t,
        clip_fea,
        context,
        seq_len,
        y=None,
        pose_latents=None, 
        face_pixel_values=None,
        cond_flag=True
    ):
        r"""
        Forward pass through the Animate diffusion model.
        
        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            clip_fea (Tensor):
                CLIP image features for image-to-video mode
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode
            pose_latents (List[Tensor], *optional*):
                Pose latent representations
            face_pixel_values (Tensor, *optional*):
                Face pixel values for motion encoding
            cond_flag (`bool`, *optional*, defaults to True):
                Flag for conditional vs unconditional forward pass
        
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes
        """
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
        x, motion_vec = self.after_patch_embedding(x, pose_latents, face_pixel_values)

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        
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

        # Add image embeddings if enabled
        if self.use_img_emb:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
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
                    modulated_inp = e0[0][:, -1, :]
                else:
                    modulated_inp = e0[0]
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
            def create_custom_forward_with_adapter(module, block_idx, motion_vec_ref):
                def custom_forward(*inputs):
                    x = module(*inputs)
                    x = x.to(inputs[0].dtype)
                    x = self.after_transformer_block(block_idx, x, motion_vec_ref.to(inputs[0].dtype))
                    return x
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
                
                for idx, block in enumerate(self.blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward_with_adapter(block, idx, motion_vec),
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
                        x, motion_vec = x.to(dtype), motion_vec.to(dtype)
                        x = self.after_transformer_block(idx, x, motion_vec)
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for idx, block in enumerate(self.blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_with_adapter(block, idx, motion_vec),
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
                    x, motion_vec = x.to(dtype), motion_vec.to(dtype)
                    x = self.after_transformer_block(idx, x, motion_vec)

        # Head: project to output space
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.head),
                x,
                e,
                **ckpt_kwargs
            )
        else:
            x = self.head(x, e)

        # Context Parallel: gather results from all GPUs
        if self.sp_world_size > 1:
            x = self.all_gather(x.contiguous(), dim=1)

        # Unpatchify: reconstruct video from patches
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)

        # Increment teacache counter and reset if completed
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x
