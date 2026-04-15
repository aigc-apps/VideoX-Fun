# Modified from https://github.com/Fantasy-AMAP/fantasy-talking/blob/main/diffsynth/models
# Copyright Alibaba Inc. All Rights Reserved.
import math
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.utils import is_torch_version

from .attention_utils import attention
from .wan_transformer3d import (WanAttentionBlock, WanLayerNorm, WanRMSNorm,
                                WanSelfAttention, WanTransformer3DModel,
                                sinusoidal_embedding_1d)


class AudioProjModel(nn.Module):
    r"""
    Audio projection model to process audio embeddings.
    """
    
    def __init__(self, audio_in_dim=1024, cross_attention_dim=1024):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.proj = torch.nn.Linear(audio_in_dim, cross_attention_dim, bias=False)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, audio_embeds):
        r"""
        Args:
            audio_embeds (`Tensor`):
                Input audio embeddings
            
        Returns:
            `Tensor`:
                Projected and normalized context tokens
        """
        context_tokens = self.proj(audio_embeds)
        context_tokens = self.norm(context_tokens)
        return context_tokens  # [B,L,C]


class AudioCrossAttentionProcessor(nn.Module):
    r"""
    Processor for audio cross-attention mechanism.
    """
    
    def __init__(self, context_dim, hidden_dim):
        super().__init__()

        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.k_proj.weight)
        nn.init.zeros_(self.v_proj.weight)

        self.sp_world_size = 1
        self.sp_world_rank = 0
        self.all_gather = None

    def __call__(
        self,
        attn: nn.Module,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        audio_proj: torch.Tensor,
        audio_context_lens: torch.Tensor,
        latents_num_frames: int = 21,
        audio_scale: float = 1.0,
    ) -> torch.Tensor:
        r"""
        Args:
            x (`Tensor`):
                Input tensor with shape [B, L1, C]
            context (`Tensor`):
                Context tensor with shape [B, L2, C]
            context_lens (`Tensor`):
                Context lengths with shape [B]
            audio_proj (`Tensor`):
                Audio projections with shape [B, 21, L3, C]
            audio_context_lens (`Tensor`):
                Audio context lengths with shape [B*21]
            latents_num_frames (`int`, *optional*, defaults to 21):
                Number of latent frames
            audio_scale (`float`, *optional*, defaults to 1.0):
                Audio attention scaling factor
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), attn.num_heads, attn.head_dim

        # Compute query, key, value
        q = attn.norm_q(attn.q(x)).view(b, -1, n, d)
        k = attn.norm_k(attn.k(context)).view(b, -1, n, d)
        v = attn.v(context).view(b, -1, n, d)
        k_img = attn.norm_k_img(attn.k_img(context_img)).view(b, -1, n, d)
        v_img = attn.v_img(context_img).view(b, -1, n, d)
        img_x = attention(q, k_img, v_img, k_lens=None)
        # Compute attention
        x = attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        img_x = img_x.flatten(2)

        if len(audio_proj.shape) == 4:
            if self.sp_world_size > 1:
                q = self.all_gather(q, dim=1)

                length = int(np.floor(q.size()[1] / latents_num_frames) * latents_num_frames)
                origin_length = q.size()[1]
                if origin_length > length:
                    q_pad = q[:, length:]
                    q = q[:, :length]
            audio_q = q.view(b * latents_num_frames, -1, n, d)  # [B, 21, L1, N, D]
            ip_key = self.k_proj(audio_proj).view(b * latents_num_frames, -1, n, d)
            ip_value = self.v_proj(audio_proj).view(b * latents_num_frames, -1, n, d)
            audio_x = attention(
                audio_q, ip_key, ip_value, k_lens=audio_context_lens, attention_type="NORMAL"
            )
            audio_x = audio_x.view(b, q.size(1), n, d)
            if self.sp_world_size > 1:
                if origin_length > length:
                    audio_x = torch.cat([audio_x, q_pad], dim=1)
                audio_x = torch.chunk(audio_x, self.sp_world_size, dim=1)[self.sp_world_rank]
            audio_x = audio_x.flatten(2)
        elif len(audio_proj.shape) == 3:
            ip_key = self.k_proj(audio_proj).view(b, -1, n, d)
            ip_value = self.v_proj(audio_proj).view(b, -1, n, d)
            audio_x = attention(q, ip_key, ip_value, k_lens=audio_context_lens, attention_type="NORMAL")
            audio_x = audio_x.flatten(2)
        if isinstance(audio_scale, torch.Tensor):
            audio_scale = audio_scale[:, None, None]

        x = x + img_x + audio_x * audio_scale
        x = attn.o(x)
        return x


class AudioCrossAttention(WanSelfAttention):
    r"""
    Audio cross-attention layer with image context support.
    """
    
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        r"""
        Args:
            dim (`int`):
                Transformer dimension
            num_heads (`int`):
                Number of attention heads
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for windowed attention
            qk_norm (`bool`, *optional*, defaults to True):
                Whether to apply QK normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon for layer normalization
        """
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)

        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.processor = AudioCrossAttentionProcessor(2048, dim)

    def forward(
        self,
        x,
        context,
        context_lens,
        audio_proj,
        audio_context_lens,
        latents_num_frames,
        audio_scale: float = 1.0,
        **kwargs,
    ):
        r"""
        Forward pass through audio cross-attention.

        Args:
            x (`Tensor`):
                Input tensor with shape [B, L1, C]
            context (`Tensor`):
                Context tensor with shape [B, L2, C]
            context_lens (`Tensor`):
                Context lengths with shape [B]
            audio_proj (`Tensor`, *optional*):
                Audio projections
            audio_context_lens (`Tensor`, *optional*):
                Audio context lengths
            latents_num_frames (`int`, *optional*):
                Number of latent frames
            audio_scale (`float`, *optional*, defaults to 1.0):
                Audio attention scaling factor
        
        Returns:
            `Tensor`:
                Output tensor after cross-attention
        """
        if audio_proj is None:
            return self.processor(self, x, context, context_lens)
        else:
            return self.processor(
                self,
                x,
                context,
                context_lens,
                audio_proj,
                audio_context_lens,
                latents_num_frames,
                audio_scale,
            )


class AudioAttentionBlock(nn.Module):
    r"""
    Attention block with audio cross-attention support.
    """
    
    def __init__(
        self,
        cross_attn_type, # Useless
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        r"""
        Args:
            cross_attn_type (`str`):
                Cross-attention type (unused)
            dim (`int`):
                Transformer dimension
            ffn_dim (`int`):
                Feed-forward network dimension
            num_heads (`int`):
                Number of attention heads
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for windowed attention
            qk_norm (`bool`, *optional*, defaults to True):
                Whether to apply QK normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Whether to apply cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon for layer normalization
        """
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
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = AudioCrossAttention(
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

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
        audio_proj=None,
        audio_context_lens=None,
        audio_scale=1,
        dtype=torch.bfloat16,
        t=0,
    ):
        r"""
        Forward pass through the attention block.

        Args:
            x (`Tensor`):
                Input tensor with shape [B, L, C]
            e (`Tensor`):
                Time embedding modulation with shape [B, 6, C]
            seq_lens (`Tensor`):
                Sequence lengths with shape [B]
            grid_sizes (`Tensor`):
                Grid sizes with shape [B, 3]
            freqs (`Tensor`):
                RoPE frequencies
            context (`Tensor`):
                Text context embeddings
            context_lens (`Tensor`, *optional*):
                Context lengths with shape [B]
            audio_proj (`Tensor`, *optional*):
                Audio projections
            audio_context_lens (`Tensor`, *optional*):
                Audio context lengths
            audio_scale (`int`, *optional*, defaults to 1):
                Audio attention scaling factor
            dtype (`torch.dtype`, *optional*, defaults to torch.bfloat16):
                Output dtype to match transformer precision
            t (`int`, *optional*, defaults to 0):
                Timestep (unused, kept for API compatibility)
        
        Returns:
            `Tensor`:
                Output tensor after attention and FFN
        """
        e = (self.modulation + e).chunk(6, dim=1)

        # Self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs, dtype, t=t
        )
        x = x + y * e[2]

        # Cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens, dtype=dtype, t=t,
                audio_proj=audio_proj, audio_context_lens=audio_context_lens, audio_scale=audio_scale,
                latents_num_frames=grid_sizes[0][0],
            )
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class FantasyTalkingTransformer3DModel(WanTransformer3DModel):
    r"""
    FantasyTalking Transformer 3D model with audio integration.
    """
    
    @register_to_config
    def __init__(self,
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
                 cross_attn_type=None,
                 audio_in_dim=768):
        r"""
        Initialize the FantasyTalking diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 'i2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to True):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            cross_attn_type (`str`, *optional*, defaults to None):
                Cross-attention type
            audio_in_dim (`int`, *optional*, defaults to 768):
                Input dimension for audio embeddings
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
            cross_attn_type=cross_attn_type,
        )

        if cross_attn_type is None:
            cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            AudioAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        self.proj_model = AudioProjModel(audio_in_dim, 2048)

    def split_audio_sequence(self, audio_proj_length, num_frames=81):
        r"""
        Map the audio feature sequence to corresponding latent frame slices.

        Args:
            audio_proj_length (`int`):
                The total length of the audio feature sequence
            num_frames (`int`, *optional*, defaults to 81):
                The number of video frames in the training data
        
        Returns:
            `list`:
                A list of [start_idx, end_idx] pairs representing index ranges
        """
        # Average number of tokens per original video frame
        tokens_per_frame = audio_proj_length / num_frames

        # Each latent frame covers 4 video frames, and we want the center
        tokens_per_latent_frame = tokens_per_frame * 4
        half_tokens = int(tokens_per_latent_frame / 2)

        pos_indices = []
        for i in range(int((num_frames - 1) / 4) + 1):
            if i == 0:
                pos_indices.append(0)
            else:
                start_token = tokens_per_frame * ((i - 1) * 4 + 1)
                end_token = tokens_per_frame * (i * 4 + 1)
                center_token = int((start_token + end_token) / 2) - 1
                pos_indices.append(center_token)

        # Build index ranges centered around each position
        pos_idx_ranges = [[idx - half_tokens, idx + half_tokens] for idx in pos_indices]

        # Adjust the first range to avoid negative start index
        pos_idx_ranges[0] = [
            -(half_tokens * 2 - pos_idx_ranges[1][0]),
            pos_idx_ranges[1][0],
        ]

        return pos_idx_ranges

    def split_tensor_with_padding(self, input_tensor, pos_idx_ranges, expand_length=0):
        r"""
        Split the input tensor into subsequences based on index ranges with zero-padding.

        Args:
            input_tensor (`Tensor`):
                Input audio tensor of shape [1, L, 768]
            pos_idx_ranges (`list`):
                A list of index ranges
            expand_length (`int`, *optional*, defaults to 0):
                Number of tokens to expand on both sides
        
        Returns:
            `tuple`:
                sub_sequences: Tensor of shape [1, F, L, 768]
                k_lens: Tensor of shape [F] with actual lengths
        """
        pos_idx_ranges = [
            [idx[0] - expand_length, idx[1] + expand_length] for idx in pos_idx_ranges
        ]
        sub_sequences = []
        seq_len = input_tensor.size(1)  # 173
        max_valid_idx = seq_len - 1  # 172
        k_lens_list = []
        for start, end in pos_idx_ranges:
            # Calculate the fill amount
            pad_front = max(-start, 0)
            pad_back = max(end - max_valid_idx, 0)

            # Calculate the start and end indices of the valid part
            valid_start = max(start, 0)
            valid_end = min(end, max_valid_idx)

            # Extract the valid part
            if valid_start <= valid_end:
                valid_part = input_tensor[:, valid_start : valid_end + 1, :]
            else:
                valid_part = input_tensor.new_zeros((1, 0, input_tensor.size(2)))

            # In the sequence dimension (the 1st dimension) perform padding
            padded_subseq = F.pad(
                valid_part,
                (0, 0, 0, pad_back + pad_front, 0, 0),
                mode="constant",
                value=0,
            )
            k_lens_list.append(padded_subseq.size(-2) - pad_back - pad_front)

            sub_sequences.append(padded_subseq)
        return torch.stack(sub_sequences, dim=1), torch.tensor(
            k_lens_list, dtype=torch.long
        )

    def enable_multi_gpus_inference(self,):
        r"""
        Enable multi-GPU inference using sequence parallel.
        """
        super().enable_multi_gpus_inference()
        for name, module in self.named_modules():
            if module.__class__.__name__ == 'AudioCrossAttentionProcessor':
                module.sp_world_size = self.sp_world_size
                module.sp_world_rank = self.sp_world_rank
                module.all_gather = self.all_gather

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        audio_wav2vec_fea=None,
        clip_fea=None,
        y=None,
        audio_scale=1,
        cond_flag=True
    ):
        r"""
        Forward pass through the diffusion model.

        Args:
            x (`List[Tensor]`):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (`Tensor`):
                Diffusion timesteps tensor of shape [B]
            context (`List[Tensor]`):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            audio_wav2vec_fea (`Tensor`, *optional*):
                Audio wav2vec features
            clip_fea (`Tensor`, *optional*):
                CLIP image features for image-to-video mode
            y (`List[Tensor]`, *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            audio_scale (`int`, *optional*, defaults to 1):
                Audio attention scaling factor
            cond_flag (`bool`, *optional*, defaults to True):
                Whether this is conditional forward pass

        Returns:
            `List[Tensor]`:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
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

        # Get grid_sizes
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
        if t.dim() != 1:
            if t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                last_elements = t[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                t = torch.cat([t, padding], dim=1)
            bt = t.size(0)
            ft = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, ft).unflatten(0, (bt, seq_len)).float()).to(dtype)
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        else:
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

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # Audio proj
        num_frames = (grid_sizes[0][0] - 1) * 4 + 1
        audio_proj_fea = self.proj_model(audio_wav2vec_fea)
        pos_idx_ranges = self.split_audio_sequence(audio_proj_fea.size(1), num_frames=num_frames)
        audio_proj, audio_context_lens = self.split_tensor_with_padding(
            audio_proj_fea, pos_idx_ranges, expand_length=4
        )

        # Context Parallel: split input across GPUs
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            if t.dim() != 1:
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
                e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]
        
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
        
        # Prepare checkpointing utilities
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
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

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            audio_proj,
                            audio_context_lens,
                            audio_scale,
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
                            audio_proj=audio_proj,
                            audio_context_lens=audio_context_lens,
                            audio_scale=audio_scale,
                            dtype=dtype,
                            t=t  
                        )
                        x = block(x, **kwargs)
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        e0,
                        seq_lens,
                        grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        audio_proj,
                        audio_context_lens,
                        audio_scale,
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
                        audio_proj=audio_proj,
                        audio_context_lens=audio_context_lens,
                        audio_scale=audio_scale,
                        dtype=dtype,
                        t=t  
                    )
                    x = block(x, **kwargs)

        # Head: project to output space
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x, e, **ckpt_kwargs)
        else:
            x = self.head(x, e)

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # Unpatchify: reconstruct video from patches
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)

        # Increment teacache counter and reset if completed
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x