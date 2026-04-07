# Modified from https://github.com/Fantasy-AMAP/fantasy-talking/blob/main/diffsynth/models
# and https://github.com/Soul-AILab/SoulX-FlashHead/blob/main/flash_head/src/modules/flash_head_model.py
# Copyright Alibaba Inc. All Rights Reserved.
import math
from einops import rearrange
from typing import Any, Dict, Tuple

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.utils import is_torch_version

from ..utils import cfg_skip
from .attention_utils import attention
from .wan_transformer3d import (WanLayerNorm, WanRMSNorm,
                                WanSelfAttention, WanTransformer3DModel,
                                sinusoidal_embedding_1d)


class AudioMLP(nn.Module):
    """MLP matching official flash_head_model.py MLP class structure.
    Weight paths: audio_emb.proj.0.*, audio_emb.proj.1.*, audio_emb.proj.3.*, audio_emb.proj.4.*
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class AudioProjModel(nn.Module):
    """
    Multi-stage audio projection model.
    Projects audio features from wav2vec blocks into context tokens for cross-attention.

    Args:
        seq_len (int): Audio window size for the first frame.
        seq_len_vf (int): Audio window size for subsequent (video) frames.
        blocks (int): Number of wav2vec blocks (e.g. 12).
        channels (int): Feature dimension per block (e.g. 768).
        intermediate_dim (int): Hidden dimension for MLP projections.
        output_dim (int): Output context token dimension.
        context_tokens (int): Number of context tokens per frame.
        norm_output_audio (bool): Whether to apply LayerNorm on output tokens.
    """
    def __init__(
        self,
        seq_len=5,
        seq_len_vf=8,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=1536,
        context_tokens=32,
        norm_output_audio=True,
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

        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds, audio_embeds_vf, dtype=torch.bfloat16):
        """
        Args:
            audio_embeds (Tensor):    [B, 1, seq_len, blocks, channels]  - first frame audio
            audio_embeds_vf (Tensor): [B, F-1, seq_len_vf, blocks, channels] - subsequent frames audio
            dtype: Output dtype to match transformer precision
        Returns:
            context_tokens (Tensor):  [B, F, context_tokens, output_dim]
        """
        # Ensure input dtype matches target dtype
        if audio_embeds.dtype != dtype:
            audio_embeds = audio_embeds.to(dtype=dtype)
        if audio_embeds_vf.dtype != dtype:
            audio_embeds_vf = audio_embeds_vf.to(dtype=dtype)
        
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B = audio_embeds.shape[0]

        # process first frame audio
        audio_embeds = rearrange(audio_embeds, "b f w s c -> (b f) w s c")
        bf, w, s, c = audio_embeds.shape
        audio_embeds = audio_embeds.view(bf, w * s * c)

        # process subsequent frames audio
        audio_embeds_vf = rearrange(audio_embeds_vf, "b f w s c -> (b f) w s c")
        bf_vf, w_vf, s_vf, c_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(bf_vf, w_vf * s_vf * c_vf)

        # first projection
        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))
        audio_embeds = rearrange(audio_embeds, "(b f) c -> b f c", b=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(b f) c -> b f c", b=B)
        audio_embeds_c = torch.cat([audio_embeds, audio_embeds_vf], dim=1)
        b_c, n_t, c_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(b_c * n_t, c_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))
        context_tokens = self.proj3(audio_embeds_c).reshape(b_c * n_t, self.context_tokens, self.output_dim)

        # normalization and reshape
        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(b f) m c -> b f m c", f=video_length)

        # Ensure output dtype matches transformer precision
        if context_tokens.dtype != dtype:
            context_tokens = context_tokens.to(dtype=dtype)

        return context_tokens  # [B, F, context_tokens, output_dim]


class AudioCrossAttention(WanSelfAttention):
    """
    Cross-attention module for audio context.
    Inherits q/k/v/o projections and norm from WanSelfAttention,
    overrides forward to attend x (query) to context (key/value).
    """
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

    def forward(self, x, context, dtype=torch.bfloat16, **kwargs):
        """
        x:       [(B*F), L_x, C]   - query (per-frame patch tokens)
        context: [F, context_tokens, C] - key/value (per-frame audio context tokens)
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).to(dtype=dtype).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).to(dtype=dtype).view(b, -1, n, d)
        v = self.v(context.to(dtype=dtype)).view(b, -1, n, d)
        out = attention(q, k, v, k_lens=None)
        out = out.flatten(2)
        out = self.o(out).to(dtype=dtype)
        return out


class FlashHeadAttentionBlock(nn.Module):
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
        dtype=torch.bfloat16,
        t=0,
    ):
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation.to(dtype=e.dtype, device=e.device) + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).float() * (1 + e[1]) + e[0]).to(dtype=dtype), seq_lens, grid_sizes, freqs, dtype, t=t
        )
        x = (x + y * e[2]).to(dtype=dtype)

        # cross-attention: distribute context per latent frame, ref flash_head_model.py DiTAudioBlock
        # context shape: [B, F, context_tokens, dim]
        # For multi-GPU: audio_cross_attn expects full sequence, so all_gather x first
        if hasattr(self, 'sp_world_size') and self.sp_world_size > 1 and self.all_gather is not None:
            # All gather x to get full sequence for audio cross attention
            x_full = self.all_gather(x, dim=1)
            x_norm_full = self.norm3(x_full)
            num_latent_frames = context.shape[1]
            x_1_full = rearrange(x_norm_full, 'b (f l) c -> (b f) l c', f=num_latent_frames)
            context_1 = context.squeeze(0).to(dtype=dtype)
            
            x_a_full = self.cross_attn(x_1_full, context_1, dtype=dtype)
            # Chunk result back to local rank
            x_a = torch.chunk(x_a_full.flatten(0, 1).unsqueeze(0), self.sp_world_size, dim=1)[self.sp_world_rank]
            x = x + x_a
        else:
            num_latent_frames = context.shape[1]
            x_norm = self.norm3(x)
            x_1 = rearrange(x_norm, 'b (f l) c -> (b f) l c', f=num_latent_frames)
            context_1 = context.squeeze(0).to(dtype=dtype)
            
            x = x + self.cross_attn(
                x_1, context_1, dtype=dtype,
            ).flatten(0, 1).unsqueeze(0)

        y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).to(dtype=dtype))
        x = (x + y * e[5]).to(dtype=dtype)
        return x


class FlashHeadTransformer3DModel(WanTransformer3DModel):
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
        # audio proj params, ref flash_head_model.py AudioProjModel
        audio_window=5,
        vae_scale=4,
        audio_blocks=12,
        audio_channels=768,
        intermediate_dim=512,
        context_tokens=32,
        audio_output_dim=1536,
        norm_output_audio=True
    ):
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim,
                         num_heads, num_layers, window_size, qk_norm, cross_attn_norm, eps)

        if cross_attn_type is None:
            cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            FlashHeadAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        # audio window params, ref flash_head_model.py WanModelAudioProject.__init__
        self.audio_window = audio_window
        self.vae_scale = vae_scale

        # seq_len_vf: matches official flash_head_model.py: audio_window + vae_scale - 1
        seq_len_vf = audio_window + vae_scale - 1

        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=seq_len_vf,
            blocks=audio_blocks,
            channels=audio_channels,
            intermediate_dim=intermediate_dim,
            output_dim=audio_output_dim,
            context_tokens=context_tokens,
            norm_output_audio=norm_output_audio,
        )

        # audio_emb: MLP(768, dim) - matches official audio_emb, used for direct audio embedding
        self.audio_emb = AudioMLP(audio_channels, dim)

    def prepare_audio_context(self, audio_wav2vec_fea: torch.Tensor, num_latent_frames: int, dtype=torch.bfloat16):
        """
        Prepare per-latent-frame audio context from raw wav2vec features.
        Ref: flash_head_model.py WanModelAudioProject.forward audio condition processing.

        Args:
            audio_wav2vec_fea (Tensor): [B, num_video_frames, audio_window, blocks, channels]
                - num_video_frames = num_latent_frames * vae_scale - (vae_scale - 1)
                  e.g. 9 latent frames * 4 - 3 = 33 video frames, but here we expect
                  the caller to pass (1 + (num_latent_frames-1)*vae_scale) frames total.
            num_latent_frames (int): Number of latent frames (e.g. 9 for a 33-frame video).
            dtype: Output dtype to match transformer precision

        Returns:
            context (Tensor): [B, num_latent_frames, context_tokens, audio_output_dim]
        """
        audio_cond = audio_wav2vec_fea  # [B, total_video_frames, audio_window, blocks, channels]

        # first frame: directly use the full audio window
        first_frame_audio = audio_cond[:, :1, ...]  # [B, 1, audio_window, blocks, channels]

        # subsequent frames: rearrange into (n_latent, vae_scale) groups
        latter_frames_audio = rearrange(
            audio_cond[:, 1:, ...],
            "b (n_latent n_frame) w s c -> b n_latent n_frame w s c",
            n_frame=self.vae_scale
        )  # [B, num_latent_frames-1, vae_scale, audio_window, blocks, channels]

        mid_idx = self.audio_window // 2

        # select audio window per sub-frame position within each latent group
        first_of_group = latter_frames_audio[:, :, :1, :mid_idx + 1, ...]    # [B, F-1, 1, mid_idx+1, S, C]
        middle_of_group = latter_frames_audio[:, :, 1:-1, mid_idx:mid_idx + 1, ...]  # [B, F-1, vae_scale-2, 1, S, C]
        last_of_group = latter_frames_audio[:, :, -1:, mid_idx:, ...]         # [B, F-1, 1, audio_window-mid_idx, S, C]

        # flatten sub-window dim: (n_frame, window) -> (n_frame * window)
        latter_frames_processed = torch.cat([
            rearrange(first_of_group,  "b f nf w s c -> b f (nf w) s c"),
            rearrange(middle_of_group, "b f nf w s c -> b f (nf w) s c"),
            rearrange(last_of_group,   "b f nf w s c -> b f (nf w) s c"),
        ], dim=2)  # [B, num_latent_frames-1, seq_len_vf, blocks, channels]

        # project to context tokens: [B, num_latent_frames, context_tokens, audio_output_dim]
        context = self.audio_proj(first_frame_audio, latter_frames_processed, dtype=dtype)
        return context

    def enable_multi_gpus_inference(self,):
        """Enable multi-GPU inference using sequence parallel."""
        from ..dist import (get_sequence_parallel_rank,
                            get_sequence_parallel_world_size, get_sp_group,
                            usp_attn_flashhead_forward)
        import types
        
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather

        # Replace self_attn forward with xfuser version for all blocks
        # and pass sp parameters to each block for audio cross_attn multi-GPU support
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_flashhead_forward, block.self_attn)
            # Pass sp parameters to block for audio cross_attn multi-GPU support
            block.sp_world_size = self.sp_world_size
            block.sp_world_rank = self.sp_world_rank
            block.all_gather = self.all_gather

    def forward(
        self,
        x,
        t,
        seq_len,
        audio_wav2vec_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model.

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W].
            t (Tensor):
                Diffusion timesteps tensor of shape [B].
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]. (Unused in FlashHead)
            seq_len (int):
                Maximum sequence length for positional encoding.
            audio_wav2vec_fea (Tensor, optional):
                Raw wav2vec audio features with shape
                [B, 1 + (num_latent_frames-1)*vae_scale, audio_window, blocks, channels].
            y (List[Tensor], optional):
                Conditional video inputs for image-to-video mode, same shape as x.

        Returns:
            Tensor: Denoised video tensor of shape [B, C_out, F, H/8, W/8].
        """
        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # patch embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            if t.dim() != 1:
                if t.size(1) < seq_len:
                    pad_size = seq_len - t.size(1)
                    last_elements = t[:, -1].unsqueeze(1)
                    padding = last_elements.repeat(1, pad_size)
                    t = torch.cat([t, padding], dim=1)
                bt = t.size(0)
                ft = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, ft).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # audio context: [B, num_latent_frames, context_tokens, audio_output_dim]
        # Replaces text context for cross-attention (ref: flash_head_model.py)
        num_latent_frames = int(grid_sizes[0][0].item())
        audio_context = self.prepare_audio_context(
            audio_wav2vec_fea.to(device=x.device, dtype=x.dtype),
            num_latent_frames=num_latent_frames,
            dtype=x.dtype,
        )

        # context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            if t.dim() != 1:
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
                e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]

        # build block kwargs: pass audio_context as context for per-frame cross-attention
        # Run transformer blocks
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
                    audio_context,
                    dtype, t,
                    **ckpt_kwargs,
                )
            else:
                x = block(
                    x,
                    e=e0,
                    seq_lens=seq_lens,
                    grid_sizes=grid_sizes,
                    freqs=self.freqs,
                    context=audio_context,
                    dtype=dtype,
                    t=t,
                )

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
        return x


# Alias for backward compatibility
WanModelAudioProject = FlashHeadTransformer3DModel