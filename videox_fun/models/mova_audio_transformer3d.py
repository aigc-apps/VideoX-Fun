# Modified from https://github.com/OpenMOSS/MOVA/blob/main/mova/diffusion/models/wan_audio_dit.py
"""
Code adapted from DiffSynth-Studio's Wan DiT implementation:
https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/models/wan_video_dit.py
"""

import math
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version
from einops import rearrange

from .attention_utils import attention
from .wan_transformer3d import (Head, WanCrossAttention, WanRMSNorm,
                                WanTransformer3DModel)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 16384, theta: float = 10000.0, s: float = 1.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    pos = torch.arange(end, dtype=torch.float64, device=freqs.device) * s
    freqs = torch.outer(pos, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def legacy_precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0, base_tps=4.0, target_tps=44100/2048):
    s = float(base_tps) / float(target_tps)
    # 1d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta, s)
    # Do not apply positional encoding to the remaining dimensions.
    no_freqs_cis = precompute_freqs_cis(dim // 3, end, theta, s)
    no_freqs_cis = torch.ones_like(no_freqs_cis)
    return f_freqs_cis, no_freqs_cis, no_freqs_cis


def precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim, end, theta)
    return f_freqs_cis.chunk(3, dim=-1)


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def rope_apply_head_dim(x, freqs, head_dim):
    x = rearrange(x, "b s (n d) -> b s n d", d=head_dim)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    # print(f"{x_out.shape = }, {freqs.shape = }")
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class AudioSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def forward(self, x, freqs):
        b, s = x.shape[:2]
        q = self.norm_q(self.q(x)).view(b, s, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(x)).view(b, s, self.num_heads, self.head_dim)
        v = self.v(x).view(b, s, self.num_heads, self.head_dim)
        
        q = rope_apply_head_dim(q.flatten(2), freqs, self.head_dim).view(b, s, self.num_heads, self.head_dim)
        k = rope_apply_head_dim(k.flatten(2), freqs, self.head_dim).view(b, s, self.num_heads, self.head_dim)
        
        x = attention(q, k, v)
        return self.o(x.flatten(2))


class AudioWanAttentionBlock(nn.Module):
    def __init__(
        self, 
        has_image_input: bool, 
        dim,
        ffn_dim,
        num_heads,
        eps: float = 1e-6
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.has_image_input = has_image_input

        self.self_attn = AudioSelfAttention(dim, num_heads, eps)
        # Use WanCrossAttention directly with default parameters
        self.cross_attn = WanCrossAttention(
            dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))

        # modulation
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
        dtype=torch.bfloat16,
        t=0,
    ):
        has_seq = len(e.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=e.dtype, device=e.device) + e).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        # Inline modulate: x * (1 + scale) + shift
        input_x = self.norm1(x) * (1 + scale_msa) + shift_msa
        # Inline gate: x + gate * residual
        x = x + gate_msa * self.self_attn(input_x, freqs)
        # Adapt interface for WanCrossAttention: (x, context, context_lens, dtype, t)
        x = x + self.cross_attn(self.norm3(x), context, context_lens=None, dtype=x.dtype, t=0)
        # Inline modulate: x * (1 + scale) + shift
        input_x = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        # Inline gate: x + gate * residual
        x = x + gate_mlp * self.ffn(input_x)
        return x


class AudioMLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class WanAudioTransformer3DModel(WanTransformer3DModel):
    _repeated_blocks = ("AudioWanAttentionBlock",)

    @register_to_config
    def __init__(
        self,
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
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        cross_attn_type='cross_attn',
        has_image_pos_emb: bool = False,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        vae_type: Literal["oobleck", "dac"] = "oobleck",
    ):
        # Call parent __init__ with compatible parameters
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
            in_channels=in_channels,
            hidden_size=hidden_size,
            add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            downscale_factor_control_adapter=downscale_factor_control_adapter,
            add_ref_conv=add_ref_conv,
            in_dim_ref_conv=in_dim_ref_conv,
            cross_attn_type=cross_attn_type,
        )
        
        # Override patch_embedding with Conv1d for audio
        self.patch_embedding = nn.Conv1d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Override blocks with AudioWanAttentionBlock
        self.blocks = nn.ModuleList([
            AudioWanAttentionBlock(model_type == 'i2v', dim, ffn_dim, num_heads, eps)
            for _ in range(num_layers)
        ])
        
        # Override freqs with audio-specific freqs (DO NOT modify this)
        head_dim = dim // num_heads
        if vae_type == "oobleck":
            self.freqs = legacy_precompute_freqs_cis_1d(head_dim, base_tps=4.0, target_tps=44100/2048)
        elif vae_type == "dac":
            self.freqs = precompute_freqs_cis_1d(head_dim)
        else:
            raise ValueError(f"Invalid VAE type: {vae_type}")
        
        # Override img_emb with AudioMLPProj if has_image_input
        if model_type == 'i2v':
            self.img_emb = AudioMLPProj(1280, dim, has_pos_emb=has_image_pos_emb)
        
        # Store audio-specific config
        self.vae_type = vae_type
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.has_image_pos_emb = has_image_pos_emb


    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b f (p c) -> b c (f p)',
            f=grid_size[0],
            p=self.patch_size[0]
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        seq_len: Optional[int] = None,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        y_camera: Optional[torch.Tensor] = None,
        full_ref: Optional[torch.Tensor] = None,
        subject_ref: Optional[torch.Tensor] = None,
        cond_flag: bool = True,
    ):
        # params
        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs[0].device != device and torch.device(type="meta") != device:
            self.freqs = tuple(freq.to(device) for freq in self.freqs)
        
        if y is not None:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)

        # Inline patchify logic
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f -> b f c').contiguous()
        f = grid_size[0]

        # Prepare parameters for AudioWanAttentionBlock
        audio_seq_lens = torch.tensor([f], dtype=torch.long, device=x.device)
        audio_grid_sizes = torch.tensor([[f]], dtype=torch.long, device=x.device)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, -1).expand(f, -1),
            self.freqs[1][:f].view(f, -1).expand(f, -1),
            self.freqs[2][:f].view(f, -1).expand(f, -1),
        ], dim=-1).reshape(f, 1, -1).to(x.device)

        # time embeddings
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t))
        e = self.time_projection(t).unflatten(1, (6, self.dim))
        
        # context
        context = self.text_embedding(context)
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    e,
                    audio_seq_lens,
                    audio_grid_sizes,
                    freqs,
                    context,
                    None,
                    x.dtype,
                    t,
                    **ckpt_kwargs,
                )
            else:
                # arguments
                kwargs = dict(
                    e=e,
                    seq_lens=audio_seq_lens,
                    grid_sizes=audio_grid_sizes,
                    freqs=freqs,
                    context=context,
                    context_lens=None,
                    dtype=x.dtype,
                    t=t,
                )
                x = block(x, **kwargs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, ))
        return x
