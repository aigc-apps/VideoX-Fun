# Modified from https://github.com/Robbyant/lingbot-world/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# LingBot-World (camera-pose controlled) transformer for Wan2.2 I2V.
# Reference: repo/lingbot-world/wan/modules/model.py
#
# Mirrors the "self-forcing" integration pattern: it reuses the Wan2.2 backbone
# and only adds the extra camera-injection layers found in the lingbot-world
# checkpoints, so it can be loaded and run with minimal changes.

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from einops import rearrange
from torch import nn

from .wan_transformer3d import Wan2_2Transformer3DModel, WanAttentionBlock


def unwrap_block_module(module):
    r"""
    Return the module wrapped by FSDP / activation-checkpoint wrappers.

    When accelerate auto-wraps the attention blocks (e.g. FSDP
    ``TRANSFORMER_BASED_WRAP`` on ``LingbotWorldWanAttentionBlock``),
    ``self.blocks`` holds wrapper modules. A plain ``setattr`` on the wrapper
    would NOT reach the wrapped block: the block keeps reading its own
    ``_cam_hidden`` attribute (initialized to None), silently disabling the
    camera injection. Attributes therefore must be set on the unwrapped module.
    """
    while True:
        if hasattr(module, "_fsdp_wrapped_module"):
            module = module._fsdp_wrapped_module
        elif hasattr(module, "_checkpoint_wrapped_module"):
            module = module._checkpoint_wrapped_module
        else:
            return module


class LingbotWorldWanAttentionBlock(WanAttentionBlock):
    """Wan transformer block with per-block camera (plücker) injection."""

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads,
                         window_size, qk_norm, cross_attn_norm, eps)

        # Camera injection layers (names must match the checkpoint)
        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

        # Per-token camera hidden states, set per-forward by the parent
        # transformer and broadcast over the batch dimension
        self._cam_hidden = None

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
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C] or [B, L, 6, C] for modulation
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            context(Tensor): Shape [B, L_context, C]
            context_lens(Tensor): Shape [B]
        """
        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [ei.squeeze(2) for ei in e]
        else:
            e = (self.modulation + e).chunk(6, dim=1)

        # Self-attention with modulation
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)

        y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype, t=t)
        x = x + y * e[2]

        # Camera injection: shift/scale the features using the plücker condition
        if self._cam_hidden is not None:
            cam = self._cam_hidden.to(x.dtype)
            # Align the camera token length with the (possibly padded) sequence
            if cam.shape[1] != x.shape[1]:
                if cam.shape[1] < x.shape[1]:
                    cam = F.pad(cam, (0, 0, 0, x.shape[1] - cam.shape[1]))
                else:
                    cam = cam[:, :x.shape[1]]
            cam_hidden = self.cam_injector_layer2(F.silu(self.cam_injector_layer1(cam)))
            cam_hidden = cam_hidden + cam
            cam_scale = self.cam_scale_layer(cam_hidden)
            cam_shift = self.cam_shift_layer(cam_hidden)
            x = (1.0 + cam_scale) * x + cam_shift

        # Cross-attention and FFN with modulation
        def cross_attn_ffn(x, context, context_lens, e):
            # Cross-attention: attend to text context
            x = x + self.cross_attn(self.norm3(x).to(x.dtype), context, context_lens, dtype, t=t)

            # FFN with modulation
            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)

            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class WanTransformer3DModel_LingbotWorld(Wan2_2Transformer3DModel):
    r"""
    Wan2.2 I2V backbone with lingbot-world camera-pose (plücker) control.

    Compared with :class:`Wan2_2Transformer3DModel`, it adds the global camera
    embedding layers (``patch_embedding_wancamctrl`` and
    ``c2ws_hidden_states_layer{1,2}``) and per-block camera-injection layers
    (see :class:`LingbotWorldWanAttentionBlock`). The camera condition is passed
    through ``dit_cond_dict``, either as a forward argument or via the
    ``self.dit_cond_dict`` attribute, so the standard Wan2.2 pipeline can be
    reused unchanged.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='i2v',
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
        in_channels=36,
        hidden_size=5120,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        control_type='cam',
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 'i2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 36):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 5120):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 13824):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 40):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 40):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to True):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            in_channels (`int`, *optional*, defaults to 36):
                Alias for in_dim (diffusers compatibility)
            hidden_size (`int`, *optional*, defaults to 5120):
                Alias for dim (diffusers compatibility)
            add_control_adapter (`bool`, *optional*, defaults to False):
                Enable camera control adapter
            in_dim_control_adapter (`int`, *optional*, defaults to 24):
                Input channels for control adapter
            downscale_factor_control_adapter (`int`, *optional*, defaults to 8):
                Downscale factor for control adapter
            add_ref_conv (`bool`, *optional*, defaults to False):
                Enable reference frame convolution
            in_dim_ref_conv (`int`, *optional*, defaults to 16):
                Input channels for reference convolution
            control_type (`str`, *optional*, defaults to 'cam'):
                Camera control type - 'cam' (6-dim plücker) or 'act' (7-dim)
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
            in_channels=in_channels,
            hidden_size=hidden_size,
            add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            downscale_factor_control_adapter=downscale_factor_control_adapter,
            add_ref_conv=add_ref_conv,
            in_dim_ref_conv=in_dim_ref_conv,
        )

        self.control_type = control_type
        control_dim = 6 if control_type == 'cam' else 7

        # Global camera embeddings: project the patchified plücker condition into
        # the transformer hidden space
        self.patch_embedding_wancamctrl = nn.Linear(
            control_dim * 64 * patch_size[0] * patch_size[1] * patch_size[2], dim)
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)

        # Blocks (Wan2.2 uses the "cross_attn" cross-attention type)
        self.blocks = nn.ModuleList([
            LingbotWorldWanAttentionBlock('cross_attn', dim, ffn_dim, num_heads,
                                          window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        # Optional condition holder so the standard pipeline can stay unchanged
        self.dit_cond_dict = None

    def _prepare_cam_hidden(self, dit_cond_dict, device):
        r"""
        Embed the raw plücker condition into per-token camera hidden states.

        Args:
            dit_cond_dict (`dict`):
                Condition dict containing "c2ws_plucker_emb", a tensor (or list of
                tensors) with shape [1, C, F, H, W]
            device (`torch.device`):
                Target device for the produced hidden states

        Returns:
            Tensor:
                Camera hidden states with shape [1, L, C], broadcastable over the
                batch dimension (the camera condition is identical for cond / uncond)
        """
        emb = dit_cond_dict["c2ws_plucker_emb"]
        emb_list = list(emb) if isinstance(emb, (list, tuple)) else [emb]

        # Fold the 3D patches into the channel dimension, matching patch_embedding
        p0, p1, p2 = self.patch_size
        weight_dtype = self.patch_embedding_wancamctrl.weight.dtype
        rearranged = []
        for e in emb_list:
            e = e.to(device=device, dtype=weight_dtype)
            e = rearrange(
                e,
                '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                c1=p0, c2=p1, c3=p2,
            )
            rearranged.append(e)
        emb = torch.cat(rearranged, dim=1)

        emb = self.patch_embedding_wancamctrl(emb)
        hidden = self.c2ws_hidden_states_layer2(
            F.silu(self.c2ws_hidden_states_layer1(emb)))
        return emb + hidden

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        full_ref=None,
        subject_ref=None,
        dit_cond_dict=None,
        **kwargs,
    ):
        r"""
        Forward pass through the diffusion model.

        Prepares the camera hidden states from ``dit_cond_dict`` (falling back to
        ``self.dit_cond_dict``), shares them with every block, and then delegates
        to :meth:`Wan2_2Transformer3DModel.forward`. The signature mirrors the base
        class (so every argument is forwarded explicitly, including ``cond_flag``
        used by the cfg-skip / teacache paths); ``dit_cond_dict`` is the only extra
        argument.

        Args:
            dit_cond_dict (`dict`, *optional*):
                Camera condition dict containing "c2ws_plucker_emb". If None, the
                value stored in ``self.dit_cond_dict`` is used instead.

        Returns:
            Tensor:
                Denoised video tensor with shape [B, C_out, F, H / 8, W / 8]
        """
        if dit_cond_dict is None:
            dit_cond_dict = getattr(self, "dit_cond_dict", None)

        # Prepare the per-token camera hidden states and share them with the
        # blocks. The attribute must be set on the *unwrapped* blocks (see
        # ``unwrap_block_module``) so it survives FSDP auto-wrapping, and it is
        # intentionally NOT cleared after forward: with ``use_reentrant=False``
        # gradient checkpointing the block forwards are re-run during backward
        # and must still see the same condition. Every forward overwrites the
        # value (with None when no condition is given), so nothing goes stale.
        cam_hidden = None
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            cam_hidden = self._prepare_cam_hidden(dit_cond_dict, device=x.device)
        for block in self.blocks:
            unwrap_block_module(block)._cam_hidden = cam_hidden

        # Delegate to the Wan2.2 backbone
        return super().forward(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=y,
            y_camera=y_camera,
            full_ref=full_ref,
            subject_ref=subject_ref,
            **kwargs,
        )
