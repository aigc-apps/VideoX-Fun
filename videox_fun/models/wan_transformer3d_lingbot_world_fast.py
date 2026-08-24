# Modified from https://github.com/Robbyant/lingbot-world/blob/main/wan/modules/model_fast.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# LingBot-World (camera-pose controlled) causal transformer for Wan2.2 I2V.
# Reference: repo/lingbot-world/wan/modules/model_fast.py
#
# Combines the Self-Forcing causal attention infrastructure (KV-cache, causal
# RoPE, block-wise streaming) with lingbot-world camera injection layers.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from einops import rearrange

from .wan_transformer3d import sinusoidal_embedding_1d
from .wan_transformer3d_lingbot_world import unwrap_block_module
from .wan_transformer3d_self_forcing import (
    CasualWanAttentionBlock,
    CausalHead,
    WanTransformer3DModel_SelfForcing,
)


class LingbotWorldCasualWanAttentionBlock(CasualWanAttentionBlock):
    r"""
    Causal Wan transformer block with per-block camera (plücker) injection.

    Inherits the Self-Forcing causal self-attention (KV-cache, flex-attention)
    and adds the four camera-injection layers from lingbot-world. The camera
    hidden states are injected after the self-attention residual connection:
        x = (1.0 + cam_scale) * x + cam_shift
    """

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 local_attn_size=-1,
                 sink_size=0):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads,
                         window_size, qk_norm, cross_attn_norm, eps,
                         local_attn_size, sink_size)

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
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        block_mask=None,
        dtype=torch.bfloat16,
        t=0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C] for per-frame modulation
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            context(Tensor): Shape [B, L_context, C]
            context_lens(Tensor): Shape [B]
            kv_cache: KV cache for causal self-attention
            crossattn_cache: Cross-attention cache
            current_start: Current starting position in token sequence
            cache_start: Cache starting position
            block_mask: Block mask for flex attention
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # Self-attention with modulation
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        # Residual connection with modulation
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

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
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class WanTransformer3DModel_LingbotWorldFast(WanTransformer3DModel_SelfForcing):
    r"""
    Wan2.2 I2V backbone with lingbot-world camera-pose (plücker) control and
    Self-Forcing causal streaming inference.

    Combines:
      * Self-Forcing causal attention (KV-cache, causal RoPE, block-wise mask)
      * LingBot-World global camera embeddings (``patch_embedding_wancamctrl``,
        ``c2ws_hidden_states_layer{1,2}``)
      * Per-block camera injection (``cam_injector_layer{1,2}``,
        ``cam_scale_layer``, ``cam_shift_layer``)

    The camera condition is passed through ``dit_cond_dict``, either as a
    forward argument or via the ``self.dit_cond_dict`` attribute. For streaming
    inference, if the full-sequence plücker is provided, the model automatically
    slices the temporal dimension based on ``current_start``.
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
        cross_attn_type='cross_attn',
        # Self-Forcing causal inference parameters
        local_attn_size=-1,
        sink_size=0,
        # LingBot-World camera control
        control_type='cam',
    ):
        r"""
        Initialize the causal diffusion model backbone with camera control.

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
            cross_attn_type (`str`, *optional*, defaults to 'cross_attn'):
                Cross-attention type (lingbot uses generic cross_attn, no CLIP)
            local_attn_size (`int`, *optional*, defaults to -1):
                Local attention window size (-1 for global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Sink token size for local attention cache eviction
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
            cross_attn_type=cross_attn_type,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )

        # Remove img_emb created by base for i2v (lingbot uses no CLIP)
        if hasattr(self, "img_emb"):
            del self.img_emb

        self.control_type = control_type
        control_dim = 6 if control_type == 'cam' else 7

        # Global camera embeddings: project the patchified plücker condition into
        # the transformer hidden space
        self.patch_embedding_wancamctrl = nn.Linear(
            control_dim * 64 * patch_size[0] * patch_size[1] * patch_size[2], dim)
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)

        # Blocks (replace parent blocks with camera-aware causal blocks)
        self.blocks = nn.ModuleList([
            LingbotWorldCasualWanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                window_size, qk_norm, cross_attn_norm, eps,
                local_attn_size, sink_size)
            for _ in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        # Optional condition holder so the standard pipeline can stay unchanged
        self.dit_cond_dict = None

        # Re-initialize weights for the new layers
        self.init_weights()

    def _prepare_cam_hidden(self, dit_cond_dict, grid_sizes, current_start, device):
        r"""
        Embed the raw plücker condition into per-token camera hidden states,
        with optional temporal slicing for streaming inference.

        Args:
            dit_cond_dict (`dict`):
                Condition dict containing "c2ws_plucker_emb", a tensor (or tuple
                of tensors) with shape [1, C, F, H, W]
            grid_sizes (Tensor):
                Shape [B, 3], the second dimension contains (F, H, W) of the
                current block's patch-embedded latent
            current_start (`int`):
                Token offset in the full sequence (for temporal slice computation)
            device (`torch.device`):
                Target device for the produced hidden states

        Returns:
            Tensor:
                Camera hidden states with shape [1, L_block, dim], where
                L_block = f_block * h_grid * w_grid matches current x tokens
        """
        emb = dit_cond_dict["c2ws_plucker_emb"]
        emb_list = list(emb) if isinstance(emb, (list, tuple)) else [emb]

        # Current block dimensions from grid_sizes
        f_current = grid_sizes[0, 0].item()
        h_grid = grid_sizes[0, 1].item()
        w_grid = grid_sizes[0, 2].item()
        frame_seqlen = h_grid * w_grid
        start_frame = current_start // frame_seqlen if frame_seqlen > 0 else 0

        p0, p1, p2 = self.patch_size
        weight_dtype = self.patch_embedding_wancamctrl.weight.dtype
        rearranged = []
        for e in emb_list:
            e = e.to(device=device, dtype=weight_dtype)
            # Slice temporal dimension if full-sequence plücker is provided
            f_total = e.shape[2]  # temporal dim of plücker
            f_needed = f_current * p0  # frames needed after patch fold
            if f_total > f_needed:
                t_start = start_frame * p0
                e = e[:, :, t_start:t_start + f_needed, :, :]
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
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=0,
        clean_x=None,
        aug_t=None,
        dit_cond_dict=None,
    ):
        r"""
        Forward pass through the causal diffusion model with camera control.

        Prepares camera hidden states from ``dit_cond_dict`` (falling back to
        ``self.dit_cond_dict``), slices temporally for streaming, shares with
        every block, then runs the causal Self-Forcing forward.

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B] or [B, F]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode
            kv_cache (`list[dict]`, *optional*):
                Per-layer self-attention KV cache
            crossattn_cache (`list[dict]`, *optional*):
                Per-layer cross-attention KV cache
            current_start (`int`, *optional*, defaults to 0):
                Token offset of the current block in the full sequence
            cache_start (`int`, *optional*, defaults to 0):
                Cache starting position
            clean_x (List[Tensor], *optional*):
                Clean frames for teacher forcing training
            aug_t (Tensor, *optional*):
                Augmented timesteps for teacher forcing
            dit_cond_dict (`dict`, *optional*):
                Camera condition dict containing "c2ws_plucker_emb"

        Returns:
            Tensor:
                Denoised video tensor with shape [B, C_out, F, H, W]
        """
        # Resolve camera condition
        if dit_cond_dict is None:
            dit_cond_dict = getattr(self, "dit_cond_dict", None)

        if self.model_type == 'i2v':
            assert y is not None

        # Params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        # Padding for multi-gpu inference
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # Concatenate clean features for teacher forcing
        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]
            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat(clean_x)
            x = torch.cat([clean_x, x], dim=1)

        # Block mask management for training (kv_cache is None during training)
        num_frames_actual = None
        if kv_cache is None:
            num_frames_actual = grid_sizes[0, 0].item()
            frame_seqlen_actual = grid_sizes[0, 1].item() * grid_sizes[0, 2].item()

            if clean_x is not None:
                expected_mask_len = num_frames_actual * frame_seqlen_actual * 2
                is_teacher_forcing_mask = True
            else:
                expected_mask_len = num_frames_actual * frame_seqlen_actual
                is_teacher_forcing_mask = False

            if (self.block_mask is None or
                getattr(self, '_block_mask_expected_len', None) != expected_mask_len or
                getattr(self, '_block_mask_is_teacher_forcing', None) != is_teacher_forcing_mask):

                if is_teacher_forcing_mask:
                    if self.independent_first_frame:
                        raise NotImplementedError(
                            "Teacher forcing with independent first frame is not supported")
                    self.create_teacher_forcing_mask(
                        num_frames=num_frames_actual,
                        frame_seqlen=frame_seqlen_actual,
                        num_frame_per_block=self.num_frame_per_block,
                        device=device,
                    )
                else:
                    self.create_block_mask_for_training(
                        num_frames=num_frames_actual,
                        frame_seqlen=frame_seqlen_actual,
                        num_frame_per_block=self.num_frame_per_block,
                        independent_first_frame=self.independent_first_frame,
                        device=device
                    )
                self._block_mask_expected_len = expected_mask_len
                self._block_mask_is_teacher_forcing = is_teacher_forcing_mask

        # Time embeddings
        if t.dim() == 1:
            num_frames_actual = grid_sizes[0, 0].item()
            t = t.unsqueeze(1).expand(-1, num_frames_actual)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        # Handle teacher forcing: concatenate clean and noisy time embeddings
        if clean_x is not None:
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            if aug_t.dim() == 1:
                aug_t = aug_t.unsqueeze(1).expand(-1, num_frames_actual)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # Context: text embeddings (padded to fixed length)
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # Camera condition: prepare per-token hidden states.
        cam_hidden = None
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            cam_hidden = self._prepare_cam_hidden(
                dit_cond_dict, grid_sizes, current_start, device)

        # Context Parallel: split input across GPUs
        if self.sp_world_size > 1:
            if t.dim() != 1:
                F_curr = e0.shape[1]
                assert x.shape[1] % F_curr == 0
                frame_seqlen_e0 = x.shape[1] // F_curr
                e0 = e0.repeat_interleave(frame_seqlen_e0, dim=1)
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            # Shard the camera hidden states with the SAME chunking as x so each
            # rank injects the plucker tokens for its own sequence slice. Without
            # this the per-block injection would fall back to cam[:, :x.shape[1]]
            # (the first slice) on every rank, misaligning camera control on all
            # ranks except rank 0.
            if cam_hidden is not None:
                cam_hidden = torch.chunk(cam_hidden, self.sp_world_size, dim=1)[self.sp_world_rank]

        # Share the (possibly sharded) camera condition with every block. The
        # attribute is set on the *unwrapped* blocks so it survives FSDP /
        # activation-checkpoint auto-wrapping, and it is intentionally NOT
        # cleared after forward: with ``use_reentrant=False`` gradient
        # checkpointing the block forwards are re-run during backward and must
        # still see the same condition. Every forward overwrites the value
        # (with None when no condition is given), so nothing goes stale.
        for block in self.blocks:
            unwrap_block_module(block)._cam_hidden = cam_hidden

        # Arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update({
                    "kv_cache": kv_cache[block_index] if kv_cache else None,
                    "current_start": current_start,
                    "cache_start": cache_start
                })
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update({
                    "kv_cache": kv_cache[block_index] if kv_cache else None,
                    "crossattn_cache": crossattn_cache[block_index] if crossattn_cache else None,
                    "current_start": current_start,
                    "cache_start": cache_start
                })
                x = block(x, **kwargs)

        # Remove clean part for teacher forcing output
        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # Context Parallel: gather results from all GPUs
        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # Head: project to output space
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # Unpatchify: reconstruct video from patches
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)
