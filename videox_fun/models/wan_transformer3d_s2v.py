# Modified from https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/s2v/model_s2v.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math
import types
from copy import deepcopy
from typing import Any, Dict

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.utils import is_torch_version
from einops import rearrange

from ..dist import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    usp_attn_s2v_forward)
from .attention_utils import attention
from .wan_audio_injector import (AudioInjector_WAN, CausalAudioEncoder,
                                 FramePackMotioner, MotionerTransformers,
                                 rope_precompute)
from .wan_transformer3d import (Wan2_2Transformer3DModel, WanAttentionBlock,
                                WanLayerNorm, WanSelfAttention,
                                sinusoidal_embedding_1d)
from ..utils import cfg_skip


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)

    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


@amp.autocast(enabled=False)
@torch.compiler.disable()
def s2v_rope_apply(x, grid_sizes, freqs, start=None):
    dtype = x.dtype
    n, c = x.size(2), x.size(3) // 2
    # Loop over samples
    output = []
    for i, _ in enumerate(x):
        s = x.size(1)
        x_i = torch.view_as_complex(x[i, :s].to(torch.float32).reshape(s, n, -1, 2))
        freqs_i = freqs[i, :s]
        # Apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])
        # Append to collection
        output.append(x_i)
    return torch.stack(output).to(dtype)


def s2v_rope_apply_qk(q, k, grid_sizes, freqs):
    q = s2v_rope_apply(q, grid_sizes, freqs)
    k = s2v_rope_apply(k, grid_sizes, freqs)
    return q, k


class WanS2VSelfAttention(WanSelfAttention):

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0):
        """
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # Query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q, k = s2v_rope_apply_qk(q, k, grid_sizes, freqs)

        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v=v.to(dtype),
            k_lens=seq_lens,
            window_size=self.window_size)

        # Output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanS2VAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__(
            cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps
        )
        self.self_attn = WanS2VSelfAttention(dim, num_heads, window_size,qk_norm, eps)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dtype=torch.bfloat16, t=0):
        seg_idx = e[1].item()
        seg_idx = min(max(0, seg_idx), x.size(1))
        seg_idx = [0, seg_idx, x.size(1)]
        e = e[0]
        modulation = self.modulation.unsqueeze(2)
        e = (modulation + e).chunk(6, dim=1)

        e = [element.squeeze(1) for element in e]
        norm_x = self.norm1(x)
        parts = []
        for i in range(2):
            parts.append(norm_x[:, seg_idx[i]:seg_idx[i + 1]] *
                         (1 + e[1][:, i:i + 1]) + e[0][:, i:i + 1])
        norm_x = torch.cat(parts, dim=1)
        # Self-attention
        y = self.self_attn(norm_x, seq_lens, grid_sizes, freqs)
        z = []
        for i in range(2):
            z.append(y[:, seg_idx[i]:seg_idx[i + 1]] * e[2][:, i:i + 1])
        y = torch.cat(z, dim=1)
        x = x + y
        # Cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            norm2_x = self.norm2(x)
            parts = []
            for i in range(2):
                parts.append(norm2_x[:, seg_idx[i]:seg_idx[i + 1]] *
                             (1 + e[4][:, i:i + 1]) + e[3][:, i:i + 1])
            norm2_x = torch.cat(parts, dim=1)
            y = self.ffn(norm2_x)
            z = []
            for i in range(2):
                z.append(y[:, seg_idx[i]:seg_idx[i + 1]] * e[5][:, i:i + 1])
            y = torch.cat(z, dim=1)
            x = x + y
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Wan2_2Transformer3DModel_S2V(Wan2_2Transformer3DModel):
    """Wan Transformer 3D model for Speech-to-Video generation."""
    
    # ignore_for_config = [
    #     'args', 'kwargs', 'patch_size', 'cross_attn_norm', 'qk_norm',
    #     'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanS2VAttentionBlock']

    @register_to_config
    def __init__(
        self,
        cond_dim=0,
        audio_dim=5120,
        num_audio_token=4,
        enable_adain=False,
        adain_mode="attn_norm",
        audio_inject_layers=[0, 4, 8, 12, 16, 20, 24, 27],
        zero_init=False,
        zero_timestep=False,
        enable_motioner=True,
        add_last_motion=True,
        enable_tsm=False,
        trainable_token_pos_emb=False,
        motion_token_num=1024,
        enable_framepack=False,  # Mutually exclusive with enable_motioner
        framepack_drop_mode="drop",
        model_type='s2v',
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
        *args,
        **kwargs
    ):
        r"""
        Initialize the S2V diffusion model backbone.
        
        Args:
            cond_dim (`int`, *optional*, defaults to 0):
                Condition dimension for pose/control
            audio_dim (`int`, *optional*, defaults to 5120):
                Audio embedding dimension
            num_audio_token (`int`, *optional*, defaults to 4):
                Number of audio tokens
            enable_adain (`bool`, *optional*, defaults to False):
                Enable adaptive instance normalization
            adain_mode (`str`, *optional*, defaults to "attn_norm"):
                AdaIN mode for audio injection
            audio_inject_layers (`list`, *optional*, defaults to [0, 4, 8, ...]):
                Layer indices for audio injection
            zero_init (`bool`, *optional*, defaults to False):
                Initialize audio injector with zeros
            zero_timestep (`bool`, *optional*, defaults to False):
                Use zero timestep for ref/motion
            enable_motioner (`bool`, *optional*, defaults to True):
                Enable motion encoder
            add_last_motion (`bool`, *optional*, defaults to True):
                Add last motion frame
            enable_tsm (`bool`, *optional*, defaults to False):
                Enable temporal shift module
            trainable_token_pos_emb (`bool`, *optional*, defaults to False):
                Enable trainable token position embedding
            motion_token_num (`int`, *optional*, defaults to 1024):
                Number of motion tokens
            enable_framepack (`bool`, *optional*, defaults to False):
                Enable frame packing (mutually exclusive with enable_motioner)
            framepack_drop_mode (`str`, *optional*, defaults to "drop"):
                Frame packing drop mode
            model_type (`str`, *optional*, defaults to 's2v'):
                Model variant - speech-to-video
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
            in_channels (`int`, *optional*, defaults to 16):
                Alias for in_dim (diffusers compatibility)
            hidden_size (`int`, *optional*, defaults to 2048):
                Alias for dim (diffusers compatibility)
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
            hidden_size=hidden_size
        )

        assert model_type == 's2v'
        self.enable_adain = enable_adain
        self.adain_mode = adain_mode
        self.zero_timestep = zero_timestep  
        self.enable_motioner = enable_motioner
        self.add_last_motion = add_last_motion
        self.enable_framepack = enable_framepack

        # Replace blocks with S2V attention blocks
        self.blocks = nn.ModuleList([
            WanS2VAttentionBlock("cross_attn", dim, ffn_dim, num_heads, window_size, qk_norm,
                                 cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # Initialize audio injector and related components
        all_modules, all_modules_names = torch_dfs(self.blocks, parent_name="root.transformer_blocks")
        if cond_dim > 0:
            # Condition encoder for pose/control
            self.cond_encoder = nn.Conv3d(
                cond_dim,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size)
        self.trainable_cond_mask = nn.Embedding(3, self.dim)
        
        # Causal audio encoder
        self.casual_audio_encoder = CausalAudioEncoder(
            dim=audio_dim,
            out_dim=self.dim,
            num_token=num_audio_token,
            need_global=enable_adain)
        
        # Audio injector for injecting audio features
        self.audio_injector = AudioInjector_WAN(
            all_modules,
            all_modules_names,
            dim=self.dim,
            num_heads=self.num_heads,
            inject_layer=audio_inject_layers,
            root_net=self,
            enable_adain=enable_adain,
            adain_dim=self.dim,
            need_adain_ont=adain_mode != "attn_norm",
        )

        if zero_init:
            self.zero_init_weights()

        # Initialize motioner
        if enable_motioner and enable_framepack:
            raise ValueError(
                "enable_motioner and enable_framepack are mutually exclusive, please set one of them to False"
            )
        if enable_motioner:
            motioner_dim = 2048
            self.motioner = MotionerTransformers(
                patch_size=(2, 4, 4),
                dim=motioner_dim,
                ffn_dim=motioner_dim,
                freq_dim=256,
                out_dim=16,
                num_heads=16,
                num_layers=13,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=False,
                eps=1e-6,
                motion_token_num=motion_token_num,
                enable_tsm=enable_tsm,
                motion_stride=4,
                expand_ratio=2,
                trainable_token_pos_emb=trainable_token_pos_emb,
            )
            self.zip_motion_out = torch.nn.Sequential(
                WanLayerNorm(motioner_dim),
                zero_module(nn.Linear(motioner_dim, self.dim)))

            self.trainable_token_pos_emb = trainable_token_pos_emb
            if trainable_token_pos_emb:
                d = self.dim // self.num_heads
                x = torch.zeros([1, motion_token_num, self.num_heads, d])
                x[..., ::2] = 1

                gride_sizes = [[
                    torch.tensor([0, 0, 0]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([
                        1, self.motioner.motion_side_len,
                        self.motioner.motion_side_len
                    ]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([
                        1, self.motioner.motion_side_len,
                        self.motioner.motion_side_len
                    ]).unsqueeze(0).repeat(1, 1),
                ]]
                token_freqs = s2v_rope_apply(x, gride_sizes, self.freqs)
                token_freqs = token_freqs[0, :,
                                          0].reshape(motion_token_num, -1, 2)
                token_freqs = token_freqs * 0.01
                self.token_freqs = torch.nn.Parameter(token_freqs)

        if enable_framepack:
            self.frame_packer = FramePackMotioner(
                inner_dim=self.dim,
                num_heads=self.num_heads,
                zip_frame_buckets=[1, 2, 16],
                drop_mode=framepack_drop_mode)

    def enable_multi_gpus_inference(self,):
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_s2v_forward, block.self_attn)

    def process_motion(self, motion_latents, drop_motion_frames=False):
        if drop_motion_frames or motion_latents[0].shape[1] == 0:
            return [], []
        self.lat_motion_frames = motion_latents[0].shape[1]
        mot = [self.patch_embedding(m.unsqueeze(0)) for m in motion_latents]
        batch_size = len(mot)

        mot_remb = []
        flattern_mot = []
        for bs in range(batch_size):
            height, width = mot[bs].shape[3], mot[bs].shape[4]
            flat_mot = mot[bs].flatten(2).transpose(1, 2).contiguous()
            motion_grid_sizes = [[
                torch.tensor([-self.lat_motion_frames, 0,
                              0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([0, height, width]).unsqueeze(0).repeat(1, 1),
                torch.tensor([self.lat_motion_frames, height,
                              width]).unsqueeze(0).repeat(1, 1)
            ]]
            motion_rope_emb = rope_precompute(
                flat_mot.detach().view(1, flat_mot.shape[1], self.num_heads,
                                       self.dim // self.num_heads),
                motion_grid_sizes,
                self.freqs,
                start=None)
            mot_remb.append(motion_rope_emb)
            flattern_mot.append(flat_mot)
        return flattern_mot, mot_remb

    def process_motion_frame_pack(self,
                                  motion_latents,
                                  drop_motion_frames=False,
                                  add_last_motion=2):
        flattern_mot, mot_remb = self.frame_packer(motion_latents,
                                                   add_last_motion)
        if drop_motion_frames:
            return [m[:, :0] for m in flattern_mot
                   ], [m[:, :0] for m in mot_remb]
        else:
            return flattern_mot, mot_remb

    def process_motion_transformer_motioner(self,
                                            motion_latents,
                                            drop_motion_frames=False,
                                            add_last_motion=True):
        """
        Process motion frames using transformer-based motioner.
        
        Args:
            motion_latents: List of motion latent tensors
            drop_motion_frames: Whether to drop motion frame information
            add_last_motion: Whether to add the last motion frame
        
        Returns:
            Tuple of (motion tensors, motion rope embeddings)
        """
        batch_size, height, width = len(
            motion_latents), motion_latents[0].shape[2] // self.patch_size[
                1], motion_latents[0].shape[3] // self.patch_size[2]

        freqs = self.freqs
        device = self.patch_embedding.weight.device
        if freqs.device != device:
            freqs = freqs.to(device)
        if self.trainable_token_pos_emb:
            token_freqs = self.token_freqs
            token_freqs = token_freqs / token_freqs.norm(
                dim=-1, keepdim=True)
            freqs = [freqs, torch.view_as_complex(token_freqs)]

        # Prepare last motion frame
        if not drop_motion_frames and add_last_motion:
            last_motion_latent = [u[:, -1:] for u in motion_latents]
            last_mot = [
                self.patch_embedding(m.unsqueeze(0)) for m in last_motion_latent
            ]
            last_mot = [m.flatten(2).transpose(1, 2) for m in last_mot]
            last_mot = torch.cat(last_mot)
            gride_sizes = [[
                torch.tensor([-1, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                torch.tensor([0, height,
                              width]).unsqueeze(0).repeat(batch_size, 1),
                torch.tensor([1, height,
                              width]).unsqueeze(0).repeat(batch_size, 1)
            ]]
        else:
            last_mot = torch.zeros([batch_size, 0, self.dim],
                                   device=motion_latents[0].device,
                                   dtype=motion_latents[0].dtype)
            gride_sizes = []

        # Encode motion with motioner (with optional gradient checkpointing)
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            zip_motion = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.motioner), 
                motion_latents, 
                **ckpt_kwargs
            )
        else:
            zip_motion = self.motioner(motion_latents)
        
        # Project motioner output
        zip_motion = self.zip_motion_out(zip_motion)
        if drop_motion_frames:
            zip_motion = zip_motion * 0.0
        
        # Prepare grid sizes for motion rope embedding
        zip_motion_grid_sizes = [[
            torch.tensor([-1, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
            torch.tensor([
                0, self.motioner.motion_side_len, self.motioner.motion_side_len
            ]).unsqueeze(0).repeat(batch_size, 1),
            torch.tensor(
                [1 if not self.trainable_token_pos_emb else -1, height,
                 width]).unsqueeze(0).repeat(batch_size, 1),
        ]]

        mot = torch.cat([last_mot, zip_motion], dim=1)
        gride_sizes = gride_sizes + zip_motion_grid_sizes

        # Compute motion rope embeddings
        motion_rope_emb = rope_precompute(
            mot.detach().view(batch_size, mot.shape[1], self.num_heads,
                              self.dim // self.num_heads),
            gride_sizes,
            freqs,
            start=None)
        return [m.unsqueeze(0) for m in mot
               ], [r.unsqueeze(0) for r in motion_rope_emb]

    def inject_motion(self,
                      x,
                      seq_lens,
                      rope_embs,
                      mask_input,
                      motion_latents,
                      drop_motion_frames=False,
                      add_last_motion=True):
        # Inject the motion frames token to the hidden states
        if self.enable_motioner:
            mot, mot_remb = self.process_motion_transformer_motioner(
                motion_latents,
                drop_motion_frames=drop_motion_frames,
                add_last_motion=add_last_motion)
        elif self.enable_framepack:
            mot, mot_remb = self.process_motion_frame_pack(
                motion_latents,
                drop_motion_frames=drop_motion_frames,
                add_last_motion=add_last_motion)
        else:
            mot, mot_remb = self.process_motion(
                motion_latents, drop_motion_frames=drop_motion_frames)

        if len(mot) > 0:
            x = [torch.cat([u, m], dim=1) for u, m in zip(x, mot)]
            seq_lens = seq_lens + torch.tensor([r.size(1) for r in mot],
                                               dtype=torch.long)
            rope_embs = [
                torch.cat([u, m], dim=1) for u, m in zip(rope_embs, mot_remb)
            ]
            mask_input = [
                torch.cat([
                    m, 2 * torch.ones([1, u.shape[1] - m.shape[1]],
                                      device=m.device,
                                      dtype=m.dtype)
                ],
                          dim=1) for m, u in zip(mask_input, x)
            ]
        return x, seq_lens, rope_embs, mask_input

    def after_transformer_block(self, block_idx, hidden_states):
        """
        Post-processing after each transformer block with audio injection.
        
        Args:
            block_idx: Current transformer block index
            hidden_states: Hidden states from the transformer block
        
        Returns:
            Updated hidden states with audio features injected
        """
        if block_idx in self.audio_injector.injected_block_id.keys():
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            audio_emb = self.merged_audio_emb  # b f n c
            num_frames = audio_emb.shape[1]

            if self.sp_world_size > 1:
                hidden_states = self.all_gather(hidden_states, dim=1)

            input_hidden_states = hidden_states[:, :self.original_seq_len].clone()
            input_hidden_states = rearrange(
                input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)

            # Apply AdaIN for audio injection
            if self.enable_adain and self.adain_mode == "attn_norm":
                audio_emb_global = self.audio_emb_global
                audio_emb_global = rearrange(audio_emb_global,
                                             "b t n c -> (b t) n c")
                adain_hidden_states = self.audio_injector.injector_adain_layers[audio_attn_id](
                    input_hidden_states, temb=audio_emb_global[:, 0]
                )
                attn_hidden_states = adain_hidden_states
            else:
                attn_hidden_states = self.audio_injector.injector_pre_norm_feat[audio_attn_id](
                    input_hidden_states
                )
            audio_emb = rearrange(audio_emb, "b t n c -> (b t) n c", t=num_frames)
            attn_audio_emb = audio_emb
            context_lens = torch.ones(
                attn_hidden_states.shape[0], dtype=torch.long, device=attn_hidden_states.device
            ) * attn_audio_emb.shape[1]

            # Audio cross-attention (with optional gradient checkpointing)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                residual_out = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.audio_injector.injector[audio_attn_id]), 
                    attn_hidden_states,
                    attn_audio_emb,
                    context_lens,
                    **ckpt_kwargs
                )
            else:
                residual_out = self.audio_injector.injector[audio_attn_id](
                    x=attn_hidden_states,
                    context=attn_audio_emb,
                    context_lens=context_lens)
            
            # Reshape and add residual to hidden states
            residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
            hidden_states[:, :self.original_seq_len] = hidden_states[:, :self.original_seq_len] + residual_out

            if self.sp_world_size > 1:
                hidden_states = torch.chunk(
                    hidden_states, self.sp_world_size, dim=1)[self.sp_world_rank]

        return hidden_states

    @cfg_skip()
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        ref_latents,
        motion_latents,
        cond_states,
        audio_input=None,
        motion_frames=[17, 5],
        add_last_motion=2,
        drop_motion_frames=False,
        cond_flag=True,
        *extra_args,
        **extra_kwargs
    ):
        r"""
        Forward pass through the S2V diffusion model.
        
        Args:
            x (List[Tensor]):
                List of input videos each with shape [C, T, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Video token length (not used for this model)
            ref_latents (List[Tensor]):
                List of reference images for each video with shape [C, 1, H, W]
            motion_latents (List[Tensor]):
                List of motion frames for each video with shape [C, T_m, H, W]
            cond_states (List[Tensor]):
                List of condition frames (i.e. pose) each with shape [C, T, H, W]
            audio_input (Tensor, *optional*):
                Input audio embedding [B, num_wav2vec_layer, C_a, T_a]
            motion_frames (`list`, *optional*, defaults to [17, 5]):
                Number of motion frames and motion latents frames encoded by VAE
            add_last_motion (`int`, *optional*, defaults to 2):
                For motioner: if > 0, adds the most recent frame.
                For frame packing: 0=only clean_latents_4x, 1=clean_latents_2x+4x, 2=all
            drop_motion_frames (`bool`, *optional*, defaults to False):
                Whether to drop the motion frames info
            cond_flag (`bool`, *optional*, defaults to True):
                Flag for conditional vs unconditional forward pass
        
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes
        """
        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)
        add_last_motion = self.add_last_motion * add_last_motion

        # Parse motion frames configuration
        if isinstance(motion_frames[0], list):
            motion_frames_0 = motion_frames[0][0]
            motion_frames_1 = motion_frames[0][1]
        else:
            motion_frames_0 = motion_frames[0]
            motion_frames_1 = motion_frames[1]
        
        # Prepare checkpointing utilities
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            def create_custom_forward_with_audio(module, block_idx):
                def custom_forward(*inputs):
                    x = module(*inputs)
                    x = self.after_transformer_block(block_idx, x)
                    return x
                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

        # Patch embedding: convert video to sequence of patches
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        
        # Encode audio features (with optional gradient checkpointing)
        audio_input = torch.cat([audio_input[..., 0:1].repeat(1, 1, 1, motion_frames_0), audio_input], dim=-1)
        
        # Encode audio (with optional gradient checkpointing)
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            audio_emb_res = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.casual_audio_encoder), 
                audio_input, 
                **ckpt_kwargs
            )
        else:
            audio_emb_res = self.casual_audio_encoder(audio_input)
        if self.enable_adain:
            audio_emb_global, audio_emb = audio_emb_res
            self.audio_emb_global = audio_emb_global[:, motion_frames_1:].clone()
        else:
            audio_emb = audio_emb_res
        self.merged_audio_emb = audio_emb[:, motion_frames_1:, :]

        # Encode condition states (with optional gradient checkpointing)
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            cond = [
                torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.cond_encoder), 
                    c.unsqueeze(0), 
                    **ckpt_kwargs
                ) for c in cond_states
            ]
        else:
            cond = [self.cond_encoder(c.unsqueeze(0)) for c in cond_states]
        
        # Add condition features to input
        x = [x_ + pose for x_, pose in zip(x, cond)]

        # Get grid sizes and flatten sequences
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

        original_grid_sizes = deepcopy(grid_sizes)
        grid_sizes = [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]]

        # Add reference image embeddings
        ref = [self.patch_embedding(r.unsqueeze(0)) for r in ref_latents]
        batch_size = len(ref)
        height, width = ref[0].shape[3], ref[0].shape[4]
        ref = [r.flatten(2).transpose(1, 2) for r in ref]  # r: 1 c f h w
        x = [torch.cat([u, r], dim=1) for u, r in zip(x, ref)]

        # Store original sequence length before adding reference
        self.original_seq_len = seq_lens[0]
        seq_lens = seq_lens + torch.tensor([r.size(1) for r in ref], dtype=torch.long)
        ref_grid_sizes = [
            [
                torch.tensor([30, 0, 0]).unsqueeze(0).repeat(batch_size, 1),  # the start index
                torch.tensor([31, height,width]).unsqueeze(0).repeat(batch_size, 1),  # the end index
                torch.tensor([1, height, width]).unsqueeze(0).repeat(batch_size, 1),
            ]  # the range           
        ]
        grid_sizes = grid_sizes + ref_grid_sizes

        # Compute rope embeddings for the input
        x = torch.cat(x)
        b, s, n, d = x.size(0), x.size(1), self.num_heads, self.dim // self.num_heads
        self.pre_compute_freqs = rope_precompute(
            x.detach().view(b, s, n, d), grid_sizes, self.freqs, start=None)
        x = [u.unsqueeze(0) for u in x]
        self.pre_compute_freqs = [u.unsqueeze(0) for u in self.pre_compute_freqs]

        # Inject motion latents and initialize masks
        # Masks indicate: 0=noisy latent, 1=ref latent, 2=motion latent
        mask_input = [
            torch.zeros([1, u.shape[1]], dtype=torch.long, device=x[0].device)
            for u in x
        ]
        for i in range(len(mask_input)):
            mask_input[i][:, self.original_seq_len:] = 1

        self.lat_motion_frames = motion_latents[0].shape[1]
        x, seq_lens, self.pre_compute_freqs, mask_input = self.inject_motion(
            x,
            seq_lens,
            self.pre_compute_freqs,
            mask_input,
            motion_latents,
            drop_motion_frames=drop_motion_frames,
            add_last_motion=add_last_motion)
        x = torch.cat(x, dim=0)
        self.pre_compute_freqs = torch.cat(self.pre_compute_freqs, dim=0)
        mask_input = torch.cat(mask_input, dim=0)

        # Apply trainable condition mask
        x = x + self.trainable_cond_mask(mask_input).to(x.dtype)

        seq_len = seq_lens.max()
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u.unsqueeze(0), u.new_zeros(1, seq_len - u.size(0), u.size(1))],
                      dim=1) for u in x
        ])

        # Compute time embeddings with sinusoidal encoding
        if self.zero_timestep:
            t = torch.cat([t, torch.zeros([1], dtype=t.dtype, device=t.device)])
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float()).to(dtype)
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        if self.zero_timestep:
            e = e[:-1]
            zero_e0 = e0[-1:]
            e0 = e0[:-1]
            token_len = x.shape[1]

            e0 = torch.cat(
                [
                    e0.unsqueeze(2),
                    zero_e0.unsqueeze(2).repeat(e0.size(0), 1, 1, 1)
                ],
                dim=2
            )
            e0 = [e0, self.original_seq_len]
        else:
            e0 = e0.unsqueeze(2).repeat(1, 1, 2, 1)
            e0 = [e0, 0]

        # Encode text context (padded to fixed length)
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if self.sp_world_size > 1:
            # Sharded tensors for long context attn
            x = torch.chunk(x, self.sp_world_size, dim=1)
            sq_size = [u.shape[1] for u in x]
            sq_start_size = sum(sq_size[:self.sp_world_rank])
            x = x[self.sp_world_rank]
            # Confirm the application range of the time embedding in e0[0] for each sequence:
            # - For tokens before seg_id: apply e0[0][:, :, 0]
            # - For tokens after seg_id: apply e0[0][:, :, 1]
            sp_size = x.shape[1]
            seg_idx = e0[1] - sq_start_size
            e0[1] = seg_idx

            self.pre_compute_freqs = torch.chunk(self.pre_compute_freqs, self.sp_world_size, dim=1)
            self.pre_compute_freqs = self.pre_compute_freqs[self.sp_world_rank]

        # TeaCache optimization: skip computation when change is small
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
                            create_custom_forward_with_audio(block, idx),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            self.pre_compute_freqs,
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
                            freqs=self.pre_compute_freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype,
                            t=t  
                        )
                        x = block(x, **kwargs)
                        x = self.after_transformer_block(idx, x)
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for idx, block in enumerate(self.blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_with_audio(block, idx),
                        x,
                        e0,
                        seq_lens,
                        grid_sizes,
                        self.pre_compute_freqs,
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
                        freqs=self.pre_compute_freqs,
                        context=context,
                        context_lens=context_lens,
                        dtype=dtype,
                        t=t  
                    )
                    x = block(x, **kwargs)
                    x = self.after_transformer_block(idx, x)

        # Context Parallel: gather results from all GPUs
        if self.sp_world_size > 1:
            x = self.all_gather(x.contiguous(), dim=1)

        # Truncate to original sequence length
        x = x[:, :self.original_seq_len]
        
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

        # Unpatchify: reconstruct video from patches
        x = self.unpatchify(x, original_grid_sizes)
        x = torch.stack(x)

        # Increment TeaCache counter and reset if completed
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x


    def zero_init_weights(self):
        with torch.no_grad():
            self.trainable_cond_mask = zero_module(self.trainable_cond_mask)
            if hasattr(self, "cond_encoder"):
                self.cond_encoder = zero_module(self.cond_encoder)

            for i in range(self.audio_injector.injector.__len__()):
                self.audio_injector.injector[i].o = zero_module(
                    self.audio_injector.injector[i].o)
                if self.enable_adain:
                    self.audio_injector.injector_adain_layers[i].linear = \
                        zero_module(self.audio_injector.injector_adain_layers[i].linear)