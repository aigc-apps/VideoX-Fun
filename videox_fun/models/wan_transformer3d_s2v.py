# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import glob
import json
import math
import os
import types
from copy import deepcopy

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from ..dist import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    usp_attn_s2v_forward, xFuserLongContextAttention)
from ..utils import cfg_skip
from .attention_utils import attention, flash_attention
from .cache_utils import TeaCache
from .wan_audio_injector import (AudioInjector_WAN, CausalAudioEncoder,
                                 FramePackMotioner, MotionerTransformers,
                                 rope_precompute)
from .wan_transformer3d import (Head, WanAttentionBlock, WanLayerNorm,
                                WanSelfAttention, rope_params,
                                sinusoidal_embedding_1d)


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
    n, c = x.size(2), x.size(3) // 2
    # loop over samples
    output = []
    for i, _ in enumerate(x):
        s = x.size(1)
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, n, -1, 2))
        freqs_i = freqs[i, :s]
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])
        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


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

        # query, key, value function
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
        x = x.to(dtype)

        # output
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
        # e
        seg_idx = e[1].item()
        seg_idx = min(max(0, seg_idx), x.size(1))
        seg_idx = [0, seg_idx, x.size(1)]
        e = e[0]
        modulation = self.modulation.unsqueeze(2)
        e = (modulation + e).chunk(6, dim=1)
        e = [element.squeeze(1) for element in e]

        # norm
        norm_x = self.norm1(x).float()
        parts = []
        for i in range(2):
            parts.append(norm_x[:, seg_idx[i]:seg_idx[i + 1]] *
                         (1 + e[1][:, i:i + 1]) + e[0][:, i:i + 1])
        norm_x = torch.cat(parts, dim=1)
        # self-attention
        y = self.self_attn(norm_x, seq_lens, grid_sizes, freqs)
        with amp.autocast(dtype=torch.float32):
            z = []
            for i in range(2):
                z.append(y[:, seg_idx[i]:seg_idx[i + 1]] * e[2][:, i:i + 1])
            y = torch.cat(z, dim=1)
            x = x + y

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            norm2_x = self.norm2(x).float()
            parts = []
            for i in range(2):
                parts.append(norm2_x[:, seg_idx[i]:seg_idx[i + 1]] *
                             (1 + e[4][:, i:i + 1]) + e[3][:, i:i + 1])
            norm2_x = torch.cat(parts, dim=1)
            y = self.ffn(norm2_x)
            with amp.autocast(dtype=torch.float32):
                z = []
                for i in range(2):
                    z.append(y[:, seg_idx[i]:seg_idx[i + 1]] * e[5][:, i:i + 1])
                y = torch.cat(z, dim=1)
                x = x + y
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Wan2_2Transformer3DModel_S2V(ModelMixin, ConfigMixin):
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
        *args,
        **kwargs
    ):
        super().__init__()

        assert model_type == 's2v'
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.enbale_adain = enable_adain
        # Whether to assign 0 value timestep to ref/motion
        self.adain_mode = adain_mode
        self.zero_timestep = zero_timestep  
        self.enable_motioner = enable_motioner
        self.add_last_motion = add_last_motion
        self.enable_framepack = enable_framepack

        # Embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # Blocks
        self.blocks = nn.ModuleList([
            WanS2VAttentionBlock("cross_attn", dim, ffn_dim, num_heads, window_size, qk_norm,
                                 cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # Head
        self.head = Head(dim, out_dim, patch_size, eps)

        # Buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1
        )

        # Initialize weights
        self.init_weights()

        all_modules, all_modules_names = torch_dfs(self.blocks, parent_name="root.transformer_blocks")

        if cond_dim > 0:
            self.cond_encoder = nn.Conv3d(
                cond_dim,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size)
        self.trainable_cond_mask = nn.Embedding(3, self.dim)
        self.casual_audio_encoder = CausalAudioEncoder(
            dim=audio_dim,
            out_dim=self.dim,
            num_token=num_audio_token,
            need_global=enable_adain)
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

        # init motioner
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

        self.use_context_parallel = False  
        self.teacache = None
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None
        self.gradient_checkpointing = False
        self.sp_world_size = 1
        self.sp_world_rank = 0
        self.init_weights()

    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            self.gradient_checkpointing = kwargs["value"]
        elif "enable" in kwargs:
            self.gradient_checkpointing = kwargs["enable"]
        else:
            raise ValueError("Invalid set gradient checkpointing")

    def enable_teacache(
        self,
        coefficients,
        num_steps: int,
        rel_l1_thresh: float,
        num_skip_start_steps: int = 0,
        offload: bool = True,
    ):
        self.teacache = TeaCache(
            coefficients, num_steps, rel_l1_thresh=rel_l1_thresh, num_skip_start_steps=num_skip_start_steps, offload=offload
        )

    def share_teacache(
        self,
        transformer = None,
    ):
        self.teacache = transformer.teacache

    def disable_teacache(self):
        self.teacache = None

    def enable_cfg_skip(self, cfg_skip_ratio, num_steps):
        if cfg_skip_ratio != 0:
            self.cfg_skip_ratio = cfg_skip_ratio
            self.current_steps = 0
            self.num_inference_steps = num_steps
        else:
            self.cfg_skip_ratio = None
            self.current_steps = 0
            self.num_inference_steps = None

    def share_cfg_skip(
        self,
        transformer = None,
    ):
        self.cfg_skip_ratio = transformer.cfg_skip_ratio
        self.current_steps = transformer.current_steps
        self.num_inference_steps = transformer.num_inference_steps

    def disable_cfg_skip(self):
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None

    def enable_riflex(
        self,
        k = 6,
        L_test = 66,
        L_test_scale = 4.886,
    ):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                get_1d_rotary_pos_embed_riflex(1024, self.d - 4 * (self.d // 6), use_real=False, k=k, L_test=L_test, L_test_scale=L_test_scale),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def disable_riflex(self):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                rope_params(1024, self.d - 4 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

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
        batch_size, height, width = len(
            motion_latents), motion_latents[0].shape[2] // self.patch_size[
                1], motion_latents[0].shape[3] // self.patch_size[2]

        freqs = self.freqs
        device = self.patch_embedding.weight.device
        if freqs.device != device:
            freqs = freqs.to(device)
        if self.trainable_token_pos_emb:
            with amp.autocast(dtype=torch.float64):
                token_freqs = self.token_freqs.to(torch.float64)
                token_freqs = token_freqs / token_freqs.norm(
                    dim=-1, keepdim=True)
                freqs = [freqs, torch.view_as_complex(token_freqs)]

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

        zip_motion = self.motioner(motion_latents)
        zip_motion = self.zip_motion_out(zip_motion)
        if drop_motion_frames:
            zip_motion = zip_motion * 0.0
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
        if block_idx in self.audio_injector.injected_block_id.keys():
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            audio_emb = self.merged_audio_emb  # b f n c
            num_frames = audio_emb.shape[1]

            if self.sp_world_size > 1:
                hidden_states = self.all_gather(hidden_states, dim=1)

            input_hidden_states = hidden_states[:, :self.original_seq_len].clone()
            input_hidden_states = rearrange(
                input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)

            if self.enbale_adain and self.adain_mode == "attn_norm":
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
            residual_out = self.audio_injector.injector[audio_attn_id](
                x=attn_hidden_states,
                context=attn_audio_emb,
                context_lens=context_lens)
            residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
            hidden_states[:, :self.original_seq_len] = hidden_states[:, :self.original_seq_len] + residual_out

            if self.sp_world_size > 1:
                hidden_states = torch.chunk(
                    hidden_states, self.sp_world_size, dim=1)[self.sp_world_rank]

        return hidden_states

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
        """
        x:                  A list of videos each with shape [C, T, H, W].
        t:                  [B].
        context:            A list of text embeddings each with shape [L, C].
        seq_len:            A list of video token lens, no need for this model.
        ref_latents         A list of reference image for each video with shape [C, 1, H, W].
        motion_latents      A list of  motion frames for each video with shape [C, T_m, H, W].
        cond_states         A list of condition frames (i.e. pose) each with shape [C, T, H, W].
        audio_input         The input audio embedding [B, num_wav2vec_layer, C_a, T_a].
        motion_frames       The number of motion frames and motion latents frames encoded by vae, i.e. [17, 5]
        add_last_motion     For the motioner, if add_last_motion > 0, it means that the most recent frame (i.e., the last frame) will be added.
                            For frame packing, the behavior depends on the value of add_last_motion:
                            add_last_motion = 0: Only the farthest part of the latent (i.e., clean_latents_4x) is included.
                            add_last_motion = 1: Both clean_latents_2x and clean_latents_4x are included.
                            add_last_motion = 2: All motion-related latents are used.
        drop_motion_frames  Bool, whether drop the motion frames info
        """
        device = self.patch_embedding.weight.device
        dtype = x.dtype
        add_last_motion = self.add_last_motion * add_last_motion

        # Embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        # Audio process
        audio_input = torch.cat([audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input], dim=-1)
        audio_emb_res = self.casual_audio_encoder(audio_input)
        if self.enbale_adain:
            audio_emb_global, audio_emb = audio_emb_res
            self.audio_emb_global = audio_emb_global[:, motion_frames[1]:].clone()
        else:
            audio_emb = audio_emb_res
        self.merged_audio_emb = audio_emb[:, motion_frames[1]:, :]

        # Cond states
        cond = [self.cond_encoder(c.unsqueeze(0)) for c in cond_states]
        x = [x_ + pose for x_, pose in zip(x, cond)]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

        original_grid_sizes = deepcopy(grid_sizes)
        grid_sizes = [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]]

        # Ref latents 
        ref = [self.patch_embedding(r.unsqueeze(0)) for r in ref_latents]
        batch_size = len(ref)
        height, width = ref[0].shape[3], ref[0].shape[4]
        ref = [r.flatten(2).transpose(1, 2) for r in ref]  # r: 1 c f h w
        x = [torch.cat([u, r], dim=1) for u, r in zip(x, ref)]

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

        # Compute the rope embeddings for the input
        x = torch.cat(x)
        b, s, n, d = x.size(0), x.size(1), self.num_heads, self.dim // self.num_heads
        self.pre_compute_freqs = rope_precompute(
            x.detach().view(b, s, n, d), grid_sizes, self.freqs, start=None)
        x = [u.unsqueeze(0) for u in x]
        self.pre_compute_freqs = [u.unsqueeze(0) for u in self.pre_compute_freqs]

        # Inject Motion latents.
        # Initialize masks to indicate noisy latent, ref latent, and motion latent.
        # However, at this point, only the first two (noisy and ref latents) are marked;
        # the marking of motion latent will be implemented inside `inject_motion`.
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

        # Apply trainable_cond_mask
        x = x + self.trainable_cond_mask(mask_input).to(x.dtype)

        seq_len = seq_lens.max()
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u.unsqueeze(0), u.new_zeros(1, seq_len - u.size(0), u.size(1))],
                      dim=1) for u in x
        ])

        # Time embeddings
        if self.zero_timestep:
            t = torch.cat([t, torch.zeros([1], dtype=t.dtype, device=t.device)])
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
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

        # context
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

        # TeaCache
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

        # TeaCache
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for idx, block in enumerate(self.blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
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
                        x = self.after_transformer_block(idx, x)
                    else:
                        # arguments
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

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
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
                    x = self.after_transformer_block(idx, x)
                else:
                    # arguments
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

        # Context Parallel
        if self.sp_world_size > 1:
            x = self.all_gather(x.contiguous(), dim=1)

        # Unpatchify
        x = x[:, :self.original_seq_len]
        # Head
        x = self.head(x, e)
        x = self.unpatchify(x, original_grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x

    def unpatchify(self, x, grid_sizes):
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def zero_init_weights(self):
        with torch.no_grad():
            self.trainable_cond_mask = zero_module(self.trainable_cond_mask)
            if hasattr(self, "cond_encoder"):
                self.cond_encoder = zero_module(self.cond_encoder)

            for i in range(self.audio_injector.injector.__len__()):
                self.audio_injector.injector[i].o = zero_module(
                    self.audio_injector.injector[i].o)
                if self.enbale_adain:
                    self.audio_injector.injector_adain_layers[i].linear = \
                        zero_module(self.audio_injector.injector_adain_layers[i].linear)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                from diffusers.models.modeling_utils import \
                    load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate
                
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file, safe_open
                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    print(model_files_safetensors)
                    for _model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(_model_file_safetensors)
                        for key in _state_dict:
                            state_dict[key] = _state_dict[key]

                if diffusers_version >= "0.33.0":
                    # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
                    # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
                    load_model_dict_into_meta(
                        model,
                        state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(state_dict)
                    # move the params from meta device to cpu
                    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                    if len(missing_keys) > 0:
                        raise ValueError(
                            f"Cannot load {cls} from {pretrained_model_path} because the following keys are"
                            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                            " those weights or else make sure your checkpoint file is correct."
                        )

                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        print(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )
                
                return model
            except Exception as e:
                print(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )
        
        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]
        
        if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
            model.state_dict()['patch_embedding.weight'][:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight'][:, :model.state_dict()['patch_embedding.weight'].size()[1], :, :]
            model.state_dict()['patch_embedding.weight'][:, state_dict['patch_embedding.weight'].size()[1]:, :, :] = 0
            state_dict['patch_embedding.weight'] = model.state_dict()['patch_embedding.weight']
        
        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
                
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)
        
        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")
        
        model = model.to(torch_dtype)
        return model
