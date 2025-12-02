# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import inspect
import json
import os
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.utils import (USE_PEFT_BACKEND, is_torch_version, logging,
                             scale_lora_layers, unscale_lora_layers)

from .attention_utils import attention
from ..dist import (ZMultiGPUsSingleStreamAttnProcessor, get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group)


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                mid_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                mid_size,
                out_size,
                bias=True,
            ),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ZSingleStreamAttnProcessor:
    """
    Processor for Z-Image single stream attention that adapts the existing Attention class to match the behavior of the
    original Z-ImageAttention module.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "ZSingleStreamAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)  # todo

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Compute joint attention
        hidden_states = attention(
            query,
            key,
            value,
            attn_mask=attention_mask
        )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


@maybe_allow_in_graph
class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads

        # Refactored to use diffusers Attention with custom processor
        # Original Z-Image params: dim, n_heads, n_kv_heads, qk_norm
        self.attention = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // n_heads,
            heads=n_heads,
            qk_norm="rms_norm" if qk_norm else None,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=ZSingleStreamAttnProcessor(),
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN block
            x = x + gate_mlp * self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x) * scale_mlp,
                )
            )
        else:
            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)

            # FFN block
            x = x + self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x),
                )
            )

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)  # complex64
                freqs_cis.append(freqs_cis_i)

            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:
            # Ensure freqs_cis are on the same device as ids
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


class ZImageTransformer2DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True
    # _no_split_modules = ["ZImageTransformerBlock"]
    # _skip_layerwise_casting_patterns = ["t_embedder", "cap_embedder"]  # precision sensitive layers

    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)
        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

        self.sp_world_size = 1
        self.sp_world_rank = 0

    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            self.gradient_checkpointing = kwargs["value"]
        elif "enable" in kwargs:
            self.gradient_checkpointing = kwargs["enable"]
        else:
            raise ValueError("Invalid set gradient checkpointing")

    def enable_multi_gpus_inference(self,):
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather
        self.set_attn_processor(ZMultiGPUsSingleStreamAttnProcessor())

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify(
        self,
        all_image: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
        cap_padding_len: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []

        for i, image in enumerate(all_image):
            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            # pad mask
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_image_size,
            all_image_pos_ids,
            all_image_pad_mask,
        )

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            ### Process Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            all_cap_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            cap_padded_feat = torch.cat(
                [cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)],
                dim=0,
            )
            all_cap_feats_out.append(cap_padded_feat)

            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            # pad mask
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        # Match t_embedder output dtype to x for layerwise casting compatibility
        adaln_input = t.type_as(x)
        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.noise_refiner:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x, x_attn_mask, x_freqs_cis, adaln_input,
                    **ckpt_kwargs,
                )
        else:
            for layer in self.noise_refiner:
                x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

        # cap embed & refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.context_refiner:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                cap_feats = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    cap_feats, 
                    cap_attn_mask, 
                    cap_freqs_cis,
                    **ckpt_kwargs,
                )
        else:
            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

        # unified
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.layers:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                unified = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    unified, 
                    unified_attn_mask, 
                    unified_freqs_cis, 
                    adaln_input,
                    **ckpt_kwargs,
                )
        else:
            for layer in self.layers:
                unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)
        x = torch.stack(x)
        return x, {}
    

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
                if diffusers_version >= "0.33.0":
                    from diffusers.models.model_loading_utils import \
                        load_model_dict_into_meta
                else:
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

                filtered_state_dict = {}
                for key in state_dict:
                    if key in model.state_dict() and model.state_dict()[key].size() == state_dict[key].size():
                        filtered_state_dict[key] = state_dict[key]
                    else:
                        print(f"Skipping key '{key}' due to size mismatch or absence in model.")
                        
                model_keys = set(model.state_dict().keys())
                loaded_keys = set(filtered_state_dict.keys())
                missing_keys = model_keys - loaded_keys

                def initialize_missing_parameters(missing_keys, model_state_dict, torch_dtype=None):
                    initialized_dict = {}
                    
                    with torch.no_grad():
                        for key in missing_keys:
                            param_shape = model_state_dict[key].shape
                            param_dtype = torch_dtype if torch_dtype is not None else model_state_dict[key].dtype
                            if "control" in key and key.replace("control_", "") in filtered_state_dict.keys():
                                initialized_dict[key] = filtered_state_dict[key.replace("control_", "")].clone()
                                print(f"Initializing missing parameter '{key}' with model.state_dict().")
                            elif "after_proj" in key or "before_proj" in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                                print(f"Initializing missing parameter '{key}' with zero.")
                            elif 'weight' in key:
                                if any(norm_type in key for norm_type in ['norm', 'ln_', 'layer_norm', 'group_norm', 'batch_norm']):
                                    initialized_dict[key] = torch.ones(param_shape, dtype=param_dtype)
                                elif 'embedding' in key or 'embed' in key:
                                    initialized_dict[key] = torch.randn(param_shape, dtype=param_dtype) * 0.02
                                elif 'head' in key or 'output' in key or 'proj_out' in key:
                                    initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                                elif len(param_shape) >= 2:
                                    initialized_dict[key] = torch.empty(param_shape, dtype=param_dtype)
                                    nn.init.xavier_uniform_(initialized_dict[key])
                                else:
                                    initialized_dict[key] = torch.randn(param_shape, dtype=param_dtype) * 0.02
                            elif 'bias' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                            elif 'running_mean' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                            elif 'running_var' in key:
                                initialized_dict[key] = torch.ones(param_shape, dtype=param_dtype)
                            elif 'num_batches_tracked' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=torch.long)
                            else:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                            
                    return initialized_dict

                if missing_keys:
                    print(f"Missing keys will be initialized: {sorted(missing_keys)}")
                    initialized_params = initialize_missing_parameters(
                        missing_keys, 
                        model.state_dict(), 
                        torch_dtype
                    )
                    filtered_state_dict.update(initialized_params)

                if diffusers_version >= "0.33.0":
                    # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
                    # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
                    load_model_dict_into_meta(
                        model,
                        filtered_state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(filtered_state_dict)
                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        filtered_state_dict,
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
                
                params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
                print(f"### All Parameters: {sum(params) / 1e6} M")

                params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
                print(f"### attn1 Parameters: {sum(params) / 1e6} M")
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
        
        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
        
        for key in model.state_dict():
            if "control" in key and key.replace("control_", "") in state_dict.keys() and model.state_dict()[key].size() == state_dict[key.replace("control_", "")].size():
                tmp_state_dict[key] = state_dict[key.replace("control_", "")].clone()
                print(f"Initializing missing parameter '{key}' with model.state_dict().")
                
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