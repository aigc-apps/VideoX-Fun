# Control variant of the MiniMax-H3 transformer, mirroring the VACE-style control branch of
# `z_image_transformer2d_control.py`: a parallel stack of control blocks runs over the packed sequence with the
# clean control video in place of the video rows and injects per-layer skips into the main block stack. The skips
# are gated by zero-initialised projections, so a freshly initialised model is numerically identical to
# `MiniMaxH3Transformer3DModel` and only the control parameters need training.

from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple, Union

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version

from ..dist import MiniMaxH3MultiGPUsAttnProcessor
from .minimax_h3_transformer3d import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerBlock,
    MiniMaxH3TransformerOutput,
)


# `os.environ.get` hands back the raw string, so `VIDEOX_OFFLOAD_VACE_LATENTS=False` would otherwise read as True.
VIDEOX_OFFLOAD_VACE_LATENTS = os.environ.get("VIDEOX_OFFLOAD_VACE_LATENTS", "").lower() in ("1", "true", "yes", "on")

# Seed of `control_proj_in`'s default init, see `MiniMaxH3ControlTransformer3DModel.materialize_missing_control_params`.
CONTROL_PROJ_IN_INIT_SEED = 0


class BaseMiniMaxH3TransformerBlock(MiniMaxH3TransformerBlock):
    r"""
    Main-branch block of the control model. Identical to `MiniMaxH3TransformerBlock` — the state dict of a released
    MiniMax-H3 checkpoint loads unchanged — except that blocks addressed by `control_blocks_places` add the matching
    zero-initialised skip of the control branch to their output. Every block is handed the whole `hints` stack (as in
    `BaseZImageTransformerBlock`) and `block_id` decides whether it uses one; `hints=None` runs the plain base block,
    which is what the base forward of `MiniMaxH3ControlTransformer3DModel` relies on.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        block_id=None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ffn_dim=ffn_dim,
            time_embed_dim=time_embed_dim,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
        )
        self.block_id = block_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        valid_length: Optional[int] = None,
        hints=None,
        context_scale: float = 1.0,
    ) -> torch.Tensor:
        hidden_states = super().forward(hidden_states, temb, adaln_indices, rotary_emb, attention_mask, valid_length)
        if self.block_id is not None and hints is not None:
            hidden_states = hidden_states + hints[self.block_id].to(hidden_states.device) * context_scale
        return hidden_states


class MiniMaxH3ControlTransformerBlock(MiniMaxH3TransformerBlock):
    r"""
    Control-branch block, mirroring `ZImageControlTransformerBlock`. The first block re-bases the control stream on
    the main branch's packed sequence through a zero-initialised `before_proj`; every block emits a zero-initialised
    `after_proj` skip. The running stack of skips and the stream is carried through the layers as one stacked tensor,
    exactly like the z_image control branch.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        block_id=0,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ffn_dim=ffn_dim,
            time_embed_dim=time_embed_dim,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
        )
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(hidden_size, hidden_size)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(
        self,
        c: torch.Tensor,
        x: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        valid_length: Optional[int] = None,
    ) -> torch.Tensor:
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        # Move to correct device if offloading
        if VIDEOX_OFFLOAD_VACE_LATENTS:
            c = c.to(x.device)

        c = super().forward(c, temb, adaln_indices, rotary_emb, attention_mask, valid_length)
        c_skip = self.after_proj(c)

        # Offload to CPU if enabled
        if VIDEOX_OFFLOAD_VACE_LATENTS:
            c_skip = c_skip.to("cpu")
            c = c.to("cpu")

        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class MiniMaxH3ControlTransformer3DModel(MiniMaxH3Transformer3DModel):
    r"""
    MiniMax-H3 with a VACE-style control branch.

    The control video is patchified exactly like the target video and enters through `control_proj_in`. The control
    stream is the full packed sequence with the control embeddings in place of the video rows — the text and audio
    rows are the main branch's own features, mirroring the `[control_rows ; cap_feats]` stream of
    `ZImageControlTransformer2DModel` — so a parallel stack of control blocks, one per `control_blocks_places`,
    attends over exactly the main sequence layout and shares its timestep embeddings, AdaLN table rows, rotary
    coordinates and attention mask. Every block emits a zero-initialised skip that is added to the main branch's
    hidden states at the addressed layer.

    `control_apply_audio` decides whether the skips reach the audio rows: `False` zeros the audio rows out of every
    skip before injection, so the control video guides the video (and text) rows alone while the soundtrack stays
    on the base model's path; the control stream itself still attends over the full packed sequence either way, so
    its layout, AdaLN rows and rotary coordinates keep matching the main branch.

    `control_rows=None` delegates to the base forward, so the class is a drop-in replacement everywhere the base
    model is used.

    Both loading entry points initialise the control branch themselves, so no caller has to remember
    `materialize_missing_control_params`.
    """

    # The control model swaps subclasses into the main block stack and diffusers matches these entries against
    # `module.__class__.__name__`: without the subclass names `device_map` sharding and layerwise casting would find
    # no splittable / repeated block at all. FSDP is handed the same classes through
    # `--fsdp_transformer_layer_cls_to_wrap` in the launch scripts.
    _no_split_modules = [
        "BaseMiniMaxH3TransformerBlock",
        "MiniMaxH3ControlTransformerBlock",
        "MiniMaxH3TokenRefinerBlock",
        "MiniMaxH3AdaLayerNormOut",
    ]
    _repeated_blocks = [
        "BaseMiniMaxH3TransformerBlock",
        "MiniMaxH3ControlTransformerBlock",
        "MiniMaxH3TokenRefinerBlock",
    ]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        control_blocks_places=(0, 10, 20, 30, 40),
        control_in_dim=None,
        control_apply_audio: bool = True,
    ) -> None:
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=ffn_dim,
            in_channels=in_channels,
            audio_in_channels=audio_in_channels,
            patch_size=patch_size,
            text_dim=text_dim,
            freq_dim=freq_dim,
            time_embed_hidden_dim=time_embed_hidden_dim,
            time_embed_dim=time_embed_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )
        places = range(0, num_layers, 2) if control_blocks_places is None else control_blocks_places
        # Sorted and deduplicated: `block_id` addresses the position in this list rather than the layer index, so the
        # n-th control block feeds the n-th addressed main layer in depth order.
        self.control_blocks_places = sorted({int(i) for i in places})
        if not self.control_blocks_places or self.control_blocks_places[0] != 0:
            raise ValueError(
                "`control_blocks_places` must start the injection at layer 0, so that the control stream is re-based "
                f"on the main branch's input embeddings, got {self.control_blocks_places}."
            )
        if self.control_blocks_places[-1] >= num_layers:
            raise ValueError(
                f"`control_blocks_places` addresses layer {self.control_blocks_places[-1]}, but the model only has "
                f"{num_layers} layers."
            )
        self.control_in_dim = in_channels if control_in_dim is None else control_in_dim
        self.control_blocks_mapping = {i: n for n, i in enumerate(self.control_blocks_places)}
        # Registered through `register_to_config`, so the YAML `transformer_additional_kwargs` override and the
        # checkpoint's config.json carry it between training and inference; old checkpoints lack the key and fall
        # back to the default, keeping the skips applied to every row as before.
        self.control_apply_audio = control_apply_audio

        # Rebuild the main block stack with the hint-injecting blocks; the state-dict layout is unchanged, so the
        # released checkpoint loads as-is.
        del self.transformer_blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BaseMiniMaxH3TransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    block_id=self.control_blocks_mapping[i] if i in self.control_blocks_places else None,
                )
                for i in range(num_layers)
            ]
        )

        # Control branch: one control block per addressed main layer, plus the control patch projection. The branch
        # starts at identity through the zero-initialised `before_proj` / `after_proj` of its blocks, so neither
        # `control_proj_in` nor the blocks themselves are zeroed — see `materialize_missing_control_params`, where a
        # second zero gate in front of the control latents would only cost the projection its gradient and an
        # all-zero block would have no gradient at all. `control_proj_in` matches the `proj_in` substring of
        # `_keep_in_fp32_modules`, mirroring the mixed-precision contract of the video patch projections.
        self.control_blocks = nn.ModuleList(
            [
                MiniMaxH3ControlTransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    block_id=n,
                )
                for n, i in enumerate(self.control_blocks_places)
            ]
        )
        control_patch_dim = self.control_in_dim * patch_size[0] * patch_size[1] * patch_size[2]
        self.control_proj_in = nn.Linear(control_patch_dim, hidden_size, bias=True)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        r"""
        Load like `ModelMixin.from_pretrained` and initialise the control branch the checkpoint does not carry.

        Doing it here rather than in the training scripts keeps every entry point safe — EMA copies, the
        `load_model_hook` of `accelerate` and the predict scripts all go through `from_pretrained` — and mirrors how
        `WanTransformer3DModel.from_pretrained` fills its missing VACE keys. `materialize_missing_control_params`
        stays public and idempotent for callers that assemble the model by hand.
        """
        model = super().from_pretrained(*args, **kwargs)

        # Only initialise when the meta parameters are exactly the control branch. `device_map` / disk offloading
        # also leaves base parameters on meta behind an `accelerate` hook, and those must keep their hook rather
        # than be replaced by freshly allocated tensors; an explicit `materialize_missing_control_params` call still
        # reports that case.
        meta_keys = [name for name, param in model.named_parameters() if param.is_meta]
        meta_keys += [name for name, buf in model.named_buffers() if buf.is_meta]
        if meta_keys and all(key.startswith("control_") for key in meta_keys):
            model.materialize_missing_control_params()
        return model

    @classmethod
    def from_pretrained_original(cls, checkpoint_path, torch_dtype=None, device=None):
        r"""
        Load an original MiniMax-H3 partition and initialise the control branch it does not carry.

        The base implementation would `Module.to(device)` while the control branch is still on the meta device, which
        raises "Cannot copy out of meta tensor", so the branch is initialised before the model is moved.
        """
        model = super().from_pretrained_original(checkpoint_path, torch_dtype=torch_dtype, device=None)
        model.materialize_missing_control_params()
        if device is not None:
            model.to(device)
        return model.eval()

    def _reset_control_proj_in(self) -> None:
        r"""
        Draw `nn.Linear`'s default init for `control_proj_in` off a fixed seed.

        The draw has to be identical on every rank — `accelerate` prepares FSDP with `sync_module_states=False`, so
        nothing broadcasts rank 0's parameters — and across the three transformers of `train_control_distill.py`,
        which materialize the same checkpoint one after the other.
        """
        weight = self.control_proj_in.weight
        generator = torch.Generator(device=weight.device).manual_seed(CONTROL_PROJ_IN_INIT_SEED)
        nn.init.kaiming_uniform_(weight, a=5 ** 0.5, generator=generator)
        bound = weight.shape[1] ** -0.5
        nn.init.uniform_(self.control_proj_in.bias, -bound, bound, generator=generator)

    def materialize_missing_control_params(self, device=None):
        r"""
        Materialize and initialize the control-branch parameters that are absent from the checkpoint.

        Called by `from_pretrained` / `from_pretrained_original` already, so a caller only needs it when it builds
        the model by hand; calling it again is a no-op, since nothing is left on the meta device.

        Under `low_cpu_mem_usage=True` `from_pretrained` builds the model on the meta device and only fills the
        keys present in the checkpoint; the released MiniMax-H3 weights have no control-branch entries, so those
        parameters stay on meta (a later `Module.to` then raises "Cannot copy out of meta tensor").

        Every materialized control block is then initialized from the main block it is attached to, which is what
        VACE does. Leaving a block at the zeros it was materialized with is not an option: with a zero AdaLN
        projection the gate multiplies the attention / feed-forward gradient to zero, and with zero attention and
        feed-forward weights their output multiplies the gate gradient to zero, so *every* parameter inside the
        block has exactly zero gradient and stays dead for the whole run — the branch would collapse to
        `after_proj(before_proj(c) + x)`, a per-row linear map with no attention and no feed-forward.
        `before_proj` / `after_proj` keep their zeros, so the branch still starts at identity and a freshly loaded
        control model behaves exactly like the base model. The copy involves no RNG, so it is identical on every
        rank — `accelerate` prepares FSDP with `sync_module_states=False`, so nothing broadcasts rank 0's
        parameters — and across the three transformers of `train_control_distill.py`.

        The materialized tensors take the loaded block stack's dtype rather than the meta parameter's: the meta
        build of `low_cpu_mem_usage` leaves every parameter at the default float32 and `from_pretrained` only
        downcasts the keys the checkpoint carries, so the control branch would otherwise stay float32 — the copy
        below would pull the bf16 main blocks up to float32 (`load_state_dict` casts into the destination
        parameters) and `before_proj` would reject the bf16 packed sequence with a dtype error at inference.

        `control_proj_in` is initialized from `proj_in` when the control latents are patchified like the video
        latents, and off a fixed seed otherwise (see `_reset_control_proj_in`); zeros there would only gate the
        control latents behind a second zero layer. Either way it ends up at `proj_in`'s dtype, since it matches the
        `proj_in` substring of `_keep_in_fp32_modules`.

        Blocks the checkpoint did provide are left untouched, so resuming a trained control branch never overwrites
        it.

        Args:
            device (`str` or `torch.device`, *optional*):
                Where to allocate the materialized parameters. `None` follows the weights the checkpoint did load,
                so the branch never ends up on another device than the rest of the model.
        """
        if device is None:
            loaded = next((param for param in self.parameters() if not param.is_meta), None)
            device = "cpu" if loaded is None else loaded.device
        # The block stack is outside the `_keep_in_fp32_modules` exceptions, so its dtype is the requested
        # `torch_dtype`; `None` when nothing loaded yet (a hand-built meta model) and each tensor keeps its own.
        dtype = next((param.dtype for param in self.transformer_blocks.parameters() if not param.is_meta), None)
        materialized = {}
        for module_name, module in self.named_modules():
            names = []
            for name, param in list(module.named_parameters(recurse=False)):
                if param.is_meta:
                    setattr(module, name, nn.Parameter(
                        torch.zeros(param.shape, dtype=dtype if dtype is not None else param.dtype, device=device),
                        requires_grad=param.requires_grad,
                    ))
                    names.append(name)
            for name, buf in list(module.named_buffers(recurse=False)):
                if buf.is_meta:
                    setattr(module, name, torch.zeros(buf.shape, dtype=dtype if dtype is not None else buf.dtype, device=device))
                    names.append(name)
            if names:
                materialized[module_name] = names

        # Only the control branch is expected to be missing from the checkpoint; zeroing a base weight instead would
        # silently produce a broken model, so report it rather than hide it.
        base_keys = sorted(
            key
            for module_name, names in materialized.items()
            for key in (f"{module_name}.{name}" if module_name else name for name in names)
            if not key.startswith("control_")
        )
        if base_keys:
            raise ValueError(
                "The base weights did not load correctly: the checkpoint left non-control parameters on the meta "
                f"device: {base_keys}."
            )

        copied_blocks = []
        for n, place in enumerate(self.control_blocks_places):
            prefix = f"control_blocks.{n}"
            if not any(name == prefix or name.startswith(f"{prefix}.") for name in materialized):
                continue
            # The main block carries no `before_proj` / `after_proj`, so `strict=False` leaves those two at zero.
            incompatible = self.control_blocks[n].load_state_dict(
                self.transformer_blocks[place].state_dict(), strict=False
            )
            if incompatible.unexpected_keys:
                raise ValueError(
                    f"`transformer_blocks[{place}]` carries keys `control_blocks[{n}]` does not have: "
                    f"{sorted(incompatible.unexpected_keys)}."
                )
            copied_blocks.append((n, place))

        control_proj_in_init = None
        if "control_proj_in" in materialized:
            # `control_proj_in` matches the `proj_in` substring of `_keep_in_fp32_modules`, but the meta build gave it
            # the requested `torch_dtype` and `restore_fp32_modules` only re-reads keys the checkpoint carries, so
            # align it with the video patch projection by hand.
            self.control_proj_in.to(self.proj_in.weight.dtype)
            if self.control_proj_in.weight.shape == self.proj_in.weight.shape:
                self.control_proj_in.load_state_dict(self.proj_in.state_dict())
                control_proj_in_init = "the weights of `proj_in`"
            else:
                self._reset_control_proj_in()
                control_proj_in_init = "`nn.Linear`'s default init off a fixed seed"

        if materialized:
            num_materialized = sum(len(names) for names in materialized.values())
            print(f"Materialized {num_materialized} meta parameters left by the base checkpoint.")
        if copied_blocks:
            print(
                "Initialized the control blocks from their main blocks (`before_proj` / `after_proj` stay zero): "
                + ", ".join(f"control_blocks.{n} <- transformer_blocks.{place}" for n, place in copied_blocks)
                + "."
            )
        if control_proj_in_init is not None:
            print(f"Initialized `control_proj_in` with {control_proj_in_init}.")
        return self

    def enable_multi_gpus_inference(self):
        r"""
        Enable multi-GPU inference on both block stacks.

        The base implementation sets up the sequence-parallel state and hands the main blocks the multi-GPU
        attention processor; the control blocks attend over the same packed sequence, so they are handed the
        same processor.
        """
        super().enable_multi_gpus_inference()
        for block in self.control_blocks:
            block.attn.set_processor(MiniMaxH3MultiGPUsAttnProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        control_rows: Optional[torch.Tensor] = None,
        control_context_scale: float = 1.0,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[MiniMaxH3TransformerOutput, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Same contract as `MiniMaxH3Transformer3DModel.forward`, plus:

        Args:
            control_rows (`torch.Tensor` of shape `(batch_size, num_video_tokens, in_channels * prod(patch_size))`,
                *optional*):
                Patchified clean control latents, one row per video row of `hidden_states` (same order, so
                conditioning rows need a control row too). `None` disables the control branch and runs the base
                forward.
            control_context_scale (`float`, defaults to `1.0`):
                Scale applied to every control skip before it is added to the main branch.
        """
        if control_rows is None:
            return super().forward(
                hidden_states,
                audio_hidden_states,
                encoder_hidden_states,
                timestep,
                timestep_indices,
                token_tags,
                position_ids,
                video_indices,
                audio_indices,
                text_indices,
                attention_kwargs=attention_kwargs,
                return_dict=return_dict,
            )
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            from diffusers.utils import scale_lora_layers
            scale_lora_layers(self, lora_scale)

        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching `position_ids`, got "
                f"{list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}."
            )

        # 1. Modality projections and the packed sequence buffer, identical to the base forward.
        video_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
        text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        text_embeds = self.token_refiner(text_embeds)

        packed_hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        packed_hidden_states = packed_hidden_states.index_copy(1, text_indices, text_embeds)
        packed_hidden_states = packed_hidden_states.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        packed_hidden_states = packed_hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype)).to(packed_hidden_states.dtype)

        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)

        attention_mask = None
        is_pad = token_tags < 0
        if bool(is_pad.any()):
            attention_mask = is_pad[None, :] == is_pad[:, None]

        # Sequence parallel, mirroring the base forward: pad the packed sequence up to a multiple of the group
        # size and keep only this rank's rows. The control stream below is built over the same slice, so both
        # block stacks run on this rank's rows and the attention processor all-gathers the keys and values,
        # dropping the padding rows via `valid_length`.
        padded_length, valid_length = sequence_length, None
        if self.sp_world_size > 1:
            if attention_mask is not None:
                raise ValueError(
                    "Multi-GPU inference expects a padless packed sequence; drop the padding rows (tag -1) instead."
                )
            padded_length = sequence_length + (-sequence_length) % self.sp_world_size
            if padded_length != sequence_length:
                pad = padded_length - sequence_length
                packed_hidden_states = F.pad(packed_hidden_states, (0, 0, 0, pad))
                position_ids = F.pad(position_ids, (0, 0, 0, pad))
                adaln_indices = F.pad(adaln_indices, (0, pad))
            valid_length = sequence_length

        rotary_emb = self.rope(position_ids)

        if self.sp_world_size > 1:
            chunk_size = padded_length // self.sp_world_size
            rows = slice(self.sp_world_rank * chunk_size, (self.sp_world_rank + 1) * chunk_size)
            packed_hidden_states = packed_hidden_states[:, rows]
            adaln_indices = adaln_indices[rows]
            rotary_emb = (rotary_emb[0][rows], rotary_emb[1][rows])

        # 2. Control branch over the packed sequence (this rank's slice of it under sequence parallelism): the
        # control stream carries the control latents at the video rows and the main branch's own features at the
        # text and audio rows, mirroring the `[control_rows ; cap_feats]` stream of
        # `ZImageControlTransformerBlock`. It therefore shares the main branch's timestep embeddings, AdaLN table
        # rows, rotary coordinates and attention mask, and the first control block re-bases it on the main
        # branch's packed sequence via the zero-initialised `before_proj`.
        if control_rows.shape[1] != video_indices.shape[0]:
            raise ValueError(
                f"`control_rows` carries {control_rows.shape[1]} rows but the packed sequence has "
                f"{video_indices.shape[0]} video rows. One control row per video row of `hidden_states` is required, "
                "in the same order — a layout with conditioning rows (keyframes) needs those rows covered too."
            )
        control_embeds = self.control_proj_in(control_rows.to(self.control_proj_in.weight.dtype))
        if self.sp_world_size > 1:
            # `video_indices` addresses the unpadded packed sequence; this rank's slice holds the control rows of
            # the video rows inside it, re-based to the slice. The order is kept, so the row-for-row pairing of
            # control rows and video rows is preserved. `video_indices` itself is left alone: the output heads
            # still select from the gathered full sequence.
            local_video_rows = (video_indices >= rows.start) & (video_indices < rows.stop)
            control_embeds = control_embeds[:, local_video_rows]
            local_video_indices = video_indices[local_video_rows] - rows.start
        else:
            local_video_indices = video_indices
        c = packed_hidden_states.index_copy(1, local_video_indices, control_embeds.to(text_embeds.dtype))

        for layer in self.control_blocks:
            with torch.autograd.graph.save_on_cpu() if self.gradient_checkpointing_save_on_cpu else nullcontext():
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    c = torch.utils.checkpoint.checkpoint(
                        layer,
                        c,
                        packed_hidden_states,
                        temb,
                        adaln_indices,
                        rotary_emb,
                        attention_mask,
                        valid_length,
                        **ckpt_kwargs,
                    )
                else:
                    c = layer(c, packed_hidden_states, temb, adaln_indices, rotary_emb, attention_mask, valid_length)

        # 3. The skips line up row-for-row with the main stack's hidden states (this rank's slice of the sequence
        # under sequence parallelism), so the main blocks add them as they are. Under
        # `VIDEOX_OFFLOAD_VACE_LATENTS` they live on CPU and each main block moves its hint on-device at injection
        # time.
        hints = torch.unbind(c)[:-1]

        # `control_apply_audio=False` keeps the skips off the audio rows: zero them out of every hint before
        # injection. The mask spans the padded sequence and is sliced like `adaln_indices` under sequence
        # parallelism; pad rows stay unmasked, which is harmless since `valid_length` drops them. The per-hint
        # `.to` keeps `VIDEOX_OFFLOAD_VACE_LATENTS` runs working, where the hints sit on CPU until a main block
        # moves its hint on-device at injection time.
        if not self.control_apply_audio and audio_indices.numel():
            keep = torch.ones(padded_length, dtype=hints[0].dtype, device=audio_indices.device)
            keep[audio_indices] = 0
            if self.sp_world_size > 1:
                keep = keep[rows]
            hints = tuple(h * keep[None, :, None].to(device=h.device, dtype=h.dtype) for h in hints)

        # 4. Main block stack with the control skips injected at the addressed layers.
        hidden_states = packed_hidden_states
        for block in self.transformer_blocks:
            with torch.autograd.graph.save_on_cpu() if self.gradient_checkpointing_save_on_cpu else nullcontext():
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        block,
                        hidden_states,
                        temb,
                        adaln_indices,
                        rotary_emb,
                        attention_mask,
                        valid_length,
                        hints,
                        control_context_scale,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = block(
                        hidden_states, temb, adaln_indices, rotary_emb, attention_mask, valid_length,
                        hints=hints, context_scale=control_context_scale,
                    )

        # Sequence parallel: gather the full sequence back, as the base forward does, and drop the padding rows.
        if self.sp_world_size > 1:
            hidden_states = self.all_gather(hidden_states.contiguous(), dim=1)[:, :sequence_length]

        # 5. Output heads, identical to the base forward.
        hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(self.proj_out.weight.dtype)
        video_output = self.proj_out(hidden_states).index_select(1, video_indices)
        audio_output = self.audio_proj_out(hidden_states).index_select(1, audio_indices)

        if USE_PEFT_BACKEND:
            from diffusers.utils import unscale_lora_layers
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (video_output, audio_output)
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)
