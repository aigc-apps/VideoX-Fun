r"""
Shared mapping and loading contract for MiniMax-H3 checkpoints.

This module is the single source of truth for the conversion: both the offline converter
(`scripts/minimax_h3/convert_minimax_h3_to_diffusers.py`) and the load-time converters
(`from_pretrained_original` of the three model classes, `MiniMaxH3Pipeline.from_pretrained_original`)
import the same constants and functions, so the two paths cannot drift apart.

The module also hosts ``MiniMaxH3MixedPrecisionLoaderMixin``, the loading base of the three MiniMax-H3 model
classes. The mixin auto-detects the checkpoint layout (original vs. diffusers) and, for a diffusers layout, restores
the ``_keep_in_fp32_modules`` tensors the diffusers loader would otherwise round down when ``torch_dtype`` is not
float32. A diffusers-layout checkpoint that misses the keys a subclass adds on top of it (the control branch, the
multiview modules) is loaded shard-wise into a meta build through ``load_model_dict_into_meta`` instead of through
``ModelMixin.from_pretrained``, which every diffusers version tolerates; the missing parameters stay on the meta
device for the subclass's ``from_pretrained`` to materialise.

The differences between the two formats are

* the transformer keys are named after the reference's sglang-native modules, so every key is renamed
  (`convert_transformer_key`),
* the fused QKV projection is stored per-head interleaved; the shard reader first reorders the rows into the
  reference's in-memory `[q_all; k_all; v_all]` layout (`reorder_interleaved_qkv`, mirroring the reference's
  load-time transform) and then splits contiguous thirds (`split_fused_qkv`),
* the gated FFN's fused `mlp.fc1` becomes `ff.net.0.proj` with its two halves swapped, because diffusers' `SwiGLU`
  reads `[value; gate]` where the reference stores `[gate; value]` (the video VAE's `ff.w1` needs the same swap),
* `rope.inv_freq` is dropped: it is a pure function of `rope_theta` and `rope_freq_dim` and the port recomputes it
  in a non-persistent buffer,
* the video VAE weights live one level deeper (`video_vae/source/...`) and the scheduler shifts hide in the root
  `model_index.json._minimax_h3.sigma_shift_scales`.

There are no transposes anywhere.
"""

import glob
import json
import os
import struct

import torch
from diffusers.utils import logging
from safetensors import safe_open

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ── Mixed-precision loader mixin ─────────────────────────────────────────────


class MiniMaxH3MixedPrecisionLoaderMixin:
    r"""
    Loading contract of the mixed-precision MiniMax-H3 checkpoints.

    MiniMax-H3 pins part of every model in float32 through `_keep_in_fp32_modules`: the transformer's two patch
    projections, its timestep MLP and its two output heads, and — since the decode recipe is float16 *autocast over
    float32 weights* — the whole of both autoencoders. `diffusers` only honours `_keep_in_fp32_modules` when the
    requested dtype is float16, so a `torch_dtype=torch.bfloat16` load would round those weights down.

    This mixin restores them: the model is loaded with the requested dtype as usual, and the pinned tensors are then
    read back from the checkpoint at float32. Only checkpoints that live in a local directory can be read back that
    way; a repository id is loaded as usual and warned about.

    The mixin also auto-detects the checkpoint layout: a path that points at an *original* MiniMax-H3 partition
    (e.g. `MiniMax-H3/FL2VA`) is stream-converted through ``from_pretrained_original`` on the fly, so the caller never
    has to branch on ``is_raw_minimax_h3_format`` itself. A diffusers layout or a repository id falls through to the
    standard ``from_pretrained``.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, **kwargs):
        # Auto-detect an original MiniMax-H3 checkpoint layout and stream-convert it on the fly, so the caller never
        # has to branch on the format itself.
        if pretrained_model_name_or_path is not None:
            if is_raw_minimax_h3_format(pretrained_model_name_or_path):
                torch_dtype = kwargs.pop("torch_dtype", kwargs.pop("dtype", None))
                # diffusers-only kwargs that the original-format loader does not use.
                kwargs.pop("subfolder", None)
                kwargs.pop("low_cpu_mem_usage", None)
                return cls.from_pretrained_original(
                    pretrained_model_name_or_path, torch_dtype=torch_dtype, **kwargs
                )

        torch_dtype = kwargs.pop("torch_dtype", None)
        dtype = kwargs.pop("dtype", None)
        torch_dtype = torch_dtype if torch_dtype is not None else dtype
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
        # The missing-keys loader below cannot hand a partially-meta model to accelerate's dispatcher, so it pops
        # `device_map` instead of honouring it; a subclass that materialises the missing parameters re-applies the
        # device map afterwards (see `MiniMaxH3MultiViewsTransformer3DModel.from_pretrained`).
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)

        # A checkpoint that misses the keys this class adds on top of it (the control branch, the multiview
        # modules) cannot go through `ModelMixin.from_pretrained` under `low_cpu_mem_usage`: diffusers before 0.33
        # feeds the sharded layout to `accelerate.load_checkpoint_and_dispatch`, which moves the model while the
        # missing parameters are still on the meta device ("Cannot copy out of meta tensor"), and the non-sharded
        # layout raises on the missing keys outright. Load the shards into a meta build ourselves instead; whatever
        # the checkpoint does not carry stays on meta for the class's `from_pretrained` to materialise.
        if low_cpu_mem_usage is not False:
            model = cls._from_pretrained_with_missing_keys(pretrained_model_name_or_path, torch_dtype, **kwargs)
            if model is not None:
                if torch_dtype not in (None, torch.float32) and cls._keep_in_fp32_modules:
                    restore_fp32_modules(model, pretrained_model_name_or_path, kwargs.get("subfolder", None))
                if device_map is not None:
                    model._pending_device_map = device_map
                    model._pending_max_memory = max_memory
                return model

        if low_cpu_mem_usage is not None:
            kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        if device_map is not None:
            kwargs["device_map"] = device_map
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
        model = super().from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype, **kwargs)
        if torch_dtype not in (None, torch.float32) and cls._keep_in_fp32_modules:
            restore_fp32_modules(model, pretrained_model_name_or_path, kwargs.get("subfolder", None))
        return model

    @classmethod
    def _from_pretrained_with_missing_keys(cls, pretrained_model_name_or_path, torch_dtype, **kwargs):
        r"""
        Load a local diffusers-layout checkpoint the model outgrows — e.g. the released MiniMax-H3 weights loaded
        into the control or multiview transformer — or return `None` when the checkpoint carries every key of the
        model, or is not a local safetensors directory, so the standard `ModelMixin.from_pretrained` takes over.

        Mirrors the manual loading of `WanTransformer3DModel.from_pretrained`: the model is built on the meta
        device and the checkpoint's shards are written into it one by one through `load_model_dict_into_meta`, the
        one load primitive every diffusers version offers, without dispatching the model while parameters are still
        on meta. Parameters the checkpoint does not carry stay on the meta device for the class's `from_pretrained`
        to materialise.
        """
        if pretrained_model_name_or_path is None:
            return None
        subfolder = kwargs.pop("subfolder", None)
        directory = (
            os.path.join(str(pretrained_model_name_or_path), subfolder) if subfolder
            else str(pretrained_model_name_or_path)
        )
        if not os.path.isdir(directory):
            return None
        checkpoint_keys = _read_safetensors_keys(directory)
        if checkpoint_keys is None:
            return None

        import accelerate
        from diffusers import __version__ as diffusers_version
        from packaging import version as pkg_version
        is_new_load_utils = pkg_version.parse(diffusers_version) >= pkg_version.parse("0.33.0")
        if is_new_load_utils:
            # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
            # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
            from diffusers.models.model_loading_utils import load_model_dict_into_meta
        else:
            from diffusers.models.modeling_utils import load_model_dict_into_meta

        # Loading-only kwargs of `ModelMixin.from_pretrained` that `load_config` / `from_config` must not see.
        for key in (
            "variant", "use_safetensors", "device_map", "max_memory", "offload_folder", "offload_state_dict",
            "cache_dir", "force_download", "proxies", "local_files_only", "token", "revision",
            "output_loading_info", "quantization_config", "dduf_entries", "disable_mmap",
        ):
            kwargs.pop(key, None)
        config, unused_kwargs = cls.load_config(
            pretrained_model_name_or_path, subfolder=subfolder, return_unused_kwargs=True, **kwargs
        )
        with accelerate.init_empty_weights():
            model = cls.from_config(config, **unused_kwargs)
        model_state_dict = model.state_dict()
        missing_keys = [key for key in model_state_dict if key not in checkpoint_keys]
        if not missing_keys:
            return None
        logger.info(
            f"The checkpoint at {directory} misses {len(missing_keys)} key(s) of {cls.__name__}; loading the "
            "shards through `load_model_dict_into_meta` and leaving those parameters on the meta device for the "
            "class's `from_pretrained` to materialise."
        )

        for file in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
            with safe_open(file, framework="pt") as reader:
                state_dict = {}
                for key in reader.keys():
                    if key not in model_state_dict:
                        continue
                    if model_state_dict[key].shape != torch.Size(reader.get_slice(key).get_shape()):
                        logger.warning(f"Skipping key '{key}' of {file}: shape mismatch with the model.")
                        continue
                    state_dict[key] = reader.get_tensor(key)
            if not state_dict:
                continue
            if is_new_load_utils:
                load_model_dict_into_meta(
                    model, state_dict, dtype=torch_dtype, model_name_or_path=pretrained_model_name_or_path
                )
            else:
                load_model_dict_into_meta(
                    model, state_dict, device="cpu", dtype=torch_dtype,
                    model_name_or_path=pretrained_model_name_or_path,
                )

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.eval()
        return model


def _read_safetensors_keys(directory):
    r"""The keys of every safetensors shard in `directory`, or `None` when it holds none."""
    files = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
    if not files:
        return None
    keys = set()
    for file in files:
        with safe_open(file, framework="pt") as reader:
            keys.update(reader.keys())
    return keys


def _assign_tensor(model: torch.nn.Module, key: str, tensor: torch.Tensor) -> None:
    r"""Replace the parameter or the buffer `key` addresses, keeping `requires_grad`."""
    *path, attribute = key.split(".")
    module = model
    for name in path:
        module = getattr(module, name)
    existing = getattr(module, attribute)
    if attribute in module._buffers:
        module._buffers[attribute] = tensor
    else:
        module._parameters[attribute] = torch.nn.Parameter(tensor, requires_grad=existing.requires_grad)


def restore_fp32_modules(model: torch.nn.Module, pretrained_model_name_or_path, subfolder=None) -> None:
    r"""
    Re-read the `_keep_in_fp32_modules` tensors of `model` from its checkpoint at float32.

    Args:
        model (`torch.nn.Module`): The freshly loaded model, already cast to the requested dtype.
        pretrained_model_name_or_path (`str` or `os.PathLike`): Where the model was loaded from.
        subfolder (`str`, *optional*): The subfolder it was loaded from.
    """
    patterns = model._keep_in_fp32_modules or []
    directory = str(pretrained_model_name_or_path)
    if subfolder:
        directory = os.path.join(directory, subfolder)
    if not os.path.isdir(directory):
        logger.warning(
            f"MiniMax-H3 keeps {patterns} in float32, but {directory} is not a local directory, so those weights "
            "cannot be read back at float32 and stay at the requested dtype. Download the checkpoint first to keep "
            "the mixed precision of the released weights."
        )
        return

    files = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
    if not files:
        logger.warning(
            f"MiniMax-H3 keeps {patterns} in float32, but no safetensors file was found in {directory}, so those "
            "weights stay at the requested dtype."
        )
        return

    state_dict = model.state_dict()
    restored = 0
    for file in files:
        with safe_open(file, framework="pt") as reader:
            for key in reader.keys():
                if key not in state_dict or not any(pattern in key for pattern in patterns):
                    continue
                tensor = reader.get_tensor(key)
                if not torch.is_floating_point(tensor):
                    continue
                _assign_tensor(model, key, tensor.to(device=state_dict[key].device, dtype=torch.float32))
                restored += 1
    logger.info(f"Restored {restored} float32 tensors of {type(model).__name__} matching {patterns}.")


# ── Conversion constants ─────────────────────────────────────────────────────

SAFE_WEIGHTS_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
DIFFUSERS_VERSION = "0.32.2"

# `MiniMaxH3Transformer3DModel` argument names. The original config uses the sglang-native names listed in the
# comments; everything else in the original config (`adaln_out_features`, `final_adaln_out_features`) is derived.
MINIMAX_H3_TRANSFORMER_CONFIG = {
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "hidden_size": 5376,
    "num_layers": 50,
    "num_refiner_layers": 2,  # token_refiner_num_layers
    "ffn_dim": 14336,  # ffn_hidden_size
    "in_channels": 24,  # latents_dim
    "audio_in_channels": 32,  # audio_latents_dim
    "patch_size": [1, 2, 2],
    "text_dim": 5120,
    "freq_dim": 256,  # timestep_input_dim
    "time_embed_hidden_dim": 5376,  # time_embed_hidden_size
    "time_embed_dim": 2688,
    "rope_freq_dim": 16,  # rope_inv_freq_len
    "rope_theta": 10000.0,
    "norm_eps": 1e-05,
    "qk_norm_eps": 1e-05,
    "final_norm_eps": 1e-05,
}

# MiniMax-H3 ships a mixed-precision checkpoint. These *original* keys are float32; everything else is bfloat16 —
# including the AdaLN projections.
MINIMAX_H3_FP32_SOURCE_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
)

# `rope.inv_freq` is recomputed by `MiniMaxH3RotaryPosEmbed` into a non-persistent buffer.
MINIMAX_H3_TRANSFORMER_DROPPED_KEYS = ("rope.inv_freq",)

# `AutoencoderKLMiniMaxH3` argument names, field-for-field equal to `video_vae/source/config.json`.
MINIMAX_H3_VIDEO_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,  # out_ch
    "latent_channels": 24,  # z_channels == embed_dim
    "block_out_channels": [128, 256, 256, 512, 512, 1024],  # ch * ch_mult
    "layers_per_block": 2,  # num_res_blocks
    "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],  # space_down
    "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],  # time_down
    "norm_num_groups": 32,
    "norm_eps": 1e-06,
    "spatial_padding_mode": "reflect",  # padding_mode
    "decoder_num_layers": 36,  # vit_decoder_kwargs.num_layers
    "decoder_num_attention_heads": 32,  # vit_decoder_kwargs.heads
    "decoder_attention_head_dim": 64,  # vit_decoder_kwargs.dim_head
    "decoder_num_register_tokens": 4,  # ViT3DDecoder default
    "decoder_ffn_mult": 4,  # FeedForward default
    "decoder_rope_theta": 100.0,  # vit_decoder_kwargs.rope_theta
    "decoder_rope_dim_ratio": 0.75,  # vit_decoder_kwargs.rope_dim_ratio
    "decoder_norm_eps": 1e-05,  # ViT3DDecoder eps
    "clip_length": 17,  # video_vae/config.json vae_clip_length
    "token_drop": 3,  # video_vae/config.json vae_token_drop
}

# `decoder.mask_token` is an all-zero buffer of the masked-autoencoding training objective; the released decoder
# never reads it, so the port does not carry the module and the conversion drops the key.
MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS = ("decoder.mask_token",)

# Not present in `audio_vae/metadata.json`: the reference implementation hardcodes these in its DAC audio VAE and
# its attention projection, keyed off the sample rate.
MINIMAX_H3_AUDIO_VAE_FIXED_CONFIG = {
    "num_attention_heads": 8,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
}

# The conditioner folders, which are already in the HuggingFace layout.
MINIMAX_H3_CONDITIONER_FOLDERS = ("text_encoder", "tokenizer", "processor")

# `MiniMaxH3Pipeline.from_pretrained` component specs. Everything lives in `videox_fun`; the conditioner classes are
# the patched ones re-exported from `videox_fun.models`, not stock transformers.
MINIMAX_H3_MODEL_INDEX = {
    "_class_name": "MiniMaxH3Pipeline",
    "_diffusers_version": DIFFUSERS_VERSION,
    "audio_scheduler": ["videox_fun.utils", "MiniMaxH3Scheduler"],
    "audio_vae": ["videox_fun.models", "AutoencoderKLMiniMaxH3Audio"],
    "processor": ["videox_fun.models", "Qwen3VLProcessor"],
    "scheduler": ["videox_fun.utils", "MiniMaxH3Scheduler"],
    "text_encoder": ["videox_fun.models", "Qwen3VLForConditionalGeneration"],
    "tokenizer": ["videox_fun.models", "Qwen2TokenizerFast"],
    "transformer": ["videox_fun.models", "MiniMaxH3Transformer3DModel"],
    "vae": ["videox_fun.models", "AutoencoderKLMiniMaxH3"],
}


def reorder_interleaved_qkv(weight, num_attention_heads, attention_head_dim):
    """Reorder a *raw-checkpoint* per-head-interleaved fused QKV weight into `[q_all; k_all; v_all]`.

    The original checkpoint shards store rows as `[head0: q(head_dim) k(head_dim) v(head_dim), head1: q, k, v, ...]`.
    The reference implementation applies exactly this reorder at load time, so `[q_all; k_all; v_all]` is the
    reference's in-memory / state-dict layout. There is no transpose.
    """
    expected_rows = num_attention_heads * 3 * attention_head_dim
    if weight.shape[0] != expected_rows:
        raise ValueError(
            f"fused qkv weight has {weight.shape[0]} rows, expected "
            f"{expected_rows} = {num_attention_heads} heads * 3 * {attention_head_dim}."
        )
    grouped = weight.reshape(num_attention_heads, 3 * attention_head_dim, *weight.shape[1:])
    query, key, value = grouped.split(attention_head_dim, dim=1)
    return torch.cat(
        [
            tensor.reshape(num_attention_heads * attention_head_dim, *weight.shape[1:])
            for tensor in (query, key, value)
        ],
        dim=0,
    )


def split_fused_qkv(weight, num_attention_heads, attention_head_dim):
    """Split a fused `[q_all; k_all; v_all]` QKV weight into separate `to_q` / `to_k` / `to_v` weights."""
    inner_dim = num_attention_heads * attention_head_dim
    if weight.shape[0] != 3 * inner_dim:
        raise ValueError(
            f"fused qkv weight has {weight.shape[0]} rows, expected "
            f"{3 * inner_dim} = 3 * {num_attention_heads} heads * {attention_head_dim}."
        )
    query, key, value = weight.split(inner_dim, dim=0)
    return tuple(tensor.contiguous() for tensor in (query, key, value))


def get_transformer_key_plan(config):
    """Map every original transformer key to the target key(s) it produces, with the resulting shapes."""
    hidden_size = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = config["attention_head_dim"]
    inner_dim = heads * head_dim
    ffn_dim = config["ffn_dim"]
    time_embed_dim = config["time_embed_dim"]
    video_patch_dim = (
        config["in_channels"] * config["patch_size"][0] * config["patch_size"][1] * config["patch_size"][2]
    )

    plan = {
        "video_patch_proj.weight": [("proj_in.weight", [hidden_size, video_patch_dim])],
        "video_patch_proj.bias": [("proj_in.bias", [hidden_size])],
        "audio_patch_proj.weight": [("audio_proj_in.weight", [hidden_size, config["audio_in_channels"]])],
        "audio_patch_proj.bias": [("audio_proj_in.bias", [hidden_size])],
        "condition_proj.weight": [("context_embedder.weight", [hidden_size, config["text_dim"]])],
        "condition_proj.bias": [("context_embedder.bias", [hidden_size])],
        # `Timesteps` + `TimestepEmbedding` reproduce the reference sinusoid and MLP exactly, so the timestep MLP is
        # renamed onto `TimestepEmbedding`'s `linear_1` / `linear_2`.
        "time_embedder.proj_in.weight": [
            ("time_embedder.linear_1.weight", [config["time_embed_hidden_dim"], config["freq_dim"]])
        ],
        "time_embedder.proj_in.bias": [("time_embedder.linear_1.bias", [config["time_embed_hidden_dim"]])],
        "time_embedder.proj_out.weight": [
            ("time_embedder.linear_2.weight", [time_embed_dim, config["time_embed_hidden_dim"]])
        ],
        "time_embedder.proj_out.bias": [("time_embedder.linear_2.bias", [time_embed_dim])],
        "token_refiner.final_norm.weight": [("token_refiner.final_norm.weight", [hidden_size])],
        "final_layer.norm.weight": [("norm_out.norm.weight", [hidden_size])],
        "final_layer.adaln_proj.linear.weight": [("norm_out.linear.weight", [2 * hidden_size, time_embed_dim])],
        "final_layer.adaln_proj.linear.bias": [("norm_out.linear.bias", [2 * hidden_size])],
        "final_layer.video_out.weight": [("proj_out.weight", [video_patch_dim, hidden_size])],
        "final_layer.video_out.bias": [("proj_out.bias", [video_patch_dim])],
        "final_layer.audio_out.weight": [("audio_proj_out.weight", [config["audio_in_channels"], hidden_size])],
        "final_layer.audio_out.bias": [("audio_proj_out.bias", [config["audio_in_channels"]])],
    }
    for key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
        plan[key] = []

    block_specs = [
        ("blocks", "transformer_blocks", config["num_layers"], True),
        ("token_refiner.blocks", "token_refiner.refiner_blocks", config["num_refiner_layers"], False),
    ]
    for source_prefix, target_prefix, num_layers, has_adaln in block_specs:
        for i in range(num_layers):
            source = f"{source_prefix}.{i}"
            target = f"{target_prefix}.{i}"
            plan[f"{source}.norm1.weight"] = [(f"{target}.norm1.weight", [hidden_size])]
            plan[f"{source}.norm2.weight"] = [(f"{target}.norm2.weight", [hidden_size])]
            plan[f"{source}.attn.qkv_proj.weight"] = [
                (f"{target}.attn.to_q.weight", [inner_dim, hidden_size]),
                (f"{target}.attn.to_k.weight", [inner_dim, hidden_size]),
                (f"{target}.attn.to_v.weight", [inner_dim, hidden_size]),
            ]
            plan[f"{source}.attn.q_norm.weight"] = [(f"{target}.attn.norm_q.weight", [head_dim])]
            plan[f"{source}.attn.k_norm.weight"] = [(f"{target}.attn.norm_k.weight", [head_dim])]
            plan[f"{source}.attn.out_proj.weight"] = [(f"{target}.attn.to_out.0.weight", [hidden_size, inner_dim])]
            # `fc1` stays fused, as diffusers' `SwiGLU` also fuses its two projections, but the halves are swapped
            # from `[gate; value]` to `[value; gate]`.
            plan[f"{source}.mlp.fc1.weight"] = [(f"{target}.ff.net.0.proj.weight", [2 * ffn_dim, hidden_size])]
            plan[f"{source}.mlp.fc2.weight"] = [(f"{target}.ff.net.2.weight", [hidden_size, ffn_dim])]
            if has_adaln:
                plan[f"{source}.adaln_proj.linear.weight"] = [
                    (f"{target}.adaln_proj.linear.weight", [6 * 3 * hidden_size, time_embed_dim])
                ]
                plan[f"{source}.adaln_proj.linear.bias"] = [
                    (f"{target}.adaln_proj.linear.bias", [6 * 3 * hidden_size])
                ]

    return plan


def convert_transformer_key(source_key, tensor, config):
    """Convert one original key/tensor pair into the target key/tensor pair(s) it maps to."""
    if source_key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
        return []

    target_key = source_key
    if target_key.startswith("token_refiner.blocks."):
        target_key = target_key.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif target_key.startswith("blocks."):
        target_key = target_key.replace("blocks.", "transformer_blocks.", 1)
    target_key = target_key.replace("time_embedder.proj_in.", "time_embedder.linear_1.")
    target_key = target_key.replace("time_embedder.proj_out.", "time_embedder.linear_2.")
    target_key = target_key.replace("video_patch_proj.", "proj_in.")
    target_key = target_key.replace("audio_patch_proj.", "audio_proj_in.")
    target_key = target_key.replace("condition_proj.", "context_embedder.")
    target_key = target_key.replace("final_layer.norm.", "norm_out.norm.")
    target_key = target_key.replace("final_layer.adaln_proj.linear.", "norm_out.linear.")
    target_key = target_key.replace("final_layer.video_out.", "proj_out.")
    target_key = target_key.replace("final_layer.audio_out.", "audio_proj_out.")
    target_key = target_key.replace(".attn.q_norm.", ".attn.norm_q.")
    target_key = target_key.replace(".attn.k_norm.", ".attn.norm_k.")
    target_key = target_key.replace(".attn.out_proj.", ".attn.to_out.0.")

    if target_key.endswith(".attn.qkv_proj.weight"):
        # The fused QKV rows are already `[q_all; k_all; v_all]` here: the shard streamer normalizes the raw
        # per-head interleave with `reorder_interleaved_qkv` before calling this.
        query, key, value = split_fused_qkv(tensor, config["num_attention_heads"], config["attention_head_dim"])
        prefix = target_key[: -len("qkv_proj.weight")]
        return [(f"{prefix}to_q.weight", query), (f"{prefix}to_k.weight", key), (f"{prefix}to_v.weight", value)]

    if target_key.endswith(".mlp.fc1.weight"):
        # The reference computes `fc2(silu(gate) * value)` from a fused `[gate; value]`; diffusers' `SwiGLU` computes
        # `value * silu(gate)` from a fused `[value; gate]`, so the two halves swap places.
        gate, value = tensor.chunk(2, dim=0)
        target_key = target_key.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight")
        return [(target_key, torch.cat([value, gate], dim=0).contiguous())]

    target_key = target_key.replace(".mlp.fc2.", ".ff.net.2.")
    return [(target_key, tensor)]


def rename_video_vae_key(source_key):
    """Rename one original video-VAE key onto its target module path (no tensor transform)."""
    target_key = source_key
    if target_key.startswith("encoder.down."):
        level, rest = target_key[len("encoder.down.") :].split(".", 1)
        rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        target_key = f"encoder.down_blocks.{level}.{rest}"
    target_key = target_key.replace("decoder.x_embedder.", "decoder.proj_in.")
    target_key = target_key.replace(".attn.to_out.", ".attn.to_out.0.")
    target_key = target_key.replace(".ff.w1.", ".ff.net.0.proj.")
    target_key = target_key.replace(".ff.w2.", ".ff.net.2.")
    return target_key


def convert_video_vae_key(source_key, tensor, config):
    """Convert one original video-VAE key/tensor pair into the target key/tensor pair(s) it maps to.

    `quant_conv` / `post_quant_conv`, the encoder's `conv_in` / `norm_out` / `conv_out` and the ViT decoder's
    `register_tokens` / `norm_out` / `proj_out` / `norm{1,2}` / `scale{1,2}` are pure pass-throughs. What moves:

    * the encoder's CNN levels are renamed from the original CompVis spelling onto the diffusers autoencoder idiom,
    * the ViT decoder's `x_embedder` becomes `proj_in`,
    * the fused per-head-interleaved `attn.to_qkv` is split into `attn.to_q` / `to_k` / `to_v`,
    * `attn.to_out` becomes `attn.to_out.0`,
    * `ff.w1` / `ff.w2` become `ff.net.0.proj` / `ff.net.2`, with the two halves of `w1` swapped.
    """
    if source_key in MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS:
        return []

    if ".attn.to_qkv." in source_key:
        # Same per-head interleave as the DiT: `[head0: q k v, head1: q k v, ...]`.
        reordered = reorder_interleaved_qkv(
            tensor, config["decoder_num_attention_heads"], config["decoder_attention_head_dim"]
        )
        query, key, value = split_fused_qkv(
            reordered, config["decoder_num_attention_heads"], config["decoder_attention_head_dim"]
        )
        prefix, suffix = source_key.split(".attn.to_qkv.")
        return [
            (f"{prefix}.attn.to_q.{suffix}", query),
            (f"{prefix}.attn.to_k.{suffix}", key),
            (f"{prefix}.attn.to_v.{suffix}", value),
        ]

    target_key = rename_video_vae_key(source_key)
    if ".ff.w1." in source_key:
        gate, up = tensor.chunk(2, dim=0)
        return [(target_key, torch.cat([up, gate], dim=0).contiguous())]
    return [(target_key, tensor)]


def get_audio_vae_config(checkpoint_path):
    """Build the `AutoencoderKLMiniMaxH3Audio` config from the original audio-VAE metadata.

    `audio_vae/metadata.json` carries the constructor kwargs the checkpoint was built with, and
    `audio_vae/config.json` carries the per-channel `latents_mean` / `latents_std`. The two are cross-checked here
    because they duplicate the latent width and the sample rate.
    """
    source_dir = os.path.join(checkpoint_path, "audio_vae")
    with open(os.path.join(source_dir, "metadata.json")) as f:
        kwargs = json.load(f)["metadata"]["kwargs"]
    with open(os.path.join(source_dir, "config.json")) as f:
        wrapper_config = json.load(f)

    if kwargs["decoder_type"] != "bigvgan":
        raise ValueError(f"Only the BigVGAN decoder is supported, got {kwargs['decoder_type']!r}.")
    if not kwargs["attn_proj"]:
        raise ValueError("The audio VAE is expected to carry the causal-attention latent projection.")
    latent_channels = kwargs["vae_latent_channels"]
    if wrapper_config["latent_channels"] != latent_channels:
        raise ValueError(
            f"latent width disagreement: metadata.json says {latent_channels}, "
            f"config.json says {wrapper_config['latent_channels']}."
        )
    if wrapper_config["sample_rate"] != kwargs["sample_rate"]:
        raise ValueError(
            f"sample rate disagreement: metadata.json says {kwargs['sample_rate']}, "
            f"config.json says {wrapper_config['sample_rate']}."
        )
    for key in ("latents_mean", "latents_std"):
        if len(wrapper_config[key]) != latent_channels:
            raise KeyError(f"{source_dir}/config.json `{key}` does not have {latent_channels} entries.")

    return {
        "encoder_dim": kwargs["encoder_dim"],
        "encoder_rates": kwargs["encoder_rates"],
        "latent_dim": kwargs["latent_dim"],
        "latent_channels": latent_channels,
        "decoder_dim": kwargs["decoder_dim"],
        "decoder_rates": kwargs["decoder_rates"],
        # The reference's two hardcoded BigVGAN tables (16 kHz and 32 kHz) both pair rate `u` with kernel `2u` for
        # even `u` and `2u - 1` for odd `u`, i.e. [5, 5, 2, ...] -> [9, 9, 4, ...].
        "decoder_kernel_sizes": [2 * rate - (rate % 2) for rate in kwargs["decoder_rates"]],
        **MINIMAX_H3_AUDIO_VAE_FIXED_CONFIG,
        # Renamed from the original `sample_rate` to the diffusers audio convention.
        "sampling_rate": kwargs["sample_rate"],
        "latents_mean": wrapper_config["latents_mean"],
        "latents_std": wrapper_config["latents_std"],
    }


def read_safetensors_header(path):
    """Read the metadata header of a safetensors file without touching the tensor payload."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return header


def is_raw_minimax_h3_format(path):
    r"""
    Tell whether `path` points at an *original* MiniMax-H3 checkpoint (the released layout) rather than at the
    diffusers layout `videox_fun` loads.

    The original root `model_index.json` carries a `_minimax_h3` block (the schedule constants) instead of component
    specs; as a fallback the first transformer shard header is sniffed for an original key.
    """
    if not os.path.isdir(path):
        return False
    index_path = os.path.join(path, "model_index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path) as f:
                if "_minimax_h3" in json.load(f):
                    return True
        except (OSError, json.JSONDecodeError):
            pass
    shards = sorted(glob.glob(os.path.join(path, "transformer", "*.safetensors")))
    if shards:
        try:
            header = read_safetensors_header(shards[0])
        except (OSError, struct.error, json.JSONDecodeError):
            return False
        return any(key.startswith("video_patch_proj.") or ".attn.qkv_proj." in key for key in header)
    return False


def read_original_sigma_shifts(checkpoint_path):
    """Read the per-modality sigma shifts of an original checkpoint, `{"video": ..., "audio": ...}`.

    The source `model_index.json` leaves `scheduler` null and instead carries the schedule constants in its
    `_minimax_h3.sigma_shift_scales` block.
    """
    with open(os.path.join(checkpoint_path, "model_index.json")) as f:
        return json.load(f)["_minimax_h3"]["sigma_shift_scales"]


def read_original_video_vae_wrapper_config(checkpoint_path):
    """Read `video_vae/config.json`, which carries `latents_mean` / `latents_std` and the source weights path."""
    source_dir = os.path.join(checkpoint_path, "video_vae")
    with open(os.path.join(source_dir, "config.json")) as f:
        wrapper_config = json.load(f)
    for key in ("latents_mean", "latents_std"):
        if key not in wrapper_config:
            raise KeyError(f"{source_dir}/config.json does not carry `{key}`.")
    return wrapper_config


def original_video_vae_weights_path(checkpoint_path):
    """Resolve the original video-VAE weights file, one level deeper than the rest of the checkpoint."""
    wrapper_config = read_original_video_vae_wrapper_config(checkpoint_path)
    return os.path.join(
        checkpoint_path, "video_vae", wrapper_config["source_path"], wrapper_config["source_safetensors_path"]
    )


def iter_original_transformer_tensors(checkpoint_path, config=None, torch_dtype=None):
    r"""
    Stream the original transformer shards and yield `(target_key, tensor)` pairs in the `videox_fun` naming.

    The shards are memory-mapped, so only the tensor being read is materialized and peak memory stays close to one
    shard. Tensors of the mixed-precision checkpoint keep their released dtype: the `MINIMAX_H3_FP32_SOURCE_PREFIXES`
    modules stay float32, everything else is cast to `torch_dtype` (bfloat16 as released when `None`).
    """
    config = config or MINIMAX_H3_TRANSFORMER_CONFIG
    plan = get_transformer_key_plan(config)
    transformer_dir = os.path.join(checkpoint_path, "transformer")
    shards = sorted(glob.glob(os.path.join(transformer_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No `*.safetensors` shards found under {transformer_dir}.")

    seen_source_keys = set()
    for shard in shards:
        # `safe_open` memory-maps the file, so only the tensor being read is materialized.
        with safe_open(shard, framework="pt", device="cpu") as f:
            for source_key in f.keys():
                if source_key not in plan:
                    raise KeyError(f"Unexpected key in {os.path.basename(shard)}: {source_key}")
                seen_source_keys.add(source_key)
                if source_key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
                    # Dropped keys (e.g. the recomputed `rope.inv_freq`) carry no dtype contract.
                    continue
                source_tensor = f.get_tensor(source_key)
                if source_key.endswith(".attn.qkv_proj.weight"):
                    source_tensor = reorder_interleaved_qkv(
                        source_tensor, config["num_attention_heads"], config["attention_head_dim"]
                    )
                expected_dtype = (
                    torch.float32 if source_key.startswith(MINIMAX_H3_FP32_SOURCE_PREFIXES) else torch.bfloat16
                )
                if source_tensor.dtype != expected_dtype:
                    raise ValueError(f"{source_key}: expected {expected_dtype}, got {source_tensor.dtype}.")
                keep_fp32 = source_key.startswith(MINIMAX_H3_FP32_SOURCE_PREFIXES)
                for target_key, tensor in convert_transformer_key(source_key, source_tensor, config):
                    if not keep_fp32 and torch_dtype is not None and tensor.is_floating_point():
                        tensor = tensor.to(torch_dtype)
                    yield target_key, tensor

    missing = sorted(set(plan) - seen_source_keys)
    if missing:
        raise KeyError(f"{len(missing)} planned key(s) missing from the checkpoint, e.g. {missing[:5]}.")


def iter_original_video_vae_tensors(checkpoint_path, config=None):
    r"""
    Stream the original video-VAE weights and yield `(target_key, tensor)` pairs in the `videox_fun` naming.

    The released weights are float32 and the decode recipe is float16 autocast over float32 weights, so the tensors
    keep their float32 dtype whatever the caller requests elsewhere.
    """
    config = config or MINIMAX_H3_VIDEO_VAE_CONFIG
    weights_path = original_video_vae_weights_path(checkpoint_path)
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for source_key in f.keys():
            for target_key, tensor in convert_video_vae_key(source_key, f.get_tensor(source_key), config):
                if tensor.dtype != torch.float32:
                    raise ValueError(f"{source_key}: expected torch.float32, got {tensor.dtype}.")
                yield target_key, tensor


def load_original_audio_vae_state_dict(checkpoint_path):
    r"""
    Read the original audio-VAE weights into a state dict of the `videox_fun` naming.

    The mapping is an identity: `AutoencoderKLMiniMaxH3Audio` reproduces the original module tree name for name,
    including `torch.nn.utils.weight_norm`'s `weight_g` / `weight_v` spelling and the Kaiser-window `filter` buffers
    of the anti-aliased activations. The audio VAE is small (~0.6 GiB), so it is read in one go.
    """
    weights_path = os.path.join(checkpoint_path, "audio_vae", "model.safetensors")
    state_dict = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.dtype != torch.float32:
                raise ValueError(f"{key}: expected torch.float32, got {tensor.dtype}.")
            state_dict[key] = tensor
    return state_dict


def assign_original_tensors(model, tensors):
    r"""
    Assign `(key, tensor)` pairs onto `model` in place and verify every model parameter / buffer was covered.

    The model is expected to have been built empty (e.g. on the meta device) by the caller, so assigning replaces the
    placeholder tensors one by one and frees them as it goes — the peak memory stays the model itself plus one
    streamed tensor, never a second copy of the weights.
    """
    expected_keys = set(model.state_dict().keys())
    assigned = set()
    for key, tensor in tensors:
        if key in assigned:
            raise KeyError(f"Duplicate target key while converting: {key}")
        if key not in expected_keys:
            raise KeyError(f"Converted key {key} does not address any parameter of {type(model).__name__}.")
        _assign_tensor(model, key, tensor)
        assigned.add(key)

    missing = sorted(expected_keys - assigned)
    if missing:
        raise KeyError(f"{len(missing)} parameter(s) of {type(model).__name__} left unassigned, e.g. {missing[:5]}.")
    return model
