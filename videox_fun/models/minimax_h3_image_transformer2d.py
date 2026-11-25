# Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_minimax_h3_image.py
# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

"""MiniMax-H3-Image denoiser.

MiniMax-H3-Image is MiniMax-H3 without the audio branch: the same single-stream block stack over one packed
sequence, the same `(t, h, w)` rotary grid, the same per-`(timestep, modality)` AdaLN table, and the image
checkpoint does not carry the two audio projections. Everything that is not image-specific — the blocks, the text
refiner, the rotary embedding, the AdaLN modules, the sequence-parallel attention processor — is reused from
`minimax_h3_transformer3d`.

The default output is the original single `final_layer.video_out` projection, exposed as `proj_out`, like the video
model. The released image checkpoint also supplies six PDD output heads; using them is an explicit choice. The
released image backbone remains guidance-distilled in either mode.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import (USE_PEFT_BACKEND, is_torch_version, logging,
                             scale_lora_layers, unscale_lora_layers)
from safetensors import safe_open

from ..dist import (MiniMaxH3MultiGPUsAttnProcessor, get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group)
from ..utils.lora_utils_pdd import PDDPreFusedHeads
from .minimax_h3_conversion import (MiniMaxH3MixedPrecisionLoaderMixin,
                                    assign_original_tensors, get_transformer_key_plan,
                                    iter_original_transformer_tensors)
from .minimax_h3_transformer3d import (MINIMAX_H3_MODALITY_NUM,
                                       MiniMaxH3AdaLayerNormOut,
                                       MiniMaxH3RotaryPosEmbed,
                                       MiniMaxH3TokenRefiner,
                                       MiniMaxH3TransformerBlock)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Original output-head and component helpers.
_ORIGINAL_KEYS = ("final_layer.video_out.weight", "final_layer.video_out.bias")
_HEAD_KEYS = ("weight", "bias")
_CONVERTED_KEYS = ("proj_out.weight", "proj_out.bias")
ORIGINAL_IMAGE_HEAD_FILE = "original_output_head.safetensors"


def is_original_image_transformer(directory):
    """Identify the released native transformer without opening large shards."""
    directory = Path(directory).expanduser()
    config_path = directory / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError, AttributeError):
        return False
    # Recognize the native layout even if its index is absent or damaged: the
    # native loader must report that error instead of silently choosing PDD.
    return config.get("_class_name") == "MiniMaxH3DiTModel" and config.get("audio_enabled") is False


def find_original_image_transformer(model_path, subfolder=None):
    """Return the native component directory if this request selects one."""
    path = Path(model_path).expanduser()
    if subfolder is not None:
        candidate = path / subfolder
        return candidate if is_original_image_transformer(candidate) else None
    if is_original_image_transformer(path):
        return path
    candidate = path / "transformer"
    return candidate if is_original_image_transformer(candidate) else None


def resolve_image_transformer_component(model_path):
    """Prefer a native transformer; otherwise select the existing diffusers layout."""
    root = Path(model_path).expanduser()
    native = find_original_image_transformer(root)
    if native is not None:
        return native
    for name in ("transformer_diffusers", "transformer"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No Image transformer directory found in {root}")


def resolve_original_image_output_head(model_path, subfolder=None):
    """Locate the original head shipped alongside a snapshot.

    A converted PDD transformer does not contain the original projection. A
    snapshot can provide it as ``original_output_head.safetensors`` at its root
    or as the original indexed ``transformer`` directory. Never substitute a
    PDD projection or randomly initialize a missing original projection.
    """
    root = Path(model_path).expanduser()
    component = root / subfolder if subfolder else root
    candidates = []
    for directory in (root, component, component.parent):
        for candidate in (directory / ORIGINAL_IMAGE_HEAD_FILE,
                          directory / "transformer" / "model.safetensors.index.json",
                          directory / "model.safetensors.index.json"):
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        if candidate.is_file():
            # Indexes are resolved by load_original_image_output_head at their parent.
            return candidate.parent if candidate.name == "model.safetensors.index.json" else candidate
    raise FileNotFoundError(
        f"The checkpoint at {component} has PDD heads but no original single output projection. "
        f"Place {ORIGINAL_IMAGE_HEAD_FILE} in the model root or explicitly request output_head_type='pdd'."
    )


def image_output_head_sha256(weight, bias):
    """Fingerprint FP32 projection tensors in the released shard's byte order."""
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError(f"Original output head must be FP32; got {weight.dtype} / {bias.dtype}")
    if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
        raise ValueError(f"Invalid single output head shapes: {tuple(weight.shape)}, {tuple(bias.shape)}")
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("Original single output head contains non-finite values")
    digest = hashlib.sha256()
    digest.update(bias.detach().cpu().contiguous().numpy().tobytes())
    digest.update(weight.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _read_tensors(path, keys):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        available = set(f.keys())
        missing = set(keys) - available
        if missing:
            raise ValueError(f"Missing original output head tensors in {path}: {sorted(missing)}")
        return tuple(f.get_tensor(k) for k in keys)


def load_original_image_output_head(source):
    """Read only the original projection tensors from a head file, shard or HF snapshot.

    A converted six-head checkpoint does not contain this projection. The source is
    either a safetensors file with original keys (or `weight`/`bias`), or an
    official raw snapshot containing `transformer/model.safetensors.index.json`.
    """
    path = Path(source).expanduser()
    if path.is_dir():
        indexes = (path / "transformer" / "model.safetensors.index.json", path / "model.safetensors.index.json")
        index = next((p for p in indexes if p.is_file()), None)
        if index is None:
            raise FileNotFoundError(
                f"No original transformer index in {path}; a converted PDD snapshot does not contain the original "
                "single output head. Provide the official raw snapshot, raw shard, or extracted head file."
            )
        mapping = json.loads(index.read_text())["weight_map"]
        base = index.parent.resolve()
        tensors = []
        for key in _ORIGINAL_KEYS:
            if key not in mapping:
                raise ValueError(f"Original output head key {key} not found in {index}")
            shard = (base / mapping[key]).resolve()
            if not shard.is_relative_to(base):
                raise ValueError(f"Unsafe shard path in {index}: {mapping[key]}")
            tensors.append(_read_tensors(shard, (key,))[0])
        weight, bias = tensors
    elif path.is_file():
        with safe_open(str(path), framework="pt", device="cpu") as f:
            available = set(f.keys())
            keys = next((pair for pair in (_ORIGINAL_KEYS, _HEAD_KEYS, _CONVERTED_KEYS)
                         if set(pair).issubset(available)), None)
            if keys is None:
                raise ValueError(f"No original single output head in {path}")
            weight, bias = (f.get_tensor(k) for k in keys)
    else:
        raise FileNotFoundError(f"Original single output head not found: {path}")
    weight = weight.contiguous()
    bias = bias.contiguous()
    return weight, bias, image_output_head_sha256(weight, bias)


# Streaming loader for the released native Image transformer.
def original_image_transformer_config(directory):
    """Translate the released native config to this repository's Image model config."""
    source = json.loads((Path(directory) / "config.json").read_text())
    if source.get("_class_name") != "MiniMaxH3DiTModel" or source.get("audio_enabled") is not False:
        raise ValueError(f"Not an audio-disabled original MiniMax-H3-Image transformer: {directory}")
    aliases = {
        "hidden_size": "hidden_size", "num_layers": "num_layers",
        "num_refiner_layers": "token_refiner_num_layers", "num_attention_heads": "num_attention_heads",
        "attention_head_dim": "attention_head_dim", "ffn_dim": "ffn_hidden_size",
        "in_channels": "latents_dim", "patch_size": "patch_size", "text_dim": "text_dim",
        "freq_dim": "timestep_input_dim", "time_embed_hidden_dim": "time_embed_hidden_size",
        "time_embed_dim": "time_embed_dim", "rope_freq_dim": "rope_inv_freq_len",
        "norm_eps": "norm_eps", "qk_norm_eps": "qk_norm_eps", "final_norm_eps": "final_norm_eps",
    }
    missing = [name for name in aliases.values() if name not in source]
    if missing:
        raise ValueError(f"Native Image transformer config is missing {missing}")
    config = {target: source[name] for target, name in aliases.items()}
    config["rope_theta"] = source.get("rope_theta", 10000.0)
    if len(config["patch_size"]) != 3 or any(int(x) <= 0 for x in config["patch_size"]):
        raise ValueError("Native Image transformer patch_size must have three positive dimensions")
    hidden = config["hidden_size"]
    if source.get("adaln_out_features", 18 * hidden) != 18 * hidden:
        raise ValueError("Unexpected native Image AdaLN output dimension")
    if source.get("final_adaln_out_features", 2 * hidden) != 2 * hidden:
        raise ValueError("Unexpected native Image final AdaLN output dimension")
    return config


def _validate_native_index(directory, config):
    directory = Path(directory).resolve()
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict):
        raise ValueError("Native Image transformer index is missing its weight map")
    expected = set(get_transformer_key_plan(config))
    actual = set(mapping)
    if expected != actual:
        raise ValueError(f"Native Image transformer keys differ: missing={sorted(expected - actual)[:5]}, "
                         f"unexpected={sorted(actual - expected)[:5]}")
    for name in set(mapping.values()):
        if not isinstance(name, str):
            raise ValueError("Invalid native Image transformer shard path")
        shard = (directory / name).resolve()
        if not shard.is_relative_to(directory) or not shard.is_file() or shard.suffix != ".safetensors":
            raise FileNotFoundError(f"Missing or unsafe native Image transformer shard: {name}")


def _pdd_tensors(model_root, config):
    root = Path(model_root)
    pdd_config_path = root / "pdd_config.json"
    pdd_file = root / "pdd_fused_heads.safetensors"
    if not pdd_config_path.is_file() or not pdd_file.is_file():
        raise FileNotFoundError(f"PDD mode requires pdd_config.json and pdd_fused_heads.safetensors in {root}")
    pdd_config = json.loads(pdd_config_path.read_text())
    count = pdd_config.get("num_inference_steps")
    if count != 6:
        raise ValueError(f"The released Image PDD schedule requires six heads, got {count!r}")
    output_dim = config["in_channels"]
    for size in config["patch_size"]:
        output_dim *= size
    with safe_open(str(pdd_file), framework="pt", device="cpu") as f:
        if set(f.keys()) != {"video_out.weight", "video_out.bias"}:
            raise ValueError("PDD fused head file must contain video_out.weight and video_out.bias")
        weight, bias = f.get_tensor("video_out.weight"), f.get_tensor("video_out.bias")
    if (weight.dtype != torch.float32 or bias.dtype != torch.float32 or
            tuple(weight.shape) != (count, output_dim, config["hidden_size"]) or
            tuple(bias.shape) != (count, output_dim) or
            not torch.isfinite(weight).all() or not torch.isfinite(bias).all()):
        raise ValueError("Invalid FP32 Image PDD head tensors or shapes")
    for i in range(count):
        yield f"pdd_proj_out.{i}.weight", weight[i].contiguous()
        yield f"pdd_proj_out.{i}.bias", bias[i].contiguous()


def load_original_image_transformer(cls, directory, model_root, torch_dtype=None,
                                    output_head_type="single", device=None):
    """Build on meta and assign each converted native tensor as it is read."""
    directory = Path(directory)
    config = original_image_transformer_config(directory)
    _validate_native_index(directory, config)
    if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
        raise ValueError(f"Unsupported native Image transformer dtype: {torch_dtype!r}")
    if output_head_type not in ("single", "pdd"):
        raise ValueError(f"Unsupported Image output head type: {output_head_type!r}")
    # Reject missing or invalid optional heads before streaming a large backbone.
    pdd_tensors = list(_pdd_tensors(model_root, config)) if output_head_type == "pdd" else []
    with torch.device("meta"):
        model = cls(**config, output_head_type=output_head_type)
    model.rope.inv_freq = model.rope._compute_inv_freq()
    shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}

    def tensors():
        for name, tensor in iter_original_transformer_tensors(
                directory, config=config, torch_dtype=torch_dtype, transformer_subfolder=None):
            if output_head_type == "pdd" and name in ("proj_out.weight", "proj_out.bias"):
                continue
            if name not in shapes or tuple(tensor.shape) != shapes[name]:
                raise ValueError(f"Native Image tensor {name} has unexpected shape {tuple(tensor.shape)}")
            yield name, tensor
        if output_head_type == "pdd":
            for name, tensor in pdd_tensors:
                if name not in shapes or tuple(tensor.shape) != shapes[name]:
                    raise ValueError(f"PDD Image tensor {name} has unexpected shape {tuple(tensor.shape)}")
                yield name, tensor

    assign_original_tensors(model, tensors())
    meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta:
        raise ValueError(f"Native Image transformer has uninitialized parameters: {meta[:5]}")
    if output_head_type == "single":
        model.register_to_config(single_head_sha256=image_output_head_sha256(model.proj_out.weight,
                                                                            model.proj_out.bias))
    if device is not None:
        model.to(device)
    return model.eval()


class MiniMaxH3ImageTransformer2DModel(MiniMaxH3MixedPrecisionLoaderMixin, ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    A Transformer model for image generation, introduced in MiniMax-H3-Image.

    The model denoises one packed 1-D sequence that holds the text condition, the rows of the reference images and the
    rows of the generated image. Attention is full self-attention over that sequence; there is no cross-attention and
    no per-modality block weights. Modality-specific behaviour comes only from the input patch projection, the per-row
    AdaLN modality tag and the output projection.

    The caller is responsible for building the packed layout: patchifying the image latents, ordering the rows, and
    producing the `(t, h, w)` position grid, the per-row modality tags and the per-row timestep indices. Prefer a
    padless sequence, which keeps the flash-attention backends available.

    Args:
        num_attention_heads (`int`, defaults to `56`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each attention head. Note that `num_attention_heads * attention_head_dim` is
            *larger* than `hidden_size` in MiniMax-H3-Image.
        hidden_size (`int`, defaults to `5376`):
            The number of channels of the packed sequence (the residual stream).
        num_layers (`int`, defaults to `50`):
            The number of transformer blocks.
        num_refiner_layers (`int`, defaults to `2`):
            The number of token refiner blocks applied to the projected text stream.
        ffn_dim (`int`, defaults to `14336`):
            The inner dimension of the SwiGLU feed-forward layers.
        in_channels (`int`, defaults to `24`):
            The number of channels of the image latents.
        patch_size (`tuple[int, int, int]`, defaults to `(1, 2, 2)`):
            The `(t, h, w)` patch used to pack the image latents into rows.
        text_dim (`int`, defaults to `5120`):
            The number of channels of the text conditioning produced by the text encoder.
        freq_dim (`int`, defaults to `256`):
            The dimension of the sinusoidal timestep embedding. Timesteps are consumed unscaled in `[0, 1]`.
        time_embed_hidden_dim (`int`, defaults to `5376`):
            The inner dimension of the timestep MLP.
        time_embed_dim (`int`, defaults to `2688`):
            The output dimension of the timestep MLP, i.e. the input of every AdaLN projection.
        rope_freq_dim (`int`, defaults to `16`):
            The number of rotary frequencies per axis. The `(t, h, w)` axes share one `inv_freq` buffer of this length
            and `2 * 3 * rope_freq_dim` of the `attention_head_dim` channels are rotated.
        rope_theta (`float`, defaults to `10000.0`):
            The base of the rotary frequency schedule the `rope.inv_freq` buffer is computed from.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon of the pre-attention and pre-feed-forward norms.
        qk_norm_eps (`float`, defaults to `1e-5`):
            Epsilon of the per-head query/key norms.
        final_norm_eps (`float`, defaults to `1e-5`):
            Epsilon of the token refiner output norm and of `norm_out`.
        output_head_type (`str`, defaults to `"single"`):
            `"pdd"` uses per-step output heads; `"single"` uses one shared output projection.
        single_head_sha256 (`str`, *optional*):
            Fingerprint of the original single output projection used to initialize this model.
        num_pdd_heads (`int`, *optional*):
            One for the single projection; six by default when PDD is explicitly selected.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxH3TransformerBlock", "MiniMaxH3TokenRefinerBlock", "MiniMaxH3AdaLayerNormOut"]
    _repeated_blocks = ["MiniMaxH3TransformerBlock", "MiniMaxH3TokenRefinerBlock"]
    _skip_layerwise_casting_patterns = ["norm"]
    # The checkpoint is mixed precision, as MiniMax-H3's: the input patch projection, the timestep MLP and the output
    # heads are float32 while everything else (including the AdaLN projections) is bfloat16. Entries are matched as
    # substrings of the parameter name, so `pdd_proj_out` covers every head of the output list.
    _keep_in_fp32_modules = ["proj_in", "time_embedder", "pdd_proj_out", "proj_out", "rope"]

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
        num_pdd_heads: Optional[int] = None,
        output_head_type: str = "single",
        single_head_sha256: Optional[str] = None,
    ) -> None:
        super().__init__()

        patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]

        # 1. Input projections. There is no audio branch, so a single patch projection feeds the sequence.
        self.proj_in = nn.Linear(patch_dim, hidden_size, bias=True)
        self.context_embedder = nn.Linear(text_dim, hidden_size, bias=True)

        # 2. Timestep embedding, shared by every AdaLN projection
        self.time_proj = Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(
            in_channels=freq_dim, time_embed_dim=time_embed_hidden_dim, out_dim=time_embed_dim
        )

        # 3. Rotary embedding over the packed (t, h, w) grid
        self.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=rope_freq_dim, rope_theta=rope_theta)

        # 4. Text stream refiner
        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ffn_dim=ffn_dim,
            num_layers=num_refiner_layers,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )

        # 5. The block stack, shared field-for-field with MiniMax-H3 video
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3TransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. The ordinary model has one projection; PDD can select one head per step.
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            hidden_size=hidden_size, time_embed_dim=time_embed_dim, eps=final_norm_eps
        )
        if num_pdd_heads is None:
            num_pdd_heads = 6 if output_head_type == "pdd" else 1
            self.register_to_config(num_pdd_heads=num_pdd_heads)
        if output_head_type == "pdd":
            if num_pdd_heads < 1:
                raise ValueError("A PDD transformer needs at least one output head")
            self.pdd_proj_out = PDDPreFusedHeads(
                [nn.Linear(hidden_size, patch_dim, bias=True) for _ in range(num_pdd_heads)]
            )
        elif output_head_type == "single":
            if num_pdd_heads != 1:
                raise ValueError("A single-output transformer must have num_pdd_heads=1")
            self.proj_out = nn.Linear(hidden_size, patch_dim, bias=True)
        else:
            raise ValueError(f"Unsupported output_head_type: {output_head_type!r}")

        self.gradient_checkpointing = False
        self.sp_world_size = 1
        self.sp_world_rank = 0
        self.all_gather = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, **kwargs):
        """Load the single projection by default; PDD must be explicitly selected.

        Older converted image snapshots contain only PDD heads. For those, load
        the shared backbone and restore the original FP32 projection from the
        snapshot root. A saved single-head transformer already contains its own
        projection and is loaded without replacing it.
        """

        output_head_type = kwargs.pop("output_head_type", "single")
        requested_heads = kwargs.pop("num_pdd_heads", None)
        wants_loading_info = kwargs.pop("output_loading_info", False)
        if output_head_type not in ("single", "pdd"):
            raise ValueError(f"Unsupported output_head_type: {output_head_type!r}")
        if pretrained_model_name_or_path is None:
            raise ValueError("A pretrained image transformer path is required")

        native = find_original_image_transformer(pretrained_model_name_or_path, kwargs.get("subfolder"))
        if native is not None:
            logger.info(f"Loading original MiniMax-H3-Image transformer from {native} (output_head_type={output_head_type})")
            expected_heads = 1 if output_head_type == "single" else 6
            if requested_heads is not None and int(requested_heads) != expected_heads:
                raise ValueError(f"Requested {requested_heads} output heads, expected {expected_heads}")
            torch_dtype = kwargs.pop("torch_dtype", None)
            dtype = kwargs.pop("dtype", None)
            if torch_dtype is not None and dtype is not None and torch_dtype != dtype:
                raise ValueError("Conflicting torch_dtype and dtype for the native Image transformer")
            torch_dtype = torch_dtype if torch_dtype is not None else dtype
            subfolder = kwargs.pop("subfolder", None)
            device = kwargs.pop("device", None)
            for name in ("low_cpu_mem_usage", "cache_dir", "force_download", "local_files_only", "token",
                         "revision", "proxies", "use_safetensors"):
                kwargs.pop(name, None)
            if kwargs.get("device_map") is None:
                kwargs.pop("device_map", None)
            if kwargs:
                raise ValueError(f"Unsupported native Image loading options: {sorted(kwargs)}")
            # When a component directory is passed directly, PDD assets live in its parent.
            requested = Path(pretrained_model_name_or_path).expanduser()
            model_root = requested if subfolder is not None or native != requested else (
                native.parent if native.name == "transformer" else native
            )
            model = cls.from_pretrained_original(
                native, model_root=model_root, torch_dtype=torch_dtype, output_head_type=output_head_type,
                device=device,
            )
            info = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
            return (model, info) if wants_loading_info else model

        config_kwargs = {key: kwargs[key] for key in (
            "subfolder", "cache_dir", "force_download", "local_files_only", "token", "revision", "proxies"
        ) if key in kwargs}
        source_config = cls.load_config(pretrained_model_name_or_path, **config_kwargs)
        source_mode = source_config.get("output_head_type")
        source_heads = source_config.get("num_pdd_heads")
        if source_mode is None:
            # The released converted snapshot predates output_head_type and has
            # num_pdd_heads=6. Its tensors are PDD projections, not proj_out.
            source_mode = "pdd" if source_heads is not None and int(source_heads) > 1 else "single"
        if source_mode not in ("single", "pdd"):
            raise ValueError(f"Unsupported checkpoint output_head_type: {source_mode!r}")
        source_heads = int(source_heads) if source_heads is not None else (6 if source_mode == "pdd" else 1)
        if source_mode == "single" and source_heads != 1:
            raise ValueError("A single-output checkpoint must have num_pdd_heads=1")
        if source_mode == "pdd" and source_heads < 1:
            raise ValueError("A PDD checkpoint must contain at least one head")
        if output_head_type == "pdd" and source_mode != "pdd":
            raise ValueError("A single-head checkpoint does not contain PDD heads; provide a PDD transformer checkpoint")
        expected_heads = 1 if output_head_type == "single" else source_heads
        if requested_heads is not None and int(requested_heads) != expected_heads:
            raise ValueError(f"Requested {requested_heads} output heads, expected {expected_heads}")

        head_source = None
        if output_head_type == "single" and source_mode == "pdd":
            head_source = resolve_original_image_output_head(
                pretrained_model_name_or_path, subfolder=kwargs.get("subfolder")
            )
            # Fail before loading a large backbone if the discovered head is invalid.
            load_original_image_output_head(head_source)

        result = super().from_pretrained(
            pretrained_model_name_or_path, output_head_type=source_mode, num_pdd_heads=source_heads,
            output_loading_info=True, **kwargs
        )
        model, info = result if isinstance(result, tuple) else (result, {})
        missing = info.get("missing_keys", [])
        mismatched = info.get("mismatched_keys", [])
        if missing or mismatched:
            raise ValueError(f"Image transformer checkpoint has missing or mismatched weights: {missing[:5]} {mismatched[:5]}")
        meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
        if meta:
            raise ValueError(f"Image transformer checkpoint left parameters uninitialized: {meta[:5]}")
        if head_source is not None:
            model.use_single_output_head(head_source)
        elif output_head_type == "single" and not getattr(model.config, "single_head_sha256", None):
            digest = image_output_head_sha256(model.proj_out.weight, model.proj_out.bias)
            model.register_to_config(single_head_sha256=digest)
        return (model, info) if wants_loading_info else model

    @classmethod
    def from_pretrained_original(cls, checkpoint_path, torch_dtype=None, device=None, model_root=None,
                                 output_head_type="single"):
        """Stream a native Image transformer into this model without an on-disk conversion."""

        native = find_original_image_transformer(checkpoint_path)
        if native is None:
            raise ValueError(f"No original MiniMax-H3-Image transformer found in {checkpoint_path}")
        if model_root is None:
            requested = Path(checkpoint_path).expanduser()
            model_root = requested if native != requested else (native.parent if native.name == "transformer" else native)
        return load_original_image_transformer(cls, native, model_root, torch_dtype=torch_dtype,
                                               output_head_type=output_head_type, device=device)

    def use_single_output_head(self, source):
        """Replace the step-specific heads with the original FP32 output projection.

        Call before FSDP/DeepSpeed wrapping and LoRA injection. `source` is an
        extracted head safetensors file, an official raw shard, or an official raw
        model snapshot. The returned fingerprint can be stored with an adapter.
        """

        weight, bias, digest = load_original_image_output_head(source)
        old = self.proj_out if self.config.output_head_type == "single" else self.pdd_proj_out[0]
        if old.weight.device.type == "meta":
            raise ValueError("Load the transformer weights before selecting the single output head")
        if tuple(weight.shape) != tuple(old.weight.shape) or tuple(bias.shape) != tuple(old.bias.shape):
            raise ValueError(
                f"Single output head shape {tuple(weight.shape)}, {tuple(bias.shape)} does not match "
                f"the transformer output {tuple(old.weight.shape)}, {tuple(old.bias.shape)}"
            )
        new = nn.Linear(old.in_features, old.out_features, bias=True, dtype=torch.float32, device="meta")
        new.to_empty(device=old.weight.device)
        with torch.no_grad():
            new.weight.copy_(weight)
            new.bias.copy_(bias)
        new.weight.requires_grad_(old.weight.requires_grad)
        new.bias.requires_grad_(old.bias.requires_grad)
        if hasattr(self, "pdd_proj_out"):
            del self.pdd_proj_out
        self.proj_out = new
        self.register_to_config(output_head_type="single", num_pdd_heads=1, single_head_sha256=digest)
        return digest

    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            value = kwargs["value"]
        elif "enable" in kwargs:
            value = kwargs["enable"]
        else:
            raise ValueError("Invalid set gradient checkpointing")
        self.gradient_checkpointing = value
        self.token_refiner.gradient_checkpointing = value

    def enable_multi_gpus_inference(self):
        """
        Enable multi-GPU inference by splitting the packed sequence across the sequence-parallel group.

        Every rank keeps its own slice of the rows — norms, feed-forwards and the projections run on that slice only —
        and the attention processor turns the row split into a head split over the whole sequence through the group's
        all-to-all. The rows are gathered back after the block stack, before the per-step output head.
        """
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather

        # The all-to-all of the attention processor splits the heads across the group, so the head count has to be
        # divisible by it — a mismatch would silently drop heads rather than fail inside the kernel.
        num_heads = self.transformer_blocks[0].attn.heads
        if num_heads % self.sp_world_size != 0:
            raise ValueError(
                f"Sequence-parallel inference splits the {num_heads} attention heads across the group, so the group "
                f"size must divide them; got a sequence-parallel world size of {self.sp_world_size}."
            )

        for block in self.transformer_blocks:
            block.attn.set_processor(MiniMaxH3MultiGPUsAttnProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        image_indices: torch.Tensor,
        text_indices: torch.Tensor,
        pdd_step_index: int = 0,
        attention_kwargs: Optional[dict] = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, num_image_tokens, in_channels * prod(patch_size))`):
                Patchified latent rows — the rows of the reference images and the rows of the generated image —
                ordered as they appear in the packed sequence, i.e. matching `image_indices`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, num_text_tokens, text_dim)`):
                Text conditioning, ordered to match `text_indices`.
            timestep (`torch.Tensor` of shape `(num_timesteps,)`):
                The *distinct* timestep values present in the packed sequence, in `[0, 1]` and unscaled. One forward
                serves rows at different noise levels (generated rows, reference rows).
            timestep_indices (`torch.Tensor` of shape `(seq_len,)`):
                For every row of the packed sequence, the index of its timestep in `timestep`.
            token_tags (`torch.Tensor` of shape `(seq_len,)`):
                For every row of the packed sequence, its modality: `0` image, `1` text.
            position_ids (`torch.Tensor` of shape `(seq_len, 3)`):
                The `(t, h, w)` rotary coordinates of every row of the packed sequence.
            image_indices (`torch.Tensor` of shape `(num_image_tokens,)`):
                Positions of the image rows in the packed sequence.
            text_indices (`torch.Tensor` of shape `(num_text_tokens,)`):
                Positions of the text rows in the packed sequence.
            pdd_step_index (`int`, defaults to `0`):
                The sampling step, which selects the output head.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that, if specified, may carry a `scale` entry which is applied to the LoRA layers.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.modeling_outputs.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or `tuple`:
                The velocity of every image row, of shape `(batch_size, num_image_tokens, in_channels *
                prod(patch_size))`, in the row order of `image_indices`.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching `position_ids`, got "
                f"{list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}."
            )

        # 1. Project each modality and scatter the rows into the packed sequence buffer. The checkpoint is
        # mixed-precision (the patch projection is float32 while `context_embedder` and the block stack are bfloat16),
        # so every input is aligned with its projection's parameter dtype. The text stream sets the sequence dtype.
        image_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        text_embeds = self.token_refiner(text_embeds)

        hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
        hidden_states = hidden_states.index_copy(1, image_indices, image_embeds.to(text_embeds.dtype))

        # 2. One timestep embedding per distinct noise level. `temb` stays at the time embedder's precision: each
        # AdaLN module applies its own activation to it and casts to its projection's dtype afterwards.
        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype)).to(hidden_states.dtype)

        # 3. Row -> AdaLN table row. `clamp(min=0)` mirrors the video model, where padding rows carry the tag `-1`.
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)

        # 4. Padding rows (tag `-1`) must not exchange attention with live rows; a boolean mask that pairs live rows
        # with live rows and padding rows with padding rows reproduces the reference's attention-document split.
        attention_mask = None
        is_pad = token_tags < 0
        if bool(is_pad.any()):
            attention_mask = is_pad[None, :] == is_pad[:, None]

        # 5. Sequence parallel: pad the sequence up to a multiple of the group size and keep only this rank's rows.
        padded_length, valid_length = sequence_length, None
        if self.sp_world_size > 1:
            if attention_mask is not None:
                raise ValueError(
                    "Multi-GPU inference expects a padless packed sequence; drop the padding rows (tag -1) instead."
                )
            padded_length = sequence_length + (-sequence_length) % self.sp_world_size
            if padded_length != sequence_length:
                pad = padded_length - sequence_length
                hidden_states = F.pad(hidden_states, (0, 0, 0, pad))
                position_ids = F.pad(position_ids, (0, 0, 0, pad))
                adaln_indices = F.pad(adaln_indices, (0, pad))
            valid_length = sequence_length

        rotary_emb = self.rope(position_ids)

        if self.sp_world_size > 1:
            chunk_size = padded_length // self.sp_world_size
            rows = slice(self.sp_world_rank * chunk_size, (self.sp_world_rank + 1) * chunk_size)
            hidden_states = hidden_states[:, rows]
            adaln_indices = adaln_indices[rows]
            rotary_emb = (rotary_emb[0][rows], rotary_emb[1][rows])

        for block in self.transformer_blocks:
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
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, attention_mask,
                                      valid_length)

        if self.sp_world_size > 1:
            hidden_states = self.all_gather(hidden_states.contiguous(), dim=1)[:, :sequence_length]

        # 6. The head of this step runs over every row, then the image rows are selected. The heads are listed in
        # `_keep_in_fp32_modules`, so they stay float32 while the block stack runs in the requested `torch_dtype`;
        # align the activation with their parameter dtype.
        if self.config.output_head_type == "single":
            if pdd_step_index != 0:
                raise ValueError("A single output head only accepts pdd_step_index=0")
            hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(self.proj_out.weight.dtype)
            output = self.proj_out(hidden_states)
        else:
            hidden_states = self.norm_out(hidden_states, temb, timestep_indices)
            output = self.pdd_proj_out(hidden_states, pdd_step_index)
        output = output.index_select(1, image_indices)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
