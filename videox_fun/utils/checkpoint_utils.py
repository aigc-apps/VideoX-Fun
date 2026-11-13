"""Model weight I/O and transactional Accelerate training checkpoints."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import errno
import functools
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import types
import uuid
from weakref import WeakKeyDictionary, ref

try:
    import fcntl
except ImportError:  # Lock recovery is unavailable on platforms without flock.
    fcntl = None

import numpy as np
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file


def _remove_prefix(text, prefix):
    """str.removeprefix backport for Python < 3.9."""
    return text[len(prefix):] if prefix and text.startswith(prefix) else text


def checkpoint_fingerprint(path):
    """Content identity for a frozen model source, without loading tensor payloads.

    Include the index and config when present. Stream the actual weights so an
    in-place replacement is detected even if filenames and sizes are unchanged.
    Paths are relative, allowing an identical checkpoint to be moved.
    """
    path = Path(path)
    if path.is_dir():
        files = safetensor_checkpoint_files(path)
        if files is None:
            index = path / "diffusion_pytorch_model.bin.index.json"
            files = ([path / name for name in sorted(set(json.loads(index.read_text())["weight_map"].values()))]
                     if index.exists() else [path / "diffusion_pytorch_model.bin"])
        files = [Path(name) for name in files]
        files += [file for file in (path / "config.json", path / "diffusion_pytorch_model.safetensors.index.json",
                                   path / "diffusion_pytorch_model.bin.index.json") if file.is_file()]
    else:
        files = [path]
    result = []
    for file in sorted(set(files)):
        before = file.stat()
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        after = file.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RuntimeError(f"Checkpoint source changed while hashing: {file}")
        result.append({"name": file.relative_to(path).as_posix() if path.is_dir() else "weights",
                       "size": after.st_size, "sha256": digest.hexdigest()})
    return result


def save_diffusers_state_dict(state_dict, model, output_dir, max_shard_size=5 * 1024**3, *, dtype=None):
    """Save weights in bounded-size standard Diffusers shards.

    Consume the gathered dictionary as shards are written, avoiding a second
    full-model serialization buffer. Floating dtypes are preserved unless
    ``dtype`` is supplied; integer and boolean buffers are never cast.
    This consumes ``state_dict``. All ranks must gather before rank zero calls it.
    """
    from safetensors.torch import save_file
    if not state_dict:
        raise ValueError('Cannot save an empty full transformer checkpoint')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups, group_bytes, total_bytes = [[]], 0, 0
    for key, value in state_dict.items():
        size = value.numel() * (torch.empty((), dtype=dtype).element_size() if dtype is not None and value.is_floating_point() else value.element_size())
        if size > max_shard_size:
            raise ValueError(f"{key} exceeds the checkpoint shard size")
        if groups[-1] and group_bytes + size > max_shard_size:
            groups.append([])
            group_bytes = 0
        groups[-1].append(key)
        group_bytes += size
        total_bytes += size
    weight_map = {}
    for index, keys in enumerate(groups, 1):
        filename = f"diffusion_pytorch_model-{index:05d}-of-{len(groups):05d}.safetensors"
        tensors = {}
        for key in keys:
            value = state_dict.pop(key)
            tensors[key] = value.detach().to(device='cpu', dtype=dtype if dtype is not None and value.is_floating_point() else value.dtype).contiguous()
            weight_map[key] = filename
        save_file(tensors, output_dir / filename, metadata={'format': 'pt'})
        del tensors, value
    (output_dir / 'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps(
        {'metadata': {'total_size': total_bytes}, 'weight_map': weight_map}, indent=2))
    model.save_config(output_dir)
    for path in output_dir.glob('diffusion_pytorch_model*.safetensors'):
        if path.name not in set(weight_map.values()):
            path.unlink()


def read_diffusers_state_dict(directory):
    """Read a standard single-file or sharded safetensors checkpoint on CPU."""
    directory = Path(directory)
    index = directory / 'diffusion_pytorch_model.safetensors.index.json'
    weight_map = json.loads(index.read_text())['weight_map'] if index.exists() else None
    names = sorted(set(weight_map.values())) if weight_map is not None else ['diffusion_pytorch_model.safetensors']
    state = {}
    for name in names:
        shard = load_file(directory / name)
        if state.keys() & shard.keys():
            raise ValueError(f'Duplicate tensors in checkpoint shard {name}')
        if weight_map is not None and any(weight_map.get(key) != name for key in shard):
            raise ValueError(f'Checkpoint index does not match shard {name}')
        state.update(shard)
    if not state or (weight_map is not None and set(state) != set(weight_map)):
        raise ValueError(f'Incomplete checkpoint index: {directory}')
    return state


def load_diffusers_checkpoint(model_cls, model_name, checkpoint, torch_dtype, *, subfolder=None):
    """Validate all tensors before loading one model; partial/LoRA files are rejected."""
    if checkpoint is None:
        raise ValueError("A full model checkpoint is required")
    checkpoint = Path(checkpoint)
    config = model_cls.load_config(model_name, subfolder=subfolder)
    with init_empty_weights(include_buffers=True):
        expected_model = model_cls.from_config(config)
    expected = {key: tuple(value.shape) for key, value in expected_model.state_dict().items()}
    del expected_model
    weight_map = None
    if checkpoint.is_dir():
        if not (checkpoint / "config.json").is_file():
            raise ValueError(f"Missing transformer config: {checkpoint}")
        index = checkpoint / "diffusion_pytorch_model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            files = [checkpoint / name for name in sorted(set(weight_map.values()))]
        else:
            files = [checkpoint / "diffusion_pytorch_model.safetensors"]
    else:
        files = [checkpoint]
    found = {}
    state = None
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint weights: {path}")
        if path.suffix == ".safetensors":
            with safe_open(path, framework="pt", device="cpu") as reader:
                shapes = {key: tuple(reader.get_slice(key).get_shape()) for key in reader.keys()}
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
            shapes = {key: tuple(value.shape) for key, value in state.items()}
        for key, shape in shapes.items():
            if key in found:
                raise ValueError(f"Duplicate checkpoint tensor: {key}")
            if weight_map is not None and weight_map.get(key) != path.name:
                raise ValueError(f"Checkpoint index does not match shard for {key}")
            found[key] = shape
    missing = sorted(expected.keys() - found.keys())
    unexpected = sorted(found.keys() - expected.keys())
    mismatched = [key for key in expected.keys() & found.keys() if expected[key] != found[key]]
    if missing or unexpected or mismatched or (weight_map is not None and set(weight_map) != set(found)):
        raise ValueError(
            f"Incomplete or incompatible full checkpoint {checkpoint}: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, shape_mismatch={mismatched[:5]}"
        )
    if checkpoint.is_dir():
        model, info = model_cls.from_pretrained(
            checkpoint, low_cpu_mem_usage=True, torch_dtype=torch_dtype, output_loading_info=True,
        )
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Checkpoint could not be loaded completely: {info}")
        return model
    # A standalone file needs the base config/precision contract. Only this path
    # loads base weights; directory checkpoints never instantiate a base model.
    model = model_cls.from_pretrained(
        model_name, subfolder=subfolder, low_cpu_mem_usage=True, torch_dtype=torch_dtype,
    )
    if state is None:
        state = load_file(str(checkpoint))
    model.load_state_dict(state, strict=True)
    return model


def safetensor_checkpoint_files(path):
    path = os.fspath(path)
    if not os.path.isdir(path):
        return [path] if path.endswith(".safetensors") else None
    index_path = os.path.join(path, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as reader:
            filenames = sorted(set(json.load(reader)["weight_map"].values()))
        return [os.path.join(path, filename) for filename in filenames]
    single_file = os.path.join(path, "diffusion_pytorch_model.safetensors")
    if os.path.isfile(single_file):
        return [single_file]
    files = sorted(os.path.join(path, name) for name in os.listdir(path) if name.endswith(".safetensors"))
    return files or None


def is_complete_diffusers_checkpoint(model_cls, path, base_config):
    from accelerate import init_empty_weights
    from safetensors import safe_open

    if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, "config.json")):
        return False
    files = safetensor_checkpoint_files(path)
    if files is None:
        return False
    with init_empty_weights():
        expected_model = model_cls.from_config(base_config)
        candidate_model = model_cls.from_config(model_cls.load_config(path))
    configs = [dict(model.config) for model in (expected_model, candidate_model)]
    if json.dumps({k: v for k, v in configs[0].items() if not k.startswith("_")}, sort_keys=True) != json.dumps(
        {k: v for k, v in configs[1].items() if not k.startswith("_")}, sort_keys=True
    ):
        return False
    expected = {key: tuple(value.shape) for key, value in expected_model.state_dict().items()}
    found = {}
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in found:
                    return False
                found[key] = tuple(reader.get_slice(key).get_shape())
    return found == expected


def copy_diffusers_checkpoint(model, path, weight_dtype, *, strip_prefix=None, state_dict_key=None):
    from safetensors import safe_open

    files = safetensor_checkpoint_files(path)
    if files is None:
        if os.path.isdir(path):
            candidate = type(model).from_pretrained(path, low_cpu_mem_usage=True, torch_dtype=weight_dtype)
            state_dict = candidate.state_dict()
            del candidate
        else:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            state_dict = state_dict.get("state_dict", state_dict)
            if state_dict_key is not None:
                state_dict = state_dict.get(state_dict_key, state_dict)
        if strip_prefix:
            state_dict = {_remove_prefix(key, strip_prefix): value for key, value in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        del state_dict
    else:
        destination = model.state_dict()
        found_shapes = {}
        for filename in files:
            with safe_open(filename, framework="pt", device="cpu") as reader:
                for source_key in reader.keys():
                    key = _remove_prefix(source_key, strip_prefix) if strip_prefix else source_key
                    if key in found_shapes:
                        raise ValueError(f"Duplicate transformer key: {key}")
                    found_shapes[key] = tuple(reader.get_slice(source_key).get_shape())
        if found_shapes != {key: tuple(value.shape) for key, value in destination.items()}:
            raise ValueError(f"Incomplete or incompatible full transformer checkpoint: {path}")
        seen, unexpected = set(), []
        with torch.no_grad():
            for filename in files:
                with safe_open(filename, framework="pt", device="cpu") as reader:
                    for source_key in reader.keys():
                        key = _remove_prefix(source_key, strip_prefix) if strip_prefix else source_key
                        if key in seen:
                            raise ValueError(f"Duplicate transformer key: {key}")
                        seen.add(key)
                        if key not in destination:
                            unexpected.append(key)
                            continue
                        value = reader.get_tensor(source_key)
                        if value.shape != destination[key].shape:
                            raise ValueError(f"{key}: checkpoint shape {value.shape} != model shape {destination[key].shape}")
                        destination[key].copy_(value)
                        del value
        missing = sorted(set(destination) - seen)
    if unexpected:
        raise ValueError(f"Unexpected transformer keys in {path}: {unexpected[:3]}")
    gc.collect()


def load_model_checkpoint(model, checkpoint):
    """Load a full local checkpoint into an existing model, one shard at a time.

    Validate all keys, shapes and index entries before modifying parameters.
    Supports standalone PyTorch/safetensors files and Diffusers directories.
    Uses model.load_state_dict to preserve model-specific loading hooks and
    returns (missing_keys, unexpected_keys), as expected by training callers.
    """
    checkpoint = Path(checkpoint)
    weight_map = None
    if checkpoint.is_dir():
        index = checkpoint / "diffusion_pytorch_model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            if not weight_map:
                raise ValueError(f"Empty checkpoint weight map: {index}")
            files = [checkpoint / name for name in sorted(set(weight_map.values()))]
        else:
            files = [checkpoint / "diffusion_pytorch_model.safetensors"]
    else:
        files = [checkpoint]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing transformer checkpoint shard: {path}")
    expected_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    found_shapes = {}
    for path in files:
        if path.suffix == ".safetensors":
            from safetensors import safe_open
            with safe_open(str(path), framework="pt", device="cpu") as reader:
                shapes = {key: tuple(reader.get_slice(key).get_shape()) for key in reader.keys()}
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
            shapes = {key: tuple(value.shape) for key, value in state.items()}
            del state
        for key, shape in shapes.items():
            if key in found_shapes:
                raise ValueError(f"Duplicate checkpoint tensor: {key}")
            if weight_map is not None and weight_map.get(key) != path.name:
                raise ValueError(f"Checkpoint index does not match shard for {key}")
            found_shapes[key] = shape
    if found_shapes != expected_shapes or (weight_map is not None and set(weight_map) != set(found_shapes)):
        raise ValueError(f"Incomplete or incompatible full transformer checkpoint: {checkpoint}")
    expected = set(expected_shapes)
    loaded = set()
    unexpected = set()
    for path in files:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(path))
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        if weight_map is not None:
            shard_keys = {key for key, name in weight_map.items() if name == path.name}
            if not shard_keys.issubset(state):
                raise ValueError(f"Checkpoint index references missing tensors in {path}")
        _, extra = model.load_state_dict(state, strict=False)
        loaded.update(state)
        unexpected.update(extra)
        del state
    return sorted(expected - loaded), sorted(unexpected)


MANIFEST = "checkpoint_manifest.json"
PROTOCOL = 1
_CONFIGS = WeakKeyDictionary()
_CHECKPOINT_NAME = re.compile(r"checkpoint[-_](\d+)\Z")


@dataclass
class CheckpointConfig:
    model_ids: WeakKeyDictionary = field(default_factory=WeakKeyDictionary)
    allow_legacy: bool = True
    check_finite: bool = False
    checksum: str = "metadata"
    private_states: dict = field(default_factory=dict)
    private_load_hook_ids: set = field(default_factory=set)
    validators: dict = field(default_factory=dict)
    run_id: str = None
    metadata: dict = field(default_factory=dict)
    legacy_metadata_keys: tuple = ()
    legacy_metadata_groups: tuple = ()
    step_guards: list = field(default_factory=list)


def checkpoint_config(owner):
    if owner not in _CONFIGS:
        _CONFIGS[owner] = CheckpointConfig(
            check_finite=os.environ.get("VIDEOX_CHECKPOINT_FINITE", "0") == "1",
        )
    return _CONFIGS[owner]


def configure_checkpointing(owner, *, model_ids=None, allow_legacy=None, check_finite=None, checksum=None, metadata=None,
                            legacy_metadata_keys=(), legacy_metadata_groups=(), _validate=True):
    """Configure stable model roles and explicit compatibility/integrity options.

    ``model_ids`` maps role names to prepared models. When omitted, stable
    default roles are assigned, including for multiple models of the same class.
    Restore uses native Accelerate loading; manifest and configuration checks
    do not block it.
    ``allow_legacy`` and the legacy metadata arguments remain accepted for
    compatibility with existing training scripts.
    ``checksum='full'`` adds streaming hashes of large payloads; the default
    hashes control files and records the size of every payload.
    The legacy metadata arguments only describe the saved configuration; they
    do not impose extra restrictions on loading an older checkpoint.
    """
    config = checkpoint_config(owner)
    if _validate and getattr(owner, "_models", None):
        validate_checkpoint_owner(owner)
    if model_ids is not None:
        if not all(isinstance(name, str) and name for name in model_ids):
            raise ValueError("Checkpoint model IDs must be nonempty strings")
        if len(set(map(id, model_ids.values()))) != len(model_ids):
            raise ValueError("Each checkpoint model must have exactly one ID")
        config.model_ids = WeakKeyDictionary((model, name) for name, model in model_ids.items())
    if metadata is not None:
        try:
            config.metadata = _manifest_value(metadata)
        except Exception:
            config.metadata = {"type": _class_name(metadata), "omitted": True}
    config.legacy_metadata_keys = tuple(legacy_metadata_keys)
    config.legacy_metadata_groups = tuple(tuple(group) for group in legacy_metadata_groups)
    if allow_legacy is not None:
        config.allow_legacy = bool(allow_legacy)
    if check_finite is not None:
        config.check_finite = bool(check_finite)
    if checksum is not None:
        if checksum not in ("metadata", "full"):
            raise ValueError("Checkpoint checksum must be 'metadata' or 'full'")
        config.checksum = checksum
    return owner


def checkpoint_phase(function):
    """Synchronize errors from a local phase, including native DCP exceptions."""
    result, failure = None, None
    try:
        result = function()
    except BaseException as exc:
        # DCP CheckpointException derives directly from BaseException.
        failure = f"{type(exc).__name__}: {exc}"
    failures = [failure]
    if dist.is_initialized():
        failures = [None] * dist.get_world_size()
        dist.all_gather_object(failures, failure)
    if any(failures):
        detail = "; ".join(f"rank {rank}: {error}" for rank, error in enumerate(failures) if error)
        raise RuntimeError("Training checkpoint failed: " + detail)
    return result


def _agree(value):
    if dist.is_initialized():
        values = [None] * dist.get_world_size()
        dist.all_gather_object(values, value)
        if any(other != values[0] for other in values[1:]):
            raise RuntimeError("Checkpoint configuration differs between ranks")


def _main_value(function):
    value = checkpoint_phase(lambda: function() if not dist.is_initialized() or dist.get_rank() == 0 else None)
    if dist.is_initialized():
        values = [value]
        dist.broadcast_object_list(values, src=0)
        value = values[0]
    return value


def checkpoint_source_identity(path):
    """Hash frozen checkpoint inputs once on rank zero and share their identity."""

    _agree(os.fspath(path))
    return _main_value(lambda: checkpoint_fingerprint(path))


def _class_name(value):
    return type(value).__module__ + "." + type(value).__qualname__


def _plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    raise ValueError(f"Unsupported checkpoint configuration value: {_class_name(value)}")


def _manifest_value(value, seen=None):
    """Describe optional metadata deterministically without blocking native save."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (torch.dtype, torch.device, os.PathLike)):
        return str(value)
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        try:
            return _manifest_value(value.item(), seen)
        except (RuntimeError, ValueError, TypeError):
            pass
    if isinstance(value, (tuple, list, dict)):
        seen = set() if seen is None else seen
        if id(value) in seen:
            return {"type": _class_name(value), "omitted": "recursive"}
        seen.add(id(value))
        try:
            if isinstance(value, dict):
                items = ((key if isinstance(key, str) else str(key), item) for key, item in value.items()
                         if key is None or isinstance(key, (str, int, float, bool)))
                return {key: _manifest_value(item, seen) for key, item in sorted(items, key=lambda pair: pair[0])}
            return [_manifest_value(item, seen) for item in value]
        finally:
            seen.remove(id(value))
    return {"type": _class_name(value), "omitted": True}


def _callable_signature(function):
    if isinstance(function, functools.partial):
        return {"partial": _callable_signature(function.func), "args": _plain(function.args),
                "kwargs": _plain(function.keywords)}
    if isinstance(function, types.FunctionType):
        return {"function": function.__name__, "defaults": _plain(function.__defaults__),
                "keywords": _plain(function.__kwdefaults__),
                "closure": [_plain(cell.cell_contents) for cell in function.__closure__ or ()]}
    return {"callable_type": _class_name(function)}


def _scheduler_callable_description(function):
    # This is informational metadata. Arbitrary callable internals are already
    # handled by the scheduler's native state_dict and must not block saving.
    try:
        return _callable_signature(function)
    except Exception:
        return {"callable_type": _class_name(function)}



def _save_hook_counts(owners):
    # Only save hooks run during saving, and each wrapper invokes a collective.
    return {name: len(owner._save_model_state_pre_hook) for name, owner in owners.items()}


def _raw_optimizer(optimizer):
    return getattr(optimizer, "optimizer", optimizer)


def validate_checkpoint_owner(owner):
    from .. import _supported_accelerate_versions
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

    if not _supported_accelerate_versions():
        raise ValueError("Protected checkpoints require Accelerate 1.7.0 / PyTorch 2.7.0")
    mode = owner.distributed_type.name
    if mode not in ("FSDP", "NO", "MULTI_GPU"):
        raise ValueError(f"Protected checkpoint backend is not validated: {mode}")
    if not owner._models or len(owner._models) != len(owner._optimizers):
        raise ValueError("Protected checkpoints require one optimizer per prepared model")
    if owner.project_configuration.save_on_each_node:
        raise ValueError("Protected checkpoints require one shared checkpoint filesystem")
    if any(type(_raw_optimizer(optimizer)) not in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD)
           for optimizer in owner._optimizers):
        raise ValueError("Protected checkpoints support native Adam, AdamW and SGD; this optimizer is not validated")
    if mode == "FSDP":
        plugin = owner.state.fsdp_plugin
        if getattr(plugin, "fsdp_version", 1) != 1 or plugin.state_dict_type.name != "SHARDED_STATE_DICT":
            raise ValueError("Protected FSDP checkpoints require FSDP1 SHARDED_STATE_DICT")
        for model in owner._models:
            for unit in FSDP.fsdp_modules(model):
                if (unit.process_group is not model.process_group or unit._device_mesh is not None
                        or dist.get_world_size(unit.process_group) != owner.num_processes
                        or unit.sharding_strategy not in (ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP)):
                    raise ValueError("Only ordinary FSDP1 sharding over the world group is validated")
                if unit.cpu_offload.offload_params and not unit._use_orig_params:
                    raise ValueError("CPU offload checkpoints require use_orig_params=True for reliable continuation")


def _descriptor(owner, *, require_model_ids=True):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    validate_checkpoint_owner(owner)
    config = checkpoint_config(owner)
    models, optimizers = [], []
    if config.model_ids and set(config.model_ids) != set(owner._models):
        raise ValueError("Checkpoint model_ids must identify every currently prepared model")
    classes = [_class_name(owner.unwrap_model(model, keep_torch_compile=False)) for model in owner._models]
    for index, (model, wrapped_optimizer) in enumerate(zip(owner._models, owner._optimizers)):
        optimizer = _raw_optimizer(wrapped_optimizer)
        if owner.distributed_type.name == "FSDP":
            groups, shapes = _optimizer_layout(model, optimizer)
            from torch.distributed.fsdp._common_utils import _get_param_to_fqns
            names = _get_param_to_fqns(model)
            units = [{"original_params": unit._use_orig_params, "cpu_offload": unit.cpu_offload.offload_params,
                      "sharding": unit.sharding_strategy.name} for unit in FSDP.fsdp_modules(model)]
        else:
            names = {parameter: [name] for name, parameter in model.named_parameters()}
            shapes = {name: parameter.shape for name, parameter in model.named_parameters()}
            groups = [[name for parameter in group["params"] for name in names[parameter]] for group in optimizer.param_groups]
            units = []
        role = config.model_ids.get(model)
        if role is None:
            role = (f"model_{index}" if classes.count(classes[index]) > 1
                    else "model" if len(classes) == 1 else classes[index])
        parameters = {name: {"shape": list(shapes[name]), "dtype": str(parameter.dtype),
                             "requires_grad": parameter.requires_grad}
                      for parameter, fqns in names.items() for name in fqns}
        unwrapped = owner.unwrap_model(model, keep_torch_compile=False)
        try:
            model_config = getattr(unwrapped, "config", {})
            model_config = {key: value for key, value in dict(model_config).items()
                            if isinstance(key, str) and not key.startswith("_")} if hasattr(model_config, "items") else {}
            model_config = _manifest_value(model_config)
        except Exception:
            model_config = {"omitted": True}
        models.append({"id": role, "class": classes[index], "parameters": parameters,
                       "config": model_config, "fsdp": units})
        optimizers.append({"class": _class_name(optimizer), "groups": groups,
                           "execution": [{key: _plain(group.get(key, False)) for key in
                                          ("capturable", "fused", "differentiable")} for group in optimizer.param_groups]})
    if len({model["id"] for model in models}) != len(models):
        raise ValueError("Checkpoint model IDs must be unique")
    schedulers = []
    for wrapped in owner._schedulers:
        scheduler = getattr(wrapped, "scheduler", wrapped)
        index = next((i for i, optimizer in enumerate(owner._optimizers)
                      if scheduler.optimizer in (optimizer, _raw_optimizer(optimizer))), None)
        if index is None:
            raise ValueError("Scheduler is not bound to a prepared optimizer")
        row = {"class": type(scheduler).__name__, "optimizer": index}
        if hasattr(scheduler, "lr_lambdas"):
            row["functions"] = [_scheduler_callable_description(function) for function in scheduler.lr_lambdas]
        schedulers.append(row)
    return {"world_size": owner.num_processes, "backend": owner.distributed_type.name,
            "mixed_precision": owner.mixed_precision, "accumulation_steps": owner.gradient_accumulation_steps,
            "models": models, "optimizers": optimizers, "schedulers": schedulers,
            "custom_objects": [_class_name(obj) for obj in owner._custom_objects],
            "scaler": _class_name(owner.scaler) if owner.scaler else None,
            "private_states": config.private_states, "training": config.metadata,
            "dataloaders": [{"class": _class_name(loader),
                             "dataset": _class_name(getattr(loader.dataset, "checkpoint_dataset", loader.dataset)),
                             "workers": loader.num_workers, "stateful": getattr(loader, "use_stateful_dataloader", False)}
                            for loader in owner._dataloaders]}


def _relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Checkpoint paths must be relative and remain inside the checkpoint")
    return path


def _participants(owners):
    owners = owners if isinstance(owners, dict) else {"": owners}
    result = {}
    for name, owner in sorted(owners.items()):
        path = _relative(name)
        key = "" if str(path) == "." else path.as_posix()
        if key in result or owner in result.values():
            raise ValueError("Checkpoint components and Accelerators must be unique")
        if any(other and (path.is_relative_to(other) or Path(other).is_relative_to(path)) for other in result):
            raise ValueError("Non-root checkpoint components must not overlap")
        result[key] = owner
    if not result:
        raise ValueError("At least one checkpoint participant is required")
    return result


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_schema(path):
    from torch.distributed.checkpoint import FileSystemReader, TensorStorageMetadata

    metadata = FileSystemReader(path.parent).read_metadata()
    rows = {}
    for key, value in metadata.state_dict_metadata.items():
        row = {"path": list(metadata.planner_data[key])}
        if isinstance(value, TensorStorageMetadata):
            row.update(shape=list(value.size), dtype=str(value.properties.dtype))
        rows[key] = row
    return rows


def _inventory(directory, checksum):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Checkpoint cannot contain symbolic links: {path}")
        if not path.is_file() or path.name == MANIFEST:
            continue
        row = {"size": path.stat().st_size}
        # Large native payloads retain bounded I/O by default. All control files,
        # including DCP metadata and RNG/scheduler/custom states, are hashed.
        if checksum == "full" or path.suffix not in (".distcp", ".safetensors", ".bin") or path.stat().st_size <= 16 * 1024 * 1024:
            row["sha256"] = _sha(path)
        if path.name == ".metadata":
            row["schema"] = _metadata_schema(path)
        result[path.relative_to(directory).as_posix()] = row
    return result


def _verify_inventory(directory, manifest, *, full=True):
    if manifest.get("protocol") != PROTOCOL or manifest.get("status") != "complete":
        raise ValueError("Checkpoint is not a committed supported checkpoint")
    if not all(isinstance(manifest.get(key), str) and manifest[key] for key in ("run_id", "transaction_id")):
        raise ValueError("Checkpoint identity is missing")
    for name, expected in manifest["files"].items():
        path = directory / _relative(name)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["size"]:
            raise ValueError(f"Checkpoint file missing or changed: {name}")
        if "sha256" in expected and (full or "schema" in expected or expected["size"] <= 16 * 1024 * 1024):
            if _sha(path) != expected["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
        if "schema" in expected and _metadata_schema(path) != expected["schema"]:
            raise ValueError(f"Checkpoint state schema changed: {name}")


def latest_training_checkpoint(output_dir, owners=None, *, native_names=False):
    """Select the latest named checkpoint without imposing a file format.

    Like Accelerator.load_state, actual compatibility is decided by the
    loaders. Atomic saves never publish the staging directory under this name.
    ``owners`` remains accepted for compatibility with existing callers.
    """
    root = Path(output_dir)
    if not root.is_dir():
        return None
    candidates = []
    for path in root.iterdir():
        if path.name.startswith(".") or not path.is_dir():
            continue
        match = (_CHECKPOINT_NAME.fullmatch(path.name) or
                 (re.search(r"(\d+)", path.name) if native_names else None))
        if match:
            candidates.append((int(match.group(1)), path))
    return str(max(candidates)[1]) if candidates else None


def _coordinated_latest_training_checkpoint(output_dir, *, native_names=False):
    """Choose one checkpoint name on rank zero and rebase it on each rank."""
    def choose_name():
        selected = latest_training_checkpoint(output_dir, native_names=native_names)
        return None if selected is None else Path(selected).name

    name = _main_value(choose_name)
    if name is None:
        def require_absent():
            if latest_training_checkpoint(output_dir, native_names=native_names) is not None:
                raise RuntimeError("Rank 0 found no checkpoint, but a rank found one")

        checkpoint_phase(require_absent)
        return None
    return checkpoint_phase(lambda: str(Path(output_dir) / name))


def _rng_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"step", "random_state", "numpy_random_seed", "torch_manual_seed"}
    if torch.cuda.is_available():
        required.add("torch_cuda_manual_seed")
        if len(state.get("torch_cuda_manual_seed", ())) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG topology does not match this process")
    if required - state.keys():
        raise ValueError("Checkpoint RNG state is incomplete")
    # Validate using the public setters, then restore the caller's RNG exactly.
    previous = {"random_state": random.getstate(), "numpy_random_seed": np.random.get_state(),
                "torch_manual_seed": torch.get_rng_state()}
    if torch.cuda.is_available():
        previous["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()
    try:
        _apply_rng(state)
    finally:
        _apply_rng(previous)
    return state


def _apply_rng(state):
    random.setstate(state["random_state"])
    np.random.set_state(state["numpy_random_seed"])
    torch.set_rng_state(state["torch_manual_seed"])
    if "torch_cuda_manual_seed" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_manual_seed"])


def _required_files(owner, directory, *, all_ranks=False, safe_serialization=True):
    if owner.distributed_type.name == "FSDP":
        for index in range(len(owner._models)):
            for stem in ("pytorch_model_fsdp", "optimizer"):
                path = directory / f"{stem}_{index}" / ".metadata"
                if not path.is_file():
                    raise ValueError(f"Missing native FSDP state: {path}")
                from torch.distributed.checkpoint import FileSystemReader
                metadata = FileSystemReader(path.parent).read_metadata()
                for storage in metadata.storage_data.values():
                    payload = path.parent / _relative(storage.relative_path)
                    if not payload.is_file() or payload.stat().st_size < storage.offset + storage.length:
                        raise ValueError(f"Missing or truncated DCP payload: {payload}")
    else:
        from accelerate.utils.constants import OPTIMIZER_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_NAME

        def require_payload(name, kind):
            path = directory / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty native {kind} state: {path}")

        # A save hook may remove entries from Accelerate's weights list and
        # replace them with its own format, so native model files are only
        # predictable when no such hooks are registered.
        if not owner._save_model_state_pre_hook:
            model_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
            for index in range(len(owner._models)):
                name = model_name if index == 0 else model_name.replace(".", f"_{index}.")
                require_payload(name, "model")
        for index in range(len(owner._optimizers)):
            name = f"{OPTIMIZER_NAME}.bin" if index == 0 else f"{OPTIMIZER_NAME}_{index}.bin"
            require_payload(name, "optimizer")
    for index in range(len(owner._schedulers)):
        name = "scheduler.bin" if index == 0 else f"scheduler_{index}.bin"
        if not (directory / name).is_file():
            raise ValueError(f"Missing scheduler state: {name}")
    if owner.scaler is not None and not (directory / "scaler.pt").is_file():
        raise ValueError("Missing scaler state")
    for index in range(len(owner._custom_objects)):
        if not (directory / f"custom_checkpoint_{index}.pkl").is_file():
            raise ValueError("Missing registered checkpoint state")
    ranks = range(owner.num_processes) if all_ranks else (owner.process_index,)
    current_rng = None
    for rank in ranks:
        for validator in checkpoint_config(owner).validators.values():
            validator(directory, rank=rank)
        rng = _rng_state(directory / f"random_states_{rank}.pkl")
        if rank == owner.process_index:
            current_rng = rng
    return current_rng


def _tensor_values(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_values(item)


def check_finite_state(value):
    """Bound temporary validation buffers while checking local tensor state."""
    from torch.distributed._shard.sharded_tensor import ShardedTensor

    for tensor in _tensor_values(value):
        tensors = [shard.tensor for shard in tensor.local_shards()] if isinstance(tensor, ShardedTensor) else [tensor]
        for local in tensors:
            if not local.is_floating_point() and not local.is_complex():
                continue
            for chunk in local.detach().reshape(-1).split(1024 * 1024):
                if not torch.isfinite(chunk).all().item():
                    raise ValueError("Checkpoint contains NaN or Inf")


def _set_failed(owners, failed):
    from .. import _failed_checkpoint_owners

    for owner in owners:
        config = checkpoint_config(owner)
        if failed:
            _failed_checkpoint_owners.add(owner)
            if not config.step_guards:
                owner_ref = ref(owner)
                def guard(optimizer, args, kwargs, owner_ref=owner_ref):
                    if owner_ref() in _failed_checkpoint_owners:
                        raise RuntimeError("Checkpoint restore failed; reload successfully before updating parameters")
                for optimizer in owner._optimizers:
                    register = getattr(_raw_optimizer(optimizer), "register_step_pre_hook", None)
                    if callable(register):
                        config.step_guards.append(register(guard))
        else:
            _failed_checkpoint_owners.discard(owner)
            for handle in config.step_guards:
                handle.remove()
            config.step_guards.clear()


def _native(owner, method, *args, **kwargs):
    function = getattr(type(owner), method)
    function = getattr(function, "_videox_native_checkpoint", function)
    return function(owner, *args, **kwargs)


@contextmanager
def _synchronized_hooks(owner, *, load_only=False):
    """Make local hook failures visible before the next native phase."""
    changes = []
    hook_sets = ((owner._load_model_state_pre_hook,) if load_only else
                 (owner._save_model_state_pre_hook,))
    for hooks in hook_sets:
        for key, function in list(hooks.items()):
            def synchronized(*args, function=function):
                return checkpoint_phase(lambda: function(*args))
            changes.append((hooks, key, function, synchronized))
            hooks[key] = synchronized
    try:
        yield
    finally:
        for hooks, key, function, synchronized in changes:
            # Preserve hook removal or replacement performed by user code.
            if hooks.get(key) is synchronized:
                hooks[key] = function


def _synchronize_load_hooks(owner):
    """Use synchronized load hooks only when their structure matches on all ranks."""
    config = checkpoint_config(owner)
    if not dist.is_initialized() or not config.private_states:
        return False
    signature = [key in config.private_load_hook_ids for key in owner._load_model_state_pre_hook]
    signatures = [None] * dist.get_world_size()
    dist.all_gather_object(signatures, signature)
    return any(signature) and all(other == signature for other in signatures)


def _load_native(owner, path, kwargs):
    if _synchronize_load_hooks(owner):
        with _synchronized_hooks(owner, load_only=True):
            return _native(owner, "load_state", str(path), **kwargs)
    return _native(owner, "load_state", str(path), **kwargs)



def _save_native(owner, path, safe_serialization, kwargs):
    project = owner.project_configuration
    automatic = project.automatic_checkpoint_naming
    project.automatic_checkpoint_naming = False
    try:
        with _synchronized_hooks(owner):
            return _native(owner, "save_state", str(path), safe_serialization=safe_serialization, **kwargs)
    finally:
        project.automatic_checkpoint_naming = automatic


def _save_preflight(owners):
    from .. import _failed_checkpoint_owners

    descriptions = {}
    for name, owner in owners.items():
        if owner in _failed_checkpoint_owners:
            raise ValueError("Cannot save after a failed restore; reload successfully first")
        descriptions[name] = _descriptor(owner)
        def has_nonzero_gradient(parameter):
            grad = parameter.grad
            if grad is None:
                return False
            if grad.layout != torch.strided:
                grad = grad.coalesce().values() if grad.layout == torch.sparse_coo else grad.values()
            return bool(torch.count_nonzero(grad).item())
        if not owner.sync_gradients or any(has_nonzero_gradient(parameter)
                                           for model in owner._models for parameter in model.parameters()):
            raise ValueError("Save checkpoints after an optimizer update and clearing gradients")
        if checkpoint_config(owner).check_finite:
            check_finite_state([list(model.parameters()) for model in owner._models])
            check_finite_state([_raw_optimizer(optimizer).state for optimizer in owner._optimizers])
    return descriptions


def _save_agreement_descriptors(descriptions):
    """Compare structural state while keeping optional metadata informational."""
    result = {}
    for name, descriptor in descriptions.items():
        shared = dict(descriptor)
        shared.pop("training", None)
        shared["models"] = [{key: value for key, value in model.items() if key != "config"}
                            for model in descriptor["models"]]
        shared["schedulers"] = [{key: value for key, value in scheduler.items() if key != "functions"}
                                for scheduler in descriptor["schedulers"]]
        result[name] = shared
    return result


def _cleanup_save_stage(stage, guard):
    """Release the lock even if removing the staging directory fails."""
    errors = []
    try:
        if stage and Path(stage).exists():
            shutil.rmtree(stage)
    except Exception as exc:
        errors.append(f"could not remove staging directory {stage}: {type(exc).__name__}: {exc}")
    finally:
        if guard is not None:
            try:
                guard.release()
            except Exception as exc:
                errors.append(f"could not release checkpoint lock {guard.path}: {type(exc).__name__}: {exc}")
    return errors


def _prune_checkpoints(parent, current, limit, run_id, *, marker, components=None):
    """Prune matching old checkpoints without changing the published save result."""
    errors, candidates = [], []
    try:
        for candidate in parent.iterdir():
            match = _CHECKPOINT_NAME.fullmatch(candidate.name)
            file = candidate / marker
            if not match or candidate.is_symlink() or not candidate.is_dir() or not file.is_file():
                continue
            try:
                value = json.loads(file.read_text())
            except (OSError, ValueError):
                continue
            if (isinstance(value, dict) and value.get("run_id") == run_id
                    and (components is None or value.get("components") == components)):
                candidates.append((int(match.group(1)), candidate))
        excess = max(0, len(candidates) - limit)
        removable = [candidate for _, candidate in sorted(candidates) if candidate != current]
        for candidate in removable[:excess]:
            try:
                shutil.rmtree(candidate)
            except Exception as exc:
                errors.append(f"could not remove {candidate}: {type(exc).__name__}: {exc}")
    except Exception as exc:
        errors.append(f"could not scan {parent}: {type(exc).__name__}: {exc}")
    return errors


def _report_checkpoint_maintenance(context, errors):
    if errors and (not dist.is_initialized() or dist.get_rank() == 0):
        from .. import _checkpoint_warning
        _checkpoint_warning(f"{context}: " + "; ".join(errors))


def _stage_was_committed(path, identity, committed):
    """Distinguish our published directory from an unrelated destination."""
    if committed:
        return True
    if identity is None:
        return False
    try:
        st = os.stat(path, follow_symlinks=False)
        return (st.st_dev, st.st_ino) == identity
    except OSError:
        return False


def _process_start_time(pid):
    """Return a Linux process start token, None if absent, or False if unknown."""
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text()
        return text.rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        # hidepid may conceal a live process; only ESRCH proves absence.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except OSError:
            return False
        return False
    except (OSError, ValueError, IndexError):
        return False


def _local_lock_identity():
    try:
        return {"host": os.uname().nodename,
                "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "pid_namespace": os.readlink("/proc/self/ns/pid")}
    except (OSError, AttributeError):
        return None


class _CheckpointLock:
    """NFS-compatible exclusive lock with conservative same-host stale recovery.

    O_EXCL-compatible hard-link publication also excludes older writers. Only
    locks from this process namespace with a provably dead owner are recovered;
    unknown/legacy/remote locks require manual verification.
    """
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / ".training-checkpoints.lock"
        self.inode = None

    def _stale_local_owner(self):
        try:
            owner = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return False
        identity = _local_lock_identity()
        if (not isinstance(owner, dict) or owner.get("format") != "videox-checkpoint-lock-v1"
                or identity is None or any(owner.get(key) != value for key, value in identity.items())
                or not isinstance(owner.get("pid"), int) or owner["pid"] <= 0
                or not isinstance(owner.get("start"), str)):
            return False
        current = _process_start_time(owner["pid"])
        return current is None or (current is not False and current != owner["start"])

    def _publish_owner(self, source):
        source_stat = os.stat(source)
        try:
            os.link(source, self.path)
            return source_stat.st_dev, source_stat.st_ino
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS, errno.EXDEV):
                raise
            # Some filesystems do not support hard links. Preserve the original
            # O_EXCL behavior; a crash during this short write may leave an
            # unverifiable lock that requires manual inspection.
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            created = os.fstat(fd)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(Path(source).read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                try:
                    current = self.path.stat()
                    if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                        self.path.unlink()
                except OSError:
                    pass
                raise
            return created.st_dev, created.st_ino

    def acquire(self):
        identity = _local_lock_identity() or {}
        owner = {"format": "videox-checkpoint-lock-v1", **identity, "pid": os.getpid(),
                 "start": _process_start_time(os.getpid()), "token": uuid.uuid4().hex}
        fd, name = tempfile.mkstemp(prefix=".training-checkpoint-owner-", dir=self.directory)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(owner, stream)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                self.inode = self._publish_owner(name)
            except FileExistsError:
                # On NFS mounts with local-only flock, this guards only recovery
                # for processes proven to be in this same PID namespace.
                if fcntl is None:
                    raise FileExistsError(f"Checkpoint lock cannot be safely recovered: {self.path}")
                parent_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    fcntl.flock(parent_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if not self._stale_local_owner():
                        raise FileExistsError(f"Checkpoint lock is active or cannot be safely recovered: {self.path}")
                    self.path.unlink()
                    self.inode = self._publish_owner(name)
                finally:
                    os.close(parent_fd)
        finally:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass
            except OSError as exc:
                from .. import _checkpoint_warning
                _checkpoint_warning(f"Could not remove temporary checkpoint lock owner file {name}: {exc}")

    def release(self):
        if self.inode is None:
            return
        try:
            st = self.path.stat()
            if (st.st_dev, st.st_ino) == self.inode:
                self.path.unlink()
        except FileNotFoundError:
            pass
        self.inode = None


def save_training_checkpoint(owners, output_dir=None, *, checkpoint_limit=None, safe_serialization=True, **kwargs):
    """Atomically publish one Accelerator or a named group of Accelerators."""
    owners = checkpoint_phase(lambda: _participants(owners))
    primary = next(iter(owners.values()))
    if primary.project_configuration.automatic_checkpoint_naming and output_dir is None:
        output_dir = Path(primary.project_dir) / "checkpoints" / f"checkpoint_{primary.save_iteration}"
        if checkpoint_limit is None:
            checkpoint_limit = primary.project_configuration.total_limit
    path = checkpoint_phase(lambda: Path(output_dir).expanduser().absolute())
    descriptions = checkpoint_phase(lambda: _save_preflight(owners))
    policies = {name: {"finite": checkpoint_config(owner).check_finite, "checksum": checkpoint_config(owner).checksum}
                for name, owner in owners.items()}
    _agree([str(path), _save_agreement_descriptors(descriptions), policies, checkpoint_limit, _save_hook_counts(owners)])
    if checkpoint_limit is not None and checkpoint_limit < 1:
        raise ValueError("checkpoint_limit must be at least 1")
    run_ids = {checkpoint_config(owner).run_id for owner in owners.values()} - {None}
    _agree(sorted(run_ids))
    if len(run_ids) > 1:
        raise ValueError("Checkpoint participants belong to different training runs")
    run_id = _main_value(lambda: next(iter(run_ids), None) or uuid.uuid4().hex)
    lock, stage = None, None
    committed = [False]
    stage_identity = [None]
    held_lock = [None]
    iterations = {owner: owner.project_configuration.iteration for owner in owners.values()}

    def reserve():
        path.parent.mkdir(parents=True, exist_ok=True)
        guard = _CheckpointLock(path.parent)
        guard.acquire()
        held_lock[0] = guard
        try:
            if os.path.lexists(path):
                raise ValueError(f"Refusing to overwrite an existing checkpoint: {path}")
            stage_path = tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent)
            return str(guard.path), stage_path
        except BaseException:
            _report_checkpoint_maintenance("Checkpoint lock cleanup failed", _cleanup_save_stage(None, guard))
            held_lock[0] = None
            raise

    lock, stage = _main_value(reserve)
    try:
        for name, owner in owners.items():
            checkpoint_phase(lambda owner=owner, name=name: _save_native(owner, Path(stage) / name, safe_serialization, kwargs))
            checkpoint_phase(lambda owner=owner, name=name: _required_files(
                owner, Path(stage) / name, safe_serialization=safe_serialization))

        def publish():
            directory = Path(stage)
            checksum = "full" if any(checkpoint_config(owner).checksum == "full" for owner in owners.values()) else "metadata"
            manifest = {"protocol": PROTOCOL, "status": "complete", "run_id": run_id,
                        "transaction_id": uuid.uuid4().hex, "components": descriptions,
                        "versions": {"accelerate": "1.7.0", "torch": "2.7.0"},
                        "integrity": checksum, "files": _inventory(directory, checksum)}
            with (directory / MANIFEST).open("x") as stream:
                json.dump(manifest, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            if os.path.lexists(path):
                raise ValueError("Checkpoint destination appeared while saving")
            st = os.stat(directory, follow_symlinks=False)
            stage_identity[0] = (st.st_dev, st.st_ino)
            os.rename(directory, path)
            committed[0] = True
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        _main_value(publish)
        for owner in owners.values():
            checkpoint_config(owner).run_id = run_id
        if checkpoint_limit is not None:
            errors = _main_value(lambda: _prune_checkpoints(
                path.parent, path, checkpoint_limit, run_id, marker=MANIFEST))
            _report_checkpoint_maintenance("Checkpoint saved, but retention cleanup failed", errors)
    except BaseException:
        # Native save increments this counter even before a later component fails.
        if _main_value(lambda: _stage_was_committed(path, stage_identity[0], committed[0])):
            for owner in owners.values():
                checkpoint_config(owner).run_id = run_id
        else:
            for owner, iteration in iterations.items():
                owner.project_configuration.iteration = iteration
        raise
    finally:
        errors = _main_value(lambda: _cleanup_save_stage(stage, held_lock[0]))
        _report_checkpoint_maintenance("Checkpoint staging cleanup failed", errors)
    return path



def _checkpoint_run_id(path):
    """Read optional save metadata without making it a restore requirement."""
    def identity(directory):
        for name in (MANIFEST, ".videox_native_checkpoint.json"):
            try:
                value = json.loads((directory / name).read_text())
                run_id = value.get("run_id") if isinstance(value, dict) else None
                if isinstance(run_id, str) and run_id:
                    return run_id
            except (OSError, ValueError, TypeError):
                continue
        return None

    run_id = identity(path)
    if run_id:
        return run_id
    # A sibling's identity does not prove the legacy checkpoint belongs to the
    # same run. Starting a new identity keeps retention from deleting another
    # run's checkpoints while leaving native restore semantics unchanged.
    return uuid.uuid4().hex


def load_training_checkpoint(owners, input_dir=None, **kwargs):
    """Load each registered participant using Accelerate's native semantics."""
    owners = checkpoint_phase(lambda: _participants(owners))
    primary = next(iter(owners.values()))
    # These values control which collectives each rank enters below (including
    # private RNG hooks). They do not constrain the checkpoint's file format.
    _agree([list(owners), input_dir is None,
            input_dir is None and primary.project_configuration.automatic_checkpoint_naming,
            [(owner.distributed_type.name, checkpoint_config(owner).check_finite,
              len(checkpoint_config(owner).private_states)) for owner in owners.values()]])
    if input_dir is None and primary.project_configuration.automatic_checkpoint_naming:
        input_dir = _coordinated_latest_training_checkpoint(
            Path(primary.project_dir) / "checkpoints", native_names=True)
    if input_dir is None:
        raise ValueError("No checkpoint directory was provided or found")
    path = checkpoint_phase(lambda: Path(input_dir).expanduser().absolute())
    try:
        def require_directories():
            for name in owners:
                directory = path / name
                if not directory.is_dir():
                    raise ValueError(f"Tried to find {directory} but folder does not exist")

        # Accelerate requires these directories too; check all ranks before
        # any model enters native distributed loading collectives.
        checkpoint_phase(require_directories)
        for name, owner in owners.items():
            checkpoint_phase(lambda owner=owner, name=name:
                             _load_native(owner, path / name, kwargs))
            if checkpoint_config(owner).check_finite:
                checkpoint_phase(lambda: check_finite_state([list(model.parameters()) for model in owner._models]))
                if owner.distributed_type.name != "FSDP":
                    checkpoint_phase(lambda: check_finite_state([_raw_optimizer(optimizer).state for optimizer in owner._optimizers]))
    except BaseException:
        _set_failed(owners.values(), True)
        raise
    _set_failed(owners.values(), False)
    run_id = _main_value(lambda: _checkpoint_run_id(path))
    for owner in owners.values():
        checkpoint_config(owner).run_id = run_id
        if owner.project_configuration.automatic_checkpoint_naming:
            match = _CHECKPOINT_NAME.fullmatch(path.name)
            if match:
                owner.project_configuration.iteration = max(owner.save_iteration, int(match.group(1)) + 1)


def dispatch_checkpoint(owner, method, original, *args, **kwargs):
    """Agree on FSDP dispatch before choosing native or protected checkpoints."""
    from .. import (_accelerate_patch_requested, _accelerate_patch_owners, _checkpoint_warning,
                    _accelerate_patch_disabled_owners, _supported_accelerate_versions,
                    _eligible_fsdp_checkpoint, _failed_checkpoint_owners)

    enabled = ((owner in _accelerate_patch_owners or _accelerate_patch_requested())
               and owner not in _accelerate_patch_disabled_owners)
    failed = owner in _failed_checkpoint_owners
    plugin = owner.state.fsdp_plugin
    _agree([enabled, _supported_accelerate_versions(), owner.distributed_type.name,
            getattr(plugin, "fsdp_version", 1), getattr(plugin.state_dict_type, "name", None), failed])
    if failed and method == "save_state":
        raise RuntimeError("Cannot save after a failed restore; reload successfully first")

    def native_fallback():
        if method == "load_state" and failed:
            result = checkpoint_phase(lambda: original(owner, *args, **kwargs))
            _set_failed([owner], False)
            return result
        return original(owner, *args, **kwargs)

    if not enabled or not _eligible_fsdp_checkpoint(plugin):
        return native_fallback()
    available = _protected_checkpoint_available({"": owner})
    _agree(available)
    if not available:
        if not dist.is_initialized() or dist.get_rank() == 0:
            _checkpoint_warning("This Accelerator configuration is not supported by protected checkpoints; "
                                "using native Accelerate state")
        return native_fallback()
    if method == "save_state":
        return save_training_checkpoint(owner, *args, **kwargs)
    return load_training_checkpoint(owner, *args, **kwargs)


def patch_accelerate_fsdp(accelerator):
    """Explicitly enable FSDP1 sharded restore for an instance.

    Repository imports enable automatic dispatch by default. This opt-in API
    also works with VIDEOX_ACCELERATE_PATCHES=0 and checks supported versions.
    """
    from .. import enable_accelerate_patch

    return enable_accelerate_patch(accelerator)


def unpatch_accelerate_fsdp(accelerator):
    """Disable automatic and explicit patch dispatch for this instance."""
    from .. import disable_accelerate_patch

    disable_accelerate_patch(accelerator)


def _checkpoint_phase(function, process_group):
    """Propagate local preflight failures before entering FSDP collectives."""
    import torch.distributed as dist

    result, error = None, None
    try:
        result = function()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size(process_group)
    dist.all_gather_object(errors, error, group=process_group)
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
    if failures:
        raise ValueError("FSDP optimizer restore failed: " + "; ".join(failures))
    return result


def _optimizer_layout(model, optimizer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp._common_utils import _get_param_to_fqns

    names = _get_param_to_fqns(model)
    groups = [[name for parameter in group["params"] for name in names[parameter]]
              for group in optimizer.param_groups]
    shapes = {name: parameter.shape for parameter, fqns in names.items() for name in fqns}
    for unit in FSDP.fsdp_modules(model):
        if unit._handle is None:
            continue
        flat = unit._handle.flat_param
        if unit._use_orig_params:
            for parameter, shape in zip(flat._params, flat._shapes):
                for name in names[parameter]:
                    shapes[name] = shape
        else:
            shapes.update(zip(names[flat], flat._shapes))
    return groups, shapes


def _optimizer_load_destination(metadata, shapes, process_group):
    """Allocate only this rank's CPU chunks, directly from saved metadata.

    Do not derive the schema from live optimizer state: it may be empty, partial,
    or use different options. PyTorch's optimizer-specific DCP loader allocates
    full CUDA tensors before sharding on NCCL, which is too large for CPU offload.
    These native ShardedTensors use the same dim-0 layout without that allocation.
    """
    import math
    import torch.distributed as dist
    from torch.distributed.checkpoint import TensorStorageMetadata
    from torch.distributed.checkpoint._nested_dict import unflatten_state_dict
    from torch.distributed._shard.sharded_tensor import Shard, ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata, TensorProperties
    from torch.distributed._shard.metadata import ShardMetadata

    rank, world_size = dist.get_rank(process_group), dist.get_world_size(process_group)
    flat = {}
    for key, stored in metadata.state_dict_metadata.items():
        path = metadata.planner_data[key]
        if not path or path[0] != "optimizer":
            raise ValueError(f"Unexpected optimizer checkpoint entry: {key}")
        if not isinstance(stored, TensorStorageMetadata):
            flat[key] = None
            continue
        size = stored.size
        if len(path) >= 3 and path[1] == "state":
            if path[2] not in shapes:
                raise ValueError(f"Unknown optimizer parameter: {path[2]}")
            is_moment = path[-1] in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
            if (len(size) or is_moment) and tuple(size) != tuple(shapes[path[2]]):
                raise ValueError(f"Optimizer checkpoint tensor shape mismatch: {key}")
        if not len(size) or path[1] != "state":
            flat[key] = torch.empty(size, dtype=stored.properties.dtype, device="cpu")
            continue
        chunk = max(1, math.ceil(size[0] / world_size))
        shards, local = [], []
        for shard_rank, offset in enumerate(range(0, size[0], chunk)):
            offsets = [offset] + [0] * (len(size) - 1)
            lengths = [min(chunk, size[0] - offset)] + list(size[1:])
            shard = ShardMetadata(offsets, lengths,
                                  f"rank:{dist.get_global_rank(process_group, shard_rank)}/cpu")
            shards.append(shard)
            if shard_rank == rank:
                local.append(Shard(torch.empty(lengths, dtype=stored.properties.dtype, device="cpu"), shard))
        properties = TensorProperties(dtype=stored.properties.dtype, requires_grad=False)
        flat[key] = ShardedTensor._init_from_local_shards_and_global_metadata(
            local, ShardedTensorMetadata(shards, size, properties), process_group=process_group,
        )
    return unflatten_state_dict(flat, metadata.planner_data)


def _align_optimizer_groups(state, expected_groups, optimizer, *, allow_reorder=True):
    """Match groups by parameter names, including reordered and empty groups."""
    # DCP omits empty dictionaries (e.g. a checkpoint before the first step).
    state.setdefault("state", {})
    saved = state["param_groups"]
    if len(saved) != len(expected_groups):
        raise ValueError("Optimizer checkpoint parameter group count differs")
    if not allow_reorder and [group["params"] for group in saved] != expected_groups:
        raise ValueError("Parameter group reordering with a scheduler is not supported")
    remaining, aligned = list(saved), []
    for expected in expected_groups:
        matches = [index for index, group in enumerate(remaining)
                   if len(group["params"]) == len(expected) and set(group["params"]) == set(expected)]
        if not matches:
            raise ValueError("Optimizer checkpoint parameter groups differ from the prepared optimizer")
        group = dict(remaining.pop(matches[0]), params=list(expected))
        required_options = set(optimizer.defaults)
        if required_options - group.keys():
            raise ValueError("Optimizer checkpoint options do not match this optimizer type")
        current = optimizer.param_groups[len(aligned)]
        # Tensor learning rates may live on CUDA (notably capturable Adam).
        # DCP loads on CPU; retain the runtime device when restoring options.
        for key, value in group.items():
            if isinstance(value, torch.Tensor):
                if isinstance(current.get(key), torch.Tensor):
                    group[key] = value.to(current[key].device)
                elif key == "lr" and group.get("capturable", False) and current["params"]:
                    group[key] = value.to(current["params"][0].device)
        aligned.append(group)
    expected_names = {name for group in expected_groups for name in group}
    if set(state["state"]) - expected_names:
        raise ValueError("Optimizer checkpoint contains state outside its parameter groups")
    # A parameter may legitimately have no state before its first update. Once
    # initialized, Adam needs all its moments, including the saved AMSGrad option.
    if type(optimizer) in (torch.optim.Adam, torch.optim.AdamW):
        for group in aligned:
            required = {"step", "exp_avg", "exp_avg_sq"}
            if group.get("amsgrad", False):
                required.add("max_exp_avg_sq")
            for name in group["params"]:
                if name not in state["state"]:
                    continue
                values = state["state"][name]
                if required - values.keys():
                    raise ValueError(f"Incomplete Adam optimizer state for {name}")
                if any(not isinstance(values[key], torch.Tensor) for key in required):
                    raise ValueError(f"Invalid Adam optimizer tensor state for {name}")
                if values["step"].ndim != 0:
                    raise ValueError(f"Invalid Adam optimizer step for {name}")
    state["param_groups"] = aligned


class _OptimizerAdapterFallback(Exception):
    """The adapter failed before changing optimizer state; native load may proceed."""


def _load_generic_fsdp_optimizer(accelerator, optimizer, model, input_dir, optimizer_index):
    """Restore an unknown optimizer from checkpoint metadata, including fresh state."""
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    plugin = accelerator.state.fsdp_plugin
    path = Path(input_dir) / f"optimizer_{optimizer_index}"
    try:
        reader = FileSystemReader(path)
        _checkpoint_phase(reader.read_metadata, model.process_group)
        with FSDP.state_dict_type(model, plugin.state_dict_type,
                                  plugin.state_dict_config, plugin.optim_state_dict_config):
            model_state = model.state_dict()
            state = _checkpoint_phase(lambda: load_sharded_optimizer_state_dict(
                model_state, "optimizer", reader)["optimizer"], model.process_group)
            local = _checkpoint_phase(lambda: FSDP.optim_state_dict_to_load(model, optimizer, state),
                                      model.process_group)
    except Exception as exc:
        raise _OptimizerAdapterFallback(str(exc)) from exc
    # This may mutate local optimizer state, so propagate any rank-local
    # failure to every rank without falling back to another loader.
    _checkpoint_phase(lambda: optimizer.load_state_dict(local), model.process_group)


def _load_sharded_fsdp_optimizer(accelerator, optimizer, model, input_dir, optimizer_index):
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from .fsdp_utils import release_completed_fsdp_gradients

    optimizer = getattr(optimizer, "optimizer", optimizer)
    process_group = model.process_group
    checkpoint_dir = Path(input_dir) / f"optimizer_{optimizer_index}"

    def prepare():
        if len(accelerator._optimizers) != len(accelerator._models):
            raise ValueError("FSDP checkpoint restore requires one optimizer per prepared model")
        for unit in FSDP.fsdp_modules(model):
            if (unit.process_group is not process_group or unit._device_mesh is not None
                    or dist.get_world_size(unit.process_group) != accelerator.num_processes
                    or unit.sharding_strategy not in (ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP)):
                raise ValueError("FSDP layout is outside the low-memory optimizer adapter")
        groups, shapes = _optimizer_layout(model, optimizer)
        metadata = FileSystemReader(checkpoint_dir).read_metadata()
        if getattr(accelerator.state.fsdp_plugin.cpu_offload, "offload_params", False):
            # Model restore creates idle pinned gradients. Free these before
            # allocating moments, selecting each model independently: a single
            # Accelerator can contain both original and flat parameter layouts.
            original_param_models = [prepared for prepared in accelerator._models
                                     if all(unit._use_orig_params for unit in FSDP.fsdp_modules(prepared))]
            release_completed_fsdp_gradients(*original_param_models)
        return groups, shapes, metadata

    try:
        expected_groups, shapes, metadata = _checkpoint_phase(prepare, process_group)
        state = _checkpoint_phase(
            lambda: _optimizer_load_destination(metadata, shapes, process_group), process_group,
        )
        # No optimizer state has been mutated before the native-fallback boundary.
        _checkpoint_phase(lambda: load(state, checkpoint_id=checkpoint_dir, process_group=process_group,
                                       planner=DefaultLoadPlanner(flatten_sharded_tensors=False)), process_group)
        saved_groups = _checkpoint_phase(lambda: [group["params"] for group in
                                                   state["optimizer"]["param_groups"]], process_group)
        def warn_reordered_groups():
            if accelerator._schedulers and saved_groups != expected_groups and accelerator.is_main_process:
                from .. import _checkpoint_warning
                _checkpoint_warning("Optimizer parameter groups changed order; their state is matched by name, "
                                    "but scheduler-specific group state may need adjustment")
        _checkpoint_phase(warn_reordered_groups, process_group)
        _checkpoint_phase(lambda: _align_optimizer_groups(state["optimizer"], expected_groups, optimizer),
                          process_group)
    except Exception as exc:
        raise _OptimizerAdapterFallback(str(exc)) from exc
    if checkpoint_config(accelerator).check_finite:
        _checkpoint_phase(lambda: check_finite_state(state["optimizer"]), process_group)
    _restore_fsdp_optimizer_state(model, optimizer, state["optimizer"])


@torch.no_grad()
def _restore_fsdp_optimizer_state(model, optimizer, state):
    """Release superseded moments before native FSDP conversion.

    All saved leaves and parameter groups have already been read and validated.
    Custom optimizers and load pre-hooks retain access to the original state.
    No dummy optimizer step is needed, even when restoring an empty optimizer.
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP, StateDictType,
        ShardedStateDictConfig, ShardedOptimStateDictConfig,
    )

    if (type(optimizer) in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD)
            and not optimizer._optimizer_load_state_dict_pre_hooks):
        optimizer.state.clear()
    with FSDP.state_dict_type(
        model, StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
        ShardedOptimStateDictConfig(offload_to_cpu=True),
    ):
        local = _checkpoint_phase(lambda: FSDP.optim_state_dict_to_load(model, optimizer, state), model.process_group)
    _checkpoint_phase(lambda: optimizer.load_state_dict(local), model.process_group)


def register_checkpoint_state(
    accelerator, generators, progress, *, filename="training_state_{rank}.pt", progress_keys=None,
):
    """Save private noise streams and consumed batches separately for each rank."""

    template = filename
    filename = filename.format(rank=accelerator.process_index)
    progress_keys = tuple(progress) if progress_keys is None else tuple(progress_keys)
    config = checkpoint_config(accelerator)
    config.private_states[template] = {
        "schema_version": 1,
        "generators": {name: generator.device.type for name, generator in generators.items()},
        "progress_keys": list(progress_keys),
    }

    def read_state(directory, *, rank=None):
        path = Path(directory) / template.format(rank=accelerator.process_index if rank is None else rank)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if set(state["generators"]) != set(generators) or set(progress_keys) - state["progress"].keys():
            raise ValueError("Checkpoint private RNG/progress schema differs")
        for name, generator in generators.items():
            previous = generator.get_state()
            try:
                generator.set_state(state["generators"][name])
            finally:
                generator.set_state(previous)
        return state

    config.validators[template] = read_state

    def save_state(models, weights, output_dir):
        path = Path(output_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"generators": {name: g.get_state() for name, g in generators.items()},
                    "progress": {key: progress[key] for key in progress_keys}}, path)

    def load_state(models, input_dir):
        state, error = None, None
        try:
            state = read_state(input_dir)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [error]
        if dist.is_initialized():
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
        if any(errors):
            def warn_private_fallback():
                if accelerator.is_main_process:
                    from .. import _checkpoint_warning
                    _checkpoint_warning("Checkpoint private RNG/data progress is missing or incompatible; "
                                        "resuming with the current initialization and training progress fallback. "
                                        "Continuation will not be exact. " +
                                        "; ".join(f"rank {i}: {value}" for i, value in enumerate(errors) if value))
            checkpoint_phase(warn_private_fallback)
            return
        for name, generator in generators.items():
            generator.set_state(state["generators"][name])
        progress.update(state["progress"], restored=True)

    save_handle = accelerator.register_save_state_pre_hook(save_state)
    load_handle = accelerator.register_load_state_pre_hook(load_state)
    config.private_load_hook_ids.add(load_handle.id)
    return save_handle, load_handle


def check_finite_training_value(value, name):
    if value is not None and not torch.isfinite(torch.as_tensor(value)).all():
        raise FloatingPointError(f"Non-finite {name}; refusing to update/save corrupted training weights")


def _protected_checkpoint_available(owners):
    """Select native fallback for configurations outside the validated adapter."""
    from .. import (_accelerate_patch_requested, _accelerate_patch_owners,
                    _accelerate_patch_disabled_owners, _supported_accelerate_versions, _eligible_fsdp_checkpoint)

    if not _supported_accelerate_versions():
        return False
    for owner in owners.values():
        if owner in _accelerate_patch_disabled_owners:
            return False
        if owner not in _accelerate_patch_owners and not _accelerate_patch_requested():
            return False
        mode = owner.distributed_type.name
        if mode not in ("FSDP", "NO", "MULTI_GPU"):
            return False
        if mode == "FSDP" and not _eligible_fsdp_checkpoint(owner.state.fsdp_plugin):
            return False
        if (not owner._models or len(owner._models) != len(owner._optimizers)
                or owner.project_configuration.save_on_each_node):
            return False
        if any(type(_raw_optimizer(opt)) not in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD)
               for opt in owner._optimizers):
            return False
        if mode != "FSDP":
            # The protected descriptor pairs each optimizer with one model.
            # Native Accelerate also supports other registration layouts.
            for model, wrapped_optimizer in zip(owner._models, owner._optimizers):
                parameters = {id(parameter) for parameter in model.parameters()}
                if any(id(parameter) not in parameters
                       for group in _raw_optimizer(wrapped_optimizer).param_groups
                       for parameter in group["params"]):
                    return False
        if mode == "FSDP":
            try:
                validate_checkpoint_owner(owner)
            except (ValueError, AttributeError, TypeError):
                return False
    return True



class AcceleratorCheckpoint:
    """Checkpoint registration and standard Accelerator save/load dispatch.

    Registered participants are saved together. Supported configurations use
    the protected format; other backends and explicit opt-outs use Accelerate's
    native format. Training code does not need to choose the implementation.
    """

    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.owners = {"": accelerator}
        self.checkpoint_limit = None
        self.configured = False
        self._warned_fallback = False
        self._native_run_id = None

    def configure(self, *, model_ids=None, companions=None, generators=None, progress=None,
                  filename="training_state_{rank}.pt", progress_keys=None, checkpoint_limit=None, **options):
        if self.configured:
            raise RuntimeError("Checkpoint registration must only happen once per Accelerator")
        owners = {"": self.accelerator}
        roles = {"": model_ids}
        for name, (owner, owner_roles) in (companions or {}).items():
            owners[name] = owner
            roles[name] = owner_roles
        self.owners = _participants(owners)
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if (len(self.owners) > 1 and any(owner.project_configuration.save_on_each_node
                                        and owner.num_processes > local_world for owner in self.owners.values())):
            raise ValueError("Multi-node grouped checkpoints with save_on_each_node require shared storage")
        if checkpoint_limit is not None and checkpoint_limit < 1:
            raise ValueError("checkpoint_limit must be at least 1")
        self.checkpoint_limit = checkpoint_limit
        for name, owner in self.owners.items():
            configure_checkpointing(owner, model_ids=roles[name], _validate=False, **options)
        if generators is not None:
            if progress is None:
                raise ValueError("Checkpoint progress is required with private generators")
            register_checkpoint_state(self.accelerator, generators, progress, filename=filename,
                                      progress_keys=progress_keys)
        self.configured = True
        return self

    def _protected(self):
        return _protected_checkpoint_available(self.owners)

    def _use_protected(self, *, automatic, native_per_node):
        from .. import _checkpoint_warning

        protected = self._protected()
        _agree([protected, automatic, native_per_node])
        if not protected and not self._warned_fallback:
            if self.accelerator.is_main_process:
                _checkpoint_warning("Checkpoint protection is unavailable or disabled; using native Accelerate state "
                                    "without manifest configuration checks")
            self._warned_fallback = True
        return protected

    def save_state(self, output_dir=None, *, safe_serialization=True, **kwargs):
        automatic = self.accelerator.project_configuration.automatic_checkpoint_naming
        native_per_node = self.accelerator.project_configuration.save_on_each_node and len(self.owners) == 1
        projects = {id(owner.project_configuration): (owner.project_configuration, owner.project_configuration.iteration)
                    for owner in self.owners.values()}
        if automatic:
            output_dir = None
        if self._use_protected(automatic=automatic, native_per_node=native_per_node):
            result = save_training_checkpoint(self.owners, output_dir, checkpoint_limit=self.checkpoint_limit,
                                              safe_serialization=safe_serialization, **kwargs)
        elif native_per_node:
            from .. import _failed_checkpoint_owners

            def require_restored():
                if self.accelerator in _failed_checkpoint_owners:
                    raise RuntimeError("Cannot save after a failed restore; reload successfully first")

            checkpoint_phase(require_restored)
            result = _native(self.accelerator, "save_state", output_dir, safe_serialization=safe_serialization, **kwargs)
        else:
            result = self._save_native_group(output_dir, safe_serialization, kwargs)
        if automatic:
            for project, iteration in projects.values():
                project.iteration = iteration + 1
        return result

    def load_state(self, input_dir=None, **kwargs):
        result = load_training_checkpoint(self.owners, input_dir, **kwargs)
        self._native_run_id = checkpoint_config(self.accelerator).run_id
        return result

    def _save_native_group(self, output_dir, safe_serialization, kwargs):
        from .. import _failed_checkpoint_owners
        if any(owner in _failed_checkpoint_owners for owner in self.owners.values()):
            raise RuntimeError("Cannot save after a failed restore; reload successfully first")
        if output_dir is None:
            if not self.accelerator.project_configuration.automatic_checkpoint_naming:
                raise ValueError("output_dir is required for a checkpoint group")
            output_dir = Path(self.accelerator.project_dir) / "checkpoints" / f"checkpoint_{self.accelerator.save_iteration}"
        path = checkpoint_phase(lambda: Path(output_dir).expanduser().absolute())
        _agree([str(path), _save_hook_counts(self.owners)])
        run_id = _main_value(lambda: self._native_run_id or uuid.uuid4().hex)
        held_lock = [None]
        committed = [False]
        stage_identity = [None]
        def reserve():
            path.parent.mkdir(parents=True, exist_ok=True)
            guard = _CheckpointLock(path.parent)
            guard.acquire()
            held_lock[0] = guard
            try:
                if os.path.lexists(path):
                    raise ValueError(f"Refusing to overwrite an existing checkpoint: {path}")
                return str(guard.path), tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent)
            except BaseException:
                _report_checkpoint_maintenance("Checkpoint lock cleanup failed", _cleanup_save_stage(None, guard))
                held_lock[0] = None
                raise
        lock, stage = _main_value(reserve)
        iterations = {owner: owner.project_configuration.iteration for owner in self.owners.values()}
        try:
            for name, owner in self.owners.items():
                checkpoint_phase(lambda owner=owner, name=name: _save_native(owner, Path(stage) / name,
                                                                             safe_serialization, kwargs))
            def publish():
                if os.path.lexists(path):
                    raise ValueError("Checkpoint destination appeared while saving")
                (Path(stage) / ".videox_native_checkpoint.json").write_text(json.dumps(
                    {"run_id": run_id, "components": list(self.owners)}, sort_keys=True))
                st = os.stat(stage, follow_symlinks=False)
                stage_identity[0] = (st.st_dev, st.st_ino)
                os.rename(stage, path)
                committed[0] = True
            _main_value(publish)
            self._native_run_id = run_id
            limit = self.checkpoint_limit
            if limit is None and self.accelerator.project_configuration.automatic_checkpoint_naming:
                limit = self.accelerator.project_configuration.total_limit
            if limit is not None:
                errors = _main_value(lambda: _prune_checkpoints(
                    path.parent, path, limit, run_id, marker=".videox_native_checkpoint.json",
                    components=list(self.owners)))
                _report_checkpoint_maintenance("Checkpoint saved, but retention cleanup failed", errors)
        except BaseException:
            if _main_value(lambda: _stage_was_committed(path, stage_identity[0], committed[0])):
                self._native_run_id = run_id
            else:
                for owner, iteration in iterations.items():
                    owner.project_configuration.iteration = iteration
            raise
        finally:
            errors = _main_value(lambda: _cleanup_save_stage(stage, held_lock[0]))
            _report_checkpoint_maintenance("Checkpoint staging cleanup failed", errors)
        return path

    def latest(self, output_dir):
        return _coordinated_latest_training_checkpoint(output_dir)

    @staticmethod
    def phase(function): return checkpoint_phase(function)
    @staticmethod
    def fingerprint(path): return checkpoint_source_identity(path)
    @staticmethod
    def load_model(*args, **kwargs): return load_model_checkpoint(*args, **kwargs)
    @staticmethod
    def copy_model(*args, **kwargs): return copy_diffusers_checkpoint(*args, **kwargs)
    @staticmethod
    def complete_model(*args, **kwargs): return is_complete_diffusers_checkpoint(*args, **kwargs)
    @staticmethod
    def save_model(*args, **kwargs): return save_diffusers_state_dict(*args, **kwargs)
    @staticmethod
    def read_model(*args, **kwargs): return read_diffusers_state_dict(*args, **kwargs)
    @staticmethod
    def check_finite(value, name): return check_finite_training_value(value, name)
