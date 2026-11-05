"""Model weight I/O and transactional Accelerate training checkpoints."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import types
import uuid
import warnings
from weakref import WeakKeyDictionary, ref

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

    ``model_ids`` maps role names to prepared models. It is required for multiple
    models of the same class when saving new checkpoints. Legacy checkpoints
    are detected automatically; ``allow_legacy=False`` opts into manifest-only
    restore. Missing required training state is an error in either format.
    ``checksum='full'`` adds streaming hashes of large payloads; the default
    hashes control files and records the size of every payload.
    ``legacy_metadata_keys`` names fields newly added by the caller. Older
    manifests missing that entire set can load with a warning. A partially
    missing set is invalid; existing values must still match. No model,
    optimizer or topology checks are relaxed.
    ``legacy_metadata_groups`` allows separate generations of fields to be
    migrated independently, each subject to the same all-or-none check.
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
        config.metadata = _plain(metadata)
    config.legacy_metadata_keys = tuple(legacy_metadata_keys)
    config.legacy_metadata_groups = tuple(tuple(group) for group in legacy_metadata_groups)
    if any(key not in config.metadata for group in (config.legacy_metadata_keys, *config.legacy_metadata_groups) for key in group):
        raise ValueError("Legacy metadata keys must exist in the current training metadata")
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


def _callable_signature(function):
    if isinstance(function, functools.partial):
        return {"partial": _callable_signature(function.func), "args": _plain(function.args),
                "kwargs": _plain(function.keywords)}
    if isinstance(function, types.FunctionType):
        return {"function": function.__name__, "defaults": _plain(function.__defaults__),
                "keywords": _plain(function.__kwdefaults__),
                "closure": [_plain(cell.cell_contents) for cell in function.__closure__ or ()]}
    # Callable scheduler objects may carry arbitrary mutable behavior. Require
    # an explicit implementation instead of claiming generic compatibility.
    raise ValueError("Unsupported scheduler callable; use a supported function or functools.partial")


def _normalize_scheduler_callable(signature):
    """Read older signatures without tying compatibility to module or bytecode."""
    signature = dict(signature)
    if "partial" in signature:
        signature["partial"] = _normalize_scheduler_callable(signature["partial"])
    if "function" in signature:
        signature["function"] = signature["function"].rsplit(".", 1)[-1]
        signature.pop("code", None)
    return signature


def _hook_counts(owners):
    # The synchronized wrapper invokes one collective per registered hook.
    # Agree on their counts across ranks, not their Python implementation.
    return {name: [len(owner._save_model_state_pre_hook), len(owner._load_model_state_pre_hook)]
            for name, owner in owners.items()}


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
            if classes.count(classes[index]) > 1 and require_model_ids:
                raise ValueError("Multiple models of the same class require configure_checkpointing(model_ids={role: model})")
            role = (f"legacy_model_{index}" if classes.count(classes[index]) > 1
                    else "model" if len(classes) == 1 else classes[index])
        parameters = {name: {"shape": list(shapes[name]), "dtype": str(parameter.dtype),
                             "requires_grad": parameter.requires_grad}
                      for parameter, fqns in names.items() for name in fqns}
        unwrapped = owner.unwrap_model(model, keep_torch_compile=False)
        model_config = getattr(unwrapped, "config", {})
        model_config = {key: value for key, value in dict(model_config).items() if not key.startswith("_")} if hasattr(model_config, "items") else {}
        models.append({"id": role, "class": classes[index], "parameters": parameters,
                       "config": _plain(model_config), "fsdp": units})
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
            row["functions"] = [_callable_signature(function) for function in scheduler.lr_lambdas]
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


def latest_training_checkpoint(output_dir, owners=None):
    """Select a complete checkpoint; pass owners to also inspect legacy formats.

    Legacy files cannot describe required models, components or private state,
    so their requirements come from the prepared Accelerator(s). Inspect every
    rank's files so all callers select the same checkpoint. This function uses
    no collectives; distributed callers synchronize errors at the call site.
    """
    participants = _participants(owners) if owners is not None else None
    root = Path(output_dir)
    if not root.is_dir():
        return None
    candidates = []
    for path in root.iterdir():
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))
    candidates.sort()
    for _, path in reversed(candidates):
        try:
            manifest_path = path / MANIFEST
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                _verify_inventory(path, manifest, full=False)
                if participants is not None and set(manifest["components"]) != set(participants):
                    raise ValueError("Checkpoint participants differ")
            elif participants is None or not all(checkpoint_config(owner).allow_legacy for owner in participants.values()):
                continue
            if participants is not None:
                for name, owner in participants.items():
                    _required_files(owner, path / name, all_ranks=True)
            return str(path)
        except Exception:
            # A broken candidate must not prevent trying the preceding one.
            # Direct load still reports the actual validation error.
            continue
    if candidates:
        raise ValueError("No complete checkpoint matching the required training state")
    return None


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


def _required_files(owner, directory, *, all_ranks=False):
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
def _synchronized_hooks(owner):
    """Make local hook failures visible before the next native phase."""
    changes = []
    for hooks in (owner._save_model_state_pre_hook, owner._load_model_state_pre_hook):
        for key, function in list(hooks.items()):
            def synchronized(*args, function=function):
                return checkpoint_phase(lambda: function(*args))
            changes.append((hooks, key, function))
            hooks[key] = synchronized
    try:
        yield
    finally:
        for hooks, key, function in changes:
            hooks[key] = function


@contextmanager
def _optimizer_adapter(owner):
    from .. import (enable_accelerate_patch, _accelerate_patch_owners,
                    _accelerate_patch_disabled_owners)

    explicit = owner in _accelerate_patch_owners
    disabled = owner in _accelerate_patch_disabled_owners
    if owner.distributed_type.name == "FSDP":
        enable_accelerate_patch(owner)
    try:
        yield
    finally:
        if not explicit:
            _accelerate_patch_owners.discard(owner)
        if disabled:
            _accelerate_patch_disabled_owners.add(owner)


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
        if not owner.sync_gradients or any(parameter.grad is not None for model in owner._models for parameter in model.parameters()):
            raise ValueError("Save checkpoints after an optimizer update and zero_grad(set_to_none=True)")
        if checkpoint_config(owner).check_finite:
            check_finite_state([list(model.parameters()) for model in owner._models])
            check_finite_state([_raw_optimizer(optimizer).state for optimizer in owner._optimizers])
    return descriptions


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
    _agree([str(path), descriptions, policies, checkpoint_limit, _hook_counts(owners)])
    if checkpoint_limit is not None and checkpoint_limit < 1:
        raise ValueError("checkpoint_limit must be at least 1")
    run_ids = {checkpoint_config(owner).run_id for owner in owners.values()} - {None}
    _agree(sorted(run_ids))
    if len(run_ids) > 1:
        raise ValueError("Checkpoint participants belong to different training runs")
    run_id = _main_value(lambda: next(iter(run_ids), None) or uuid.uuid4().hex)
    lock, stage = None, None
    iterations = {owner: owner.project_configuration.iteration for owner in owners.values()}

    def reserve():
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.parent / ".training-checkpoints.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            if path.exists():
                raise ValueError(f"Refusing to overwrite an existing checkpoint: {path}")
            for sibling in path.parent.iterdir():
                if _CHECKPOINT_NAME.fullmatch(sibling.name) and (sibling / MANIFEST).is_file():
                    other = json.loads((sibling / MANIFEST).read_text())
                    if other.get("run_id") != run_id:
                        raise ValueError("Output directory contains checkpoints from another run; resume that run or use a new output directory")
            stage_path = tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent)
            return str(lock_path), stage_path
        except BaseException:
            lock_path.unlink()
            raise

    lock, stage = _main_value(reserve)
    try:
        for name, owner in owners.items():
            checkpoint_phase(lambda owner=owner, name=name: _save_native(owner, Path(stage) / name, safe_serialization, kwargs))
            checkpoint_phase(lambda owner=owner, name=name: _required_files(owner, Path(stage) / name))

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
            if path.exists():
                raise ValueError("Checkpoint destination appeared while saving")
            os.rename(directory, path)
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        _main_value(publish)
        for owner in owners.values():
            checkpoint_config(owner).run_id = run_id
        if checkpoint_limit is not None:
            def prune():
                checkpoints = []
                for candidate in path.parent.iterdir():
                    match = _CHECKPOINT_NAME.fullmatch(candidate.name)
                    if match and (candidate / MANIFEST).is_file():
                        checkpoints.append((int(match.group(1)), candidate))
                checkpoints.sort()
                for _, candidate in checkpoints[:-checkpoint_limit]:
                    manifest = json.loads((candidate / MANIFEST).read_text())
                    if candidate != path and manifest.get("run_id") == run_id:
                        shutil.rmtree(candidate)
            _main_value(prune)
    except BaseException:
        # Native save increments this counter even before a later component fails.
        if not path.exists():
            for owner, iteration in iterations.items():
                owner.project_configuration.iteration = iteration
        raise
    finally:
        def cleanup():
            if stage and Path(stage).exists():
                shutil.rmtree(stage)
            if lock:
                try:
                    Path(lock).unlink()
                except FileNotFoundError:
                    pass
        _main_value(cleanup)
    return path


def _compare_descriptor(saved, current, *, legacy_metadata_keys=(), legacy_metadata_groups=()):
    # Preserve the previously supported no-scheduler group reordering. Group
    # membership changes remain invalid; scheduler-bound group order is exact.
    saved, current = json.loads(json.dumps(saved)), json.loads(json.dumps(current))
    for descriptor in (saved, current):
        descriptor.pop("hooks", None)  # Legacy manifests recorded Python hook identities.
        for state in descriptor.get("private_states", {}).values():
            # The existing RNG/progress schema predates explicit versioning.
            state.setdefault("schema_version", 1)
        for scheduler in descriptor["schedulers"]:
            scheduler["class"] = scheduler["class"].rsplit(".", 1)[-1]
            if "functions" in scheduler:
                scheduler["functions"] = [_normalize_scheduler_callable(fn) for fn in scheduler["functions"]]
    missing = []
    for group in (legacy_metadata_keys, *legacy_metadata_groups):
        absent = sorted(set(group) - saved["training"].keys())
        if absent and set(group) & saved["training"].keys():
            raise ValueError("Checkpoint training metadata is incomplete; refusing partial legacy metadata migration")
        missing.extend(absent)
    for key in missing:
        saved["training"][key] = current["training"][key]
    if saved["backend"] == current["backend"] == "FSDP" and not saved["schedulers"] and not current["schedulers"]:
        for descriptor in (saved, current):
            for optimizer in descriptor["optimizers"]:
                pairs = sorted(zip(optimizer["groups"], optimizer["execution"]), key=lambda pair: sorted(pair[0]))
                optimizer["groups"] = [sorted(group) for group, _ in pairs]
                optimizer["execution"] = [execution for _, execution in pairs]
    if saved != current:
        raise ValueError("Checkpoint training configuration differs: verify model roles, optimizer type/groups, scheduler, precision and topology")
    if missing and (not dist.is_initialized() or dist.get_rank() == 0):
        warnings.warn("Older checkpoint lacks training metadata for " + ", ".join(missing) +
                      "; using the current configuration for these fields. Historical values cannot be verified. "
                      "The next save will record them.", RuntimeWarning, stacklevel=2)


def load_training_checkpoint(owners, input_dir=None, **kwargs):
    """Validate every participant before mutating any model; stop training on failure."""
    owners = checkpoint_phase(lambda: _participants(owners))
    primary = next(iter(owners.values()))
    if input_dir is None and primary.project_configuration.automatic_checkpoint_naming:
        input_dir = _main_value(lambda: latest_training_checkpoint(Path(primary.project_dir) / "checkpoints", owners))
    path = checkpoint_phase(lambda: Path(input_dir).expanduser().absolute())
    def read_manifest():
        manifest_path = path / MANIFEST
        if not manifest_path.exists():
            return None
        value = json.loads(manifest_path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Invalid checkpoint manifest")
        return value
    manifest = checkpoint_phase(read_manifest)
    descriptions = checkpoint_phase(lambda: {
        name: _descriptor(owner, require_model_ids=manifest is not None) for name, owner in owners.items()
    })
    _agree([str(path), descriptions, _hook_counts(owners), {name: {"legacy": checkpoint_config(owner).allow_legacy,
                                           "finite": checkpoint_config(owner).check_finite}
                                    for name, owner in owners.items()}])

    def preflight():
        if any(parameter.grad is not None for owner in owners.values()
               for model in owner._models for parameter in model.parameters()):
            raise ValueError("Clear accumulated gradients with zero_grad(set_to_none=True) before restoring")
        if manifest is None:
            if not all(checkpoint_config(owner).allow_legacy for owner in owners.values()):
                raise ValueError("Checkpoint has no completion manifest; allow_legacy=False requires a new-format checkpoint")
        else:
            if set(manifest["components"]) != set(owners):
                raise ValueError("Checkpoint participants differ; restore the complete named group")
            if manifest.get("versions") != {"accelerate": "1.7.0", "torch": "2.7.0"}:
                raise ValueError("Checkpoint library versions differ")
            _verify_inventory(path, manifest)
            for name, current in descriptions.items():
                config = checkpoint_config(owners[name])
                _compare_descriptor(manifest["components"][name], current,
                                    legacy_metadata_keys=config.legacy_metadata_keys if config.allow_legacy else (),
                                    legacy_metadata_groups=config.legacy_metadata_groups if config.allow_legacy else ())
        rng_states = {name: _required_files(owner, path / name) for name, owner in owners.items()}
        return manifest, rng_states

    manifest, rng_states = checkpoint_phase(preflight)
    _agree(hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
    if manifest is None:
        def warn_legacy():
            if primary.is_main_process:
                warnings.warn("Loading a legacy checkpoint using native component order; required files were checked, "
                              "but historical component identity and atomic completion cannot be verified. "
                              "The next save will include a completion manifest.", RuntimeWarning, stacklevel=2)
        checkpoint_phase(warn_legacy)
    try:
        for name, owner in owners.items():
            def restore(owner=owner, name=name):
                with _optimizer_adapter(owner), _synchronized_hooks(owner):
                    _native(owner, "load_state", str(path / name), **kwargs)
            checkpoint_phase(restore)
            owner.step = rng_states[name]["step"]
            if checkpoint_config(owner).check_finite:
                checkpoint_phase(lambda: check_finite_state([list(model.parameters()) for model in owner._models]))
                if owner.distributed_type.name != "FSDP":
                    checkpoint_phase(lambda: check_finite_state([_raw_optimizer(optimizer).state for optimizer in owner._optimizers]))
        # Native Accelerate swallows RNG errors. Apply the prevalidated state
        # explicitly after all hooks/components, and propagate any failure.
        checkpoint_phase(lambda: _apply_rng(next(reversed(rng_states.values()))))
    except BaseException:
        _set_failed(owners.values(), True)
        raise
    _set_failed(owners.values(), False)
    run_id = manifest["run_id"] if manifest else _main_value(lambda: uuid.uuid4().hex)
    for owner in owners.values():
        checkpoint_config(owner).run_id = run_id
        if owner.project_configuration.automatic_checkpoint_naming:
            match = _CHECKPOINT_NAME.fullmatch(path.name)
            if match:
                owner.project_configuration.iteration = max(owner.save_iteration, int(match.group(1)) + 1)


def dispatch_checkpoint(owner, method, original, *args, **kwargs):
    """Agree on FSDP dispatch before choosing native or protected checkpoints."""
    from .. import (_accelerate_patch_requested, _accelerate_patch_owners,
                    _accelerate_patch_disabled_owners, _supported_accelerate_versions,
                    _eligible_fsdp_checkpoint)

    enabled = ((owner in _accelerate_patch_owners or _accelerate_patch_requested())
               and owner not in _accelerate_patch_disabled_owners)
    plugin = owner.state.fsdp_plugin
    _agree([enabled, _supported_accelerate_versions(), owner.distributed_type.name,
            getattr(plugin, "fsdp_version", 1), getattr(plugin.state_dict_type, "name", None)])
    if not enabled or not _eligible_fsdp_checkpoint(plugin):
        return original(owner, *args, **kwargs)
    available = _protected_checkpoint_available({"": owner})
    _agree(available)
    if not available:
        if method == "load_state" and args and args[0] is not None and (Path(args[0]) / MANIFEST).is_file():
            raise ValueError("Protected checkpoint requires a compatible Accelerate/PyTorch backend and optimizer")
        if not dist.is_initialized() or dist.get_rank() == 0:
            warnings.warn("This Accelerator configuration is not supported by protected checkpoints; "
                          "using native Accelerate state", RuntimeWarning, stacklevel=2)
        return original(owner, *args, **kwargs)
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


def _load_generic_fsdp_optimizer(accelerator, optimizer, model, input_dir, optimizer_index):
    """Restore an unknown optimizer from checkpoint metadata, including fresh state."""
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    plugin = accelerator.state.fsdp_plugin
    path = Path(input_dir) / f"optimizer_{optimizer_index}"
    with FSDP.state_dict_type(model, plugin.state_dict_type,
                              plugin.state_dict_config, plugin.optim_state_dict_config):
        state = load_sharded_optimizer_state_dict(
            model.state_dict(), "optimizer", FileSystemReader(path),
        )["optimizer"]
        local = FSDP.optim_state_dict_to_load(model, optimizer, state)
        optimizer.load_state_dict(local)


def _load_sharded_fsdp_optimizer(accelerator, optimizer, model, input_dir, optimizer_index):
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from .fsdp_utils import release_completed_fsdp_gradients

    optimizer = getattr(optimizer, "optimizer", optimizer)
    process_group = model.process_group
    _checkpoint_phase(lambda: validate_checkpoint_owner(accelerator), process_group)
    checkpoint_dir = Path(input_dir) / f"optimizer_{optimizer_index}"

    def prepare():
        if len(accelerator._optimizers) != len(accelerator._models):
            raise ValueError("FSDP checkpoint restore requires one optimizer per prepared model")
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

    expected_groups, shapes, metadata = _checkpoint_phase(prepare, process_group)
    state = _checkpoint_phase(
        lambda: _optimizer_load_destination(metadata, shapes, process_group), process_group,
    )
    # These are already one-dimensional shards. The nested-shard flattener
    # drops empty local shards and can mis-detect the checkpoint schema.
    load(state, checkpoint_id=checkpoint_dir, process_group=process_group,
         planner=DefaultLoadPlanner(flatten_sharded_tensors=False))

    _checkpoint_phase(
        lambda: _align_optimizer_groups(state["optimizer"], expected_groups, optimizer,
                                        allow_reorder=not accelerator._schedulers), process_group,
    )
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
        state = read_state(input_dir)
        if state is None:
            return
        for name, generator in generators.items():
            generator.set_state(state["generators"][name])
        progress.update(state["progress"], restored=True)

    return (accelerator.register_save_state_pre_hook(save_state),
            accelerator.register_load_state_pre_hook(load_state))


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
    return True


def _required_native_files(owner, directory, *, all_ranks=False):
    """Validate registered rank state without assuming an FSDP/DeepSpeed layout."""
    if not directory.is_dir():
        raise ValueError(f"Missing checkpoint component: {directory}")
    ranks = range(owner.num_processes) if all_ranks else (owner.process_index,)
    for rank in ranks:
        for validator in checkpoint_config(owner).validators.values():
            validator(directory, rank=rank)
        _rng_state(directory / f"random_states_{rank}.pkl")


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

    def _use_protected(self):
        protected = self._protected()
        _agree(protected)
        if not protected and not self._warned_fallback:
            warnings.warn("Checkpoint protection is unavailable or disabled; using native Accelerate state "
                          "without manifest configuration checks", RuntimeWarning, stacklevel=2)
            self._warned_fallback = True
        return protected

    def save_state(self, output_dir=None, *, safe_serialization=True, **kwargs):
        automatic = self.accelerator.project_configuration.automatic_checkpoint_naming
        projects = {id(owner.project_configuration): (owner.project_configuration, owner.project_configuration.iteration)
                    for owner in self.owners.values()}
        if automatic:
            output_dir = None
        if self._use_protected():
            result = save_training_checkpoint(self.owners, output_dir, checkpoint_limit=self.checkpoint_limit,
                                              safe_serialization=safe_serialization, **kwargs)
        elif self.accelerator.project_configuration.save_on_each_node and len(self.owners) == 1:
            result = _native(self.accelerator, "save_state", output_dir, safe_serialization=safe_serialization, **kwargs)
        else:
            result = self._save_native_group(output_dir, safe_serialization, kwargs)
        if automatic:
            for project, iteration in projects.values():
                project.iteration = iteration + 1
        return result

    def load_state(self, input_dir=None, **kwargs):
        if self._use_protected():
            return load_training_checkpoint(self.owners, input_dir, **kwargs)
        if self.accelerator.project_configuration.save_on_each_node and len(self.owners) == 1:
            return _native(self.accelerator, "load_state", input_dir, **kwargs)
        if input_dir is None and self.accelerator.project_configuration.automatic_checkpoint_naming:
            input_dir = self.latest(Path(self.accelerator.project_dir) / "checkpoints")
        if input_dir is None:
            raise ValueError("A checkpoint directory is required")
        path = checkpoint_phase(lambda: Path(input_dir).expanduser().absolute())
        from .. import _accelerate_patch_requested, _accelerate_patch_disabled_owners
        opted_out = not _accelerate_patch_requested() or any(
            owner in _accelerate_patch_disabled_owners for owner in self.owners.values())
        if not opted_out and (path / MANIFEST).is_file():
            raise ValueError("Protected checkpoint requires a compatible Accelerate/PyTorch backend and optimizer")
        _agree([str(path), _hook_counts(self.owners)])
        checkpoint_phase(lambda: [_required_native_files(owner, path / name) for name, owner in self.owners.items()])
        def identity():
            marker = path / ".videox_native_checkpoint.json"
            if not marker.exists():
                return None
            value = json.loads(marker.read_text())
            if value.get("components") != list(self.owners) or not isinstance(value.get("run_id"), str):
                raise ValueError("Native checkpoint group metadata differs")
            return value["run_id"]
        self._native_run_id = _main_value(identity)
        try:
            for name, owner in self.owners.items():
                def restore(owner=owner, name=name):
                    with _synchronized_hooks(owner):
                        return _native(owner, "load_state", str(path / name), **kwargs)
                checkpoint_phase(restore)
        except BaseException:
            _set_failed(self.owners.values(), True)
            raise
        _set_failed(self.owners.values(), False)
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match:
            for owner in self.owners.values():
                if owner.project_configuration.automatic_checkpoint_naming:
                    owner.project_configuration.iteration = max(owner.save_iteration, int(match.group(1)) + 1)

    def _save_native_group(self, output_dir, safe_serialization, kwargs):
        from .. import _failed_checkpoint_owners
        if any(owner in _failed_checkpoint_owners for owner in self.owners.values()):
            raise RuntimeError("Cannot save after a failed restore; reload successfully first")
        if output_dir is None:
            if not self.accelerator.project_configuration.automatic_checkpoint_naming:
                raise ValueError("output_dir is required for a checkpoint group")
            output_dir = Path(self.accelerator.project_dir) / "checkpoints" / f"checkpoint_{self.accelerator.save_iteration}"
        path = checkpoint_phase(lambda: Path(output_dir).expanduser().absolute())
        _agree([str(path), _hook_counts(self.owners)])
        run_id = _main_value(lambda: self._native_run_id or uuid.uuid4().hex)
        def reserve():
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = path.parent / ".training-checkpoints.lock"
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            try:
                if path.exists():
                    raise ValueError(f"Refusing to overwrite an existing checkpoint: {path}")
                return str(lock), tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent)
            except BaseException:
                lock.unlink()
                raise
        lock, stage = _main_value(reserve)
        iterations = {owner: owner.project_configuration.iteration for owner in self.owners.values()}
        try:
            for name, owner in self.owners.items():
                checkpoint_phase(lambda owner=owner, name=name: _save_native(owner, Path(stage) / name,
                                                                             safe_serialization, kwargs))
            def publish():
                if path.exists():
                    raise ValueError("Checkpoint destination appeared while saving")
                (Path(stage) / ".videox_native_checkpoint.json").write_text(json.dumps(
                    {"run_id": run_id, "components": list(self.owners)}, sort_keys=True))
                os.rename(stage, path)
            _main_value(publish)
            self._native_run_id = run_id
        except BaseException:
            for owner, iteration in iterations.items():
                owner.project_configuration.iteration = iteration
            raise
        finally:
            def cleanup():
                if Path(stage).exists():
                    shutil.rmtree(stage)
                try:
                    Path(lock).unlink()
                except FileNotFoundError:
                    pass
            _main_value(cleanup)
        limit = self.checkpoint_limit
        if limit is None and self.accelerator.project_configuration.automatic_checkpoint_naming:
            limit = self.accelerator.project_configuration.total_limit
        if limit is not None:
            def prune():
                candidates = []
                for candidate in path.parent.iterdir():
                    match = _CHECKPOINT_NAME.fullmatch(candidate.name)
                    marker = candidate / ".videox_native_checkpoint.json"
                    if not match or not candidate.is_dir() or candidate.is_symlink() or not marker.is_file():
                        continue
                    try:
                        value = json.loads(marker.read_text())
                    except (OSError, ValueError):
                        continue
                    if value.get("run_id") == run_id and value.get("components") == list(self.owners):
                        candidates.append((int(match.group(1)), candidate))
                for _, candidate in sorted(candidates)[:-limit]:
                    if candidate != path:
                        shutil.rmtree(candidate)
            _main_value(prune)
        return str(path)

    def latest(self, output_dir):
        if self._use_protected():
            result = checkpoint_phase(lambda: latest_training_checkpoint(output_dir, self.owners))
            _agree(result)
            return result
        def find_native():
            root = Path(output_dir)
            if not root.is_dir():
                return None
            candidates = sorted(((int(match.group(1)), path) for path in root.iterdir()
                                 if (match := _CHECKPOINT_NAME.fullmatch(path.name)) and path.is_dir()), reverse=True)
            for _, path in candidates:
                try:
                    manifest = path / MANIFEST
                    if manifest.exists():
                        value = json.loads(manifest.read_text())
                        _verify_inventory(path, value, full=False)
                        if set(value["components"]) != set(self.owners):
                            continue
                    marker = path / ".videox_native_checkpoint.json"
                    if marker.exists() and json.loads(marker.read_text()).get("components") != list(self.owners):
                        continue
                    for name, owner in self.owners.items():
                        _required_native_files(owner, path / name,
                                               all_ranks=not owner.project_configuration.save_on_each_node)
                    return str(path)
                except (OSError, ValueError, KeyError, RuntimeError):
                    continue
            if candidates:
                raise ValueError("No complete native checkpoint matching the required training state")
            return None
        result = checkpoint_phase(find_native)
        _agree(result)
        return result

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
