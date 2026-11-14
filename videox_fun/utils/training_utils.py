"""Training optimizer, precision, and checkpoint utilities."""
import errno
import functools
import gc
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from weakref import WeakKeyDictionary, ref

try:
    import fcntl
except ImportError:  # Lock recovery is unavailable on platforms without flock.
    fcntl = None

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
    Legacy integrity settings remain accepted for API compatibility; native
    save and load use Accelerate's format without a separate manifest.
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







def _save_hook_counts(owners):
    # Only save hooks run during saving, and each wrapper invokes a collective.
    return {name: len(owner._save_model_state_pre_hook) for name, owner in owners.items()}


def _raw_optimizer(optimizer):
    return getattr(optimizer, "optimizer", optimizer)


def validate_checkpoint_owner(owner):
    """Check only the public capabilities needed by native checkpointing."""
    if not callable(getattr(owner, "save_state", None)) or not callable(getattr(owner, "load_state", None)):
        raise TypeError("Checkpoint owner must provide save_state and load_state")




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
        if hasattr(tensor, "to_local") and callable(tensor.to_local):
            tensor = tensor.to_local()
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
    def __init__(self, directory, name=".training-checkpoints.lock"):
        self.directory = Path(directory)
        self.path = self.directory / name
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


def _check_checkpoint_finite(owner):
    if checkpoint_config(owner).check_finite:
        check_finite_state([list(model.parameters()) for model in owner._models])
        check_finite_state([_raw_optimizer(optimizer).state for optimizer in owner._optimizers])


def _verify_native_dcp_payloads(directory):
    """Check recognizable DCP shards without imposing a load-time format."""
    try:
        from torch.distributed.checkpoint import FileSystemReader
    except ImportError:
        return
    for child in directory.iterdir():
        if not child.is_dir() or not re.fullmatch(r"(?:pytorch_model_fsdp|optimizer)_\d+", child.name):
            continue
        metadata_path = child / ".metadata"
        if not metadata_path.is_file():
            raise ValueError(f"Missing native DCP metadata: {metadata_path}")
        metadata = FileSystemReader(child).read_metadata()
        storage_data = getattr(metadata, "storage_data", None)
        if not isinstance(storage_data, dict):
            continue
        sizes = {}
        for item in storage_data.values():
            if not all(hasattr(item, name) for name in ("relative_path", "offset", "length")):
                break  # Unknown runtime metadata layout: leave validation to Accelerate.
            payload = child / _relative(item.relative_path)
            if payload not in sizes:
                try:
                    sizes[payload] = payload.stat().st_size
                except OSError as exc:
                    raise ValueError(f"Missing native DCP payload: {payload}") from exc
            if sizes[payload] < item.offset + item.length:
                raise ValueError(f"Missing or truncated native DCP payload: {payload}")


def _sync_checkpoint_parent(path):
    """Request directory durability when the backing filesystem supports it."""
    fd = None
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        os.fsync(fd)
    except OSError as exc:
        from .. import _checkpoint_warning
        _checkpoint_warning(f"Could not synchronize checkpoint directory metadata: {exc}")
    finally:
        if fd is not None:
            os.close(fd)


def _checkpoint_target_lock(path):
    name = hashlib.sha256(os.fsencode(path.name)).hexdigest()[:24]
    return _CheckpointLock(path.parent, f".videox-checkpoint-{name}.lock")


def _transaction_paths(path):
    return (path.parent / f".{path.name}.videox-backup",
            path.parent / f".{path.name}.videox-transaction.json")


def _recover_native_transaction(path):
    """Recover an interrupted overwrite while the parent checkpoint lock is held."""
    backup, journal = _transaction_paths(path)
    if not journal.exists():
        if os.path.lexists(backup):
            raise FileExistsError(f"Checkpoint backup exists without a transaction journal: {backup}")
        return
    entry = json.loads(journal.read_text())
    if entry.get("version") != 1 or entry.get("target") != path.name or entry.get("backup") != backup.name:
        raise ValueError(f"Invalid checkpoint transaction journal: {journal}")
    if os.path.lexists(backup):
        if not os.path.lexists(path):
            os.rename(backup, path)
        else:
            info = os.stat(path, follow_symlinks=False)
            if [info.st_dev, info.st_ino] != entry.get("new_identity"):
                raise RuntimeError(f"Cannot safely reconcile checkpoint and backup: {path}")
            shutil.rmtree(backup)
    elif not os.path.lexists(path):
        raise RuntimeError(f"Checkpoint and transaction backup are both missing: {path}")
    journal.unlink()


def recover_native_checkpoint(owner, input_dir):
    """Restore an interrupted overwrite before native Accelerate loading."""
    if input_dir is None:
        return
    path = Path(input_dir).expanduser().absolute()
    backup, journal = _transaction_paths(path)
    # All ranks must agree before any of them enters a collective.
    needs_recovery = _main_value(lambda: journal.exists() or os.path.lexists(backup))
    if not needs_recovery:
        return
    def recover():
        guard = _checkpoint_target_lock(path)
        guard.acquire()
        try:
            _recover_native_transaction(path)
        finally:
            guard.release()
    _main_value(recover)


def recover_checkpoint_directories(accelerator, output_dir):
    """Recover interrupted overwrites before selecting the latest checkpoint.

    This only visits VideoX transaction journals; native and older checkpoints
    are otherwise left to Accelerate's normal loading behavior. All ranks in
    a distributed training job must call this method.
    """
    from .. import (_accelerate_patch_disabled_owners,
                    _accelerate_patch_requested)

    if (not _accelerate_patch_requested()
            or accelerator in _accelerate_patch_disabled_owners
            or getattr(accelerator.project_configuration, "save_on_each_node", False)):
        return
    root = Path(output_dir).expanduser().absolute()

    def recover():
        if not root.is_dir():
            return
        suffix = ".videox-transaction.json"
        for journal in root.glob(".checkpoint*" + suffix):
            name = journal.name[1:-len(suffix)]
            if not _CHECKPOINT_NAME.fullmatch(name):
                continue
            path = root / name
            guard = _checkpoint_target_lock(path)
            guard.acquire()
            try:
                _recover_native_transaction(path)
            finally:
                guard.release()

    _main_value(recover)


def _publish_native_stage(path, directory, published):
    backup, journal = _transaction_paths(path)
    if os.path.lexists(path):
        info = os.stat(directory, follow_symlinks=False)
        entry = {"version": 1, "target": path.name, "backup": backup.name,
                 "new_identity": [info.st_dev, info.st_ino]}
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.transaction-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(entry, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, journal)
            _sync_checkpoint_parent(path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        os.rename(path, backup)
        _sync_checkpoint_parent(path)
        try:
            os.rename(directory, path)
        except BaseException:
            _recover_native_transaction(path)
            raise
        published[0] = True
        _sync_checkpoint_parent(path)
        try:
            _recover_native_transaction(path)
            _sync_checkpoint_parent(path)
        except OSError as exc:
            from .. import _checkpoint_warning
            _checkpoint_warning(f"Checkpoint published but backup cleanup failed: {exc}")
    else:
        os.rename(directory, path)
        published[0] = True
        _sync_checkpoint_parent(path)


def save_native_transaction(owner, original, output_dir, *, safe_serialization=True, **kwargs):
    """Stage native Accelerate output and publish only after all save calls succeed."""
    from .. import _failed_checkpoint_owners

    project = owner.project_configuration
    automatic = project.automatic_checkpoint_naming
    if automatic:
        output_dir = Path(owner.project_dir) / "checkpoints" / f"checkpoint_{owner.save_iteration}"
    path = checkpoint_phase(lambda: Path(output_dir).expanduser().absolute())
    _agree([str(path), automatic, project.total_limit, len(getattr(owner, "_save_model_state_pre_hook", {}))])
    def require_restored():
        if owner in _failed_checkpoint_owners:
            raise RuntimeError("Cannot save after a failed restore; reload successfully first")
    checkpoint_phase(require_restored)
    checkpoint_phase(lambda: _check_checkpoint_finite(owner))
    held, stage = [None], [None]
    previous_iteration = owner.project_configuration.iteration

    def reserve():
        path.parent.mkdir(parents=True, exist_ok=True)
        guard = _checkpoint_target_lock(path)
        guard.acquire()
        held[0] = guard
        try:
            _recover_native_transaction(path)
            if automatic and os.path.lexists(path):
                raise ValueError(f"Automatic checkpoint destination already exists: {path}")
            if os.path.lexists(path) and (path.is_symlink() or not path.is_dir()):
                raise ValueError(f"Checkpoint destination must be a real directory: {path}")
            stage[0] = tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent)
            return stage[0]
        except BaseException:
            guard.release()
            held[0] = None
            raise

    directory = _main_value(reserve)
    published = [False]
    try:
        def save():
            project.automatic_checkpoint_naming = False
            try:
                with _synchronized_hooks(owner):
                    return original(owner, directory, safe_serialization=safe_serialization, **kwargs)
            finally:
                project.automatic_checkpoint_naming = automatic
        checkpoint_phase(save)
        _main_value(lambda: _verify_native_dcp_payloads(Path(directory)))

        _main_value(lambda: _publish_native_stage(path, directory, published))
        if automatic and project.total_limit is not None:
            def prune():
                candidates = []
                for item in path.parent.iterdir():
                    match = re.fullmatch(r"checkpoint_(\d+)", item.name)
                    if match and item.is_dir() and not item.is_symlink():
                        candidates.append((int(match.group(1)), item))
                excess = max(0, len(candidates) - max(1, project.total_limit))
                for _, item in sorted(candidates):
                    if not excess:
                        break
                    if item != path:
                        try:
                            shutil.rmtree(item)
                            excess -= 1
                        except OSError as exc:
                            from .. import _checkpoint_warning
                            _checkpoint_warning(f"Checkpoint saved but old checkpoint cleanup failed: {exc}")
            _main_value(prune)
    except BaseException:
        if not _main_value(lambda: published[0]):
            owner.project_configuration.iteration = previous_iteration
        raise
    finally:
        errors = _main_value(lambda: _cleanup_save_stage(directory, held[0]))
        _report_checkpoint_maintenance("Checkpoint staging cleanup failed", errors)
    return str(path)


_native_checkpoint_groups = {}


def _native_group_for(path):
    if path is None:
        return None
    path = Path(path).expanduser().absolute()
    for root, group in _native_checkpoint_groups.items():
        if path == root or root in path.parents:
            return group
    return None


def checkpoint_staging_path(accelerator, path):
    """Resolve a path inside an active multi-component checkpoint."""
    group = _native_group_for(path)
    if group is None:
        return os.fspath(path)
    relative = Path(path).expanduser().absolute().relative_to(group["path"])
    return str(Path(group["stage"]) / relative)


def _abort_native_group(group):
    for owner, iteration in group["iterations"].items():
        owner.project_configuration.iteration = iteration
        owner.__dict__.pop("videox_checkpoint_grouped", None)
    errors = _main_value(lambda: _cleanup_save_stage(group["stage"], group["lock"][0]))
    _report_checkpoint_maintenance("Checkpoint staging cleanup failed", errors)
    _native_checkpoint_groups.pop(group["path"], None)


def _begin_native_group(path):
    _agree([str(path), True])
    held = [None]
    def reserve():
        path.parent.mkdir(parents=True, exist_ok=True)
        guard = _checkpoint_target_lock(path)
        guard.acquire()
        held[0] = guard
        try:
            _recover_native_transaction(path)
            if os.path.lexists(path) and (path.is_symlink() or not path.is_dir()):
                raise ValueError(f"Checkpoint destination must be a real directory: {path}")
            return tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent)
        except BaseException:
            guard.release()
            held[0] = None
            raise
    stage = _main_value(reserve)
    group = {"path": path, "stage": stage, "lock": held, "iterations": {}, "published": [False]}
    _native_checkpoint_groups[path] = group
    return group


def _register_group_owner(group, owner):
    if owner in group["iterations"]:
        return
    project = owner.project_configuration
    # Different Accelerator instances can share the same ProjectConfiguration.
    # Keep its value from before the first component, not from after a save.
    initial = next((iteration for participant, iteration in group["iterations"].items()
                    if participant.project_configuration is project), project.iteration)
    group["iterations"][owner] = initial


def begin_checkpoint_group(accelerator, output_dir):
    """Start a transaction for custom checkpoint files on all participating ranks."""
    from .. import (_accelerate_patch_disabled_owners,
                    _accelerate_patch_requested)

    if (not _accelerate_patch_requested()
            or accelerator in _accelerate_patch_disabled_owners
            or getattr(accelerator.project_configuration, "save_on_each_node", False)):
        return os.fspath(output_dir)
    path = Path(output_dir).expanduser().absolute()
    group = _native_group_for(path)
    if group is None:
        try:
            group = _begin_native_group(path)
        except BaseException:
            accelerator.__dict__.pop("videox_checkpoint_grouped", None)
            raise
    elif group["path"] != path:
        raise ValueError("Checkpoint group must be started at its root directory")
    _register_group_owner(group, accelerator)
    return checkpoint_staging_path(accelerator, path)


def abort_checkpoint_group(accelerator, output_dir):
    """Discard an unfinished group after a synchronized failure."""
    group = _native_group_for(output_dir)
    if group is not None:
        _abort_native_group(group)


def save_native_group_component(owner, original, output_dir, *, safe_serialization=True, **kwargs):
    """Stage one native save; publish the group only when its caller finishes."""
    path = Path(output_dir).expanduser().absolute()
    group = _native_group_for(path)
    if group is None:
        try:
            group = _begin_native_group(path)
        except BaseException:
            owner.__dict__.pop("videox_checkpoint_grouped", None)
            raise
    _register_group_owner(group, owner)
    destination = str(Path(group["stage"]) / path.relative_to(group["path"]))
    try:
        _agree([str(path), len(getattr(owner, "_save_model_state_pre_hook", {}))])
        from .. import _failed_checkpoint_owners
        def require_restored():
            if owner in _failed_checkpoint_owners:
                raise RuntimeError("Cannot save after a failed restore; reload successfully first")
        checkpoint_phase(require_restored)
        checkpoint_phase(lambda: _check_checkpoint_finite(owner))
        def save():
            automatic = owner.project_configuration.automatic_checkpoint_naming
            owner.project_configuration.automatic_checkpoint_naming = False
            try:
                with _synchronized_hooks(owner):
                    return original(owner, destination, safe_serialization=safe_serialization, **kwargs)
            finally:
                owner.project_configuration.automatic_checkpoint_naming = automatic
        checkpoint_phase(save)
        _main_value(lambda: _verify_native_dcp_payloads(Path(destination)))
    except BaseException:
        _abort_native_group(group)
        raise
    return str(path)


def finish_native_group(accelerator, output_dir):
    """Publish all staged components before checkpoint retention runs."""
    group = _native_group_for(output_dir)
    if group is None:
        return
    path = Path(output_dir).expanduser().absolute()
    if path != group["path"]:
        raise ValueError("Checkpoint group must be finished at its root directory")
    try:
        _main_value(lambda: _publish_native_stage(path, group["stage"], group["published"]))
    except BaseException:
        if not _main_value(lambda: group["published"][0]):
            for owner, iteration in group["iterations"].items():
                owner.project_configuration.iteration = iteration
        raise
    finally:
        errors = _main_value(lambda: _cleanup_save_stage(group["stage"], group["lock"][0]))
        _report_checkpoint_maintenance("Checkpoint staging cleanup failed", errors)
        _native_checkpoint_groups.pop(path, None)
        for owner in group["iterations"]:
            owner.__dict__.pop("videox_checkpoint_grouped", None)


def save_training_checkpoint(owners, output_dir=None, *, checkpoint_limit=None, safe_serialization=True, **kwargs):
    """Save registered participants using native Accelerate checkpoint formats."""
    participants = checkpoint_phase(lambda: _participants(owners))
    primary = next(iter(participants.values()))
    if (len(participants) == 1 and "" in participants and output_dir is not None
            and checkpoint_limit is None and not primary.project_configuration.automatic_checkpoint_naming):
        return save_native_transaction(primary, lambda owner, *a, **k: _native(owner, "save_state", *a, **k),
                                       output_dir, safe_serialization=safe_serialization, **kwargs)
    manager = AcceleratorCheckpoint(primary)
    manager.owners = participants
    manager.checkpoint_limit = checkpoint_limit
    return manager._save_native_group(output_dir, safe_serialization, kwargs)


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
    if not getattr(primary.project_configuration, "save_on_each_node", False):
        recover_native_checkpoint(primary, path)
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
    """Preserve native recovery semantics and clear prior failed-restore guards."""
    from .. import _failed_checkpoint_owners
    if owner in _failed_checkpoint_owners:
        if method == "save_state":
            raise RuntimeError("Cannot save after a failed restore; reload successfully first")
        result = checkpoint_phase(lambda: original(owner, *args, **kwargs))
        _set_failed([owner], False)
        return result
    return original(owner, *args, **kwargs)


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
    from torch.distributed._shard.metadata import ShardMetadata
    from torch.distributed._shard.sharded_tensor import Shard, ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import (
        ShardedTensorMetadata, TensorProperties)
    from torch.distributed.checkpoint import TensorStorageMetadata
    from torch.distributed.checkpoint._nested_dict import unflatten_state_dict

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
    plugin = accelerator.state.fsdp_plugin
    path = Path(input_dir) / f"optimizer_{optimizer_index}"
    try:
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.optimizer import \
            load_sharded_optimizer_state_dict
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

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
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import (ShardedOptimStateDictConfig,
                                        ShardedStateDictConfig, StateDictType)

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


class AcceleratorCheckpoint:
    """Checkpoint registration and standard Accelerator save/load dispatch.

    Registered participants use native Accelerate checkpoint formats and are
    published together after the save succeeds.
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

    def save_state(self, output_dir=None, *, safe_serialization=True, **kwargs):
        automatic = self.accelerator.project_configuration.automatic_checkpoint_naming
        native_per_node = self.accelerator.project_configuration.save_on_each_node and len(self.owners) == 1
        projects = {id(owner.project_configuration): (owner.project_configuration, owner.project_configuration.iteration)
                    for owner in self.owners.values()}
        if automatic:
            output_dir = None
        if native_per_node:
            from .. import _failed_checkpoint_owners

            def require_restored():
                if self.accelerator in _failed_checkpoint_owners:
                    raise RuntimeError("Cannot save after a failed restore; reload successfully first")

            checkpoint_phase(require_restored)
            result = _native(self.accelerator, "save_state", output_dir, safe_serialization=safe_serialization, **kwargs)
        elif len(self.owners) == 1 and not automatic and self.checkpoint_limit is None:
            result = save_native_transaction(self.accelerator,
                lambda owner, *a, **k: _native(owner, "save_state", *a, **k),
                output_dir, safe_serialization=safe_serialization, **kwargs)
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
        if output_dir is None:
            if not self.accelerator.project_configuration.automatic_checkpoint_naming:
                raise ValueError("output_dir is required for a checkpoint group")
            output_dir = Path(self.accelerator.project_dir) / "checkpoints" / f"checkpoint_{self.accelerator.save_iteration}"
        path = checkpoint_phase(lambda: Path(output_dir).expanduser().absolute())
        _agree([str(path), _save_hook_counts(self.owners)])
        run_id = _main_value(lambda: self._native_run_id or uuid.uuid4().hex)
        group = _begin_native_group(path)
        try:
            for name, owner in self.owners.items():
                save_native_group_component(owner,
                    lambda participant, *a, **k: _native(participant, "save_state", *a, **k),
                    path / name, safe_serialization=safe_serialization, **kwargs)
            def marker():
                (Path(group["stage"]) / ".videox_native_checkpoint.json").write_text(json.dumps(
                    {"run_id": run_id, "components": list(self.owners)}, sort_keys=True))
            _main_value(marker)
            finish_native_group(self.accelerator, path)
        except BaseException:
            if _native_checkpoint_groups.get(path) is group:
                _abort_native_group(group)
            raise
        self._native_run_id = run_id
        limit = self.checkpoint_limit
        if limit is None and self.accelerator.project_configuration.automatic_checkpoint_naming:
            limit = self.accelerator.project_configuration.total_limit
        if limit is not None:
            def prune():
                guard = _CheckpointLock(path.parent)
                try:
                    guard.acquire()
                    return _prune_checkpoints(path.parent, path, limit, run_id,
                                              marker=".videox_native_checkpoint.json",
                                              components=list(self.owners))
                except OSError as exc:
                    return [f"could not acquire checkpoint cleanup lock: {exc}"]
                finally:
                    guard.release()
            errors = _main_value(prune)
            _report_checkpoint_maintenance("Checkpoint saved, but retention cleanup failed", errors)
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


# Gradient checkpointing and distributed training setup.

_checkpoint_cpu_policy = {"enabled": False}


def configure_gradient_checkpointing_save_on_cpu(enabled):
    """Control CPU saved tensors for PyTorch and FSDP activation checkpoints.

    Training programs call this once after argument parsing. The policy is
    process-wide and applies only to checkpoint calls with autograd enabled.
    """
    import functools
    import importlib

    import torch.utils.checkpoint as checkpoint_module

    _checkpoint_cpu_policy["enabled"] = bool(enabled)
    if not enabled:
        return

    def wrap(checkpoint_fn):
        if getattr(checkpoint_fn, "videox_cpu_checkpoint_policy", None) is _checkpoint_cpu_policy:
            return checkpoint_fn

        @functools.wraps(checkpoint_fn)
        def checkpoint_with_cpu_offload(*args, **kwargs):
            if not _checkpoint_cpu_policy["enabled"] or not torch.is_grad_enabled():
                return checkpoint_fn(*args, **kwargs)
            with torch.autograd.graph.save_on_cpu():
                return checkpoint_fn(*args, **kwargs)

        checkpoint_with_cpu_offload.videox_cpu_checkpoint_policy = _checkpoint_cpu_policy
        return checkpoint_with_cpu_offload

    checkpoint_module.checkpoint = wrap(checkpoint_module.checkpoint)
    try:
        fsdp_checkpoint = importlib.import_module(
            "torch.distributed.algorithms._checkpoint.checkpoint_wrapper"
        )
    except ImportError:
        return
    if hasattr(fsdp_checkpoint, "torch_utils_checkpoint"):
        # PyTorch's wrapper captures this alias when its module is imported.
        fsdp_checkpoint.torch_utils_checkpoint = wrap(fsdp_checkpoint.torch_utils_checkpoint)


def initialize_training_seed(args, accelerator):
    """Choose one shared seed for model initialization and distributed sampling."""
    if args.seed is None:
        import secrets

        from accelerate.utils import broadcast_object_list

        seed = [secrets.randbelow(2**31) if accelerator.is_main_process else None]
        broadcast_object_list(seed, from_process=0)
        args.seed = seed[0]
        accelerator.print(f"No seed specified; using shared training seed {args.seed}.")
    return args.seed


# Optimizer and learning-rate configuration.

def parse_trainable_module_learning_rates(args, parser=None, *, preserve_unspecified_modules=False):
    """Validate the optional low learning rate and normalize module lists."""
    for name in ("low_learning_rate", "low_learning_rate_critic"):
        rate = getattr(args, name, None)
        if rate is not None and (not math.isfinite(rate) or rate < 0):
            message = f"--{name} must be finite and nonnegative."
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
    if not preserve_unspecified_modules:
        args.trainable_modules = args.trainable_modules or []
    return args


def apply_trainable_module_learning_rates(groups, named_parameters, args, accelerator, *, base="learning_rate", include_legacy=False):
    """Apply the optional rate to selected modules, retaining original groups."""
    rate = args.low_learning_rate
    if base == "learning_rate_critic" and getattr(args, "low_learning_rate_critic", None) is not None:
        rate = args.low_learning_rate_critic
    patterns = args.trainable_modules_low_learning_rate or []
    if not patterns:
        return groups
    factor = 1
    if base == "learning_rate" and getattr(args, "scale_lr", False):
        factor = args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    rate = getattr(args, base) / 2 if rate is None else rate * factor
    selected = set()
    for name, param in named_parameters:
        if param.requires_grad and any(
            pattern in name or (include_legacy and pattern.replace(".", "_") in name)
            for pattern in patterns
        ):
            selected.add(id(param))
    if not include_legacy:
        # Regular training scripts already use two groups. Reuse the low-LR
        # group so the optimizer and scheduler keep their normal group count.
        if len(groups) != 2:
            raise ValueError("Expected the standard two optimizer parameter groups")
        high = []
        moved = []
        for param in groups[0]["params"]:
            (moved if id(param) in selected else high).append(param)
        groups[0]["params"] = high
        groups[1]["params"].extend(moved)
        groups[1]["lr"] = rate
        count = sum(id(param) in selected for group in groups for param in group["params"])
    else:
        # PEFT and Kohya may provide multiple groups with distinct options.
        groups = list(groups)
        if groups and not isinstance(groups[0], dict):
            groups = [{"params": groups, "lr": getattr(args, base)}]
        result, extra = [], []
        count = 0
        for group in groups:
            normal, low = [], []
            for param in group["params"]:
                (low if id(param) in selected else normal).append(param)
            count += len(low)
            original = dict(group, params=normal)
            result.append(original)
            if low:
                extra.append(dict(group, params=low, lr=rate))
        groups = result + extra
    if accelerator.is_main_process:
        if count:
            accelerator.print(f"Low learning rate {rate:g} applied to {count} tensors matching {patterns}.")
        else:
            accelerator.print(f"Warning: low learning rate modules {patterns} matched no trainable optimizer parameters.")
    return groups


def build_trainable_optimizer_groups(named_parameters, args, accelerator, *, base="learning_rate", only_trainable=False):
    """Build standard main/low-LR groups from named model parameters.

    A parameter entry may contain a third, qualified name for selecting a
    component while retaining the original local-name matching behavior.
    """
    rate = getattr(args, base)
    groups = [{"params": [], "lr": rate}, {"params": [], "lr": rate / 2}]
    high_patterns = args.trainable_modules or []
    low_patterns = args.trainable_modules_low_learning_rate or []
    candidates = []
    seen = set()
    for entry in named_parameters:
        name, param = entry[:2]
        qualified_name = entry[2] if len(entry) == 3 else name
        if qualified_name in seen or (only_trainable and not param.requires_grad):
            continue
        seen.add(qualified_name)
        candidates.append((qualified_name, param))
        if any(pattern in name for pattern in high_patterns):
            groups[0]["params"].append(param)
        elif any(pattern in qualified_name for pattern in low_patterns):
            groups[1]["params"].append(param)
    return apply_trainable_module_learning_rates(groups, candidates, args, accelerator, base=base)


def create_optimizer(parameters, args, *, learning_rate=None):
    """Create the training optimizer while preserving the existing defaults."""
    use_8bit = getattr(args, "use_8bit_adam", False)
    use_came = getattr(args, "use_came", False)
    if use_8bit and use_came:
        raise ValueError("--use_8bit_adam and --use_came cannot be used together")
    lr = args.learning_rate if learning_rate is None else learning_rate
    if use_came:
        try:
            from came_pytorch import CAME
        except ImportError as exc:
            raise ImportError("Please install came_pytorch to use CAME.") from exc
        return CAME(parameters, lr=lr, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
    if use_8bit:
        try:
            from bitsandbytes.optim import AdamW8bit
        except ImportError as exc:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.") from exc
        optimizer_cls = AdamW8bit
    else:
        from torch.optim import AdamW
        optimizer_cls = AdamW
    return optimizer_cls(parameters, lr=lr, betas=(args.adam_beta1, args.adam_beta2),
                         weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)


# Training precision and trainable-state handling.

_full_bf16 = False
_gradient_reduce_dtype = "auto"
_fsdp2_precision_lock = threading.RLock()


def add_training_precision_args(parser):
    """Register the shared training and gradient communication precision options."""
    parser.add_argument(
        "--full_bf16", action="store_true",
        help="Train with BF16 model parameters and optimizer state where supported.",
    )
    parser.add_argument(
        "--gradient_reduce_dtype", choices=("auto", "fp32"), default="auto",
        help="Gradient communication dtype: backend default (auto) or FP32.",
    )


def configure_training_precision(args):
    """Apply precision choices before Accelerator and distributed model creation."""
    global _full_bf16, _gradient_reduce_dtype
    _full_bf16 = bool(getattr(args, "full_bf16", False))
    _gradient_reduce_dtype = getattr(args, "gradient_reduce_dtype", "auto")
    if _gradient_reduce_dtype not in ("auto", "fp32"):
        raise ValueError("--gradient_reduce_dtype must be auto or fp32.")
    if _full_bf16:
        args.mixed_precision = "bf16"


def configure_gradient_reduction(accelerator):
    """Configure FSDP or DeepSpeed before they wrap models and optimizers."""
    if _gradient_reduce_dtype == "auto":
        return
    mode = getattr(accelerator.distributed_type, "name", "")
    if mode == "FSDP":
        from dataclasses import is_dataclass, replace

        plugin = getattr(accelerator.state, "fsdp_plugin", None)
        if plugin is None:
            raise RuntimeError("FP32 gradient communication requires an FSDP plugin.")
        policy = plugin.mixed_precision_policy
        if policy is None:
            if getattr(plugin, "fsdp_version", 1) == 2:
                try:
                    from torch.distributed.fsdp import MixedPrecisionPolicy
                except ImportError as exc:
                    raise RuntimeError("This PyTorch version cannot configure FSDP2 gradient precision.") from exc
                policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
            else:
                from torch.distributed.fsdp import MixedPrecision
                policy = MixedPrecision(reduce_dtype=torch.float32)
        elif isinstance(policy, dict):
            policy = dict(policy, reduce_dtype=torch.float32)
        elif is_dataclass(policy):
            policy = replace(policy, reduce_dtype=torch.float32)
        else:
            raise RuntimeError("Unsupported FSDP mixed precision policy for FP32 gradient communication.")
        plugin.mixed_precision_policy = policy
    elif mode == "DEEPSPEED":
        plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        config = getattr(plugin, "deepspeed_config", None)
        if not isinstance(config, dict):
            raise RuntimeError("FP32 gradient communication requires a mutable DeepSpeed configuration.")
        zero = config.get("zero_optimization", {})
        if isinstance(zero, dict) and zero.get("zero_quantized_gradients", False):
            raise RuntimeError("FP32 gradient communication is incompatible with DeepSpeed quantized gradients.")
        autocast_config = config.get("torch_autocast", {})
        if isinstance(autocast_config, dict) and autocast_config.get("enabled", False):
            raise RuntimeError("FP32 gradient communication is not validated with DeepSpeed torch_autocast.")
        # DeepSpeed can replace communication_data_type with its sequence-parallel
        # setting during engine initialization. Override an explicit setting too,
        # otherwise FP32 would silently stop applying when that mode is enabled.
        for key in ("communication_data_type", "seq_parallel_communication_data_type"):
            if key == "seq_parallel_communication_data_type" and key not in config:
                continue
            previous = config.get(key)
            if previous is not None and str(previous).lower() not in ("fp32", "torch.float32"):
                accelerator.print(f"Overriding DeepSpeed {key}={previous} with fp32.")
            config[key] = "fp32"
    elif mode != "NO" and not mode.startswith("MULTI_"):
        raise RuntimeError(f"FP32 gradient communication is not supported for {mode or 'this backend'}.")
    if mode != "NO" and not accelerator.__dict__.get("videox_gradient_reduce_reported", False):
        accelerator.print(f"Gradient communication dtype: fp32 ({mode}).")
        accelerator.__dict__["videox_gradient_reduce_reported"] = True


def validate_deepspeed_gradient_reduction(accelerator, result):
    """Check the final engine setting after DeepSpeed applies its own overrides."""
    if _gradient_reduce_dtype != "fp32" or getattr(accelerator.distributed_type, "name", "") != "DEEPSPEED":
        return
    from deepspeed import DeepSpeedEngine

    objects = result if isinstance(result, (tuple, list)) else (result,)
    for obj in objects:
        engine = getattr(obj, "_orig_mod", obj)
        if not isinstance(engine, DeepSpeedEngine):
            continue
        dtype = getattr(engine, "communication_data_type", None)
        optimizer_dtype = getattr(engine.optimizer, "communication_data_type", dtype)
        if dtype != torch.float32 or optimizer_dtype != torch.float32:
            raise RuntimeError(
                "DeepSpeed changed FP32 gradient communication during engine initialization "
                f"(engine={dtype}, optimizer={optimizer_dtype}). Check its parallel configuration."
            )


def validate_fsdp_gradient_reduction(accelerator, result):
    """Reject FSDP1 child wrappers that override the requested reduce dtype."""
    if _gradient_reduce_dtype != "fp32" or getattr(accelerator.distributed_type, "name", "") != "FSDP":
        return
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if getattr(plugin, "fsdp_version", 1) != 1:
        return
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    objects = result if isinstance(result, (tuple, list)) else (result,)
    for obj in objects:
        if not isinstance(obj, torch.nn.Module):
            continue
        for name, module in obj.named_modules():
            if not isinstance(module, FSDP):
                continue
            handle = getattr(module, "_handle", None)
            flat_param = getattr(handle, "flat_param", None)
            if flat_param is None or not flat_param.requires_grad:
                continue
            policy = module.mixed_precision
            reduce_dtype = policy.reduce_dtype or policy.param_dtype or flat_param.dtype
            if reduce_dtype != torch.float32:
                raise RuntimeError(
                    f"FSDP module {name or '<root>'} overrides FP32 gradient communication "
                    f"with {reduce_dtype}. Check the auto-wrap policy and low-precision "
                    "parameters in modules excluded from mixed precision (such as BatchNorm)."
                )


def configure_ddp_gradient_reduction(accelerator, result):
    """Use an FP32 all-reduce for low-precision DDP buckets without changing grads."""
    if _gradient_reduce_dtype != "fp32" or not getattr(accelerator.distributed_type, "name", "").startswith("MULTI_"):
        return
    from torch.nn.parallel import DistributedDataParallel

    objects = result if isinstance(result, (tuple, list)) else (result,)
    for obj in objects:
        model = getattr(obj, "_orig_mod", obj)
        if not isinstance(model, DistributedDataParallel):
            continue
        if getattr(model, "_videox_fp32_gradient_reduction", False):
            continue
        if getattr(model, "mixed_precision", None) is not None:
            raise RuntimeError("FP32 gradient communication is not validated with DDP native mixed_precision.")
        hooks = getattr(model, "_comm_hooks", ())
        logging_data = model._get_ddp_logging_data() if hasattr(model, "_get_ddp_logging_data") else {}
        if hooks or logging_data.get("comm_hook"):
            raise RuntimeError("FP32 gradient communication cannot replace an existing DDP communication hook.")
        trainable = [p for p in model.parameters() if p.requires_grad]
        if any(p.dtype not in (torch.float16, torch.bfloat16, torch.float32) for p in trainable):
            raise RuntimeError("FP32 gradient communication does not support non-FP16/BF16/FP32 DDP parameters.")
        if any(p.dtype in (torch.float16, torch.bfloat16) for p in trainable):
            group = model.process_group

            def fp32_allreduce(process_group, bucket):
                tensor = bucket.buffer()
                reduced = tensor if tensor.dtype == torch.float32 else tensor.float()
                reduced.div_(dist.get_world_size(process_group))
                future = dist.all_reduce(reduced, group=process_group, async_op=True).get_future()

                def finish(value):
                    averaged = value.value()[0]
                    return averaged if averaged.dtype == tensor.dtype else averaged.to(dtype=tensor.dtype)

                return future.then(finish)

            try:
                model.register_comm_hook(group, fp32_allreduce)
            except RuntimeError as exc:
                raise RuntimeError("Cannot register FP32 DDP communication: another hook may already be installed.") from exc
        model._videox_fp32_gradient_reduction = True


def prepare_full_bf16(accelerator, args):
    """Convert training models before distributed wrapping and optimizer state creation."""
    import torch

    models = [arg for arg in args if isinstance(arg, torch.nn.Module)]
    optimizers = [arg for arg in args if isinstance(arg, torch.optim.Optimizer)]
    if not models:
        return
    mode = getattr(accelerator.distributed_type, "name", "")
    if mode in ("DEEPSPEED", "MEGATRON_LM"):
        raise RuntimeError("--full_bf16 is not supported with DeepSpeed or Megatron: their optimizers may maintain FP32 master state.")
    if not optimizers:
        return
    if accelerator.mixed_precision != "bf16":
        raise RuntimeError("--full_bf16 requires BF16 mixed precision.")
    supported = (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD)
    for optimizer in optimizers:
        if type(optimizer) not in supported:
            raise RuntimeError("--full_bf16 currently supports torch Adam, AdamW and SGD optimizers only.")
        if optimizer.state:
            raise RuntimeError("--full_bf16 must be configured before optimizer state is created.")
    parameters = [p for model in models for p in model.parameters()]
    owned = {id(p) for p in parameters}
    if any(id(p) not in owned for optimizer in optimizers for group in optimizer.param_groups for p in group["params"]):
        raise RuntimeError("--full_bf16 requires every optimizer parameter to belong to a model passed to prepare.")
    if any(p.grad is not None for p in parameters):
        raise RuntimeError("--full_bf16 must be configured before gradients are created.")
    tensors = parameters + [b for model in models for b in model.buffers()]
    storages = {}
    for tensor in tensors:
        if tensor.device.type != "meta" and tensor.numel() and tensor.is_floating_point():
            key = (tensor.device, tensor.untyped_storage().data_ptr())
            storages.setdefault(key, {})[id(tensor)] = tensor
    if any(len(group) > 1 and any(p.dtype != torch.bfloat16 for p in group.values())
           for group in storages.values()):
        raise RuntimeError("--full_bf16 cannot safely convert distinct tensors sharing storage.")
    for model in models:
        # Complex buffers (for example rotary frequencies) must stay complex.
        model._apply(lambda tensor: tensor.to(dtype=torch.bfloat16) if tensor.is_floating_point() else tensor)
    converted = {id(p) for model in models for p in model.parameters()}
    if any(id(p) not in converted for optimizer in optimizers for group in optimizer.param_groups for p in group["params"]):
        raise RuntimeError("Model conversion changed optimizer parameter identities; cannot train safely.")


def cast_fsdp1_flat_parameters_to_bf16(accelerator, result):
    """Undo Accelerate FSDP1 master upcasting before FSDP lazy initialization."""
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if getattr(accelerator.distributed_type, "name", "") != "FSDP":
        return
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if getattr(plugin, "fsdp_version", 1) != 1:
        return
    objects = result if isinstance(result, (tuple, list)) else (result,)
    seen = set()
    for obj in objects:
        if not isinstance(obj, torch.nn.Module):
            continue
        for unit in FSDP.fsdp_modules(obj):
            if id(unit) in seen:
                continue
            seen.add(id(unit))
            param = getattr(unit, "_flat_param", None)
            if param is None or not param.requires_grad or param.dtype == torch.bfloat16:
                continue
            handle = getattr(unit, "_handle", None)
            if param.device.type == "meta" or handle is None or not hasattr(handle, "_orig_param_dtype") or getattr(unit, "_is_root", None) is not None:
                raise RuntimeError("Cannot safely set BF16 FSDP1 parameters before lazy initialization.")
            param.data = param.data.to(torch.bfloat16)
            handle._orig_param_dtype = torch.bfloat16


def promote_optimizer_parameters(accelerator, args):
    """Promote optimizer-owned parameters before ordinary DDP/FSDP2 wrapping."""
    import torch

    mode = getattr(accelerator.distributed_type, "name", "")
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    fsdp2 = mode == "FSDP" and getattr(plugin, "fsdp_version", 1) == 2
    if accelerator.mixed_precision not in ("fp16", "bf16"):
        return
    if mode in ("DEEPSPEED", "MEGATRON_LM") or (mode == "FSDP" and not fsdp2):
        return

    models = [arg for arg in args if isinstance(arg, torch.nn.Module)]
    optimizers = [arg for arg in args if isinstance(arg, torch.optim.Optimizer)]
    if not models or not optimizers:
        return

    parameters = [parameter for model in models for parameter in model.parameters()]
    owned = {id(parameter) for parameter in parameters}
    seen, to_promote = set(), []
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) not in owned or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                if not parameter.requires_grad or parameter.dtype not in (torch.float16, torch.bfloat16):
                    continue
                if parameter.grad is not None or optimizer.state.get(parameter):
                    raise RuntimeError(
                        "FP32 master weights must be prepared before gradients or optimizer state are created."
                    )
                if hasattr(parameter, "to_local"):
                    raise RuntimeError("Cannot promote an already sharded low-precision parameter safely.")
                to_promote.append(parameter)

    if not to_promote:
        return
    aliases = {}
    for parameter in parameters:
        if parameter.device.type == "meta" or parameter.numel() == 0:
            continue
        # Prepared FSDP2 models expose DTensors. There is no need to examine
        # their storage when all optimizer-owned parameters are already FP32.
        if hasattr(parameter, "to_local"):
            continue
        key = (parameter.device, parameter.untyped_storage().data_ptr())
        aliases.setdefault(key, set()).add(id(parameter))
    for parameter in to_promote:
        if parameter.device.type != "meta" and parameter.numel():
            key = (parameter.device, parameter.untyped_storage().data_ptr())
            if len(aliases.get(key, ())) > 1:
                raise RuntimeError(
                    "Cannot safely promote distinct trainable parameters sharing storage; "
                    "preserve the tie explicitly before Accelerator.prepare."
                )
    # Validate every candidate before mutating any parameter. Preserve identity,
    # since the optimizer already owns these objects and DDP is not wrapped yet.
    for parameter in to_promote:
        parameter.data = parameter.data.to(torch.float32)


def promote_fsdp1_flat_parameters(accelerator, result):
    # Complete FP32 master promotion on Accelerate versions that lack it.
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if getattr(accelerator.distributed_type, "name", "") != "FSDP":
        return
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if getattr(plugin, "fsdp_version", 1) != 1 or accelerator.mixed_precision not in ("fp16", "bf16"):
        return
    objects = result if isinstance(result, (tuple, list)) else (result,)
    units, seen = [], set()
    for obj in objects:
        if not isinstance(obj, torch.nn.Module):
            continue
        for unit in FSDP.fsdp_modules(obj):
            if id(unit) in seen:
                continue
            seen.add(id(unit))
            param = getattr(unit, "_flat_param", None)
            if param is None or not param.requires_grad or param.dtype not in (torch.float16, torch.bfloat16):
                continue
            handle = getattr(unit, "_handle", None)
            if param.device.type == "meta" or handle is None or not hasattr(handle, "_orig_param_dtype"):
                raise RuntimeError("Cannot safely promote this FSDP1 flat parameter to FP32.")
            if getattr(unit, "_is_root", None) is not None:
                raise RuntimeError("FSDP1 FP32 promotion must occur before lazy initialization.")
            units.append((param, handle))
    for param, handle in units:
        # This matches Accelerate FSDP1 promotion and runs before lazy init.
        param.data = param.data.to(torch.float32)
        handle._orig_param_dtype = torch.float32



def configure_fsdp1_optimizer_identity(accelerator, args):
    # Optimizers in repository trainers are built before FSDP wrapping. Flat
    # parameters cannot preserve their references or per-parameter groups.
    import logging

    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if getattr(accelerator.distributed_type, "name", "") != "FSDP":
        return
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if getattr(plugin, "fsdp_version", 1) != 1 or getattr(plugin, "use_orig_params", False):
        return
    models = [obj for obj in args if isinstance(obj, torch.nn.Module) and not isinstance(obj, FSDP)]
    optimizers = [obj for obj in args if isinstance(obj, torch.optim.Optimizer)]
    parameters = {id(p) for model in models for p in model.parameters() if p.requires_grad}
    if any(id(p) in parameters for optimizer in optimizers for group in optimizer.param_groups for p in group["params"]):
        plugin.use_orig_params = True
        if accelerator.is_main_process:
            logging.getLogger(__name__).warning(
                "FSDP use_orig_params enabled to preserve optimizer parameters and parameter groups created before prepare."
            )


def configure_declared_fsdp_precision(accelerator, args):
    # Keep declared FP32 modules separate when FSDP1 flattens mixed weights.
    import torch

    if getattr(accelerator.distributed_type, "name", "") != "FSDP":
        return
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if getattr(plugin, "fsdp_version", 1) != 1:
        return
    models = [arg for arg in args if isinstance(arg, torch.nn.Module)
              and getattr(arg, "_keep_in_fp32_modules", None)]
    mixed = []
    for model in models:
        dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
        if torch.float32 in dtypes and dtypes.intersection((torch.float16, torch.bfloat16)):
            mixed.append(model)
    if not mixed:
        return
    if getattr(getattr(plugin, "cpu_offload", None), "offload_params", False):
        # Keep the legacy uniform dtype for the separate CPU-offload path.
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            accelerator.mixed_precision, torch.float32
        )
        for model in mixed:
            model.to(dtype=dtype)
        return
    if len(mixed) != 1:
        raise RuntimeError("Mixed-dtype FSDP1 models must be prepared separately.")
    model = mixed[0]
    pinned = tuple(name for name, module in model.named_modules() if name and
                   any(p.dtype == torch.float32 for p in module.parameters(recurse=False)))
    if not pinned:
        raise RuntimeError("Cannot safely separate model FP32 parameters for FSDP1.")
    from .fsdp_utils import configure_fsdp_precision
    configure_fsdp_precision(model, accelerator, keep_in_fp32_modules=pinned)

def save_trainable_lora_state(accelerator, models, weights, output_dir):
    """Save exact trainable state if a custom hook removed the native model."""
    if not accelerator.is_main_process or weights:
        return
    path = Path(output_dir)
    adapter_export = any(
        "lora" in item.name.lower() or item.name.lower().startswith("pdd")
        for item in path.rglob("*.safetensors")
    )
    if not adapter_export:
        return

    import torch
    from safetensors.torch import save_file

    for index, model in enumerate(models):
        model = accelerator.unwrap_model(model)
        parameters = {
            name: parameter.detach().cpu().contiguous().clone()
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        }
        state_keys = set(model.state_dict())
        buffers = {
            name: buffer.detach().cpu().contiguous().clone()
            for name, buffer in model.named_buffers(remove_duplicate=False)
            if name in state_keys
        }
        state = {**parameters, **buffers}
        # Reuse the exported adapter only when it losslessly represents every
        # trainable parameter and persistent buffer. Otherwise keep the exact
        # training sidecar, including for unsupported adapter variants.
        if state and len(models) == 1:
            try:
                exported = legacy_trainable_lora_state(model, path)
                if exported is not None and state.keys() <= exported.keys() and all(
                    value.dtype == exported[name].dtype and torch.equal(value, exported[name])
                    for name, value in state.items()
                ):
                    continue
            except (OSError, ValueError, RuntimeError, KeyError):
                pass
        if state:
            save_file(
                state,
                str(path / f"training_trainable_model_{index}.safetensors"),
                metadata={"format": "pt"},
            )

def legacy_trainable_lora_state(model, path):
    # Older custom hooks only exported inference-format LoRA weights. Recover
    # them when all trainable parameters can be matched without ambiguity.
    from safetensors.torch import load_file

    export = next((path / name for name in ("lora_diffusion_pytorch_model.safetensors",
                   "pdd_live.safetensors", "pdd.safetensors") if (path / name).is_file()), None)
    if export is None:
        return None
    source = load_file(str(export), device="cpu")
    parameters = dict(model.named_parameters(remove_duplicate=False))
    expected = {name: value for name, value in parameters.items() if value.requires_grad}
    if not expected:
        return None
    result, used = {}, set()

    def matches(name, value):
        names = {name, name.replace("network.", ""), name.replace(".default.", ".")}
        if ".lora_adapter." in name:
            prefix, suffix = name.split(".lora_adapter.", 1)
            names.add("lora_unet_" + prefix.replace(".", "_") + "." + suffix)
        return [key for key in names if key in source and source[key].shape == value.shape]

    for name, value in expected.items():
        keys = matches(name, value)
        if len(keys) != 1 or keys[0] in used:
            return None
        used.add(keys[0])
        result[name] = source[keys[0]]
    # Restore matching persistent adapter buffers, such as the LoRA alpha.
    state_keys = set(model.state_dict())
    for name, value in model.named_buffers(remove_duplicate=False):
        if name not in state_keys:
            continue
        keys = matches(name, value)
        if len(keys) == 1 and keys[0] not in used:
            result[name] = source[keys[0]]
            used.add(keys[0])
    return result

def load_trainable_lora_state(accelerator, models, input_dir):
    """Restore compact training state when no native model state was saved."""
    from safetensors.torch import load_file

    path = Path(input_dir)
    for index in range(len(models) - 1, -1, -1):
        suffix = "" if index == 0 else f"_{index}"
        native_exists = any(
            (path / filename).is_file()
            for filename in (f"model{suffix}.safetensors", f"pytorch_model{suffix}.bin")
        )
        original_index = next(
            (position for position, model in enumerate(accelerator._models) if model is models[index]),
            index,
        )
        state_file = path / f"training_trainable_model_{original_index}.safetensors"
        if native_exists:
            continue
        model = accelerator.unwrap_model(models[index])
        if state_file.is_file():
            state = load_file(str(state_file), device="cpu")
        elif len(models) == 1:
            state = legacy_trainable_lora_state(model, path)
            if state is None:
                continue
        else:
            continue
        expected = {
            name for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        }
        if not expected.issubset(state):
            raise RuntimeError("Training checkpoint is missing trainable model parameters.")
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected trainable model keys: {result.unexpected_keys}")
        models.pop(index)

def defer_checkpoint_removal(accelerator, path):
    # All ranks may discover the same old checkpoint. Only the main rank owns
    # its deletion; it is not removed until a later save succeeds everywhere.
    save_on_each_node = getattr(accelerator.project_configuration, "save_on_each_node", False)
    writer = accelerator.is_local_main_process if save_on_each_node else accelerator.is_main_process
    if not writer:
        return
    path = Path(path)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    pending = accelerator.__dict__.setdefault("videox_checkpoint_removals", {})
    pending[str(path)] = (info.st_dev, info.st_ino)

def finish_checkpoint_removals(accelerator, saved_path=None):
    import logging
    import shutil

    if saved_path is not None:
        finish_native_group(accelerator, saved_path)
    if getattr(accelerator.distributed_type, "name", "") in ("FSDP", "DEEPSPEED"):
        # Native distributed saves require all ranks. Wait for rank-local RNG
        # and other files too, before deleting a previously usable checkpoint.
        accelerator.wait_for_everyone()
    accelerator.__dict__.pop("videox_checkpoint_grouped", None)
    pending = accelerator.__dict__.get("videox_checkpoint_removals", {})
    destination = os.path.abspath(os.fspath(saved_path)) if saved_path is not None else None
    save_on_each_node = getattr(accelerator.project_configuration, "save_on_each_node", False)
    for name, identity in list(pending.items()):
        lock_fd = None
        try:
            if destination and os.path.commonpath((os.path.abspath(name), destination)) == os.path.abspath(name):
                pending.pop(name, None)
                continue
            if save_on_each_node:
                # Local main ranks on different nodes may share this parent.
                # Lock the existing directory, so no persistent lock file is needed.
                import fcntl
                lock_fd = os.open(os.path.dirname(os.path.abspath(name)), os.O_RDONLY)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            info = os.lstat(name)
            if (info.st_dev, info.st_ino) != identity:
                logging.getLogger(__name__).warning("Checkpoint changed before cleanup; keeping %s", name)
                pending.pop(name, None)
                continue
            shutil.rmtree(name)
            pending.pop(name, None)
        except FileNotFoundError:
            pending.pop(name, None)
        except OSError as exc:
            logging.getLogger(__name__).warning("Checkpoint cleanup failed for %s: %s", name, exc)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)


def register_ema_checkpoint(accelerator, ema, prefix="training_ema"):
    # FSDP/ZeRO EMA tensors are rank-local; each participating rank saves its
    # own state. DDP saves on the main rank and other ranks load that copy.
    import torch

    status = {"loaded": False}

    def save_ema(models, weights, output_dir):
        path = Path(output_dir) / f"{prefix}_{accelerator.process_index}.pt"
        torch.save(ema.state_dict(), path)

    def load_ema(models, input_dir):
        status["loaded"] = False
        path = Path(input_dir) / f"{prefix}_{accelerator.process_index}.pt"
        mode = getattr(accelerator.distributed_type, "name", "")
        if not path.is_file() and mode in ("NO", "MULTI_GPU", "MULTI_CPU"):
            if getattr(accelerator.project_configuration, "save_on_each_node", False):
                rank = accelerator.process_index - accelerator.local_process_index
            else:
                rank = 0
            path = Path(input_dir) / f"{prefix}_{rank}.pt"
        if path.is_file():
            ema.load_state_dict(torch.load(path, map_location=accelerator.device, weights_only=False))
            status["loaded"] = True

    accelerator.register_save_state_pre_hook(save_ema)
    accelerator.register_load_state_pre_hook(load_ema)
    return status


def install_training_precision():
    """Install common mixed-precision and compact LoRA checkpoint handling."""
    from accelerate import Accelerator

    current_prepare = Accelerator.prepare
    if getattr(current_prepare, "videox_training_precision", False):
        return

    @functools.wraps(current_prepare)
    def prepare(accelerator, *args, **kwargs):
        enabled = os.environ.get("VIDEOX_FP32_MASTER_WEIGHTS", "1").strip().lower() not in ("0", "false", "no", "off")
        if _full_bf16:
            prepare_full_bf16(accelerator, args)
        elif enabled:
            promote_optimizer_parameters(accelerator, args)
        configure_fsdp1_optimizer_identity(accelerator, args)
        if not _full_bf16:
            configure_declared_fsdp_precision(accelerator, args)
        configure_gradient_reduction(accelerator)
        plugin = getattr(accelerator.state, "fsdp_plugin", None)
        fsdp2 = getattr(accelerator.distributed_type, "name", "") == "FSDP" and getattr(plugin, "fsdp_version", 1) == 2
        if _full_bf16 and fsdp2:
            # Accelerate's FSDP2 wrapper otherwise upcasts trainable parameters
            # before sharding. The already configured FSDP BF16 policy remains
            # active; restore the state even when prepare raises.
            with _fsdp2_precision_lock:
                state = accelerator.state
                if not hasattr(state, "_mixed_precision"):
                    raise RuntimeError("Unsupported Accelerate FSDP2 precision runtime.")
                previous_precision = state._mixed_precision
                state._mixed_precision = "no"
                try:
                    result = current_prepare(accelerator, *args, **kwargs)
                finally:
                    state._mixed_precision = previous_precision
        else:
            result = current_prepare(accelerator, *args, **kwargs)
        if _full_bf16:
            cast_fsdp1_flat_parameters_to_bf16(accelerator, result)
            import torch
            objects = result if isinstance(result, (tuple, list)) else (result,)
            if any(p.is_floating_point() and p.dtype != torch.bfloat16
                   for obj in objects if isinstance(obj, torch.nn.Module) for p in obj.parameters()):
                raise RuntimeError("--full_bf16 could not preserve BF16 model parameters on this runtime.")
        elif enabled:
            promote_fsdp1_flat_parameters(accelerator, result)
        validate_deepspeed_gradient_reduction(accelerator, result)
        validate_fsdp_gradient_reduction(accelerator, result)
        configure_ddp_gradient_reduction(accelerator, result)
        return result

    current_save, current_load = Accelerator.save_state, Accelerator.load_state

    @functools.wraps(current_save)
    def save_state(accelerator, *args, **kwargs):
        from .. import _accelerate_patch_requested
        mode = getattr(accelerator.distributed_type, "name", "")
        handle = None
        if _accelerate_patch_requested() and mode in ("NO", "MULTI_GPU", "MULTI_CPU"):
            handle = accelerator.register_save_state_pre_hook(
                functools.partial(save_trainable_lora_state, accelerator)
            )
        try:
            result = current_save(accelerator, *args, **kwargs)
        finally:
            if handle is not None:
                handle.remove()
        destination = result if isinstance(result, (str, os.PathLike)) else kwargs.get("output_dir", args[0] if args else None)
        if not accelerator.__dict__.get("videox_checkpoint_grouped", False) and _native_group_for(destination) is None:
            finish_checkpoint_removals(accelerator, destination)
        return result

    @functools.wraps(current_load)
    def load_state(accelerator, *args, **kwargs):
        from .. import _accelerate_patch_requested
        if not _accelerate_patch_requested() or getattr(accelerator.distributed_type, "name", "") not in ("NO", "MULTI_GPU", "MULTI_CPU"):
            return current_load(accelerator, *args, **kwargs)
        handle = accelerator.register_load_state_pre_hook(
            functools.partial(load_trainable_lora_state, accelerator)
        )
        try:
            return current_load(accelerator, *args, **kwargs)
        finally:
            handle.remove()

    prepare.videox_training_precision = True
    def accelerator_checkpoint_staging_path(accelerator, path):
        return checkpoint_staging_path(accelerator, path)

    Accelerator.checkpoint_staging_path = accelerator_checkpoint_staging_path
    Accelerator.begin_checkpoint_group = begin_checkpoint_group
    Accelerator.abort_checkpoint_group = abort_checkpoint_group
    Accelerator.recover_checkpoint_directories = recover_checkpoint_directories
    Accelerator.defer_checkpoint_removal = defer_checkpoint_removal
    Accelerator.finish_checkpoint_removals = finish_checkpoint_removals
    Accelerator.register_ema_checkpoint = register_ema_checkpoint
    Accelerator.prepare = prepare
    Accelerator.save_state = save_state
    Accelerator.load_state = load_state
