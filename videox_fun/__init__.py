import functools
import importlib.util
import logging
import os
from weakref import WeakSet

if importlib.util.find_spec("turbox") is not None:
    import turbox
elif importlib.util.find_spec("paifuser") is not None:
    import paifuser

# Imported conditionally rather than unconditionally-then-bailing the way `videox_fun.pipeline` does it, because
# this runs on `import videox_fun` itself: with the variable unset that import does not load `perf_metrics` at all,
# and costs exactly what it did before. (`videox_fun.utils` exports from it and so loads it either way, which buys
# nothing but a bytecode load -- the module imports only the standard library and `torch`.) Here rather than in
# `videox_fun.pipeline` because the training scripts are what this measures, and three of them never import that
# package -- but all 113 of them import `videox_fun.models`, which this reaches through.
if os.environ.get("VIDEOX_PERF", "0").strip() not in ("", "0"):
    from .utils.perf_metrics import install_training as _install_perf_training

    _install_perf_training()

# Repository-wide FSDP optimizer restore compatibility. Keep this bootstrap here
# so importing the package installs it without eagerly importing videox_fun.utils.
# Preserve explicit opt-ins and opt-outs when the package is reloaded.
if "_accelerate_patch_owners" not in globals():
    _accelerate_patch_owners = WeakSet()
    _accelerate_patch_disabled_owners = WeakSet()
    _accelerate_patches_automatic = False
    _accelerate_patch_warned_unsupported = False

if "_failed_checkpoint_owners" not in globals():
    _failed_checkpoint_owners = WeakSet()


def _accelerate_patch_requested():
    return os.environ.get("VIDEOX_ACCELERATE_PATCHES", "1").strip().lower() not in ("", "0", "false", "no", "off")


def _eligible_fsdp_checkpoint(plugin):
    return (getattr(plugin, "fsdp_version", 1) == 1
            and getattr(getattr(plugin, "state_dict_type", None), "name", None) == "SHARDED_STATE_DICT")


def _checkpoint_warning(message):
    """Log without allowing a warning filter or logging handler to desynchronize ranks."""
    try:
        logging.getLogger("videox_fun.checkpoint").warning("%s", message)
    except Exception:
        pass


def supports_fsdp_optimizer_adapter(plugin=None):
    """Check the interfaces needed by the FSDP1 restore adapter."""
    import inspect

    try:
        import accelerate.accelerator as implementation
        import torch.distributed as dist
        from torch.distributed._shard.metadata import ShardMetadata
        from torch.distributed._shard.sharded_tensor import (Shard,
                                                             ShardedTensor)
        from torch.distributed._shard.sharded_tensor.metadata import (
            ShardedTensorMetadata, TensorProperties)
        from torch.distributed.checkpoint import (FileSystemReader,
                                                  TensorStorageMetadata, load)
        from torch.distributed.checkpoint._nested_dict import \
            unflatten_state_dict
        from torch.distributed.checkpoint.default_planner import \
            DefaultLoadPlanner
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp._common_utils import _get_param_to_fqns

        inspect.signature(implementation.load_fsdp_optimizer).bind(None, None, None, None, None, 0)
        required = (ShardMetadata, Shard, ShardedTensor, ShardedTensorMetadata, TensorProperties,
                    FileSystemReader, TensorStorageMetadata, load, unflatten_state_dict,
                    DefaultLoadPlanner, _get_param_to_fqns,
                    getattr(dist, "get_global_rank", None), getattr(FSDP, "optim_state_dict_to_load", None),
                    getattr(ShardedTensor, "_init_from_local_shards_and_global_metadata", None))
        if not all(callable(value) for value in required):
            return False
        offload = getattr(plugin, "cpu_offload", None)
        offload_enabled = offload if isinstance(offload, bool) else getattr(offload, "offload_params", False)
        if offload_enabled:
            # The optional CPU-offload cleanup uses a separate runtime contract.
            from .utils.fsdp_utils import _require_cpu_offload_runtime
            _require_cpu_offload_runtime()
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
        return False
    return True


def _install_accelerate_dispatch():
    import inspect

    import accelerate.accelerator as implementation

    original = getattr(implementation, "load_fsdp_optimizer", None)
    if original is None or getattr(original, "_videox_optimizer_dispatch", False) is True:
        return
    try:
        accepts_adapter_only = "adapter_only" in inspect.signature(original).parameters
    except (TypeError, ValueError):
        _checkpoint_warning("Cannot inspect Accelerate optimizer restore; leaving its native implementation unchanged")
        return

    def native_load(plugin, owner, optimizer, model, input_dir, optimizer_index, adapter_only, kwargs):
        if accepts_adapter_only:
            kwargs = {**kwargs, "adapter_only": adapter_only}
        elif adapter_only:
            raise TypeError("This Accelerate version does not support adapter_only optimizer restore.")
        return original(plugin, owner, optimizer, model, input_dir, optimizer_index, **kwargs)

    @functools.wraps(original)
    def load_optimizer(plugin, owner, optimizer, model, input_dir, optimizer_index=0, adapter_only=False, **kwargs):
        global _accelerate_patch_warned_unsupported
        automatic = _accelerate_patches_automatic and _accelerate_patch_requested()
        enabled = owner in _accelerate_patch_owners or automatic
        if enabled and owner not in _accelerate_patch_disabled_owners and not adapter_only and _eligible_fsdp_checkpoint(plugin):
            if supports_fsdp_optimizer_adapter(plugin) and kwargs.get("use_dcp", True):
                import torch

                from .utils.training_utils import (
                    _load_generic_fsdp_optimizer, _load_sharded_fsdp_optimizer,
                    _OptimizerAdapterFallback)

                raw = getattr(optimizer, "optimizer", optimizer)
                optimized = (type(raw) in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD)
                             and len(getattr(owner, "_models", ())) == len(getattr(owner, "_optimizers", ())))
                loader = _load_sharded_fsdp_optimizer if optimized else _load_generic_fsdp_optimizer
                try:
                    return loader(owner, optimizer, model, input_dir, optimizer_index)
                except _OptimizerAdapterFallback as exc:
                    if optimized:
                        # The generic PyTorch DCP reader builds state from the
                        # checkpoint, even for a newly constructed optimizer.
                        try:
                            return _load_generic_fsdp_optimizer(owner, optimizer, model, input_dir, optimizer_index)
                        except _OptimizerAdapterFallback as generic_exc:
                            exc = generic_exc
                    if owner.is_main_process:
                        _checkpoint_warning(f"VideoX optimizer adapter could not prepare this checkpoint; "
                                            f"using native Accelerate restore, which may not recover an empty "
                                            f"optimizer's saved state on this version: {exc}")
                    return native_load(plugin, owner, optimizer, model, input_dir, optimizer_index, adapter_only, kwargs)
            if not _accelerate_patch_warned_unsupported and kwargs.get("use_dcp", True):
                _checkpoint_warning("VideoX FSDP optimizer adapter is unavailable for this runtime or CPUOffload; "
                                    "using native Accelerate restore")
                _accelerate_patch_warned_unsupported = True
        return native_load(plugin, owner, optimizer, model, input_dir, optimizer_index, adapter_only, kwargs)

    load_optimizer._videox_optimizer_dispatch = True
    # Accelerator imports this symbol directly; existing Accelerator class aliases
    # and instances therefore use the same dispatch without constructor wrapping.
    implementation.load_fsdp_optimizer = load_optimizer


def _install_checkpoint_lifecycle():
    from accelerate import Accelerator

    if getattr(Accelerator.save_state, "_videox_native_checkpoint", None) is not None:
        return
    native_save, native_load, native_backward = Accelerator.save_state, Accelerator.load_state, Accelerator.backward

    def checkpoint(owner):
        manager = owner.__dict__.get("_videox_checkpoint_manager")
        if manager is None:
            from .utils.training_utils import AcceleratorCheckpoint
            manager = AcceleratorCheckpoint(owner)
            owner.__dict__["_videox_checkpoint_manager"] = manager
        return manager

    if hasattr(Accelerator, "checkpoint"):
        raise RuntimeError("Accelerator.checkpoint already exists; cannot install the VideoX checkpoint extension")
    Accelerator.checkpoint = property(checkpoint)

    @functools.wraps(native_save)
    def save_state(owner, output_dir=None, safe_serialization=True, **kwargs):
        manager = owner.__dict__.get("_videox_checkpoint_manager")
        if manager is not None and manager.configured:
            return manager.save_state(output_dir, safe_serialization=safe_serialization, **kwargs)
        mode = getattr(owner.distributed_type, "name", None)
        collective_ddp = (mode in ("MULTI_GPU", "MULTI_CPU")
                          and owner not in _accelerate_patch_disabled_owners
                          and owner.__dict__.get("videox_checkpoint_all_ranks", False))
        if (_accelerate_patch_requested() and owner not in _accelerate_patch_disabled_owners
                and (mode in ("NO", "FSDP", "DEEPSPEED") or collective_ddp)
                and (output_dir is not None or owner.project_configuration.automatic_checkpoint_naming)
                and not getattr(owner.project_configuration, "save_on_each_node", False)):
            from .utils.training_utils import (_native_group_for,
                                               save_native_group_component,
                                               save_native_transaction)
            if output_dir is not None and (_native_group_for(output_dir) is not None or
                    (owner.__dict__.get("videox_checkpoint_grouped", False)
                     and not owner.project_configuration.automatic_checkpoint_naming)):
                return save_native_group_component(owner, native_save, output_dir,
                                                   safe_serialization=safe_serialization, **kwargs)
            return save_native_transaction(owner, native_save, output_dir,
                                           safe_serialization=safe_serialization, **kwargs)
        return native_save(owner, output_dir, safe_serialization=safe_serialization, **kwargs)

    @functools.wraps(native_load)
    def load_state(owner, input_dir=None, **kwargs):
        manager = owner.__dict__.get("_videox_checkpoint_manager")
        if manager is not None and manager.configured:
            return manager.load_state(input_dir, **kwargs)
        mode = getattr(owner.distributed_type, "name", None)
        collective_ddp = (mode in ("MULTI_GPU", "MULTI_CPU")
                          and owner not in _accelerate_patch_disabled_owners
                          and owner.__dict__.get("videox_checkpoint_all_ranks", False))
        if (_accelerate_patch_requested() and (mode in ("NO", "FSDP", "DEEPSPEED") or collective_ddp)
                and input_dir is not None
                and not getattr(owner.project_configuration, "save_on_each_node", False)):
            from .utils.training_utils import recover_native_checkpoint
            recover_native_checkpoint(owner, input_dir)
        if mode == "FSDP":
            from .utils.training_utils import dispatch_checkpoint
            return dispatch_checkpoint(owner, "load_state", native_load, input_dir, **kwargs)
        return native_load(owner, input_dir, **kwargs)

    @functools.wraps(native_backward)
    def backward(owner, *args, **kwargs):
        if owner in _failed_checkpoint_owners:
            raise RuntimeError("Checkpoint restore failed; reload successfully before continuing training")
        return native_backward(owner, *args, **kwargs)

    save_state._videox_native_checkpoint = native_save
    load_state._videox_native_checkpoint = native_load
    backward._videox_native_checkpoint = native_backward
    Accelerator.save_state, Accelerator.load_state, Accelerator.backward = save_state, load_state, backward


def install_accelerate_patches():
    """Install lazy lifecycle guards and the version-specific optimizer adapter."""
    global _accelerate_patches_automatic
    if importlib.util.find_spec("accelerate") is None:
        return
    # The guards also compare rank opt-out/version settings before FSDP I/O.
    _install_checkpoint_lifecycle()
    if not _accelerate_patch_requested() or not supports_fsdp_optimizer_adapter():
        return
    _install_accelerate_dispatch()
    _accelerate_patches_automatic = True


def enable_accelerate_patch(accelerator):
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if getattr(accelerator.distributed_type, "name", None) != "FSDP" or not _eligible_fsdp_checkpoint(plugin):
        return accelerator
    if not supports_fsdp_optimizer_adapter(plugin):
        raise RuntimeError("FSDP optimizer restore adapter is not validated for this runtime or CPUOffload")
    _install_checkpoint_lifecycle()
    _install_accelerate_dispatch()
    _accelerate_patch_disabled_owners.discard(accelerator)
    _accelerate_patch_owners.add(accelerator)
    return accelerator


def disable_accelerate_patch(accelerator):
    _accelerate_patch_owners.discard(accelerator)
    _accelerate_patch_disabled_owners.add(accelerator)


install_accelerate_patches()

# Load training support once without importing optional image/video utilities.
# Register the canonical name temporarily during execution for dataclasses;
# utils.__init__ restores it once the parent package is loaded.
import sys

_training_module_name = f"{__name__}.utils.training_utils"
_training_bootstrap_name = f"{__name__}._training_utils_bootstrap"
_training_module = sys.modules.get(_training_module_name) or sys.modules.get(_training_bootstrap_name)
if _training_module is None:
    _training_path = os.path.join(os.path.dirname(__file__), "utils", "training_utils.py")
    _training_spec = importlib.util.spec_from_file_location(_training_module_name, _training_path)
    if _training_spec is None or _training_spec.loader is None:
        raise ImportError(f"Cannot load training support from {_training_path}")
    _training_module = importlib.util.module_from_spec(_training_spec)
    sys.modules[_training_module_name] = _training_module
    sys.modules[_training_bootstrap_name] = _training_module
    try:
        _training_spec.loader.exec_module(_training_module)
    except BaseException:
        sys.modules.pop(_training_bootstrap_name, None)
        raise
    finally:
        sys.modules.pop(_training_module_name, None)
_training_module.install_training_precision()
