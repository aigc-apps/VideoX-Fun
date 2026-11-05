import functools
import importlib.util
import os
import warnings
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


def _supported_accelerate_versions():
    import accelerate
    import torch
    from packaging.version import Version

    versions = (Version(accelerate.__version__), Version(torch.__version__))
    return tuple(version.public for version in versions) == ("1.7.0", "2.7.0")


def _install_accelerate_dispatch():
    import accelerate.accelerator as implementation

    original = implementation.load_fsdp_optimizer
    if getattr(original, "_videox_optimizer_dispatch", False) is True:
        return

    @functools.wraps(original)
    def load_optimizer(plugin, owner, optimizer, model, input_dir, optimizer_index=0, adapter_only=False):
        global _accelerate_patch_warned_unsupported
        automatic = _accelerate_patches_automatic and _accelerate_patch_requested()
        enabled = owner in _accelerate_patch_owners or automatic
        if enabled and owner not in _accelerate_patch_disabled_owners and not adapter_only and _eligible_fsdp_checkpoint(plugin):
            if _supported_accelerate_versions():
                import torch
                from .utils.checkpoint_utils import \
                    _load_sharded_fsdp_optimizer, _load_generic_fsdp_optimizer

                raw = getattr(optimizer, "optimizer", optimizer)
                optimized = (type(raw) in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD)
                             and len(getattr(owner, "_models", ())) == len(getattr(owner, "_optimizers", ())))
                loader = _load_sharded_fsdp_optimizer if optimized else _load_generic_fsdp_optimizer
                return loader(owner, optimizer, model, input_dir, optimizer_index)
            if not _accelerate_patch_warned_unsupported:
                warnings.warn("VideoX FSDP optimizer patch supports Accelerate 1.7.0 / PyTorch 2.7.0; "
                              "using native restore on this version", RuntimeWarning, stacklevel=2)
                _accelerate_patch_warned_unsupported = True
        return original(plugin, owner, optimizer, model, input_dir, optimizer_index, adapter_only)

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
            from .utils.checkpoint_utils import AcceleratorCheckpoint
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
        if getattr(owner.distributed_type, "name", None) == "FSDP":
            from .utils.checkpoint_utils import dispatch_checkpoint

            if owner.project_configuration.automatic_checkpoint_naming:
                output_dir = None
            return dispatch_checkpoint(owner, "save_state", native_save, output_dir,
                                       safe_serialization=safe_serialization, **kwargs)
        return native_save(owner, output_dir, safe_serialization=safe_serialization, **kwargs)

    @functools.wraps(native_load)
    def load_state(owner, input_dir=None, **kwargs):
        manager = owner.__dict__.get("_videox_checkpoint_manager")
        if manager is not None and manager.configured:
            return manager.load_state(input_dir, **kwargs)
        if getattr(owner.distributed_type, "name", None) == "FSDP":
            from .utils.checkpoint_utils import dispatch_checkpoint

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
    if not _accelerate_patch_requested() or not _supported_accelerate_versions():
        return
    _install_accelerate_dispatch()
    _accelerate_patches_automatic = True


def enable_accelerate_patch(accelerator):
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if getattr(accelerator.distributed_type, "name", None) != "FSDP" or not _eligible_fsdp_checkpoint(plugin):
        return accelerator
    if not _supported_accelerate_versions():
        raise RuntimeError("FSDP optimizer patch supports Accelerate 1.7.0 / PyTorch 2.7.0; "
                           "validate the adapter before enabling it on another version")
    _install_checkpoint_lifecycle()
    _install_accelerate_dispatch()
    _accelerate_patch_disabled_owners.discard(accelerator)
    _accelerate_patch_owners.add(accelerator)
    return accelerator


def disable_accelerate_patch(accelerator):
    _accelerate_patch_owners.discard(accelerator)
    _accelerate_patch_disabled_owners.add(accelerator)


install_accelerate_patches()
