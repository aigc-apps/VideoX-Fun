import importlib.util
import os

if importlib.util.find_spec("paifuser") is not None:
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
