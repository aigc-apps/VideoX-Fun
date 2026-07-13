import importlib.util

# The pai_fuser is an internally developed acceleration package, which can be used on PAI.
# Importing paifuser registers post-import hooks that automatically patch the
# videox_fun submodules (models / dist / utils) as they are imported, so no
# per-submodule patching code is required (see paifuser/patch/videox_fun.py).
if importlib.util.find_spec("paifuser") is not None:
    import paifuser
