import importlib.util

if importlib.util.find_spec("paifuser") is not None:
    import paifuser
