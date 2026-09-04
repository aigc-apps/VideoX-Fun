# This file intentionally exists (empty) to make `midas` a regular package.
#
# Without it, `midas` is only a namespace-package portion, and Python's import
# machinery lets ANY regular package named `midas` found elsewhere on sys.path
# (e.g. comfyui_controlnet_aux's .../src/custom_controlnet_aux/midas/) win the
# resolution, even though torch.hub.load inserts this midas_repo dir at
# sys.path[0]. That collision produces:
#   ImportError: attempted relative import beyond top-level package
# See https://github.com/aigc-apps/VideoX-Fun/issues/502
