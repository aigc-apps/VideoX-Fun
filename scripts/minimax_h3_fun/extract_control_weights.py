# Extract the control branch of a trained MiniMaxH3ControlTransformer3DModel checkpoint.
#
# `train_control.py` saves the whole transformer (main branch + control branch) in the diffusers layout
# (`<checkpoint>/transformer/diffusion_pytorch_model.safetensors` plus `config.json`). The control branch is
# everything that `MiniMaxH3ControlTransformer3DModel` adds on top of the base model: the `control_blocks.*`
# list (one block per `control_blocks_places` entry) and the `control_proj_in.*` patch projection. This script
# writes just those tensors to a standalone safetensors file, which can be re-applied onto a fresh base model
# with `MiniMaxH3ControlTransformer3DModel.materialize_missing_control_params(...)`.
#
# Usage:
#   python scripts/minimax_h3_fun/extract_control_weights.py \
#       --model_path /path/to/train_control/checkpoint-xxx/transformer \
#       --output_path /path/to/control_weights.safetensors
import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file

CONTROL_PREFIXES = ("control_blocks.", "control_proj_in.")
# FSDP / DeepSpeed unwrap may leave wrapper prefixes on the keys; strip them to the bare model namespace.
WRAPPER_PREFIXES = ("_fsdp_wrapped_module.", "_fsdp_wrapped_module_", "module.", "_orig_mod.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract the control-branch weights (control_blocks / control_proj_in) of a trained "
        "MiniMax-H3 control transformer into a standalone safetensors file."
    )
    parser.add_argument(
        "--model_path", type=str, default="output_dir_minimax_h3_control_distill/checkpoint-4000/transformer/diffusion_pytorch_model.safetensors",
        help="Path to the saved transformer: a directory containing diffusion_pytorch_model*.safetensors "
        "(e.g. `<checkpoint>/transformer`), or a single .safetensors file.",
    )
    parser.add_argument(
        "--output_path", type=str, default="output_dir_minimax_h3_control_distill/checkpoint-4000/transformer/diffusion_pytorch_model_control.safetensors",
        help="Where to write the extracted control weights.",
    )
    return parser.parse_args()


def resolve_safetensor_files(model_path):
    if os.path.isdir(model_path):
        shards = sorted(
            os.path.join(model_path, name)
            for name in os.listdir(model_path)
            if name.endswith(".safetensors")
        )
        if not shards:
            raise FileNotFoundError(f"No .safetensors files found under {model_path}.")
        return shards
    if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
        return [model_path]
    raise FileNotFoundError(f"--model_path must be a safetensors file or a directory of them, got {model_path}.")


def unwrap_key(key):
    changed = True
    while changed:
        changed = False
        for prefix in WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def main():
    args = parse_args()

    state_dict = {}
    for shard in resolve_safetensor_files(args.model_path):
        state_dict.update(load_file(shard, device="cpu"))

    control_state_dict = {}
    for key, value in state_dict.items():
        bare_key = unwrap_key(key)
        if bare_key.startswith(CONTROL_PREFIXES):
            control_state_dict[bare_key] = value.contiguous()

    if not control_state_dict:
        raise ValueError(
            f"No control-branch keys (control_blocks.* / control_proj_in.*) found in {args.model_path}; "
            "this checkpoint does not look like a MiniMax-H3 control training output."
        )

    # Carry the branch layout next to the weights so a loader can rebuild the same control model without
    # inspecting the full training config. `config.json` of the saved transformer records both fields; keep
    # the safetensors metadata strings-only.
    metadata = {"format": "pt"}
    config_path = os.path.join(args.model_path, "config.json") if os.path.isdir(args.model_path) else None
    if config_path is not None and os.path.isfile(config_path):
        with open(config_path, "r") as file:
            config = json.load(file)
        for field in ("control_blocks_places", "control_in_dim"):
            if field in config:
                metadata[field] = json.dumps(config[field])

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    save_file(control_state_dict, args.output_path, metadata=metadata)

    num_params = sum(value.numel() for value in control_state_dict.values())
    block_ids = sorted({
        int(key.split(".")[1]) for key in control_state_dict if key.startswith("control_blocks.")
    })
    print(f"Extracted {len(control_state_dict)} control tensors ({num_params / 1e9:.3f}B params) -> {args.output_path}")
    print(f"  control_blocks indices: {block_ids}")
    if "control_in_dim" in metadata:
        print(f"  control_in_dim: {metadata['control_in_dim']}, control_blocks_places: {metadata['control_blocks_places']}")
    for key in sorted(control_state_dict):
        if key.endswith(".weight") and control_state_dict[key].dim() >= 2:
            print(f"  {key}: {list(control_state_dict[key].shape)} {control_state_dict[key].dtype}")


if __name__ == "__main__":
    main()
