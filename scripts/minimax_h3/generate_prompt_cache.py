r"""Cache Qwen3-VL conditioning for data-free PDD (`fl2va`), multi-GPU, in safetensors.

Rewritten from the single-GPU `encode_prompts.py` to follow the repository's preprocessing convention
(mirrored on `scripts/wan2.1_self_forcing/generate_ode_pairs.py`):
`accelerate launch` over every rank, interleaved rank sharding, resume on files that already exist, per-entry
`.safetensors` (never `.pt` / LMDB), `wait_for_everyone`, then rank0 writes an `outputs.json` that
`ImageVideoSafetensorsDataset` consumes. Each entry holds the exact keys `FL2VATrajectory.reset` reads —
`prompt_embeds` (`hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]`) and `text_token_tags` — so the ~62 GB conditioner
stays out of the PDD training run under `--enable_preprocess_training`.

    accelerate launch --mixed_precision="bf16" scripts/minimax_h3/generate_prompt_cache.py \
        --pretrained_model_name_or_path=models/Diffusion_Transformer/MiniMax-H3 \
        --train_data_meta=datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json \
        --output_folder=datasets/minimax_h3_pdd_prompt_cache

`--train_data_meta` is the official demo dataset's annotation JSON (never an ad-hoc prompt set). PDD is
data-free, so only its `text` captions are read — the `file_path` / `audio_path` / `control_file_path` / `width` /
`height` fields the audio-visual dataset carries are ignored here. `load_prompts` also accepts a bare list of prompt
strings or `{"prompt": ...}` records. Data-free PDD has no train/val split: cache once and point both
`--train_data_meta` and `--val_data_meta` of `train_pdd_lora.py` at the same `outputs.json`.
"""

import argparse
import json
import math
import os
import sys

import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration

# Reuse the canonical MiniMax-H3 conditioner recipe rather than re-deriving it: `train_lora.encode_prompt` builds the
# presentation and reads `hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]` exactly as the pipeline does. For a text-only
# `fl2va` request it never touches `processor`, so `None` is passed.
from train_lora import encode_prompt


def load_prompts(path):
    r"""Read the annotation JSON into a list of prompt strings.

    Accepts the repository annotation format (a list of `{"text": ...}`), a bare list of strings, and the legacy
    `{"prompt": ...}` job records / `{"examples": [...]}` wrapper the old `encode_prompts.py` took.
    """
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if isinstance(document, dict):
        document = document.get("examples", document)
    if not isinstance(document, list) or not document:
        raise ValueError(f"{path} must be a non-empty list of prompts or `{{'text': ...}}` records.")
    prompts = []
    for index, entry in enumerate(document, start=1):
        if isinstance(entry, str):
            prompt = entry.strip()
        elif isinstance(entry, dict) and isinstance(entry.get("text"), str):
            prompt = entry["text"].strip()
        elif isinstance(entry, dict) and isinstance(entry.get("prompt"), str):
            prompt = entry["prompt"].strip()
        else:
            raise ValueError(f"Entry {index} in {path} is not a prompt string or a `{{'text'/'prompt': ...}}` record.")
        if not prompt:
            raise ValueError(f"Entry {index} in {path} is empty.")
        prompts.append(prompt)
    return prompts


def parse_args():
    parser = argparse.ArgumentParser(description="Cache Qwen3-VL `fl2va` conditioning for data-free PDD (multi-GPU).")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to the MiniMax-H3 partition; its `tokenizer/` and `text_encoder/` subfolders are read.",
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        required=True,
        help="Annotation JSON of prompts: a list of `{\"text\": ...}` records (bare strings / `{\"prompt\": ...}` also accepted).",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Directory the per-prompt `.safetensors` and the rank0 `outputs.json` are written to (one split).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision the conditioner runs at. MiniMax-H3 conditions in bfloat16. Default: bf16.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed preprocessing: local_rank.")
    return parser.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    prompts = load_prompts(args.train_data_meta)
    tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(args.pretrained_model_name_or_path, "tokenizer"))
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "text_encoder"), low_cpu_mem_usage=True, torch_dtype=weight_dtype,
    ).to(device).eval()
    text_encoder.requires_grad_(False)

    os.makedirs(args.output_folder, exist_ok=True)
    total_per_rank = int(math.ceil(len(prompts) / world_size))

    # Each rank walks an interleaved slice of the prompt list; a file that already exists is skipped so a killed run
    # resumes instead of recomputing.
    for index in tqdm(range(total_per_rank), disable=rank != 0, desc="Caching fl2va prompts"):
        prompt_index = index * world_size + rank
        if prompt_index >= len(prompts):
            continue
        prompt = prompts[prompt_index]
        output_path = os.path.join(args.output_folder, f"{prompt_index:05d}.safetensors")
        if os.path.exists(output_path):
            continue

        prompt_embeds, text_token_tags = encode_prompt(
            text_encoder, tokenizer, None, prompt, device=device, dtype=weight_dtype,
        )
        save_file(
            {
                "prompt_embeds": prompt_embeds.to(torch.bfloat16).cpu().contiguous(),
                "text_token_tags": text_token_tags.cpu().contiguous().long(),
            },
            output_path,
            metadata={"format": "pt", "prompt": prompt},
        )

    accelerator.wait_for_everyone()

    # rank0 lists every generated file so `ImageVideoSafetensorsDataset` can load the split.
    if accelerator.is_main_process:
        records = []
        for prompt_index in range(len(prompts)):
            safetensor_path = os.path.join(args.output_folder, f"{prompt_index:05d}.safetensors")
            if os.path.exists(safetensor_path):
                records.append({"file_path": safetensor_path})
        json_path = os.path.join(args.output_folder, "outputs.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=4)
        print(f"Done. Cached {len(records)} fl2va prompts to {args.output_folder}")
        print(f"Annotation JSON: {json_path}")


if __name__ == "__main__":
    main()
