r"""Cache Qwen3-VL conditioning for data-free PDD.

    /opt/pdd/bin/python scripts/minimax_h3_fun/encode_prompts.py \
        --model /root/models/MiniMax-H3/FL2VA \
        --prompts-json prompts_t2va_test_768p_val8.json \
        --output datasets/minimax_h3_pdd_prompt_cache \
        --split val
"""

import argparse
import json
import os
import sys

import torch

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration
from videox_fun.pipeline.pipeline_minimax_h3 import MINIMAX_H3_TEXT_ENCODER_LAYER, MINIMAX_H3_TEXT_TAG


def load_prompts(path):
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if isinstance(document, list) and all(isinstance(item, str) for item in document):
        return document
    examples = document.get("examples") if isinstance(document, dict) else document
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"{path} must be a prompt list or a jobs JSON with an `examples` list.")
    prompts = []
    for index, example in enumerate(examples, start=1):
        if isinstance(example, str):
            prompt = example.strip()
        elif isinstance(example, dict) and isinstance(example.get("prompt"), str):
            prompt = example["prompt"].strip()
        else:
            raise ValueError(f"Entry {index} in {path} is not a prompt string or a job with `prompt`.")
        if not prompt:
            raise ValueError(f"Entry {index} in {path} is empty.")
        prompts.append(prompt)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val", help="Subfolder under --output (train or val).")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_json)
    tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(args.model, "tokenizer"))
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(args.model, "text_encoder"), low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    ).eval()
    text_encoder.to("cuda")

    folder = os.path.join(args.output, args.split)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "prompts.json"), "w", encoding="utf-8") as handle:
        json.dump(prompts, handle, indent=1, ensure_ascii=False)
    for index, prompt in enumerate(prompts):
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            outputs = text_encoder.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                output_hidden_states=True,
            )
        torch.save(
            {
                "prompt": prompt,
                "prompt_embeds": outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to("cpu", torch.bfloat16),
                "text_token_tags": torch.full((len(token_ids),), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
            },
            os.path.join(folder, f"{index:04d}.pt"),
        )
        print(f"{args.split}/{index:04d}: {len(token_ids)} tokens", flush=True)


if __name__ == "__main__":
    main()
