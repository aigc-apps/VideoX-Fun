r"""Cache `ref2va` request conditioning for data-free PDD, multi-GPU, in safetensors.

The `ref2va` counterpart of `generate_prompt_cache.py`: a `ref2va` request conditions on reference media, so
besides the Qwen3-VL `prompt_embeds` / `text_token_tags` it also needs the VAE-encoded reference latents
`Ref2VATrajectory.reset` lays in front of the generated rows. This cache is the *optional* pre-encode route
(README_TRAIN_PDD_LORA.md §3.2.1 Route B): `train_pdd_lora.py --train_mode=ref2va` can instead encode those
latents on the fly from a request annotation (Route A, launched without `--enable_preprocess_training`), but
pre-encoding once keeps the ~62 GB conditioner and both VAEs out of long or repeated training runs.

Follows the same preprocessing convention as `generate_prompt_cache.py` (mirrored on
`scripts/wan2.1_self_forcing/generate_ode_pairs.py`): `accelerate launch`, interleaved rank
sharding, resume on existing files, per-request `.safetensors`, `wait_for_everyone`, then rank0 `outputs.json`.

    accelerate launch --mixed_precision="bf16" scripts/minimax_h3/generate_ref2va_request_cache.py \
        --pretrained_model_name_or_path=models/Diffusion_Transformer/MiniMax-H3 \
        --train_data_meta=datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json \
        --output_folder=datasets/minimax_h3_pdd_request_cache \
        --transformer_subfolder=transformer_ref \
        --video_sample_n_frames=124

`--train_data_meta` is a JSON list of requests. Either an explicit `{"prompt": ..., "references": ["image=...",
"video=...", "audio=..."]}` record (the `predict_ref2va.py` schema, references in the order the model reads them), or
the official audio-visual demo record `{"text", "file_path", "audio_path"}` (never an ad-hoc request set),
whose own video + audio become the `video=`/`audio=` references — see `load_requests`.
"""

import argparse
import json
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

from videox_fun.models import (AutoencoderKLMiniMaxH3,
                               AutoencoderKLMiniMaxH3Audio,
                               MiniMaxH3Transformer3DModel, Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor)
from videox_fun.pipeline.pipeline_minimax_h3 import (
    MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference,
    align_num_frames, check_ref2va_references, normalize_ref2va_references)

# Reuse the canonical recipes rather than re-deriving them: `encode_prompt(references=...)` builds the ref2va
# presentation and `encode_reference_latents_for_training` mirrors `MiniMaxH3Pipeline.encode_reference_latents`
# without needing a pipeline instance.
from train_lora import encode_prompt, encode_reference_latents_for_training

_REFERENCE_KIND_IDS = {"image": 0, "video": 1, "audio": 2}


def parse_reference(entry: str):
    r"""Decode one `image=path` / `video=path` / `audio=path` entry into its `MiniMaxH3Reference` (the
    `predict_ref2va.py` schema)."""
    kind, _, media = entry.partition("=")
    kind, media = kind.strip().lower(), media.strip()
    if not media:
        raise ValueError(f"A reference entry must be `image=path`, `video=path` or `audio=path`, got {entry!r}.")
    if kind == "image":
        return MiniMaxH3ImageReference.from_file(media)
    if kind == "video":
        return MiniMaxH3VideoReference.from_file(media)
    if kind == "audio":
        return MiniMaxH3AudioReference.from_file(media)
    raise ValueError(f"A reference entry must start with `image=`, `video=` or `audio=`, got {entry!r}.")


def load_requests(path):
    r"""Read the request annotation JSON into a list of `{"prompt": str, "references": [str, ...]}`.

    Two record shapes are accepted. An explicit `ref2va` request carries its own `references` (the `predict_ref2va.py`
    schema). The official audio-visual demo record (`{"text", "file_path", "audio_path"}`) has none, so its
    own video + audio become the references — `video=<file_path>` and `audio=<audio_path>` — which lets `ref2va` be
    driven straight from `datasets/X-Fun-Videos-Audios-Demo` exactly as the `fl2va` prompt cache is. A demo media path
    is relative to the dataset, so it resolves against the annotation file's own directory.
    """
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if isinstance(document, dict):
        document = document.get("examples", document)
    if not isinstance(document, list) or not document:
        raise ValueError(f"{path} must be a non-empty list of ref2va requests.")
    root = os.path.dirname(os.path.abspath(path))
    requests = []
    for index, entry in enumerate(document, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {index} in {path} is not a request record.")
        prompt = entry.get("prompt", entry.get("text"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Entry {index} in {path} needs a non-empty `prompt`/`text`.")
        references = entry.get("references")
        if references is None:
            # Official audio-visual demo record: derive the references from its own video + audio (an audio reference
            # cannot stand alone, and every demo entry ships both), so no ad-hoc request file is needed.
            references = []
            for key, kind in (("file_path", "video"), ("audio_path", "audio")):
                media = entry.get(key)
                if isinstance(media, str) and media.strip():
                    media = media.strip()
                    references.append(f"{kind}={media if os.path.isabs(media) else os.path.join(root, media)}")
        if not isinstance(references, list) or not references or not all(isinstance(r, str) for r in references):
            raise ValueError(f"Entry {index} in {path} needs a non-empty `references` list of strings.")
        requests.append({"prompt": prompt.strip(), "references": list(references)})
    return requests


def parse_args():
    parser = argparse.ArgumentParser(description="Cache `ref2va` request conditioning for data-free PDD (multi-GPU).")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to the MiniMax-H3 `ref2va` partition; its `tokenizer/`, `processor/`, `text_encoder/`, `vae/`, "
             "`audio_vae/` and `transformer_ref/` subfolders are read.",
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        required=True,
        help="Annotation JSON of requests: a list of `{\"prompt\": ..., \"references\": [\"image=...\", ...]}`.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Directory the per-request `.safetensors` and the rank0 `outputs.json` are written to (one split).",
    )
    parser.add_argument(
        "--transformer_subfolder",
        type=str,
        default="transformer_ref",
        help="Subfolder the `patch_size` / `audio_in_channels` are read from (config only, no weights). Default: transformer_ref.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=124,
        help="Generated frame count (form 17 * n + 5) the references are normalized onto. Default: 124.",
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

    requests = load_requests(args.train_data_meta)
    num_frames = align_num_frames(int(args.video_sample_n_frames))

    # `patch_size` / `audio_in_channels` / `sampling_rate` are read from config files alone — neither the 33 B
    # transformer nor the audio VAE weights are loaded just to read them.
    transformer_config = MiniMaxH3Transformer3DModel.load_config(
        args.pretrained_model_name_or_path, subfolder=args.transformer_subfolder
    )
    patch_size = tuple(transformer_config["patch_size"])
    audio_channels = transformer_config["audio_in_channels"]
    audio_sr = AutoencoderKLMiniMaxH3Audio.load_config(
        args.pretrained_model_name_or_path, subfolder="audio_vae"
    ).get("sampling_rate", 32000)

    os.makedirs(args.output_folder, exist_ok=True)

    # Interleaved rank sharding with resume: `pending` holds this rank's not-yet-cached request indices. Both passes
    # walk the same list, so every rank stays independent and no collective depends on the request count dividing
    # evenly across ranks — which is why this generator does not shard the conditioner with FSDP the way inference does.
    pending = [
        request_index
        for request_index in range(rank, len(requests), world_size)
        if not os.path.exists(os.path.join(args.output_folder, f"{request_index:05d}.safetensors"))
    ]

    def normalized_references(request):
        # Deterministic, so pass 2 reproduces exactly the references pass 1 encoded the prompt against.
        references = [parse_reference(entry) for entry in request["references"]]
        references = check_ref2va_references(references)
        return normalize_ref2va_references(references, num_frames, audio_sr)

    # ---- Pass 1/2: the ~62 GB Qwen3-VL conditioner is resident; encode every prompt, then release it. A 124-frame
    # reference video does not VAE-encode alongside the conditioner within 80 GB, so the two big models never overlap.
    tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(args.pretrained_model_name_or_path, "tokenizer"))
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(args.pretrained_model_name_or_path, "processor"))
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "text_encoder"), low_cpu_mem_usage=True, torch_dtype=weight_dtype,
    ).to(device).eval()
    text_encoder.requires_grad_(False)

    prompt_cache = {}
    for request_index in tqdm(pending, disable=rank != 0, desc="Caching ref2va prompts (pass 1/2)"):
        request = requests[request_index]
        references = normalized_references(request)
        prompt_embeds, text_token_tags = encode_prompt(
            text_encoder, tokenizer, processor, request["prompt"], references=references, device=device, dtype=weight_dtype,
        )
        prompt_cache[request_index] = (
            prompt_embeds.to(torch.bfloat16).cpu().contiguous(),
            text_token_tags.cpu().contiguous().long(),
        )
        del references, prompt_embeds, text_token_tags

    del text_encoder, tokenizer, processor
    torch.cuda.empty_cache()

    # ---- Pass 2/2: only the two VAEs are resident; encode the reference latents and write each request. The VAEs stay
    # float32 as released (the encode recipe is float16 autocast over float32 weights).
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", low_cpu_mem_usage=True,
    ).to(device).eval()
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="audio_vae", low_cpu_mem_usage=True,
    ).to(device).eval()
    vae.requires_grad_(False)
    audio_vae.requires_grad_(False)

    for request_index in tqdm(pending, disable=rank != 0, desc="Caching ref2va latents (pass 2/2)"):
        request = requests[request_index]
        references = normalized_references(request)
        condition_latents, audio_condition_latents = encode_reference_latents_for_training(
            vae, audio_vae, references, patch_size, device, audio_latent_channels=audio_channels,
        )
        reference_kinds = [(reference.kind, bool(reference.has_audio)) for reference in references]
        prompt_embeds, text_token_tags = prompt_cache.pop(request_index)

        tensors = {
            "prompt_embeds": prompt_embeds,
            "text_token_tags": text_token_tags,
            # safetensors holds tensors only, so the ragged reference structure is flattened: the kind / has-audio
            # pairs become two int vectors and the per-reference latents become indexed tensors under a count.
            "reference_kind_ids": torch.tensor([_REFERENCE_KIND_IDS[kind] for kind, _ in reference_kinds], dtype=torch.long),
            "reference_has_audio": torch.tensor([int(has_audio) for _, has_audio in reference_kinds], dtype=torch.long),
            "num_condition_latents": torch.tensor(len(condition_latents), dtype=torch.long),
            "num_audio_condition_latents": torch.tensor(len(audio_condition_latents), dtype=torch.long),
        }
        for position, latent in enumerate(condition_latents):
            tensors[f"condition_latents_{position}"] = latent.cpu().contiguous().float()
        for position, latent in enumerate(audio_condition_latents):
            tensors[f"audio_condition_latents_{position}"] = latent.cpu().contiguous().float()
        save_file(
            tensors,
            os.path.join(args.output_folder, f"{request_index:05d}.safetensors"),
            metadata={"format": "pt", "prompt": request["prompt"], "references": json.dumps(request["references"])},
        )
        del references, condition_latents, audio_condition_latents

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        records = []
        for request_index in range(len(requests)):
            safetensor_path = os.path.join(args.output_folder, f"{request_index:05d}.safetensors")
            if os.path.exists(safetensor_path):
                records.append({"file_path": safetensor_path})
        json_path = os.path.join(args.output_folder, "outputs.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=4)
        print(f"Done. Cached {len(records)} ref2va requests to {args.output_folder}")
        print(f"Annotation JSON: {json_path}")


if __name__ == "__main__":
    main()
