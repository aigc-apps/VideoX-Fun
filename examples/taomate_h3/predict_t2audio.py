import json
import os
import sys

import numpy as np
import torch
import torchaudio

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (AutoencoderKLMiniMaxH3,
                               AutoencoderKLMiniMaxH3Audio,
                               MiniMaxH3Transformer3DModel, Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor)
from videox_fun.pipeline import MiniMaxH3Pipeline
from videox_fun.pipeline.pipeline_minimax_h3 import (MINIMAX_H3_AUDIO_TAG,
                                                     MINIMAX_H3_TEXT_TAG,
                                                     _spatial_position_grid)
from videox_fun.pipeline.pipeline_taomate_h3 import (
    TAOMATE_H3_AUDIO_LATENT_CHANNELS, TAOMATE_H3_AUDIO_SIGMA_SHIFT,
    TAOMATE_H3_DISTILLED_STATE_INDICES, TAOMATE_H3_REQUEST_AUDIO_LATENTS,
    TAOMATE_H3_REQUEST_VIDEO_LATENTS, TAOMATE_H3_ROLLOVER_REFERENCE_LATENTS,
    TAOMATE_H3_SUPPORTED_SHORT_EDGES, TAOMATE_H3_TEACHER_STATE_NUMBERS,
    TAOMATE_H3_VIDEO_SIGMA_SHIFT, taomate_h3_canonical_continuation_plan,
    taomate_h3_direct_5s_plan, taomate_h3_select_time_shift_sigmas,
    taomate_h3_teacher_geometry)
from videox_fun.utils import (MiniMaxH3Scheduler, register_auto_device_hook,
                              safe_enable_group_offload)

# GPU memory mode, which can be chosen in [model_full_load, model_cpu_offload, model_group_offload, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_group_offload transfers internal layer groups between CPU/CUDA, 
# balancing memory efficiency and speed between full-module and leaf-level offloading methods.
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
# The transformer alone is 61.7 GB in bfloat16 and the Qwen3-VL conditioner another 62.1 GB, so a single 80 GB
# card needs an offload mode. The float8-quantizing modes are deliberately absent: the audio path is the Base10
# teacher's, running the *base* weights exactly as released (no LoRA — the TaoMate-H3 adapter only steers the
# video), and the artifact records `base_precision=bf16`.
GPU_memory_mode     = "model_cpu_offload"
# Multi GPUs config. The audio-only loop drives the transformer directly and runs one request's whole packed
# sequence on one GPU, so keep ulysses_degree = ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with sequential_cpu_offload.
compile_dit         = False

# model path
model_name          = "models/Diffusion_Transformer/MiniMax-H3"

# Other params
# The canvas: the short edge must be 480, 768 or 1088 and both edges 32-aligned (480x864 is the resolution
# the official TaoMate-H3 demo ships). The teacher artifact is bound to this geometry, and the audio noise
# identity depends on it too (the discarded video-noise draw is canvas-shaped).
sample_size         = [864, 480]
# How many 5-second requests to generate: the first one runs the direct 124-frame plan, every following one
# the canonical continuation spliced behind its predecessor — it denoises the previous request's clean tail
# (`TAOMATE_H3_ROLLOVER_REFERENCE_LATENTS` latents per channel, read-only) together with its own fresh noise.
request_count       = 2

# Use torch.float16 if GPU does not support torch.bfloat16
# Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
# The prompt the default Base10 teacher artifact (`samples/taomate_h3_teacher/00000000`) was generated for —
# `Self-Forcing/prompts/self_forcing_all_prompts.json` entry 0. A list gives one prompt per stream request
# (the official `--prompt-json` shape) and the artifact then carries exactly those prompts; a string covers
# the whole timeline.
prompt              = (
    "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city "
    "signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red "
    "dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a "
    "bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a "
    "mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the "
    "lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to "
    "one side, highlighting her graceful strides."
)
# The authored seed. Request j draws its audio noise from seed + j — the streaming runtime replays the same
# sequence from the artifact's `audio_noise_seed_sequence`.
seed                = 43
# The offline Base10 audio-teacher artifact directory to write: `predict_t2av_streaming.py` reads this
# very value back through its own `audio_teacher_dir`, so keep the two identical. The artifact
# (`complete.json` + `request_XX.pt`) holds the clean audio rows after denoising steps 3, 6 and 9 and
# is bound to the prompt(s), seed and canvas above. Set to None to only save the wav.
audio_teacher_dir   = "samples/taomate_h3_teacher/00000000"
save_path           = "samples/taomate-h3-audios-t2a"

# `sample_size` must fit the streaming canvas contract, and the artifact is only ever `base_precision=bf16`.
if min(sample_size) not in TAOMATE_H3_SUPPORTED_SHORT_EDGES or sample_size[0] % 32 or sample_size[1] % 32:
    raise ValueError(
        f"`sample_size` {sample_size} must use a 480-, 768- or 1088-pixel short edge and be 32-pixel "
        "aligned, matching the streaming canvas contract (e.g. [864, 480])."
    )
if request_count < 1:
    raise ValueError(f"`request_count` must be positive, got {request_count}.")
if audio_teacher_dir is not None and weight_dtype != torch.bfloat16:
    raise ValueError(
        "the Base10 teacher artifact records `base_precision=bf16`; set `audio_teacher_dir = None` to "
        "only save the wav, or run with `weight_dtype = torch.bfloat16`."
    )

device = set_multi_gpus_devices(ulysses_degree, ring_degree)

# `model_name` may point either at a converted diffusers layout or at an *original* MiniMax-H3 partition (e.g.
# `MiniMax-H3/FL2VA`); the original shards are converted on the fly while loading, no intermediate copy on disk.
# Transformer
transformer = MiniMaxH3Transformer3DModel.from_pretrained(
    model_name,
    subfolder="transformer",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

# Video VAE. The released weights are float32 and the decode runs under float16 autocast, so the VAE is not
# downcast even when the rest of the pipeline is bfloat16 (this is also how the training scripts load it).
# The audio-only path never decodes video — the container holds it only to satisfy `MiniMaxH3Pipeline`.
vae = AutoencoderKLMiniMaxH3.from_pretrained(
    model_name,
    subfolder="vae",
    low_cpu_mem_usage=True,
)

# Audio VAE, waveform in / waveform out: MiniMax-H3 has no separate vocoder. Float32 as released, like the video VAE.
audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
    model_name,
    subfolder="audio_vae",
    low_cpu_mem_usage=True,
)

# Get Tokenizer and Processor
tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(model_name, "tokenizer"))
processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_name, "processor"))

# Get Text encoder. MiniMax-H3 reads the unnormalized hidden state after the 50th decoder layer of Qwen3-VL.
text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    os.path.join(model_name, "text_encoder"),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Schedulers. The 10-step Base schedule is rebuilt from the checkpoint's own sigma shifts
# (`taomate_h3_select_time_shift_sigmas`), so the checkpoint schedules only seed the class.
scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="scheduler")
audio_scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="audio_scheduler")

pipeline = MiniMaxH3Pipeline(
    vae=vae,
    audio_vae=audio_vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    processor=processor,
    transformer=transformer,
    scheduler=scheduler,
    audio_scheduler=audio_scheduler,
)

if compile_dit:
    for i in range(len(pipeline.transformer.transformer_blocks)):
        pipeline.transformer.transformer_blocks[i] = torch.compile(pipeline.transformer.transformer_blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_group_offload":
    register_auto_device_hook(pipeline.transformer)
    safe_enable_group_offload(pipeline, onload_device=device, offload_device="cpu", offload_type="leaf_level", use_stream=True)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load":
    pipeline.to(device=device)
else:
    raise ValueError(
        f"`GPU_memory_mode` must be one of ['model_full_load', 'model_cpu_offload', 'model_group_offload', "
        f"'sequential_cpu_offload'], got {GPU_memory_mode}."
    )


def official_audio_noise(*, video_latent_t, video_latent_h, video_latent_w, audio_latent_t, seed):
    """The exact initial audio noise of one request, the offline teacher's arithmetic.

    H3 draws full-AV video noise before audio. The audio-only path discards those values, but
    advancing this exact CPU generator is part of the audio identity, and the streaming student
    reuses the very same rows as its initial audio noise. `24` is MiniMax-H3's video VAE latent
    channel count; the canvas enters the identity through this draw's shape.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.randn(
        1, 24, video_latent_t, video_latent_h, video_latent_w,
        generator=generator, dtype=torch.float32, device="cpu",
    )
    return torch.randn(
        2 * audio_latent_t, TAOMATE_H3_AUDIO_LATENT_CHANNELS,
        generator=generator, dtype=torch.float32, device="cpu",
    )


def audio_only_packed_layout(
    text_len, ref_audio_t, audio_t, latent_h, latent_w, *, reference_time_start, target_time_start
):
    """The audio-only packed sequence `[text | (reference audio |) target audio]`, the offline teacher's
    arithmetic.

    Mirrors the official `minimax_h3_audio_only_packed_sequence` and
    `minimax_h3_audio_only_frozen_prefix_packed_sequence` builders, minus the 64-row attention
    padding (a single-GPU sequence needs no alignment). Rows order their time axis per channel
    block — `[ch0 rows; ch1 rows]` — matching the audio-row storage order.
    """
    patch = 2
    ref_rows = ref_audio_t * 2
    target_rows = audio_t * 2
    total_rows = ref_rows + target_rows
    sequence_length = text_len + total_rows

    sqrt_area = np.sqrt(latent_h * latent_w)
    width_grid = _spatial_position_grid(latent_w, patch, sqrt_area)

    grid = torch.zeros(sequence_length, 3, dtype=torch.float64)
    grid[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
    row_index = text_len
    for temporal_rows, time_start in ((ref_audio_t, reference_time_start), (audio_t, target_time_start)):
        if temporal_rows <= 0:
            continue
        times = (float(time_start) + torch.arange(temporal_rows, dtype=torch.float64)).repeat(2)
        grid[row_index : row_index + 2 * temporal_rows, 0] = times
        grid[row_index : row_index + temporal_rows, 2] = float(width_grid[0])
        grid[row_index + temporal_rows : row_index + 2 * temporal_rows, 2] = float(width_grid[-1])
        row_index += 2 * temporal_rows

    token_tags = torch.full((sequence_length,), -1, dtype=torch.long)
    token_tags[:text_len] = MINIMAX_H3_TEXT_TAG
    token_tags[text_len:] = MINIMAX_H3_AUDIO_TAG

    return {
        "sequence_length": sequence_length,
        "position_ids": grid,
        "token_tags": token_tags,
        "text_indices": torch.arange(text_len),
        "audio_indices": torch.arange(text_len, sequence_length),
        "ref_rows": ref_rows,
        "target_rows": target_rows,
    }


def audio_only_step_timesteps(t_video, t_audio, *, has_reference):
    """The `(timestep, timestep_indices)` pair of one audio-only forward, the offline teacher's arithmetic.

    Text rows ride the video clock, target audio rows the audio clock, and the frozen reference
    rows stay pinned at `1.0` — the timestep of a clean row (`t = 1 - sigma` with `sigma = 0`),
    exactly as the reference audio in the student's layouts.
    """
    candidates = [float(t_video), float(t_audio)]
    if has_reference:
        candidates.append(1.0)
    unique_timesteps, slot_to_unique = torch.unique(
        torch.tensor(candidates, dtype=torch.float32), sorted=True, return_inverse=True
    )
    text_slot = int(slot_to_unique[0])
    target_slot = int(slot_to_unique[1])
    reference_slot = int(slot_to_unique[2]) if has_reference else None

    def expand(text_rows, ref_rows, total_rows):
        indices = torch.empty(total_rows, dtype=torch.long)
        indices[:text_rows] = text_slot
        if ref_rows:
            indices[text_rows : text_rows + ref_rows] = reference_slot
        indices[text_rows + ref_rows :] = target_slot
        return indices

    return unique_timesteps, expand


_OFFLOAD_MODES = ("model_cpu_offload", "model_group_offload", "sequential_cpu_offload")


def _park_on_cpu(module):
    """Send a component back to the CPU under an offload mode, so the next component fits.

    `model_cpu_offload` only orchestrates component transfers inside `pipeline.__call__`; this
    script drives the components directly, so each pass parks what it just used. Under
    `model_full_load` nothing is parked.
    """
    if GPU_memory_mode not in _OFFLOAD_MODES:
        return
    if next(module.parameters()).device.type != "cpu":
        module.to("cpu")
        torch.cuda.empty_cache()


def generate_audio_track(pipeline, *, prompt, seed, request_count, height, width, device):
    """Denoise `request_count` 5-second audio-only requests and return their clean packed rows.

    Returns `(clean_rows, request_records)`: `clean_rows` is `(2 * total_audio_latents,
    audio_latent_channels)` float32 on the CPU — the exact rows `MiniMaxH3StreamingPipeline`
    publishes (it hard-asserts they equal the Base10 teacher's final clean state), before that
    pipeline's one-shot VAE decode — and `request_records` carries one record per request with its
    captured 3/6/9 milestones plus the metadata `write_teacher_artifact` needs.
    """
    if isinstance(prompt, str):
        prompts = [prompt] * request_count
    else:
        prompts = list(prompt)
        if len(prompts) != request_count:
            raise ValueError(
                f"`prompt` lists {len(prompts)} entries but `request_count` is {request_count}: a list gives "
                "one prompt per stream request."
            )

    latent_height = height // pipeline.vae_spatial_compression_ratio
    latent_width = width // pipeline.vae_spatial_compression_ratio
    video_row_width = int(pipeline.vae_latent_channels * pipeline.patch_size[1] * pipeline.patch_size[2])

    # The full Base schedule: ten steps at the checkpoint's sigma shifts, no distilled state
    # subsampling. The forward clock rides the Python-float timesteps while the update rides the
    # tensorized float32 sigma_t / ratio, mirroring the exact arithmetic chain of the reference
    # denoise loop bit for bit.
    video_sigmas = taomate_h3_select_time_shift_sigmas(shift_scale=TAOMATE_H3_VIDEO_SIGMA_SHIFT, num_steps=10)
    audio_sigmas = taomate_h3_select_time_shift_sigmas(shift_scale=TAOMATE_H3_AUDIO_SIGMA_SHIFT, num_steps=10)
    video_timesteps = [1.0 - sigma for sigma in video_sigmas[:-1]]
    audio_timesteps = [1.0 - sigma for sigma in audio_sigmas[:-1]]
    audio_sigmas_tensor = torch.tensor(audio_sigmas, dtype=torch.float32)
    audio_sigma_t = 1.0 - torch.tensor(audio_timesteps, dtype=torch.float32)
    audio_sigma_ratios = audio_sigmas_tensor[1:] / audio_sigmas_tensor[:-1]
    audio_one_minus_ratios = 1.0 - audio_sigma_ratios
    base_plan = taomate_h3_direct_5s_plan()

    # The empty video stream: the audio-only path never executes the video projection or head.
    empty_video_rows = torch.zeros((0, video_row_width), dtype=torch.float32, device=device)

    segments = []
    request_records = []
    prompt_cache = {}
    previous_clean = None
    previous_audio_latent_count = None
    for request_index in range(request_count):
        prompt_text = prompts[request_index]
        if prompt_text not in prompt_cache:
            # The transformer's 9 forwards keep it on the GPU under an offload mode; park it before
            # the conditioner comes in for its own pass (the teacher's explicit offload order).
            _park_on_cpu(pipeline.transformer)
            with torch.no_grad():
                prompt_cache[prompt_text] = pipeline.encode_prompt(
                    prompt_text, device=device, dtype=pipeline.transformer.dtype
                )
            _park_on_cpu(pipeline.text_encoder)
        prompt_embeds, text_token_tags = prompt_cache[prompt_text]
        text_len = int(text_token_tags.shape[0])

        active_plan = (
            base_plan
            if request_index == 0
            else taomate_h3_canonical_continuation_plan(base_plan, request_index=request_index)
        )
        active_audio_latents = active_plan.phases[-1].audio_latent_stop
        transport_prefix = TAOMATE_H3_REQUEST_AUDIO_LATENTS - active_audio_latents

        official_audio = official_audio_noise(
            video_latent_t=TAOMATE_H3_REQUEST_VIDEO_LATENTS,
            video_latent_h=latent_height,
            video_latent_w=latent_width,
            audio_latent_t=TAOMATE_H3_REQUEST_AUDIO_LATENTS,
            seed=seed + request_index,
        )
        # A continuation slices its own fresh noise down to the steady geometry; the published
        # prefix of the request is the previous request's tail.
        target_noise = (
            official_audio.view(2, TAOMATE_H3_REQUEST_AUDIO_LATENTS, -1)[:, transport_prefix:]
            .contiguous()
            .view(-1, TAOMATE_H3_AUDIO_LATENT_CHANNELS)
        )
        reference_tail = (
            None
            if previous_clean is None
            else previous_clean.view(2, previous_audio_latent_count, -1)[
                :, -TAOMATE_H3_ROLLOVER_REFERENCE_LATENTS:
            ]
            .contiguous()
            .view(2 * TAOMATE_H3_ROLLOVER_REFERENCE_LATENTS, TAOMATE_H3_AUDIO_LATENT_CHANNELS)
        )
        initial_audio = target_noise if reference_tail is None else torch.cat((reference_tail, target_noise), dim=0)
        ref_audio_t = 0 if reference_tail is None else TAOMATE_H3_ROLLOVER_REFERENCE_LATENTS
        ref_rows = ref_audio_t * 2

        layout = audio_only_packed_layout(
            text_len,
            ref_audio_t,
            active_audio_latents,
            latent_height,
            latent_width,
            reference_time_start=text_len + previous_audio_latent_count - TAOMATE_H3_ROLLOVER_REFERENCE_LATENTS
            if reference_tail is not None
            else 0,
            target_time_start=text_len + (previous_audio_latent_count or 0),
        )
        position_ids = layout["position_ids"].to(device)
        token_tags = layout["token_tags"].to(device)
        text_indices = layout["text_indices"].to(device)
        audio_indices = layout["audio_indices"].to(device)

        audio_rows = initial_audio.to(device=device, dtype=torch.float32)
        captured = {}

        def run_forward(audio_rows, t_video, t_audio, has_reference):
            unique_timesteps, expand = audio_only_step_timesteps(t_video, t_audio, has_reference=has_reference)
            timestep_indices = expand(text_len, ref_rows, int(layout["sequence_length"])).to(device)
            _, audio_velocity = pipeline.transformer(
                hidden_states=empty_video_rows[None],
                audio_hidden_states=audio_rows[None],
                encoder_hidden_states=prompt_embeds,
                timestep=unique_timesteps.to(device),
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                position_ids=position_ids,
                video_indices=torch.empty(0, dtype=torch.long, device=device),
                audio_indices=audio_indices,
                text_indices=text_indices,
                return_dict=False,
            )
            return unique_timesteps, audio_velocity[0].float()

        with torch.no_grad():
            for step in range(len(audio_timesteps)):
                _, audio_velocity = run_forward(
                    audio_rows,
                    video_timesteps[step],
                    audio_timesteps[step],
                    has_reference=ref_rows > 0,
                )
                # Euler over the target rows only; the reference rows stay clean.
                target = audio_rows[ref_rows:]
                sigma_t = float(audio_sigma_t[step])
                sigma_ratio = float(audio_sigma_ratios[step])
                one_minus_ratio = float(audio_one_minus_ratios[step])
                denoised = target + sigma_t * audio_velocity[ref_rows:]
                audio_rows = torch.cat(
                    (
                        audio_rows[:ref_rows],
                        sigma_ratio * target + one_minus_ratio * denoised,
                    ),
                    dim=0,
                )
                # The Base10 teacher contract: the clean audio rows after states 3, 6 and 9; state 9
                # is this request's final clean target.
                state_number = step + 1
                if state_number in TAOMATE_H3_TEACHER_STATE_NUMBERS:
                    captured[state_number] = (
                        audio_rows[ref_rows:].detach().to(device="cpu", dtype=torch.float32).contiguous()
                    )

        milestones = [captured[state_number] for state_number in TAOMATE_H3_TEACHER_STATE_NUMBERS]
        clean_target = milestones[-1]
        segments.append(clean_target)
        previous_clean = clean_target
        previous_audio_latent_count = active_audio_latents
        request_records.append(
            {
                "prompt": prompt_text,
                "audio_noise_seed": seed + request_index,
                "audio_latent_count": active_audio_latents,
                "transport_prefix": transport_prefix,
                "packed_text_rows": text_len,
                "packed_audio_rows": int(layout["audio_indices"].shape[0]),
                "reference_latents_per_channel": ref_audio_t,
                "milestones": milestones,
            }
        )
        print(
            f"[audio] request {request_index}: audio latents/channel={active_audio_latents}, "
            f"reference latents/channel={ref_audio_t}",
            flush=True,
        )

    return torch.cat(segments, dim=0), request_records


def write_teacher_artifact(output_dir, *, pipeline, request_records, request_count, seed, height, width):
    """Write the offline Base10 teacher artifact the streaming runtime reads back.

    The directory contract `TaomateH3TeacherArtifact.open` validates: `complete.json` plus one
    `request_XX.pt` per stream request, holding the request's `prompt` / `seed` /
    `audio_latent_count`, the contract keys (`teacher_state_numbers = (3, 6, 9)`,
    `stage3_target_state_indices = (16, 33, 49)`) and the three `(2 * audio_latent_count, 32)`
    float32 milestones captured in `generate_audio_track`.
    """
    latent_height = height // pipeline.vae_spatial_compression_ratio
    latent_width = width // pipeline.vae_spatial_compression_ratio
    os.makedirs(output_dir, exist_ok=True)

    for request_index, record in enumerate(request_records):
        torch.save(
            {
                "prompt": record["prompt"],
                "seed": seed,
                "audio_latent_count": record["audio_latent_count"],
                "teacher_state_numbers": TAOMATE_H3_TEACHER_STATE_NUMBERS,
                "stage3_target_state_indices": TAOMATE_H3_DISTILLED_STATE_INDICES[1:],
                "milestones": record["milestones"],
            },
            os.path.join(output_dir, f"request_{request_index:02d}.pt"),
        )

    completion = {
        "mode": "base10_milestones",
        "strategy": "previous_clean_audio_tail_reference_then_new_noise",
        "partition": "fl2va",
        "base_precision": "bf16",
        "request_count": request_count,
        "request_seconds": 5,
        "request_seeds": [seed] * request_count,
        "audio_noise_seed_sequence": [record["audio_noise_seed"] for record in request_records],
        "producer_geometry": taomate_h3_teacher_geometry(
            width, height, video_latent_h=latent_height, video_latent_w=latent_width
        ),
        "official_audio_latents_per_channel": TAOMATE_H3_REQUEST_AUDIO_LATENTS,
        "active_audio_latents_per_channel": [record["audio_latent_count"] for record in request_records],
        "request_receipts": [
            {
                "transport_prefix_audio_latents_per_channel": record["transport_prefix"],
                "packed_text_rows": record["packed_text_rows"],
                "packed_audio_rows": record["packed_audio_rows"],
                "reference_latents_per_channel": record["reference_latents_per_channel"],
                "reference_duration_seconds": 0.0 if record["reference_latents_per_channel"] == 0 else 1.0,
                "audio_noise_seed": record["audio_noise_seed"],
                "prefix_source_request": None if request_index == 0 else request_index - 1,
            }
            for request_index, record in enumerate(request_records)
        ],
        "full_state_count": 10,
        "executed_forwards_per_request": 9,
        "teacher_state_numbers": list(TAOMATE_H3_TEACHER_STATE_NUMBERS),
        "stage3_target_state_indices": list(TAOMATE_H3_DISTILLED_STATE_INDICES[1:]),
        "artifact_storage_dtype": "float32",
        "video_rows": 0,
        "noise_order": "draw_and_discard_full_video_then_draw_audio",
        "reference_latents_per_channel": TAOMATE_H3_ROLLOVER_REFERENCE_LATENTS,
        "reference_duration_seconds": 1.0,
        "persistent_kv": False,
        "waveform_crossfade": False,
        "audio_vae_decode_count": 0,
        "adapter_loaded": False,
    }
    with open(os.path.join(output_dir, "complete.json"), "w", encoding="utf-8") as handle:
        json.dump(completion, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


# One continuous audio timeline 5 seconds at a time: every request after the first denoises the previous
# request's clean tail as a frozen reference, and the captured 3/6/9 rows are the Base10 teacher artifact.
audio_rows, request_records = generate_audio_track(
    pipeline,
    prompt=prompt,
    seed=seed,
    request_count=request_count,
    height=sample_size[0],
    width=sample_size[1],
    device=device,
)

# Deliver the offline Base10 teacher artifact the streaming runtime consumes: these are exactly the
# rows `TaomateH3TeacherArtifact.open` reads back, so `predict_t2av_streaming.py` can point its
# `audio_teacher_dir` straight at this directory.
if audio_teacher_dir is not None:
    write_teacher_artifact(
        audio_teacher_dir,
        pipeline=pipeline,
        request_records=request_records,
        request_count=request_count,
        seed=seed,
        height=sample_size[0],
        width=sample_size[1],
    )
    print(
        f"saved Base10 teacher artifact: {audio_teacher_dir} "
        f"({request_count} request(s), base_precision=bf16)",
        flush=True,
    )

# One-shot publication, exactly like the streaming pipeline's own ending: splice the requests
# (already prefix-free) and decode the full timeline once.
total_audio_latents = int(audio_rows.shape[0]) // 2
_park_on_cpu(pipeline.transformer)
with torch.no_grad():
    audio = pipeline.decode_audio_latents(audio_rows.to(device), 0, total_audio_latents)

waveform = audio[0].float().cpu()
sample_rate = pipeline.audio_sampling_rate
duration = waveform.shape[-1] / sample_rate

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    audio_path = os.path.join(save_path, prefix + ".wav")
    torchaudio.save(audio_path, waveform, sample_rate)
    print(
        f"saved {audio_path}: {total_audio_latents} latents/channel, {duration:.3f}s @ {sample_rate} Hz, "
        f"{waveform.shape[0]} channels",
        flush=True,
    )

save_results()
