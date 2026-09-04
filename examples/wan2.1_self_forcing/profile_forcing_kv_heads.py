# Offline head profiling for Forcing-KV (arXiv 2605.09681).
#
# Runs the Self-Forcing causal pipeline on one prompt while forward hooks on
# every CasualWanSelfAttention accumulate per-head attention mass by region
# (sink / last-K frames / distant history). A head is classified Static when
# the last-K frames hold >= threshold of the post-sink attention mass (the
# official simplified Eq. 1 criterion from zju-jiyicheng/Forcing-KV
# configs_head/head_profile.py: THRESHOLD=0.8, LAST_K=4, skip sink frames).
# The result is written to forcing_kv_head_profile.json in the official
# {"format": "forcingkv_offline", "layers": [...]} format consumed by
# predict_t2v_forcing_kv.py via forcing_kv_head_profile.
#
# [NOTE]: profile with the SAME local_attn_size / sink_size /
# num_frame_per_block you intend to use at inference, since region boundaries
# depend on them. Single GPU only.
import json
import math
import os
import sys

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (AutoencoderKLWan, AutoTokenizer,
                               WanT5EncoderModel,
                               WanTransformer3DModel_SelfForcing)
from videox_fun.pipeline import WanSelfForcingPipeline
from videox_fun.utils.utils import filter_kwargs

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# model path
model_name          = "models/Diffusion_Transformer/Wan2.1-T2V-1.3B"

# Load pretrained model if need
transformer_path    = "models/Diffusion_Transformer/Self-Forcing/checkpoints/self_forcing_dmd.pt"

# Self-Forcing causal inference config (MUST match the target inference setup)
# Number of frames to generate per block
num_frame_per_block     = 3
# Local attention window size (-1 for global attention)
local_attn_size         = 6
# Sink frames always kept at the start of the rolling KV cache
sink_size               = 1
# Others
independent_first_frame = False
context_noise           = 0.0

# Profiling config
# Official criterion (configs_head/head_profile.py): a head is Static when
# last_k frames / post-sink total attention mass >= threshold (default 0.8).
threshold           = 0.8
last_k              = 4
# Frames to generate while profiling (more frames = better stats, slower)
video_length        = 81
sample_size         = [480, 832]
shift               = 5 
guidance_scale      = 1.0
num_inference_steps = 4
seed                = 43
prompt              = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
# Output head profile JSON path
output_path         = "asset/forcing_kv_head_profile.json"

# Use torch.float16 if GPU does not support torch.bfloat16
weight_dtype        = torch.bfloat16

device = set_multi_gpus_devices(1, 1)
config = OmegaConf.load(config_path)

# Load transformer with causal inference support if enabled
transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
transformer_additional_kwargs['local_attn_size'] = local_attn_size
transformer_additional_kwargs['sink_size'] = sink_size

transformer = WanTransformer3DModel_SelfForcing.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=transformer_additional_kwargs,
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")

    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    state_dict = state_dict["generator_ema"] if "generator_ema" in state_dict else state_dict
    state_dict = state_dict["generator"] if "generator" in state_dict else state_dict
    if any("._fsdp_wrapped_module." in k for k in state_dict.keys()):
        state_dict = {k.replace("model._fsdp_wrapped_module.", "model.", 1) if k.startswith("model._fsdp_wrapped_module.") else k: v for k, v in state_dict.items()}
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()}

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

# Get Scheduler
scheduler = FlowMatchEulerDiscreteScheduler(
    **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = WanSelfForcingPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)
pipeline.to(device=device)

# Profiling hooks: after every cached self-attn forward, the layer exposes
# kv_cache["_fkv_last_q"] / "_fkv_window_start" / "_fkv_local_end". Recompute
# softmax attention mass in fp32 (row-chunked) and partition it by region.
layer_stats = []
hooks = []
ROW_CHUNK = 128

for layer_idx, block in enumerate(pipeline.transformer.blocks):
    attn = block.self_attn
    acc = {
        "sink": torch.zeros(attn.num_heads, dtype=torch.float64),
        "last_k": torch.zeros(attn.num_heads, dtype=torch.float64),
        "distant": torch.zeros(attn.num_heads, dtype=torch.float64),
        "total": 0.0,
        "seen_starts": set(),
    }
    layer_stats.append(acc)

    def make_hook(acc):
        def hook(module, inputs, output):
            kv_cache = inputs[5] if len(inputs) > 5 else None
            if kv_cache is None or "_fkv_last_q" not in kv_cache:
                return
            current_start = int(inputs[6])
            # Only profile the first forward per chunk position (denoise step 0);
            # later steps attend over the identical window with noisier keys.
            if current_start in acc["seen_starts"]:
                return
            acc["seen_starts"].add(current_start)

            q = kv_cache["_fkv_last_q"]                      # [B, s, n, d]
            window_start = int(kv_cache["_fkv_window_start"])
            local_end = int(kv_cache["_fkv_local_end"])
            grid_sizes = inputs[2]
            frame_seqlen = int(math.prod(grid_sizes[0][1:]))

            k_win = kv_cache["k"][:, window_start:local_end]  # [B, L, n, d]
            sink_tokens = module.sink_size * frame_seqlen
            local_start = local_end - q.shape[1]
            # Regions cover HISTORY only ([0, local_start)) for sink clamping;
            # clamping by local_start avoids overlap with the current chunk on
            # early chunks, which would double-count attention mass (score > 1).
            sink_end = min(sink_tokens, local_start)
            # Last-K frames (inclusive of the current chunk), mirroring the
            # official LAST_K criterion; clamped against the sink region.
            last_k_start = max(sink_end, local_end - last_k * frame_seqlen)
            rel_sink = max(0, sink_end - window_start)
            rel_last = max(rel_sink, last_k_start - window_start)

            qf = q[0].float()                                 # [s, n, d]
            kf = k_win[0].float()                             # [L, n, d]
            scale = kf.shape[-1] ** -0.5
            for r0 in range(0, qf.shape[0], ROW_CHUNK):
                qc = qf[r0:r0 + ROW_CHUNK]                    # [c, n, d]
                probs = torch.einsum(
                    "cnd,lnd->ncl", qc, kf).mul_(scale).softmax(dim=-1)
                acc["sink"] += probs[:, :, :rel_sink].sum(dim=(1, 2)).double().cpu()
                acc["last_k"] += probs[:, :, rel_last:].sum(dim=(1, 2)).double().cpu()
                acc["distant"] += probs[:, :, rel_sink:rel_last].sum(dim=(1, 2)).double().cpu()
                acc["total"] += float(qc.shape[0])
        return hook

    hooks.append(attn.register_forward_hook(make_hook(acc)))

generator = torch.Generator(device=device).manual_seed(seed)

with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1

    pipeline(
        prompt, 
        num_frames = video_length,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale          = guidance_scale,
        num_inference_steps     = num_inference_steps,
        shift                   = shift,
        num_frame_per_block     = num_frame_per_block,
        independent_first_frame = independent_first_frame,
        context_noise           = context_noise,
        stochastic_sampling     = True,
        output_type             = "latent",
    )

for h in hooks:
    h.remove()

# Classify heads (official criterion: last-K / post-sink >= threshold) and
# dump the profile JSON in the official forcingkv_offline format.
layers_out = []
num_static = 0
print(f"\n{'layer':>5} {'static heads':<40} {'mean score':>10}")
for layer_idx, acc in enumerate(layer_stats):
    denom = (torch.clamp(torch.tensor(acc["total"]), min=1e-8) - acc["sink"]).clamp(min=1e-8)
    score = acc["last_k"] / denom
    static = [h for h in range(score.shape[0]) if score[h].item() >= threshold]
    dynamic = [h for h in range(score.shape[0]) if h not in set(static)]
    num_heads = score.shape[0]
    num_static += len(static)
    layers_out.append({
        "layer_idx": layer_idx,
        "static_head": static,
        "dynamic_head": dynamic,
    })
    print(f"{layer_idx:>5} {str(static):<40} {score.mean().item():>10.4f}")

profile = {
    "format": "forcingkv_offline",
    "num_layers": len(layer_stats),
    "num_heads": num_heads,
    "layers": layers_out,
}
with open(output_path, "w") as f:
    json.dump(profile, f, indent=2)

total_layers = len(layer_stats)
print(f"\nthreshold={threshold}, last_k={last_k}: {num_static}/{total_layers * num_heads} heads static "
      f"({num_static / max(1, total_layers * num_heads) * 100:.1f}%), "
      f"dynamic {100 - num_static / max(1, total_layers * num_heads) * 100:.1f}%")
print(f"Profile saved to: {output_path}")
