# Modified from https://github.com/Robbyant/lingbot-video/blob/main/lingbot_video/rewriter
# Copyright 2025 The LingBot-Video Team and The HuggingFace Team. All rights reserved.
#
# Prompt-rewriter integration for LingBot-Video: wraps the official two-step (EXPAND -> MAP)
# rewriter pipeline and provides the structured-caption JSON schema helpers.
# Reference: https://github.com/Robbyant/lingbot-video/tree/main/lingbot_video/rewriter
#
"""Official prompt-rewriter integration for LingBot-Video (in-process).

ALL prompt input for LingBot-Video must go through the official rewriter:
the DiT was trained only on structured JSON captions, so natural-language
prompts are converted with the two-step official pipeline before use:

    step 1 EXPAND : base VLM (Qwen3.6-27B), LoRA disabled -> detailed prose
    step 2 MAP    : same base VLM + rewriter LoRA          -> JSON caption

The official implementation (``repo/lingbot-video/rewriter``: TransformersBackend
+ Rewriter + NegativePromptEditor) is imported and loaded directly in-process.
Run the host script with the dedicated venv python so the rewriter's modern
``transformers`` (>=5.x, with the ``qwen3_5`` module) and matching
``diffusers``/``peft`` are used end-to-end:

    /root/rewriter_venv/bin/python examples/lingbot_video/predict_t2v.py

The 27B base VLM is loaded once per ``LingBotVideoRewriter`` instance and stays
resident across all its rewrite/auto_negative calls; ``close()`` frees it (call
before loading the DiT pipeline so the two never share the GPU at once).

Weights (set via args or environment):
    REWRITER_BASE_MODEL -> Qwen/Qwen3.6-27B (default models/Diffusion_Transformer/Qwen3.6-27B)
    REWRITER_ADAPTER    -> Robbyant/lingbot-video-rewriter-lora
                           (default models/Diffusion_Transformer/lingbot-video-rewriter-lora)

The structured-caption schema helpers (CAMERA_CHOICES / build_caption / element /
cam / load_caption / is_valid_caption) are co-located here as the single source
of truth for the rewriter's JSON schema; ``is_valid_caption`` is also used by
train.py / prepare_captions.py to validate dataset metadata WITHOUT loading any
model (the rewriter VLM is only loaded on LingBotVideoRewriter instantiation).

Usage:
    from videox_fun.models.lingbot_video_rewriter import ensure_json_caption
    prompt = ensure_json_caption("a red ball rolls across the floor", mode="t2v",
                                 duration=3.3, cache_file="samples/caption_cache.json")
"""
import gc
import hashlib
import json
import os
import sys


# ==================== Structured-caption schema (official rewriter JSON) ====================

# Valid choices for camera_info fields (official rewriter schema).
CAMERA_CHOICES = {
    "color": ["Warm", "Cool", "Mixed", "Saturated", "Desaturated", "Black and White",
              "Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Magenta", "Pink"],
    "frame_size": ["Extreme Wide", "Wide", "Medium Wide", "Medium",
                   "Medium Close Up", "Close Up", "Extreme Close Up"],
    "shot_type_angle": ["High angle", "Low angle", "Dutch angle", "Overhead", "Aerial", "Eye level"],
    "lens_size": ["Ultra Wide / Fisheye", "Wide", "Medium", "Long Lens", "Telephoto"],
    "composition": ["Center", "Balanced", "Symmetrical", "Left heavy", "Right heavy", "Short side"],
    "lighting": ["Hard light", "Soft light", "High contrast", "Low contrast", "Side light",
                 "Top light", "Underlight", "Backlight", "Edge light", "Silhouette"],
    "lighting_type": ["Daylight", "Sunny", "Overcast", "Moonlight", "Artificial light",
                      "Practical light", "Tungsten", "Fluorescent", "Firelight", "Mixed light"],
}

_CAMERA_DEFAULTS = {
    "color": "Warm",
    "frame_size": "Medium",
    "shot_type_angle": "Eye level",
    "lens_size": "Medium",
    "composition": "Center",
    "lighting": "Soft light",
    "lighting_type": "Daylight",
}

_ELEMENT_FIELDS = (
    "name", "description", "actions", "location", "relative_size", "shape_and_color",
    "texture", "appearance_details", "relationship", "orientation",
    "pose", "expression", "clothing", "gender", "skin_tone_and_texture",
)


def cam(**kwargs):
    """Build the camera_info object. Unset keys get safe defaults.
    Values are validated against CAMERA_CHOICES."""
    info = dict(_CAMERA_DEFAULTS)
    for key, value in kwargs.items():
        if key not in CAMERA_CHOICES:
            raise KeyError(f"unknown camera_info key: {key!r}; valid keys: {sorted(CAMERA_CHOICES)}")
        if value and value not in CAMERA_CHOICES[key]:
            raise ValueError(
                f"camera_info[{key!r}]={value!r} not allowed; choices: {CAMERA_CHOICES[key]}")
        info[key] = value
    return info


def element(name, description="", actions=(), location="", relative_size="medium",
            shape_and_color="", texture="", appearance_details="", relationship="",
            orientation="", pose="", expression="", clothing="", gender="",
            skin_tone_and_texture=""):
    """Build one prominent_element.

    actions: iterable of (timestamp, action) tuples, e.g.
             [("[0.0s - 3.3s]", "walking slowly to the left")].
             Pass () for a static element (schema: one entry with empty action).
    Human-only fields (pose/expression/clothing/gender/skin_tone_and_texture)
    stay empty for non-human objects.
    """
    action_list = [{"timestamp": ts, "action": act} for ts, act in actions]
    if not action_list:
        action_list = [{"timestamp": "", "action": ""}]
    elem = {
        "name": name,
        "description": description,
        "actions": action_list,
        "location": location,
        "relative_size": relative_size,
        "shape_and_color": shape_and_color,
        "texture": texture,
        "appearance_details": appearance_details,
        "relationship": relationship,
        "orientation": orientation,
        "pose": pose,
        "expression": expression,
        "clothing": clothing,
        "gender": gender,
        "skin_tone_and_texture": skin_tone_and_texture,
    }
    # Keep only schema keys (guards against typos in kwargs via dict literals).
    return {k: elem[k] for k in _ELEMENT_FIELDS}


def build_caption(scene, elements, camera_movement="", camera_info=None, indent=None):
    """Assemble the full structured caption and return the JSON string that the
    pipeline expects as `prompt`.

    scene:             scene_content_description (<= 800 words; no camera info here)
    camera_movement:   camera_movement_description (<= 100 words, '' if static)
    elements:          list from element(...)
    camera_info:       dict from cam(...) (defaults applied when None)
    """
    caption = {
        "comprehensive_description": {
            "scene_content_description": scene,
            "camera_movement_description": camera_movement,
        },
        "prominent_elements": list(elements),
        "camera_info": camera_info if camera_info is not None else cam(),
    }
    return json.dumps(caption, ensure_ascii=False, indent=indent)


def load_caption(path):
    """Load a caption saved by the official rewriter (`--output` of
    rewriter/inference.py, i.e. {caption, duration}) or a plain caption JSON."""
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    caption = obj.get("caption", obj) if isinstance(obj, dict) else obj
    return caption if isinstance(caption, str) else json.dumps(caption, ensure_ascii=False)


def is_valid_caption(text):
    """Check that `text` looks like a LingBot-Video structured JSON caption:
    parseable JSON containing comprehensive_description / prominent_elements /
    camera_info. Plain natural-language strings fail this check."""
    if not isinstance(text, str) or not text.strip().startswith("{"):
        return False
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return False
    if not isinstance(obj, dict):
        return False
    keys = {"comprehensive_description", "prominent_elements", "camera_info"}
    return keys.issubset(obj.keys())


# ==================== Prompt rewriter (in-process two-step EXPAND -> MAP) ====================

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REWRITER_PKG_DIR = os.path.join(_REPO_ROOT, "repo", "lingbot-video", "rewriter")

DEFAULT_REWRITER_BASE = os.environ.get(
    "REWRITER_BASE_MODEL", "models/Diffusion_Transformer/Qwen3.6-27B")
DEFAULT_REWRITER_ADAPTER = os.environ.get(
    "REWRITER_ADAPTER", "models/Diffusion_Transformer/lingbot-video-rewriter-lora")

_WEIGHT_HELP = (
    "Rewriter weights missing. Download them first:\n"
    "  modelscope download --model Qwen/Qwen3.6-27B --local_dir {base}\n"
    "  modelscope download --model Robbyant/lingbot-video-rewriter-lora --local_dir {adapter}\n"
    "or set REWRITER_BASE_MODEL / REWRITER_ADAPTER to existing paths."
)


def _patch_peft_compat_checks():
    """peft 0.20.0's ``is_gptqmodel_available`` / ``is_torchao_available`` raise
    ImportError when the inherited system package is too old (instead of
    returning False), which aborts the awq/gptq/torchao dispatchers while
    loading a plain bf16 LoRA and thus breaks ``PeftModel.from_pretrained``.
    The rewriter LoRA is a plain bf16 adapter (not gptq/awq/torchao), so these
    checks are irrelevant; short-circuit them to False in the dispatcher
    modules that bound the names by import."""
    _PATCHES = {
        "peft.tuners.lora.awq": ("is_gptqmodel_available", False),
        "peft.tuners.lora.gptq": ("is_gptqmodel_available", False),
        "peft.tuners.lora.torchao": ("is_torchao_available", False),
    }
    for mod_name, (attr, val) in _PATCHES.items():
        try:
            mod = __import__(mod_name, fromlist=["x"])
            setattr(mod, attr, lambda: val)
        except Exception:
            pass


class LingBotVideoRewriter:
    """Loads the official two-step rewriter (EXPAND -> MAP) in-process on a local
    base VLM + LoRA adapter. Also provides the official auto-negative editor.

    The 27B base VLM is loaded once in ``__init__`` and stays resident for all
    subsequent rewrite/auto_negative calls; ``close()`` frees it (call before
    loading the DiT pipeline)."""

    def __init__(self, base=None, adapter=None, device="auto", max_new_tokens=6144):
        base = base or DEFAULT_REWRITER_BASE
        adapter = adapter or DEFAULT_REWRITER_ADAPTER
        if not os.path.isdir(REWRITER_PKG_DIR):
            raise FileNotFoundError(
                f"official rewriter package not found at {REWRITER_PKG_DIR}; "
                "clone the LingBot-Video release repo into repo/lingbot-video first.")
        if not os.path.isdir(base) or not os.path.isdir(adapter):
            raise FileNotFoundError(_WEIGHT_HELP.format(base=base, adapter=adapter))

        # The rewriter package uses flat imports (rewriter_core/system_prompts).
        if REWRITER_PKG_DIR not in sys.path:
            sys.path.insert(0, REWRITER_PKG_DIR)
        _patch_peft_compat_checks()
        from inference import TransformersBackend       # noqa: E402
        from rewriter_core import Rewriter               # noqa: E402

        self.base, self.adapter = base, adapter
        self._backend = TransformersBackend(base, adapter, device=device,
                                            max_new_tokens=max_new_tokens)
        self._rewriter = Rewriter(self._backend)

    def rewrite(self, prompt, mode="t2v", first_frame=None, duration=5.0,
                return_raw=False):
        """Run EXPAND + MAP. Returns the JSON caption string (and the full raw
        result dict when return_raw=True). mode: t2v | ti2v | t2i."""
        out = self._rewriter.rewrite(prompt, mode, first_frame, duration)
        if out.get("json") is None:
            raise RuntimeError(
                "rewriter did not produce a valid JSON caption; raw step2 output: "
                f"{str(out.get('json_raw'))[:500]}")
        caption = json.dumps(out["json"], ensure_ascii=False)
        return (caption, out) if return_raw else caption

    def auto_negative(self, caption, mode="t2v", first_frame=None):
        """Official per-sample negative pruning (base VLM, LoRA disabled).
        `caption` may be a JSON caption string or a dict."""
        from auto_negative import NegativePromptEditor  # noqa: E402
        cap = json.loads(caption) if isinstance(caption, str) else caption
        editor = NegativePromptEditor(self._backend)
        return editor.edit(cap, mode, first_frame)["negative_str"]

    def close(self):
        """Free the base VLM (call before loading the DiT pipeline)."""
        for attr in ("_rewriter", "_backend"):
            obj = getattr(self, attr, None)
            if obj is not None:
                model = getattr(obj, "model", None)
                if model is not None:
                    del model
                delattr(self, attr)
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def _cache_key(prompt, mode, duration, first_frame):
    ff = first_frame if isinstance(first_frame, str) else None
    raw = json.dumps({"prompt": prompt, "mode": mode,
                      "duration": float(duration), "first_frame": ff},
                     ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cached_caption(cache_file, prompt, mode, duration, first_frame=None):
    """Return the cached JSON caption string, or None on miss."""
    if not cache_file or not os.path.isfile(cache_file):
        return None
    try:
        with open(cache_file, encoding="utf-8") as f:
            cache = json.load(f)
    except (ValueError, OSError):
        return None
    entry = cache.get(_cache_key(prompt, mode, duration, first_frame))
    return entry.get("caption") if isinstance(entry, dict) else None


def save_cached_caption(cache_file, caption, prompt, mode, duration, first_frame=None):
    if not cache_file:
        return
    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    cache = {}
    if os.path.isfile(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as f:
                cache = json.load(f)
        except (ValueError, OSError):
            cache = {}
    cache[_cache_key(prompt, mode, duration, first_frame)] = {
        "caption": caption, "prompt": prompt, "mode": mode, "duration": float(duration),
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def ensure_json_caption(prompt, mode="t2v", duration=5.0, first_frame=None,
                        cache_file=None, base=None, adapter=None, device="auto"):
    """The single entry point for ALL LingBot-Video prompt input.

    - already-valid JSON captions pass through (they are rewriter output);
    - otherwise the official rewriter is loaded, the prompt is rewritten, the
      rewriter is freed, and (when cache_file is given) the result is cached so
      later runs with the same prompt/mode/duration skip the rewrite.
    """
    if is_valid_caption(prompt):
        return prompt
    cached = load_cached_caption(cache_file, prompt, mode, duration, first_frame)
    if cached is not None:
        return cached
    rewriter = LingBotVideoRewriter(base=base, adapter=adapter, device=device)
    try:
        caption = rewriter.rewrite(prompt, mode=mode, first_frame=first_frame,
                                   duration=duration)
    finally:
        rewriter.close()
    save_cached_caption(cache_file, caption, prompt, mode, duration, first_frame)
    return caption
