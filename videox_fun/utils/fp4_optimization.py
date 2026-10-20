import torch
import torch.nn as nn

from .fp8_optimization import _is_fsdp_managed

# E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit -> 8 magnitudes, 16 codes.
FLOAT4_MAX = 6.0
# NVFP4 blocks are 16 elements wide and scale them with a single E4M3 byte (plus one
# fp32 global scale per tensor); block 32 with an fp32 scale is the OCP MXFP4 flavour.
FLOAT4_BLOCK_SIZE = 16
FLOAT8_E4M3 = torch.float8_e4m3fn
FLOAT8_E4M3_MAX = torch.finfo(FLOAT8_E4M3).max  # 448.0
FLOAT4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Mid-points between consecutive magnitudes; a value is rounded to the bucket this
# falls into. Boundaries are exclusive for the lower bucket (see _float4_round).
FLOAT4_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Index = code (magnitude index | sign << 3).
FLOAT4_LUT = FLOAT4_MAGNITUDES + tuple(-v for v in FLOAT4_MAGNITUDES)

_FLOAT4_TABLES = {}


def _float4_tables(device):
    """The two constant tables, built once per device."""
    tables = _FLOAT4_TABLES.get(device)
    if tables is None:
        tables = (
            torch.tensor(FLOAT4_LUT, dtype=torch.float32, device=device),
            torch.tensor(FLOAT4_MAGNITUDES, dtype=torch.float32, device=device),
            torch.tensor(FLOAT4_BOUNDARIES, dtype=torch.float32, device=device),
        )
        _FLOAT4_TABLES[device] = tables
    return tables


def _float4_scale_name(param_name):
    return param_name + "_fp4_scale"


def _float4_global_scale_name(param_name):
    # NVFP4's second-level, per-tensor scale (a single fp32 value), stored next to the
    # per-block E4M3 scales so both move together under the offload hooks.
    return param_name + "_fp4_global_scale"


def _float4_meta_name(param_name):
    # Plain attribute (not a buffer): the unpacked shape and the block size are Python
    # constants that never need to move between devices.
    return param_name + "_fp4_meta"


def _float4_round(magnitudes, boundaries):
    """Round non-negative scaled values to the nearest E2M1 magnitude index.

    ``bucketize(..., right=True)`` is round-half-up; a value sitting exactly on a
    boundary is then pushed down to the even code, which makes the whole thing
    round-nearest-even like the hardware FP8 cast in fp8_optimization. Adjacent codes
    always differ in parity, so on a tie either neighbour can be the even one and one
    where covers both directions.
    """
    idx = torch.bucketize(magnitudes, boundaries, right=True)
    lower = (idx - 1).clamp(min=0)
    # `lower` indexes the boundary between the two candidate codes, not a grid value.
    tie = (magnitudes == boundaries[lower]) & (idx > 0)
    return torch.where(tie & (lower % 2 == 0), lower, idx)


@torch.no_grad()
def quantize_weight_to_float4(tensor, block_size=FLOAT4_BLOCK_SIZE):
    """Return ``(packed_uint8, block_scale_fp8, global_scale)`` for a 2-D weight.

    NVFP4's two-level scaling: the per-tensor ``global_scale`` maps the largest block
    scale onto the E4M3 grid, and each 16-element block stores its scale as one E4M3
    byte. The element codes are quantized with the *dequantized* block scale
    (``block_scale_fp8.float() / global_scale``), so the stored nibbles reproduce
    exactly what the unpack divides back out -- no scale mismatch between the two.
    """
    rows, features = tensor.shape[0], tensor.shape[1]
    x = tensor.detach().float().reshape(rows, features)
    pad = (-features) % block_size
    if pad:
        # Only reachable for weights whose input dim is not a multiple of the block;
        # the padding is dropped again by the unpack through the stored shape.
        x = nn.functional.pad(x, (0, pad))
    padded = x.shape[1]
    blocks = x.reshape(-1, block_size)

    # Second-level (per-tensor) scale: (E4M3_MAX * E2M1_MAX) / global_amax, so the
    # widest block (amax == global_amax) scales to exactly E4M3_MAX and no block scale
    # overflows fp8. global_amax is clamped so an all-zero weight yields codes of 0.
    global_amax = x.abs().amax().clamp(min=1e-12)
    global_scale = (FLOAT8_E4M3_MAX * FLOAT4_MAX) / global_amax

    # First-level (per-block) scale, rounded to E4M3 (NVFP4's storage type) then read
    # back, so quantization uses the identical scale the unpack will use. A block whose
    # scale underflows to 0 in fp8 (or a zero-initialized control projection) is clamped
    # instead of dividing by zero, keeping its codes at 0 rather than NaN.
    block_amax = blocks.abs().amax(dim=1, keepdim=True)
    block_scale_fp8 = ((block_amax / FLOAT4_MAX) * global_scale).to(FLOAT8_E4M3)
    block_scale = (block_scale_fp8.float() / global_scale).clamp(min=1e-12)

    _, _, boundaries = _float4_tables(x.device)
    scaled = (blocks / block_scale).clamp(-FLOAT4_MAX, FLOAT4_MAX)
    idx = _float4_round(scaled.abs().reshape(rows, padded), boundaries)
    code = torch.where(scaled.reshape(rows, padded) < 0, idx + 8, idx).to(torch.uint8)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed, block_scale_fp8.reshape(rows, padded // block_size, 1), global_scale


@torch.no_grad()
def dequantize_weight_from_float4(packed, block_scale_fp8, global_scale, shape,
                                  block_size=FLOAT4_BLOCK_SIZE, dtype=torch.bfloat16):
    """Rebuild the ``(out, in)`` weight from the nibbles and the two-level scales."""
    rows, half = packed.shape
    code = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(rows, half * 2)
    lut, _, _ = _float4_tables(packed.device)
    values = lut[code.long()]
    # E4M3 block scale back to fp32, then undo the per-tensor global scale.
    block_scale = block_scale_fp8.float() / global_scale
    blocks = values.reshape(rows, block_scale_fp8.shape[1], block_size) * block_scale
    return blocks.reshape(rows, -1)[:, :shape[1]].reshape(shape).to(dtype)


def _float4_pack_weight(module, device, block_size):
    """Pack ``module.weight`` (a 2-D floating tensor) into E2M1 nibbles + scale/meta buffers.

    Single source of truth for the packing, shared by the whole-model
    :func:`convert_model_weight_to_float4` and the per-layer
    :func:`float4_quantize_linear` used around a LoRA merge.
    """
    weight = module.weight
    packed, block_scale_fp8, global_scale = quantize_weight_to_float4(weight.data, block_size)
    to_device = (lambda t: t.to(device)) if device is not None else (lambda t: t)
    module.register_buffer(_float4_scale_name("weight"), to_device(block_scale_fp8))
    module.register_buffer(_float4_global_scale_name("weight"), to_device(global_scale))
    setattr(module, _float4_meta_name("weight"), (tuple(weight.shape), block_size))
    # A Parameter may only hold a non-floating tensor while it does not require
    # grad, and a nibble-packed weight could not be trained through anyway.
    weight.requires_grad_(False)
    weight.data = to_device(packed)


def _float4_wrap_forward(module, origin_dtype):
    """Install the per-forward FP4 unpack on ONE module holding a packed weight."""
    if getattr(module, _float4_meta_name("weight"), None) is None:
        return
    if hasattr(module, "original_forward_fp4"):
        return
    module.original_forward_fp4 = module.forward
    setattr(
        module,
        "forward",
        lambda *inputs, m=module, **kwargs: _float4_autocast_forward(m, origin_dtype, *inputs, **kwargs)
    )


def convert_model_weight_to_float4(model, exclude_module_name=("modulation",),
                                   device=None, block_size=FLOAT4_BLOCK_SIZE):
    """Pack every ``nn.Linear`` weight of ``model`` into E2M1 nibbles, in place."""
    if _is_fsdp_managed(model):
        raise NotImplementedError(
            "GPU_memory_mode 'qfloat4' is not supported under FSDP: the managed "
            "parameters are flat-storage views, so a block-scaled 4-bit repacking of "
            "`weight` has no shape to keep. Use 'qfloat8' (which dequantizes on the "
            "output side instead) or no quantization.")
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(keyword in name for keyword in exclude_module_name):
            continue
        weight = module.weight
        if weight is None or weight.dtype == torch.uint8 or weight.dim() != 2:
            continue
        _float4_pack_weight(module, device, block_size)
    return model


@torch.no_grad()
def _float4_autocast_forward(module, origin_dtype, *inputs, **kwargs):
    """Swap the packed weight for its bf16 reconstruction around one forward.

    Unlike the FP8 wrapper this never re-quantizes on the way out: the packed nibbles
    are simply put back, so the stored bits cannot drift across steps and the dequant
    costs one gather plus one mul instead of a division.
    """
    restored = []
    for param_name, param in list(module.named_parameters(recurse=False)):
        meta = getattr(module, _float4_meta_name(param_name), None)
        if meta is None:
            if param.is_floating_point() and param.dtype != origin_dtype:
                # An unquantized fp32 bias next to a bf16 weight would make F.linear
                # reject the pair; fp8_optimization normalizes them the same way.
                restored.append((param, param.data))
                param.data = param.data.to(origin_dtype)
            continue
        if param.dtype != torch.uint8:
            # `module.to(dtype)` walks every parameter and would silently turn the
            # packed bytes into the bf16 numbers 0..255. Refuse instead of generating
            # with a destroyed weight.
            raise RuntimeError(
                "The FP4-packed parameter '%s' holds a %s tensor, not the packed uint8 "
                "nibbles: something called .to(dtype) on a qfloat4 model. Apply the "
                "memory mode after the dtype move, and move devices with "
                "apply_gpu_memory_mode()/offload hooks, which keep the dtype."
                % (param_name, param.dtype))
        shape, block_size = meta
        restored.append((param, param.data))
        param.data = dequantize_weight_from_float4(
            param.data, getattr(module, _float4_scale_name(param_name)),
            getattr(module, _float4_global_scale_name(param_name)),
            shape, block_size, origin_dtype)

    # Integer tensors (indices) must keep their dtype or the kernel rejects them.
    inputs = [input.to(origin_dtype) if torch.is_tensor(input) and input.is_floating_point()
              else input for input in inputs]
    kwargs = {key: value.to(origin_dtype) if torch.is_tensor(value) and value.is_floating_point()
              else value for key, value in kwargs.items()}
    try:
        return module.original_forward_fp4(*inputs, **kwargs)
    finally:
        for param, data in restored:
            param.data = data


def convert_float4_weight_dtype_wrapper(module, origin_dtype):
    """Install the per-forward unpack on every module holding a packed weight."""
    for name, child in module.named_modules():
        if name == "" or "embed_tokens" in name or isinstance(child, nn.Embedding):
            continue
        _float4_wrap_forward(child, origin_dtype)
    return module


def undo_convert_float4_weight_dtype_wrapper(module, origin_dtype):
    """Drop the wrappers and unpack in place, leaving a plain full-precision module.

    The mirror of the FP8 undo. Afterwards the model runs its own forwards again and no
    per-forward reconstruction is needed -- at the cost of holding the weights in
    `origin_dtype` again, so this is what to call before a LoRA merge or a save.
    """
    for _, child in module.named_modules():
        if not hasattr(child, "original_forward_fp4"):
            continue
        setattr(child, "forward", child.original_forward_fp4)
        delattr(child, "original_forward_fp4")
        meta = getattr(child, _float4_meta_name("weight"), None)
        if meta is None:
            continue
        shape, block_size = meta
        weight = child.weight
        scale = getattr(child, _float4_scale_name("weight"), None)
        global_scale = getattr(child, _float4_global_scale_name("weight"), None)
        if weight is not None and weight.dtype == torch.uint8 and scale is not None:
            weight.requires_grad_(False)
            weight.data = dequantize_weight_from_float4(
                weight.data, scale, global_scale, shape, block_size, origin_dtype)
        delattr(child, _float4_meta_name("weight"))
        if scale is not None:
            delattr(child, _float4_scale_name("weight"))
        if global_scale is not None:
            delattr(child, _float4_global_scale_name("weight"))
    return module


def float4_dequantize_linear(module, origin_dtype):
    """Unpack ONE FP4-packed ``nn.Linear`` back to ``origin_dtype``, in place.

    Drops this layer's scale/meta buffers and forward wrapper, leaving a plain
    full-precision Linear. Returns True if the layer was packed (so the caller knows to
    re-pack it), False if it was already full precision and was left untouched.

    Deliberately per layer rather than whole-model: a LoRA merge edits one layer at a
    time, and materializing the *entire* transformer in ``origin_dtype`` would need the
    very memory the 4-bit packing was chosen to avoid. Only this layer's weight is
    dequantized, so the peak overhead is one layer.
    """
    meta = getattr(module, _float4_meta_name("weight"), None)
    if meta is None:
        return False
    if _is_fsdp_managed(module):
        raise NotImplementedError(
            "Cannot dequantize an FP4-packed weight under FSDP: the managed parameters "
            "are flat-storage DTensor views with no shape to unpack. Merge the LoRA "
            "before the FSDP sharding (or before apply_gpu_memory_mode).")
    shape, block_size = meta
    weight = module.weight
    scale = getattr(module, _float4_scale_name("weight"), None)
    global_scale = getattr(module, _float4_global_scale_name("weight"), None)
    if hasattr(module, "original_forward_fp4"):
        setattr(module, "forward", module.original_forward_fp4)
        delattr(module, "original_forward_fp4")
    if weight is not None and weight.dtype == torch.uint8 and scale is not None:
        weight.requires_grad_(False)
        weight.data = dequantize_weight_from_float4(
            weight.data, scale, global_scale, shape, block_size, origin_dtype)
    delattr(module, _float4_meta_name("weight"))
    if scale is not None:
        delattr(module, _float4_scale_name("weight"))
    if global_scale is not None:
        delattr(module, _float4_global_scale_name("weight"))
    return True


def float4_quantize_linear(module, origin_dtype, device=None, block_size=FLOAT4_BLOCK_SIZE):
    """Re-pack ONE ``nn.Linear`` into FP4 and reinstall its forward wrapper.

    The exact inverse of :func:`float4_dequantize_linear`, called after a LoRA delta has
    been folded into the layer's full-precision weight so the model stays 4-bit for
    generation. Returns True if the layer was packed.
    """
    if not isinstance(module, nn.Linear):
        return False
    weight = module.weight
    if weight is None or weight.dtype == torch.uint8 or weight.dim() != 2:
        return False
    _float4_pack_weight(module, device, block_size)
    _float4_wrap_forward(module, origin_dtype)
    return True
