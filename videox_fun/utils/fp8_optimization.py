"""Modified from https://github.com/kijai/ComfyUI-MochiWrapper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

FLOAT8_DTYPE = torch.float8_e4m3fn
FLOAT8_MAX = torch.finfo(FLOAT8_DTYPE).max


def replace_parameters_by_name(module, name_keywords, device):
    from torch import nn
    for name, param in list(module.named_parameters(recurse=False)):
        if any(keyword in name for keyword in name_keywords):
            if isinstance(param, nn.Parameter):
                tensor = param.data
                delattr(module, name)
                setattr(module, name, tensor.to(device=device))
    for child_name, child_module in module.named_children():
        replace_parameters_by_name(child_module, name_keywords, device)

def _float8_scale_name(param_name):
    return param_name + "_fp8_scale"

def _quantize_param_to_float8(module, param_name, param):
    # Scale-aware quantization: a per-row absmax scale (per-tensor for 1D params) stretches the weight's dynamic
    # range onto the full e4m3 grid before the cast, instead of clipping everything above 448 and coarsely rounding
    # everything small. The scale rides along as a buffer of the owning module, so device-offload hooks move it
    # together with the weight.
    tensor = param.data
    if tensor.dim() > 1:
        reduce_dims = tuple(range(1, tensor.dim()))
        scale = tensor.float().abs().amax(dim=reduce_dims, keepdim=True) / FLOAT8_MAX
    else:
        scale = tensor.float().abs().amax() / FLOAT8_MAX
    # Zero-initialised projections (the control branch's before_proj / after_proj) would divide by zero.
    scale = scale.clamp(min=1e-8)
    module.register_buffer(_float8_scale_name(param_name), scale)
    param.data = (tensor.float() / scale).to(FLOAT8_DTYPE)

def convert_model_weight_to_float8(model, exclude_module_name=['embed_tokens'], device=None, with_scale=True):
    for name, module in model.named_modules():
        flag = False
        for _exclude_module_name in exclude_module_name:
            if _exclude_module_name in name:
                flag = True
        if flag:
            continue
        # recurse=False so every parameter is quantized exactly once at its owning module; a recursive walk would
        # visit it once per ancestor and quantize it repeatedly.
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{name}.{param_name}" if name else param_name
            flag = False
            for _exclude_module_name in exclude_module_name:
                if _exclude_module_name in full_name:
                    flag = True
            if flag:
                continue
            if param.dtype == FLOAT8_DTYPE:
                continue
            if with_scale:
                _quantize_param_to_float8(module, param_name, param)
            else:
                param.data = param.data.to(FLOAT8_DTYPE)

def _dequantize_float8_weights(module, origin_dtype):
    for param_name, param in module.named_parameters(recurse=False):
        scale = getattr(module, _float8_scale_name(param_name), None)
        if param.dtype == FLOAT8_DTYPE:
            data = param.data.to(torch.float32)
            if scale is not None:
                data = data * scale
            param.data = data.to(origin_dtype)
        elif param.dtype != origin_dtype:
            param.data = param.data.to(origin_dtype)
    for child in module.children():
        _dequantize_float8_weights(child, origin_dtype)

def _requantize_float8_weights(module, storage_dtype):
    for param_name, param in module.named_parameters(recurse=False):
        scale = getattr(module, _float8_scale_name(param_name), None)
        if scale is not None:
            if param.dtype != FLOAT8_DTYPE:
                param.data = (param.data.to(torch.float32) / scale).to(FLOAT8_DTYPE)
        elif param.dtype != storage_dtype:
            param.data = param.data.to(storage_dtype)
    for child in module.children():
        _requantize_float8_weights(child, storage_dtype)

def autocast_model_forward(cls, origin_dtype, *inputs, **kwargs):
    storage_dtype = cls.weight.dtype
    _dequantize_float8_weights(cls, origin_dtype)

    # Convert all inputs to the original dtype
    inputs = [input.to(origin_dtype) if torch.is_tensor(input) else input for input in inputs]
    out = cls.original_forward(*inputs, **kwargs)

    _requantize_float8_weights(cls, storage_dtype)
    return out

def _is_fsdp_managed(module):
    # FSDP1 wraps modules in `FullyShardedDataParallel`; FSDP2 (`fully_shard`) instead turns the managed
    # parameters into DTensors without any wrapper class.
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
        if isinstance(module, FullyShardedDataParallel) or any(
                isinstance(m, FullyShardedDataParallel) for m in module.modules()):
            return True
    except ImportError:
        pass
    try:
        from torch.distributed.tensor import DTensor
        return any(isinstance(p, DTensor) for p in module.parameters())
    except ImportError:
        return False

def convert_weight_dtype_wrapper(module, origin_dtype, fsdp=None):
    # `fsdp` defaults to detecting the sharding from the module itself, so pass it explicitly only when
    # installing on a not-yet-wrapped module that is going to be FSDP-sharded afterwards.
    if fsdp is None:
        fsdp = _is_fsdp_managed(module)
    if not fsdp:
        for name, module in module.named_modules():
            if name == "" or "embed_tokens" in name:
                continue
            original_forward = module.forward
            if hasattr(module, "weight") and module.weight is not None:
                setattr(module, "original_forward", original_forward)
                setattr(
                    module,
                    "forward",
                    lambda *inputs, m=module, **kwargs: autocast_model_forward(m, origin_dtype, *inputs, **kwargs)
                )
        return
    # Under FSDP the dequant must never rewrite `param.data` (the params are flat-storage views), so replace
    # only the forwards of the module types the DiT blocks quantize (`nn.Linear` and RMSNorm) and read
    # `self.weight` / the scale buffer at call time, which stays valid inside the FSDP forward while the
    # unit's parameters are unsharded.
    for _, child in module.named_modules():
        if isinstance(child, nn.Linear) and child.weight.dtype == FLOAT8_DTYPE:
            child.original_forward = child.forward
            child.forward = (
                lambda *inputs, m=child, **kwargs: _fsdp_dequant_linear_forward(m, origin_dtype, *inputs, **kwargs)
            )
        elif hasattr(child, "normalized_shape") and getattr(child, "weight", None) is not None \
                and child.weight.dtype == FLOAT8_DTYPE:
            child.original_forward = child.forward
            child.forward = (
                lambda *inputs, m=child, **kwargs: _fsdp_dequant_rmsnorm_forward(m, origin_dtype, *inputs, **kwargs)
            )

def undo_convert_weight_dtype_wrapper(module):
    for name, module in module.named_modules():
        if hasattr(module, "original_forward") and module.weight is not None:
            setattr(module, "forward", module.original_forward)
            delattr(module, "original_forward")


def _fsdp_dequant_linear_forward(module, origin_dtype, *inputs, **kwargs):
    # Non-mutating dequant for FSDP-sharded models: the scale-aware storage holds `w / scale` in fp8 and the
    # dequant must never rewrite `param.data` (FSDP flat-storage views), so the per-row scale is applied on
    # the output side instead — the scale indexes the output channels of the matmul, which makes
    # `(w / scale).to(dtype) @ x * scale` equivalent to dequantizing the weight first.
    weight = module.weight
    scale = getattr(module, _float8_scale_name("weight"), None)
    inputs = [input.to(origin_dtype) if torch.is_tensor(input) else input for input in inputs]
    out = F.linear(inputs[0], weight.to(origin_dtype))
    if scale is not None:
        # The per-row scale is stored as `(out_features, 1...)`; flatten it so it broadcasts over the output's
        # last (channel) dim regardless of the leading batch / sequence dims.
        out = out * scale.flatten().to(out.device, out.dtype)
    if module.bias is not None:
        bias = module.bias.to(origin_dtype)
        bias_scale = getattr(module, _float8_scale_name("bias"), None)
        if bias_scale is not None:
            bias = bias * bias_scale.to(bias.device, bias.dtype)
        out = out + bias
    return out

def _fsdp_dequant_rmsnorm_forward(module, origin_dtype, *inputs, **kwargs):
    # RMSNorm has no bias or additive term, so `norm(x) * (w / scale) * scale` folds the scale out exactly.
    hidden_states = inputs[0]
    weight = module.weight.to(origin_dtype)
    scale = getattr(module, _float8_scale_name("weight"), None)
    if scale is not None:
        weight = weight * scale.to(weight.device, weight.dtype)
    return F.rms_norm(hidden_states, module.normalized_shape, weight, module.eps)
