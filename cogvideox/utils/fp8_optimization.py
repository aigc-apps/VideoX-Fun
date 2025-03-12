"""Modified from https://github.com/kijai/ComfyUI-MochiWrapper
"""
import torch
import torch.nn as nn

def autocast_model_forward(cls, origin_dtype, *inputs, **kwargs):
    weight_dtype = cls.weight.dtype
    cls.to(origin_dtype)

    # Convert all inputs to the original dtype
    inputs = [input.to(origin_dtype) for input in inputs]
    out = cls.original_forward(*inputs, **kwargs)

    cls.to(weight_dtype)
    return out

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

def convert_model_weight_to_float8(model, exclude_module_name=['embed_tokens']):
    for name, module in model.named_modules():
        flag = False
        for _exclude_module_name in exclude_module_name:
            if _exclude_module_name in name:
                flag = True
        if flag:
            continue
        for param_name, param in module.named_parameters():
            flag = False
            for _exclude_module_name in exclude_module_name:
                if _exclude_module_name in param_name:
                    flag = True
            if flag:
                continue
            param.data = param.data.to(torch.float8_e4m3fn)

def convert_weight_dtype_wrapper(module, origin_dtype):
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
