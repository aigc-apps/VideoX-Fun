# Copied from https://github.com/Wan-Video/Wan2.1/blob/main/wan/distributed/fsdp.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (lambda_auto_wrap_policy,
                                         transformer_auto_wrap_policy)
from torch.distributed.utils import _free_storage


def find_classes_in_model(model, class_names):
    """
    Recursively find unique module classes in the model that match the given class names.
    
    Args:
        model: The PyTorch model to traverse.
        class_names: A list of class name strings to look for.
        
    Returns:
        A set of matched class types.
    """
    found_classes = set()
    class_names_set = set(class_names)
    
    def traverse(module):
        if module.__class__.__name__ in class_names_set:
            found_classes.add(module.__class__)
        for child in module.children():
            traverse(child)
    
    traverse(model)
    print(f"Found transformer classes: {found_classes}")
    return found_classes


def create_transformer_auto_wrap_policy(
    model,
    transformer_layer_cls_to_wrap,
):
    """
    Creates an auto wrap policy that only wraps modules belonging to the specified 
    transformer layer classes.
    
    Args:
        model: The PyTorch model to analyze for class types.
        transformer_layer_cls_to_wrap: A list of class name strings to wrap.
        
    Returns:
        A callable auto wrap policy function.
    """
    # Dynamically find the actual class types corresponding to the provided names
    transformer_classes = find_classes_in_model(model, transformer_layer_cls_to_wrap)
    
    if not transformer_classes:
        raise ValueError(
            f"No modules found with class names {transformer_layer_cls_to_wrap}. "
            "Please check the class names or the model structure."
        )
    
    def transformer_policy(module, recurse, unwrapped_params):
        # Use the standard transformer auto wrap policy with the discovered classes
        return transformer_auto_wrap_policy(
            module=module,
            recurse=recurse,
            unwrapped_params=unwrapped_params,
            transformer_layer_cls=transformer_classes,
        )
    
    return transformer_policy


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
    module_to_wrapper=None,
    transformer_layer_cls_to_wrap=None,
):  
    """
    Wraps the model with FSDP using the specified configuration.
    
    Args:
        model: The PyTorch model to shard.
        device_id: The CUDA device ID.
        param_dtype: Data type for parameters.
        reduce_dtype: Data type for gradient reduction.
        buffer_dtype: Data type for buffers.
        process_group: The process group for distributed training.
        sharding_strategy: The FSDP sharding strategy.
        sync_module_states: Whether to sync module states across ranks.
        module_to_wrapper: Specific modules to wrap if using lambda policy.
        transformer_layer_cls_to_wrap: List of class names to wrap using transformer policy.
        
    Returns:
        The FSDP-wrapped model.
    """
    if transformer_layer_cls_to_wrap is not None:
        # Create policy based strictly on transformer layer classes
        auto_wrap_policy = create_transformer_auto_wrap_policy(
            model=model,
            transformer_layer_cls_to_wrap=transformer_layer_cls_to_wrap,
        )
    else:
        # Fallback to lambda policy if no transformer classes are specified
        auto_wrap_policy = partial(
            lambda_auto_wrap_policy, 
            lambda_fn=lambda m: m in (model.blocks if module_to_wrapper is None else module_to_wrapper)
        )

    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states)
    
    return model


def free_model(model):
    """
    Frees memory associated with the FSDP model.
    
    Args:
        model: The FSDP-wrapped model to free.
    """
    for m in model.modules():
        if isinstance(m, FSDP):
            _free_storage(m._handle.flat_param.data)
    del model
    gc.collect()
    torch.cuda.empty_cache()