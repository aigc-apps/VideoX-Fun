# FSDP-aware EMA shadow copies shared by the training scripts.
"""FSDP-aware EMA shadows: the full-module `FSDPEMA` and the LoRA-only `LORAFSDPEMA`."""
import os

import torch

__all__ = ["FSDPEMA", "LORAFSDPEMA"]


class FSDPEMA:
    r"""
    FSDP1-compatible drop-in replacement for `diffusers.training_utils.EMAModel`.

    The EMA copy is a second instance of the same module, wrapped with the same
    FSDP kwargs that `accelerator.prepare()` applies to the live model (the
    FSDP1 branch of `Accelerator.prepare_model`). Both copies therefore flatten
    and shard their parameters identically, which means every *local* shard of
    the EMA pairs 1:1 with the matching local shard of the live model. The
    polyak update runs purely on those local shards -- no communication, so the
    per-step cost is the same as the single-GPU case.

    The shadow copy is kept in fp32 on purpose: with `decay=0.9999` the update
    term `(1 - decay) * delta` has a magnitude of ~1e-8, which underflows in
    bf16 and would freeze the EMA at its initial value.

    The public surface matches `EMAModel` closely enough for the training
    loops: `step`, `store`, `copy_to`, `restore`, `to`, `save_pretrained` and
    `load_pretrained`. Only a fixed `decay` is supported. The class must be
    instantiated *before* the live model goes through `accelerator.prepare()`
    (i.e. while it is still unwrapped), which is the order used by the training
    scripts.

    Checkpointing uses `FULL_STATE_DICT`: rank 0 offloads the whole fp32 state
    dict to CPU and writes `diffusion_pytorch_model.safetensors`, while every
    rank reads that file back on resume and lets FSDP scatter it into local
    shards (the same trade-off accelerate itself makes for model checkpoints,
    at the cost of one full fp32 copy in host RAM per rank).
    """

    def __init__(self, module, source, accelerator, fsdp_plugin, decay=0.9999):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        self.accelerator = accelerator
        self.decay = decay
        self.optimization_step = 0
        self.temp_stored_params = None
        # Kept for the per-step buffer mirroring below; buffers are never
        # sharded, so this reference stays valid after FSDP wrapping.
        self._source_module = source

        # Initialize the fp32 shadow copy from the live weights so the two
        # copies start aligned (the live model may already carry a finetuned
        # checkpoint at this point).
        module = module.to(torch.float32)
        module.requires_grad_(False)
        with torch.no_grad():
            module.load_state_dict(source.state_dict(), strict=True)

        # Wrap with the same kwargs as the live model goes through in
        # `accelerator.prepare_model`, minus the mixed-precision policy: the
        # EMA never runs forward/backward, and a `param_dtype` cast would
        # quantize the shadow copy away from fp32 (see the note in the class
        # docstring), so no casting is wanted here.
        fsdp_plugin.set_auto_wrap_policy(module)
        self.module = FSDP(
            module,
            sharding_strategy=fsdp_plugin.sharding_strategy or fsdp_plugin.reshard_after_forward,
            cpu_offload=fsdp_plugin.cpu_offload,
            auto_wrap_policy=fsdp_plugin.auto_wrap_policy,
            mixed_precision=None,
            sync_module_states=fsdp_plugin.sync_module_states,
            backward_prefetch=fsdp_plugin.backward_prefetch,
            forward_prefetch=fsdp_plugin.forward_prefetch,
            use_orig_params=fsdp_plugin.use_orig_params,
            ignored_modules=fsdp_plugin.ignored_modules,
            limit_all_gathers=fsdp_plugin.limit_all_gathers,
            device_id=accelerator.device,
        )
        self.module.eval()

    def to(self, *args, **kwargs):
        # The sharded copy already lives on its own device (`device_id` above);
        # kept for interface parity with `EMAModel`.
        return self

    def _paired_params(self, parameters):
        source_params = list(parameters)
        shadow_params = list(self.module.parameters())
        if len(source_params) != len(shadow_params):
            raise RuntimeError(
                f"FSDP layout mismatch between the live model ({len(source_params)} params) "
                f"and the EMA copy ({len(shadow_params)} params); the EMA must be wrapped "
                "with the same FSDP kwargs as the live model."
            )
        return source_params, shadow_params

    @torch.no_grad()
    def step(self, parameters):
        source_params, shadow_params = self._paired_params(parameters)
        for p_shadow, p_source in zip(shadow_params, source_params):
            # `use_orig_params=True` leaves 0-numel views on ranks that do not
            # own a slice of a parameter.
            if p_shadow.numel() == 0:
                continue
            source = p_source.detach().to(device=p_shadow.device, dtype=p_shadow.dtype)
            if p_shadow.dtype.is_floating_point:
                p_shadow.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                p_shadow.copy_(source)
        # Buffers are never sharded by FSDP; mirror them directly. The Wan
        # backbones register no buffers today, kept for parity with the CCD
        # trainer's shard-wise polyak.
        for b_shadow, b_source in zip(self.module.buffers(), self._source_module.buffers()):
            b_shadow.copy_(b_source.to(device=b_shadow.device, dtype=b_shadow.dtype))
        self.optimization_step += 1

    @torch.no_grad()
    def store(self, parameters):
        # Snapshots the live local shards so `copy_to` can be undone with
        # `restore` (used to swap the EMA weights in for validation).
        self.temp_stored_params = [p.detach().clone() for p in parameters]

    @torch.no_grad()
    def copy_to(self, parameters):
        source_params, shadow_params = self._paired_params(parameters)
        for p_source, p_shadow in zip(source_params, shadow_params):
            if p_source.numel() == 0:
                continue
            p_source.data.copy_(p_shadow.data)

    @torch.no_grad()
    def restore(self, parameters):
        if self.temp_stored_params is None:
            raise RuntimeError("`restore` called without a matching `store`.")
        for p_source, p_stored in zip(parameters, self.temp_stored_params):
            if p_source.numel() == 0:
                continue
            p_source.data.copy_(p_stored)
        self.temp_stored_params = None

    def save_pretrained(self, save_directory):
        from safetensors.torch import save_file
        from torch.distributed.fsdp import FullStateDictConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        # Every rank has to join the all-gather; only rank 0 receives the full
        # state dict (`rank0_only=True`) offloaded to CPU.
        with FSDP.state_dict_type(
            self.module,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = self.module.state_dict()
        if self.accelerator.is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            # Keep fp32 on disk: the EMA target must not round-trip through
            # bf16 (see the fp32 note in the class docstring).
            state_dict = {k: v.detach().contiguous() for k, v in state_dict.items()}
            save_file(
                state_dict,
                os.path.join(save_directory, "diffusion_pytorch_model.safetensors"),
                metadata={"format": "pt"},
            )
            self.module.module.save_config(save_directory)
            print(f"Saved EMA weights to {save_directory}.")
        del state_dict

    def load_pretrained(self, load_directory):
        from safetensors.torch import load_file
        from torch.distributed.fsdp import FullStateDictConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        ema_path = os.path.join(load_directory, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(ema_path):
            return
        # Every rank feeds the full state dict; under the FULL_STATE_DICT
        # context FSDP scatters it into that rank's local shards.
        state_dict = load_file(ema_path)
        with FSDP.state_dict_type(
            self.module,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        ):
            self.module.load_state_dict(state_dict, strict=True)
        del state_dict
        if self.accelerator.is_main_process:
            print(f"Loaded EMA weights from {load_directory}.")


class LORAFSDPEMA:
    r"""A polyak average of the generator's LoRA weights (`--ema_decay`, the official 0.99).

    The full-parameter `FSDPEMA` above cannot be kept here: a second 33 B copy is a third of the
    training memory, and under FSDP a second full model does not exist at all. Only the LoRA weights
    are averaged, one local shard per rank — the frozen base weights are never touched, so `step`
    needs no communication and stays valid under FSDP. The full-precision export happens in the save
    hook: the shadow is copied back into the parameters, rides the same all-gather as the training
    weights and is restored afterwards.

    The shadow is built from the `(name, parameter)` pairs *as `accelerator.prepare` left them*:
    plain tensors in the single-process and DeepSpeed ZeRO-2 runs, rank partitions under ZeRO-3,
    flat-parameter views under FSDP1 and `DTensor` shards under FSDP2. Each shadow therefore pairs
    1:1 with its local parameter, every method runs purely on those local shards, and a rank that
    owns no slice of a parameter (the 0-numel views `use_orig_params=True` leaves) skips it. The
    shards are disjoint slices of the same tensors, so the copies stay in sync without any
    all-gather. The shadow is kept in fp32 on purpose: the update term `(1 - decay) * delta`
    shrinks with the training rate down to ~1e-8, which underflows in bf16 and would freeze the
    EMA at its initial value.

    `store`/`copy_to`/`restore` swap the shadow in for a validation rollout; `save_shards`/
    `load_shards` keep the rank-local shadow next to the `accelerator.save_state` checkpoints, so
    a resumed run continues averaging from where it stopped. Must be built *after*
    `accelerator.prepare` (from the unwrapped module), so the shadow adopts the sharded layout
    instead of a full per-rank copy of every parameter.
    """

    def __init__(self, named_parameters, decay):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().to(torch.float32).clone()
            for name, param in named_parameters
        }
        if not self.shadow:
            raise ValueError("LORAFSDPEMA was handed no parameters.")
        self._backup = None

    @torch.no_grad()
    def step(self, named_parameters):
        for name, param in named_parameters:
            shadow = self.shadow[name]
            # `use_orig_params=True` can leave 0-numel views on ranks that own no slice of a parameter.
            if shadow.numel() == 0:
                continue
            # Outside forward/backward the shards are the fp32 master weights, so a per-shard polyak
            # update is bit-identical to a full-tensor update.
            source = param.detach().to(device=shadow.device, dtype=shadow.dtype)
            if shadow.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                shadow.copy_(source)

    @torch.no_grad()
    def copy_to(self, named_parameters):
        r"""Overwrite the live weights with the shadow (the validation swap and the EMA export)."""
        for name, param in named_parameters:
            if param.numel() == 0:
                continue
            param.detach().copy_(self.shadow[name])

    @torch.no_grad()
    def store(self, named_parameters):
        r"""Stash the live weights, for a `copy_to` that `restore` undoes (the validation swap)."""
        self._backup = {name: param.detach().clone() for name, param in named_parameters}

    @torch.no_grad()
    def restore(self, named_parameters):
        if self._backup is None:
            raise RuntimeError("`restore` called without a matching `store`.")
        for name, param in named_parameters:
            if param.numel() == 0:
                continue
            param.detach().copy_(self._backup[name])
        self._backup = None

    def save_shards(self, output_dir, rank):
        r"""Write the rank-local shadow shards next to the checkpoint.

        A `DTensor` shadow (FSDP2) is stored as its plain local tensor, so the file holds ordinary
        tensors that `load_shards` reads back without a process group.
        """
        state = {
            name: shard.to_local() if hasattr(shard, "to_local") else shard
            for name, shard in self.shadow.items()
        }
        torch.save(state, os.path.join(output_dir, f"lora_ema_shadow.rank{rank}.pt"))

    def load_shards(self, input_dir, rank):
        r"""Restore the rank-local shadow saved by `save_shards`; a missing file is not an error."""
        path = os.path.join(input_dir, f"lora_ema_shadow.rank{rank}.pt")
        if not os.path.exists(path):
            return False
        state = torch.load(path, map_location="cpu")
        for name, value in state.items():
            shadow = self.shadow.get(name)
            if shadow is None:
                continue
            target = shadow.to_local() if hasattr(shadow, "to_local") else shadow
            target.copy_(value.to(device=target.device, dtype=target.dtype))
        return True
