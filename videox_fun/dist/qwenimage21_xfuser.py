from typing import Any, List, Optional, Tuple

import torch

from .fuser import ulysses_all_to_all


class QwenImage21MultiGPUsAttnProcessor:
    r"""Ulysses (head-parallel) sequence-parallel processor for Qwen-Image 2.1.

    Outside attention the joint sequence is split across GPUs. This processor runs an all-to-all to swap
    q/k/v into a head-split layout (full sequence, a subset of heads), so the exact block-causal multi-pass
    prefill and the prefix KV cache run unchanged, then swaps the output back. Only Ulysses is supported
    (``ring_degree`` must be 1): ring attention rotates KV chunks and cannot express the block-causal mask
    or the prefix cache. ``attention_mask`` / ``segments`` / ``key_valid`` are full-sequence and are applied
    after the all-to-all, when every rank again sees the whole sequence.
    """

    def __call__(
        self,
        attn: "QwenImage21Attention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[Any] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        layer_cache: Optional["QwenImage21KVLayerCache"] = None,
        kv_cache_mode: Optional[str] = None,
        cache_write_slice: Optional[slice] = None,
        segments: Optional[List[Tuple[int, int, bool]]] = None,
        key_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Imported lazily: `models.qwenimage21_transformer2d` imports this processor from `..dist` at module
        # load, and these helpers are only needed at call time, so a top-level import would be circular. The
        # helpers stay in the model file because the single-GPU `QwenImage21AttnProcessor` shares them.
        from ..models.qwenimage21_transformer2d import (
            _qwenimage21_apply_cache,
            _qwenimage21_block_causal_attention,
            _qwenimage21_project_qkv,
        )

        # Sequence-split slice -> q/k/v [B, S_local, H, D].
        query, key, value = _qwenimage21_project_qkv(attn, hidden_states, rotary_emb)

        # Ulysses all-to-all: [B, S_local, H, D] -> [B, S_full, H/P, D].
        query = ulysses_all_to_all(query, 2, 1)
        key = ulysses_all_to_all(key, 2, 1)
        value = ulysses_all_to_all(value, 2, 1)

        # Prefix cache lives in the head-split layout (full-sequence prefix, subset of heads).
        key, value = _qwenimage21_apply_cache(key, value, layer_cache, kv_cache_mode, cache_write_slice)
        seq_len_q = query.shape[1]

        # Exact block-causal attention over the full sequence with this rank's heads.
        hidden_states = _qwenimage21_block_causal_attention(
            query, key, value, seq_len_q, attention_mask, segments, key_valid
        )

        # Ulysses all-to-all back: [B, S_full_q, H/P, D] -> [B, S_local_q, H, D].
        hidden_states = ulysses_all_to_all(hidden_states, 1, 2)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)
