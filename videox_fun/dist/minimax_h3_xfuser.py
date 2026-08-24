import torch

from .fuser import xFuserLongContextAttention


class MiniMaxH3MultiGPUsAttnProcessor:
    """
    Sequence parallel attention processor for MiniMax-H3.

    MiniMax-H3 runs full self-attention over one packed sequence, so every rank holds a contiguous slice of the rows
    and the sequence-parallel group turns that row split into a head split: `xFuserLongContextAttention` all-to-alls
    the queries, keys and values so that every rank attends over the whole sequence with a subset of the heads, then
    all-to-alls the context back onto the rank's own rows. This is the path every other model of the repository takes
    (Wan, HunyuanVideo, QwenImage, ...), and it keeps the local attention square — `q_len == k_len` — so the very
    same kernel serves the single-GPU and the multi-GPU runs.

    The packed sequence is padded up to a multiple of the group size before it is split, so the sequence the heads
    attend over carries at most `sp_world_size - 1` padding rows. Their projections are zero, which leaves them as
    `exp(0)` terms of a softmax over ~1e5 rows — a ~1e-5 relative dilution. `valid_length` is therefore only accepted
    for signature compatibility with the single-GPU processor and is not used.
    """

    def __init__(self):
        if xFuserLongContextAttention is not None:
            try:
                self.hybrid_seq_parallel_attn = xFuserLongContextAttention()
            except Exception:
                self.hybrid_seq_parallel_attn = None
        else:
            self.hybrid_seq_parallel_attn = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb=None,
        attention_mask=None,
        valid_length=None,
    ) -> torch.Tensor:
        from ..models.minimax_h3_transformer3d import \
            apply_minimax_h3_rotary_emb

        if attention_mask is not None:
            raise ValueError("MiniMaxH3MultiGPUsAttnProcessor does not support a masked (padded) packed sequence.")
        if self.hybrid_seq_parallel_attn is None:
            raise RuntimeError(
                "Multi-GPU inference needs the sequence-parallel attention of xfuser / paifuser, which could not be "
                "instantiated."
            )

        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # The rotary coordinates are the ones of this rank's rows, so they are applied before the all-to-all moves the
        # rows around.
        if rotary_emb is not None:
            query = apply_minimax_h3_rotary_emb(query, *rotary_emb)
            key = apply_minimax_h3_rotary_emb(key, *rotary_emb)

        # The flash-attention backend behind the all-to-all only takes half precision; the single-GPU path casts the
        # same way inside `attention()`.
        half_dtypes = (torch.float16, torch.bfloat16)

        def half(x):
            return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

        hidden_states = self.hybrid_seq_parallel_attn(
            None,
            half(query),
            half(key),
            half(value),
            dropout_p=0.0,
            causal=False,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
