import torch
import torch.nn.functional as F

from .fuser import get_sp_group, ulysses_all_to_all
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
                "Multi-GPU inference needs the sequence-parallel attention of xfuser, which could not be "
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


class MiniMaxH3MultiGPUsVSAAttnProcessor:
    """Sequence-parallel attention processor for the FastVideo VSA student on MiniMax-H3.

    Mirrors ``MiniMaxH3MultiGPUsAttnProcessor``'s row-split -> head-split Ulysses scheme, but swaps
    the dense ``xFuserLongContextAttention`` core for the VSA block-sparse op (``attn.attn_op``) so
    the trained coarse-to-fine selection and the ``to_gate_compress`` compensation branch are kept
    instead of being silently dropped by a dense fallback.

    Two VSA-specific points under head-parallelism:
      * ``to_gate_compress`` is a ``Linear(dim, heads*head_dim)`` that MIXES heads, so it cannot run
        in the head-shard layout. It is strictly per-token and SP shards rows while keeping the full
        hidden dim, so the gate is evaluated here on this rank's ROW shard (yielding all heads), then
        all-to-all'd alongside q/k/v -- element-wise identical to computing it on the full sequence
        and slicing heads, and it lands on the same contiguous head block the all-to-all hands back.
      * the parent forward pads the packed sequence up to a multiple of the group size, so the
        gathered sequence carries ``<= sp_world_size - 1`` zero rows at its tail. VSA builds its tile
        geometry from ``vsa_segments``, whose lengths sum to the PADLESS ``valid_length``; the padding
        tail is therefore sliced off before the op and the output padded back afterwards. Unlike the
        dense processor (which lets the pad rows dilute the softmax by ~1e-5), this keeps VSA
        bitwise-equal to the single-GPU run -- the pad rows never enter the kernel.

    Requires a padless packed sequence (``attention_mask is None``) and a pure-Ulysses split
    (``num_heads`` divisible by the group size, checked by the model's ``enable_multi_gpus_inference``).
    A dense student (``topk >= 1.0``) carries no ``attn_op`` and must keep the dense processor.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb=None,
        attention_mask=None,
        valid_length=None,
        vsa_segments=None,
    ) -> torch.Tensor:
        from ..models.minimax_h3_transformer3d import \
            apply_minimax_h3_rotary_emb

        if attention_mask is not None:
            raise ValueError(
                "MiniMaxH3MultiGPUsVSAAttnProcessor does not support a masked (padded) packed sequence.")
        if not hasattr(attn, "attn_op"):
            raise RuntimeError(
                "The VSA sequence-parallel processor needs attn.attn_op; a dense student (topk>=1.0) "
                "must bind MiniMaxH3MultiGPUsAttnProcessor instead.")

        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # The rotary coordinates are this rank's rows, so apply them before the all-to-all moves rows.
        if rotary_emb is not None:
            query = apply_minimax_h3_rotary_emb(query, *rotary_emb)
            key = apply_minimax_h3_rotary_emb(key, *rotary_emb)

        # Cross-head compress gate on this rank's ROW shard (full hidden dim present): [B, chunk, H, D].
        gate = attn.attn_op.to_gate_compress(hidden_states).unflatten(-1, (attn.heads, -1))

        half_dtypes = (torch.float16, torch.bfloat16)

        def half(x):
            return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

        # row-split -> head-split over the whole (padded) packed sequence, gate alongside q/k/v.
        query = ulysses_all_to_all(half(query), scatter_dim=2, gather_dim=1)
        key = ulysses_all_to_all(half(key), scatter_dim=2, gather_dim=1)
        value = ulysses_all_to_all(half(value), scatter_dim=2, gather_dim=1)
        gate = ulysses_all_to_all(half(gate), scatter_dim=2, gather_dim=1)

        # Drop the SP padding tail so VSA sees exactly the padless length its segments describe.
        padded_length = query.shape[1]
        trim = valid_length is not None and valid_length < padded_length
        if trim:
            query = query[:, :valid_length]
            key = key[:, :valid_length]
            value = value[:, :valid_length]
            gate = gate[:, :valid_length]

        out = attn.attn_op(query, key, value, None, segments=vsa_segments, gate=gate)

        # Pad the output back to the gathered length before the inverse all-to-all; the zero rows are
        # dropped by the model's final all_gather and never touch a real row's attention.
        if trim:
            out = F.pad(out, (0, 0, 0, 0, 0, padded_length - valid_length))

        # head-split -> row-split, back to this rank's rows.
        out = ulysses_all_to_all(out.to(hidden_states.dtype), scatter_dim=1, gather_dim=2)
        out = out.flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out
