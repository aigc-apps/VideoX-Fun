import torch


class MiniMaxH3MultiGPUsAttnProcessor:
    """
    Sequence parallel attention processor for MiniMax-H3.

    MiniMax-H3 runs full self-attention over one packed sequence, so every rank holds a contiguous slice of the rows
    and all-gathers the keys and values before attending: queries stay local, keys and values cover the whole
    sequence, and the result is numerically identical to the single-GPU path.

    The packed sequence is padded up to a multiple of the group size before it is split, so the gathered keys and
    values carry the padding rows at their tail; `valid_length` is the unpadded length and the padding is sliced off
    again here.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb=None,
        attention_mask=None,
        valid_length=None,
    ) -> torch.Tensor:
        from ..models.attention_utils import attention
        from ..models.minimax_h3_transformer3d import apply_minimax_h3_rotary_emb

        if attention_mask is not None:
            raise ValueError("MiniMaxH3MultiGPUsAttnProcessor does not support a masked (padded) packed sequence.")

        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # The rotary coordinates are the ones of this rank's rows, so they are applied before the keys leave the rank.
        if rotary_emb is not None:
            query = apply_minimax_h3_rotary_emb(query, *rotary_emb)
            key = apply_minimax_h3_rotary_emb(key, *rotary_emb)

        key = attn.all_gather(key.contiguous(), dim=1)
        value = attn.all_gather(value.contiguous(), dim=1)
        if valid_length is not None:
            key = key[:, :valid_length]
            value = value[:, :valid_length]

        hidden_states = attention(query, key, value, causal=False)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
