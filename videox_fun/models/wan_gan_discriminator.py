# Modified from https://github.com/Aurora-edu/One-Forcing/blob/main/wan/modules/model.py
r"""One-Forcing GAN discriminator head for the Wan fake score.

The discriminator shares the fake-score backbone: one learnable register token per selected block cross-attends to
that block's hidden states, and the concatenated tokens go through a small classification branch to a single logit.
[`add_wan_gan_discriminator`] attaches the head as the `gan_head` submodule of a [`WanTransformer3DModel`] and wraps its
forward, so calling the model with `gan_classify=True` returns `(flow_pred, logits)`. The head runs inside the forward,
so FSDP shards and trains it together with the backbone.
"""

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_transformer3d import WanLayerNorm, WanRMSNorm


class GanAttentionBlock(nn.Module):
    """Register-token cross-attention block of the One-Forcing discriminator head."""

    def __init__(self, dim, ffn_dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

    def forward(self, x, token):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        x = self.norm3(x)
        q = self.norm_q(self.q(token)).view(b, -1, n, d).transpose(1, 2)
        k = self.norm_k(self.k(x)).view(b, -1, n, d).transpose(1, 2)
        v = self.v(x).view(b, -1, n, d).transpose(1, 2)
        token = token + self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))
        return token + self.ffn(self.norm2(token))


class GanDiscriminatorHead(nn.Module):
    """One-Forcing discriminator head: one register token per fake_score feature layer."""

    def __init__(self, dim, num_layers, ffn_dim, num_heads):
        super().__init__()
        self.register_tokens = nn.Parameter(torch.randn(num_layers, dim) * 0.02)
        self.register_norm = WanRMSNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList([GanAttentionBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.cls_pred_branch = nn.Sequential(
            nn.LayerNorm(dim * num_layers), nn.Linear(dim * num_layers, dim), nn.SiLU(), nn.Linear(dim, 1)
        )

    def forward(self, features):
        registers = self.register_norm(self.register_tokens)
        tokens = []
        for i, (block, feature) in enumerate(zip(self.blocks, features)):
            token = registers[i:i + 1].unsqueeze(0).expand(feature.size(0), -1, -1)
            # Checkpoint each cross-attention block so its q/k/v activations are recomputed during
            # backward instead of being retained; one register token keeps the recompute cost tiny.
            if torch.is_grad_enabled() and self.training:
                token = torch.utils.checkpoint.checkpoint(block, feature, token, use_reentrant=False)
            else:
                token = block(feature, token)
            tokens.append(token)
        return self.cls_pred_branch(torch.cat(tokens, dim=1).flatten(1))


def add_wan_gan_discriminator(transformer, feature_layers, ffn_dim, num_heads, dtype, seed=0):
    """
    Attach the One-Forcing discriminator head to fake_score before accelerator.prepare.
    Calling the model with gan_classify=True returns (flow_pred, logits); the head runs
    inside the wrapped forward so it is sharded and trained together with fake_score.
    """
    feature_layers = sorted(set(feature_layers))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        transformer.gan_head = GanDiscriminatorHead(transformer.dim, len(feature_layers), ffn_dim, num_heads)
    transformer.gan_head.to(device=next(transformer.parameters()).device, dtype=dtype)
    transformer.gan_head.requires_grad_(True)
    transformer._gan_features = None

    def capture_feature(module, inputs, output):
        if transformer._gan_features is not None:
            transformer._gan_features.append(output)

    for layer in feature_layers:
        transformer.blocks[layer].register_forward_hook(capture_feature)

    def forward(self, *args, gan_classify=False, **kwargs):
        if not gan_classify:
            return type(self).forward(self, *args, **kwargs)
        self._gan_features = []
        try:
            flow = type(self).forward(self, *args, **kwargs)
            features = self._gan_features
        finally:
            self._gan_features = None
        num_tokens = math.prod(flow.shape[2:]) // math.prod(self.patch_size)
        return flow, self.gan_head([feature[:, :num_tokens] for feature in features])

    transformer.forward = types.MethodType(forward, transformer)
