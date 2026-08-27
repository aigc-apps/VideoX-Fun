# Ported from https://github.com/madebyollin/taehv
# Tiny AutoEncoder for Wan 2.1 (taew2_1, 16ch latent) and Wan 2.2 5B (taew2_2, 48ch latent).
# TAE shares the exact latent space of the full-size Wan VAEs, so it can be used
# as a cheap drop-in encoder/decoder (e.g. for live previewing or low-memory decoding)
# at the cost of slightly lower reconstruction quality.
from typing import Tuple, Union

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.autoencoders.vae import (DecoderOutput,
                                               DiagonalGaussianDistribution)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from tqdm.auto import tqdm

# Deterministic-logvar used to wrap the (non-stochastic) TAE encoder output
# into a DiagonalGaussianDistribution compatible with the full Wan VAE interface.
DETERMINISTIC_LOGVAR = -30.0

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class SuperMemBlock(nn.Module):
    """MemBlock variant used by the Super decoder (ConvNeXt-style: 7x7 depthwise conv + inverted bottleneck)."""
    def __init__(self, n_f):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_f * 2, n_f * 2, 7, padding=3, groups=n_f * 2, bias=False),
            nn.Conv2d(n_f * 2, n_f * 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(n_f * 4, n_f, 1, bias=False),
        )

    def forward(self, x, past):
        return self.conv(torch.cat([x, past], 1)) + x


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


def apply_model_with_memblocks_parallel(model, x, show_progress_bar):
    """Apply a sequential model with memblocks over the time axis in parallel.

    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAE operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    x = x.reshape(N * T, C, H, W)

    # parallel over input timesteps, iterate over blocks
    for b in tqdm(model, disable=not show_progress_bar):
        if isinstance(b, (MemBlock, SuperMemBlock)):
            NT, C, H, W = x.shape
            T = NT // N
            _x = x.reshape(N, T, C, H, W)
            # pad with zeros along time axis (i.e. empty memory), slice
            block_memory = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
            x = b(x, block_memory)
        else:
            x = b(x)
    NT, C, H, W = x.shape
    T = NT // N
    return x.view(N, T, C, H, W)


def apply_model_with_memblocks_sequential_single_step(model, memory, work_queue, progress_bar=None):
    """Process the work queue (a graph traversal over blocks and timesteps)
    until an output frame is produced or the queue is empty.
    Mutates memory and work_queue in place.

    Returns N1CHW output tensor, or None if the queue needs more input.
    """
    while work_queue:
        xt, i = work_queue.pop(0)
        if progress_bar is not None and i == 0:
            progress_bar.update(1)
        if i == len(model):
            return xt.unsqueeze(1)
        b = model[i]
        if isinstance(b, (MemBlock, SuperMemBlock)):
            # mem blocks are simple since we're visiting the graph in causal order
            if memory[i] is None:
                xt_new = b(xt, xt * 0)
            else:
                xt_new = b(xt, memory[i])
            memory[i] = xt
            work_queue.insert(0, TWorkItem(xt_new, i + 1))
        elif isinstance(b, TPool):
            # pool blocks accumulate inputs until they have enough to pool
            if memory[i] is None:
                memory[i] = []
            memory[i].append(xt)
            if len(memory[i]) > b.stride:
                raise ValueError(f"TPool memory overflow: {len(memory[i])} items for stride {b.stride}")
            elif len(memory[i]) == b.stride:
                N, C, H, W = xt.shape
                xt = b(torch.cat(memory[i], 1).view(N * b.stride, C, H, W))
                memory[i] = []
                work_queue.insert(0, TWorkItem(xt, i + 1))
        elif isinstance(b, TGrow):
            xt = b(xt)
            NT, C, H, W = xt.shape
            for xt_next in reversed(xt.view(NT // b.stride, b.stride * C, H, W).chunk(b.stride, 1)):
                work_queue.insert(0, TWorkItem(xt_next, i + 1))
        else:
            xt = b(xt)
            work_queue.insert(0, TWorkItem(xt, i + 1))
    return None


def apply_model_with_memblocks_sequential(model, x, show_progress_bar):
    """Apply a sequential model with memblocks, iterating over timesteps as
    well as blocks (slow but uses O(1) memory w.r.t. sequence length).

    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAE operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    work_queue = [TWorkItem(xt, 0) for xt in x.unbind(1)]
    memory = [None] * len(model)
    progress_bar = tqdm(range(len(work_queue)), disable=not show_progress_bar)
    out = []
    while work_queue:
        xt = apply_model_with_memblocks_sequential_single_step(model, memory, work_queue, progress_bar)
        if xt is not None:
            out.append(xt)
    progress_bar.close()
    return torch.cat(out, 1)


def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    """Apply a sequential model with memblocks to the given input.

    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - parallel: if True, parallelize over timesteps (fast but uses O(T) memory)
        if False, each timestep will be processed sequentially (slow but uses O(1) memory)
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    if parallel:
        return apply_model_with_memblocks_parallel(model, x, show_progress_bar)
    else:
        return apply_model_with_memblocks_sequential(model, x, show_progress_bar)


class TAEWan(nn.Module):
    """Tiny AutoEncoder core for the Wan family (Wan 2.1 16ch / Wan 2.2 48ch latents).

    Operates on NTCHW tensors with pixel values in [0, 1] and unnormalized
    (~Gaussian) latents, matching the reference taehv implementation.
    """

    def __init__(self, encoder_time_downscale=(True, True, False), decoder_time_upscale=(False, True, True), decoder_space_upscale=(True, True, True), patch_size=1, latent_channels=16, arch_variant=None):
        """
        Args:
            encoder_time_downscale: whether temporal downsampling is enabled for each block.
            decoder_time_upscale: whether temporal upsampling is enabled for each block. upsampling can be disabled for a cheaper preview.
            decoder_space_upscale: whether spatial upsampling is enabled for each block. upsampling can be disabled for a cheaper preview.
            patch_size: input/output pixelshuffle patch-size for this model.
            latent_channels: number of latent channels (z dim) for this model.
            arch_variant: decoder architecture variant. None (base) or "super" (higher-quality, ~2x decoder params).
        """
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        if len(decoder_time_upscale) == 2:
            decoder_time_upscale = (False, *decoder_time_upscale)
        assert arch_variant in (None, "super"), f"unrecognized arch_variant {arch_variant!r}"
        self.encoder = nn.Sequential(
            conv(self.image_channels * self.patch_size ** 2, 64), nn.ReLU(inplace=True),
            TPool(64, 2 if encoder_time_downscale[0] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[1] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[2] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, self.latent_channels),
        )
        if arch_variant == "super":
            n_f = [512, 256, 128, 64]
            self.decoder = nn.Sequential(
                nn.Conv2d(self.latent_channels, n_f[0], 1, bias=False),
                SuperMemBlock(n_f[0]), SuperMemBlock(n_f[0]), SuperMemBlock(n_f[0]), conv(n_f[0], n_f[1] * (2 if decoder_space_upscale[0] else 1) ** 2), nn.ReLU(inplace=True), nn.PixelShuffle(2 if decoder_space_upscale[0] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
                SuperMemBlock(n_f[1]), SuperMemBlock(n_f[1]), SuperMemBlock(n_f[1]), conv(n_f[1], n_f[2] * (2 if decoder_space_upscale[1] else 1) ** 2), nn.ReLU(inplace=True), nn.PixelShuffle(2 if decoder_space_upscale[1] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
                SuperMemBlock(n_f[2]), SuperMemBlock(n_f[2]), SuperMemBlock(n_f[2]), conv(n_f[2], n_f[3] * (2 if decoder_space_upscale[2] else 1) ** 2), nn.ReLU(inplace=True), nn.PixelShuffle(2 if decoder_space_upscale[2] else 1), TGrow(n_f[3], 2 if decoder_time_upscale[2] else 1),
                conv(n_f[3], self.image_channels * self.patch_size ** 2),
            )
        else:
            n_f = [256, 128, 64, 64]
            self.decoder = nn.Sequential(
                Clamp(), conv(self.latent_channels, n_f[0]), nn.ReLU(inplace=True),
                MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1), conv(n_f[0], n_f[1], bias=False),
                MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1), conv(n_f[1], n_f[2], bias=False),
                MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1), conv(n_f[2], n_f[3], bias=False),
                nn.ReLU(inplace=True), conv(n_f[3], self.image_channels * self.patch_size ** 2),
            )
        # computed properties
        self.t_downscale = 2 ** sum(t.stride == 2 for t in self.encoder if isinstance(t, TPool))
        self.t_upscale = 2 ** sum(t.stride == 2 for t in self.decoder if isinstance(t, TGrow))
        self.frames_to_trim = self.t_upscale - 1

    def patch_tgrow_layers(self, sd):
        """Patch TGrow layers to use a smaller kernel if temporal upscaling
        was partially disabled relative to the checkpoint.

        Args:
            sd: state dict to patch
        """
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if key in sd and sd[key].shape[0] > new_sd[key].shape[0]:
                    # take the last-timestep output channels
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def preprocess_input_frames(self, x):
        """Preprocess RGB input frames prior to the main encoder sequence."""
        if self.patch_size > 1:
            x = F.pixel_unshuffle(x, self.patch_size)
        return x

    def postprocess_output_frames(self, x):
        """Postprocess RGB frames after the main decoder sequence."""
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        return x.clamp_(0, 1)

    def encode_video(self, x, parallel=True, show_progress_bar=False):
        """Encode a sequence of frames.

        Args:
            x: input NTCHW RGB (C=3) tensor with values in [0, 1].
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW latent tensor with ~Gaussian values.
        """
        x = self.preprocess_input_frames(x)
        if x.shape[1] % self.t_downscale != 0:
            # pad at end to multiple of self.t_downscale
            n_pad = self.t_downscale - x.shape[1] % self.t_downscale
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def decode_video(self, x, parallel=True, show_progress_bar=False):
        """Decode a sequence of frames.

        Args:
            x: input NTCHW latent (C=self.latent_channels) tensor with ~Gaussian values.
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW RGB tensor with ~[0, 1] values.
        """
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        x = self.postprocess_output_frames(x)
        return x[:, self.frames_to_trim:]


class AutoencoderTinyWan(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """VideoX-Fun-compatible wrapper around TAEWan.

    Implements the same encode/decode protocol as AutoencoderKLWan /
    AutoencoderKLWan3_8: encode/decode take BCTHW tensors in [-1, 1] and
    latents are normalized with the per-channel mean/std of the corresponding
    full-size Wan VAE, so TAE latents are interchangeable with the real ones.

    Weight files (from https://github.com/madebyollin/taehv):
    - taew2_1*.pth / .safetensors: Wan 2.1 & Wan 2.2 14B (16ch latent)
    - taew2_2*.pth / .safetensors: Wan 2.2 5B (48ch latent, patch_size=2)
    """

    # per-channel statistics of the full-size Wan VAE latent spaces
    WAN2_1_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                   0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    WAN2_1_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                  3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
    WAN2_2_MEAN = [
        -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
        -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
        -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
        -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
        -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
        0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667]
    WAN2_2_STD = [
        0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
        0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
        0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
        0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
        0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
        0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744]

    @register_to_config
    def __init__(
        self,
        latent_channels=16,
        patch_size=1,
        arch_variant=None,
        encoder_time_downscale=(True, True, False),
        decoder_time_upscale=(False, True, True),
        decoder_space_upscale=(True, True, True),
        temporal_compression_ratio=4,
        spatial_compression_ratio=None,
        parallel=True,
    ):
        super().__init__()
        if latent_channels == 16:
            mean, std = self.WAN2_1_MEAN, self.WAN2_1_STD
        elif latent_channels == 48:
            mean, std = self.WAN2_2_MEAN, self.WAN2_2_STD
        else:
            raise ValueError(f"AutoencoderTinyWan supports 16ch (taew2_1) or 48ch (taew2_2) latents, got {latent_channels}")
        assert len(mean) == latent_channels and len(std) == latent_channels
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.scale = [self.mean, 1.0 / self.std]

        # spatial compression is 8 from convs times the pixel-shuffle patch size
        if spatial_compression_ratio is None:
            spatial_compression_ratio = 8 * patch_size
        self.spatial_compression_ratio = spatial_compression_ratio

        self.model = TAEWan(
            encoder_time_downscale=tuple(encoder_time_downscale),
            decoder_time_upscale=tuple(decoder_time_upscale),
            decoder_space_upscale=tuple(decoder_space_upscale),
            patch_size=patch_size,
            latent_channels=latent_channels,
            arch_variant=arch_variant,
        )
        self.parallel = parallel
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, *args, **kwargs):
        # TAE has no gradient checkpointing path; accept the flag for API parity.
        if "value" in kwargs:
            self.gradient_checkpointing = kwargs["value"]
        elif "enable" in kwargs:
            self.gradient_checkpointing = kwargs["enable"]
        else:
            raise ValueError("Invalid set gradient checkpointing")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: BCTHW in [-1, 1] -> normalized latents BCLHW
        scale = [item.to(x.device, x.dtype) for item in self.scale]
        x = (x.add(1).div_(2)).clamp_(0, 1)     # [-1, 1] -> [0, 1]
        x = x.permute(0, 2, 1, 3, 4)            # BCTHW -> NTCHW
        zs = []
        for u in x:
            z = self.model.encode_video(u.unsqueeze(0), parallel=self.parallel).squeeze(0)
            z = (z - scale[0].view(1, -1, 1, 1)) * scale[1].view(1, -1, 1, 1)
            zs.append(z.permute(1, 0, 2, 3))    # TCHW -> CTHW
        return torch.stack(zs)

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        h = self._encode(x)
        # TAE encoder is deterministic; wrap the output in a degenerate
        # gaussian (near-zero variance) for interface compatibility.
        logvar = torch.full_like(h, DETERMINISTIC_LOGVAR)
        posterior = DiagonalGaussianDistribution(torch.cat([h, logvar], dim=1))

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, zs):
        # zs: BCLHW normalized latents -> BCTHW in [-1, 1]
        scale = [item.to(zs.device, zs.dtype) for item in self.scale]
        zs = zs / scale[1].view(1, -1, 1, 1, 1) + scale[0].view(1, -1, 1, 1, 1)
        zs = zs.permute(0, 2, 1, 3, 4)          # BCTHW -> NTCHW
        dec = []
        for u in zs:
            d = self.model.decode_video(u.unsqueeze(0), parallel=self.parallel).squeeze(0)
            d = d.mul_(2).sub_(1).clamp_(-1, 1)  # [0, 1] -> [-1, 1]
            dec.append(d)
        dec = torch.stack(dec).permute(0, 2, 1, 3, 4)  # NTCHW -> BCTHW
        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    @classmethod
    def from_pretrained(cls, pretrained_model_path, additional_kwargs={}, **kwargs):
        def filter_kwargs(cls, kwargs):
            import inspect
            sig = inspect.signature(cls.__init__)
            valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            return filtered_kwargs

        # standard diffusers directory kwargs (e.g. subfolder= from load_model_hook)
        subfolder = kwargs.get("subfolder", None)
        if subfolder:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        kwargs = filter_kwargs(cls, additional_kwargs)
        # support diffusers directory checkpoints (save_pretrained output)
        if os.path.isdir(pretrained_model_path):
            if os.path.exists(os.path.join(pretrained_model_path, "config.json")):
                return super().from_pretrained(pretrained_model_path, **kwargs)
            candidates = sorted(f for f in os.listdir(pretrained_model_path)
                                if f.endswith((".pth", ".safetensors")))
            if not candidates:
                raise ValueError(f"No TAE weights found in {pretrained_model_path}")
            pretrained_model_path = os.path.join(pretrained_model_path, candidates[0])
        # infer model flavour from the checkpoint filename when not given explicitly
        checkpoint_name = pretrained_model_path.lower()
        if "latent_channels" not in kwargs:
            if "taew2_2" in checkpoint_name:
                kwargs["latent_channels"], kwargs["patch_size"] = 48, 2
            else:
                kwargs["latent_channels"], kwargs["patch_size"] = 16, 1
        if kwargs.get("arch_variant") is None and "_super" in checkpoint_name:
            kwargs["arch_variant"] = "super"

        model = cls(**kwargs)
        if pretrained_model_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(pretrained_model_path)
        else:
            state_dict = torch.load(pretrained_model_path, map_location="cpu", weights_only=True)
        # TAE checkpoints already use "encoder.*" / "decoder.*" keys, no prefix needed
        state_dict = model.model.patch_tgrow_layers(state_dict)
        m, u = model.model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m, u)
        return model
