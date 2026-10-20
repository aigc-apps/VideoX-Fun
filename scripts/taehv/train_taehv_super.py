"""Super decoder training, copied and adapted from train_taehv.py.

Frozen official Wan2.2 encoder -> trainable Super decoder, using the four
TinyVAE losses and the stage 4a B training recipe.
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The VideoX-Fun Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# ruff: noqa: E402 -- local imports follow the repository sys.path bootstrap.
import argparse
import json
import logging
import math
import os
import shutil
import sys

import diffusers
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import transforms

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data import ImageVideoSampler, RandomSampler, VideoDataset
from videox_fun.models import AutoencoderKLWan3_8, AutoencoderTinyWan
from videox_fun.utils.tqdm_bar import PauseAwareTqdm
from videox_fun.utils.utils import save_videos_grid

check_min_version("0.18.0.dev0")
logger = get_logger(__name__, log_level="INFO")


# TinyVAE reconstruction losses: RGB [0,1], BCTHW, FP32 reductions.
def grid_loss_per_frame(pred, target, period=16):
    """Reference: ZYT-World: A Real-Time Controllable World Model for
    Closed-Loop Autonomous-Driving Simulation (https://arxiv.org/abs/2609.21712).

    Penalize NCHW residual phase means (16x16 by default) after per-frame/channel
    DC removal to suppress periodic grid artifacts.
    """
    n, c, h, w = pred.shape
    h, w = h // period * period, w // period * period
    if min(h, w) == 0:
        raise ValueError("Images must be at least one grid period in size")
    residual = pred.float()[..., :h, :w] - target.float()[..., :h, :w]
    phase = residual.reshape(n, c, h // period, period, w // period, period).mean((2, 4))
    return (phase - phase.mean((-2, -1), keepdim=True)).abs().mean((1, 2, 3))

class TinyVAEReconstructionLoss(nn.Module):
    def __init__(
        self,
        weights,
        dino_repo=None,
        dino_weights=None,
        feature_batch_size=13,
        feature_checkpointing=False,
        grid_period=16,
        lpips_model=None,
        dino_model=None,
    ):
        super().__init__()
        self.weights = {
            k: float(weights[k]) for k in ("l1", "lpips", "dino", "grid")
        }
        if any(not math.isfinite(v) or v < 0 for v in self.weights.values()):
            raise ValueError("Loss weights must be finite and nonnegative")
        if not any(self.weights.values()) or min(feature_batch_size, grid_period) < 1:
            raise ValueError("Need a positive objective, feature batch size and grid period")
        if lpips_model is None:
            import lpips

            lpips_model = lpips.LPIPS(net="alex", version="0.1", verbose=False)
        self.lpips = lpips_model.eval().requires_grad_(False)
        if self.weights["dino"] and dino_model is None:
            dino_model = torch.hub.load(
                dino_repo or "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8",
                "dinov2_vitb14",
                source="local" if dino_repo else "github",
                pretrained=not bool(dino_weights),
                trust_repo=True,
                skip_validation=True,
            )
            if dino_weights:
                dino_model.load_state_dict(
                    torch.load(dino_weights, map_location="cpu", weights_only=True), strict=True
                )
        self.dino = dino_model.eval().requires_grad_(False) if self.weights["dino"] else None
        self.feature_batch_size = feature_batch_size
        self.feature_checkpointing = feature_checkpointing
        self.grid_period = grid_period
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def dino_features(self, x):
        x = F.interpolate(x.float(), (224, 560), mode="bilinear", align_corners=False)
        return self.dino.get_intermediate_layers(
            (x - self.mean) / self.std,
            n=(2, 5, 8, 11),
            reshape=False,
            return_class_token=False,
            norm=True,
        )

    def feature_terms(self, pred, target):
        lp = (
            self.lpips(pred * 2 - 1, target * 2 - 1).float().flatten()
            if self.weights["lpips"]
            else pred.new_zeros(len(pred))
        )
        semantic = pred.new_zeros(len(pred))
        if self.dino is not None:
            with torch.no_grad():
                reference = self.dino_features(target)
            features = self.dino_features(pred)
            semantic = torch.stack(
                [
                    (fp.float() - ft.float()).square().sum((-2, -1))
                    / ft.float().square().sum((-2, -1)).clamp_min(1e-8)
                    for fp, ft in zip(features, reference, strict=True)
                ]
            ).mean(0)
        return lp, semantic

    def forward(self, pred, target):
        if pred.ndim != 5 or pred.shape != target.shape or pred.shape[1] != 3:
            raise ValueError("Expected equal BCTHW RGB tensors in [0,1]")
        p = pred.permute(0, 2, 1, 3, 4).flatten(0, 1).float()
        x = target.permute(0, 2, 1, 3, 4).flatten(0, 1).float()
        terms = {
            "l1": (p - x).abs().mean(),
            "grid": grid_loss_per_frame(p, x, self.grid_period).mean(),
        }
        chunks = []
        for start in range(0, len(p), self.feature_batch_size):
            inputs = (
                p[start : start + self.feature_batch_size],
                x[start : start + self.feature_batch_size],
            )
            if self.feature_checkpointing and torch.is_grad_enabled() and p.requires_grad:
                chunks.append(checkpoint(self.feature_terms, *inputs, use_reentrant=False))
            else:
                chunks.append(self.feature_terms(*inputs))
        terms.update(
            lpips=torch.cat([v[0] for v in chunks]).mean(),
            dino=torch.cat([v[1] for v in chunks]).mean(),
        )
        return sum(self.weights[k] * v for k, v in terms.items()), terms

    @torch.no_grad()
    def metrics(self, pred, target):
        """Unweighted frame means; RGB Gaussian 11x11 valid SSIM, no DINO in eval."""
        p = pred.permute(0, 2, 1, 3, 4).flatten(0, 1).float()
        x = target.permute(0, 2, 1, 3, 4).flatten(0, 1).float()
        rows = []
        for start in range(0, len(p), self.feature_batch_size):
            a, b = (
                p[start : start + self.feature_batch_size],
                x[start : start + self.feature_batch_size],
            )
            lp = self.lpips(a * 2 - 1, b * 2 - 1).float().flatten()
            with torch.autocast(p.device.type, enabled=False):
                q = torch.arange(11, device=p.device, dtype=torch.float32) - 5
                kernel = torch.exp(-q.square() / (2 * 1.5**2))
                kernel /= kernel.sum()
                window = (kernel[:, None] * kernel[None, :])[None, None].expand(3, 1, 11, 11)
                mu_a, mu_b = (F.conv2d(v, window, groups=3) for v in (a, b))
                va = (F.conv2d(a.square(), window, groups=3) - mu_a.square()).clamp_min(0)
                vb = (F.conv2d(b.square(), window, groups=3) - mu_b.square()).clamp_min(0)
                cov = F.conv2d(a * b, window, groups=3) - mu_a * mu_b
                ss = ((2 * mu_a * mu_b + 0.01**2) * (2 * cov + 0.03**2)) / (
                    (mu_a.square() + mu_b.square() + 0.01**2) * (va + vb + 0.03**2)
                )
                psnr = -10 * (a - b).square().mean((1, 2, 3)).clamp_min(1e-12).log10()
                grid = grid_loss_per_frame(a, b, self.grid_period)
            rows.append(torch.stack((psnr, ss.mean((1, 2, 3)), lp, grid), dim=1))
        return dict(zip(("psnr", "ssim", "lpips", "grid"), torch.cat(rows).mean(0).tolist()))

@torch.no_grad()
def log_validation(vae, tae, args, accelerator, weight_dtype, global_step,
                   criterion, validation_dataloader, selection_state, ema_tae=None):
    """Validate the Super decoder using official latents, then select inference weights."""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Running validation...")
        unwrapped_tae = accelerator.unwrap_model(tae)
        trainable_params = list(unwrapped_tae.model.decoder.parameters())
        was_training = unwrapped_tae.training
        if ema_tae is not None:
            ema_tae.store(trainable_params)
            ema_tae.copy_to(trainable_params)
        unwrapped_tae.eval()
        rows = []
        try:
            for i, batch in enumerate(validation_dataloader):
                if args.validation_max_samples and i >= args.validation_max_samples:
                    break
                pixel_values = batch["pixel_values"].to(accelerator.device)
                with accelerator.autocast():
                    gt_latents = vae.encode(pixel_values.to(weight_dtype)).latent_dist.mode()
                    decoded = unwrapped_tae(gt_latents).sample[:, :, :pixel_values.shape[2]]
                    decoded = decoded.float().add(1).mul(0.5)
                    target = pixel_values.float().add(1).mul(0.5)
                    rows.append(criterion.metrics(decoded, target))
                if i == 0:
                    save_videos_grid(
                        torch.cat([target, decoded]),
                        os.path.join(args.output_dir, f"sample/step-{global_step}.mp4"), n_rows=2,
                    )
            if not rows:
                raise ValueError("Validation set is empty")
            metrics = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
            if selection_state["reference"] is None:
                selection_state["reference"] = metrics
            reference = selection_state["reference"]
            eligible = (metrics["lpips"] <= reference["lpips"] * args.selection_lpips_ratio and
                        metrics["grid"] <= reference["grid"] * args.selection_grid_ratio)
            best = eligible and (selection_state["best_psnr"] is None or metrics["psnr"] > selection_state["best_psnr"])
            if best:
                selection_state["best_psnr"] = metrics["psnr"]
                unwrapped_tae.save_pretrained(os.path.join(args.output_dir, "best", "taehv"))
            row = {"step": global_step, "weights": "ema" if ema_tae is not None else "raw",
                   "eligible": eligible, "best": best, **metrics}
            with open(os.path.join(args.output_dir, "validation.jsonl"), "a") as file:
                file.write(json.dumps(row) + "\n")
            accelerator.log({f"validation/{key}": value for key, value in metrics.items()}, step=global_step)
            logger.info(row)
        finally:
            if ema_tae is not None:
                ema_tae.restore(trainable_params)
            unwrapped_tae.train(was_training)
    accelerator.wait_for_everyone()



def parse_args():
    parser = argparse.ArgumentParser(description="Training script for the Wan2.2 Super decoder.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/wan2.2/wan_civitai_5b.yaml",
        help="Path to the Wan2.2 model config yaml; vae_kwargs configures the frozen official VAE.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model directory (contains the full VAE weights).",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="datasets/X-Fun-Videos-Demo",
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json",
        help=("A JSON array containing the training video metadata."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dir_taehv_super",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=1, help="Mini batch size for VAE encoding."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=200, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_min_ratio", type=float, default=0.2,
        help="Final LR / peak LR for the cosine scheduler (stage 4a: 1e-6 / 5e-6).",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training."
        ),
    )
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=False, help="Whether to use EMA model.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to.',
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="taehv-train",
        help="The `project_name` argument passed to Accelerator.init_trackers.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Training video sample size.",
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=1,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=25,
        help="Num frame of video. Must be 4k+1 to match the causal temporal compression.",
    )
    parser.add_argument(
        "--tae_path",
        type=str,
        default=None,
        help="Optional Super weights to warm-start from (taew2_2_super.pth/.safetensors, a directory, "
             "or a previous checkpoint-xxx/taehv dir). If not given, train from scratch.",
    )
    parser.add_argument(
        "--use_taehv_sequential", action="store_true",
        help="Use the sequential TAE decoder instead of the parallel path.",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Initialize a new Super decoder without student weights.",
    )
    parser.add_argument(
        "--fix_sample_size",
        nargs=2,
        type=int,
        default=None,
        help="Fixed [height, width] overriding video_sample_size.",
    )
    parser.add_argument(
        "--video_sample_strides",
        nargs="+",
        type=int,
        default=None,
        help="Optional per-video random sampling strides.",
    )
    parser.add_argument(
        "--validation_data_meta",
        type=str,
        default=None,
        help="Held-out validation JSON index.",
    )
    parser.add_argument(
        "--validation_n_frames",
        type=int,
        default=81,
        help="Validation clip length, 4k+1.",
    )
    parser.add_argument(
        "--validation_max_samples",
        type=int,
        default=0,
        help="Validation sample cap; zero evaluates the complete index.",
    )
    parser.add_argument(
        "--dino_repo",
        type=str,
        default=None,
        help="Optional local pinned DINOv2 source directory.",
    )
    parser.add_argument(
        "--dino_weights",
        type=str,
        default=None,
        help="Optional local ViT-B/14 weights.",
    )
    parser.add_argument(
        "--feature_batch_size",
        type=int,
        default=13,
        help="Frame batch size for perceptual networks.",
    )
    parser.add_argument(
        "--feature_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Checkpoint perceptual feature computation.",
    )
    parser.add_argument(
        "--grid_period",
        type=int,
        default=16,
        help="Spatial residual period for grid loss.",
    )
    parser.add_argument(
        "--l1_loss_weight",
        type=float,
        default=0.1,
        help="RGB L1 reconstruction loss weight.",
    )
    parser.add_argument(
        "--lpips_loss_weight",
        type=float,
        default=0.1,
        help="LPIPS reconstruction loss weight.",
    )
    parser.add_argument(
        "--dino_loss_weight",
        type=float,
        default=0.02,
        help="DINOv2 patch feature loss weight.",
    )
    parser.add_argument(
        "--grid_loss_weight",
        type=float,
        default=1.0,
        help="Periodic spatial residual loss weight.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay for decoder parameters.",
    )
    parser.add_argument(
        "--selection_lpips_ratio",
        type=float,
        default=1.02,
        help="Maximum LPIPS relative to the first validation.",
    )
    parser.add_argument(
        "--selection_grid_ratio",
        type=float,
        default=1.05,
        help="Maximum grid loss relative to the first validation.",
    )

    args = parser.parse_args()

    if args.fix_sample_size is not None:
        args.video_sample_height, args.video_sample_width = args.fix_sample_size
    else:
        args.video_sample_height = args.video_sample_width = args.video_sample_size
    if any(size < 32 or size % 16 for size in (args.video_sample_height, args.video_sample_width)):
        parser.error("Video dimensions must be >=32 and divisible by 16")
    if any(frames < 1 or (frames - 1) % 4 for frames in (args.video_sample_n_frames, args.validation_n_frames)):
        parser.error("Video frame counts must be 4k+1")
    if args.from_scratch and (args.tae_path or args.resume_from_checkpoint):
        parser.error("--from_scratch cannot be combined with weight initialization or resume")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        parser.error("max_train_steps must be positive")
    if min(args.num_train_epochs, args.train_batch_size, args.vae_mini_batch,
           args.gradient_accumulation_steps, args.checkpointing_steps, args.video_sample_stride,
           args.feature_batch_size, args.grid_period) < 1:
        parser.error("Epochs, batch sizes, checkpoint interval, stride and grid period must be positive")
    if min(args.dataloader_num_workers, args.validation_steps, args.validation_max_samples, args.lr_warmup_steps) < 0:
        parser.error("Workers, validation intervals/sample cap and warmup cannot be negative")
    if args.video_sample_strides is not None and (not args.video_sample_strides or min(args.video_sample_strides) < 1):
        parser.error("video_sample_strides must be positive")
    if args.checkpoints_total_limit is not None and args.checkpoints_total_limit < 1:
        parser.error("checkpoints_total_limit must be positive")
    if not 0 <= args.ema_decay < 1 or not 0 <= args.adam_beta1 < 1 or not 0 <= args.adam_beta2 < 1:
        parser.error("Invalid EMA decay or Adam betas")
    if args.adam_weight_decay < 0:
        parser.error("Adam weight decay cannot be negative")
    if not 0 <= args.lr_min_ratio <= 1:
        parser.error("lr_min_ratio must be between 0 and 1")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.learning_rate, args.adam_epsilon, args.max_grad_norm,
            args.selection_lpips_ratio, args.selection_grid_ratio)):
        parser.error("Learning rate, epsilon, gradient norm and selection ratios must be finite and positive")
    loss_weights = (args.l1_loss_weight, args.lpips_loss_weight, args.dino_loss_weight,
                    args.grid_loss_weight)
    if not any(loss_weights) or any(not math.isfinite(value) or value < 0 for value in loss_weights):
        parser.error("Loss weights must be finite and nonnegative, with at least one positive term")
    return args




def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=accelerator_project_config,
    )
    if accelerator.distributed_type.name not in ("NO", "MULTI_GPU", "MULTI_CPU"):
        raise ValueError("This decoder trainer supports single device and ordinary DDP")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()
    set_seed(args.seed)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if not args.resume_from_checkpoint and (
            os.path.exists(os.path.join(args.output_dir, "train.jsonl")) or
            any(name.startswith("checkpoint-") for name in os.listdir(args.output_dir))
        ):
            raise FileExistsError("Use a new output_dir or --resume_from_checkpoint")

    # For mixed precision training, keep trainable parameters and Adam moments FP32.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ==================== Load Config ====================
    config = OmegaConf.load(args.config_path)
    if config['vae_kwargs'].get('vae_type') != 'AutoencoderKLWan3_8':
        raise ValueError("Super training requires the official 48-channel Wan2.2 VAE")

    # ==================== Load Models ====================
    vae = AutoencoderKLWan3_8.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    )
    vae.eval()
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)

    tae_load_kwargs = {"latent_channels": 48, "patch_size": 2, "arch_variant": "super"}
    if args.tae_path is not None and not args.resume_from_checkpoint:
        logger.info(f"Warm-starting Super TAE from checkpoint: {args.tae_path}")
        tae = AutoencoderTinyWan.from_pretrained(
            args.tae_path, additional_kwargs=tae_load_kwargs,
        )
    else:
        logger.info("Initializing Super TAE (training state is restored below when resuming)")
        tae = AutoencoderTinyWan(**tae_load_kwargs)
    if tae.config.arch_variant != "super" or tae.config.latent_channels != 48 or tae.config.patch_size != 2:
        raise ValueError("Expected a 48-channel, patch-size-2 Super checkpoint")
    tae.parallel = not args.use_taehv_sequential
    tae.float()
    tae.requires_grad_(False)
    tae.model.decoder.requires_grad_(True)
    tae.train()

    trainable_params = list(filter(lambda p: p.requires_grad, tae.parameters()))
    total_params = sum(p.numel() for p in tae.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    logger.info(f"Total params: {total_params / 1e6:.2f}M, Trainable params: {trainable_count / 1e6:.2f}M")
    ema_tae = None
    if args.use_ema:
        ema_tae = EMAModel(
            trainable_params, decay=args.ema_decay, min_decay=args.ema_decay, use_ema_warmup=False,
        )

    # ==================== Save/Load Hooks ====================
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            accelerator.unwrap_model(models[0]).save_pretrained(os.path.join(output_dir, "taehv"))
            trainer_state = {"step": global_step, "epoch": epoch, "batch": resume_step, **selection_state}
            with open(os.path.join(output_dir, "trainer_state.json"), "w") as file:
                json.dump(trainer_state, file, indent=2)
        # The model is saved in Diffusers format by this hook, as in train_taehv.py.
        weights.pop()

    def load_model_hook(models, input_dir):
        while models:
            model = accelerator.unwrap_model(models.pop())
            load_model = AutoencoderTinyWan.from_pretrained(input_dir, subfolder="taehv")
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict(), strict=True)
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # ==================== Optimizer ====================
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )

    # ==================== Dataset ====================
    train_dataset = VideoDataset(
        args.train_data_meta, args.train_data_dir,
        sample_size=args.video_sample_size,
        sample_stride=args.video_sample_strides or args.video_sample_stride,
        sample_n_frames=args.video_sample_n_frames,
        enable_bucket=True, enable_inpaint=False, text_drop_ratio=0,
    )
    if not len(train_dataset):
        raise ValueError("Training set is empty")

    def collate_fn(examples):
        """Fixed canvas, aspect-preserving resize/crop; align the shortest clip to 4k+1."""
        frames = (min(len(e["pixel_values"]) for e in examples) - 1) // 4 * 4 + 1
        height, width = args.video_sample_height, args.video_sample_width
        videos = []
        for example in examples:
            x = torch.from_numpy(example["pixel_values"][:frames]).permute(0, 3, 1, 2).float()
            h, w = x.shape[-2:]
            scale = max(height / h, width / w)
            resize_size = (math.ceil(h * scale), math.ceil(w * scale))
            x = transforms.functional.resize(x, resize_size, antialias=True)
            x = transforms.functional.center_crop(x, (height, width))
            videos.append(x.div_(127.5).sub_(1))
        return {"pixel_values": torch.stack(videos).permute(0, 2, 1, 3, 4).contiguous()}

    batch_sampler_generator = torch.Generator().manual_seed(args.seed)
    batch_sampler = ImageVideoSampler(
        RandomSampler(train_dataset, generator=batch_sampler_generator),
        train_dataset, args.train_batch_size, drop_last=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        persistent_workers=True if args.dataloader_num_workers != 0 else False,
        num_workers=args.dataloader_num_workers,
    )
    validation_dataloader = None
    if args.validation_data_meta is not None:
        validation_dataset = VideoDataset(
            args.validation_data_meta, args.train_data_dir,
            sample_n_frames=args.validation_n_frames, sample_stride=1,
            enable_bucket=True, enable_inpaint=False, text_drop_ratio=0,
            random_start=False,
        )
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=1, collate_fn=collate_fn,
        )

    # ==================== LR Scheduler ====================
    tae, optimizer, train_dataloader = accelerator.prepare(tae, optimizer, train_dataloader)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if num_update_steps_per_epoch == 0:
        raise ValueError("Training data does not contain a complete batch")
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer.optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    if args.lr_scheduler == "cosine":
        # Keep Diffusers warmup/cosine and add stage 4a's nonzero final LR.
        cosine_schedule = lr_scheduler.lr_lambdas[0]

        def lr_lambda(step):
            factor = cosine_schedule(step)
            if step < args.lr_warmup_steps:
                return factor
            return args.lr_min_ratio + (1 - args.lr_min_ratio) * factor

        lr_scheduler.lr_lambdas = [lr_lambda] * len(lr_scheduler.lr_lambdas)
    # Register explicitly so it advances once per optimizer update, at any world size.
    accelerator.register_for_checkpointing(lr_scheduler)
    if args.use_ema:
        ema_tae.to(accelerator.device)
        accelerator.register_for_checkpointing(ema_tae)

    # ==================== TinyVAE Losses ====================
    loss_weights = {
        "l1": args.l1_loss_weight,
        "lpips": args.lpips_loss_weight,
        "dino": args.dino_loss_weight,
        "grid": args.grid_loss_weight,
    }
    criterion = TinyVAEReconstructionLoss(
        loss_weights,
        dino_repo=args.dino_repo, dino_weights=args.dino_weights,
        feature_batch_size=args.feature_batch_size,
        feature_checkpointing=args.feature_checkpointing, grid_period=args.grid_period,
    ).to(accelerator.device)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        keys_to_pop = [key for key, value in tracker_config.items() if isinstance(value, list)]
        for key in keys_to_pop:
            tracker_config.pop(key)
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # ==================== Training ====================
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running Super decoder training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Loss weights = {loss_weights}")
    global_step = 0
    first_epoch = 0
    resume_step = 0
    selection_state = {"reference": None, "best_psnr": None}

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = args.resume_from_checkpoint
        else:
            dirs = [name for name in os.listdir(args.output_dir)
                    if name.startswith("checkpoint-") and name.split("-")[-1].isdigit()]
            dirs = sorted(dirs, key=lambda name: int(name.split("-")[-1]))
            if not dirs:
                raise FileNotFoundError(f"No checkpoints in {args.output_dir}")
            path = os.path.join(args.output_dir, dirs[-1])
        with open(os.path.join(path, "trainer_state.json")) as file:
            trainer_state = json.load(file)
        global_step = trainer_state["step"]
        first_epoch = trainer_state["epoch"]
        resume_step = trainer_state["batch"]
        selection_state["reference"] = trainer_state["reference"]
        selection_state["best_psnr"] = trainer_state["best_psnr"]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(path)
        if args.use_ema:
            ema_tae.to(accelerator.device)
    initial_global_step = global_step
    epoch = first_epoch
    if accelerator.is_main_process:
        OmegaConf.save(OmegaConf.create(vars(args)), os.path.join(args.output_dir, f"args-step-{global_step}.yaml"))
    if validation_dataloader is not None and selection_state["reference"] is None:
        log_validation(vae, tae, args, accelerator, weight_dtype, global_step,
                       criterion, validation_dataloader, selection_state, ema_tae)

    progress_bar = PauseAwareTqdm(
        range(0, args.max_train_steps), initial=initial_global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    optimizer.zero_grad()
    for epoch in range(first_epoch, args.num_train_epochs):
        if global_step >= args.max_train_steps:
            break
        train_loss = torch.zeros(1 + len(loss_weights), device=accelerator.device)
        micro_steps = 0
        train_dataloader.set_epoch(epoch)
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        batch_sampler.sampler._pos_start = 0
        batch_sampler.bucket = {"image": [], "video": []}
        epoch_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step) if resume_step else train_dataloader
        for step, batch in enumerate(epoch_dataloader, start=resume_step):
            with accelerator.accumulate(tae):
                pixel_values = batch["pixel_values"].to(accelerator.device)  # [B, C, F, H, W], FP32 target
                bsz = pixel_values.shape[0]

                # 1. Official latents, frozen.
                with torch.no_grad(), accelerator.autocast():
                    gt_latents_list = []
                    for i in range(0, bsz, args.vae_mini_batch):
                        encoded = vae.encode(pixel_values[i:i + args.vae_mini_batch].to(weight_dtype)).latent_dist.mode()
                        gt_latents_list.append(encoded)
                    gt_latents = torch.cat(gt_latents_list, dim=0).detach()

                # 2. Student forward: official latent -> Super decoder, through DDP.
                with accelerator.autocast():
                    decoded = tae(gt_latents).sample[:, :, :pixel_values.shape[2]]
                    decoded = decoded.float().add(1).mul(0.5)
                    target = pixel_values.float().add(1).mul(0.5)
                    loss, loss_terms = criterion(decoded, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {global_step}")

                # 3. Gather losses for logging and backpropagate.
                train_loss += torch.stack([loss.detach(), *[loss_terms[key].detach() for key in loss_weights]])
                micro_steps += 1
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                    if args.use_ema:
                        ema_tae.step(trainable_params)

            resume_step = step + 1
            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                progress_bar.update(1)
                global_step += 1
                averaged_loss = accelerator.reduce(train_loss / micro_steps, reduction="mean").tolist()
                logs = dict(zip(("loss", *loss_weights), averaged_loss))
                logs.update(step=global_step, frames=pixel_values.shape[2], lr=lr_scheduler.get_last_lr()[0])
                accelerator.log(logs, step=global_step)
                if accelerator.is_main_process:
                    with open(os.path.join(args.output_dir, "train.jsonl"), "a") as file:
                        file.write(json.dumps(logs) + "\n")
                train_loss.zero_()
                micro_steps = 0
                progress_bar.set_postfix(step_loss=logs["loss"], lr=logs["lr"])

                # Validation
                if validation_dataloader is not None and args.validation_steps and global_step % args.validation_steps == 0:
                    with progress_bar.paused():
                        log_validation(vae, tae, args, accelerator, weight_dtype, global_step,
                                       criterion, validation_dataloader, selection_state, ema_tae)

                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                        checkpoints = [name for name in os.listdir(args.output_dir)
                                       if name.startswith("checkpoint-") and name.split("-")[-1].isdigit()]
                        checkpoints = sorted(checkpoints, key=lambda name: int(name.split("-")[-1]))
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        for removing_checkpoint in checkpoints[:max(num_to_remove, 0)]:
                            shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))
                    accelerator.wait_for_everyone()
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    with progress_bar.paused():
                        accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break
        resume_step = 0
    progress_bar.close()

    # Final validation and save. Restore raw parameters after EMA validation.
    if validation_dataloader is not None and (not args.validation_steps or global_step % args.validation_steps):
        log_validation(vae, tae, args, accelerator, weight_dtype, global_step,
                       criterion, validation_dataloader, selection_state, ema_tae)
    accelerator.wait_for_everyone()
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
