"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import contextlib
import gc
import logging
import math
import os
import pickle
import random
import shutil
import sys
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Tuple,
                    Union)

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from omegaconf import OmegaConf
from packaging import version
from torch.utils.data import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data.bucket_sampler import ASPECT_RATIO_512, RandomSampler
from videox_fun.data.dataset_image_video import TextDataset
from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKL, AutoTokenizer,
                               CLIPImageProcessor,
                               CLIPVisionModelWithProjection,
                               Qwen2_5_VLForConditionalGeneration,
                               Qwen2Tokenizer, Qwen3ForCausalLM,
                               QwenImageTransformer2DModel,
                               ZImageTransformer2DModel)
from videox_fun.pipeline import ZImagePipeline
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.lora_utils import (convert_peft_lora_to_kohya_lora,
                                         create_network, merge_lora,
                                         unmerge_lora)
from videox_fun.utils.sd3_sde_with_logprob import sde_step_with_logprob
from videox_fun.utils.utils import get_image_to_video_latent, save_videos_grid

if is_wandb_available():
    import wandb

def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def encode_prompt(
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    text_encoder = None, 
    tokenizer = None,
    max_sequence_length: int = 512,
) -> List[torch.FloatTensor]:
    if isinstance(prompt, str):
        prompt = [prompt]

    for i, prompt_item in enumerate(prompt):
        messages = [
            {"role": "user", "content": prompt_item},
        ]
        prompt_item = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt[i] = prompt_item

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    embeddings_list = []

    for i in range(len(prompt_embeds)):
        embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

    return embeddings_list

# Fallback implementation if flow_grpo is not available
class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, type='grpo'):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards) * 0.0
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            if type == 'grpo':
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts
    
    def clear(self):
        self.stats = {}

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """Calculate the proportion of unique prompts whose reward standard deviation is zero.

    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.

    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

@torch.no_grad()
def sample_with_cfg(
    model,
    vae,
    noise,
    prompt_embeds,
    neg_prompt_embeds,
    num_steps=25,
    cfg_scale=7.5,
    noise_scheduler=None,
    device='cuda',
    dtype=torch.float32,
    noise_level: float = 0.7,
    sde_window_size: int = 0,
    sde_window_range: tuple[int, int] = (0, 5),
):
    batch_size = noise.shape[0]
    image_seq_len = (noise.shape[2] // 2) * (noise.shape[3] // 2)
    latents = noise.clone().to(torch.float32)

    mu = calculate_shift(
        image_seq_len,
        noise_scheduler.config.get("base_image_seq_len", 256),
        noise_scheduler.config.get("max_image_seq_len", 4096),
        noise_scheduler.config.get("base_shift", 0.5),
        noise_scheduler.config.get("max_shift", 1.15),
    )
    scheduler_kwargs = {"mu": mu}
    timesteps, num_inference_steps = retrieve_timesteps(
        noise_scheduler,
        num_steps,
        device,
        sigmas=None,
        **scheduler_kwargs,
    )

    if sde_window_size > 0:
        assert sde_window_range[1] - sde_window_size >= sde_window_range[0], (
            f"sde_window_range {sde_window_range} 与 sde_window_size {sde_window_size} 不兼容，"
            f"请保证 range[1] - window_size >= range[0]"
        )
        start = random.randint(
            sde_window_range[0],
            sde_window_range[1] - sde_window_size
        )
        end = start + sde_window_size
        sde_window = (start, end)
    else:
        sde_window = (0, len(timesteps) - 1)

    all_latents   = []
    all_log_probs = []
    all_timesteps = []

    apply_cfg = cfg_scale > 1.0

    sampling_bar = tqdm(enumerate(timesteps), total=len(timesteps), desc="Sampling", leave=False, disable=not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
    for i, t in sampling_bar:
        if i < sde_window[0]:
            cur_noise_level = 0.0
        elif i == sde_window[0]:
            cur_noise_level = noise_level
            all_latents.append(latents.clone())
        elif sde_window[0] < i < sde_window[1]:
            cur_noise_level = noise_level
        else:
            cur_noise_level = 0.0

        timestep = t.expand(batch_size)
        timestep_normalized = (1000 - timestep) / 1000.0

        if apply_cfg:
            latents_typed      = latents.to(dtype)
            latent_model_input = latents_typed.repeat(2, 1, 1, 1, 1)
            prompt_embeds_input = prompt_embeds + neg_prompt_embeds
            timestep_input     = timestep_normalized.repeat(2)
        else:
            latent_model_input  = latents.to(dtype)
            prompt_embeds_input = prompt_embeds
            timestep_input      = timestep_normalized

        latent_model_input_list = list(latent_model_input.unbind(dim=0))
        model_out_list = model(
            latent_model_input_list,
            timestep_input,
            prompt_embeds_input,
        )[0]

        if apply_cfg:
            pos_out = model_out_list[:batch_size]
            neg_out = model_out_list[batch_size:]
            noise_pred = torch.stack(
                [neg_out[j].float() + cfg_scale * (pos_out[j].float() - neg_out[j].float())
                 for j in range(batch_size)],
                dim=0,
            )
        else:
            noise_pred = torch.stack(
                [out.float() for out in model_out_list], dim=0
            )

        noise_pred = -noise_pred   # 原代码符号约定

        latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
            noise_scheduler,
            noise_pred.float(),
            t.unsqueeze(0).repeat(batch_size),
            latents.float(),
            noise_level=cur_noise_level,
        )
        if sde_window[0] <= i < sde_window[1]:
            all_latents.append(latents.clone())
            all_log_probs.append(log_prob)
            all_timesteps.append(t)


    _latents = latents.to(vae.dtype).squeeze(2)
    _latents = (_latents / vae.config.scaling_factor) + vae.config.shift_factor
    images = vae.decode(_latents, return_dict=False)[0].float().unsqueeze(2)
    images = (images / 2 + 0.5).clamp(0, 1)

    ret = {
        "latents"                   : latents.double(),
        "all_latents"               : all_latents,    # List[Tensor], len = window_size+1
        "all_log_probs"             : all_log_probs,  # List[Tensor], len = window_size
        "all_timesteps"             : all_timesteps,  # List[Tensor], len = window_size
        "prompt_embeds"             : prompt_embeds,
        "negative_prompt_embeds"    : neg_prompt_embeds,
        "prompt_embeds_mask"            : None,
        "negative_prompt_embeds_mask"   : None,
        "images"                        : images,
    }
    return ret

def compute_log_prob(
    model,
    vae,
    sample,
    step_idx,
    noise_scheduler,
    prompt_embeds,
    neg_prompt_embeds,
    cfg_scale=4.5,
    noise_level=0.7,
    dtype=torch.float32,
    ref_model=None,
):
    """
    Compute log probability for GRPO training.
    
    Args:
        model: The transformer model
        vae: VAE for decoding (not used in log_prob computation)
        sample: Dict containing latents, next_latents, timesteps
        step_idx: Index of the timestep to compute log_prob for
        noise_scheduler: The noise scheduler
        prompt_embeds: Prompt embeddings (list of tensors)
        neg_prompt_embeds: Negative prompt embeddings (list of tensors)
        cfg_scale: Classifier-free guidance scale
        noise_level: Noise level for SDE
        dtype: Data type for computation
        ref_model: Reference model for KL computation (optional)
    
    Returns:
        log_prob: Log probability of the transition
        prev_sample_mean: Mean of the predicted previous sample
        std_dev_t: Standard deviation at timestep t
        ref_prev_sample_mean: Mean from reference model (if ref_model provided)
    """
    batch_size = sample["latents"].shape[0]
    latents = sample["latents"][:, step_idx]  # [B, C, 1, H, W]
    next_latents = sample["next_latents"][:, step_idx]  # [B, C, 1, H, W]
    timesteps = sample["timesteps"][:, step_idx]  # [B]
    
    apply_cfg = cfg_scale > 1.0
    
    # Prepare timestep
    timestep_normalized = (1000 - timesteps) / 1000.0
    
    if apply_cfg:
        latents_typed = latents.to(dtype)
        latent_model_input = latents_typed.repeat(2, 1, 1, 1, 1)
        prompt_embeds_input = prompt_embeds + neg_prompt_embeds
        timestep_input = timestep_normalized.repeat(2)
    else:
        latent_model_input = latents.to(dtype)
        prompt_embeds_input = prompt_embeds
        timestep_input = timestep_normalized
    
    # Forward pass through transformer
    latent_model_input_list = list(latent_model_input.unbind(dim=0))
    model_out_list = model(
        latent_model_input_list,
        timestep_input,
        prompt_embeds_input,
    )[0]
    
    if apply_cfg:
        pos_out = model_out_list[:batch_size]
        neg_out = model_out_list[batch_size:]
        noise_pred = torch.stack(
            [neg_out[j].float() + cfg_scale * (pos_out[j].float() - neg_out[j].float())
             for j in range(batch_size)],
            dim=0,
        )
    else:
        noise_pred = torch.stack(
            [out.float() for out in model_out_list], dim=0
        )
    
    noise_pred = -noise_pred  # Sign convention
    
    # Compute log prob using SDE step
    _, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        noise_scheduler,
        noise_pred.float(),
        timesteps,
        latents.float(),
        prev_sample=next_latents.float(),
        noise_level=noise_level,
    )
    
    # Compute reference model prediction if provided
    ref_prev_sample_mean = None
    if ref_model is not None:
        with torch.no_grad():
            if apply_cfg:
                ref_model_out_list = ref_model(
                    latent_model_input_list,
                    timestep_input,
                    prompt_embeds_input,
                )[0]
                ref_pos_out = ref_model_out_list[:batch_size]
                ref_neg_out = ref_model_out_list[batch_size:]
                ref_noise_pred = torch.stack(
                    [ref_neg_out[j].float() + cfg_scale * (ref_pos_out[j].float() - ref_neg_out[j].float())
                     for j in range(batch_size)],
                    dim=0,
                )
            else:
                ref_model_out_list = ref_model(
                    latent_model_input_list,
                    timestep_input,
                    prompt_embeds_input,
                )[0]
                ref_noise_pred = torch.stack(
                    [out.float() for out in ref_model_out_list], dim=0
                )
            
            ref_noise_pred = -ref_noise_pred
            
            _, _, ref_prev_sample_mean, _ = sde_step_with_logprob(
                noise_scheduler,
                ref_noise_pred.float(),
                timesteps,
                latents.float(),
                prev_sample=next_latents.float(),
                noise_level=noise_level,
            )
    
    return log_prob, prev_sample_mean, std_dev_t, ref_prev_sample_mean

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def log_validation(vae, text_encoder, tokenizer, transformer3d, network, args, accelerator, weight_dtype, global_step):
    try:
        is_deepspeed = type(transformer3d).__name__ == 'DeepSpeedEngine'
        if is_deepspeed:
            origin_config = transformer3d.config
            transformer3d.config = accelerator.unwrap_model(transformer3d).config
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            logger.info("Running validation... ")
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                args.pretrained_model_name_or_path, 
                subfolder="scheduler"
            )
            pipeline = ZImagePipeline(
                vae=vae, 
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d,
                scheduler=scheduler,
            )
            pipeline = pipeline.to(accelerator.device)

            if args.seed is None:
                generator = None
            else:
                rank_seed = args.seed + accelerator.process_index
                generator = torch.Generator(device=accelerator.device).manual_seed(rank_seed)
                logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

            for i in range(len(args.validation_prompts)):
                sample = pipeline(
                    args.validation_prompts[i], 
                    negative_prompt = "bad detailed",
                    height      = args.image_sample_size,
                    width       = args.image_sample_size,
                    generator   = generator,
                    guidance_scale = 0 if  "Turbo" in args.pretrained_model_name_or_path else 4.5,
                    num_inference_steps = 8 if  "Turbo" in args.pretrained_model_name_or_path else 25,
                ).images
                os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                image = sample[0].save(
                    os.path.join(
                        args.output_dir, 
                        f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.jpg"
                    )
                )

            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
            transformer3d.to(accelerator.device, dtype=weight_dtype)
            if not args.enable_text_encoder_in_dataloader:
                text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        if is_deepspeed:
            transformer3d.config = origin_config
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error on rank {accelerator.process_index} with info {e}")
        vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        transformer3d.to(accelerator.device, dtype=weight_dtype)
        if not args.enable_text_encoder_in_dataloader:
            text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
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
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
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
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--use_peft_lora", action="store_true", help="Whether or not to use peft lora."
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true", help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Sample size of the image.",
    )
    parser.add_argument(
        "--fix_sample_size", 
        nargs=2, type=int, default=None,
        help="Fix Sample size [height, width] when using bucket and collate_fn."
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )
    parser.add_argument("--save_state", action="store_true", help="Whether or not to save state.")

    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Whether or not to use fsdp."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--lora_skip_name",
        type=str,
        default=None,
        help=("The module is not trained in loras. "),
    )
    parser.add_argument(
        "--target_name",
        type=str,
        default=None,
        help=("The module is trained in loras. "),
    )
    parser.add_argument(
        "--training_with_video_token_length",
        action="store_true",
        help="Whether to train with video token length. When set, the text encoder is not trained in lora mode.",
    )
    # GRPO specific arguments
    parser.add_argument(
        "--grpo_num_steps",
        type=int,
        default=20,
        help="Number of inference steps for GRPO sampling.",
    )
    parser.add_argument(
        "--grpo_cfg_scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale for GRPO sampling.",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=1.2,
        help="Noise level for SDE sampling in GRPO.",
    )
    parser.add_argument(
        "--sde_window_size",
        type=int,
        default=2,
        help="SDE window size for GRPO training. 0 means use all steps.",
    )
    parser.add_argument(
        "--sde_window_range",
        nargs=2,
        type=int,
        default=[0, 5],
        help="SDE window range [start, end] for GRPO training.",
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=1e-5,
        help="PPO clip range for GRPO training.",
    )
    parser.add_argument(
        "--adv_clip_max",
        type=float,
        default=5.0,
        help="Maximum value for advantage clipping.",
    )
    parser.add_argument(
        "--grpo_beta",
        type=float,
        default=0.0,
        help="KL divergence coefficient for GRPO. 0 means no KL regularization.",
    )
    parser.add_argument(
        "--per_prompt_stat_tracking",
        action="store_true",
        default=True,
        help="Whether to use per-prompt statistics tracking for advantage normalization.",
    )
    parser.add_argument(
        "--global_std",
        action="store_true",
        default=True,
        help="Whether to use global std for advantage normalization.",
    )
    parser.add_argument(
        "--num_image_per_prompt",
        type=int,
        default=16,
        help="Number of images to generate per prompt for GRPO group comparison. "
             "Each prompt is repeated this many times consecutively by the sampler, and the advantage is "
             "normalized within the group, so it is recommended that `num_batches_per_epoch * train_batch_size` "
             "is divisible by this value.",
    )
    parser.add_argument(
        "--num_batches_per_epoch",
        type=int,
        default=16,
        help="Number of sampling batches to collect per epoch before training. "
             "All batches are sampled with the same model, then concatenated for advantage computation and training.",
    )
    parser.add_argument(
        "--reward_fn",
        type=str,
        default="MPSReward",
        help="Reward function to use for GRPO training. For multiple rewards, use comma-separated names like 'HPSReward,MPSReward'.",
    )
    parser.add_argument(
        "--reward_fn_kwargs",
        type=str,
        default=None,
        help="JSON string of kwargs for the reward function. For multiple rewards, use JSON dict like '{\"HPSReward\": {\"version\": \"v2.1\"}, \"MPSReward\": {}}'.",
    )
    parser.add_argument(
        "--multi_reward_weights",
        type=str,
        default=None,
        help="JSON string of weights for combining advantages from multiple rewards, e.g., '{\"HPSReward\": 0.5, \"MPSReward\": 0.5}'. If None, use equal weights.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    deepspeed_plugin = accelerator.state.deepspeed_plugin if hasattr(accelerator.state, "deepspeed_plugin") else None
    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        fsdp_stage = 0
        print(f"Using DeepSpeed Zero stage: {zero_stage}")

        args.use_deepspeed = True
        if zero_stage == 3:
            print(f"Auto set save_state to True because zero_stage == 3")
            args.save_state = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None: # The fsdp_plugin.sharding_strategy is None in FSDP 2.
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"Using FSDP stage: {fsdp_stage}")

        args.use_fsdp = True
        if fsdp_stage == 3:
            print(f"Auto set save_state to True because fsdp_stage == 3")
            args.save_state = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("DeepSpeed is not enabled.")

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # When num_image_per_prompt == 1, per-prompt stat tracking is meaningless (no group to compare)
    if args.num_image_per_prompt == 1:
        args.per_prompt_stat_tracking = False
        logger.info(f"num_image_per_prompt=1, auto-disabling per_prompt_stat_tracking")
    else:
        logger.info(f"num_image_per_prompt={args.num_image_per_prompt})")

    # Initialize per-prompt stat tracker for advantage normalization
    if args.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(global_std=args.global_std)
    else:
        stat_tracker = None

    # Calculate number of train timesteps based on SDE window
    if args.sde_window_size > 0:
        num_train_timesteps = args.sde_window_size
    else:
        num_train_timesteps = args.grpo_num_steps - 1

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So Qwen3ForCausalLM and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Get Text encoder
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
        )
        text_encoder = text_encoder.eval()
        # Get Vae
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="vae"
        ).to(weight_dtype)
        vae.eval()

        import json
        import videox_fun.reward.reward_fn as reward_fn

        reward_fn_names = [name.strip() for name in args.reward_fn.split(',')]
        is_multi_reward = len(reward_fn_names) > 1
        
        # Parse kwargs for each reward function
        reward_fn_kwargs_all = {}
        if args.reward_fn_kwargs is not None:
            parsed_kwargs = json.loads(args.reward_fn_kwargs)
            if isinstance(parsed_kwargs, dict):
                # Check if it's a nested dict (multi-reward format) or flat dict (single reward)
                if is_multi_reward or any(name in parsed_kwargs for name in reward_fn_names):
                    reward_fn_kwargs_all = parsed_kwargs
                else:
                    # Single reward with flat kwargs
                    reward_fn_kwargs_all = {reward_fn_names[0]: parsed_kwargs}
            else:
                reward_fn_kwargs_all = {reward_fn_names[0]: parsed_kwargs}
        
        # Parse weights for multi-reward advantage combination
        multi_reward_weights = {}
        if args.multi_reward_weights is not None:
            multi_reward_weights = json.loads(args.multi_reward_weights)
        # Normalize weights to sum to 1
        if is_multi_reward:
            if not multi_reward_weights:
                # Equal weights by default
                multi_reward_weights = {name: 1.0 / len(reward_fn_names) for name in reward_fn_names}
            else:
                total = sum(multi_reward_weights.values())
                multi_reward_weights = {name: multi_reward_weights.get(name, 0.0) / total for name in reward_fn_names}
        
        # Initialize all reward functions
        loss_fns = {}
        if accelerator.is_main_process:
            # Check if the models are downloaded in the main process
            for fn_name in reward_fn_names:
                fn_kwargs = reward_fn_kwargs_all.get(fn_name, {})
                logger.info(f"Loading reward function: {fn_name} with kwargs: {fn_kwargs}")
                loss_fns[fn_name] = getattr(reward_fn, fn_name)(device="cpu", dtype=weight_dtype, **fn_kwargs)
        accelerator.wait_for_everyone()
        
        # Re-initialize on correct device
        loss_fns = {}
        for fn_name in reward_fn_names:
            fn_kwargs = reward_fn_kwargs_all.get(fn_name, {})
            loss_fns[fn_name] = getattr(reward_fn, fn_name)(device=accelerator.device, dtype=weight_dtype, **fn_kwargs)
        
        if is_multi_reward:
            logger.info(f"Multi-reward enabled with {len(reward_fn_names)} rewards: {reward_fn_names}")
            logger.info(f"Advantage combination weights: {multi_reward_weights}")
        else:
            logger.info(f"Single reward function: {reward_fn_names[0]}")
        
        # For backward compatibility, keep loss_fn as the first reward function
        loss_fn = loss_fns[reward_fn_names[0]]

    # Get Transformer
    transformer3d = ZImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="transformer",
        torch_dtype=weight_dtype,
    ).to(weight_dtype)
    ref_transformer3d = ZImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="transformer",
        torch_dtype=weight_dtype,
    ).to(weight_dtype)

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)
    ref_transformer3d.requires_grad_(False)

    # Lora will work with this...
    if args.use_peft_lora:
        from peft import (LoraConfig, get_peft_model_state_dict,
                          inject_adapter_in_model)
        lora_config = LoraConfig(r=args.rank, lora_alpha=args.network_alpha, target_modules=args.target_name.split(","))
        transformer3d = inject_adapter_in_model(lora_config, transformer3d)

        network = None
    else:
        network = create_network(
            1.0,
            args.rank,
            args.network_alpha,
            text_encoder,
            transformer3d,
            neuron_dropout=None,
            target_name=args.target_name,
            skip_name=args.lora_skip_name,
        )
        network = network.to(weight_dtype)
        network.apply_to(text_encoder, transformer3d, args.train_text_encoder and not args.training_with_video_token_length, True)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if fsdp_stage != 0 or zero_stage == 3:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file
                    safetensor_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model.safetensors")
                    if args.use_peft_lora:
                        network_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(models[-1]), accelerate_state_dict)
                        network_state_dict_kohya = convert_peft_lora_to_kohya_lora(network_state_dict)
                        safetensor_kohya_format_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model_compatible_with_comfyui.safetensors")
                        save_model(safetensor_kohya_format_save_path, network_state_dict_kohya)
                    else:
                        network_state_dict = {}
                        for key in accelerate_state_dict:
                            if "network" in key:
                                network_state_dict[key.replace("network.", "")] = accelerate_state_dict[key].to(weight_dtype)
                    save_file(network_state_dict, safetensor_save_path, metadata={"format": "pt"})

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

            def load_model_hook(models, input_dir):
                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                        batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                    print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        else:
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file
                    safetensor_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model.safetensors")
                    if args.use_peft_lora:
                        network_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(models[-1]), accelerate_state_dict)
                        network_state_dict_kohya = convert_peft_lora_to_kohya_lora(network_state_dict)
                        safetensor_kohya_format_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model_compatible_with_comfyui.safetensors")
                        save_model(safetensor_kohya_format_save_path, network_state_dict_kohya)
                    else:
                        network_state_dict = {}
                        for key in accelerate_state_dict:
                            if "network" in key:
                                network_state_dict[key.replace("network.", "")] = accelerate_state_dict[key].to(weight_dtype)
                    save_file(network_state_dict, safetensor_save_path, metadata={"format": "pt"})

                    if not args.use_deepspeed:
                        for _ in range(len(weights)):
                            weights.pop()

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

            def load_model_hook(models, input_dir):
                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                        batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                    print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except Exception:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    if args.use_peft_lora:
        logging.info("Add peft parameters")
        trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
        trainable_params_optim = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
    else:
        logging.info("Add network parameters")
        trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
        trainable_params_optim = network.prepare_optimizer_params(args.learning_rate / 2, args.learning_rate, args.learning_rate)

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Get the training dataset
    if args.fix_sample_size is not None and args.enable_bucket:
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.random_hw_adapt = False

    # Get the dataset
    train_dataset = TextDataset(
        args.train_data_meta,
        text_drop_ratio=0.0,
    )

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            print(f"worker_init_fn with {_seed + worker_id}")
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
        return _worker_init_fn
    
    if args.enable_bucket:
        def collate_fn(examples):
            new_examples = {}
            new_examples["text"] = []
            for example in examples:
                new_examples["text"].append(example["text"])

            # Encode prompts when enable_text_encoder_in_dataloader=True
            if args.enable_text_encoder_in_dataloader:
                prompt_embeds = encode_prompt(
                    new_examples['text'], device="cpu",
                    text_encoder=text_encoder, 
                    tokenizer=tokenizer,
                )
                new_examples['prompt_embeds'] = prompt_embeds

                neg_prompt_embeds = encode_prompt(
                    ["亮度过高，过曝，严重的色彩失真，低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"], device="cpu",
                    text_encoder=text_encoder, 
                    tokenizer=tokenizer,
                )
                new_examples['neg_prompt_embeds'] = neg_prompt_embeds

            return new_examples

        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = BatchSampler(
            RandomSampler(train_dataset, generator=batch_sampler_generator, k_repeat=args.num_image_per_prompt),
            batch_size=args.train_batch_size, drop_last=True,
        )

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )
    else:
        # DataLoaders creation:
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = BatchSampler(
            RandomSampler(train_dataset, generator=batch_sampler_generator, k_repeat=args.num_image_per_prompt),
            batch_size=args.train_batch_size, drop_last=True,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    # Prepare everything with our `accelerator`.
    if args.use_peft_lora:
        transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer3d, optimizer, train_dataloader, lr_scheduler
        )
    else:
        transformer3d.network = network
        transformer3d = transformer3d.to(dtype=weight_dtype)
        transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer3d, optimizer, train_dataloader, lr_scheduler
        )

    if fsdp_stage != 0 or zero_stage != 0:
        from functools import partial

        from videox_fun.dist import set_multi_gpus_devices, shard_model
        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype, module_to_wrapper=text_encoder.model.layers)
        text_encoder = shard_fn(text_encoder)

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    ref_transformer3d.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    if not args.enable_text_encoder_in_dataloader:
        text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        keys_to_pop = [k for k, v in tracker_config.items() if isinstance(v, list)]
        for k in keys_to_pop:
            tracker_config.pop(k)
            print(f"Removed tracker_config['{k}']")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])

            initial_global_step = global_step

            pkl_path = os.path.join(os.path.join(args.output_dir, path), "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, first_epoch = pickle.load(file)
            else:
                first_epoch = global_step // num_update_steps_per_epoch
            print(f"Load pkl from {pkl_path}. Get first_epoch = {first_epoch}.")

            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
    else:
        initial_global_step = 0

    # function for saving/removing
    def save_model(ckpt_file, unwrapped_nw):
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        if isinstance(unwrapped_nw, dict):
            from safetensors.torch import save_file
            save_file(unwrapped_nw, ckpt_file, metadata={"format": "pt"})
            return ckpt_file
        unwrapped_nw.save_weights(ckpt_file, weight_dtype, None)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.multi_stream:
        # create extra cuda streams to speedup inpaint vae computation
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        transformer3d.eval()
        all_samples = []

        for step, batch in enumerate(train_dataloader):
            #################### SAMPLING (eval mode, fixed model params) ####################
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                texts = batch['text']
                for idx, text_item in enumerate(texts):
                    print(f"[Sanity Check] Sample {idx}: {text_item[:100]}...")

            with torch.no_grad():
                text = batch['text']
                if args.fix_sample_size is not None:
                    local_sample_size = [int(x / 16) * 16 for x in args.fix_sample_size]
                else:
                    if args.random_hw_adapt:
                        aspect_ratio_sample_size = {key : [x / 512 * args.image_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                        if rng is None:
                            aspect_ratio_key = np.random.choice(list(aspect_ratio_sample_size.keys()))
                        else:
                            aspect_ratio_key = rng.choice(list(aspect_ratio_sample_size.keys()))
                        local_sample_size = aspect_ratio_sample_size[aspect_ratio_key]
                        local_sample_size = [int(x / 16) * 16 for x in local_sample_size]
                    else:
                        local_sample_size = [args.image_sample_size, args.image_sample_size]

                vae_scale_factor = (
                    2 ** (len(vae.config.block_out_channels) - 1)
                )
                target_shape = (
                    len(text),
                    vae.latent_channels, 
                    1, 
                    int(local_sample_size[0] // vae_scale_factor),
                    int(local_sample_size[1] // vae_scale_factor), 
                )

                if args.low_vram:
                    vae.to('cpu')
                    torch.cuda.empty_cache()
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to(accelerator.device)

                if args.enable_text_encoder_in_dataloader:
                    prompt_embeds = batch['prompt_embeds'].to(dtype=weight_dtype, device=accelerator.device)
                    neg_prompt_embeds = batch['neg_prompt_embeds'].to(dtype=weight_dtype, device=accelerator.device)
                else:
                    with torch.no_grad():
                        prompt_embeds = encode_prompt(
                            text, 
                            device=accelerator.device,
                            text_encoder=text_encoder, 
                            tokenizer=tokenizer,
                        )
                        # Generate negative embeddings and repeat for batch size
                        neg_prompt_single = encode_prompt(
                            ["亮度过高，过曝，严重的色彩失真，低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"], 
                            device=accelerator.device,
                            text_encoder=text_encoder, 
                            tokenizer=tokenizer,
                        )
                        # Repeat negative embedding for each sample in batch
                        neg_prompt_embeds = neg_prompt_single * len(text)

                if args.low_vram and not args.enable_text_encoder_in_dataloader:
                    text_encoder.to('cpu')
                    torch.cuda.empty_cache()

            shared_noise = torch.randn(
                target_shape, 
                device=accelerator.device, 
                generator=torch_rng, 
                dtype=weight_dtype
            )

            with torch.no_grad():
                collected_data = sample_with_cfg(
                    model=transformer3d,
                    vae=vae,
                    noise=shared_noise.clone(),
                    prompt_embeds=prompt_embeds,
                    neg_prompt_embeds=neg_prompt_embeds,
                    num_steps=args.grpo_num_steps,
                    cfg_scale=args.grpo_cfg_scale,
                    noise_scheduler=noise_scheduler,
                    device=accelerator.device,
                    dtype=weight_dtype,
                    noise_level=args.noise_level,
                    sde_window_size=args.sde_window_size,
                    sde_window_range=tuple(args.sde_window_range),
                )

            latents = torch.stack(collected_data["all_latents"], dim=1) 
            log_probs = torch.stack(collected_data["all_log_probs"], dim=1)  
            timesteps = torch.stack(collected_data["all_timesteps"]).unsqueeze(0).repeat(len(text), 1).to(accelerator.device)
            images = collected_data["images"]
            
            # Compute rewards (supports multiple reward functions)
            if is_multi_reward:
                # Compute rewards from all reward functions
                rewards_dict_local = {}
                for fn_name, fn in loss_fns.items():
                    rewards_dict_local[fn_name] = fn.get_reward(images, text)
                
                individual_rewards = rewards_dict_local
                # Use first reward as placeholder for shape compatibility
                rewards = rewards_dict_local[reward_fn_names[0]]
            else:
                # Single reward function
                rewards = loss_fn.get_reward(images, text)
                individual_rewards = {reward_fn_names[0]: rewards}
            
            # Save sample images to output_dir for logging
            if step == 0:
                num_log_samples = min(4, images.shape[0])
                sample_indices = random.sample(range(images.shape[0]), num_log_samples)
                os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                for log_idx, img_idx in enumerate(sample_indices):
                    log_img = images[img_idx, :, 0].permute(1, 2, 0).cpu().float()
                    log_img = log_img.clamp(0, 1).numpy() * 255
                    from PIL import Image as PILImage
                    PILImage.fromarray(np.uint8(log_img)).save(
                        os.path.join(args.output_dir, "sample", f"epoch{epoch}_rank{accelerator.process_index}_{log_idx}.jpg")
                    )
            
            # Append this batch's samples
            all_samples.append({
                "prompt_embeds": collected_data["prompt_embeds"],
                "negative_prompt_embeds": collected_data["negative_prompt_embeds"],
                "timesteps": timesteps,
                "latents": latents[:, :-1],
                "next_latents": latents[:, 1:],
                "log_probs": log_probs,
                "rewards": rewards,
                "individual_rewards": individual_rewards,
                "prompts": text,
            })

            # Only proceed to training when we've collected enough batches
            if len(all_samples) < args.num_batches_per_epoch:
                continue

            #################### COLLATE SAMPLES & COMPUTE ADVANTAGES ####################
            # Concatenate all sample batches into one large batch
            all_prompts_local = [p for s in all_samples for p in s["prompts"]]
            
            # Collect individual rewards
            all_individual_rewards = {}
            for fn_name in reward_fn_names:
                if fn_name in all_samples[0].get("individual_rewards", {}):
                    all_individual_rewards[fn_name] = torch.cat(
                        [s["individual_rewards"][fn_name] for s in all_samples], dim=0
                    )

            world_size = accelerator.num_processes
            
            # Gather individual rewards from all GPUs
            gathered_individual_rewards = {}
            for fn_name, rewards_tensor in all_individual_rewards.items():
                gathered_individual_rewards[fn_name] = accelerator.gather(rewards_tensor)
            
            # Gather prompts
            gathered_prompts_list = [None] * world_size
            import torch.distributed as dist
            if dist.is_initialized():
                dist.all_gather_object(gathered_prompts_list, all_prompts_local)
                gathered_prompts = [p for sublist in gathered_prompts_list for p in sublist]
            else:
                gathered_prompts = all_prompts_local

            if accelerator.is_main_process:
                for fn_name, ind_rewards in gathered_individual_rewards.items():
                    logger.info(f"Epoch {epoch} Batch index {step} - {fn_name}: mean={ind_rewards.mean():.4f}, std={ind_rewards.std():.4f}")

            # Compute advantages for each reward independently, then combine
            if is_multi_reward and stat_tracker is not None:
                # For multi-reward: compute advantage per reward, then weighted sum
                advantages_per_reward = {}
                for fn_name, gathered_rewards in gathered_individual_rewards.items():
                    gathered_rewards_np = gathered_rewards.cpu().float().numpy()
                    # Use a separate stat_tracker instance per reward (or share with clear)
                    adv = stat_tracker.update(gathered_prompts, gathered_rewards_np)
                    advantages_per_reward[fn_name] = adv
                    stat_tracker.clear()  # Clear for next reward
                
                # Weighted combination of advantages
                advantages_all = None
                for fn_name, adv in advantages_per_reward.items():
                    weight = multi_reward_weights.get(fn_name, 1.0 / len(reward_fn_names))

                    weighted_adv = adv * weight
                    if advantages_all is None:
                        advantages_all = weighted_adv
                    else:
                        advantages_all = advantages_all + weighted_adv
                
                if accelerator.is_main_process:
                    group_size, trained_prompt_num = stat_tracker.get_stats()
                    # Use first reward for zero_std_ratio calculation
                    first_reward_np = list(gathered_individual_rewards.values())[0].cpu().float().numpy()
                    zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(
                        gathered_prompts,
                        {"ori_avg": first_reward_np}
                    )
                    logger.info(f"  Per-prompt stats: group_size={group_size:.2f}, trained_prompts={trained_prompt_num}")
                    logger.info(f"  Combined advantage: mean={advantages_all.mean():.4f}, std={advantages_all.std():.4f}")
                    
                    # Build log dict
                    log_dict = {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                        "combined_advantage_mean": float(advantages_all.mean()),
                        "combined_advantage_std": float(advantages_all.std()),
                    }
                    
                    # Add individual reward/advantage stats
                    for fn_name, ind_rewards in gathered_individual_rewards.items():
                        log_dict[f"{fn_name}_reward_mean"] = ind_rewards.mean().item()
                        log_dict[f"{fn_name}_reward_std"] = ind_rewards.std().item()
                        log_dict[f"{fn_name}_advantage_mean"] = float(advantages_per_reward[fn_name].mean())
                        log_dict[f"{fn_name}_advantage_std"] = float(advantages_per_reward[fn_name].std())
                        log_dict[f"{fn_name}_weight"] = multi_reward_weights.get(fn_name, 1.0 / len(reward_fn_names))
                    
                    accelerator.log(log_dict, step=global_step)

                advantages_all = torch.as_tensor(advantages_all, device=accelerator.device, dtype=weight_dtype)
                if world_size > 1:
                    local_total = len(all_prompts_local)
                    advantages_local = advantages_all.reshape(world_size, local_total)[accelerator.process_index]
                else:
                    advantages_local = advantages_all
                    
            elif stat_tracker is not None:
                # Single reward path
                all_rewards = torch.cat([s["rewards"] for s in all_samples], dim=0)
                gathered_rewards = accelerator.gather(all_rewards)
                gathered_rewards_np = gathered_rewards.cpu().float().numpy()

                advantages_all = stat_tracker.update(gathered_prompts, gathered_rewards_np)

                if accelerator.is_main_process:
                    group_size, trained_prompt_num = stat_tracker.get_stats()
                    zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(
                        gathered_prompts,
                        {"ori_avg": gathered_rewards_np}
                    )
                    logger.info(f"Epoch {epoch} Step {step}: gathered rewards mean={gathered_rewards.mean():.4f}, std={gathered_rewards.std():.4f}")
                    logger.info(f"  Per-prompt stats: group_size={group_size:.2f}, trained_prompts={trained_prompt_num}")
                    
                    log_dict = {
                        "reward_mean": gathered_rewards.mean().item(),
                        "reward_std": gathered_rewards.std().item(),
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    }
                    accelerator.log(log_dict, step=global_step)

                stat_tracker.clear()

                advantages_all = torch.as_tensor(advantages_all, device=accelerator.device, dtype=weight_dtype)
                if world_size > 1:
                    local_total = len(all_prompts_local)
                    advantages_local = advantages_all.reshape(world_size, local_total)[accelerator.process_index]
                else:
                    advantages_local = advantages_all
            else:
                # No stat_tracker - use global normalization
                all_rewards = torch.cat([s["rewards"] for s in all_samples], dim=0)
                gathered_rewards = accelerator.gather(all_rewards)
                advantages_all_global = (gathered_rewards - gathered_rewards.mean()) / (gathered_rewards.std() + 1e-4)
                if world_size > 1:
                    local_total = len(all_prompts_local)
                    advantages_local = advantages_all_global.reshape(world_size, local_total)[accelerator.process_index]
                else:
                    advantages_local = advantages_all_global

            # Expand advantages to timestep dimension and assign back to each sample batch
            advantages_local = advantages_local.unsqueeze(1).repeat(1, num_train_timesteps)
            offset = 0
            for s in all_samples:
                bs = s["rewards"].shape[0]
                s["advantages"] = advantages_local[offset:offset+bs]
                offset += bs
                del s["rewards"]
                del s["prompts"]
                if "individual_rewards" in s:
                    del s["individual_rewards"]

            # Concatenate all samples into one big dict (tensor fields only)
            tensor_keys = [k for k in all_samples[0].keys() if isinstance(all_samples[0][k], torch.Tensor)]
            list_keys = [k for k in all_samples[0].keys() if not isinstance(all_samples[0][k], torch.Tensor)]
            samples_concat = {}
            for k in tensor_keys:
                samples_concat[k] = torch.cat([s[k] for s in all_samples], dim=0)
            for k in list_keys:
                samples_concat[k] = [item for s in all_samples for item in s[k]]

            # Also collect prompt_embeds and neg_prompt_embeds for training
            all_prompt_embeds_flat = []
            all_neg_prompt_embeds_flat = []
            for s in all_samples:
                pe = s["prompt_embeds"]
                ne = s["negative_prompt_embeds"]
                if isinstance(pe, list):
                    all_prompt_embeds_flat.extend(pe)
                    all_neg_prompt_embeds_flat.extend(ne)
                else:
                    all_prompt_embeds_flat.append(pe)
                    all_neg_prompt_embeds_flat.append(ne)

            total_batch_size_collected = samples_concat["timesteps"].shape[0]

            #################### TRAINING ####################
            # Move ref model to GPU for training phase in low_vram mode
            if args.low_vram and args.grpo_beta > 0:
                ref_transformer3d.to(accelerator.device)

            gradient_accumulation_steps_total = (args.gradient_accumulation_steps * num_train_timesteps * args.num_batches_per_epoch) // 2
        
            # Rebatch: split the big concatenated samples back into smaller batches for training
            per_batch_size = total_batch_size_collected // args.num_batches_per_epoch
            samples_batched = []
            for bi in range(args.num_batches_per_epoch):
                batch_slice = slice(bi * per_batch_size, (bi + 1) * per_batch_size)
                one_batch = {k: samples_concat[k][batch_slice] for k in tensor_keys}
                samples_batched.append(one_batch)

            transformer3d.train()
            info = {"approx_kl": [], "clipfrac": [], "policy_loss": [], "kl_loss": [], "loss": []}
            for batch_idx, sample in enumerate(samples_batched):
                # Get corresponding prompt_embeds for this sub-batch
                sub_prompt_embeds = all_prompt_embeds_flat[batch_idx * per_batch_size:(batch_idx + 1) * per_batch_size]
                sub_neg_prompt_embeds = all_neg_prompt_embeds_flat[batch_idx * per_batch_size:(batch_idx + 1) * per_batch_size]

                for j in range(num_train_timesteps):
                    # Manual gradient accumulation: sync every gradient_accumulation_steps_total steps
                    flat_step = batch_idx * num_train_timesteps + j
                    should_sync = (flat_step + 1) % gradient_accumulation_steps_total == 0

                    # Disable gradient sync for accumulation steps, enable for sync steps
                    context = contextlib.nullcontext if should_sync else accelerator.no_sync
                    with context(transformer3d) if not should_sync else contextlib.nullcontext():
                        log_prob, prev_sample_mean, std_dev_t, ref_prev_sample_mean = compute_log_prob(
                            model=transformer3d,
                            vae=vae,
                            sample=sample,
                            step_idx=j,
                            noise_scheduler=noise_scheduler,
                            prompt_embeds=sub_prompt_embeds,
                            neg_prompt_embeds=sub_neg_prompt_embeds,
                            cfg_scale=args.grpo_cfg_scale,
                            noise_level=args.noise_level,
                            dtype=weight_dtype,
                            ref_model=ref_transformer3d if args.grpo_beta > 0 else None,
                        )

                        # GRPO loss computation
                        adv = torch.clamp(
                            sample["advantages"][:, j],
                            -args.adv_clip_max,
                            args.adv_clip_max,
                        )

                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])

                        unclipped_loss = -adv * ratio
                        clipped_loss = -adv * torch.clamp(
                            ratio,
                            1.0 - args.clip_range,
                            1.0 + args.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        policy_loss = policy_loss / gradient_accumulation_steps_total

                        if args.grpo_beta > 0 and ref_prev_sample_mean is not None:
                            kl_loss = ((prev_sample_mean - ref_prev_sample_mean) ** 2).mean(dim=(1,2,3,4)) / (2 * std_dev_t.squeeze() ** 2 + 1e-8)
                            kl_loss = torch.mean(kl_loss)
                            kl_loss = kl_loss / gradient_accumulation_steps_total
                            loss = policy_loss + args.grpo_beta * kl_loss
                        else:
                            kl_loss = torch.tensor(0.0, device=accelerator.device)
                            loss = policy_loss

                        approx_kl = 0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        clipfrac = torch.mean((torch.abs(ratio - 1.0) > args.clip_range).float())
                        info["approx_kl"].append(approx_kl)
                        info["clipfrac"].append(clipfrac)
                        info["policy_loss"].append(policy_loss)
                        info["kl_loss"].append(kl_loss)
                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        train_loss += loss.detach().item() * gradient_accumulation_steps_total

                    if should_sync:
                        accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        info_log = {k: torch.mean(torch.stack([v.float() if isinstance(v, torch.Tensor) else torch.tensor(v) for v in vals])) for k, vals in info.items() if vals}
                        logs = {
                            "step_loss": loss.detach().item(),
                            "lr": lr_scheduler.get_last_lr()[0],
                            "reward": gathered_rewards.mean().item(),
                            "kl": info_log["approx_kl"].item() if "approx_kl" in info_log else 0.0,
                            "clip": info_log["clipfrac"].item() if "clipfrac" in info_log else 0.0,
                        }
                        progress_bar.set_postfix(**logs)
                        accelerator.log(
                            {
                                "train_loss": train_loss,
                                "reward_mean": gathered_rewards.mean().item(),
                                "reward_std": gathered_rewards.std().item(),
                                **{k: v.item() for k, v in info_log.items()},
                            },
                            step=global_step,
                        )
                        train_loss = 0.0
                        info = {"approx_kl": [], "clipfrac": [], "policy_loss": [], "kl_loss": [], "loss": []}

            transformer3d.eval()

            # Move ref model back to CPU after training phase in low_vram mode
            if args.low_vram and args.grpo_beta > 0:
                ref_transformer3d.to('cpu')
                torch.cuda.empty_cache()

            # Update progress and global_step
            progress_bar.update(1)
            global_step += 1

            # Reset sample buffer for next collection round
            all_samples = []

            if global_step % args.checkpointing_steps == 0:
                if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    if not args.save_state:
                        if args.use_peft_lora:
                            safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                            network_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(transformer3d))
                            save_model(safetensor_save_path, network_state_dict)
                            safetensor_kohya_format_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-compatible_with_comfyui.safetensors")
                            network_state_dict_kohya = convert_peft_lora_to_kohya_lora(network_state_dict)
                            save_model(safetensor_kohya_format_save_path, network_state_dict_kohya)
                            logger.info(f"Saved safetensor to {safetensor_save_path}")
                        else:
                            safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                            save_model(safetensor_save_path, accelerator.unwrap_model(network))
                            logger.info(f"Saved safetensor to {safetensor_save_path}")
                    else:
                        accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(accelerator_save_path)
                        logger.info(f"Saved state to {accelerator_save_path}")

            if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    transformer3d,
                    network,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                )

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            log_validation(
                vae,
                text_encoder,
                tokenizer,
                transformer3d,
                network,
                args,
                accelerator,
                weight_dtype,
                global_step,
            )

        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if not args.save_state:
            if args.use_peft_lora:
                safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                network_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(transformer3d))
                save_model(safetensor_save_path, network_state_dict)

                safetensor_kohya_format_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-compatible_with_comfyui.safetensors")
                network_state_dict_kohya = convert_peft_lora_to_kohya_lora(network_state_dict)
                save_model(safetensor_kohya_format_save_path, network_state_dict_kohya)
                logger.info(f"Saved safetensor to {safetensor_save_path}")
            else:
                safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                save_model(safetensor_save_path, accelerator.unwrap_model(network))
                logger.info(f"Saved safetensor to {safetensor_save_path}")
        else:
            accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(accelerator_save_path)
            logger.info(f"Saved state to {accelerator_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
