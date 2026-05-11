# ---------------------------------------------------------------------------
# This script computes reward scores for video-prompt pairs using any reward
# model defined in videox_fun.reward.reward_fn (e.g. HPSReward, MPSReward,
# VideoAlignReward, HPSv3Reward, etc.).
#
# Argument style is kept identical to train_reward_lora.py so that users can
# directly copy --reward_fn and --reward_fn_kwargs from their training shell
# scripts without modification.
#
# Typical usage (single model):
#   1) AestheticReward (v2.5, custom predictor)
#      python scripts/get_rewards_1video.py \
#          --reward_fn AestheticReward \
#          --reward_fn_kwargs '{"version": "v2.5"}' \
#          --video_path asset/1.mp4 --prompts "1girl"
#
#   2) HPSv3Reward
#      python scripts/get_rewards_1video.py \
#          --reward_fn HPSv3Reward \
#          --reward_fn_kwargs '{"checkpoint_path": "models/Diffusion_Transformer/HPSv3/HPSv3.safetensors"}' \
#          --video_path asset/1.mp4 --prompts "1girl"
#
#   3) MPSReward
#      python scripts/get_rewards_1video.py \
#          --reward_fn MPSReward \
#          --reward_fn_kwargs '{}' \
#          --video_path asset/1.mp4 --prompts "1girl"
#
#   4) VideoAlignReward
#      python scripts/get_rewards_1video.py \
#          --reward_fn VideoAlignReward \
#          --reward_fn_kwargs '{"model_path": "models/Diffusion_Transformer/VideoReward/", "fps": 16, "reward_dim": "Overall"}' \
#          --video_path asset/1.mp4 --prompts "1girl"
#
# Notes:
#   * --num_sampled_frames defaults to -1, meaning ALL frames are fed to the
#     reward model.  For long videos this can be very slow / VRAM-heavy.
#   * Videos are processed one-by-one (batch_size == 1) to keep peak VRAM low.
#   * Both --video_path and --prompts accept multiple values; they must have
#     the same length and are matched by position.
# ---------------------------------------------------------------------------
import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import videox_fun.reward.reward_fn as reward_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute rewards for videos using a specified reward model. "
                    "Example: python scripts/get_rewards.py --reward_fn HPSReward "
                    '--reward_fn_kwargs \'{"version": "v2.1"}\' '
                    '--video_path video1.mp4 video2.mp4 --prompts "A cat plays with a ball" "A dog runs on grass"'
    )
    parser.add_argument(
        "--reward_fn",
        type=str,
        default="HPSReward",
        help='Reward model class name. Options: AestheticReward, HPSReward, PickScoreReward, '
             'MPSReward, HPSv3Reward, VideoAlignReward.',
    )
    parser.add_argument(
        "--reward_fn_kwargs",
        type=str,
        default=None,
        help='JSON string of keyword arguments passed to the reward model constructor. '
             'E.g., \'{"version": "v2.1", "device": "cuda"}\'',
    )
    parser.add_argument(
        "--video_path",
        type=str,
        nargs="+",
        required=True,
        help="List of video file paths.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="List of text prompts corresponding to each video.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the reward model on (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for the reward model and input tensors.",
    )
    parser.add_argument(
        "--num_sampled_frames",
        type=int,
        default=-1,
        help="Number of uniformly sampled frames from each video. Set to -1 (default) to use all frames.",
    )
    args = parser.parse_args()
    return args


def parse_video_prompt_pairs(video_paths, prompts):
    """Zip video paths and prompts into (video_path, prompt) tuples."""
    if len(video_paths) != len(prompts):
        raise ValueError(
            f"Number of --video_path ({len(video_paths)}) and --prompts ({len(prompts)}) must be equal."
        )
    pairs = []
    for video_path, prompt in zip(video_paths, prompts):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        pairs.append((video_path, prompt))
    return pairs


def sample_frames_from_video(video_path, num_sampled_frames):
    """Sample frames from a video and return a [T, C, H, W] float32 tensor in [0, 1].

    If num_sampled_frames is -1, all frames are used.
    """
    vr = VideoReader(video_path)
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    if num_sampled_frames == -1:
        sampled_frame_indices = np.arange(total_frames, dtype=int)
    else:
        sampled_frame_indices = np.linspace(0, total_frames, num_sampled_frames, endpoint=False, dtype=int)
    sampled_frames = vr.get_batch(sampled_frame_indices).asnumpy()  # [T, H, W, C]
    to_tensor = transforms.ToTensor()
    frames_tensor = torch.stack([to_tensor(frame) for frame in sampled_frames], dim=0)  # [T, C, H, W]
    return frames_tensor


def main():
    args = parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Parse reward constructor kwargs
    reward_fn_kwargs = {}
    if args.reward_fn_kwargs is not None:
        reward_fn_kwargs = json.loads(args.reward_fn_kwargs)
    # Ensure device and dtype are passed (user can override via reward_fn_kwargs)
    reward_fn_kwargs.setdefault("device", args.device)
    reward_fn_kwargs.setdefault("dtype", dtype)

    # Parse video:prompt pairs
    pairs = parse_video_prompt_pairs(args.video_path, args.prompts)
    video_paths = [p[0] for p in pairs]
    prompts = [p[1] for p in pairs]

    # Sample frames from each video
    frames_list = []
    for video_path in video_paths:
        frames = sample_frames_from_video(video_path, args.num_sampled_frames)
        frames_list.append(frames)

    # Stack to [B, T, C, H, W] then rearrange to [B, C, T, H, W] as required by reward_fn
    batch_frames = torch.stack(frames_list, dim=0)
    batch_frames = rearrange(batch_frames, "b t c h w -> b c t h w")

    # Instantiate reward model
    if not hasattr(reward_fn, args.reward_fn):
        raise ValueError(
            f"Reward function '{args.reward_fn}' not found in videox_fun.reward.reward_fn. "
            f"Available: {reward_fn.__all__}"
        )
    reward_model_cls = getattr(reward_fn, args.reward_fn)
    print(f"Loading reward model: {args.reward_fn} with kwargs {reward_fn_kwargs}")
    reward_model = reward_model_cls(**reward_fn_kwargs)
    reward_model.eval()

    # Process one video at a time to minimize VRAM usage
    batch_size = 1
    num_batches = len(pairs)

    all_per_sample_rewards = []
    total_loss = 0.0
    total_reward = 0.0

    with torch.no_grad():
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, len(pairs))
            batch_frames_slice = batch_frames[start:end].to(args.device, dtype=dtype)
            batch_prompts_slice = prompts[start:end]

            loss, reward = reward_model(batch_frames_slice, batch_prompts_slice)
            per_sample_rewards = reward_model.get_reward(batch_frames_slice, batch_prompts_slice)

            total_loss += loss.item() * (end - start)
            total_reward += reward.item() * (end - start)
            all_per_sample_rewards.append(per_sample_rewards.cpu())

    # Aggregate results
    all_per_sample_rewards = torch.cat(all_per_sample_rewards, dim=0)
    avg_loss = total_loss / len(pairs)
    avg_reward = total_reward / len(pairs)

    print("\n" + "=" * 60)
    print(f"Reward Model : {args.reward_fn}")
    print(f"Num Videos   : {len(pairs)}")
    print(f"Num Frames   : {args.num_sampled_frames}")
    print(f"Device       : {args.device}")
    print(f"Dtype        : {args.dtype}")
    print("-" * 60)
    print(f"Average Loss   : {avg_loss:.6f}")
    print(f"Average Reward : {avg_reward:.6f}")
    print("-" * 60)
    print("Per-sample rewards:")
    for idx, (vp, prompt) in enumerate(pairs):
        print(f"  [{idx}] Reward: {all_per_sample_rewards[idx].item():.6f} | Video: {vp} | Prompt: {prompt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
