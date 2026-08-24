import csv
import gc
import io
import json
import math
import os
import random
from contextlib import contextmanager
from random import shuffle
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from packaging import version as pver
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset

try:
    from decord import VideoReader
except ImportError:
    from .utils import AVVideoReader as VideoReader

from .utils import (VIDEO_READER_TIMEOUT, VideoReader_contextmanager,
                    get_random_mask, get_video_reader_batch, padding_image,
                    process_pose_file, process_pose_params, resize_frame,
                    resize_image_with_target_area)


class ImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'image':[], 'video':[]}

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'image')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]


class ImageVideoDataset(Dataset):
    """Dataset for mixed image and video training with inpainting support."""
    def __init__(
        self,
        ann_path, 
        data_root=None,
        video_sample_size=512, 
        video_sample_stride=4, 
        video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        enable_inpaint=False,
        inpaint_mask_fill_value=0,
        return_file_name=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # Balance image/video ratio by duplicating video entries
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # Enable bucket training (TODO)
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.inpaint_mask_fill_value = inpaint_mask_fill_value
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params: resize, center crop, normalize to [-1, 1]
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size      = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms       = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params: resize, center crop, normalize to [-1, 1]
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        # Use larger side for consistent resizing across images and videos
        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))

    def get_batch(self, idx):
        """Load and preprocess a single video or image sample."""
        data_info = self.dataset[idx % len(self.dataset)]
        
        if data_info.get('type', 'image')=='video':
            video_id, text = data_info['file_path'], data_info['text']

            # Resolve video path
            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                # Calculate frame sampling range with length dropout
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")
                min_video_sample_n_frames = getattr(self, "min_video_sample_n_frames", None)
                if min_video_sample_n_frames is not None and min_sample_n_frames < min_video_sample_n_frames:
                    raise ValueError(
                        f"Video too short: sampled {min_sample_n_frames} frames < required {min_video_sample_n_frames}."
                    )

                # Select contiguous clip with random start position
                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    raw_frames = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    # Resize each frame and free the original array early to reduce peak memory
                    resized_frames = []
                    for i in range(len(raw_frames)):
                        resized_frames.append(resize_frame(raw_frames[i], self.larger_side_of_image_and_video))
                    del raw_frames
                    pixel_values = np.stack(resized_frames)
                    del resized_frames
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            # Release video reader early to free file handles and decode buffers
            del video_reader

            # Convert to tensor, normalize to [-1, 1], apply transforms
            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
                pixel_values = self.video_transforms(pixel_values)
            
            # Random text dropout for classifier-free guidance
            if random.random() < self.text_drop_ratio:
                text = ''
            return pixel_values, text, 'video', video_dir
        else:
            # Load and preprocess image
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            # Random text dropout for classifier-free guidance
            if random.random() < self.text_drop_ratio:
                text = ''
            return image, text, 'image', image_path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Get a sample with retry on failure."""
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, name, data_type, file_path = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            # Fill masked regions with configurable value (default -1.0, some models use 0.0)
            mask_pixel_values = torch.where(mask.bool(), torch.tensor(self.inpaint_mask_fill_value), pixel_values)
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


class LingbotImageVideoDataset(ImageVideoDataset):
    """Dataset variant for LingBot-World camera-controlled I2V training.

    Extends :class:`ImageVideoDataset` and, for each video sample, additionally
    loads the paired camera trajectory (``poses.npy``) and intrinsics
    (``intrinsics.npy``) referenced through ``action_path`` in the annotation
    file. The sampled RGB frame indices are also returned so the training loop
    can slice the trajectory in sync with the video clip.

    Annotation entries are expected to have the form::

        {
            "file_path": "videos/xxx.mp4",
            "text": "...",
            "type": "video",
            "action_path": "actions/xxx"   # dir with poses.npy + intrinsics.npy
        }

    Samples without ``action_path`` (typically images) fall back to the base
    behavior; downstream code should treat ``sample["action_c2ws"] is None``
    as "no camera control".
    """

    def __init__(self, *args, action_data_root=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Where to resolve ``action_path`` from the annotation file. Defaults
        # to ``data_root`` so a single top-level dataset directory works out of
        # the box (mirroring the video path resolution).
        self.action_data_root = action_data_root if action_data_root is not None else self.data_root

    def _resolve_action_path(self, action_path):
        if action_path is None:
            return None
        if os.path.isabs(action_path) or self.action_data_root is None:
            return action_path
        return os.path.join(self.action_data_root, action_path)

    def _load_camera(self, action_path):
        """Load poses.npy / intrinsics.npy from ``action_path``.

        Returns ``(c2ws[F, 4, 4], intrinsics[N, 4])`` numpy arrays, or
        ``(None, None)`` if either file is missing.
        """
        if action_path is None:
            return None, None
        resolved = self._resolve_action_path(action_path)
        pose_file = os.path.join(resolved, "poses.npy")
        intr_file = os.path.join(resolved, "intrinsics.npy")
        if not (os.path.isfile(pose_file) and os.path.isfile(intr_file)):
            return None, None
        c2ws = np.load(pose_file)
        intrinsics = np.load(intr_file)
        return c2ws, intrinsics

    def get_batch(self, idx):
        """Same as base class but also returns the sampled frame indices."""
        data_info = self.dataset[idx % len(self.dataset)]

        if data_info.get('type', 'image') == 'video':
            video_id, text = data_info['file_path'], data_info['text']

            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames,
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")
                min_video_sample_n_frames = getattr(self, "min_video_sample_n_frames", None)
                if min_video_sample_n_frames is not None and min_sample_n_frames < min_video_sample_n_frames:
                    raise ValueError(
                        f"Video too short: sampled {min_sample_n_frames} frames < required {min_video_sample_n_frames}."
                    )

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    raw_frames = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = [resize_frame(raw_frames[i], self.larger_side_of_image_and_video)
                                      for i in range(len(raw_frames))]
                    del raw_frames
                    pixel_values = np.stack(resized_frames)
                    del resized_frames
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            del video_reader

            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
                pixel_values = self.video_transforms(pixel_values)

            if random.random() < self.text_drop_ratio:
                text = ''

            action_path = data_info.get('action_path', None)
            c2ws, intrinsics = self._load_camera(action_path)
            if c2ws is not None:
                # Align the trajectory with the sampled frames.
                if len(c2ws) < int(batch_index.max()) + 1:
                    # Trajectory is too short: fall back to no camera control.
                    sampled_c2ws = None
                    sampled_intrinsics = None
                else:
                    sampled_c2ws = c2ws[batch_index]
                    sampled_intrinsics = intrinsics
            else:
                sampled_c2ws = None
                sampled_intrinsics = None

            # Per-sample calibration resolution of ``intrinsics.npy``. Required
            # for camera-controlled samples so ``get_Ks_transformed`` can rescale
            # the intrinsics to the training bucket. If a sample carries a camera
            # trajectory but is missing these fields, treat it as invalid here so
            # ``__getitem__`` skips it and re-samples another entry (rather than
            # crashing the training loop later).
            org_h = data_info.get('intrinsics_org_height', None)
            org_w = data_info.get('intrinsics_org_width', None)
            if sampled_c2ws is not None:
                if org_h is None or org_w is None:
                    raise ValueError(
                        f"Skipping camera-controlled sample {video_id!r}: missing "
                        "`intrinsics_org_height`/`intrinsics_org_width` in the annotation."
                    )
                sampled_intrinsics_org_hw = (int(org_h), int(org_w))
            else:
                sampled_intrinsics_org_hw = None

            return pixel_values, text, 'video', video_dir, sampled_c2ws, sampled_intrinsics, sampled_intrinsics_org_hw
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            if random.random() < self.text_drop_ratio:
                text = ''
            return image, text, 'image', image_path, None, None, None

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, name, data_type, file_path, c2ws, intrinsics, intrinsics_org_hw = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                # Camera fields are optional; None means "no camera control".
                sample["action_c2ws"] = c2ws
                sample["action_intrinsics"] = intrinsics
                # Optional per-sample intrinsics calibration resolution (H, W);
                # None means "use the global CLI default".
                sample["action_intrinsics_org_hw"] = intrinsics_org_hw
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length - 1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = torch.where(mask.bool(), torch.tensor(self.inpaint_mask_fill_value), pixel_values)
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


class ImageVideoControlDataset(Dataset):
    """Dataset for control-based image and video training (Canny, Depth, Pose, etc.)."""
    def __init__(
        self,
        ann_path, 
        data_root=None,
        video_sample_size=512, 
        video_sample_stride=4, 
        video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        enable_inpaint=False,
        inpaint_mask_fill_value=0,
        enable_camera_info=False,
        enable_subject_info=False,
        padding_subject_info=True,
        return_file_name=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # Balance image/video ratio by duplicating video entries
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # Enable bucket training (TODO)
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.inpaint_mask_fill_value = inpaint_mask_fill_value
        self.enable_camera_info = enable_camera_info
        self.enable_subject_info = enable_subject_info
        self.padding_subject_info = padding_subject_info
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params: resize, center crop, normalize to [-1, 1]
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size      = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms       = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        if self.enable_camera_info:
            # Camera info only needs resize and crop, no normalization
            self.video_transforms_camera = transforms.Compose(
                [
                    transforms.Resize(min(self.video_sample_size)),
                    transforms.CenterCrop(self.video_sample_size)
                ]
            )

        # Image params: resize, center crop, normalize to [-1, 1]
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        # Use larger side for consistent resizing across images and videos
        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))
    
    def get_batch(self, idx):
        """Load and preprocess a single video or image sample with control signals."""
        data_info = self.dataset[idx % len(self.dataset)]
        
        if data_info.get('type', 'image')=='video':
            video_id, text = data_info['file_path'], data_info['text']

            # Resolve video path
            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                # Calculate frame sampling range with length dropout
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")
                min_video_sample_n_frames = getattr(self, "min_video_sample_n_frames", None)
                if min_video_sample_n_frames is not None and min_sample_n_frames < min_video_sample_n_frames:
                    raise ValueError(
                        f"Video too short: sampled {min_sample_n_frames} frames < required {min_video_sample_n_frames}."
                    )

                # Select contiguous clip with random start position
                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    raw_frames = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    # Resize each frame and free the original array early to reduce peak memory
                    resized_frames = []
                    for i in range(len(raw_frames)):
                        resized_frames.append(resize_frame(raw_frames[i], self.larger_side_of_image_and_video))
                    del raw_frames
                    pixel_values = np.stack(resized_frames)
                    del resized_frames
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            # Release video reader early to free file handles and decode buffers
            del video_reader

            # Convert to tensor, normalize to [-1, 1], apply transforms
            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
                pixel_values = self.video_transforms(pixel_values)
            
            # Random text dropout for classifier-free guidance
            if random.random() < self.text_drop_ratio:
                text = ''

            # Load control signal (Canny/Depth/Pose/Camera)
            control_video_id = data_info['control_file_path']
            if control_video_id is not None:
                if self.data_root is None:
                    control_video_path = control_video_id
                else:
                    control_video_path = os.path.join(self.data_root, control_video_id)
            else:
                control_video_path = None
            
            if self.enable_camera_info:
                # Camera parameters from txt file
                if control_video_path is not None and control_video_path.lower().endswith('.txt'):
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)
                        control_camera_values = process_pose_file(control_video_path, width=self.video_sample_size[1], height=self.video_sample_size[0])
                        control_camera_values = torch.from_numpy(control_camera_values).permute(0, 3, 1, 2).contiguous()
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)
                        control_camera_values = self.video_transforms_camera(control_camera_values)
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)
                        control_camera_values = process_pose_file(control_video_path, width=self.video_sample_size[1], height=self.video_sample_size[0], return_poses=True)
                        control_camera_values = torch.from_numpy(np.array(control_camera_values)).unsqueeze(0).unsqueeze(0)
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)[0][0]
                        control_camera_values = np.array([control_camera_values[index] for index in batch_index])
                else:
                    control_pixel_values = torch.zeros_like(pixel_values) if not self.enable_bucket else np.zeros_like(pixel_values)
                    control_camera_values = None
            else:
                # Load control video (Canny/Depth/Pose)
                if control_video_path is not None:
                    with VideoReader_contextmanager(control_video_path, num_threads=2) as control_video_reader:
                        try:
                            sample_args = (control_video_reader, batch_index)
                            control_raw_frames = func_timeout(
                                VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                            )
                            # Resize each frame and free the original array early
                            resized_frames = []
                            for i in range(len(control_raw_frames)):
                                resized_frames.append(resize_frame(control_raw_frames[i], self.larger_side_of_image_and_video))
                            del control_raw_frames
                            control_pixel_values = np.stack(resized_frames)
                            del resized_frames
                        except FunctionTimedOut:
                            raise ValueError(f"Read {idx} timeout.")
                        except Exception as e:
                            raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                    # Release control video reader early
                    del control_video_reader

                    # Convert to tensor and apply transforms
                    if not self.enable_bucket:
                        control_pixel_values = torch.from_numpy(control_pixel_values).permute(0, 3, 1, 2).contiguous()
                        control_pixel_values = control_pixel_values / 255.
                        control_pixel_values = self.video_transforms(control_pixel_values)
                else:
                    control_pixel_values = torch.zeros_like(pixel_values) if not self.enable_bucket else np.zeros_like(pixel_values)
                control_camera_values = None
            
            # Load subject reference images (for subject-driven generation)
            if self.enable_subject_info:
                visual_height, visual_width = pixel_values.shape[-2:] if not self.enable_bucket else pixel_values.shape[1:3]

                subject_id = data_info.get('object_file_path', [])
                shuffle(subject_id)
                subject_images = []
                for i in range(min(len(subject_id), 4)):
                    subject_image_path = subject_id[i] if self.data_root is None else os.path.join(self.data_root, subject_id[i])
                    subject_image = Image.open(subject_image_path)

                    if self.padding_subject_info:
                        img = padding_image(subject_image, visual_width, visual_height)
                    else:
                        img = resize_image_with_target_area(subject_image, 1024 * 1024)

                    # Random horizontal flip for augmentation
                    if random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    subject_images.append(np.array(img))
                
                subject_image = np.array(subject_images) if self.padding_subject_info else subject_images
            else:
                subject_image = None

            return pixel_values, control_pixel_values, subject_image, control_camera_values, text, "video"
        else:
            # Load and preprocess image
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            
            # Random text dropout for classifier-free guidance
            if random.random() < self.text_drop_ratio:
                text = ''

            # Load control image
            control_image_id = data_info['control_file_path']
            if self.data_root is None:
                control_image_path = control_image_id
            else:
                control_image_path = os.path.join(self.data_root, control_image_id)

            control_image = Image.open(control_image_path).convert('RGB')
            if not self.enable_bucket:
                control_image = self.image_transforms(control_image).unsqueeze(0)
            else:
                control_image = np.expand_dims(np.array(control_image), 0)
            
            # Load subject reference images
            if self.enable_subject_info:
                visual_height, visual_width = image.shape[-2:] if not self.enable_bucket else image.shape[1:3]

                subject_id = data_info.get('object_file_path', [])
                shuffle(subject_id)
                subject_images = []
                for i in range(min(len(subject_id), 4)):
                    subject_image_path = subject_id[i] if self.data_root is None else os.path.join(self.data_root, subject_id[i])
                    subject_image = Image.open(subject_image_path).convert('RGB')

                    if self.padding_subject_info:
                        img = padding_image(subject_image, visual_width, visual_height)
                    else:
                        img = resize_image_with_target_area(subject_image, 1024 * 1024)

                    # Random horizontal flip for augmentation
                    if random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    subject_images.append(np.array(img))
                
                subject_image = np.array(subject_images) if self.padding_subject_info else subject_images
            else:
                subject_image = None

            return image, control_image, subject_image, None, text, 'image'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Get a sample with retry on failure."""
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, control_pixel_values, subject_image, control_camera_values, name, data_type = self.get_batch(idx)

                sample["pixel_values"] = pixel_values
                sample["control_pixel_values"] = control_pixel_values
                sample["subject_image"] = subject_image
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx

                if self.enable_camera_info:
                    sample["control_camera_values"] = control_camera_values
                
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(data_info['file_path'])

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            # Fill masked regions with configurable value (default -1.0, some models use 0.0)
            mask_pixel_values = torch.where(mask.bool(), torch.tensor(self.inpaint_mask_fill_value), pixel_values)
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


class ImageVideoSafetensorsDataset(Dataset):
    """Dataset for loading preprocessed latents in safetensors format.

    Supports two JSON entry formats produced by ``train_preprocess.py``:

    1. Single-file mode (default preprocess output)::

           {"file_path": "/path/to/scene.safetensors"}

       The whole state dict is loaded from a single ``.safetensors`` file.

    2. Per-tensor mode (``--save_per_tensor`` preprocess output)::

           {
               "file_path": "/path/to/scene_dir",
               "latents": "/path/to/scene_dir/latents.safetensors",
               "prompt_embeds": "/path/to/scene_dir/prompt_embeds.safetensors",
               ...
           }

       Each key whose value is a ``.safetensors`` path is loaded individually
       and merged into the returned ``state_dict``. The inner safetensors file
       stores the tensor under the same key name, so a plain ``dict.update``
       is sufficient to assemble the final state dict.
    """
    def __init__(
        self,
        ann_path,
        data_root=None,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))

        self.data_root = data_root
        self.dataset = dataset
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

    def _resolve_path(self, path):
        if self.data_root is None:
            return path
        return os.path.join(self.data_root, path)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Load preprocessed latents, supporting both single-file and per-tensor formats."""
        item = self.dataset[idx]
        file_path = item.get("file_path")

        # Single-file mode: ``file_path`` points to a ``.safetensors`` archive
        # that already holds every preprocessed tensor.
        # Fall through to per-tensor mode when the key is absent or the file does not exist.
        if (
            file_path is not None
            and file_path.endswith(".safetensors")
            and os.path.exists(self._resolve_path(file_path))
        ):
            return load_file(self._resolve_path(file_path))

        # Per-tensor mode: iterate over every ``.safetensors`` entry in the
        # JSON record and merge their contents into a single state dict.
        state_dict = {}
        for key, value in item.items():
            if key == "file_path":
                continue
            if isinstance(value, str) and value.endswith(".safetensors"):
                tensor_path = self._resolve_path(value)
                state_dict.update(load_file(tensor_path))
        return state_dict


class TextDataset(Dataset):
    """Dataset for text-only training (e.g., text encoder fine-tuning)."""
    def __init__(self, ann_path, text_drop_ratio=0.0):
        print(f"loading annotations from {ann_path} ...")
        with open(ann_path, 'r') as f:
            self.dataset = json.load(f)
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        self.text_drop_ratio = text_drop_ratio

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Get a single text sample with retry on failure."""
        while True:
            try:
                item = self.dataset[idx]
                text = item['text']

                # Randomly drop text for classifier-free guidance
                if random.random() < self.text_drop_ratio:
                    text = ''

                sample = {
                    "text": text,
                    "idx": idx
                }
                return sample

            except Exception as e:
                print(f"Error at index {idx}: {e}, retrying with random index...")
                idx = np.random.randint(0, self.length - 1)