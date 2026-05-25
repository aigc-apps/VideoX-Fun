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
from decord import VideoReader
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from packaging import version as pver
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset

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
    """Dataset for loading preprocessed latents in safetensors format."""
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Load a single safetensors file containing preprocessed latents."""
        if self.data_root is None:
            path = self.dataset[idx]["file_path"]
        else:
            path = os.path.join(self.data_root, self.dataset[idx]["file_path"])
        state_dict = load_file(path)
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