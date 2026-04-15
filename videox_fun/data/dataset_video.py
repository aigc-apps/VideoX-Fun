import csv
import json
import math
import os
import random

import cv2
import librosa
import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from PIL import Image
from torch.utils.data.dataset import Dataset

from .utils import (VIDEO_READER_TIMEOUT, VideoReader_contextmanager,
                    get_random_mask, get_video_reader_batch, resize_frame)


class WebVid10M(Dataset):
    def __init__(
        self,
        csv_path, 
        video_folder,
        sample_size=256, 
        sample_stride=4, 
        sample_n_frames=16,
        enable_bucket=False, 
        enable_inpaint=False, 
        is_image=False,
    ):
        print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.enable_bucket   = enable_bucket
        self.enable_inpaint  = enable_inpaint
        self.is_image        = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
        
        video_dir    = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        if not self.enable_bucket:
            pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.
            del video_reader
        else:
            pixel_values = video_reader.get_batch(batch_index).asnumpy()

        if self.is_image:
            pixel_values = pixel_values[0]
        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                print("Error info:", e)
                idx = random.randint(0, self.length-1)

        if not self.enable_bucket:
            pixel_values = self.pixel_transforms(pixel_values)
        if self.enable_inpaint:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample = dict(pixel_values=pixel_values, mask_pixel_values=mask_pixel_values, mask=mask, text=name)
        else:
            sample = dict(pixel_values=pixel_values, text=name)
        return sample


class VideoDataset(Dataset):
    """Dataset for video training with inpainting support."""
    def __init__(
        self,
        ann_path, 
        data_root=None,
        sample_size=256, 
        sample_stride=4, 
        sample_n_frames=16,
        enable_bucket=False, 
        enable_inpaint=False,
        inpaint_mask_fill_value=0,
        video_length_drop_start=0.0,
        video_length_drop_end=1.0,
        text_drop_ratio=0.1,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        self.dataset = json.load(open(ann_path, 'r'))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.data_root = data_root
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.enable_bucket = enable_bucket
        self.enable_inpaint = enable_inpaint
        self.inpaint_mask_fill_value = inpaint_mask_fill_value
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end
        self.text_drop_ratio = text_drop_ratio
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose(
            [
                transforms.Resize(sample_size[0]),
                transforms.CenterCrop(sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    
    def get_batch(self, idx):
        """Load and preprocess a single video sample."""
        video_dict = self.dataset[idx]
        video_id, text = video_dict['file_path'], video_dict['text']

        # Resolve video path
        if self.data_root is None:
            video_dir = video_id
        else:
            video_dir = os.path.join(self.data_root, video_id)

        with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
            # Calculate frame sampling range with length dropout
            min_sample_n_frames = min(
                self.sample_n_frames, 
                int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.sample_stride)
            )
            if min_sample_n_frames == 0:
                raise ValueError(f"No Frames in video.")

            # Select contiguous clip with random start position
            video_length = int(self.video_length_drop_end * len(video_reader))
            clip_length = min(video_length, (min_sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

            try:
                sample_args = (video_reader, batch_index)
                pixel_values = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                )
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            # Convert to tensor, normalize to [-1, 1], apply transforms
            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
                del video_reader
                pixel_values = self.pixel_transforms(pixel_values)
            
            # Random text dropout for classifier-free guidance
            if random.random() < self.text_drop_ratio:
                text = ''
            return pixel_values, text

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Get a sample with retry on failure."""
        while True:
            sample = {}
            try:
                pixel_values, name = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["idx"] = idx
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

            # Prepare CLIP pixel values for first frame
            sample["clip_pixel_values"] = (sample["pixel_values"][0].permute(1, 2, 0).contiguous() * 0.5 + 0.5) * 255

        return sample


class VideoSpeechDataset(Dataset):
    """Dataset for video-speech paired training with motion and inpainting support."""
    def __init__(
        self,
        ann_path, 
        data_root=None,
        video_sample_size=512,
        video_sample_stride=4,
        video_sample_n_frames=16,
        enable_bucket=False, 
        enable_inpaint=False,
        inpaint_mask_fill_value=0,
        audio_sr=16000,
        text_drop_ratio=0.1,
        enable_motion_info=False,
        motion_frames=73,
        return_file_name=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        self.dataset = json.load(open(ann_path, 'r'))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.data_root = data_root
        self.enable_bucket = enable_bucket
        self.enable_inpaint = enable_inpaint
        self.inpaint_mask_fill_value = inpaint_mask_fill_value
        self.audio_sr = audio_sr
        self.text_drop_ratio = text_drop_ratio
        self.enable_motion_info = enable_motion_info
        self.motion_frames = motion_frames
        self.return_file_name = return_file_name
        
        # Video params: resize, center crop, normalize to [-1, 1]
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.pixel_transforms = transforms.Compose(
            [
                transforms.Resize(self.video_sample_size[0]),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    
    def get_batch(self, idx):
        """Load and preprocess a single video sample with corresponding audio."""
        video_dict = self.dataset[idx]
        video_id, text = video_dict['file_path'], video_dict['text']
        audio_id = video_dict['audio_path']

        # Resolve video and audio paths
        if self.data_root is None:
            video_path = video_id
            audio_path = audio_id
        else:
            video_path = os.path.join(self.data_root, video_id)
            audio_path = os.path.join(self.data_root, audio_id)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found for {video_path}")

        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            total_frames = len(video_reader)
            fps = video_reader.get_avg_fps()

            # Adjust stride to avoid fps > 30
            local_video_sample_stride = self.video_sample_stride
            new_fps = int(fps // local_video_sample_stride)
            while new_fps > 30:
                local_video_sample_stride = local_video_sample_stride + 1
                new_fps = int(fps // local_video_sample_stride)

            # Calculate the actual number of sampled frames (considering boundaries)
            max_possible_frames = (total_frames - 1) // local_video_sample_stride + 1
            actual_n_frames = min(self.video_sample_n_frames, max_possible_frames)
            if actual_n_frames <= 0:
                raise ValueError(f"Video too short: {video_path}")

            # Randomly select the starting frame
            max_start = total_frames - (actual_n_frames - 1) * local_video_sample_stride - 1
            start_frame = random.randint(0, max_start) if max_start > 0 else 0
            frame_indices = [start_frame + i * local_video_sample_stride for i in range(actual_n_frames)]

            # Read video frames
            try:
                sample_args = (video_reader, frame_indices)
                raw_frames = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                )
                # Resize each frame and free the original array early to reduce peak memory
                resized_frames = []
                for i in range(len(raw_frames)):
                    resized_frames.append(resize_frame(raw_frames[i], max(self.video_sample_size)))
                del raw_frames
                pixel_values = np.array(resized_frames)
                del resized_frames
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            # Motion video processing
            _, height, width, channel = np.shape(pixel_values)
            if self.enable_motion_info:
                motion_pixel_values = np.ones([self.motion_frames, height, width, channel]) * 127.5
                if start_frame > 0:
                    # Collect motion frames before start_frame (from start_frame-stride towards 0)
                    motion_frame_indices = []
                    current_idx = start_frame - local_video_sample_stride
                    while current_idx >= 0 and len(motion_frame_indices) < self.motion_frames:
                        motion_frame_indices.append(current_idx)
                        current_idx -= local_video_sample_stride
                    motion_frame_indices = motion_frame_indices[::-1]  # Reverse to ascending order

                    _motion_sample_args = (video_reader, motion_frame_indices)
                    motion_raw_frames = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=_motion_sample_args
                    )
                    # Resize each frame and free the original array early
                    motion_resized_frames = []
                    for i in range(len(motion_raw_frames)):
                        motion_resized_frames.append(resize_frame(motion_raw_frames[i], max(self.video_sample_size)))
                    del motion_raw_frames
                    motion_pixel_values[-len(motion_resized_frames):] = motion_resized_frames
                    del motion_resized_frames

                if not self.enable_bucket:
                    motion_pixel_values = torch.from_numpy(motion_pixel_values).permute(0, 3, 1, 2).contiguous()
                    motion_pixel_values = motion_pixel_values / 255.
                    motion_pixel_values = self.pixel_transforms(motion_pixel_values)
            else:
                motion_pixel_values = None

            # Video post-processing: convert to tensor, normalize to [-1, 1], apply transforms
            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
                pixel_values = self.pixel_transforms(pixel_values)

        # Load and extract the corresponding audio segment
        # Calculate start and end times (in seconds) of the video clip
        start_time = start_frame / fps
        end_time = (start_frame + (actual_n_frames - 1) * local_video_sample_stride) / fps
        duration = end_time - start_time

        # Load entire audio and resample to target sample rate
        audio_input, sample_rate = librosa.load(audio_path, sr=self.audio_sr)

        # Convert time to sample indices
        start_sample = round(start_time * self.audio_sr)
        target_len = round(duration * self.audio_sr)
        end_sample = start_sample + target_len

        # Extract audio segment with validation
        if start_sample >= len(audio_input):
            raise ValueError(f"Audio file too short: {audio_path}")
        else:
            audio_segment = audio_input[start_sample:end_sample]
            if len(audio_segment) < target_len:
                raise ValueError(f"Audio file too short: {audio_path}")

        # Random text dropout for classifier-free guidance
        if random.random() < self.text_drop_ratio:
            text = ''

        return pixel_values, motion_pixel_values, text, audio_segment, sample_rate, new_fps

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Get a sample with retry on failure."""
        data_info = self.dataset[idx % len(self.dataset)]
        while True:
            sample = {}
            try:
                pixel_values, motion_pixel_values, text, audio, sample_rate, fps = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["motion_pixel_values"] = motion_pixel_values
                sample["text"] = text
                sample["audio"] = torch.from_numpy(audio).float()
                sample["sample_rate"] = sample_rate
                sample["fps"] = fps
                sample["idx"] = idx
                
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(data_info['file_path'])

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length - 1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size(), image_start_only=True)
            # Fill masked regions with configurable value (default -1.0, some models use 0.0)
            mask_pixel_values = torch.where(mask.bool(), torch.tensor(self.inpaint_mask_fill_value), pixel_values)
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


class VideoSpeechControlDataset(Dataset):
    """Dataset for video-speech-control paired training with motion and inpainting support."""
    def __init__(
        self,
        ann_path, 
        data_root=None,
        video_sample_size=512, 
        video_sample_stride=4, 
        video_sample_n_frames=16,
        enable_bucket=False, 
        enable_inpaint=False,
        inpaint_mask_fill_value=0,
        audio_sr=16000,
        text_drop_ratio=0.1,
        enable_motion_info=False,
        motion_frames=73,
        return_file_name=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        self.dataset = json.load(open(ann_path, 'r'))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.data_root = data_root
        self.enable_bucket = enable_bucket
        self.enable_inpaint = enable_inpaint
        self.inpaint_mask_fill_value = inpaint_mask_fill_value
        self.audio_sr = audio_sr
        self.text_drop_ratio = text_drop_ratio
        self.enable_motion_info = enable_motion_info
        self.motion_frames = motion_frames
        self.return_file_name = return_file_name
        
        # Video params: resize, center crop, normalize to [-1, 1]
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.pixel_transforms = transforms.Compose(
            [
                transforms.Resize(self.video_sample_size[0]),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    
    def get_batch(self, idx):
        """Load and preprocess a single video sample with control and audio."""
        video_dict = self.dataset[idx]
        video_id, text = video_dict['file_path'], video_dict['text']
        audio_id = video_dict['audio_path']
        control_video_id = video_dict['control_file_path']

        # Resolve video, audio, and control paths
        if self.data_root is None:
            video_path = video_id
            audio_path = audio_id
            control_path = control_video_id
        else:
            video_path = os.path.join(self.data_root, video_id)
            audio_path = os.path.join(self.data_root, audio_id)
            control_path = os.path.join(self.data_root, control_video_id)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found for {video_path}")

        # Video information
        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            total_frames = len(video_reader)
            fps = video_reader.get_avg_fps()  # Get the original video frame rate
            if fps <= 0:
                raise ValueError(f"Video has negative fps: {video_path}")
            
            # Avoid fps > 30
            local_video_sample_stride = self.video_sample_stride
            new_fps = int(fps // local_video_sample_stride)
            while new_fps > 30:
                local_video_sample_stride = local_video_sample_stride + 1
                new_fps = int(fps // local_video_sample_stride)

            # Calculate the actual number of sampled video frames (considering boundaries)
            max_possible_frames = (total_frames - 1) // local_video_sample_stride + 1
            actual_n_frames = min(self.video_sample_n_frames, max_possible_frames)
            if actual_n_frames <= 0:
                raise ValueError(f"Video too short: {video_path}")

            # Randomly select the starting frame
            max_start = total_frames - (actual_n_frames - 1) * local_video_sample_stride - 1
            start_frame = random.randint(0, max_start) if max_start > 0 else 0
            frame_indices = [start_frame + i * local_video_sample_stride for i in range(actual_n_frames)]

            # Read video frames
            try:
                sample_args = (video_reader, frame_indices)
                raw_frames = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                )
                # Resize each frame and free the original array early to reduce peak memory
                resized_frames = []
                for i in range(len(raw_frames)):
                    resized_frames.append(resize_frame(raw_frames[i], max(self.video_sample_size)))
                del raw_frames
                pixel_values = np.array(resized_frames)
                del resized_frames
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            # Motion video processing
            _, height, width, channel = np.shape(pixel_values)
            if self.enable_motion_info:
                motion_pixel_values = np.ones([self.motion_frames, height, width, channel]) * 127.5
                if start_frame > 0:
                    # Collect motion frames before start_frame (from start_frame-stride towards 0)
                    motion_frame_indices = []
                    current_idx = start_frame - local_video_sample_stride
                    while current_idx >= 0 and len(motion_frame_indices) < self.motion_frames:
                        motion_frame_indices.append(current_idx)
                        current_idx -= local_video_sample_stride
                    motion_frame_indices = motion_frame_indices[::-1]  # Reverse to ascending order

                    _motion_sample_args = (video_reader, motion_frame_indices)
                    motion_raw_frames = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=_motion_sample_args
                    )
                    # Resize each frame and free the original array early
                    motion_resized_frames = []
                    for i in range(len(motion_raw_frames)):
                        motion_resized_frames.append(resize_frame(motion_raw_frames[i], max(self.video_sample_size)))
                    del motion_raw_frames
                    motion_pixel_values[-len(motion_resized_frames):] = motion_resized_frames
                    del motion_resized_frames

                if not self.enable_bucket:
                    motion_pixel_values = torch.from_numpy(motion_pixel_values).permute(0, 3, 1, 2).contiguous()
                    motion_pixel_values = motion_pixel_values / 255.
                    motion_pixel_values = self.pixel_transforms(motion_pixel_values)
            else:
                motion_pixel_values = None

            # Video post-processing: convert to tensor, normalize to [-1, 1], apply transforms
            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
                pixel_values = self.pixel_transforms(pixel_values)

        # Control information
        with VideoReader_contextmanager(control_path, num_threads=2) as control_video_reader:
            try:
                sample_args = (control_video_reader, frame_indices)
                control_raw_frames = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                )
                # Resize each frame and free the original array early
                resized_frames = []
                for i in range(len(control_raw_frames)):
                    resized_frames.append(resize_frame(control_raw_frames[i], max(self.video_sample_size)))
                del control_raw_frames
                control_pixel_values = np.stack(resized_frames)
                del resized_frames
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            if not self.enable_bucket:
                control_pixel_values = torch.from_numpy(control_pixel_values).permute(0, 3, 1, 2).contiguous()
                control_pixel_values = control_pixel_values / 255.
                control_pixel_values = self.pixel_transforms(control_pixel_values)

        # Load and extract the corresponding audio segment
        # Calculate start and end times (in seconds) of the video clip
        start_time = start_frame / fps
        end_time = (start_frame + (actual_n_frames - 1) * local_video_sample_stride) / fps
        duration = end_time - start_time

        # Load entire audio and resample to target sample rate
        audio_input, sample_rate = librosa.load(audio_path, sr=self.audio_sr)

        # Convert time to sample indices
        start_sample = round(start_time * self.audio_sr)
        target_len = round(duration * self.audio_sr)
        end_sample = start_sample + target_len

        # Extract audio segment with validation
        if start_sample >= len(audio_input):
            raise ValueError(f"Audio file too short: {audio_path}")
        else:
            audio_segment = audio_input[start_sample:end_sample]
            if len(audio_segment) < target_len:
                raise ValueError(f"Audio file too short: {audio_path}")

        # Random text dropout for classifier-free guidance
        if random.random() < self.text_drop_ratio:
            text = ''

        return pixel_values, motion_pixel_values, control_pixel_values, text, audio_segment, sample_rate, new_fps

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Get a sample with retry on failure."""
        data_info = self.dataset[idx % len(self.dataset)]
        while True:
            sample = {}
            try:
                pixel_values, motion_pixel_values, control_pixel_values, text, audio, sample_rate, fps = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["motion_pixel_values"] = motion_pixel_values
                sample["control_pixel_values"] = control_pixel_values
                sample["text"] = text
                sample["audio"] = torch.from_numpy(audio).float()
                sample["sample_rate"] = sample_rate
                sample["fps"] = fps
                sample["idx"] = idx
                
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(data_info['file_path'])

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size(), image_start_only=True)
            # Fill masked regions with configurable value (default -1.0, some models use 0.0)
            mask_pixel_values = torch.where(mask.bool(), torch.tensor(self.inpaint_mask_fill_value), pixel_values)
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


class VideoAnimateDataset(Dataset):
    """Dataset for video animation training with control, face, background, and mask support."""
    def __init__(
        self,
        ann_path, 
        data_root=None,
        video_sample_size=512, 
        video_sample_stride=4, 
        video_sample_n_frames=16,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.1, 
        video_length_drop_end=0.9,
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
        
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params: resize, center crop, normalize to [-1, 1]
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        self.larger_side_of_image_and_video = min(self.video_sample_size)
    
    def get_batch(self, idx):
        """Load and preprocess a single video sample with control, face, background, and mask."""
        data_info = self.dataset[idx % len(self.dataset)]
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
            start_idx = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

            try:
                sample_args = (video_reader, batch_index)
                raw_frames = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                )
                # Resize each frame and free the original array early
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

            # Release video reader early
            del video_reader

            # Convert to tensor and apply transforms
            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
                pixel_values = self.video_transforms(pixel_values)
            
            # Random text dropout for classifier-free guidance
            if random.random() < self.text_drop_ratio:
                text = ''

        # Load control video
        control_video_id = data_info['control_file_path']
        if control_video_id is not None:
            control_video_path = control_video_id if self.data_root is None else os.path.join(self.data_root, control_video_id)
        else:
            control_video_path = None
        
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

        # Load face video
        face_video_id = data_info['face_file_path']
        if face_video_id is not None:
            face_video_path = face_video_id if self.data_root is None else os.path.join(self.data_root, face_video_id)
        else:
            face_video_path = None
        
        if face_video_path is not None:
            with VideoReader_contextmanager(face_video_path, num_threads=2) as face_video_reader:
                try:
                    sample_args = (face_video_reader, batch_index)
                    face_raw_frames = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    # Resize each frame and free the original array early
                    resized_frames = []
                    for i in range(len(face_raw_frames)):
                        resized_frames.append(resize_frame(face_raw_frames[i], self.larger_side_of_image_and_video))
                    del face_raw_frames
                    face_pixel_values = np.stack(resized_frames)
                    del resized_frames
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                # Release face video reader early
                del face_video_reader

                # Convert to tensor and apply transforms
                if not self.enable_bucket:
                    face_pixel_values = torch.from_numpy(face_pixel_values).permute(0, 3, 1, 2).contiguous()
                    face_pixel_values = face_pixel_values / 255.
                    face_pixel_values = self.video_transforms(face_pixel_values)
        else:
            face_pixel_values = torch.zeros_like(pixel_values) if not self.enable_bucket else np.zeros_like(pixel_values)

        # Load background video
        background_video_id = data_info.get('background_file_path', None)
        if background_video_id is not None:
            background_video_path = background_video_id if self.data_root is None else os.path.join(self.data_root, background_video_id)
        else:
            background_video_path = None
        
        if background_video_path is not None:
            with VideoReader_contextmanager(background_video_path, num_threads=2) as background_video_reader:
                try:
                    sample_args = (background_video_reader, batch_index)
                    background_raw_frames = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    # Resize each frame and free the original array early
                    resized_frames = []
                    for i in range(len(background_raw_frames)):
                        resized_frames.append(resize_frame(background_raw_frames[i], self.larger_side_of_image_and_video))
                    del background_raw_frames
                    background_pixel_values = np.stack(resized_frames)
                    del resized_frames
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                # Release background video reader early
                del background_video_reader

                # Convert to tensor and apply transforms
                if not self.enable_bucket:
                    background_pixel_values = torch.from_numpy(background_pixel_values).permute(0, 3, 1, 2).contiguous()
                    background_pixel_values = background_pixel_values / 255.
                    background_pixel_values = self.video_transforms(background_pixel_values)
        else:
            background_pixel_values = torch.ones_like(pixel_values) * 127.5 if not self.enable_bucket else np.ones_like(pixel_values) * 127.5

        # Load mask video
        mask_video_id = data_info.get('mask_file_path', None)
        if mask_video_id is not None:
            mask_video_path = mask_video_id if self.data_root is None else os.path.join(self.data_root, mask_video_id)
        else:
            mask_video_path = None
        
        if mask_video_path is not None:
            with VideoReader_contextmanager(mask_video_path, num_threads=2) as mask_video_reader:
                try:
                    sample_args = (mask_video_reader, batch_index)
                    mask_raw_frames = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    # Resize each frame and free the original array early
                    resized_frames = []
                    for i in range(len(mask_raw_frames)):
                        resized_frames.append(resize_frame(mask_raw_frames[i], self.larger_side_of_image_and_video))
                    del mask_raw_frames
                    mask = np.stack(resized_frames)
                    del resized_frames
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                # Release mask video reader early
                del mask_video_reader

                # Convert to tensor (no transforms for mask)
                if not self.enable_bucket:
                    mask = torch.from_numpy(mask).permute(0, 3, 1, 2).contiguous()
                    mask = mask / 255.
        else:
            mask = torch.ones_like(pixel_values) if not self.enable_bucket else np.ones_like(pixel_values) * 255
        
        # Extract only the first channel
        mask = mask[:, :, :, :1]
        
        # Load reference image
        ref_pixel_values_path = data_info.get('ref_file_path', [])
        if self.data_root is not None:
            ref_pixel_values_path = os.path.join(self.data_root, ref_pixel_values_path)
        ref_pixel_values = Image.open(ref_pixel_values_path).convert('RGB')

        if not self.enable_bucket:
            raise ValueError("Not enable_bucket is not supported now. ")
        else:
            ref_pixel_values = np.array(ref_pixel_values)
    
        return pixel_values, control_pixel_values, face_pixel_values, background_pixel_values, mask, ref_pixel_values, text, "video"

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

                pixel_values, control_pixel_values, face_pixel_values, background_pixel_values, mask, ref_pixel_values, name, data_type = \
                    self.get_batch(idx)

                sample["pixel_values"] = pixel_values
                sample["control_pixel_values"] = control_pixel_values
                sample["face_pixel_values"] = face_pixel_values
                sample["background_pixel_values"] = background_pixel_values
                sample["mask"] = mask
                sample["ref_pixel_values"] = ref_pixel_values
                sample["clip_pixel_values"] = ref_pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx

                if self.return_file_name:
                    sample["file_name"] = os.path.basename(data_info['file_path'])

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        return sample


if __name__ == "__main__":
    if 1:
        dataset = VideoDataset(
            json_path="./webvidval/results_2M_val.json",
            sample_size=256,
            sample_stride=4, sample_n_frames=16,
        )

    if 0:
        dataset = WebVid10M(
            csv_path="./webvid/results_2M_val.csv",
            video_folder="./webvid/2M_val",
            sample_size=256,
            sample_stride=4, sample_n_frames=16,
            is_image=False,
        )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))