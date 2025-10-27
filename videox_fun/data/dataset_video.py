import csv
import gc
import io
import json
import math
import os
import random
from contextlib import contextmanager
from threading import Thread

import albumentations
import cv2
import librosa
import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from PIL import Image
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset

from .utils import (VIDEO_READER_TIMEOUT, Camera, VideoReader_contextmanager,
                    custom_meshgrid, get_random_mask, get_relative_pose,
                    get_video_reader_batch, padding_image, process_pose_file,
                    process_pose_params, ray_condition, resize_frame,
                    resize_image_with_target_area)


class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            enable_bucket=False, enable_inpaint=False, is_image=False,
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
    def __init__(
        self,
        ann_path, data_root=None,
        sample_size=256, sample_stride=4, sample_n_frames=16,
        enable_bucket=False, enable_inpaint=False
    ):
        print(f"loading annotations from {ann_path} ...")
        self.dataset = json.load(open(ann_path, 'r'))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.data_root       = data_root
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.enable_bucket   = enable_bucket
        self.enable_inpaint  = enable_inpaint
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose(
            [
                transforms.Resize(sample_size[0]),
                transforms.CenterCrop(sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        video_id, text = video_dict['file_path'], video_dict['text']

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

            video_length = int(self.video_length_drop_end * len(video_reader))
            clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
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

            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
                del video_reader
            else:
                pixel_values = pixel_values

            if not self.enable_bucket:
                pixel_values = self.video_transforms(pixel_values)
            
            # Random use no text generation
            if random.random() < self.text_drop_ratio:
                text = ''
            return pixel_values, text

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
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
            mask_pixel_values = pixel_values * (1 - mask) + torch.zeros_like(pixel_values) * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


class VideoSpeechDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        enable_bucket=False, enable_inpaint=False,
        audio_sr=16000,  # 新增：目标音频采样率
        text_drop_ratio=0.1  # 新增：文本丢弃概率
    ):
        print(f"loading annotations from {ann_path} ...")
        self.dataset = json.load(open(ann_path, 'r'))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.data_root = data_root
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.enable_bucket = enable_bucket
        self.enable_inpaint = enable_inpaint
        self.audio_sr = audio_sr
        self.text_drop_ratio = text_drop_ratio
        
        video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.pixel_transforms = transforms.Compose(
            [
                transforms.Resize(video_sample_size[0]),
                transforms.CenterCrop(video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        video_id, text = video_dict['file_path'], video_dict['text']
        audio_id = video_dict['audio_path']

        if self.data_root is None:
            video_path = video_id
        else:
            video_path = os.path.join(self.data_root, video_id)

        if self.data_root is None:
            audio_path = audio_id
        else:
            audio_path = os.path.join(self.data_root, audio_id)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found for {video_path}")

        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            total_frames = len(video_reader)
            fps = video_reader.get_avg_fps()  # 获取原始视频帧率

            # 计算实际采样的视频帧数（考虑边界）
            max_possible_frames = (total_frames - 1) // self.video_sample_stride + 1
            actual_n_frames = min(self.video_sample_n_frames, max_possible_frames)
            if actual_n_frames <= 0:
                raise ValueError(f"Video too short: {video_path}")

            # 随机选择起始帧
            max_start = total_frames - (actual_n_frames - 1) * self.video_sample_stride - 1
            start_frame = random.randint(0, max_start) if max_start > 0 else 0
            frame_indices = [start_frame + i * self.video_sample_stride for i in range(actual_n_frames)]

            # 读取视频帧
            try:
                sample_args = (video_reader, frame_indices)
                pixel_values = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                )
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            # 视频后处理
            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
                pixel_values = self.pixel_transforms(pixel_values)

            # === 新增：加载并截取对应音频 ===
            # 视频片段的起止时间（秒）
            start_time = start_frame / fps
            end_time = (start_frame + (actual_n_frames - 1) * self.video_sample_stride) / fps
            duration = end_time - start_time

            # 使用 librosa 加载整个音频（或仅加载所需部分，但 librosa.load 不支持精确 seek，所以先加载再切）
            audio_input, sample_rate = librosa.load(audio_path, sr=self.audio_sr)  # 重采样到目标 sr

            # 转换为样本索引
            start_sample = int(start_time * self.audio_sr)
            end_sample = int(end_time * self.audio_sr)

            # 安全截取
            if start_sample >= len(audio_input):
                # 音频太短，用零填充或截断
                audio_segment = np.zeros(int(duration * self.audio_sr), dtype=np.float32)
            else:
                audio_segment = audio_input[start_sample:end_sample]
                # 如果太短，补零
                target_len = int(duration * self.audio_sr)
                if len(audio_segment) < target_len:
                    audio_segment = np.pad(audio_segment, (0, target_len - len(audio_segment)), mode='constant')

            # === 文本随机丢弃 ===
            if random.random() < self.text_drop_ratio:
                text = ''

            return pixel_values, text, audio_segment, sample_rate

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            sample = {}
            try:
                pixel_values, text, audio, sample_rate = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = text
                sample["audio"] = torch.from_numpy(audio).float()  # 转为 tensor
                sample["sample_rate"] = sample_rate
                sample["idx"] = idx
                break
            except Exception as e:
                print(f"Error processing {idx}: {e}, retrying with random idx...")
                idx = random.randint(0, self.length - 1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size(), image_start_only=True)
            mask_pixel_values = pixel_values * (1 - mask) + torch.zeros_like(pixel_values) * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


class VideoSpeechControlDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        enable_bucket=False, enable_inpaint=False,
        audio_sr=16000,
        text_drop_ratio=0.1,
        enable_motion_info=False,
        motion_frames=73,
    ):
        print(f"loading annotations from {ann_path} ...")
        self.dataset = json.load(open(ann_path, 'r'))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.data_root = data_root
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.enable_bucket = enable_bucket
        self.enable_inpaint = enable_inpaint
        self.audio_sr = audio_sr
        self.text_drop_ratio = text_drop_ratio
        self.enable_motion_info = enable_motion_info
        self.motion_frames = motion_frames
        
        video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.pixel_transforms = transforms.Compose(
            [
                transforms.Resize(video_sample_size[0]),
                transforms.CenterCrop(video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        self.video_sample_size = video_sample_size
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        video_id, text = video_dict['file_path'], video_dict['text']
        audio_id = video_dict['audio_path']
        control_video_id = video_dict['control_file_path']

        if self.data_root is None:
            video_path = video_id
        else:
            video_path = os.path.join(self.data_root, video_id)

        if self.data_root is None:
            audio_path = audio_id
        else:
            audio_path = os.path.join(self.data_root, audio_id)
        
        if self.data_root is None:
            control_video_id = control_video_id
        else:
            control_video_id = os.path.join(self.data_root, control_video_id)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found for {video_path}")

        # Video information
        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            total_frames = len(video_reader)
            fps = video_reader.get_avg_fps()
            if fps <= 0:
                raise ValueError(f"Video has negative fps: {video_path}")
            local_video_sample_stride = self.video_sample_stride
            new_fps = int(fps // local_video_sample_stride)
            while new_fps > 30:
                local_video_sample_stride = local_video_sample_stride + 1
                new_fps = int(fps // local_video_sample_stride)

            max_possible_frames = (total_frames - 1) // local_video_sample_stride + 1
            actual_n_frames = min(self.video_sample_n_frames, max_possible_frames)
            if actual_n_frames <= 0:
                raise ValueError(f"Video too short: {video_path}")

            max_start = total_frames - (actual_n_frames - 1) * local_video_sample_stride - 1
            start_frame = random.randint(0, max_start) if max_start > 0 else 0
            frame_indices = [start_frame + i * local_video_sample_stride for i in range(actual_n_frames)]

            try:
                sample_args = (video_reader, frame_indices)
                pixel_values = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                )
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            _, height, width, channel = np.shape(pixel_values)
            if self.enable_motion_info:
                motion_pixel_values = np.ones([self.motion_frames, height, width, channel]) * 127.5
                if start_frame > 0:
                    motion_max_possible_frames = (start_frame - 1) // local_video_sample_stride + 1
                    motion_frame_indices = [0 + i * local_video_sample_stride for i in range(motion_max_possible_frames)]
                    motion_frame_indices = motion_frame_indices[-self.motion_frames:]

                    _motion_sample_args = (video_reader, motion_frame_indices)
                    _motion_pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=_motion_sample_args
                    )
                    motion_pixel_values[-len(motion_frame_indices):] = _motion_pixel_values

                if not self.enable_bucket:
                    motion_pixel_values = torch.from_numpy(motion_pixel_values).permute(0, 3, 1, 2).contiguous()
                    motion_pixel_values = motion_pixel_values / 255.
                    motion_pixel_values = self.pixel_transforms(motion_pixel_values)
            else:
                motion_pixel_values = None

            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
                pixel_values = self.pixel_transforms(pixel_values)

        # Audio information
        start_time = start_frame / fps
        end_time = (start_frame + (actual_n_frames - 1) * local_video_sample_stride) / fps
        duration = end_time - start_time

        audio_input, sample_rate = librosa.load(audio_path, sr=self.audio_sr)
        start_sample = int(start_time * self.audio_sr)
        end_sample = int(end_time * self.audio_sr)

        if start_sample >= len(audio_input):
            raise ValueError(f"Audio file too short: {audio_path}")
        else:
            audio_segment = audio_input[start_sample:end_sample]
            target_len = int(duration * self.audio_sr)
            if len(audio_segment) < target_len:
                raise ValueError(f"Audio file too short: {audio_path}")

        # Control information
        with VideoReader_contextmanager(control_video_id, num_threads=2) as control_video_reader:
            try:
                sample_args = (control_video_reader, frame_indices)
                control_pixel_values = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                )
                resized_frames = []
                for i in range(len(control_pixel_values)):
                    frame = control_pixel_values[i]
                    resized_frame = resize_frame(frame, max(self.video_sample_size))
                    resized_frames.append(resized_frame)
                control_pixel_values = np.array(control_pixel_values)
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            if not self.enable_bucket:
                control_pixel_values = torch.from_numpy(control_pixel_values).permute(0, 3, 1, 2).contiguous()
                control_pixel_values = control_pixel_values / 255.
                del control_video_reader
            else:
                control_pixel_values = control_pixel_values

            if not self.enable_bucket:
                control_pixel_values = self.video_transforms(control_pixel_values)

        if random.random() < self.text_drop_ratio:
            text = ''

        return pixel_values, motion_pixel_values, control_pixel_values, text, audio_segment, sample_rate, new_fps

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            sample = {}
            try:
                pixel_values, motion_pixel_values, control_pixel_values, text, audio, sample_rate, new_fps = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["motion_pixel_values"] = motion_pixel_values
                sample["control_pixel_values"] = control_pixel_values
                sample["text"] = text
                sample["audio"] = torch.from_numpy(audio).float()  # 转为 tensor
                sample["sample_rate"] = sample_rate
                sample["fps"] = new_fps
                sample["idx"] = idx
                break
            except Exception as e:
                print(f"Error processing {idx}: {e}, retrying with random idx...")
                idx = random.randint(0, self.length - 1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size(), image_start_only=True)
            mask_pixel_values = pixel_values * (1 - mask) + torch.zeros_like(pixel_values) * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


if __name__ == "__main__":
    if 1:
        dataset = VideoDataset(
            json_path="/home/zhoumo.xjq/disk3/datasets/webvidval/results_2M_val.json",
            sample_size=256,
            sample_stride=4, sample_n_frames=16,
        )

    if 0:
        dataset = WebVid10M(
            csv_path="/mnt/petrelfs/guoyuwei/projects/datasets/webvid/results_2M_val.csv",
            video_folder="/mnt/petrelfs/guoyuwei/projects/datasets/webvid/2M_val",
            sample_size=256,
            sample_stride=4, sample_n_frames=16,
            is_image=False,
        )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))