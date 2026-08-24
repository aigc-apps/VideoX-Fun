import csv
import json
import math
import os
import random
import warnings

import cv2
import librosa
import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from PIL import Image
from torch.utils.data.dataset import Dataset

# torchaudio decodes video containers `load_audio` falls back to and resamples the waveform on the
# MiniMax-H3 inference-aligned route. A build mismatched with the installed torch (a torchaudio < 2.6 against
# torch >= 2.6) fails to import with an undefined-symbol error; keep training possible in that case by holding
# `None` here and degrading at the use sites instead of crashing at module import.
try:
    import torchaudio
except Exception as _error:
    torchaudio = None
    # Copy inside the block: Python deletes an `except ... as` target when the block exits.
    _torchaudio_import_error = _error
else:
    _torchaudio_import_error = None

try:
    from decord import VideoReader
except ImportError:
    from .utils import AVVideoReader as VideoReader

from .utils import (VIDEO_READER_TIMEOUT, VideoReader_contextmanager,
                    get_random_mask, get_video_reader_batch, resize_frame)


def load_audio(path, sr, mono=True, native_sr=False, res_type=None):
    """Load a float32 waveform from an audio *or* video file.

    librosa covers plain audio files, but its support for video containers (mp4/mov/...) rides on the audioread
    fallback which is deprecated and removed in librosa 1.0, so any failure falls back to torchaudio's ffmpeg
    backend, which decodes the audio stream of every container ffmpeg understands.

    By default the waveform is mixed down to mono and resampled onto `sr` at load time, which is the behaviour
    callers before the MiniMax-H3 alignment rely on. `native_sr=True` instead hands the samples over at the rate
    the file carries them, unresampled, so the caller slices at that rate and resamples once afterwards — the
    order MiniMax-H3's inference normalizes a reference soundtrack in (`normalize_reference_audio`) — and
    `mono=False` keeps the channels the file holds instead of mixing them down. `res_type` picks the librosa
    resampler used when resampling at load time (`None` keeps librosa's default).
    """
    try:
        # Video containers miss PySoundFile and ride the deprecated audioread fallback, whose per-file "PySoundFile
        # failed" notice would otherwise flood the dataloader workers; the torchaudio fallback below still covers
        # real decode failures.
        load_kwargs = {} if res_type is None else {"res_type": res_type}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="librosa")
            warnings.filterwarnings("ignore", message="PySoundFile failed")
            waveform, sample_rate = librosa.load(path, sr=None if native_sr else sr, mono=mono, **load_kwargs)
        return waveform, sample_rate
    except Exception:
        if torchaudio is None:
            raise ImportError(
                f"librosa could not decode {path} and the torchaudio ffmpeg fallback is unavailable "
                f"({_torchaudio_import_error}); reinstall the torchaudio matching the installed torch."
            )
        waveform, source_sr = torchaudio.load(path)
        if not native_sr and source_sr != sr:
            waveform = torchaudio.functional.resample(waveform, source_sr, sr)
        if mono:
            # Channels -> mono, matching librosa.load's default mono mixdown.
            waveform = waveform.mean(0)
        return waveform.numpy().astype(np.float32), source_sr if native_sr else sr


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
        min_video_sample_n_frames=1,
        target_video_sample_fps=None,
        video_sample_fps_tolerance=0.5,
        audio_native_sr_resample=False,
        audio_stereo=False,
        audio_span_includes_last_frame=False,
        enable_ref2va=False,
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
        # Fewest sampled frames a clip has to yield to be usable. A model whose VAE only encodes certain frame
        # counts (MiniMax-H3 needs `17n + 5`, so at least 5) raises this so that clips which cannot fill one chunk
        # are skipped by `__getitem__`'s retry instead of reaching the collate as a short batch. The default of 1 is
        # the previous behaviour: only a clip yielding no frame at all is rejected.
        self.min_video_sample_n_frames = max(1, int(min_video_sample_n_frames))
        # Frame rate the sampled clip has to land on, within `video_sample_fps_tolerance`. A model that reads its
        # frames on a fixed timeline (MiniMax-H3 has no fps input: both its temporal rotary grid and its 40 latents/s
        # audio grid assume 24 fps) sets this so that clips at another rate are skipped by `__getitem__`'s retry
        # instead of training the model on video that plays at the wrong speed against its own soundtrack. The check
        # uses the *unrounded* rate, which matters: the common 23.976 fps is within a tolerance of 24 while
        # `new_fps` floors it to 23 and loses that. `None` disables the check, which is the previous behaviour.
        self.target_video_sample_fps = target_video_sample_fps
        self.video_sample_fps_tolerance = video_sample_fps_tolerance
        # Whether the audio track is loaded on the MiniMax-H3 inference route: sliced at the file's native rate
        # and resampled once afterwards with the pipeline's torchaudio pass (`normalize_reference_audio`), rather
        # than resampled onto `audio_sr` at load time and sliced by index there. `audio_stereo` keeps the two
        # channels a stereo file carries (a mono file is upmixed by repeating its channel) instead of mixing them
        # down, and `audio_span_includes_last_frame` reads the clip's span as `num_frames / fps` seconds — the
        # last frame holding its own duration — which is what the audio latent grid keys off. All three default
        # off: the legacy behaviour is unchanged unless a training script asks for the alignment.
        self.audio_native_sr_resample = audio_native_sr_resample
        self.audio_stereo = audio_stereo
        self.audio_span_includes_last_frame = audio_span_includes_last_frame
        # When True, the dataset reads a `references` field from each annotation and decodes the
        # image/video/audio references into `MiniMaxH3Reference` objects for ref2va training.
        self.enable_ref2va = enable_ref2va
        # Resampler the degraded librosa route uses once torchaudio turns out unavailable; decided in the check
        # below, `None` keeps librosa's default.
        self.audio_fallback_res_type = None
        if self.audio_native_sr_resample and torchaudio is None:
            # The aligned route resamples with torchaudio; an ABI mismatch (a torchaudio built against another
            # torch) falls back to the legacy librosa route instead of blocking training. That route then keeps
            # the channels the file carries (a mono file is upmixed by repeating its channel, as in
            # `normalize_reference_audio`) and — where available — resamples with librosa's `kaiser_best`, the
            # closest analogue of the pipeline's torchaudio pass, so the degraded audio stays as near to the
            # inference signal as librosa can get.
            print(
                f"WARNING: audio_native_sr_resample needs a working torchaudio, but importing it failed "
                f"({_torchaudio_import_error}); falling back to the legacy librosa audio route. Reinstall the "
                "torchaudio matching the installed torch (e.g. torch 2.7.0 wants torchaudio 2.7.0) to restore the "
                "inference-aligned audio."
            )
            self.audio_native_sr_resample = False
            try:
                # `kaiser_best` rides on resampy; probe it once here instead of letting every sample fail on it.
                librosa.resample(np.zeros(2, dtype=np.float32), orig_sr=2, target_sr=1, res_type="kaiser_best")
                self.audio_fallback_res_type = "kaiser_best"
            except Exception as e:
                print(
                    f"WARNING: librosa's kaiser_best resampler is unavailable ({e}); the fallback audio keeps "
                    "librosa's default resampler."
                )
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

            # Compared on the unrounded rate: 23.976 fps passes a tolerance around 24 while `new_fps` floors it to 23.
            # A clip outside the tolerance is skipped by `__getitem__`'s retry — there is no resampling fallback, as
            # playing it on the fixed 24 fps timeline would slow the picture down and drag the soundtrack's pitch
            # down with it. `frame_step` is how many source frames one sampled frame advances.
            frame_step = float(local_video_sample_stride)
            if self.target_video_sample_fps is not None:
                effective_fps = fps / local_video_sample_stride
                if abs(effective_fps - self.target_video_sample_fps) > self.video_sample_fps_tolerance:
                    raise ValueError(
                        f"Frame rate mismatch: {video_path} samples at {effective_fps:.3f} fps (source {fps:.3f} fps "
                        f"at stride {local_video_sample_stride}), outside "
                        f"{self.target_video_sample_fps} +/- {self.video_sample_fps_tolerance} fps that this training "
                        "run reads its frames on; skipping."
                    )

            # Calculate the actual number of sampled frames (considering boundaries)
            max_possible_frames = int((total_frames - 1) / frame_step) + 1
            actual_n_frames = min(self.video_sample_n_frames, max_possible_frames)
            if actual_n_frames < self.min_video_sample_n_frames:
                raise ValueError(
                    f"Video too short: {video_path} yields {actual_n_frames} sampled frame(s) at stride "
                    f"{local_video_sample_stride}, fewer than the {self.min_video_sample_n_frames} this training "
                    "run needs; skipping."
                )

            # Randomly select the starting frame
            frame_span = (actual_n_frames - 1) * frame_step
            max_start = total_frames - 1 - int(math.ceil(frame_span))
            start_frame = random.randint(0, max_start) if max_start > 0 else 0
            frame_indices = [
                min(total_frames - 1, int(round(start_frame + index * frame_step)))
                for index in range(actual_n_frames)
            ]

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
                    if len(motion_resized_frames) > 0:
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
        # The sampled frames span `(actual_n_frames - 1) * frame_step` source frames, i.e.
        # `(actual_n_frames - 1) * frame_step / fps` seconds on the source timeline. With
        # `audio_span_includes_last_frame` the span carries one more
        # frame period: MiniMax-H3's inference reads a soundtrack over `num_frames / fps` seconds — the last frame
        # holds its own duration — and its audio latent grid (`round(L / fps * 40)`) keys off that convention.
        frame_periods = actual_n_frames if self.audio_span_includes_last_frame else actual_n_frames - 1
        duration = frame_periods * frame_step / fps
        end_time = start_time + duration

        if not self.audio_native_sr_resample:
            # Load entire audio and resample to target sample rate. `audio_stereo` only reaches this branch as the
            # degraded substitute for the torchaudio route, where it keeps the channels the file carries and uses
            # the closest librosa resampler to the pipeline's torchaudio pass; without it this is the
            # pre-alignment behaviour: mono mixdown with librosa's default resampler.
            audio_input, sample_rate = load_audio(
                audio_path, self.audio_sr, mono=not self.audio_stereo,
                res_type=self.audio_fallback_res_type if self.audio_stereo else None,
            )

            # Convert time to sample indices
            start_sample = round(start_time * self.audio_sr)
            target_len = round(duration * self.audio_sr)
            end_sample = start_sample + target_len

            if self.audio_stereo:
                # A `(channels, num_samples)` block sliced on the sample axis; a mono load arrives 1-D.
                audio_input = np.asarray(audio_input)
                if audio_input.ndim == 1:
                    audio_input = audio_input[None]
                elif audio_input.shape[0] > audio_input.shape[1]:
                    audio_input = audio_input.T
                if start_sample >= audio_input.shape[-1]:
                    raise ValueError(f"Audio file too short: {audio_path}")
                audio_segment = audio_input[..., start_sample:end_sample]
                if audio_segment.shape[-1] < target_len:
                    raise ValueError(f"Audio file too short: {audio_path}")
                waveform = torch.from_numpy(np.ascontiguousarray(audio_segment)).float()
                if waveform.shape[0] == 1:
                    # A mono soundtrack is upmixed by repeating its channel, as in `normalize_reference_audio`.
                    waveform = waveform.expand(2, -1).contiguous()
                elif waveform.shape[0] != 2:
                    raise ValueError(
                        f"MiniMax-H3 carries at most two audio channels, got {waveform.shape[0]} in {audio_path}."
                    )
                audio_segment = waveform
                audio_span_samples, audio_span_rate = audio_segment.shape[-1], self.audio_sr
            else:
                # Extract audio segment with validation
                if start_sample >= len(audio_input):
                    raise ValueError(f"Audio file too short: {audio_path}")
                else:
                    audio_segment = audio_input[start_sample:end_sample]
                    if len(audio_segment) < target_len:
                        raise ValueError(f"Audio file too short: {audio_path}")
                audio_span_samples, audio_span_rate = len(audio_segment), self.audio_sr
        else:
            # The inference-aligned route, mirroring `normalize_reference_audio` in the MiniMax-H3 pipeline: the
            # file is read at the rate it carries its samples at, the slice is taken there, and the segment is
            # resampled once afterwards with the same torchaudio pass (kaiser_best) the pipeline uses.
            audio_input, source_sr = load_audio(audio_path, self.audio_sr, mono=not self.audio_stereo, native_sr=True)
            # A `(channels, num_samples)` layout whatever the decoder handed over; a mono decode arrives 1-D.
            audio_input = np.asarray(audio_input)
            if audio_input.ndim == 1:
                audio_input = audio_input[None]
            elif audio_input.shape[0] > audio_input.shape[1]:
                audio_input = audio_input.T

            start_sample = round(start_time * source_sr)
            target_len = round(duration * source_sr)
            end_sample = start_sample + target_len
            if start_sample >= audio_input.shape[-1]:
                raise ValueError(f"Audio file too short: {audio_path}")
            audio_segment = audio_input[..., start_sample:min(end_sample, audio_input.shape[-1])]
            audio_span_samples, audio_span_rate = audio_segment.shape[-1], source_sr
            if audio_segment.shape[-1] < target_len:
                # A soundtrack that ends with the clip's last frame lacks up to one frame period of tail; pad it
                # here rather than retrying forever. Anything short beyond that is a genuinely shorter file.
                shortfall = target_len - audio_segment.shape[-1]
                if shortfall > round(frame_step / fps * source_sr):
                    raise ValueError(f"Audio file too short: {audio_path}")
                audio_segment = np.pad(audio_segment, [(0, 0)] * (audio_segment.ndim - 1) + [(0, shortfall)])

            waveform = torch.from_numpy(np.ascontiguousarray(audio_segment)).float()
            if source_sr != self.audio_sr:
                waveform = torchaudio.transforms.Resample(source_sr, self.audio_sr)(waveform)
            if self.audio_stereo:
                if waveform.shape[0] == 1:
                    # A mono soundtrack is upmixed by repeating its channel, as in `normalize_reference_audio`.
                    waveform = waveform.expand(2, -1).contiguous()
                elif waveform.shape[0] != 2:
                    raise ValueError(
                        f"MiniMax-H3 carries at most two audio channels, got {waveform.shape[0]} in {audio_path}."
                    )
            else:
                waveform = waveform[0]
            audio_segment, sample_rate = waveform, self.audio_sr

        # The sliced waveform must cover the same real-time span that the sampled frames play on the target
        # timeline: `frame_periods / target_fps` seconds. A container whose metadata fps disagrees with its real
        # frame rate slices a proportionally longer / shorter waveform — undetectable from the fps field alone,
        # which the flooring above corrupts further — and surfaces much later as an audio-latent window failure
        # in the training loop. Raise here so the retry of `__getitem__` draws another sample. The tolerance is
        # one frame period plus rounding slack for the waveform-to-latent encoder.
        if self.target_video_sample_fps is not None:
            target_span = frame_periods / self.target_video_sample_fps
            audio_span = audio_span_samples / audio_span_rate
            if abs(audio_span - target_span) > 1.0 / self.target_video_sample_fps + 0.03:
                raise ValueError(
                    f"Audio span mismatch: {video_path} plays {target_span:.3f}s on the "
                    f"{self.target_video_sample_fps} fps timeline but its waveform covers {audio_span:.3f}s, so the "
                    "clip's real frame rate disagrees with its metadata fps; skipping."
                )

        # Random text dropout for classifier-free guidance
        if random.random() < self.text_drop_ratio:
            text = ''

        return pixel_values, motion_pixel_values, text, audio_segment, sample_rate, new_fps

    def _load_references(self, data_info):
        r"""Decode the `references` field of an annotation into `MiniMaxH3Reference` objects.

        Expected format in the annotation JSON:

        ```json
        {
            "references": [
                {"type": "image", "path": "path/to/image.jpg"},
                {"type": "video", "path": "path/to/video.mp4"},
                {"type": "audio", "path": "path/to/audio.wav"}
            ]
        }
        ```

        Video references retain their container frame rate and soundtrack; audio and image references carry
        their own rates. The training loop resamples everything onto MiniMax-H3's fixed 24 fps / audio-VAE
        sample rate with the same utilities the inference pipeline uses.
        """
        if not self.enable_ref2va:
            return None
        ref_infos = data_info.get("references")
        if not ref_infos:
            return None

        # Delayed import to avoid a circular dependency between the data and pipeline modules.
        from videox_fun.pipeline.pipeline_minimax_h3 import (
            MiniMaxH3AudioReference,
            MiniMaxH3ImageReference,
            MiniMaxH3VideoReference,
        )

        references = []
        for entry in ref_infos:
            ref_type = entry.get("type")
            ref_path = entry.get("path")
            if ref_path is None:
                raise ValueError(f"A reference entry must have a `path`, got {entry}.")
            if self.data_root is not None and not os.path.isabs(ref_path):
                ref_path = os.path.join(self.data_root, ref_path)
            if ref_type == "image":
                references.append(MiniMaxH3ImageReference.from_file(ref_path))
            elif ref_type == "video":
                references.append(MiniMaxH3VideoReference.from_file(ref_path))
            elif ref_type == "audio":
                references.append(MiniMaxH3AudioReference.from_file(ref_path))
            else:
                raise ValueError(
                    f"Unsupported reference type {ref_type!r}; expected 'image', 'video' or 'audio'."
                )
        return references

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
                # The inference-aligned audio route hands over a tensor; the legacy route a numpy waveform.
                sample["audio"] = audio if torch.is_tensor(audio) else torch.from_numpy(audio).float()
                sample["sample_rate"] = sample_rate
                sample["fps"] = fps
                sample["idx"] = idx

                if self.enable_ref2va:
                    sample["references"] = self._load_references(data_info)

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
        min_video_sample_n_frames=1,
        target_video_sample_fps=None,
        video_sample_fps_tolerance=0.5,
        audio_native_sr_resample=False,
        audio_stereo=False,
        audio_span_includes_last_frame=False,
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
        # Fewest sampled frames a clip has to yield to be usable; see `VideoSpeechDataset` for the rationale.
        self.min_video_sample_n_frames = max(1, int(min_video_sample_n_frames))
        # Frame rate the sampled clip has to land on; see `VideoSpeechDataset` for the rationale.
        self.target_video_sample_fps = target_video_sample_fps
        self.video_sample_fps_tolerance = video_sample_fps_tolerance
        # Whether the audio track is loaded on the MiniMax-H3 inference route; see `VideoSpeechDataset` for the
        # rationale. All three default off: the legacy behaviour is unchanged unless a training script asks for
        # the alignment.
        self.audio_native_sr_resample = audio_native_sr_resample
        self.audio_stereo = audio_stereo
        self.audio_span_includes_last_frame = audio_span_includes_last_frame
        # Resampler the degraded librosa route uses once torchaudio turns out unavailable; decided in the check
        # below, `None` keeps librosa's default.
        self.audio_fallback_res_type = None
        if self.audio_native_sr_resample and torchaudio is None:
            # The aligned route resamples with torchaudio; an ABI mismatch (a torchaudio built against another
            # torch) falls back to the legacy librosa route instead of blocking training. That route then keeps
            # the channels the file carries (a mono file is upmixed by repeating its channel, as in
            # `normalize_reference_audio`) and — where available — resamples with librosa's `kaiser_best`, the
            # closest analogue of the pipeline's torchaudio pass, so the degraded audio stays as near to the
            # inference signal as librosa can get.
            print(
                f"WARNING: audio_native_sr_resample needs a working torchaudio, but importing it failed "
                f"({_torchaudio_import_error}); falling back to the legacy librosa audio route. Reinstall the "
                "torchaudio matching the installed torch (e.g. torch 2.7.0 wants torchaudio 2.7.0) to restore the "
                "inference-aligned audio."
            )
            self.audio_native_sr_resample = False
            try:
                # `kaiser_best` rides on resampy; probe it once here instead of letting every sample fail on it.
                librosa.resample(np.zeros(2, dtype=np.float32), orig_sr=2, target_sr=1, res_type="kaiser_best")
                self.audio_fallback_res_type = "kaiser_best"
            except Exception as e:
                print(
                    f"WARNING: librosa's kaiser_best resampler is unavailable ({e}); the fallback audio keeps "
                    "librosa's default resampler."
                )
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
        audio_id = video_dict.get('audio_path')
        control_video_id = video_dict['control_file_path']

        # Resolve video, audio, and control paths. When the annotation has no audio entry, the audio track is
        # decoded from the video container itself (`load_audio` falls back to torchaudio's ffmpeg backend).
        if self.data_root is None:
            video_path = video_id
            audio_path = audio_id if audio_id else video_id
            control_path = control_video_id
        else:
            video_path = os.path.join(self.data_root, video_id)
            audio_path = os.path.join(self.data_root, audio_id) if audio_id else video_path
            control_path = os.path.join(self.data_root, control_video_id)

        if audio_id and not os.path.exists(audio_path):
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

            # Compared on the unrounded rate: 23.976 fps passes a tolerance around 24 while `new_fps` floors it to 23.
            # A clip outside the tolerance is skipped by `__getitem__`'s retry — there is no resampling fallback, as
            # playing it on the fixed 24 fps timeline would slow the picture down and drag the soundtrack's pitch
            # down with it. `frame_step` is how many source frames one sampled frame advances.
            frame_step = float(local_video_sample_stride)
            if self.target_video_sample_fps is not None:
                effective_fps = fps / local_video_sample_stride
                if abs(effective_fps - self.target_video_sample_fps) > self.video_sample_fps_tolerance:
                    raise ValueError(
                        f"Frame rate mismatch: {video_path} samples at {effective_fps:.3f} fps (source {fps:.3f} fps "
                        f"at stride {local_video_sample_stride}), outside "
                        f"{self.target_video_sample_fps} +/- {self.video_sample_fps_tolerance} fps that this training "
                        "run reads its frames on; skipping."
                    )

            # Calculate the actual number of sampled video frames (considering boundaries)
            max_possible_frames = int((total_frames - 1) / frame_step) + 1
            actual_n_frames = min(self.video_sample_n_frames, max_possible_frames)
            if actual_n_frames < self.min_video_sample_n_frames:
                raise ValueError(
                    f"Video too short: {video_path} yields {actual_n_frames} sampled frame(s) at stride "
                    f"{local_video_sample_stride}, fewer than the {self.min_video_sample_n_frames} this training "
                    "run needs; skipping."
                )

            # Randomly select the starting frame
            frame_span = (actual_n_frames - 1) * frame_step
            max_start = total_frames - 1 - int(math.ceil(frame_span))
            start_frame = random.randint(0, max_start) if max_start > 0 else 0
            frame_indices = [
                min(total_frames - 1, int(round(start_frame + index * frame_step)))
                for index in range(actual_n_frames)
            ]

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
                    if len(motion_resized_frames) > 0:
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
        # The sampled frames span `(actual_n_frames - 1) * frame_step` source frames, i.e.
        # `(actual_n_frames - 1) * frame_step / fps` seconds on the source timeline. With
        # `audio_span_includes_last_frame` the span carries one more
        # frame period: MiniMax-H3's inference reads a soundtrack over `num_frames / fps` seconds — the last frame
        # holds its own duration — and its audio latent grid (`round(L / fps * 40)`) keys off that convention.
        frame_periods = actual_n_frames if self.audio_span_includes_last_frame else actual_n_frames - 1
        duration = frame_periods * frame_step / fps
        end_time = start_time + duration

        if not self.audio_native_sr_resample:
            # Load entire audio and resample to target sample rate. `audio_stereo` only reaches this branch as the
            # degraded substitute for the torchaudio route, where it keeps the channels the file carries and uses
            # the closest librosa resampler to the pipeline's torchaudio pass; without it this is the
            # pre-alignment behaviour: mono mixdown with librosa's default resampler.
            audio_input, sample_rate = load_audio(
                audio_path, self.audio_sr, mono=not self.audio_stereo,
                res_type=self.audio_fallback_res_type if self.audio_stereo else None,
            )

            # Convert time to sample indices
            start_sample = round(start_time * self.audio_sr)
            target_len = round(duration * self.audio_sr)
            end_sample = start_sample + target_len

            if self.audio_stereo:
                # A `(channels, num_samples)` block sliced on the sample axis; a mono load arrives 1-D.
                audio_input = np.asarray(audio_input)
                if audio_input.ndim == 1:
                    audio_input = audio_input[None]
                elif audio_input.shape[0] > audio_input.shape[1]:
                    audio_input = audio_input.T
                if start_sample >= audio_input.shape[-1]:
                    raise ValueError(f"Audio file too short: {audio_path}")
                audio_segment = audio_input[..., start_sample:end_sample]
                if audio_segment.shape[-1] < target_len:
                    raise ValueError(f"Audio file too short: {audio_path}")
                waveform = torch.from_numpy(np.ascontiguousarray(audio_segment)).float()
                if waveform.shape[0] == 1:
                    # A mono soundtrack is upmixed by repeating its channel, as in `normalize_reference_audio`.
                    waveform = waveform.expand(2, -1).contiguous()
                elif waveform.shape[0] != 2:
                    raise ValueError(
                        f"MiniMax-H3 carries at most two audio channels, got {waveform.shape[0]} in {audio_path}."
                    )
                audio_segment = waveform
                audio_span_samples, audio_span_rate = audio_segment.shape[-1], self.audio_sr
            else:
                # Extract audio segment with validation
                if start_sample >= len(audio_input):
                    raise ValueError(f"Audio file too short: {audio_path}")
                else:
                    audio_segment = audio_input[start_sample:end_sample]
                    if len(audio_segment) < target_len:
                        raise ValueError(f"Audio file too short: {audio_path}")
                audio_span_samples, audio_span_rate = len(audio_segment), self.audio_sr
        else:
            # The inference-aligned route, mirroring `normalize_reference_audio` in the MiniMax-H3 pipeline: the
            # file is read at the rate it carries its samples at, the slice is taken there, and the segment is
            # resampled once afterwards with the same torchaudio pass (kaiser_best) the pipeline uses.
            audio_input, source_sr = load_audio(audio_path, self.audio_sr, mono=not self.audio_stereo, native_sr=True)
            # A `(channels, num_samples)` layout whatever the decoder handed over; a mono decode arrives 1-D.
            audio_input = np.asarray(audio_input)
            if audio_input.ndim == 1:
                audio_input = audio_input[None]
            elif audio_input.shape[0] > audio_input.shape[1]:
                audio_input = audio_input.T

            start_sample = round(start_time * source_sr)
            target_len = round(duration * source_sr)
            end_sample = start_sample + target_len
            if start_sample >= audio_input.shape[-1]:
                raise ValueError(f"Audio file too short: {audio_path}")
            audio_segment = audio_input[..., start_sample:min(end_sample, audio_input.shape[-1])]
            audio_span_samples, audio_span_rate = audio_segment.shape[-1], source_sr
            if audio_segment.shape[-1] < target_len:
                # A soundtrack that ends with the clip's last frame lacks up to one frame period of tail; pad it
                # here rather than retrying forever. Anything short beyond that is a genuinely shorter file.
                shortfall = target_len - audio_segment.shape[-1]
                if shortfall > round(frame_step / fps * source_sr):
                    raise ValueError(f"Audio file too short: {audio_path}")
                audio_segment = np.pad(audio_segment, [(0, 0)] * (audio_segment.ndim - 1) + [(0, shortfall)])

            waveform = torch.from_numpy(np.ascontiguousarray(audio_segment)).float()
            if source_sr != self.audio_sr:
                waveform = torchaudio.transforms.Resample(source_sr, self.audio_sr)(waveform)
            if self.audio_stereo:
                if waveform.shape[0] == 1:
                    # A mono soundtrack is upmixed by repeating its channel, as in `normalize_reference_audio`.
                    waveform = waveform.expand(2, -1).contiguous()
                elif waveform.shape[0] != 2:
                    raise ValueError(
                        f"MiniMax-H3 carries at most two audio channels, got {waveform.shape[0]} in {audio_path}."
                    )
            else:
                waveform = waveform[0]
            audio_segment, sample_rate = waveform, self.audio_sr

        # The sliced waveform must cover the same real-time span that the sampled frames play on the target
        # timeline: `frame_periods / target_fps` seconds. A container whose metadata fps disagrees with its real
        # frame rate slices a proportionally longer / shorter waveform — undetectable from the fps field alone,
        # which the flooring above corrupts further — and surfaces much later as an audio-latent window failure
        # in the training loop. Raise here so the retry of `__getitem__` draws another sample. The tolerance is
        # one frame period plus rounding slack for the waveform-to-latent encoder.
        if self.target_video_sample_fps is not None:
            target_span = frame_periods / self.target_video_sample_fps
            audio_span = audio_span_samples / audio_span_rate
            if abs(audio_span - target_span) > 1.0 / self.target_video_sample_fps + 0.03:
                raise ValueError(
                    f"Audio span mismatch: {video_path} plays {target_span:.3f}s on the "
                    f"{self.target_video_sample_fps} fps timeline but its waveform covers {audio_span:.3f}s, so the "
                    "clip's real frame rate disagrees with its metadata fps; skipping."
                )

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
                sample["audio"] = audio if torch.is_tensor(audio) else torch.from_numpy(audio).float()
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