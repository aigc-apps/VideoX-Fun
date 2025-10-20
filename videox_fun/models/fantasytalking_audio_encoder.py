# Modified from https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/s2v/audio_encoder.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class FantasyTalkingAudioEncoder(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    def __init__(self, pretrained_model_path="facebook/wav2vec2-base-960h", device='cpu'):
        super(FantasyTalkingAudioEncoder, self).__init__()
        # load pretrained model
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model_path)
        self.model = Wav2Vec2Model.from_pretrained(pretrained_model_path)
        self.model = self.model.to(device)

    def extract_audio_feat(self, audio_path, num_frames = 81, fps = 16, sr = 16000):
        audio_input, sample_rate = librosa.load(audio_path, sr=sr)

        start_time = 0
        end_time = num_frames / fps

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        try:
            audio_segment = audio_input[start_sample:end_sample]
        except:
            audio_segment = audio_input

        input_values = self.processor(
            audio_segment, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values.to(self.model.device, self.model.dtype)

        with torch.no_grad():
            fea = self.model(input_values).last_hidden_state
        return fea

    def extract_audio_feat_without_file_load(self, audio_segment, sample_rate):
        input_values = self.processor(
            audio_segment, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values.to(self.model.device, self.model.dtype)

        with torch.no_grad():
            fea = self.model(input_values).last_hidden_state
        return fea