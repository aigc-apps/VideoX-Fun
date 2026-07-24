# Modified from https://github.com/meituan-longcat/LongCat-Video/blob/main/longcat_video/audio_process/wav2vec2.py
import copy
import logging
import math
import os

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Model as Wav2Vec2Model_base
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PositionalConvEmbedding, Wav2Vec2SamePadLayer)


def linear_interpolation(features, seq_len):
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


def _Wav2Vec2PositionalConvEmbedding_init_hack_(self, config):
        super(Wav2Vec2PositionalConvEmbedding, self).__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.padding = Wav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]


Wav2Vec2PositionalConvEmbedding.__init__ = _Wav2Vec2PositionalConvEmbedding_init_hack_


# the implementation of Wav2Vec2Model is borrowed from
# https://github.com/huggingface/transformers/blob/HEAD/src/transformers/models/wav2vec2/modeling_wav2vec2.py
# initialize our encoder with the pre-trained wav2vec 2.0 weights.
class Wav2Vec2Mode(Wav2Vec2Model_base):
    def __init__(self, config: Wav2Vec2Config):
        config.attn_implementation = "eager"
        super().__init__(config)


    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.config._attn_implementation = "eager"
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )
            

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Wav2Vec2ModelWrapper(nn.Module):
    def __init__(self, config_path, device='cuda', prefix='wav2vec2.'):
        super(Wav2Vec2ModelWrapper, self).__init__()

        config, model_kwargs = Wav2Vec2Config.from_pretrained(
                config_path,
                return_unused_kwargs=True,
                force_download=False,
                local_files_only=True,
            )

        model_path = os.path.join(config_path, 'pytorch_model.bin')
        state_dict = torch.load(model_path, map_location=device)

        config.name_or_path = config_path
        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        # config = Wav2Vec2Mode._autoset_attn_implementation(config, use_flash_attention_2=False)

        # init model
        with torch.device('meta'):
            model = Wav2Vec2Mode(config)

        # load checkpoint
        logging.info(f'loading {model_path}')
        if prefix is not None:
            state_dict = {i.replace(prefix, ''):state_dict[i] for i in state_dict}
        
        model.tie_weights()
        m, u = model.load_state_dict(state_dict, assign=True, strict=False)
            
        model.tie_weights()
        model.eval()

        self.model = model
    
    @property
    def feature_extractor(self):
        return self.model.feature_extractor

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return self.model(
            input_values,
            seq_len,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        return self.model.feature_extract(
            input_values,
            seq_len
        )

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return self.model.encode(
            extract_features,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class LongCatVideoAudioEncoder(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """Audio encoder for LongCatVideo Avatar pipeline.
    
    This class provides a clean interface for audio feature extraction,
    similar to FantasyTalkingAudioEncoder but with LongCatVideo-specific
    audio preprocessing (loudness normalization, noise floor, transient smoothing).
    
    Uses existing Wav2Vec2ModelWrapper and Wav2Vec2FeatureExtractor internally.
    """
    
    def __init__(self, config_path, device='cpu', prefix='wav2vec2.'):
        super(LongCatVideoAudioEncoder, self).__init__()
        
        # Use existing Wav2Vec2ModelWrapper
        self.audio_encoder = Wav2Vec2ModelWrapper(config_path, device=device, prefix=prefix)
        
        # Use existing Wav2Vec2FeatureExtractor
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config_path)
    
    @property
    def dtype(self):
        return self.audio_encoder.dtype

    @property
    def device(self):
        return self.audio_encoder.device
    
    def _loudness_norm(self, audio_array, sr=16000, lufs=-23, threshold=100):
        """Normalize audio loudness to target LUFS."""
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > threshold:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio

    def _add_noise_floor(self, audio, noise_db=-45):
        """Add noise floor to audio."""
        noise_amp = 10 ** (noise_db / 20)
        noise = np.random.randn(len(audio)) * noise_amp
        return audio + noise

    def _smooth_transients(self, audio, sr=16000):
        """Smooth audio transients using low-pass filter."""
        import scipy.signal as ss
        b, a = ss.butter(3, 3000 / (sr / 2))
        return ss.lfilter(b, a, audio)
    
    def _preprocess_audio(self, speech_array, sample_rate=16000):
        """Apply LongCatVideo-specific audio preprocessing."""
        speech_array = self._loudness_norm(speech_array, sample_rate)
        speech_array = self._add_noise_floor(speech_array)
        speech_array = self._smooth_transients(speech_array, sample_rate)
        return speech_array
    
    @torch.no_grad()
    def _extract_embedding(self, speech_array, sample_rate, num_frames, audio_stride=2):
        """Core method to extract audio embedding from preprocessed speech array.
        
        Args:
            speech_array: Preprocessed audio array.
            sample_rate: Audio sample rate.
            num_frames: Number of video frames.
            audio_stride: Audio stride for sliding window.
            
        Returns:
            Audio embeddings tensor of shape [1, num_frames, 5, 12, 768].
        """
        seq_len = int(audio_stride * num_frames)
        
        # wav2vec_feature_extractor
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=sample_rate).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device, dtype=self.dtype)
        audio_feature = audio_feature.unsqueeze(0)

        # audio embedding using Wav2Vec2ModelWrapper
        embeddings = self.audio_encoder(audio_feature, seq_len=seq_len, output_hidden_states=True)

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d").contiguous()  # T, 12, 768

        # Prepare audio embedding with sliding window
        indices = torch.arange(2 * 2 + 1) - 2  # [-2, -1, 0, 1, 2]
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + audio_stride * num_frames
        
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + \
            indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=audio_emb.shape[0] - 1)
        audio_emb = audio_emb[center_indices][None, ...]  # [1, num_frames, 5, 12, 768]
        
        return audio_emb
    
    def extract_audio_feat(
        self, 
        audio_path, 
        num_frames=49, 
        fps=16, 
        sr=16000,
        audio_stride=2
    ):
        """Extract audio features from audio file.
        
        Args:
            audio_path: Path to audio file.
            num_frames: Number of video frames.
            fps: Video frames per second.
            sr: Audio sample rate.
            audio_stride: Audio stride for sliding window.
            
        Returns:
            Audio embeddings tensor of shape [1, num_frames, 5, 12, 768].
        """
        # Load audio
        speech_array, sample_rate = librosa.load(audio_path, sr=sr)
        
        # Pad audio to target length
        generate_duration = num_frames / fps
        source_duration = len(speech_array) / sample_rate
        added_sample_nums = math.ceil((generate_duration - source_duration) * sample_rate)
        if added_sample_nums > 0:
            speech_array = np.append(speech_array, [0.] * added_sample_nums)
        
        # Preprocess and extract embedding
        speech_array = self._preprocess_audio(speech_array, sample_rate)
        return self._extract_embedding(speech_array, sample_rate, num_frames, audio_stride)
    
    def extract_audio_feat_without_file_load(
        self, 
        audio_segment, 
        sample_rate, 
        num_frames=49, 
        audio_stride=2
    ):
        """Extract audio features from audio array without file loading.
        
        Args:
            audio_segment: Audio array (numpy array).
            sample_rate: Audio sample rate.
            num_frames: Number of video frames.
            audio_stride: Audio stride for sliding window.
            
        Returns:
            Audio embeddings tensor of shape [1, num_frames, 5, 12, 768].
        """
        # Preprocess and extract embedding
        speech_array = self._preprocess_audio(audio_segment, sample_rate)
        return self._extract_embedding(speech_array, sample_rate, num_frames, audio_stride)