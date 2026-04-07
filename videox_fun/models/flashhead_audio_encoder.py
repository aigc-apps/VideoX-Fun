# Modified from https://github.com/MeiGen-AI/InfiniteTalk/blob/main/src/audio_analysis/wav2vec2.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified from InfiniteTalk original implementation

import librosa
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Model as TransformersWav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput


def linear_interpolation(features, seq_len):
    """Linear interpolation for audio features."""
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


class Wav2Vec2Model(TransformersWav2Vec2Model):
    """
    Custom Wav2Vec2Model that supports seq_len parameter for time alignment.
    This matches the official InfiniteTalk implementation.
    """
    def __init__(self, config: Wav2Vec2Config):
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


class FlashHeadAudioEncoder(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """
    Audio encoder for InfiniteTalk model.
    
    Uses Wav2Vec2Model (not Wav2Vec2ForCTC) to extract audio features,
    matching the original InfiniteTalk implementation.
    """
    
    def __init__(self, pretrained_model_path="facebook/wav2vec2-base-960h", device='cpu'):
        super(FlashHeadAudioEncoder, self).__init__()
        
        # Load pretrained model
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_path)
        self.model = Wav2Vec2Model.from_pretrained(pretrained_model_path)
        
        # Freeze feature extractor
        self.model.feature_extractor._freeze_parameters()
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Video frame rate
        self.video_rate = 25  # InfiniteTalk uses 25 fps
        
    def extract_audio_feat(
        self,
        audio_path,
        return_all_layers=True,
        sr=16000,
        video_length=None,
    ):
        """
        Extract audio features from audio file.
        
        Args:
            audio_path: Path to audio file
            return_all_layers: Whether to return all hidden states (default True for InfiniteTalk)
            sr: Sample rate (default 16000)
            video_length: Target video length in frames
            
        Returns:
            Audio features tensor
        """
        # Load audio
        audio_input, sample_rate = librosa.load(audio_path, sr=sr)
        
        # Calculate video_length if not provided
        if video_length is None:
            audio_duration = len(audio_input) / sr
            video_length = int(audio_duration * self.video_rate)
        
        # Extract features
        input_values = self.feature_extractor(
            audio_input, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values
        
        # Inference
        with torch.no_grad():
            res = self.model(
                input_values.to(self.model.device),
                seq_len=video_length,  # Custom Wav2Vec2Model supports seq_len
                output_hidden_states=True
            )
            
            if return_all_layers:
                # Stack all hidden states (excluding embedding layer)
                feat = torch.stack(res.hidden_states[1:], dim=1).squeeze(0)
                feat = rearrange(feat, "b s d -> s b d")
            else:
                feat = res.hidden_states[-1]
        
        return feat

    def extract_audio_feat_without_file_load(
        self, 
        audio_array, 
        sample_rate, 
        return_all_layers=True, 
        video_length=None
    ):
        """
        Extract audio features from audio array (streaming mode).
        Matches official FlashHead preprocess_audio logic.
        
        Args:
            audio_array: Audio array (numpy or torch)
            sample_rate: Sample rate of the audio
            return_all_layers: Whether to return all hidden states
            video_length: Target video length in frames
            
        Returns:
            Audio features tensor [T, num_layers, dim]
        """
        # Convert to numpy if tensor
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.cpu().numpy()
        
        # Extract features
        input_values = self.feature_extractor(
            audio_array, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values
        
        # Calculate video_length if not provided
        if video_length is None:
            audio_duration = len(audio_array) / sample_rate
            video_length = int(audio_duration * self.video_rate)
        
        # Inference
        with torch.no_grad():
            res = self.model(
                input_values.to(self.model.device),
                seq_len=video_length,
                output_hidden_states=True
            )
            
            if return_all_layers:
                feat = torch.stack(res.hidden_states[1:], dim=1).squeeze(0)
                feat = rearrange(feat, "b s d -> s b d")
            else:
                feat = res.hidden_states[-1]
        
        return feat
    
