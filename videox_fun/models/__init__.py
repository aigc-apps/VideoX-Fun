from diffusers import AutoencoderKL
from transformers import (AutoProcessor, AutoTokenizer, CLIPImageProcessor,
                          CLIPTextModel, CLIPTokenizer,
                          CLIPVisionModelWithProjection,
                          Gemma3ForConditionalGeneration, Gemma3Processor,
                          GemmaTokenizer, GemmaTokenizerFast, LlamaModel,
                          LlamaTokenizerFast, LlavaForConditionalGeneration,
                          Mistral3ForConditionalGeneration, PixtralProcessor,
                          Qwen3Config, Qwen3ForCausalLM, T5EncoderModel,
                          T5Tokenizer, T5TokenizerFast, UMT5EncoderModel,
                          Wav2Vec2FeatureExtractor)

try:
    from transformers import (Qwen2_5_VLConfig,
                              Qwen2_5_VLForConditionalGeneration,
                              Qwen2Tokenizer, Qwen2VLProcessor)
except Exception:
    Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer = None, None
    Qwen2VLProcessor, Qwen2_5_VLConfig = None, None
    print("Your transformers version is too old to load Qwen2_5_VLForConditionalGeneration and Qwen2Tokenizer. If you wish to use QwenImage, please upgrade your transformers package to the latest version.")

try:
    from transformers import Qwen3VLForConditionalGeneration
except:
    Qwen3VLForConditionalGeneration = None
    print("Your transformers version is too old to load Qwen3VLForConditionalGeneration. If you wish to use Qwen3VLForConditionalGeneration, please upgrade your transformers package to the latest version.")

try:
    from transformers import Ministral3ForCausalLM, Mistral3Model
except:
    Mistral3Model = None
    Ministral3ForCausalLM = None
    print("Your transformers version is too old to load Mistral3Model and Ministral3ForCausalLM. If you wish to use ErnieImage, please upgrade your transformers package to the latest version.")

try:
    from .lens_text_encoder import LensGptOssEncoder
except ImportError:
    LensGptOssEncoder = None
    print("LensGptOssEncoder not available. Lens requires transformers >= 5.8.0 for GptOssForCausalLM.")

from .cogvideox_transformer3d import CogVideoXTransformer3DModel
from .cogvideox_vae import AutoencoderKLCogVideoX
from .ernie_image_transformer import ErnieImageTransformer2DModel
from .fantasytalking_audio_encoder import FantasyTalkingAudioEncoder
from .fantasytalking_transformer3d import FantasyTalkingTransformer3DModel
from .flashhead_audio_encoder import FlashHeadAudioEncoder
from .flashhead_transformer3d import FlashHeadTransformer3DModel
from .flux2_image_processor import Flux2ImageProcessor
from .flux2_transformer2d import Flux2Transformer2DModel
from .flux2_transformer2d_control import Flux2ControlTransformer2DModel
from .flux2_vae import AutoencoderKLFlux2
from .flux_transformer2d import FluxTransformer2DModel
from .hunyuanvideo_transformer3d import HunyuanVideoTransformer3DModel
from .hunyuanvideo_vae import AutoencoderKLHunyuanVideo
from .infinitetalk_audio_encoder import InfiniteTalkAudioEncoder
from .infinitetalk_transformer3d import InfiniteTalkTransformer3DModel
from .lens_reasoner import LensPromptReasoner
from .lens_transformer2d import LensTransformer2DModel
from .longcatvideo_audio_encoder import (LongCatVideoAudioEncoder,
                                         Wav2Vec2ModelWrapper)
from .longcatvideo_transformer3d import LongCatVideoTransformer3DModel
from .longcatvideo_transformer3d_avatar import \
    LongCatVideoAvatarTransformer3DModel
from .longcatvideo_vae import AutoencoderKLLongCatVideo
from .ltx2_connecter import LTX2TextConnectors
from .ltx2_latent_upsampler import LTX2LatentUpsamplerModel
from .ltx2_transformer3d import LTX2VideoTransformer3DModel
from .ltx2_vae import AutoencoderKLLTX2Video
from .ltx2_vae_audio import AutoencoderKLLTX2Audio
from .ltx2_vocoder import LTX2Vocoder, LTX2VocoderWithBWE
from .mova_audio_transformer3d import WanAudioTransformer3DModel
from .mova_interactionv2 import MOVADualTowerConditionalBridge
from .mova_model import MOVAModel
from .mova_vae_audio import AutoencoderKLMOVAAudio
from .qwenimage_transformer2d import QwenImageTransformer2DModel
from .qwenimage_transformer2d_control import QwenImageControlTransformer2DModel
from .qwenimage_transformer2d_instantx import QwenImageInstantXControlNetModel
from .qwenimage_vae import AutoencoderKLQwenImage
from .turbowan_transformer3d import TurboWanTransformer3DModel
from .wan_audio_encoder import WanAudioEncoder
from .wan_image_encoder import CLIPModel
from .wan_text_encoder import WanT5EncoderModel
from .wan_transformer3d import (Wan2_2Transformer3DModel, WanRMSNorm,
                                WanSelfAttention, WanTransformer3DModel)
from .wan_transformer3d_animate import Wan2_2Transformer3DModel_Animate
from .wan_transformer3d_s2v import Wan2_2Transformer3DModel_S2V
from .wan_transformer3d_self_forcing import WanTransformer3DModel_SelfForcing
from .wan_transformer3d_vace import VaceWanTransformer3DModel
from .wan_vae import AutoencoderKLWan, AutoencoderKLWan_
from .wan_vae3_8 import AutoencoderKLWan2_2_, AutoencoderKLWan3_8
from .z_image_transformer2d import ZImageTransformer2DModel
from .z_image_transformer2d_control import ZImageControlTransformer2DModel
