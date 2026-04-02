from __future__ import annotations

import base64
import logging
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Literal, Optional, Union

import huggingface_hub
import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

logger = logging.getLogger(__name__)

# ========================= Constants =========================
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


# ========================= DataClasses =========================

@dataclass
class DataConfig:
    train_json_list: List[str] = field(default_factory=lambda: ["/path/to/dataset/meta_data.json"])
    val_json_list: List[str] = field(default_factory=lambda: ["/path/to/dataset/meta_data.json"])
    test_json_list: List[str] = field(default_factory=lambda: ["/path/to/dataset/meta_data.json"])
    soft_label: bool = False
    confidence_threshold: Optional[float] = None
    max_pixels: Optional[int] = 256 * 28 * 28
    min_pixels: Optional[int] = 256 * 28 * 28
    with_instruction: bool = True
    tied_threshold: Optional[float] = None


@dataclass
class PEFTLoraConfig:
    lora_enable: bool = False
    vision_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    lora_namespan_exclude: Optional[List[str]] = None
    lora_modules_to_save: Optional[List[str]] = None
    lora_task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    num_lora_modules: int = -1

    def __post_init__(self):
        if isinstance(self.lora_target_modules, list) and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]

        if isinstance(self.lora_namespan_exclude, list) and len(self.lora_namespan_exclude) == 1:
            self.lora_namespan_exclude = self.lora_namespan_exclude[0]


@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = None
    model_revision: str = "main"
    rm_head_type: str = "default"
    rm_head_kwargs: Optional[dict] = None
    output_dim: int = 1

    use_special_tokens: bool = False

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    tune_merger: bool = field(default=False)
    trainable_visual_layers: Optional[int] = -1

    torch_dtype: Optional[Literal["auto", "bfloat16", "float16", "float32"]] = None
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
    use_bnb_nested_quant: bool = False
    reward_token: Literal["last", "mean", "special"] = "last"
    loss_type: Literal["bt", "reg", "btt", "margin", "constant_margin", "scaled"] = "regular"
    loss_hyperparameters: dict = field(default_factory=lambda: {})
    checkpoint_path: Optional[str] = None

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


@dataclass
class TrainingConfig:
    max_grad_norm: Optional[float] = 1.0
    dataset_num_proc: Optional[int] = None
    center_rewards_coefficient: Optional[float] = None
    disable_flash_attn2: bool = field(default=False)
    disable_dropout: bool = field(default=False)

    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    rm_head_lr: Optional[float] = None
    special_token_lr: Optional[float] = None

    conduct_eval: Optional[bool] = True
    load_from_pretrained: str = None
    load_from_pretrained_step: int = None
    logging_epochs: Optional[float] = None
    eval_epochs: Optional[float] = None
    save_epochs: Optional[float] = None
    remove_unused_columns: Optional[bool] = False

    save_full_model: Optional[bool] = False
    gradient_checkpointing: bool = field(default=False)
    bf16: bool = field(default=False)
    fp16: bool = field(default=False)
    output_dir: str = None


# ========================= Prompt Templates =========================

INSTRUCTION = """
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. 

**Visual Quality:**  
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. 

**Text Alignment:**  
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
Textual prompt - {text_prompt}


"""

prompt_with_special_token = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

prompt_without_special_token = """
Please provide the overall ratings of this image: 
"""


# ========================= Vision Processing =========================

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, torch.Tensor):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    
    if isinstance(image_obj, Image.Image):
        image = image_obj.convert("RGB")
    
    # resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        if isinstance(image, torch.Tensor):
            shape = image.shape
            if len(shape) == 4:
                if shape[1] in [1, 3]:
                    height, width = shape[2], shape[3]
                    image_mode = 'NCHW'
                elif shape[3] in [1, 3]:
                    height, width = shape[1], shape[2]
                    image_mode = 'NHWC'
            elif len(shape) == 3:
                if shape[0] in [1, 3]:
                    height, width = shape[1], shape[2]
                    image_mode = 'CHW'
                elif shape[2] in [1, 3]:
                    height, width = shape[0], shape[1]
                    image_mode = 'HWC'
                else:
                    raise ValueError(f"Cannot determine tensor image format from shape {shape}")
            else:
                raise ValueError(f"Unsupported tensor image shape: {shape}")
        else:
            width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    if isinstance(image, torch.Tensor):
        if image_mode == 'NCHW':
            image = transforms.functional.resize(
                image, [resized_height, resized_width], interpolation=InterpolationMode.BICUBIC, antialias=True
            )
        elif image_mode == 'NHWC':
            image = transforms.functional.resize(
                image.permute(0, 3, 1, 2), [resized_height, resized_width], interpolation=InterpolationMode.BICUBIC, antialias=True
            )
        elif image_mode == 'CHW':
            image = image.unsqueeze(0)
            image = transforms.functional.resize(
                image, [resized_height, resized_width], interpolation=InterpolationMode.BICUBIC, antialias=True
            )
        elif image_mode == 'HWC':
            image = image.permute(2, 0, 1).unsqueeze(0)
            image = transforms.functional.resize(
                image, [resized_height, resized_width], interpolation=InterpolationMode.BICUBIC, antialias=True
            )
    else:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((resized_width, resized_height), Image.BICUBIC)

    return image


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    vision_infos = extract_vision_info(conversations)
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            raise ValueError("Video input is not supported in HPSv3 image scoring.")
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs


# ========================= Model =========================

class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
    def __init__(
        self,
        config,
        output_dim=4,
        reward_token="last",
        special_token_ids=None,
        rm_head_type="default",
        rm_head_kwargs=None,
    ):
        super().__init__(config)
        self.output_dim = output_dim
        if rm_head_type == "default":
            self.rm_head = nn.Linear(config.hidden_size, output_dim, bias=False)
        elif rm_head_type == "ranknet":
            if rm_head_kwargs is not None:
                for layer in range(rm_head_kwargs.get("num_layers", 3)):
                    if layer == 0:
                        self.rm_head = nn.Sequential(
                            nn.Linear(config.hidden_size, rm_head_kwargs["hidden_size"]),
                            nn.ReLU(),
                            nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                        )
                    elif layer < rm_head_kwargs.get("num_layers", 3) - 1:
                        self.rm_head.add_module(
                            f"layer_{layer}",
                            nn.Sequential(
                                nn.Linear(rm_head_kwargs["hidden_size"], rm_head_kwargs["hidden_size"]),
                                nn.ReLU(),
                                nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                            ),
                        )
                    else:
                        self.rm_head.add_module(
                            f"output_layer",
                            nn.Linear(rm_head_kwargs["hidden_size"], output_dim, bias=rm_head_kwargs.get("bias", False)),
                        )
            else:
                self.rm_head = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(1024, 16),
                    nn.ReLU(),
                    nn.Linear(16, output_dim),
                )

        self.rm_head.to(torch.float32)
        self.reward_token = reward_token

        self.special_token_ids = special_token_ids
        if self.special_token_ids is not None:
            self.reward_token = "special"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            logits = self.rm_head(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        if self.reward_token == "last":
            pooled_logits = logits[
                torch.arange(batch_size, device=logits.device), sequence_lengths
            ]
        elif self.reward_token == "mean":
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack(
                [logits[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)]
            )
        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (
                    input_ids == special_token_id
                )
            pooled_logits = logits[special_token_mask, ...]
            pooled_logits = pooled_logits.view(batch_size, 1, -1)
            pooled_logits = pooled_logits.view(batch_size, -1)
        else:
            raise ValueError("Invalid reward_token")

        return {"logits": pooled_logits}


# ========================= Model Utils =========================

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=False):
    linear_cls = torch.nn.Linear
    embedding_cls = torch.nn.Embedding
    lora_module_names = []
    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def create_model_and_processor(model_config, peft_lora_config, training_args, cache_dir=None):
    """Create model and processor for inference."""
    try:
        import flash_attn
        flash_attn_available = True
    except ImportError:
        flash_attn_available = False
        print("Flash Attention is not installed. Falling to SDPA.")

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=None,
        quantization_config=None,
        use_cache=True,
    )

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, padding_side="right", cache_dir=cache_dir
    )

    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ["<|Reward|>"]
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    model = Qwen2VLRewardModelBT.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation=(
            "flash_attention_2" if not training_args.disable_flash_attn2 and flash_attn_available else "sdpa"
        ),
        cache_dir=cache_dir,
        rm_head_type=model_config.rm_head_type,
        rm_head_kwargs=model_config.rm_head_kwargs,
        **model_kwargs,
    )

    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))

    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

    model.rm_head.to(torch.float32)

    # create lora and peft model
    if peft_lora_config.lora_enable:
        target_modules = find_target_linear_names(
            model,
            num_lora_modules=peft_lora_config.num_lora_modules,
            lora_namespan_exclude=peft_lora_config.lora_namespan_exclude,
        )
        peft_config = LoraConfig(
            target_modules=target_modules,
            r=peft_lora_config.lora_r,
            lora_alpha=peft_lora_config.lora_alpha,
            lora_dropout=peft_lora_config.lora_dropout,
            task_type=peft_lora_config.lora_task_type,
            use_rslora=peft_lora_config.use_rslora,
            bias="none",
            modules_to_save=peft_lora_config.lora_modules_to_save,
        )
        model = get_peft_model(model, peft_config)
    else:
        peft_config = None

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor, peft_config


# ========================= Default Config =========================

def get_default_config():
    """Get default HPSv3 configuration."""
    model_config = ModelConfig(
        model_name_or_path="Qwen/Qwen2-VL-7B-Instruct",
        rm_head_type="ranknet",
        output_dim=2,
        use_special_tokens=True,
        reward_token="special",
        torch_dtype="bfloat16",
    )
    
    peft_lora_config = PEFTLoraConfig(
        lora_enable=False,
        lora_namespan_exclude=['lm_head', 'rm_head', 'embed_tokens'],
    )
    
    training_args = TrainingConfig(
        disable_flash_attn2=False,
        bf16=True,
        output_dir="",
    )
    
    data_config = DataConfig(
        max_pixels=256 * 28 * 28,
        min_pixels=256 * 28 * 28,
    )
    
    return data_config, training_args, model_config, peft_lora_config


# ========================= Inference =========================

class HPSv3RewardInferencer:
    """
    HPSv3 Reward Model Inferencer - Standalone version without hpsv3 package dependency.
    
    This class provides inference functionality for HPSv3 image reward scoring.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.safetensors or .pth file).
                        If None, will download from HuggingFace Hub.
        device: Device to run inference on (default: 'cuda').
        dtype: Model dtype (default: torch.bfloat16).
    
    Example:
        >>> inferencer = HPSv3RewardInferencer()
        >>> image_paths = ["image1.png", "image2.png"]
        >>> prompts = ["a cat", "a dog"]
        >>> rewards = inferencer.reward(image_paths, prompts)
        >>> print(rewards[0][0].item())  # Mean reward for first image
    """
    
    def __init__(self, checkpoint_path=None, device='cuda', dtype=torch.bfloat16):
        if checkpoint_path is None:
            checkpoint_path = huggingface_hub.hf_hub_download(
                "MizzenAI/HPSv3", 'HPSv3.safetensors', repo_type='model'
            )

        # Get default config
        data_config, training_args, model_config, peft_lora_config = get_default_config()
        
        # Override dtype
        if dtype == torch.bfloat16:
            training_args.bf16 = True
            training_args.fp16 = False
        elif dtype == torch.float16:
            training_args.bf16 = False
            training_args.fp16 = True
        else:
            training_args.bf16 = False
            training_args.fp16 = False

        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
        )

        self.device = device
        self.use_special_tokens = model_config.use_special_tokens

        # Load checkpoint
        if checkpoint_path.endswith('.safetensors'):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)
        self.data_config = data_config

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
        """Pad the sequences to the maximum length."""
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask
        
        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(
            sequences, padding, 'constant', self.processor.tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.functional.pad(
            attention_mask, padding, 'constant', 0
        )

        return sequences_padded, attention_mask_padded
    
    def _prepare_input(self, data):
        """Prepare inputs before feeding them to the model."""
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs):
        """Prepare inputs before feeding them to the model."""
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError("Empty inputs")
        return inputs
    
    def prepare_batch(self, image_paths, prompts):
        """
        Prepare batch inputs for the model.
        
        Args:
            image_paths: List of image paths or PIL Images.
            prompts: List of text prompts corresponding to each image.
            
        Returns:
            Batch dictionary ready for model inference.
        """
        max_pixels = 256 * 28 * 28
        min_pixels = 256 * 28 * 28
        message_list = []
        for text, image in zip(prompts, image_paths):
            out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        },
                        {
                            "type": "text",
                            "text": (
                                INSTRUCTION.format(text_prompt=text)
                                + prompt_with_special_token
                                if self.use_special_tokens
                                else prompt_without_special_token
                            ),
                        },
                    ],
                }
            ]

            message_list.append(out_message)

        image_inputs, _ = process_vision_info(message_list)

        batch = self.processor(
            text=self.processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    @torch.inference_mode()
    def reward(self, image_paths, prompts):
        """
        Compute reward scores for given images and prompts.
        
        Args:
            image_paths: List of image paths, URLs, or PIL Images.
            prompts: List of text prompts corresponding to each image.
            
        Returns:
            Tensor of shape [B, output_dim] containing reward scores.
            For HPSv3, output_dim=2 where:
            - rewards[:, 0]: Mean reward score (use this as the final score)
            - rewards[:, 1]: Uncertainty/sigma
        
        Example:
            >>> rewards = inferencer.reward(["image.png"], ["a beautiful sunset"])
            >>> score = rewards[0][0].item()  # Get the mean score
        """
        batch = self.prepare_batch(image_paths, prompts)
        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"]

        return rewards


# ========================= Main =========================
if __name__ == "__main__":
    # Example usage
    checkpoint_path = None  # Will download from HuggingFace Hub
    device = 'cuda'
    dtype = torch.bfloat16
    
    inferencer = HPSv3RewardInferencer("Diffusion_Transformer/HPSv3/HPSv3.safetensors", device=device, dtype=dtype)

    image_paths = [
        "assets/example1.png",
        "assets/example2.png"
    ]
    prompts = [
        "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker",
        "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker"
    ]
    
    rewards = inferencer.reward(image_paths, prompts)
    print(f"Image 1 score: {rewards[0][0].item():.4f}")  # miu and sigma, we select miu as the final output
    print(f"Image 2 score: {rewards[1][0].item():.4f}")
