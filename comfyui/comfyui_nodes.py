from .cogvideox_fun.nodes import (CogVideoXFunInpaintSampler,
                                  CogVideoXFunT2VSampler,
                                  CogVideoXFunV2VSampler,
                                  LoadCogVideoXFunLora,
                                  LoadCogVideoXFunModel)

from .wan2_1.nodes import (LoadWanModel,
                           LoadWanLora,
                           WanT2VSampler,
                           WanI2VSampler)

from .wan2_1_fun.nodes import (LoadWanFunModel,
                           LoadWanFunLora,
                           WanFunT2VSampler,
                           WanFunInpaintSampler,
                           WanFunV2VSampler)
from .annotator.nodes import VideoToCanny, VideoToDepth, VideoToPose

class FunTextBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "",}),
            }
        }
    
    RETURN_TYPES = ("STRING_PROMPT",)
    RETURN_NAMES =("prompt",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, prompt):
        return (prompt, )


NODE_CLASS_MAPPINGS = {
    "FunTextBox": FunTextBox,

    "LoadCogVideoXFunModel": LoadCogVideoXFunModel,
    "LoadCogVideoXFunLora": LoadCogVideoXFunLora,
    "CogVideoXFunT2VSampler": CogVideoXFunT2VSampler,
    "CogVideoXFunInpaintSampler": CogVideoXFunInpaintSampler,
    "CogVideoXFunV2VSampler": CogVideoXFunV2VSampler,

    "LoadWanModel": LoadWanModel,
    "LoadWanLora": LoadWanLora,
    "WanT2VSampler": WanT2VSampler,
    "WanI2VSampler": WanI2VSampler,

    "LoadWanFunModel": LoadWanFunModel,
    "LoadWanFunLora": LoadWanFunLora,
    "WanFunT2VSampler": WanFunT2VSampler,
    "WanFunInpaintSampler": WanFunInpaintSampler,
    "WanFunV2VSampler": WanFunV2VSampler,

    "VideoToCanny": VideoToCanny,
    "VideoToDepth": VideoToDepth,
    "VideoToOpenpose": VideoToPose,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "FunTextBox": "FunTextBox",
    "LoadCogVideoXFunModel": "Load CogVideoX-Fun Model",
    "LoadCogVideoXFunLora": "Load CogVideoX-Fun Lora",
    "CogVideoXFunInpaintSampler": "CogVideoX-Fun Sampler for Image to Video",
    "CogVideoXFunT2VSampler": "CogVideoX-Fun Sampler for Text to Video",
    "CogVideoXFunV2VSampler": "CogVideoX-Fun Sampler for Video to Video",

    "LoadWanModel": "Load Wan Model",
    "LoadWanLora": "Load Wan Lora",
    "WanT2VSampler": "Wan Sampler for Text to Video",
    "WanI2VSampler": "Wan Sampler for Image to Video",

    "LoadWanFunModel": "Load Wan Fun Model",
    "LoadWanFunLora": "Load Wan Fun Lora",
    "WanFunT2VSampler": "Wan Fun Sampler for Text to Video",
    "WanFunInpaintSampler": "Wan Fun Sampler for Image to Video",
    "WanFunV2VSampler": "Wan Fun Sampler for Video to Video",

    "VideoToCanny": "Video To Canny",
    "VideoToDepth": "Video To Depth",
    "VideoToOpenpose": "Video To Pose",
}