from .cogvideox_fun.nodes import (CogVideoX_Fun_I2VSampler,
                                  CogVideoX_Fun_T2VSampler,
                                  CogVideoX_Fun_V2VSampler,
                                  LoadCogVideoX_Fun_Lora,
                                  LoadCogVideoX_Fun_Model)

from .wan2_1.nodes import (LoadWanModel,
                           LoadWanLora,
                           WanT2VSampler,
                           WanI2VSampler)

class CogVideoX_FUN_TextBox:
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
    "CogVideoX_FUN_TextBox": CogVideoX_FUN_TextBox,
    "LoadCogVideoX_Fun_Model": LoadCogVideoX_Fun_Model,
    "LoadCogVideoX_Fun_Lora": LoadCogVideoX_Fun_Lora,
    "CogVideoX_Fun_I2VSampler": CogVideoX_Fun_I2VSampler,
    "CogVideoX_Fun_T2VSampler": CogVideoX_Fun_T2VSampler,
    "CogVideoX_Fun_V2VSampler": CogVideoX_Fun_V2VSampler,
    "LoadWanModel": LoadWanModel,
    "LoadWanLora": LoadWanLora,
    "WanT2VSampler": WanT2VSampler,
    "WanI2VSampler": WanI2VSampler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CogVideoX_FUN_TextBox": "CogVideoX_FUN_TextBox",
    "LoadCogVideoX_Fun_Model": "Load CogVideoX-Fun Model",
    "LoadCogVideoX_Fun_Lora": "Load CogVideoX-Fun Lora",
    "CogVideoX_Fun_I2VSampler": "CogVideoX-Fun Sampler for Image to Video",
    "CogVideoX_Fun_T2VSampler": "CogVideoX-Fun Sampler for Text to Video",
    "CogVideoX_Fun_V2VSampler": "CogVideoX-Fun Sampler for Video to Video",

    "LoadWanModel": "Load Wan Model",
    "LoadWanLora": "Load Wan Lora",
    "WanT2VSampler": "Wan Sampler for Text to Video",
    "WanI2VSampler": "Wan Sampler for Image to Video",
}