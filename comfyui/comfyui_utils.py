import os

import numpy as np
import torch
from PIL import Image

# Compatible with Alibaba EAS for quick launch
eas_cache_dir       = '/stable-diffusion-cache/models'
# The directory of the cogvideoxfun
script_directory    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

def numpy2pil(image):
    return Image.fromarray(np.clip(255. * image, 0, 255).astype(np.uint8))

def to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        return tensor2pil(image)
    if isinstance(image, np.ndarray):
        return numpy2pil(image)
    raise ValueError(f"Cannot convert {type(image)} to PIL.Image")