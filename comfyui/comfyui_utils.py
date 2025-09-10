import os

import folder_paths
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

def search_model_in_possible_folders(possible_folders, model):
    # Check if the model exists in any of the possible folders within folder_paths.models_dir
    for folder in possible_folders:
        candidate_path = os.path.join(folder_paths.models_dir, folder, model)
        if os.path.exists(candidate_path):
            model_name = candidate_path
            break

    # If model_name is still None, check eas_cache_dir for each possible folder
    if model_name is None and os.path.exists(eas_cache_dir):
        for folder in possible_folders:
            candidate_path = os.path.join(eas_cache_dir, folder, model)
            if os.path.exists(candidate_path):
                model_name = candidate_path
                break

    # If model_name is still None, prompt the user to download the model
    if model_name is None:
        print(f"Please download cogvideoxfun model to one of the following directories:")
        for folder in possible_folders:
            print(f"- {os.path.join(folder_paths.models_dir, folder)}")
            if os.path.exists(eas_cache_dir):
                print(f"- {os.path.join(eas_cache_dir, folder)}")
        raise ValueError("Please download Fun model")

    return model_name

def search_sub_dir_in_possible_folders(possible_folders, sub_dir_name="umt5-xxl"):
    new_possible_folders = []
    # Check if the model exists in any of the possible folders within folder_paths.models_dir
    for folder in possible_folders:
        candidate_path = os.path.join(folder_paths.models_dir, folder)
        new_possible_folders.append(candidate_path)

    # If model_name is still None, check eas_cache_dir for each possible folder
    if os.path.exists(eas_cache_dir):
        for folder in possible_folders:
            candidate_path = os.path.join(eas_cache_dir, folder)
            new_possible_folders.append(candidate_path)

    for folder in new_possible_folders:
        for root, dirs, files in os.walk(folder):
            if sub_dir_name in dirs or sub_dir_name in files:
                return os.path.join(root, sub_dir_name)

    print(f"Please download {sub_dir_name} tokenizer model to one of the following directories:")
    for folder in new_possible_folders:
        print(f"- {os.path.join(folder_paths.models_dir, folder)}")
        if os.path.exists(eas_cache_dir):
            print(f"- {os.path.join(eas_cache_dir, folder)}")
    raise ValueError("Please download Fun model")
