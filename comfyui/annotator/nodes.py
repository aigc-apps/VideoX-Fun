# This folder is modified from the https://github.com/Mikubill/sd-webui-controlnet
import os

import cv2
import folder_paths
import numpy as np
import torch
from einops import rearrange

from .dwpose_utils import DWposeDetector
from .zoe.zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from .zoe.zoedepth.utils.config import get_config

remote_onnx_det = "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"
remote_onnx_pose = "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
remote_zoe= "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt"

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
    cap.release()
    return frames

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad

def load_file_from_url(
    url: str,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
    hash_prefix: str | None = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    from urllib.parse import urlparse
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress, hash_prefix=hash_prefix)
    return cached_file

class VideoToCanny:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video": ("IMAGE",),
                "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
                "video_length": (
                    "INT", {"default": 81, "min": 1, "max": 81, "step": 4}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, input_video, low_threshold, high_threshold, video_length):
        def extract_canny_frames(frames):
            canny_frames = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, low_threshold, high_threshold)
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                canny_frames.append(edges_colored)
            return canny_frames
        
        if type(input_video) is str:
            video_frames = read_video(input_video)
        else:
            video_frames = np.array(input_video * 255, np.uint8)[:video_length]
        output_video = extract_canny_frames(video_frames)
        output_video = torch.from_numpy(np.array(output_video)) / 255
        return (output_video,)

class VideoToDepth:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video": ("IMAGE",),
                "video_length": (
                    "INT", {"default": 81, "min": 1, "max": 81, "step": 4}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"


    def process_frame(self, model, image, device, weight_dtype):
        with torch.no_grad():
            image, remove_pad = resize_image_with_pad(image, 512)
            image_depth = image
            with torch.no_grad():
                image_depth = torch.from_numpy(image_depth).to(device, weight_dtype)
                image_depth = image_depth / 255.0
                image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
                depth = model.infer(image_depth)

                depth = depth[0, 0].cpu().numpy()

                vmin = np.percentile(depth, 2)
                vmax = np.percentile(depth, 85)

                depth -= vmin
                depth /= vmax - vmin
                depth = 1.0 - depth
                depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)
            image = remove_pad(depth_image)
            image = HWC3(image)
        return image

    def process(self, input_video, video_length):
        model = ZoeDepth.build_from_config(get_config("zoedepth", "infer"))

        # Detect model is existing or not
        possible_folders = ["CogVideoX_Fun/Third_Party", "Fun_Models/Third_Party", "VideoX_Fun/Third_Party"]  # Possible folder names to check

        # Check if the model exists in any of the possible folders within folder_paths.models_dir
        zoe_model_path = "ZoeD_M12_N.pt"
        for folder in possible_folders:
            candidate_path = os.path.join(folder_paths.models_dir, folder, zoe_model_path)
            if os.path.exists(candidate_path):
                zoe_model_path = candidate_path
                break
        if not os.path.exists(zoe_model_path):
            load_file_from_url(remote_zoe, model_dir=os.path.join(folder_paths.models_dir, "Fun_Models/Third_Party"))
            zoe_model_path = os.path.join(folder_paths.models_dir, "Fun_Models/Third_Party", zoe_model_path)

        model.load_state_dict(
            torch.load(zoe_model_path, map_location="cpu")['model'], 
            strict=False
        )
        if torch.cuda.is_available():
            device = "cuda"
            weight_dtype = torch.float32
        else:
            device = "cpu"
            weight_dtype = torch.float32
        model = model.to(device=device, dtype=weight_dtype).eval().requires_grad_(False)

        if isinstance(input_video, str):
            video_frames = read_video(input_video)
        else:
            video_frames = np.array(input_video * 255, np.uint8)[:video_length]

        output_video = [self.process_frame(model, frame, device, weight_dtype) for frame in video_frames]
        output_video = torch.from_numpy(np.array(output_video)) / 255

        return (output_video,)
    

class VideoToPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video": ("IMAGE",),
                "video_length": (
                    "INT", {"default": 81, "min": 1, "max": 81, "step": 4}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process_frame(self, model, image):
        with torch.no_grad():
            image, remove_pad = resize_image_with_pad(image, 512)
            pose_image = model(image)
            image = remove_pad(pose_image)
            image = HWC3(image)
        return image
    
    def process(self, input_video, video_length):
        # Detect model is existing or not
        possible_folders = ["CogVideoX_Fun/Third_Party", "Fun_Models/Third_Party", "VideoX_Fun/Third_Party"]  # Possible folder names to check

        # Check if the model exists in any of the possible folders within folder_paths.models_dir
        onnx_det = "yolox_l.onnx"
        for folder in possible_folders:
            candidate_path = os.path.join(folder_paths.models_dir, folder, onnx_det)
            if os.path.exists(candidate_path):
                onnx_det = candidate_path
                break
        if not os.path.exists(onnx_det):
            load_file_from_url(remote_onnx_det, os.path.join(folder_paths.models_dir, "Fun_Models/Third_Party"))
            onnx_det = os.path.join(folder_paths.models_dir, "Fun_Models/Third_Party", onnx_det)
            
        onnx_pose = "dw-ll_ucoco_384.onnx"
        for folder in possible_folders:
            candidate_path = os.path.join(folder_paths.models_dir, folder, onnx_pose)
            if os.path.exists(candidate_path):
                onnx_pose = candidate_path
                break
        if not os.path.exists(onnx_pose):
            load_file_from_url(remote_onnx_pose, os.path.join(folder_paths.models_dir, "Fun_Models/Third_Party"))
            onnx_pose = os.path.join(folder_paths.models_dir, "Fun_Models/Third_Party", onnx_pose)
        
        model = DWposeDetector(onnx_det, onnx_pose)

        if isinstance(input_video, str):
            video_frames = read_video(input_video)
        else:
            video_frames = np.array(input_video * 255, np.uint8)[:video_length]

        output_video = [self.process_frame(model, frame) for frame in video_frames]
        output_video = torch.from_numpy(np.array(output_video)) / 255
        return (output_video,)