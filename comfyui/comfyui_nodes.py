import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .annotator.nodes import VideoToCanny, VideoToDepth, VideoToPose
from .camera_utils import CAMERA, combine_camera_motion, get_camera_motion
from .cogvideox_fun.nodes import (CogVideoXFunInpaintSampler,
                                  CogVideoXFunT2VSampler,
                                  CogVideoXFunV2VSampler, LoadCogVideoXFunLora,
                                  LoadCogVideoXFunModel)
from .comfyui_utils import script_directory
from .qwenimage.nodes import (CombineQwenImagePipeline, LoadQwenImageLora,
                              LoadQwenImageModel, LoadQwenImageProcessor, QwenImageEditSampler,
                              LoadQwenImageTextEncoderModel,
                              LoadQwenImageTransformerModel,
                              LoadQwenImageVAEModel, QwenImageT2VSampler)
from .wan2_1.nodes import (CombineWanPipeline, LoadWanClipEncoderModel,
                           LoadWanLora, LoadWanModel, LoadWanTextEncoderModel,
                           LoadWanTransformerModel, LoadWanVAEModel,
                           WanI2VSampler, WanT2VSampler)
from .wan2_1_fun.nodes import (LoadWanFunLora, LoadWanFunModel,
                               WanFunInpaintSampler, WanFunT2VSampler,
                               WanFunV2VSampler)
from .wan2_2.nodes import (CombineWan2_2Pipeline, LoadWan2_2Lora,
                           LoadWan2_2Model, LoadWan2_2TransformerModel,
                           Wan2_2I2VSampler, Wan2_2T2VSampler)
from .wan2_2_fun.nodes import (LoadWan2_2FunLora, LoadWan2_2FunModel,
                               Wan2_2FunInpaintSampler, Wan2_2FunT2VSampler,
                               Wan2_2FunV2VSampler)
from .wan2_2_vace_fun.nodes import (CombineWan2_2VaceFunPipeline,
                                    LoadVaceWanTransformer3DModel,
                                    LoadWan2_2VaceFunModel,
                                    Wan2_2VaceFunSampler)


class FunTextBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "",}),
            },
        }
    
    RETURN_TYPES = ("STRING_PROMPT",)
    RETURN_NAMES =("prompt",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, prompt):
        return (prompt, )

class FunRiflex:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "riflex_k": ("INT", {"default": 6, "min": 0, "max": 10086}),
            },
        }
    
    RETURN_TYPES = ("RIFLEXT_ARGS",)
    RETURN_NAMES = ("riflex_k",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, riflex_k):
        return (riflex_k, )

class FunCompile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 10086}),
                "funmodels": ("FunModels",)
            }
        }
    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "compile"
    CATEGORY = "CogVideoXFUNWrapper"

    def compile(self, cache_size_limit, funmodels):
        torch._dynamo.config.cache_size_limit = cache_size_limit

        if funmodels["pipeline"].transformer.device == torch.device(type="meta"):
            if hasattr(funmodels["pipeline"].transformer, "blocks"):
                for i, block in enumerate(funmodels["pipeline"].transformer.blocks):
                    if hasattr(block, "_orig_mod"):
                        block = block._orig_mod

                if hasattr(funmodels["pipeline"], "transformer_2") and funmodels["pipeline"].transformer_2 is not None:
                    for i, block in enumerate(funmodels["pipeline"].transformer_2.blocks):
                        if hasattr(block, "_orig_mod"):
                            block = block._orig_mod

            elif hasattr(funmodels["pipeline"].transformer, "transformer_blocks"):
                for i, block in enumerate(funmodels["pipeline"].transformer.transformer_blocks):
                    if hasattr(block, "_orig_mod"):
                        block = block._orig_mod
            
                if hasattr(funmodels["pipeline"], "transformer_2") and funmodels["pipeline"].transformer_2 is not None:
                    for i, block in enumerate(funmodels["pipeline"].transformer_2.transformer_blocks):
                        if hasattr(block, "_orig_mod"):
                            block = block._orig_mod
                
            print("Sequential cpu offload can not work with compile. Continue")
            return (funmodels,)

        if hasattr(funmodels["pipeline"].transformer, "blocks"):
            for i, block in enumerate(funmodels["pipeline"].transformer.blocks):
                if hasattr(block, "_orig_mod"):
                    block = block._orig_mod
                funmodels["pipeline"].transformer.blocks[i] = torch.compile(block)
        
            if hasattr(funmodels["pipeline"], "transformer_2") and funmodels["pipeline"].transformer_2 is not None:
                for i, block in enumerate(funmodels["pipeline"].transformer_2.blocks):
                    if hasattr(block, "_orig_mod"):
                        block = block._orig_mod
                    funmodels["pipeline"].transformer_2.blocks[i] = torch.compile(block)
            
        elif hasattr(funmodels["pipeline"].transformer, "transformer_blocks"):
            for i, block in enumerate(funmodels["pipeline"].transformer.transformer_blocks):
                if hasattr(block, "_orig_mod"):
                    block = block._orig_mod
                funmodels["pipeline"].transformer.transformer_blocks[i] = torch.compile(block)
        
            if hasattr(funmodels["pipeline"], "transformer_2") and funmodels["pipeline"].transformer_2 is not None:
                for i, block in enumerate(funmodels["pipeline"].transformer_2.transformer_blocks):
                    if hasattr(block, "_orig_mod"):
                        block = block._orig_mod
                    funmodels["pipeline"].transformer_2.transformer_blocks[i] = torch.compile(block)
        
        else:
            funmodels["pipeline"].transformer.forward = torch.compile(funmodels["pipeline"].transformer.forward)

            if hasattr(funmodels["pipeline"], "transformer_2") and funmodels["pipeline"].transformer_2 is not None:
                funmodels["pipeline"].transformer_2.forward = torch.compile(funmodels["pipeline"].transformer_2.forward)

        print("Add Compile")
        return (funmodels,)
    
class FunAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "attention_type": (
                    ["flash", "sage", "torch"],
                    {"default": "flash"},
                ),
                "funmodels": ("FunModels",)
            }
        }
    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "funattention"
    CATEGORY = "CogVideoXFUNWrapper"

    def funattention(self, attention_type, funmodels):
        os.environ['VIDEOX_ATTENTION_TYPE'] = {
            "flash": "FLASH_ATTENTION",
            "sage": "SAGE_ATTENTION",
            "torch": "TORCH_SCALED_DOT"
        }[attention_type]
        return (funmodels,)

class LoadConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": (
                [
                    "wan2.1/wan_civitai.yaml",
                    "wan2.2/wan_civitai_t2v.yaml",
                    "wan2.2/wan_civitai_i2v.yaml",
                    "wan2.2/wan_civitai_5b.yaml",
                ],
                {
                    "default": "wan2.2/wan_civitai_i2v.yaml",
                }
            ),
        }
    
    RETURN_TYPES = ("FunConfig",)
    RETURN_NAMES = ("config",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, config):
        # Load config
        config_path = f"{script_directory}/config/{config}"
        config = OmegaConf.load(config_path)
        return (config, )

def gen_gaussian_heatmap(imgSize=200):
    circle_img = np.zeros((imgSize, imgSize,), np.float32) 
    circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2 - 1, 1, -1)

    isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)

    # 生成高斯图
    for i in range(imgSize):
        for j in range(imgSize):
            isotropicGrayscaleImage[i, j] = 1 / (2 * np.pi * (40 ** 2)) * np.exp(
                -1 / 2 * ((i - imgSize / 2) ** 2 / (40 ** 2) + (j - imgSize / 2) ** 2 / (40 ** 2)))

    isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage) * 255).astype(np.uint8)
    return isotropicGrayscaleImage

class CreateTrajectoryBasedOnKJNodes:
    # Modified from https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/curve_nodes.py
    # Modify to meet the trajectory control requirements of EasyAnimate.
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "createtrajectory"
    CATEGORY = "CogVideoXFUNWrapper"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coordinates": ("STRING", {"forceInput": True}),
                "masks": ("MASK", {"forceInput": True}),
            },
    } 

    def createtrajectory(self, coordinates, masks):
        # Define the number of images in the batch
        if len(coordinates) < 10:
            coords_list = []
            for coords in coordinates:
                coords = json.loads(coords.replace("'", '"'))
                coords_list.append(coords)
        else:
            coords = json.loads(coordinates.replace("'", '"'))
            coords_list = [coords]

        _, frame_height, frame_width = masks.size()
        heatmap = gen_gaussian_heatmap()

        circle_size = int(50 * ((frame_height * frame_width) / (1280 * 720)) ** (1/2))

        images_list = []
        for coords in coords_list:
            _images_list = []
            for i in range(len(coords)):
                _image = np.zeros((frame_height, frame_width, 3))
                center_coordinate = [coords[i][key] for key in coords[i]]

                y1 = max(center_coordinate[1] - circle_size, 0)
                y2 = min(center_coordinate[1] + circle_size, np.shape(_image)[0] - 1)
                x1 = max(center_coordinate[0] - circle_size, 0)
                x2 = min(center_coordinate[0] + circle_size, np.shape(_image)[1] - 1)
                
                if x2 - x1 > 3 and y2 - y1 > 3:
                    need_map = cv2.resize(heatmap, (x2 - x1, y2 - y1))[:, :, None]
                    _image[y1:y2, x1:x2] = np.maximum(need_map.copy(), _image[y1:y2, x1:x2])
                
                _image = np.expand_dims(_image, 0) / 255
                _images_list.append(_image)
            images_list.append(np.concatenate(_images_list, axis=0))
            
        out_images = torch.from_numpy(np.max(np.array(images_list), axis=0))
        return (out_images, )

class ImageMaximumNode:
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "imagemaximum"
    CATEGORY = "CogVideoXFUNWrapper"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_1": ("IMAGE",),
                "video_2": ("IMAGE",),
            },
    } 

    def imagemaximum(self, video_1, video_2):
        length_1, h_1, w_1, c_1 = video_1.size()
        length_2, h_2, w_2, c_2 = video_2.size()
        
        if h_1 != h_2 or w_1 != w_2:
            video_1, video_2 = video_1.permute([0, 3, 1, 2]), video_2.permute([0, 3, 1, 2])
            video_2 = F.interpolate(video_2, video_1.size()[-2:])
            video_1, video_2 = video_1.permute([0, 2, 3, 1]), video_2.permute([0, 2, 3, 1])

        if length_1 > length_2:
            outputs = torch.maximum(video_1[:length_2], video_2)
        else:
            outputs = torch.maximum(video_1, video_2[:length_1])
        return (outputs, )

class ImageCollectNode:
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "imagecollect"
    CATEGORY = "CogVideoXFUNWrapper"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",)
            },
            "optional": {
                "image_2": ("IMAGE",),
            }
    } 

    def imagecollect(self, image_1, image_2):
        image_out = [_image_1 for _image_1 in image_1] + [_image_2 for _image_2 in image_2]
        return (image_out, )

class CameraBasicFromChaoJie:
    # Copied from https://github.com/chaojie/ComfyUI-CameraCtrl-Wrapper/blob/main/nodes.py
    # Since ComfyUI-CameraCtrl-Wrapper requires a specific version of diffusers, which is not suitable for us. 
    # The code has been copied into the current repository.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "speed":("FLOAT",{"default":1.0}),
                "video_length":("INT",{"default":16}),
            },
        }

    RETURN_TYPES = ("CameraPose",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,camera_pose,speed,video_length):
        camera_dict = {
                "motion":[camera_pose],
                "mode": "Basic Camera Poses",  # "First A then B", "Both A and B", "Custom"
                "speed": speed,
                "complex": None
                } 
        motion_list = camera_dict['motion']
        mode = camera_dict['mode']
        speed = camera_dict['speed'] 
        angle = np.array(CAMERA[motion_list[0]]["angle"])
        T = np.array(CAMERA[motion_list[0]]["T"])
        RT = get_camera_motion(angle, T, speed, video_length)
        return (RT,)

class CameraCombineFromChaoJie:
    # Copied from https://github.com/chaojie/ComfyUI-CameraCtrl-Wrapper/blob/main/nodes.py
    # Since ComfyUI-CameraCtrl-Wrapper requires a specific version of diffusers, which is not suitable for us. 
    # The code has been copied into the current repository.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose1":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "camera_pose2":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "camera_pose3":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "camera_pose4":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "speed":("FLOAT",{"default":1.0}),
                "video_length":("INT",{"default":16}),
            },
        }

    RETURN_TYPES = ("CameraPose",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,camera_pose1,camera_pose2,camera_pose3,camera_pose4,speed,video_length):
        angle = np.array(CAMERA[camera_pose1]["angle"]) + np.array(CAMERA[camera_pose2]["angle"]) + np.array(CAMERA[camera_pose3]["angle"]) + np.array(CAMERA[camera_pose4]["angle"])
        T = np.array(CAMERA[camera_pose1]["T"]) + np.array(CAMERA[camera_pose2]["T"]) + np.array(CAMERA[camera_pose3]["T"]) + np.array(CAMERA[camera_pose4]["T"])
        RT = get_camera_motion(angle, T, speed, video_length)
        return (RT,)

class CameraJoinFromChaoJie:
    # Copied from https://github.com/chaojie/ComfyUI-CameraCtrl-Wrapper/blob/main/nodes.py
    # Since ComfyUI-CameraCtrl-Wrapper requires a specific version of diffusers, which is not suitable for us. 
    # The code has been copied into the current repository.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose1":("CameraPose",),
                "camera_pose2":("CameraPose",),
            },
        }

    RETURN_TYPES = ("CameraPose",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,camera_pose1,camera_pose2):
        RT = combine_camera_motion(camera_pose1, camera_pose2)
        return (RT,)

class CameraTrajectoryFromChaoJie:
    # Copied from https://github.com/chaojie/ComfyUI-CameraCtrl-Wrapper/blob/main/nodes.py
    # Since ComfyUI-CameraCtrl-Wrapper requires a specific version of diffusers, which is not suitable for us. 
    # The code has been copied into the current repository.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose":("CameraPose",),
                "fx":("FLOAT",{"default":0.474812461, "min": 0, "max": 1, "step": 0.000000001}),
                "fy":("FLOAT",{"default":0.844111024, "min": 0, "max": 1, "step": 0.000000001}),
                "cx":("FLOAT",{"default":0.5, "min": 0, "max": 1, "step": 0.01}),
                "cy":("FLOAT",{"default":0.5, "min": 0, "max": 1, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING","INT",)
    RETURN_NAMES = ("camera_trajectory","video_length",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,camera_pose,fx,fy,cx,cy):
        #print(camera_pose)
        camera_pose_list=camera_pose.tolist()
        trajs=[]
        for cp in camera_pose_list:
            traj=[fx,fy,cx,cy,0,0]
            traj.extend(cp[0])
            traj.extend(cp[1])
            traj.extend(cp[2])
            trajs.append(traj)
        return (json.dumps(trajs),len(trajs),)

NODE_CLASS_MAPPINGS = {
    "FunTextBox": FunTextBox,
    "FunRiflex": FunRiflex,
    "FunCompile": FunCompile,
    "FunAttention": FunAttention,

    "LoadCogVideoXFunModel": LoadCogVideoXFunModel,
    "LoadCogVideoXFunLora": LoadCogVideoXFunLora,
    "CogVideoXFunT2VSampler": CogVideoXFunT2VSampler,
    "CogVideoXFunInpaintSampler": CogVideoXFunInpaintSampler,
    "CogVideoXFunV2VSampler": CogVideoXFunV2VSampler,
    
    "LoadQwenImageLora": LoadQwenImageLora,
    "LoadQwenImageTextEncoderModel": LoadQwenImageTextEncoderModel,
    "LoadQwenImageTransformerModel": LoadQwenImageTransformerModel,
    "LoadQwenImageVAEModel": LoadQwenImageVAEModel, 
    "LoadQwenImageProcessor": LoadQwenImageProcessor,
    "CombineQwenImagePipeline": CombineQwenImagePipeline, 

    "LoadQwenImageModel": LoadQwenImageModel,
    "QwenImageT2VSampler": QwenImageT2VSampler,
    "QwenImageEditSampler": QwenImageEditSampler,
                                
    "LoadWanClipEncoderModel": LoadWanClipEncoderModel,
    "LoadWanTextEncoderModel": LoadWanTextEncoderModel,
    "LoadWanTransformerModel": LoadWanTransformerModel,
    "LoadWanVAEModel": LoadWanVAEModel,
    "CombineWanPipeline": CombineWanPipeline,
    "LoadWan2_2TransformerModel": LoadWan2_2TransformerModel, 
    "CombineWan2_2Pipeline": CombineWan2_2Pipeline,

    "LoadWanModel": LoadWanModel,
    "LoadWanLora": LoadWanLora,
    "WanT2VSampler": WanT2VSampler,
    "WanI2VSampler": WanI2VSampler,

    "LoadWanFunModel": LoadWanFunModel,
    "LoadWanFunLora": LoadWanFunLora,
    "WanFunT2VSampler": WanFunT2VSampler,
    "WanFunInpaintSampler": WanFunInpaintSampler,
    "WanFunV2VSampler": WanFunV2VSampler,

    "LoadWan2_2Model": LoadWan2_2Model,
    "LoadWan2_2Lora": LoadWan2_2Lora,
    "Wan2_2T2VSampler": Wan2_2T2VSampler,
    "Wan2_2I2VSampler": Wan2_2I2VSampler,

    "LoadWan2_2FunModel": LoadWan2_2FunModel,
    "LoadWan2_2FunLora": LoadWan2_2FunLora,
    "Wan2_2FunT2VSampler": Wan2_2FunT2VSampler,
    "Wan2_2FunInpaintSampler": Wan2_2FunInpaintSampler,
    "Wan2_2FunV2VSampler": Wan2_2FunV2VSampler,

    "LoadVaceWanTransformer3DModel": LoadVaceWanTransformer3DModel, 
    "CombineWan2_2VaceFunPipeline": CombineWan2_2VaceFunPipeline,

    "LoadWan2_2VaceFunModel": LoadWan2_2VaceFunModel,
    "Wan2_2VaceFunSampler": Wan2_2VaceFunSampler,

    "VideoToCanny": VideoToCanny,
    "VideoToDepth": VideoToDepth,
    "VideoToOpenpose": VideoToPose,

    "CreateTrajectoryBasedOnKJNodes": CreateTrajectoryBasedOnKJNodes,
    "CameraBasicFromChaoJie": CameraBasicFromChaoJie,
    "CameraTrajectoryFromChaoJie": CameraTrajectoryFromChaoJie,
    "CameraJoinFromChaoJie": CameraJoinFromChaoJie,
    "CameraCombineFromChaoJie": CameraCombineFromChaoJie,
    "ImageMaximumNode": ImageMaximumNode,
    "ImageCollectNode": ImageCollectNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FunTextBox": "FunTextBox",
    "FunRiflex": "FunRiflex",
    "FunCompile": "FunCompile",
    "FunAttention": "FunAttention",

    "LoadCogVideoXFunModel": "Load CogVideoX-Fun Model",
    "LoadCogVideoXFunLora": "Load CogVideoX-Fun Lora",
    "CogVideoXFunInpaintSampler": "CogVideoX-Fun Sampler for Image to Video",
    "CogVideoXFunT2VSampler": "CogVideoX-Fun Sampler for Text to Video",
    "CogVideoXFunV2VSampler": "CogVideoX-Fun Sampler for Video to Video",

    "LoadQwenImageLora": "Load QwenImage Lora",
    "LoadQwenImageTextEncoderModel": "Load QwenImage TextEncoder Model",
    "LoadQwenImageTransformerModel": "Load QwenImage Transformer Model",
    "LoadQwenImageVAEModel": "Load QwenImage VAE Model", 
    "LoadQwenImageProcessor": "Load QwenImage Processor",
    "CombineQwenImagePipeline": "Combine QwenImage Pipeline", 

    "LoadQwenImageModel": "Load QwenImage Model",
    "QwenImageT2VSampler": "QwenImage T2V Sampler",
    "QwenImageEditSampler": "QwenImage Edit Sampler",

    "LoadWanClipEncoderModel": "Load Wan ClipEncoder Model",
    "LoadWanTextEncoderModel": "Load Wan TextEncoder Model",
    "LoadWanTransformerModel": "Load Wan Transformer Model",
    "LoadWanVAEModel": "Load Wan VAE Model",
    "CombineWanPipeline": "Combine Wan Pipeline", 
    "LoadWan2_2TransformerModel": "Load Wan2_2 Transformer Model", 
    "CombineWan2_2Pipeline": "Combine Wan2_2 Pipeline",
    "LoadVaceWanTransformer3DModel": "Load Vace Wan Transformer 3DModel", 
    "CombineWan2_2VaceFunPipeline": "Combine Wan2_2 Vace Fun Pipeline",

    "LoadWanModel": "Load Wan Model",
    "LoadWanLora": "Load Wan Lora",
    "WanT2VSampler": "Wan Sampler for Text to Video",
    "WanI2VSampler": "Wan Sampler for Image to Video",

    "LoadWanFunModel": "Load Wan Fun Model",
    "LoadWanFunLora": "Load Wan Fun Lora",
    "WanFunT2VSampler": "Wan Fun Sampler for Text to Video",
    "WanFunInpaintSampler": "Wan Fun Sampler for Image to Video",
    "WanFunV2VSampler": "Wan Fun Sampler for Video to Video",

    "LoadWan2_2Model": "Load Wan 2.2 Model",
    "LoadWan2_2Lora": "Load Wan 2.2 Lora",
    "Wan2_2T2VSampler": "Wan 2.2 Sampler for Text to Video",
    "Wan2_2I2VSampler": "Wan 2.2 Sampler for Image to Video",

    "LoadWan2_2FunModel": "Load Wan 2.2 Fun Model",
    "LoadWan2_2FunLora": "Load Wan 2.2 Fun Lora",
    "Wan2_2FunT2VSampler": "Wan 2.2 Fun Sampler for Text to Video",
    "Wan2_2FunInpaintSampler": "Wan 2.2 Fun Sampler for Image to Video",
    "Wan2_2FunV2VSampler": "Wan 2.2 Fun Sampler for Video to Video",

    "LoadWan2_2VaceFunModel": "Load Wan2_2 Vace Fun Model",
    "Wan2_2VaceFunSampler": "Wan2_2 Vace Fun Sampler",
    
    "VideoToCanny": "Video To Canny",
    "VideoToDepth": "Video To Depth",
    "VideoToOpenpose": "Video To Pose",

    "CreateTrajectoryBasedOnKJNodes": "Create Trajectory Based On KJNodes",
    "CameraBasicFromChaoJie": "Camera Basic From ChaoJie",
    "CameraTrajectoryFromChaoJie": "Camera Trajectory From ChaoJie",
    "CameraJoinFromChaoJie": "Camera Join From ChaoJie",
    "CameraCombineFromChaoJie": "Camera Combine From ChaoJie",
    "ImageMaximumNode": "Image Maximum Node",
    "ImageCollectNode": "Image Collect Node",
}