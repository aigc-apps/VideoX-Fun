import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .annotator.nodes import VideoToCanny, VideoToDepth, VideoToPose
from .camera_utils import CAMERA, combine_camera_motion, get_camera_motion
from .cogvideox_fun.nodes import (CogVideoXFunInpaintSampler,
                                  CogVideoXFunT2VSampler,
                                  CogVideoXFunV2VSampler, LoadCogVideoXFunLora,
                                  LoadCogVideoXFunModel)
from .wan2_1.nodes import (LoadWanLora, LoadWanModel, WanI2VSampler,
                           WanT2VSampler)
from .wan2_1_fun.nodes import (LoadWanFunLora, LoadWanFunModel,
                               WanFunInpaintSampler, WanFunT2VSampler,
                               WanFunV2VSampler)

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
        if hasattr(funmodels["pipeline"].transformer, "blocks"):
            for i in range(len(funmodels["pipeline"].transformer.blocks)):
                funmodels["pipeline"].transformer.blocks[i] = torch.compile(funmodels["pipeline"].transformer.blocks[i])
        
        elif hasattr(funmodels["pipeline"].transformer, "transformer_blocks"):
                for i in range(len(funmodels["pipeline"].transformer.transformer_blocks)):
                    funmodels["pipeline"].transformer.transformer_blocks[i] = torch.compile(funmodels["pipeline"].transformer.transformer_blocks[i])
        
        else:
            funmodels["pipeline"].transformer.forward = torch.compile(funmodels["pipeline"].transformer.forward)

        print("Add Compile")
        return (funmodels,)

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

    "CreateTrajectoryBasedOnKJNodes": CreateTrajectoryBasedOnKJNodes,
    "CameraBasicFromChaoJie": CameraBasicFromChaoJie,
    "CameraTrajectoryFromChaoJie": CameraTrajectoryFromChaoJie,
    "CameraJoinFromChaoJie": CameraJoinFromChaoJie,
    "CameraCombineFromChaoJie": CameraCombineFromChaoJie,
    "ImageMaximumNode": ImageMaximumNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "FunTextBox": "FunTextBox",
    "FunRiflex": "FunRiflex",
    "FunCompile": "FunCompile",

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

    "CreateTrajectoryBasedOnKJNodes": "Create Trajectory Based On KJNodes",
    "CameraBasicFromChaoJie": "Camera Basic From ChaoJie",
    "CameraTrajectoryFromChaoJie": "Camera Trajectory From ChaoJie",
    "CameraJoinFromChaoJie": "Camera Join From ChaoJie",
    "CameraCombineFromChaoJie": "Camera Combine From ChaoJie",
    "ImageMaximumNode": "Image Maximum Node",
}