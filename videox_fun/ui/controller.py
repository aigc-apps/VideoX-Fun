"""Modified from https://github.com/guoyww/AnimateDiff/blob/main/app.py
"""
import base64
import gc
import json
import os
import random
from datetime import datetime
from glob import glob

import cv2
import gradio as gr
import numpy as np
import pkg_resources
import requests
import torch
from diffusers import (CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       FlowMatchEulerDiscreteScheduler, PNDMScheduler)
from omegaconf import OmegaConf
from PIL import Image
from safetensors import safe_open

from ..data.bucket_sampler import ASPECT_RATIO_512, get_closest_ratio
from ..utils.utils import save_videos_grid
from ..utils.fm_solvers import FlowDPMSolverMultistepScheduler
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..dist import set_multi_gpus_devices

gradio_version = pkg_resources.get_distribution("gradio").version
gradio_version_is_above_4 = True if int(gradio_version.split('.')[0]) >= 4 else False

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

ddpm_scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
    "DDIM_Origin": DDIMScheduler,
    "DDIM_Cog": CogVideoXDDIMScheduler,
}
flow_scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}
all_cheduler_dict = {**ddpm_scheduler_dict, **flow_scheduler_dict}

class Fun_Controller:
    def __init__(
        self, GPU_memory_mode, scheduler_dict, model_name=None, model_type="Inpaint", 
        config_path=None, ulysses_degree=1, ring_degree=1, 
        fsdp_dit=False, fsdp_text_encoder=False, compile_dit=False, 
        weight_dtype=None, savedir_sample=None,
    ):
        # config dirs
        self.basedir                    = os.getcwd()
        self.config_dir                 = os.path.join(self.basedir, "config")
        self.diffusion_transformer_dir  = os.path.join(self.basedir, "models", "Diffusion_Transformer")
        self.motion_module_dir          = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir     = os.path.join(self.basedir, "models", "Personalized_Model")
        if savedir_sample is None:
            self.savedir_sample         = os.path.join(self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        else:
            self.savedir_sample         = savedir_sample
        os.makedirs(self.savedir_sample, exist_ok=True)

        self.GPU_memory_mode                = GPU_memory_mode
        self.model_name                     = model_name
        self.diffusion_transformer_dropdown = model_name
        self.scheduler_dict             = scheduler_dict
        self.model_type                 = model_type
        if config_path is not None:
            self.config = OmegaConf.load(config_path)
        self.ulysses_degree             = ulysses_degree
        self.ring_degree                = ring_degree
        self.fsdp_dit                   = fsdp_dit       
        self.fsdp_text_encoder          = fsdp_text_encoder
        self.compile_dit                = compile_dit
        self.weight_dtype               = weight_dtype
        self.device                     = set_multi_gpus_devices(self.ulysses_degree, self.ring_degree)

        self.diffusion_transformer_list = []
        self.motion_module_list         = []
        self.personalized_model_list    = []

        # config models
        self.tokenizer             = None
        self.text_encoder          = None
        self.vae                   = None
        self.transformer           = None
        self.pipeline              = None
        self.base_model_path       = "none"
        self.lora_model_path       = "none"
        
        self.refresh_diffusion_transformer()
        self.refresh_personalized_model()
        if model_name != None:
            self.update_diffusion_transformer(model_name)

    def refresh_diffusion_transformer(self):
        self.diffusion_transformer_list = sorted(glob(os.path.join(self.diffusion_transformer_dir, "*/")))

    def refresh_personalized_model(self):
        personalized_model_list = sorted(glob(os.path.join(self.personalized_model_dir, "*.safetensors")))
        self.personalized_model_list = [os.path.basename(p) for p in personalized_model_list]

    def update_model_type(self, model_type):
        self.model_type = model_type

    def update_diffusion_transformer(self, diffusion_transformer_dropdown):
        pass

    def update_base_model(self, base_model_dropdown):
        self.base_model_path = base_model_dropdown
        print(f"Update base model: {base_model_dropdown}")
        if base_model_dropdown == "none":
            return gr.update()
        if self.transformer is None:
            gr.Info(f"Please select a pretrained model path.")
            print(f"Please select a pretrained model path.")
            return gr.update(value=None)
        else:
            base_model_dropdown = os.path.join(self.personalized_model_dir, base_model_dropdown)
            base_model_state_dict = {}
            with safe_open(base_model_dropdown, framework="pt", device="cpu") as f:
                for key in f.keys():
                    base_model_state_dict[key] = f.get_tensor(key)
            self.transformer.load_state_dict(base_model_state_dict, strict=False)
            print("Update base model done")
            return gr.update()

    def update_lora_model(self, lora_model_dropdown):
        print(f"Update lora model: {lora_model_dropdown}")
        if lora_model_dropdown == "none":
            self.lora_model_path = "none"
            return gr.update()
        lora_model_dropdown = os.path.join(self.personalized_model_dir, lora_model_dropdown)
        self.lora_model_path = lora_model_dropdown
        return gr.update()

    def clear_cache(self,):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def auto_model_clear_cache(self, model):
        origin_device = model.device
        model = model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        model = model.to(origin_device)
    
    def input_check(self,
        resize_method,
        generation_method, 
        start_image, 
        end_image, 
        validation_video,
        control_video,
        is_api = False,
    ):
        if self.transformer is None:
            if is_api:
                return "", f"Please select a pretrained model path."
            else:
                raise gr.Error(f"Please select a pretrained model path.")
        
        if control_video is not None and self.model_type == "Inpaint":
            if is_api:
                return "", f"If specifying the control video, please set the model_type == \"Control\". "
            else:
                raise gr.Error(f"If specifying the control video, please set the model_type == \"Control\". ")

        if control_video is None and self.model_type == "Control":
            if is_api:
                return "", f"If set the model_type == \"Control\", please specifying the control video. "
            else:
                raise gr.Error(f"If set the model_type == \"Control\", please specifying the control video. ")

        if resize_method == "Resize according to Reference":
            if start_image is None and validation_video is None and control_video is None:
                if is_api:
                    return "", f"Please upload an image when using \"Resize according to Reference\"."
                else:
                    raise gr.Error(f"Please upload an image when using \"Resize according to Reference\".")

        if self.transformer.config.in_channels == self.vae.config.latent_channels and start_image is not None:
            if is_api:
                return "", f"Please select an image to video pretrained model while using image to video."
            else:
                raise gr.Error(f"Please select an image to video pretrained model while using image to video.")

        if self.transformer.config.in_channels == self.vae.config.latent_channels and generation_method == "Long Video Generation":
            if is_api:
                return "", f"Please select an image to video pretrained model while using long video generation."
            else:
                raise gr.Error(f"Please select an image to video pretrained model while using long video generation.")
        
        if start_image is None and end_image is not None:
            if is_api:
                return "", f"If specifying the ending image of the video, please specify a starting image of the video."
            else:
                raise gr.Error(f"If specifying the ending image of the video, please specify a starting image of the video.")
        return "", "OK"

    def get_height_width_from_reference(
        self,
        base_resolution,
        start_image,
        validation_video,
        control_video,
    ):
        aspect_ratio_sample_size    = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        if self.model_type == "Inpaint":
            if validation_video is not None:
                original_width, original_height = Image.fromarray(cv2.VideoCapture(validation_video).read()[1]).size
            else:
                original_width, original_height = start_image[0].size if type(start_image) is list else Image.open(start_image).size
        else:
            original_width, original_height = Image.fromarray(cv2.VideoCapture(control_video).read()[1]).size
        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height_slider, width_slider = [int(x / 16) * 16 for x in closest_size]
        return height_slider, width_slider

    def save_outputs(self, is_image, length_slider, sample, fps):
        def save_results():
            if not os.path.exists(self.savedir_sample):
                os.makedirs(self.savedir_sample, exist_ok=True)
            index = len([path for path in os.listdir(self.savedir_sample)]) + 1
            prefix = str(index).zfill(3)

            if is_image or length_slider == 1:
                save_sample_path = os.path.join(self.savedir_sample, prefix + f".png")

                image = sample[0, :, 0]
                image = image.transpose(0, 1).transpose(1, 2)
                image = (image * 255).numpy().astype(np.uint8)
                image = Image.fromarray(image)
                image.save(save_sample_path)

            else:
                save_sample_path = os.path.join(self.savedir_sample, prefix + f".mp4")
                save_videos_grid(sample, save_sample_path, fps=fps)
            return save_sample_path

        if self.ulysses_degree * self.ring_degree > 1:
            import torch.distributed as dist
            if dist.get_rank() == 0:
                save_sample_path = save_results()
            else:
                save_sample_path = None
        else:
            save_sample_path = save_results()
        return save_sample_path

    def generate(
        self,
        diffusion_transformer_dropdown,
        base_model_dropdown,
        lora_model_dropdown, 
        lora_alpha_slider,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        resize_method,
        width_slider, 
        height_slider, 
        base_resolution, 
        generation_method, 
        length_slider, 
        overlap_video_length, 
        partial_video_length, 
        cfg_scale_slider, 
        start_image, 
        end_image, 
        validation_video,
        validation_video_mask,
        control_video,
        denoise_strength,
        seed_textbox,
        enable_teacache = None, 
        teacache_threshold = None, 
        num_skip_start_steps = None, 
        teacache_offload = None, 
        cfg_skip_ratio = None,
        enable_riflex = None, 
        riflex_k = None, 
        is_api = False,
    ):
        pass

def post_to_host(
    diffusion_transformer_dropdown,
    base_model_dropdown, lora_model_dropdown, lora_alpha_slider,
    prompt_textbox, negative_prompt_textbox, 
    sampler_dropdown, sample_step_slider, resize_method, width_slider, height_slider,
    base_resolution, generation_method, length_slider, cfg_scale_slider, 
    start_image, end_image, validation_video, validation_video_mask, denoise_strength, seed_textbox,
    ref_image = None, enable_teacache = None, teacache_threshold = None, num_skip_start_steps = None, 
    teacache_offload = None, cfg_skip_ratio = None,enable_riflex = None, riflex_k = None, 
):
    if start_image is not None:
        with open(start_image, 'rb') as file:
            file_content = file.read()
            start_image_encoded_content = base64.b64encode(file_content)
            start_image = start_image_encoded_content.decode('utf-8')

    if end_image is not None:
        with open(end_image, 'rb') as file:
            file_content = file.read()
            end_image_encoded_content = base64.b64encode(file_content)
            end_image = end_image_encoded_content.decode('utf-8')

    if validation_video is not None:
        with open(validation_video, 'rb') as file:
            file_content = file.read()
            validation_video_encoded_content = base64.b64encode(file_content)
            validation_video = validation_video_encoded_content.decode('utf-8')

    if validation_video_mask is not None:
        with open(validation_video_mask, 'rb') as file:
            file_content = file.read()
            validation_video_mask_encoded_content = base64.b64encode(file_content)
            validation_video_mask = validation_video_mask_encoded_content.decode('utf-8')

    if ref_image is not None:
        with open(ref_image, 'rb') as file:
            file_content = file.read()
            ref_image_encoded_content = base64.b64encode(file_content)
            ref_image = ref_image_encoded_content.decode('utf-8')

    datas = {
        "base_model_path": base_model_dropdown,
        "lora_model_path": lora_model_dropdown, 
        "lora_alpha_slider": lora_alpha_slider, 
        "prompt_textbox": prompt_textbox, 
        "negative_prompt_textbox": negative_prompt_textbox, 
        "sampler_dropdown": sampler_dropdown, 
        "sample_step_slider": sample_step_slider, 
        "resize_method": resize_method,
        "width_slider": width_slider, 
        "height_slider": height_slider, 
        "base_resolution": base_resolution,
        "generation_method": generation_method,
        "length_slider": length_slider,
        "cfg_scale_slider": cfg_scale_slider,
        "start_image": start_image,
        "end_image": end_image,
        "validation_video": validation_video,
        "validation_video_mask": validation_video_mask,
        "denoise_strength": denoise_strength,
        "seed_textbox": seed_textbox,

        "ref_image": ref_image,
        "enable_teacache": enable_teacache,
        "teacache_threshold": teacache_threshold,
        "num_skip_start_steps": num_skip_start_steps,
        "teacache_offload": teacache_offload,
        "cfg_skip_ratio": cfg_skip_ratio,
        "enable_riflex": enable_riflex,
        "riflex_k": riflex_k,
    }

    session = requests.session()
    session.headers.update({"Authorization": os.environ.get("EAS_TOKEN")})

    response = session.post(url=f'{os.environ.get("EAS_URL")}/videox_fun/infer_forward', json=datas, timeout=300)

    outputs = response.json()
    return outputs


class Fun_Controller_Client:
    def __init__(self, scheduler_dict, savedir_sample):
        self.basedir = os.getcwd()
        if savedir_sample is None:
            self.savedir_sample         = os.path.join(self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        else:
            self.savedir_sample         = savedir_sample
        os.makedirs(self.savedir_sample, exist_ok=True)
        
        self.scheduler_dict = scheduler_dict

    def generate(
        self,
        diffusion_transformer_dropdown,
        base_model_dropdown,
        lora_model_dropdown, 
        lora_alpha_slider,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        resize_method,
        width_slider, 
        height_slider, 
        base_resolution, 
        generation_method, 
        length_slider, 
        cfg_scale_slider, 
        start_image, 
        end_image, 
        validation_video, 
        validation_video_mask, 
        denoise_strength,
        seed_textbox,
        ref_image = None,
        enable_teacache = None, 
        teacache_threshold = None, 
        num_skip_start_steps = None, 
        teacache_offload = None, 
        cfg_skip_ratio = None,
        enable_riflex = None, 
        riflex_k = None, 
    ):
        is_image = True if generation_method == "Image Generation" else False

        outputs = post_to_host(
            diffusion_transformer_dropdown,
            base_model_dropdown, lora_model_dropdown, lora_alpha_slider,
            prompt_textbox, negative_prompt_textbox, 
            sampler_dropdown, sample_step_slider, resize_method, width_slider, height_slider,
            base_resolution, generation_method, length_slider, cfg_scale_slider, 
            start_image, end_image, validation_video, validation_video_mask, denoise_strength, 
            seed_textbox, ref_image = ref_image, enable_teacache = enable_teacache, teacache_threshold = teacache_threshold, 
            num_skip_start_steps = num_skip_start_steps, teacache_offload = teacache_offload, 
            cfg_skip_ratio = cfg_skip_ratio, enable_riflex = enable_riflex, riflex_k = riflex_k, 
        )
        print(outputs)
        try:
            base64_encoding = outputs["base64_encoding"]
        except:
            return gr.Image(visible=False, value=None), gr.Video(None, visible=True), outputs["message"]
            
        decoded_data = base64.b64decode(base64_encoding)

        if not os.path.exists(self.savedir_sample):
            os.makedirs(self.savedir_sample, exist_ok=True)
        index = len([path for path in os.listdir(self.savedir_sample)]) + 1
        prefix = str(index).zfill(3)
        
        if is_image or length_slider == 1:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".png")
            with open(save_sample_path, "wb") as file:
                file.write(decoded_data)
            if gradio_version_is_above_4:
                return gr.Image(value=save_sample_path, visible=True), gr.Video(value=None, visible=False), "Success"
            else:
                return gr.Image.update(value=save_sample_path, visible=True), gr.Video.update(value=None, visible=False), "Success"
        else:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".mp4")
            with open(save_sample_path, "wb") as file:
                file.write(decoded_data)
            if gradio_version_is_above_4:
                return gr.Image(visible=False, value=None), gr.Video(value=save_sample_path, visible=True), "Success"
            else:
                return gr.Image.update(visible=False, value=None), gr.Video.update(value=save_sample_path, visible=True), "Success"
