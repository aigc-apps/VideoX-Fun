import base64
import gc
import hashlib
import io
import os
import tempfile
from io import BytesIO

import gradio as gr
import requests
import torch
from fastapi import FastAPI
from PIL import Image


# Function to encode a file to Base64
def encode_file_to_base64(file_path):
    with open(file_path, "rb") as file:
        # Encode the data to Base64
        file_base64 = base64.b64encode(file.read())
        return file_base64

def update_diffusion_transformer_api(_: gr.Blocks, app: FastAPI, controller):
    @app.post("/videox_fun/update_diffusion_transformer")
    def _update_diffusion_transformer_api(
        datas: dict,
    ):
        diffusion_transformer_path = datas.get('diffusion_transformer_path', 'none')

        try:
            controller.update_diffusion_transformer(
                diffusion_transformer_path
            )
            comment = "Success"
        except Exception as e:
            torch.cuda.empty_cache()
            comment = f"Error. error information is {str(e)}"

        return {"message": comment}

def download_from_url(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # 检查请求是否成功
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading from {url}: {e}")
        return None

def save_base64_video(base64_string):
    video_data = base64.b64decode(base64_string)

    md5_hash = hashlib.md5(video_data).hexdigest()
    filename = f"{md5_hash}.mp4"  
    
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)

    with open(file_path, 'wb') as video_file:
        video_file.write(video_data)

    return file_path

def save_base64_image(base64_string):
    video_data = base64.b64decode(base64_string)

    md5_hash = hashlib.md5(video_data).hexdigest()
    filename = f"{md5_hash}.jpg"  
    
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)

    with open(file_path, 'wb') as video_file:
        video_file.write(video_data)

    return file_path

def save_url_video(url):
    video_data = download_from_url(url)
    if video_data:
        return save_base64_video(base64.b64encode(video_data))
    return None

def save_url_image(url):
    image_data = download_from_url(url)
    if image_data:
        return save_base64_image(base64.b64encode(image_data))
    return None

def infer_forward_api(_: gr.Blocks, app: FastAPI, controller):
    @app.post("/videox_fun/infer_forward")
    def _infer_forward_api(
        datas: dict,
    ):
        base_model_path = datas.get('base_model_path', 'none')
        lora_model_path = datas.get('lora_model_path', 'none')
        lora_alpha_slider = datas.get('lora_alpha_slider', 0.55)
        prompt_textbox = datas.get('prompt_textbox', None)
        negative_prompt_textbox = datas.get('negative_prompt_textbox', 'The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. ')
        sampler_dropdown = datas.get('sampler_dropdown', 'Euler')
        sample_step_slider = datas.get('sample_step_slider', 30)
        resize_method = datas.get('resize_method', "Generate by")
        width_slider = datas.get('width_slider', 672)
        height_slider = datas.get('height_slider', 384)
        base_resolution = datas.get('base_resolution', 512)
        is_image = datas.get('is_image', False)
        generation_method = datas.get('generation_method', False)
        length_slider = datas.get('length_slider', 49)
        overlap_video_length = datas.get('overlap_video_length', 4)
        partial_video_length = datas.get('partial_video_length', 72)
        cfg_scale_slider = datas.get('cfg_scale_slider', 6)
        start_image = datas.get('start_image', None)
        end_image = datas.get('end_image', None)
        validation_video = datas.get('validation_video', None)
        validation_video_mask = datas.get('validation_video_mask', None)
        control_video = datas.get('control_video', None)
        denoise_strength = datas.get('denoise_strength', 0.70)
        seed_textbox = datas.get("seed_textbox", 43)
        
        ref_image = datas.get('ref_image', None)
        enable_teacache = datas.get('enable_teacache', True)
        teacache_threshold = datas.get('teacache_threshold', 0.10)
        num_skip_start_steps = datas.get('num_skip_start_steps', 1)
        teacache_offload = datas.get('teacache_offload', False)
        cfg_skip_ratio = datas.get('cfg_skip_ratio', 0)
        enable_riflex = datas.get('enable_riflex', False)
        riflex_k = datas.get('riflex_k', 6)

        generation_method = "Image Generation" if is_image else generation_method

        if start_image is not None:
            if start_image.startswith('http'):
                start_image = save_url_image(start_image)
                start_image = [Image.open(start_image).convert("RGB")]
            else:
                start_image = base64.b64decode(start_image)
                start_image = [Image.open(BytesIO(start_image)).convert("RGB")]

        if end_image is not None:
            if end_image.startswith('http'):
                end_image = save_url_image(end_image)
                end_image = [Image.open(end_image).convert("RGB")]
            else:
                end_image = base64.b64decode(end_image)
                end_image = [Image.open(BytesIO(end_image)).convert("RGB")]

        if validation_video is not None:
            if validation_video.startswith('http'):
                validation_video = save_url_video(validation_video)
            else:
                validation_video = save_base64_video(validation_video)

        if validation_video_mask is not None:
            if validation_video_mask.startswith('http'):
                validation_video_mask = save_url_image(validation_video_mask)
            else:
                validation_video_mask = save_base64_image(validation_video_mask)

        if control_video is not None:
            if control_video.startswith('http'):
                control_video = save_url_video(control_video)
            else:
                control_video = save_base64_video(control_video)

        if ref_image is not None:
            if ref_image.startswith('http'):
                ref_image = save_url_image(ref_image)
                ref_image = [Image.open(ref_image).convert("RGB")]
            else:
                ref_image = base64.b64decode(ref_image)
                ref_image = [Image.open(BytesIO(ref_image)).convert("RGB")]
        
        try:
            save_sample_path, comment = controller.generate(
                "",
                base_model_path,
                lora_model_path, 
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
                ref_image = ref_image,
                enable_teacache = enable_teacache, 
                teacache_threshold = teacache_threshold, 
                num_skip_start_steps = num_skip_start_steps, 
                teacache_offload = teacache_offload,
                cfg_skip_ratio = cfg_skip_ratio, 
                enable_riflex = enable_riflex, 
                riflex_k = riflex_k, 
                is_api = True,
            )
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            save_sample_path = ""
            comment = f"Error. error information is {str(e)}"
            return {"message": comment, "save_sample_path": None, "base64_encoding": None}
        
        if save_sample_path != "":
            return {"message": comment, "save_sample_path": save_sample_path, "base64_encoding": encode_file_to_base64(save_sample_path)}
        else:
            return {"message": comment, "save_sample_path": save_sample_path, "base64_encoding": None}