# This file is modified from https://github.com/xdit-project/xDiT/blob/main/entrypoints/launch.py
import base64
import gc
import os
from io import BytesIO

import gradio as gr
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image

from .api import (encode_file_to_base64, save_base64_image, save_base64_video,
                  save_url_image, save_url_video)

try:
    import ray
except:
    print("Ray is not installed. If you want to use multi gpus api. Please install it by running 'pip install ray'.")
    ray =  None

if ray is not None:
    @ray.remote(num_gpus=1)
    class MultiNodesGenerator:
        def __init__(
            self, rank: int, world_size: int, Controller,
            GPU_memory_mode, scheduler_dict, model_name=None, model_type="Inpaint", 
            config_path=None, ulysses_degree=1, ring_degree=1,
            enable_teacache=None, teacache_threshold=None, 
            num_skip_start_steps=None, teacache_offload=None, weight_dtype=None, 
            savedir_sample=None,
        ):
            # Set PyTorch distributed environment variables
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            
            self.rank = rank
            self.controller = Controller(
                GPU_memory_mode, scheduler_dict, model_name=model_name, model_type=model_type, config_path=config_path, 
                ulysses_degree=ulysses_degree, ring_degree=ring_degree, enable_teacache=enable_teacache, teacache_threshold=teacache_threshold, num_skip_start_steps=num_skip_start_steps, 
                teacache_offload=teacache_offload, weight_dtype=weight_dtype, savedir_sample=savedir_sample,
            )

        def generate(self, datas):
            try:
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

                generation_method = "Image Generation" if is_image else generation_method

                if start_image is not None:
                    if start_image.startswith('http'):
                        start_image = save_url_image(start_image)
                        start_image = [Image.open(start_image)]
                    else:
                        start_image = base64.b64decode(start_image)
                        start_image = [Image.open(BytesIO(start_image))]

                if end_image is not None:
                    if end_image.startswith('http'):
                        end_image = save_url_image(end_image)
                        end_image = [Image.open(end_image)]
                    else:
                        end_image = base64.b64decode(end_image)
                        end_image = [Image.open(BytesIO(end_image))]
                        
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
                
                try:
                    save_sample_path, comment = self.controller.generate(
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
                        is_api = True,
                    )
                except Exception as e:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    save_sample_path = ""
                    comment = f"Error. error information is {str(e)}"
                    return {"message": comment}
                
                import torch.distributed as dist
                if dist.get_rank() == 0:
                    if save_sample_path != "":
                        return {"message": comment, "save_sample_path": save_sample_path, "base64_encoding": encode_file_to_base64(save_sample_path)}
                    else:
                        return {"message": comment, "save_sample_path": save_sample_path}
                return None

            except Exception as e:
                self.logger.error(f"Error generating image: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    class MultiNodesEngine:
        def __init__(
            self, 
            world_size, 
            Controller,
            GPU_memory_mode, 
            scheduler_dict, 
            model_name, 
            model_type, 
            config_path,
            ulysses_degree, 
            ring_degree, 
            enable_teacache, 
            teacache_threshold, 
            num_skip_start_steps, 
            teacache_offload, 
            weight_dtype,
            savedir_sample
        ):
            # Ensure Ray is initialized
            if not ray.is_initialized():
                ray.init()
            
            num_workers = world_size
            self.workers = [
                MultiNodesGenerator.remote(
                    rank, world_size, Controller, 
                    GPU_memory_mode, scheduler_dict, model_name=model_name, model_type=model_type, config_path=config_path, 
                    ulysses_degree=ulysses_degree, ring_degree=ring_degree, enable_teacache=enable_teacache, teacache_threshold=teacache_threshold, num_skip_start_steps=num_skip_start_steps, 
                    teacache_offload=teacache_offload, weight_dtype=weight_dtype, savedir_sample=savedir_sample,
                )
                for rank in range(num_workers)
            ]
            print("Update workers done")
            
        async def generate(self, data):
            results = ray.get([
                worker.generate.remote(data)
                for worker in self.workers
            ])

            return next(path for path in results if path is not None) 

    def multi_nodes_infer_forward_api(_: gr.Blocks, app: FastAPI, engine):

        @app.post("/videox_fun/infer_forward")
        async def _multi_nodes_infer_forward_api(
            datas: dict,
        ):
            try:
                result = await engine.generate(datas)
                return result
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise e
                raise HTTPException(status_code=500, detail=str(e))
else:
    MultiNodesEngine = None
    MultiNodesGenerator = None
    multi_nodes_infer_forward_api = None