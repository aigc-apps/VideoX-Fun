"""Modified from https://github.com/guoyww/AnimateDiff/blob/main/app.py
"""
import os
import random

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

from ..data.bucket_sampler import ASPECT_RATIO_512, get_closest_ratio
from ..models import (AutoencoderKLCogVideoX, CogVideoXTransformer3DModel,
                      T5EncoderModel, T5Tokenizer)
from ..pipeline import (CogVideoXFunControlPipeline,
                        CogVideoXFunInpaintPipeline, CogVideoXFunPipeline)
from ..utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                    convert_weight_dtype_wrapper)
from ..utils.lora_utils import merge_lora, unmerge_lora
from ..utils.utils import (filter_kwargs, get_image_to_video_latent, get_image_latent, timer,
                           get_video_to_video_latent, save_videos_grid)
from .controller import (Fun_Controller, Fun_Controller_Client,
                         all_cheduler_dict, css, ddpm_scheduler_dict,
                         flow_scheduler_dict, gradio_version,
                         gradio_version_is_above_4)
from .ui import (create_cfg_and_seedbox,
                 create_fake_finetune_models_checkpoints,
                 create_fake_height_width, create_fake_model_checkpoints,
                 create_fake_model_type, create_finetune_models_checkpoints,
                 create_generation_method,
                 create_generation_methods_and_video_length,
                 create_height_width, create_model_checkpoints,
                 create_model_type, create_prompts, create_samplers,
                 create_ui_outputs)
from ..dist import set_multi_gpus_devices, shard_model


class CogVideoXFunController(Fun_Controller):
    def update_diffusion_transformer(self, diffusion_transformer_dropdown):
        print(f"Update diffusion transformer: {diffusion_transformer_dropdown}")
        self.diffusion_transformer_dropdown = diffusion_transformer_dropdown
        if diffusion_transformer_dropdown == "none":
            return gr.update()
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            diffusion_transformer_dropdown, 
            subfolder="vae", 
        ).to(self.weight_dtype)

        # Get Transformer
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            diffusion_transformer_dropdown, 
            subfolder="transformer",
            low_cpu_mem_usage=True, 
        ).to(self.weight_dtype)
        
        # Get tokenizer and text_encoder
        tokenizer = T5Tokenizer.from_pretrained(
            diffusion_transformer_dropdown, subfolder="tokenizer"
        )
        text_encoder = T5EncoderModel.from_pretrained(
            diffusion_transformer_dropdown, subfolder="text_encoder", torch_dtype=self.weight_dtype
        )
    
        # Get pipeline
        if self.model_type == "Inpaint":
            if self.transformer.config.in_channels != self.vae.config.latent_channels:
                self.pipeline = CogVideoXFunInpaintPipeline(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    vae=self.vae, 
                    transformer=self.transformer,
                    scheduler=self.scheduler_dict[list(self.scheduler_dict.keys())[0]].from_pretrained(diffusion_transformer_dropdown, subfolder="scheduler"),
                )
            else:
                self.pipeline = CogVideoXFunPipeline(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    vae=self.vae, 
                    transformer=self.transformer,
                    scheduler=self.scheduler_dict[list(self.scheduler_dict.keys())[0]].from_pretrained(diffusion_transformer_dropdown, subfolder="scheduler"),
                )
        else:
            self.pipeline = CogVideoXFunControlPipeline(
                diffusion_transformer_dropdown,
                vae=self.vae, 
                transformer=self.transformer,
                scheduler=self.scheduler_dict[list(self.scheduler_dict.keys())[0]].from_pretrained(diffusion_transformer_dropdown, subfolder="scheduler"),
                torch_dtype=self.weight_dtype
            )

        if self.ulysses_degree > 1 or self.ring_degree > 1:
            from functools import partial
            self.transformer.enable_multi_gpus_inference()
            if self.fsdp_dit:
                shard_fn = partial(shard_model, device_id=self.device, param_dtype=self.weight_dtype)
                self.pipeline.transformer = shard_fn(self.pipeline.transformer)
                print("Add FSDP DIT")
            if self.fsdp_text_encoder:
                shard_fn = partial(shard_model, device_id=self.device, param_dtype=self.weight_dtype)
                self.pipeline.text_encoder = shard_fn(self.pipeline.text_encoder)
                print("Add FSDP TEXT ENCODER")

        if self.compile_dit:
            for i in range(len(self.pipeline.transformer.transformer_blocks)):
                self.pipeline.transformer.transformer_blocks[i] = torch.compile(self.pipeline.transformer.transformer_blocks[i])
            print("Add Compile")

        if self.GPU_memory_mode == "sequential_cpu_offload":
            self.pipeline.enable_sequential_cpu_offload(device=self.device)
        elif self.GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(self.pipeline.transformer, exclude_module_name=[], device=self.device)
            convert_weight_dtype_wrapper(self.pipeline.transformer, self.weight_dtype)
            self.pipeline.enable_model_cpu_offload(device=self.device)
        elif self.GPU_memory_mode == "model_cpu_offload":
            self.pipeline.enable_model_cpu_offload(device=self.device)
        elif self.GPU_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(self.pipeline.transformer, exclude_module_name=[], device=self.device)
            convert_weight_dtype_wrapper(self.pipeline.transformer, self.weight_dtype)
            self.pipeline.to(self.device)
        else:
            self.pipeline.to(self.device)
        print("Update diffusion transformer done")
        return gr.update()

    @timer
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
        ref_image = None,
        enable_teacache = None, 
        teacache_threshold = None, 
        num_skip_start_steps = None, 
        teacache_offload = None, 
        cfg_skip_ratio = None,
        enable_riflex = None, 
        riflex_k = None, 
        is_api = False,
    ):
        self.clear_cache()

        print(f"Input checking.")
        _, comment = self.input_check(
            resize_method, generation_method, start_image, end_image, validation_video,control_video, is_api
        )
        print(f"Input checking down")
        if comment != "OK":
            return "", comment
        is_image = True if generation_method == "Image Generation" else False

        if self.base_model_path != base_model_dropdown:
            self.update_base_model(base_model_dropdown)

        if self.lora_model_path != lora_model_dropdown:
            self.update_lora_model(lora_model_dropdown)

        print(f"Load scheduler.")
        self.pipeline.scheduler = self.scheduler_dict[sampler_dropdown].from_config(self.pipeline.scheduler.config)
        print(f"Load scheduler down.")

        if resize_method == "Resize according to Reference":
            print(f"Calculate height and width according to Reference.")
            height_slider, width_slider = self.get_height_width_from_reference(
                base_resolution, start_image, validation_video, control_video,
            )

        if self.lora_model_path != "none":
            print(f"Merge Lora.")
            self.pipeline = merge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)
            print(f"Merge Lora done.")

        print(f"Generate seed.")
        if int(seed_textbox) != -1 and seed_textbox != "": torch.manual_seed(int(seed_textbox))
        else: seed_textbox = np.random.randint(0, 1e10)
        generator = torch.Generator(device=self.device).manual_seed(int(seed_textbox))
        print(f"Generate seed done.")
        
        try:
            print(f"Generation.")
            if self.model_type == "Inpaint":
                if self.transformer.config.in_channels != self.vae.config.latent_channels:
                    if generation_method == "Long Video Generation":
                        if validation_video is not None:
                            raise gr.Error(f"Video to Video is not Support Long Video Generation now.")
                        init_frames = 0
                        last_frames = init_frames + partial_video_length
                        while init_frames < length_slider:
                            if last_frames >= length_slider:
                                _partial_video_length = length_slider - init_frames
                                _partial_video_length = int((_partial_video_length - 1) // self.vae.config.temporal_compression_ratio * self.vae.config.temporal_compression_ratio) + 1
                                
                                if _partial_video_length <= 0:
                                    break
                            else:
                                _partial_video_length = partial_video_length

                            if last_frames >= length_slider:
                                input_video, input_video_mask, clip_image = get_image_to_video_latent(start_image, end_image, video_length=_partial_video_length, sample_size=(height_slider, width_slider))
                            else:
                                input_video, input_video_mask, clip_image = get_image_to_video_latent(start_image, None, video_length=_partial_video_length, sample_size=(height_slider, width_slider))

                            with torch.no_grad():
                                sample = self.pipeline(
                                    prompt_textbox, 
                                    negative_prompt     = negative_prompt_textbox,
                                    num_inference_steps = sample_step_slider,
                                    guidance_scale      = cfg_scale_slider,
                                    width               = width_slider,
                                    height              = height_slider,
                                    num_frames          = _partial_video_length,
                                    generator           = generator,

                                    video        = input_video,
                                    mask_video   = input_video_mask,
                                    strength     = 1,
                                ).videos
                            
                            if init_frames != 0:
                                mix_ratio = torch.from_numpy(
                                    np.array([float(_index) / float(overlap_video_length) for _index in range(overlap_video_length)], np.float32)
                                ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                                
                                new_sample[:, :, -overlap_video_length:] = new_sample[:, :, -overlap_video_length:] * (1 - mix_ratio) + \
                                    sample[:, :, :overlap_video_length] * mix_ratio
                                new_sample = torch.cat([new_sample, sample[:, :, overlap_video_length:]], dim = 2)

                                sample = new_sample
                            else:
                                new_sample = sample

                            if last_frames >= length_slider:
                                break

                            start_image = [
                                Image.fromarray(
                                    (sample[0, :, _index].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
                                ) for _index in range(-overlap_video_length, 0)
                            ]

                            init_frames = init_frames + _partial_video_length - overlap_video_length
                            last_frames = init_frames + _partial_video_length
                    else:
                        if validation_video is not None:
                            input_video, input_video_mask, ref_image, clip_image = get_video_to_video_latent(validation_video, length_slider if not is_image else 1, sample_size=(height_slider, width_slider), validation_video_mask=validation_video_mask, fps=8)
                            strength = denoise_strength
                        else:
                            input_video, input_video_mask, clip_image = get_image_to_video_latent(start_image, end_image, length_slider if not is_image else 1, sample_size=(height_slider, width_slider))
                            strength = 1

                        sample = self.pipeline(
                            prompt_textbox,
                            negative_prompt     = negative_prompt_textbox,
                            num_inference_steps = sample_step_slider,
                            guidance_scale      = cfg_scale_slider,
                            width               = width_slider,
                            height              = height_slider,
                            num_frames          = length_slider if not is_image else 1,
                            generator           = generator,

                            video        = input_video,
                            mask_video   = input_video_mask,
                            strength     = strength,
                        ).videos
                else:
                    sample = self.pipeline(
                        prompt_textbox,
                        negative_prompt     = negative_prompt_textbox,
                        num_inference_steps = sample_step_slider,
                        guidance_scale      = cfg_scale_slider,
                        width               = width_slider,
                        height              = height_slider,
                        num_frames          = length_slider if not is_image else 1,
                        generator           = generator
                    ).videos
            else:
                input_video, input_video_mask, ref_image, clip_image = get_video_to_video_latent(control_video, length_slider if not is_image else 1, sample_size=(height_slider, width_slider), fps=8)

                sample = self.pipeline(
                    prompt_textbox,
                    negative_prompt     = negative_prompt_textbox,
                    num_inference_steps = sample_step_slider,
                    guidance_scale      = cfg_scale_slider,
                    width               = width_slider,
                    height              = height_slider,
                    num_frames          = length_slider if not is_image else 1,
                    generator           = generator,

                    control_video = input_video,
                ).videos
        except Exception as e:
            self.auto_model_clear_cache(self.pipeline.transformer)
            self.auto_model_clear_cache(self.pipeline.text_encoder)
            self.auto_model_clear_cache(self.pipeline.vae)
            self.clear_cache()
            
            print(f"Error. error information is {str(e)}")
            if self.lora_model_path != "none":
                self.pipeline = unmerge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)
            if is_api:
                return "", f"Error. error information is {str(e)}"
            else:
                return gr.update(), gr.update(), f"Error. error information is {str(e)}"

        self.clear_cache()
        # lora part
        if self.lora_model_path != "none":
            print(f"Unmerge Lora.")
            self.pipeline = unmerge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)
            print(f"Unmerge Lora done.")

        print(f"Saving outputs.")
        save_sample_path = self.save_outputs(
            is_image, length_slider, sample, fps=8
        )
        print(f"Saving outputs done.")

        if is_image or length_slider == 1:
            if is_api:
                return save_sample_path, "Success"
            else:
                if gradio_version_is_above_4:
                    return gr.Image(value=save_sample_path, visible=True), gr.Video(value=None, visible=False), "Success"
                else:
                    return gr.Image.update(value=save_sample_path, visible=True), gr.Video.update(value=None, visible=False), "Success"
        else:
            if is_api:
                return save_sample_path, "Success"
            else:
                if gradio_version_is_above_4:
                    return gr.Image(visible=False, value=None), gr.Video(value=save_sample_path, visible=True), "Success"
                else:
                    return gr.Image.update(visible=False, value=None), gr.Video.update(value=save_sample_path, visible=True), "Success"

CogVideoXFunController_Host = CogVideoXFunController
CogVideoXFunController_Client = Fun_Controller_Client

def ui(GPU_memory_mode, scheduler_dict, compile_dit, weight_dtype, savedir_sample=None):
    controller = CogVideoXFunController(
        GPU_memory_mode, scheduler_dict, model_name=None, model_type="Inpaint", 
        compile_dit=compile_dit,
        weight_dtype=weight_dtype, savedir_sample=savedir_sample,
    )

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # CogVideoX-Fun:

            A CogVideoX with more flexible generation conditions, capable of producing videos of different resolutions, around 6 seconds, and fps 8 (frames 1 to 49), as well as image generated videos. 

            [Github](https://github.com/aigc-apps/CogVideoX-Fun/)
            """
        )
        with gr.Column(variant="panel"):
            model_type = create_model_type(visible=True)
            diffusion_transformer_dropdown, diffusion_transformer_refresh_button = \
                create_model_checkpoints(controller, visible=True)
            base_model_dropdown, lora_model_dropdown, lora_alpha_slider, personalized_refresh_button = \
                create_finetune_models_checkpoints(controller, visible=True)

        with gr.Column(variant="panel"):
            prompt_textbox, negative_prompt_textbox = create_prompts()

            with gr.Row():
                with gr.Column():
                    sampler_dropdown, sample_step_slider = create_samplers(controller)

                    resize_method, width_slider, height_slider, base_resolution = create_height_width(
                        default_height = 384, default_width = 672, maximum_height = 1344,
                        maximum_width = 1344,
                    )
                    gr.Markdown(
                        """
                        V1.0 and V1.1 support up to 49 frames of video generation, while V1.5 supports up to 85 frames.  
                        (V1.0和V1.1支持最大49帧视频生成，V1.5支持最大85帧视频生成。)
                        """
                    )
                    generation_method, length_slider, overlap_video_length, partial_video_length = \
                        create_generation_methods_and_video_length(
                            ["Video Generation", "Image Generation", "Long Video Generation"],
                            default_video_length=49,
                            maximum_video_length=85,
                        )
                    image_to_video_col, video_to_video_col, control_video_col, source_method, start_image, template_gallery, end_image, validation_video, validation_video_mask, denoise_strength, control_video, ref_image = create_generation_method(
                        ["Text to Video (文本到视频)", "Image to Video (图片到视频)", "Video to Video (视频到视频)", "Video Control (视频控制)"], prompt_textbox
                    )
                    cfg_scale_slider, seed_textbox, seed_button = create_cfg_and_seedbox(gradio_version_is_above_4)

                    generate_button = gr.Button(value="Generate (生成)", variant='primary')
                    
                result_image, result_video, infer_progress = create_ui_outputs()

            model_type.change(
                fn=controller.update_model_type, 
                inputs=[model_type], 
                outputs=[]
            )

            def upload_generation_method(generation_method):
                if generation_method == "Video Generation":
                    return [gr.update(visible=True, maximum=85, value=49, interactive=True), gr.update(visible=False), gr.update(visible=False)]
                elif generation_method == "Image Generation":
                    return [gr.update(minimum=1, maximum=1, value=1, interactive=False), gr.update(visible=False), gr.update(visible=False)]
                else:
                    return [gr.update(visible=True, maximum=1344), gr.update(visible=True), gr.update(visible=True)]
            generation_method.change(
                upload_generation_method, generation_method, [length_slider, overlap_video_length, partial_video_length]
            )

            def upload_source_method(source_method):
                if source_method == "Text to Video (文本到视频)":
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None)]
                elif source_method == "Image to Video (图片到视频)":
                    return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(), gr.update(), gr.update(value=None), gr.update(value=None), gr.update(value=None)]
                elif source_method == "Video to Video (视频到视频)":
                    return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(), gr.update(), gr.update(value=None)]
                else:
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update()]
            source_method.change(
                upload_source_method, source_method, [
                    image_to_video_col, video_to_video_col, control_video_col, start_image, end_image, 
                    validation_video, validation_video_mask, control_video
                ]
            )

            def upload_resize_method(resize_method):
                if resize_method == "Generate by":
                    return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)]
                else:
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]
            resize_method.change(
                upload_resize_method, resize_method, [width_slider, height_slider, base_resolution]
            )

            generate_button.click(
                fn=controller.generate,
                inputs=[
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
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller

def ui_host(GPU_memory_mode, scheduler_dict, model_name, model_type, compile_dit, weight_dtype, savedir_sample=None):
    controller = CogVideoXFunController_Host(
        GPU_memory_mode, scheduler_dict, model_name=model_name, model_type=model_type, 
        compile_dit=compile_dit,
        weight_dtype=weight_dtype, savedir_sample=savedir_sample,
    )

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # CogVideoX-Fun

            A CogVideoX with more flexible generation conditions, capable of producing videos of different resolutions, around 6 seconds, and fps 8 (frames 1 to 49), as well as image generated videos. 

            [Github](https://github.com/aigc-apps/CogVideoX-Fun/)
            """
        )
        with gr.Column(variant="panel"):
            model_type = create_fake_model_type(visible=False)
            diffusion_transformer_dropdown = create_fake_model_checkpoints(model_name, visible=True)
            base_model_dropdown, lora_model_dropdown, lora_alpha_slider = create_fake_finetune_models_checkpoints(visible=True)
        
        with gr.Column(variant="panel"):
            prompt_textbox, negative_prompt_textbox = create_prompts()

            with gr.Row():
                with gr.Column():
                    sampler_dropdown, sample_step_slider = create_samplers(controller)

                    resize_method, width_slider, height_slider, base_resolution = create_height_width(
                        default_height = 384, default_width = 672, maximum_height = 1344,
                        maximum_width = 1344,
                    )
                    gr.Markdown(
                        """
                        V1.0 and V1.1 support up to 49 frames of video generation, while V1.5 supports up to 85 frames.  
                        (V1.0和V1.1支持最大49帧视频生成，V1.5支持最大85帧视频生成。)
                        """
                    )
                    generation_method, length_slider, overlap_video_length, partial_video_length = \
                        create_generation_methods_and_video_length(
                            ["Video Generation", "Image Generation"],
                            default_video_length=49,
                            maximum_video_length=85,
                        )
                    image_to_video_col, video_to_video_col, control_video_col, source_method, start_image, template_gallery, end_image, validation_video, validation_video_mask, denoise_strength, control_video, ref_image = create_generation_method(
                        ["Text to Video (文本到视频)", "Image to Video (图片到视频)", "Video to Video (视频到视频)", "Video Control (视频控制)"], prompt_textbox
                    )
                    cfg_scale_slider, seed_textbox, seed_button = create_cfg_and_seedbox(gradio_version_is_above_4)

                    generate_button = gr.Button(value="Generate (生成)", variant='primary')

                result_image, result_video, infer_progress = create_ui_outputs()

            def upload_generation_method(generation_method):
                if generation_method == "Video Generation":
                    return gr.update(visible=True, minimum=8, maximum=85, value=49, interactive=True)
                elif generation_method == "Image Generation":
                    return gr.update(minimum=1, maximum=1, value=1, interactive=False)
            generation_method.change(
                upload_generation_method, generation_method, [length_slider]
            )

            def upload_source_method(source_method):
                if source_method == "Text to Video (文本到视频)":
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None)]
                elif source_method == "Image to Video (图片到视频)":
                    return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(), gr.update(), gr.update(value=None), gr.update(value=None), gr.update(value=None)]
                elif source_method == "Video to Video (视频到视频)":
                    return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(), gr.update(), gr.update(value=None)]
                else:
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update()]
            source_method.change(
                upload_source_method, source_method, [
                    image_to_video_col, video_to_video_col, control_video_col, start_image, end_image, 
                    validation_video, validation_video_mask, control_video
                ]
            )

            def upload_resize_method(resize_method):
                if resize_method == "Generate by":
                    return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)]
                else:
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]
            resize_method.change(
                upload_resize_method, resize_method, [width_slider, height_slider, base_resolution]
            )

            generate_button.click(
                fn=controller.generate,
                inputs=[
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
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller

def ui_client(scheduler_dict, model_name, savedir_sample=None):
    controller = CogVideoXFunController_Client(scheduler_dict, savedir_sample)

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # CogVideoX-Fun

            A CogVideoX with more flexible generation conditions, capable of producing videos of different resolutions, around 6 seconds, and fps 8 (frames 1 to 49), as well as image generated videos. 

            [Github](https://github.com/aigc-apps/CogVideoX-Fun/)
            """
        )
        with gr.Column(variant="panel"):
            diffusion_transformer_dropdown = create_fake_model_checkpoints(model_name, visible=True)
            base_model_dropdown, lora_model_dropdown, lora_alpha_slider = create_fake_finetune_models_checkpoints(visible=True)
        
        with gr.Column(variant="panel"):
            prompt_textbox, negative_prompt_textbox = create_prompts()

            with gr.Row():
                with gr.Column():
                    sampler_dropdown, sample_step_slider = create_samplers(controller, maximum_step=50)

                    resize_method, width_slider, height_slider, base_resolution = create_fake_height_width(
                        default_height = 384, default_width = 672, maximum_height = 1344,
                        maximum_width = 1344,
                    )
                    gr.Markdown(
                        """
                        V1.0 and V1.1 support up to 49 frames of video generation, while V1.5 supports up to 85 frames.  
                        (V1.0和V1.1支持最大49帧视频生成，V1.5支持最大85帧视频生成。)
                        """
                    )
                    generation_method, length_slider, overlap_video_length, partial_video_length = \
                        create_generation_methods_and_video_length(
                            ["Video Generation", "Image Generation"],
                            default_video_length=49,
                            maximum_video_length=85,
                        )
                    image_to_video_col, video_to_video_col, control_video_col, source_method, start_image, template_gallery, end_image, validation_video, validation_video_mask, denoise_strength, control_video, ref_image = create_generation_method(
                        ["Text to Video (文本到视频)", "Image to Video (图片到视频)", "Video to Video (视频到视频)"], prompt_textbox
                    )

                    cfg_scale_slider, seed_textbox, seed_button = create_cfg_and_seedbox(gradio_version_is_above_4)

                    generate_button = gr.Button(value="Generate (生成)", variant='primary')

                result_image, result_video, infer_progress = create_ui_outputs()

            def upload_generation_method(generation_method):
                if generation_method == "Video Generation":
                    return gr.update(visible=True, minimum=5, maximum=85, value=49, interactive=True)
                elif generation_method == "Image Generation":
                    return gr.update(minimum=1, maximum=1, value=1, interactive=False)
            generation_method.change(
                upload_generation_method, generation_method, [length_slider]
            )

            def upload_source_method(source_method):
                if source_method == "Text to Video (文本到视频)":
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None)]
                elif source_method == "Image to Video (图片到视频)":
                    return [gr.update(visible=True), gr.update(visible=False), gr.update(), gr.update(), gr.update(value=None), gr.update(value=None)]
                else:
                    return [gr.update(visible=False), gr.update(visible=True), gr.update(value=None), gr.update(value=None), gr.update(), gr.update()]
            source_method.change(
                upload_source_method, source_method, [image_to_video_col, video_to_video_col, start_image, end_image, validation_video, validation_video_mask]
            )

            def upload_resize_method(resize_method):
                if resize_method == "Generate by":
                    return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)]
                else:
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]
            resize_method.change(
                upload_resize_method, resize_method, [width_slider, height_slider, base_resolution]
            )

            generate_button.click(
                fn=controller.generate,
                inputs=[
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
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller