"""Modified from https://github.com/guoyww/AnimateDiff/blob/main/app.py
"""
import os
import random

import cv2
import gradio as gr
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from safetensors import safe_open

from ..data.bucket_sampler import ASPECT_RATIO_512, get_closest_ratio
from ..dist import set_multi_gpus_devices, shard_model
from ..models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer,
                      CLIPModel, Wan2_2Transformer3DModel, WanT5EncoderModel)
from ..models.cache_utils import get_teacache_coefficients
from ..pipeline import Wan2_2FunControlPipeline, Wan2_2FunPipeline, Wan2_2FunInpaintPipeline
from ..utils.fp8_optimization import (convert_model_weight_to_float8,
                                      convert_weight_dtype_wrapper,
                                      replace_parameters_by_name)
from ..utils.lora_utils import merge_lora, unmerge_lora
from ..utils.utils import (filter_kwargs, get_image_latent,
                           get_image_to_video_latent,
                           get_video_to_video_latent, save_videos_grid, timer)
from .controller import (Fun_Controller, Fun_Controller_Client,
                         all_cheduler_dict, css, ddpm_scheduler_dict,
                         flow_scheduler_dict, gradio_version,
                         gradio_version_is_above_4)
from .ui import (create_cfg_and_seedbox, create_cfg_riflex_k,
                 create_cfg_skip_params, create_config,
                 create_fake_finetune_models_checkpoints,
                 create_fake_height_width, create_fake_model_checkpoints,
                 create_fake_model_type, create_finetune_models_checkpoints,
                 create_generation_method,
                 create_generation_methods_and_video_length,
                 create_height_width, create_model_checkpoints,
                 create_model_type, create_prompts, create_samplers,
                 create_teacache_params, create_ui_outputs)


class Wan2_2_Fun_Controller(Fun_Controller):
    def update_diffusion_transformer(self, diffusion_transformer_dropdown):
        print(f"Update diffusion transformer: {diffusion_transformer_dropdown}")
        self.model_name = diffusion_transformer_dropdown
        self.diffusion_transformer_dropdown = diffusion_transformer_dropdown
        if diffusion_transformer_dropdown == "none":
            return gr.update()
        Chosen_AutoencoderKL = {
            "AutoencoderKLWan": AutoencoderKLWan,
            "AutoencoderKLWan3_8": AutoencoderKLWan3_8
        }[self.config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
        self.vae = Chosen_AutoencoderKL.from_pretrained(
            os.path.join(diffusion_transformer_dropdown, self.config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(self.config['vae_kwargs']),
        ).to(self.weight_dtype)

        # Get Transformer
        self.transformer = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(diffusion_transformer_dropdown, self.config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(self.config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=self.weight_dtype,
        )
        if self.config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
            self.transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
                os.path.join(diffusion_transformer_dropdown, self.config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
                transformer_additional_kwargs=OmegaConf.to_container(self.config['transformer_additional_kwargs']),
                low_cpu_mem_usage=True,
                torch_dtype=self.weight_dtype,
            )
        else:
            self.transformer_2 = None

        # Get Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(diffusion_transformer_dropdown, self.config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )

        # Get Text encoder
        self.text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(diffusion_transformer_dropdown, self.config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(self.config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=self.weight_dtype,
        )
        self.text_encoder = self.text_encoder.eval()
        
        Chosen_Scheduler = self.scheduler_dict[list(self.scheduler_dict.keys())[0]]
        self.scheduler = Chosen_Scheduler(
            **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(self.config['scheduler_kwargs']))
        )

        # Get pipeline
        if self.model_type == "Inpaint":
            if self.transformer.config.in_channels != self.vae.config.latent_channels:
                self.pipeline = Wan2_2FunInpaintPipeline(
                    vae=self.vae,
                    tokenizer=self.tokenizer,
                    text_encoder=self.text_encoder,
                    transformer=self.transformer,
                    transformer_2=self.transformer_2,
                    scheduler=self.scheduler,
                )
            else:
                self.pipeline = Wan2_2FunPipeline(
                    vae=self.vae,
                    tokenizer=self.tokenizer,
                    text_encoder=self.text_encoder,
                    transformer=self.transformer,
                    transformer_2=self.transformer_2,
                    scheduler=self.scheduler,
                )
        else:
            self.pipeline = Wan2_2FunControlPipeline(
                vae=self.vae,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                transformer=self.transformer,
                transformer_2=self.transformer_2,
                scheduler=self.scheduler,
            )

        if self.ulysses_degree > 1 or self.ring_degree > 1:
            from functools import partial
            self.transformer.enable_multi_gpus_inference()
            if self.transformer_2 is not None:
                self.transformer_2.enable_multi_gpus_inference()
            if self.fsdp_dit:
                shard_fn = partial(shard_model, device_id=self.device, param_dtype=self.weight_dtype)
                self.pipeline.transformer = shard_fn(self.pipeline.transformer)
                if self.transformer_2 is not None:
                    self.pipeline.transformer_2 = shard_fn(self.pipeline.transformer_2)
                print("Add FSDP DIT")
            if self.fsdp_text_encoder:
                shard_fn = partial(shard_model, device_id=self.device, param_dtype=self.weight_dtype)
                self.pipeline.text_encoder = shard_fn(self.pipeline.text_encoder)
                print("Add FSDP TEXT ENCODER")

        if self.compile_dit:
            for i in range(len(self.pipeline.transformer.blocks)):
                self.pipeline.transformer.blocks[i] = torch.compile(self.pipeline.transformer.blocks[i])
            if self.transformer_2 is not None:
                for i in range(len(self.pipeline.transformer_2.blocks)):
                    self.pipeline.transformer_2.blocks[i] = torch.compile(self.pipeline.transformer_2.blocks[i])
            print("Add Compile")

        if self.GPU_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(self.transformer, ["modulation",], device=self.device)
            self.transformer.freqs = self.transformer.freqs.to(device=self.device)
            if self.transformer_2 is not None:
                replace_parameters_by_name(self.transformer_2, ["modulation",], device=self.device)
                self.transformer_2.freqs = self.transformer_2.freqs.to(device=self.device)
            self.pipeline.enable_sequential_cpu_offload(device=self.device)
        elif self.GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(self.transformer, exclude_module_name=["modulation",], device=self.device)
            convert_weight_dtype_wrapper(self.transformer, self.weight_dtype)
            if self.transformer_2 is not None:
                convert_model_weight_to_float8(self.transformer_2, exclude_module_name=["modulation",], device=self.device)
                convert_weight_dtype_wrapper(self.transformer_2, self.weight_dtype)
            self.pipeline.enable_model_cpu_offload(device=self.device)
        elif self.GPU_memory_mode == "model_cpu_offload":
            self.pipeline.enable_model_cpu_offload(device=self.device)
        elif self.GPU_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(self.transformer, exclude_module_name=["modulation",], device=self.device)
            convert_weight_dtype_wrapper(self.transformer, self.weight_dtype)
            if self.transformer_2 is not None:
                convert_model_weight_to_float8(self.transformer_2, exclude_module_name=["modulation",], device=self.device)
                convert_weight_dtype_wrapper(self.transformer_2, self.weight_dtype)
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
        base_model_2_dropdown=None,
        lora_model_2_dropdown=None, 
        fps = None,
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
        if self.base_model_2_path != base_model_2_dropdown:
            self.update_lora_model(base_model_2_dropdown, is_checkpoint_2=True)

        if self.lora_model_path != lora_model_dropdown:
            self.update_lora_model(lora_model_dropdown)
        if self.lora_model_2_path != lora_model_2_dropdown:
            self.update_lora_model(lora_model_2_dropdown, is_checkpoint_2=True)

        print(f"Load scheduler.")
        scheduler_config = self.pipeline.scheduler.config
        if sampler_dropdown == "Flow_Unipc" or sampler_dropdown == "Flow_DPM++":
            scheduler_config['shift'] = 1
        self.pipeline.scheduler = self.scheduler_dict[sampler_dropdown].from_config(scheduler_config)
        print(f"Load scheduler down.")

        if resize_method == "Resize according to Reference":
            print(f"Calculate height and width according to Reference.")
            height_slider, width_slider = self.get_height_width_from_reference(
                base_resolution, start_image, validation_video, control_video,
            )

        if self.lora_model_path != "none":
            print(f"Merge Lora.")
            self.pipeline = merge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)
            if self.transformer_2 is not None:
                self.pipeline = merge_lora(self.pipeline, self.lora_model_2_path, multiplier=lora_alpha_slider, sub_transformer_name="transformer_2")
            print(f"Merge Lora done.")

        coefficients = get_teacache_coefficients(self.diffusion_transformer_dropdown) if enable_teacache else None
        if coefficients is not None:
            print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
            self.pipeline.transformer.enable_teacache(
                coefficients, sample_step_slider, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
            )
            if self.transformer_2 is not None:
                self.pipeline.transformer_2.share_teacache(self.pipeline.transformer)
        else:
            print(f"Disable TeaCache.")
            self.pipeline.transformer.disable_teacache()
            if self.transformer_2 is not None:
                self.pipeline.transformer_2.disable_teacache()

        if cfg_skip_ratio is not None and cfg_skip_ratio >= 0:
            print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
            self.pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, sample_step_slider)
            if self.transformer_2 is not None:
                self.pipeline.transformer_2.share_cfg_skip(self.pipeline.transformer)

        print(f"Generate seed.")
        if int(seed_textbox) != -1 and seed_textbox != "": torch.manual_seed(int(seed_textbox))
        else: seed_textbox = np.random.randint(0, 1e10)
        generator = torch.Generator(device=self.device).manual_seed(int(seed_textbox))
        print(f"Generate seed done.")

        if fps is None:
            fps = 16
        boundary = self.config['transformer_additional_kwargs'].get('boundary', 0.875)
                
        if enable_riflex:
            print(f"Enable riflex")
            latent_frames = (int(length_slider) - 1) // self.vae.config.temporal_compression_ratio + 1
            self.pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames if not is_image else 1)
            if self.transformer_2 is not None:
                self.pipeline.transformer_2.enable_riflex(k = riflex_k, L_test = latent_frames if not is_image else 1)

        try:
            print(f"Generation.")
            if self.model_type == "Inpaint":
                if self.transformer.config.in_channels != self.vae.config.latent_channels:
                    if validation_video is not None:
                        input_video, input_video_mask, _, clip_image = get_video_to_video_latent(validation_video, length_slider if not is_image else 1, sample_size=(height_slider, width_slider), validation_video_mask=validation_video_mask, fps=fps)
                    else:
                        input_video, input_video_mask, clip_image = get_image_to_video_latent(start_image, end_image, length_slider if not is_image else 1, sample_size=(height_slider, width_slider))

                    sample = self.pipeline(
                        prompt_textbox,
                        negative_prompt     = negative_prompt_textbox,
                        num_inference_steps = sample_step_slider,
                        guidance_scale      = cfg_scale_slider,
                        width               = width_slider,
                        height              = height_slider,
                        num_frames          = length_slider if not is_image else 1,
                        generator           = generator,

                        video               = input_video,
                        mask_video          = input_video_mask,
                        boundary            = boundary
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
                        generator           = generator, 
                        boundary            = boundary
                    ).videos
            else:
                inpaint_video, inpaint_video_mask, clip_image = get_image_to_video_latent(start_image, end_image, video_length=length_slider if not is_image else 1, sample_size=(height_slider, width_slider))
                
                if ref_image is not None:
                    ref_image = get_image_latent(ref_image, sample_size=(height_slider, width_slider))

                input_video, input_video_mask, _, _ = get_video_to_video_latent(control_video, video_length=length_slider if not is_image else 1, sample_size=(height_slider, width_slider), fps=fps, ref_image=None)

                sample = self.pipeline(
                    prompt_textbox,
                    negative_prompt     = negative_prompt_textbox,
                    num_inference_steps = sample_step_slider,
                    guidance_scale      = cfg_scale_slider,
                    width               = width_slider,
                    height              = height_slider,
                    num_frames          = length_slider if not is_image else 1,
                    generator           = generator,

                    video      = inpaint_video,
                    mask_video   = inpaint_video_mask,
                    control_video = input_video,
                    ref_image = ref_image,
                    boundary = boundary,
                ).videos
            print(f"Generation done.")
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
            is_image, length_slider, sample, fps=fps
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

Wan2_2_Fun_Controller_Host = Wan2_2_Fun_Controller
Wan2_2_Fun_Controller_Client = Fun_Controller_Client

def ui(GPU_memory_mode, scheduler_dict, config_path, compile_dit, weight_dtype, savedir_sample=None):
    controller = Wan2_2_Fun_Controller(
        GPU_memory_mode, scheduler_dict, model_name=None, model_type="Inpaint", 
        config_path=config_path, compile_dit=compile_dit,
        weight_dtype=weight_dtype, savedir_sample=savedir_sample,
    )

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # Wan2.2-Fun:

            A Wan with more flexible generation conditions, capable of producing videos of different resolutions, around 5 seconds, and fps 16 (frames 1 to 81), as well as image generated videos. 

            [Github](https://github.com/aigc-apps/VideoX-Fun/)
            """
        )
        with gr.Column(variant="panel"):
            config_dropdown, config_refresh_button = create_config(controller)
            model_type = create_model_type(visible=True)
            diffusion_transformer_dropdown, diffusion_transformer_refresh_button = \
                create_model_checkpoints(controller, visible=True)
            base_model_dropdown, lora_model_dropdown, lora_alpha_slider, personalized_refresh_button = \
                create_finetune_models_checkpoints(controller, visible=True, add_checkpoint_2=True)
            base_model_dropdown, base_model_2_dropdown = base_model_dropdown
            lora_model_dropdown, lora_model_2_dropdown = lora_model_dropdown
            
            with gr.Row():
                enable_teacache, teacache_threshold, num_skip_start_steps, teacache_offload = \
                    create_teacache_params(True, 0.10, 1, False)
                cfg_skip_ratio = create_cfg_skip_params(0)
                enable_riflex, riflex_k = create_cfg_riflex_k(False, 6)

        with gr.Column(variant="panel"):
            prompt_textbox, negative_prompt_textbox = create_prompts(negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

            with gr.Row():
                with gr.Column():
                    sampler_dropdown, sample_step_slider = create_samplers(controller)

                    resize_method, width_slider, height_slider, base_resolution = create_height_width(
                        default_height = 480, default_width = 832, maximum_height = 1344,
                        maximum_width = 1344,
                    )
                    generation_method, length_slider, overlap_video_length, partial_video_length = \
                        create_generation_methods_and_video_length(
                            ["Video Generation", "Image Generation"],
                            default_video_length=81,
                            maximum_video_length=161,
                        )
                    image_to_video_col, video_to_video_col, control_video_col, source_method, start_image, template_gallery, end_image, validation_video, validation_video_mask, denoise_strength, control_video, ref_image = create_generation_method(
                        ["Text to Video (文本到视频)", "Image to Video (图片到视频)", "Video Control (视频控制)"], prompt_textbox, support_ref_image=True
                    )
                    cfg_scale_slider, seed_textbox, seed_button = create_cfg_and_seedbox(gradio_version_is_above_4)

                    generate_button = gr.Button(value="Generate (生成)", variant='primary')
                    
                result_image, result_video, infer_progress = create_ui_outputs()

            config_dropdown.change(
                fn=controller.update_config, 
                inputs=[config_dropdown], 
                outputs=[]
            )

            model_type.change(
                fn=controller.update_model_type, 
                inputs=[model_type], 
                outputs=[]
            )

            def upload_generation_method(generation_method):
                if generation_method == "Video Generation":
                    return [gr.update(visible=True, maximum=161, value=81, interactive=True), gr.update(visible=False), gr.update(visible=False)]
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
                    ref_image, 
                    enable_teacache, 
                    teacache_threshold, 
                    num_skip_start_steps, 
                    teacache_offload, 
                    cfg_skip_ratio,
                    enable_riflex, 
                    riflex_k, 
                    base_model_2_dropdown, 
                    lora_model_2_dropdown
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller

def ui_host(GPU_memory_mode, scheduler_dict, model_name, model_type, config_path, compile_dit, weight_dtype, savedir_sample=None):
    controller = Wan2_2_Fun_Controller_Host(
        GPU_memory_mode, scheduler_dict, model_name=model_name, model_type=model_type, 
        config_path=config_path, compile_dit=compile_dit,
        weight_dtype=weight_dtype, savedir_sample=savedir_sample,
    )

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # Wan2.2-Fun:

            A Wan with more flexible generation conditions, capable of producing videos of different resolutions, around 5 seconds, and fps 16 (frames 1 to 81), as well as image generated videos. 

            [Github](https://github.com/aigc-apps/VideoX-Fun/)
            """
        )
        with gr.Column(variant="panel"):
            model_type = create_fake_model_type(visible=False)
            diffusion_transformer_dropdown = create_fake_model_checkpoints(model_name, visible=True)
            base_model_dropdown, lora_model_dropdown, lora_alpha_slider = \
                create_fake_finetune_models_checkpoints(visible=True, add_checkpoint_2=True)
            base_model_dropdown, base_model_2_dropdown = base_model_dropdown
            lora_model_dropdown, lora_model_2_dropdown = lora_model_dropdown

            with gr.Row():
                enable_teacache, teacache_threshold, num_skip_start_steps, teacache_offload = \
                    create_teacache_params(True, 0.10, 1, False)
                cfg_skip_ratio = create_cfg_skip_params(0)
                enable_riflex, riflex_k = create_cfg_riflex_k(False, 6)
        
        with gr.Column(variant="panel"):
            prompt_textbox, negative_prompt_textbox = create_prompts(negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

            with gr.Row():
                with gr.Column():
                    sampler_dropdown, sample_step_slider = create_samplers(controller)

                    resize_method, width_slider, height_slider, base_resolution = create_height_width(
                        default_height = 480, default_width = 832, maximum_height = 1344,
                        maximum_width = 1344,
                    )
                    generation_method, length_slider, overlap_video_length, partial_video_length = \
                        create_generation_methods_and_video_length(
                            ["Video Generation", "Image Generation"],
                            default_video_length=81,
                            maximum_video_length=161,
                        )
                    image_to_video_col, video_to_video_col, control_video_col, source_method, start_image, template_gallery, end_image, validation_video, validation_video_mask, denoise_strength, control_video, ref_image = create_generation_method(
                        ["Text to Video (文本到视频)", "Image to Video (图片到视频)", "Video Control (视频控制)"], prompt_textbox, support_ref_image=True
                    )
                    cfg_scale_slider, seed_textbox, seed_button = create_cfg_and_seedbox(gradio_version_is_above_4)

                    generate_button = gr.Button(value="Generate (生成)", variant='primary')

                result_image, result_video, infer_progress = create_ui_outputs()

            def upload_generation_method(generation_method):
                if generation_method == "Video Generation":
                    return gr.update(visible=True, minimum=1, maximum=161, value=81, interactive=True)
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
                    ref_image, 
                    enable_teacache, 
                    teacache_threshold, 
                    num_skip_start_steps, 
                    teacache_offload, 
                    cfg_skip_ratio,
                    enable_riflex, 
                    riflex_k, 
                    base_model_2_dropdown, 
                    lora_model_2_dropdown
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller

def ui_client(scheduler_dict, model_name, savedir_sample=None):
    controller = Wan2_2_Fun_Controller_Client(scheduler_dict, savedir_sample)

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # Wan2.2-Fun:

            A Wan with more flexible generation conditions, capable of producing videos of different resolutions, around 5 seconds, and fps 16 (frames 1 to 81), as well as image generated videos. 

            [Github](https://github.com/aigc-apps/VideoX-Fun/)
            """
        )
        with gr.Column(variant="panel"):
            diffusion_transformer_dropdown = create_fake_model_checkpoints(model_name, visible=True)
            base_model_dropdown, lora_model_dropdown, lora_alpha_slider = \
                create_fake_finetune_models_checkpoints(visible=True, add_checkpoint_2=True)
            base_model_dropdown, base_model_2_dropdown = base_model_dropdown
            lora_model_dropdown, lora_model_2_dropdown = lora_model_dropdown

            with gr.Row():
                enable_teacache, teacache_threshold, num_skip_start_steps, teacache_offload = \
                    create_teacache_params(True, 0.10, 1, False)
                cfg_skip_ratio = create_cfg_skip_params(0)
                enable_riflex, riflex_k = create_cfg_riflex_k(False, 6)
        
        with gr.Column(variant="panel"):
            prompt_textbox, negative_prompt_textbox = create_prompts(negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

            with gr.Row():
                with gr.Column():
                    sampler_dropdown, sample_step_slider = create_samplers(controller, maximum_step=50)

                    resize_method, width_slider, height_slider, base_resolution = create_fake_height_width(
                        default_height = 480, default_width = 832, maximum_height = 1344,
                        maximum_width = 1344,
                    )
                    generation_method, length_slider, overlap_video_length, partial_video_length = \
                        create_generation_methods_and_video_length(
                            ["Video Generation", "Image Generation"],
                            default_video_length=81,
                            maximum_video_length=161,
                        )
                    image_to_video_col, video_to_video_col, control_video_col, source_method, start_image, template_gallery, end_image, validation_video, validation_video_mask, denoise_strength, control_video, ref_image = create_generation_method(
                        ["Text to Video (文本到视频)", "Image to Video (图片到视频)"], prompt_textbox
                    )

                    cfg_scale_slider, seed_textbox, seed_button = create_cfg_and_seedbox(gradio_version_is_above_4)

                    generate_button = gr.Button(value="Generate (生成)", variant='primary')

                result_image, result_video, infer_progress = create_ui_outputs()

            def upload_generation_method(generation_method):
                if generation_method == "Video Generation":
                    return gr.update(visible=True, minimum=5, maximum=161, value=49, interactive=True)
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
                    ref_image, 
                    enable_teacache, 
                    teacache_threshold, 
                    num_skip_start_steps, 
                    teacache_offload, 
                    cfg_skip_ratio,
                    enable_riflex, 
                    riflex_k, 
                    base_model_2_dropdown, 
                    lora_model_2_dropdown
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller