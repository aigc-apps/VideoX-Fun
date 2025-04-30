import base64
import json
import time
from datetime import datetime

import requests
import base64


def post_diffusion_transformer(diffusion_transformer_path, url='http://127.0.0.1:7860'):
    datas = json.dumps({
        "diffusion_transformer_path": diffusion_transformer_path
    })
    r = requests.post(f'{url}/cogvideox_fun/update_diffusion_transformer', data=datas, timeout=1500)
    data = r.content.decode('utf-8')
    return data

def post_update_edition(edition, url='http://0.0.0.0:7860'):
    datas = json.dumps({
        "edition": edition
    })
    r = requests.post(f'{url}/cogvideox_fun/update_edition', data=datas, timeout=1500)
    data = r.content.decode('utf-8')
    return data


def post_infer(
    generation_method, 
    length_slider, 
    url='http://127.0.0.1:7860', 
    POST_TOKEN="", 
    timeout=5000,
    base_model_path="none",
    lora_model_path="none",
    lora_alpha_slider=0.55,
    prompt_textbox="A young woman with beautiful and clear eyes and blonde hair standing and white dress in a forest wearing a crown. She seems to be lost in thought, and the camera focuses on her face. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
    negative_prompt_textbox="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
    sampler_dropdown="Flow",
    sample_step_slider=50,
    width_slider=672,
    height_slider=384,
    cfg_scale_slider=6,
    seed_textbox=43
):
    # Prepare the data payload
    datas = json.dumps({
        "base_model_path": base_model_path,
        "lora_model_path": lora_model_path,
        "lora_alpha_slider": lora_alpha_slider,
        "prompt_textbox": prompt_textbox,
        "negative_prompt_textbox": negative_prompt_textbox,
        "sampler_dropdown": sampler_dropdown,
        "sample_step_slider": sample_step_slider,
        "width_slider": width_slider,
        "height_slider": height_slider,
        "generation_method": generation_method,
        "length_slider": length_slider,
        "cfg_scale_slider": cfg_scale_slider,
        "seed_textbox": seed_textbox,
    })

    # Initialize session and set headers
    session = requests.session()
    session.headers.update({"Authorization": POST_TOKEN})

    # Send POST request
    if url[-1] == "/":
        url = url[:-1]
    post_r = session.post(f'{url}/videox_fun/infer_forward', data=datas, timeout=timeout)
    
    data = post_r.content.decode('utf-8')
    return data

if __name__ == '__main__':
    # initiate time
    time_start  = time.time()  

    # The Url you want to post
    POST_URL = 'http://0.0.0.0:7860'
    # Used in EAS. If you don't need Authorization, please set it to empty string.
    TOKEN = ''
    
    # -------------------------- #
    #  Step 1: update edition
    # -------------------------- #
    # diffusion_transformer_path = "models/Diffusion_Transformer/CogVideoX-Fun-2b-InP"
    # outputs = post_diffusion_transformer(diffusion_transformer_path)
    # print('Output update edition: ', outputs)

    # -------------------------- #
    #  Step 2: infer
    # -------------------------- #
    # "Video Generation" and "Image Generation"
    generation_method   = "Video Generation"
    # Video length
    length_slider       = 49
    # Used in Lora models
    lora_model_path     = "none"
    lora_alpha_slider   = 0.55
    # Prompts
    prompt_textbox      = "A young woman with beautiful and clear eyes and blonde hair standing and white dress in a forest wearing a crown. She seems to be lost in thought, and the camera focuses on her face. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
    negative_prompt_textbox = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion."
    # Sampler name
    sampler_dropdown    = "Euler"
    # Sampler steps
    sample_step_slider  = 50
    # height and width 
    width_slider        = 672
    height_slider       = 384
    # cfg scale
    cfg_scale_slider    = 6
    seed_textbox        = 43

    outputs = post_infer(
        generation_method, 
        length_slider, 
        lora_model_path=lora_model_path,
        lora_alpha_slider=lora_alpha_slider,
        prompt_textbox=prompt_textbox,
        negative_prompt_textbox=negative_prompt_textbox,
        sampler_dropdown=sampler_dropdown,
        sample_step_slider=sample_step_slider,
        width_slider=width_slider,
        height_slider=height_slider,
        cfg_scale_slider=cfg_scale_slider,
        seed_textbox=seed_textbox,
        url=POST_URL, 
        POST_TOKEN=TOKEN
    )
    
    # Get decoded data
    outputs = json.loads(outputs)
    base64_encoding = outputs["base64_encoding"]
    decoded_data = base64.b64decode(base64_encoding)

    is_image = True if generation_method == "Image Generation" else False
    if is_image or length_slider == 1:
        file_path = "1.png"
    else:
        file_path = "1.mp4"
    with open(file_path, "wb") as file:
        file.write(decoded_data)
        
    # End of record time
    # The calculated time difference is the execution time of the program, expressed in seconds / s
    time_end = time.time()  
    time_sum = (time_end - time_start)
    print('# --------------------------------------------------------- #')
    print(f'#   Total expenditure: {time_sum}s')
    print('# --------------------------------------------------------- #')