import os
import sys
import time

import torch

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.api.api import (infer_forward_api,
                               update_diffusion_transformer_api)
from videox_fun.ui.controller import ddpm_scheduler_dict
from videox_fun.ui.cogvideox_fun_ui import ui, ui_client, ui_host

if __name__ == "__main__":
    # Choose the ui mode  
    # "normal" refers to the standard UI, which allows users to click to switch models, change model types, and more. 
    # "host" represents the hosting mode, where the model is loaded directly at startup and can be accessed via 
    #        the API to return generation results. 
    # "client" represents the client mode, offering a simple UI that sends requests to a remote API for generation.
    ui_mode = "normal"
    
    # GPU memory mode, which can be chosen in [model_full_load, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
    # model_full_load means that the entire model will be moved to the GPU.
    # 
    # model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
    # 
    # model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
    # and the transformer model has been quantized to float8, which can save more GPU memory. 
    # 
    # sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
    # resulting in slower speeds but saving a large amount of GPU memory.
    GPU_memory_mode = "model_cpu_offload_and_qfloat8"
    # Compile will give a speedup in fixed resolution and need a little GPU memory. 
    # The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
    compile_dit = False

    # Use torch.float16 if GPU does not support torch.bfloat16
    # ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
    weight_dtype = torch.bfloat16

    # Server ip
    server_name = "0.0.0.0"
    server_port = 7860

    # Params below is used when ui_mode = "host"
    model_name = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP"
    # "Inpaint" or "Control"
    model_type = "Inpaint"

    if ui_mode == "host":
        demo, controller = ui_host(GPU_memory_mode, ddpm_scheduler_dict, model_name, model_type, compile_dit, weight_dtype)
    elif ui_mode == "client":
        demo, controller = ui_client(ddpm_scheduler_dict, model_name)
    else:
        demo, controller = ui(GPU_memory_mode, ddpm_scheduler_dict, compile_dit, weight_dtype)

    # launch gradio
    app, _, _ = demo.queue(status_update_rate=1).launch(
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=True
    )
    
    # launch api
    infer_forward_api(None, app, controller)
    update_diffusion_transformer_api(None, app, controller)
    
    # not close the python
    while True:
        time.sleep(5)