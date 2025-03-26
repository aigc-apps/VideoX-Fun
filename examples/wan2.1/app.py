import os
import sys
import time

import torch

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.api.api import (infer_forward_api,
                               update_diffusion_transformer_api,
                               update_edition_api)
from videox_fun.ui.controller import flow_scheduler_dict
from videox_fun.ui.wan_ui import ui, ui_client, ui_host

if __name__ == "__main__":
    # Choose the ui mode  
    ui_mode = "normal"
    
    # GPU memory mode, which can be choosen in [model_full_load, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
    # model_full_load means that the entire model will be moved to the GPU.
    # 
    # model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
    # 
    # model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
    # and the transformer model has been quantized to float8, which can save more GPU memory. 
    # 
    # sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
    # resulting in slower speeds but saving a large amount of GPU memory.
    GPU_memory_mode = "sequential_cpu_offload"

    # Support TeaCache.
    enable_teacache     = True
    # Recommended to be set between 0.05 and 0.20. A larger threshold can cache more steps, speeding up the inference process, 
    # but it may cause slight differences between the generated content and the original content.
    teacache_threshold  = 0.10
    # The number of steps to skip TeaCache at the beginning of the inference process, which can
    # reduce the impact of TeaCache on generated video quality.
    num_skip_start_steps = 5
    # Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
    teacache_offload    = False
    # Use torch.float16 if GPU does not support torch.bfloat16
    # ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
    weight_dtype = torch.bfloat16

    # Server ip
    server_name = "0.0.0.0"
    server_port = 7860
    
    # Config path
    config_path = "config/wan2.1/wan_civitai.yaml"
    # Params below is used when ui_mode = "host"
    # Model path of the pretrained model
    model_name = "models/Diffusion_Transformer/Wan2.1-I2V-14B-480P"
    # "Inpaint" or "Control"
    model_type = "Inpaint"

    if ui_mode == "host":
        demo, controller = ui_host(GPU_memory_mode, flow_scheduler_dict, model_name, model_type, config_path, 1, 1, enable_teacache, teacache_threshold, num_skip_start_steps, teacache_offload, weight_dtype)
    elif ui_mode == "client":
        demo, controller = ui_client(flow_scheduler_dict, model_name)
    else:
        demo, controller = ui(GPU_memory_mode, flow_scheduler_dict, config_path, 1, 1, enable_teacache, teacache_threshold, num_skip_start_steps, teacache_offload, weight_dtype)

    def gr_launch():
        # launch gradio
        app, _, _ = demo.queue(status_update_rate=1).launch(
            server_name=server_name,
            server_port=server_port,
            prevent_thread_lock=True
        )
        
        # launch api
        infer_forward_api(None, app, controller)
        update_diffusion_transformer_api(None, app, controller)
        update_edition_api(None, app, controller)
    
    gr_launch()
        
    # not close the python
    while True:
        time.sleep(5)