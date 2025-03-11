import os
import sys
import time

import torch

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from cogvideox.api.api import (infer_forward_api,
                               update_diffusion_transformer_api,
                               update_edition_api)
from cogvideox.ui.controller import flow_scheduler_dict
from cogvideox.ui.wan_ui import ui, ui_eas, ui_modelscope

if __name__ == "__main__":
    # Choose the ui mode  
    ui_mode = "normal"
    
    # GPU memory mode, which can be choosen in [model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
    # model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
    # 
    # model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
    # and the transformer model has been quantized to float8, which can save more GPU memory. 
    # 
    # sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
    # resulting in slower speeds but saving a large amount of GPU memory.
    GPU_memory_mode = "model_cpu_offload_and_qfloat8"
    # Use torch.float16 if GPU does not support torch.bfloat16
    # ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
    weight_dtype = torch.bfloat16
    # Config path
    config_path = "config/wan2.1/wan_civitai.yaml"

    # Server ip
    server_name = "0.0.0.0"
    server_port = 7860

    # Params below is used when ui_mode = "modelscope"
    model_name = "models/Diffusion_Transformer/Wan2.1-Fun-1.3B-InP"
    # "Inpaint" or "Control"
    model_type = "Inpaint"
    # Save dir of this model
    savedir_sample = "samples"

    if ui_mode == "modelscope":
        demo, controller = ui_modelscope(model_name, model_type, savedir_sample, GPU_memory_mode, flow_scheduler_dict, weight_dtype, config_path)
    elif ui_mode == "eas":
        demo, controller = ui_eas(model_name, flow_scheduler_dict, savedir_sample, config_path)
    else:
        demo, controller = ui(GPU_memory_mode, flow_scheduler_dict, weight_dtype, config_path)

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
    
    # not close the python
    while True:
        time.sleep(5)