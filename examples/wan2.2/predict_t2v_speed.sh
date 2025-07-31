export EXCEL_FILE="./speed.xlsx"

export DIT_EXCEL_COL=0 VAE_EXCEL_COL=1 TOTAL_EXCEL_COL=2

# 14B 720P
export DIT_EXCEL_ROW=1 VAE_EXCEL_ROW=1 TOTAL_EXCEL_ROW=1
python examples/wan2.2/predict_t2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.2-T2V-A14B" \
    --GPU_memory_mode="model_full_load" --ulysses_degree=1 --ring_degree=1 --fsdp_text_encoder --fsdp_dit \
    --enable_teacache --teacache_threshold=0.10 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=5 \
    --sample_size 720 1280 --num_inference_steps=40 

export DIT_EXCEL_ROW=2 VAE_EXCEL_ROW=2 TOTAL_EXCEL_ROW=2
torchrun --nproc-per-node=2 examples/wan2.2/predict_t2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.2-T2V-A14B" \
    --GPU_memory_mode="model_full_load" --ulysses_degree=2 --ring_degree=1 --fsdp_text_encoder --fsdp_dit \
    --enable_teacache --teacache_threshold=0.10 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=5 \
    --sample_size 720 1280 --num_inference_steps=40 

export DIT_EXCEL_ROW=3 VAE_EXCEL_ROW=3 TOTAL_EXCEL_ROW=3
torchrun --nproc-per-node=4 examples/wan2.2/predict_t2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.2-T2V-A14B" \
    --GPU_memory_mode="model_full_load" --ulysses_degree=4 --ring_degree=1 --fsdp_text_encoder --fsdp_dit \
    --enable_teacache --teacache_threshold=0.10 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=5 \
    --sample_size 720 1280 --num_inference_steps=40 

export DIT_EXCEL_ROW=4 VAE_EXCEL_ROW=4 TOTAL_EXCEL_ROW=4
torchrun --nproc-per-node=8 examples/wan2.2/predict_t2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.2-T2V-A14B" \
    --GPU_memory_mode="model_full_load" --ulysses_degree=4 --ring_degree=2 --fsdp_text_encoder --fsdp_dit \
    --enable_teacache --teacache_threshold=0.10 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=5 \
    --sample_size 720 1280 --num_inference_steps=40 

# 14B 480P
export DIT_EXCEL_ROW=5 VAE_EXCEL_ROW=5 TOTAL_EXCEL_ROW=5
python examples/wan2.2/predict_t2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.2-T2V-A14B" \
    --GPU_memory_mode="model_full_load" --ulysses_degree=1 --ring_degree=1 --fsdp_text_encoder --fsdp_dit \
    --enable_teacache --teacache_threshold=0.10 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=3 \
    --sample_size 480 832 --num_inference_steps=40 

export DIT_EXCEL_ROW=6 VAE_EXCEL_ROW=6 TOTAL_EXCEL_ROW=6
torchrun --nproc-per-node=2 examples/wan2.2/predict_t2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.2-T2V-A14B" \
    --GPU_memory_mode="model_full_load" --ulysses_degree=2 --ring_degree=1 --fsdp_text_encoder --fsdp_dit \
    --enable_teacache --teacache_threshold=0.10 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=3 \
    --sample_size 480 832 --num_inference_steps=40 

export DIT_EXCEL_ROW=7 VAE_EXCEL_ROW=7 TOTAL_EXCEL_ROW=7
torchrun --nproc-per-node=4 examples/wan2.2/predict_t2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.2-T2V-A14B" \
    --GPU_memory_mode="model_full_load" --ulysses_degree=4 --ring_degree=1 --fsdp_text_encoder --fsdp_dit \
    --enable_teacache --teacache_threshold=0.10 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=3 \
    --sample_size 480 832 --num_inference_steps=40 

export DIT_EXCEL_ROW=8 VAE_EXCEL_ROW=8 TOTAL_EXCEL_ROW=8
torchrun --nproc-per-node=8 examples/wan2.2/predict_t2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.2-T2V-A14B" \
    --GPU_memory_mode="model_full_load" --ulysses_degree=4 --ring_degree=2 --fsdp_text_encoder --fsdp_dit \
    --enable_teacache --teacache_threshold=0.10 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=3 \
    --sample_size 480 832 --num_inference_steps=40 
