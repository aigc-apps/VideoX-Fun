export EXCEL_FILE="./speed.xlsx"

export DIT_EXCEL_COL=0 VAE_EXCEL_COL=1 TOTAL_EXCEL_COL=2

# 14B 720P
export DIT_EXCEL_ROW=9 VAE_EXCEL_ROW=9 TOTAL_EXCEL_ROW=9
python examples/wan2.1/predict_i2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P" \
    --GPU_memory_mode="model_full_load_and_qfloat8" --ulysses_degree=1 --ring_degree=1 --fsdp_text_encoder --compile_dit \
    --enable_teacache --teacache_threshold=0.30 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=5 \
    --sample_size 720 1280 --num_inference_steps=40 

export DIT_EXCEL_ROW=10 VAE_EXCEL_ROW=10 TOTAL_EXCEL_ROW=10
torchrun --nproc-per-node=2 examples/wan2.1/predict_i2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P" \
    --GPU_memory_mode="model_full_load_and_qfloat8" --ulysses_degree=2 --ring_degree=1 --fsdp_text_encoder --fsdp_dit \
    --enable_teacache --teacache_threshold=0.30 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=5 \
    --sample_size 720 1280 --num_inference_steps=40 

export DIT_EXCEL_ROW=11 VAE_EXCEL_ROW=11 TOTAL_EXCEL_ROW=11
torchrun --nproc-per-node=4 examples/wan2.1/predict_i2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P" \
    --GPU_memory_mode="model_full_load_and_qfloat8" --ulysses_degree=4 --ring_degree=1 --fsdp_text_encoder --fsdp_dit \
    --enable_teacache --teacache_threshold=0.30 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=5 \
    --sample_size 720 1280 --num_inference_steps=40 

export DIT_EXCEL_ROW=12 VAE_EXCEL_ROW=12 TOTAL_EXCEL_ROW=12
torchrun --nproc-per-node=8 examples/wan2.1/predict_i2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P" \
    --GPU_memory_mode="model_full_load_and_qfloat8" --ulysses_degree=8 --ring_degree=1 --fsdp_text_encoder --fsdp_dit \
    --enable_teacache --teacache_threshold=0.30 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=5 \
    --sample_size 720 1280 --num_inference_steps=40 

# 14B 480P
export DIT_EXCEL_ROW=13 VAE_EXCEL_ROW=13 TOTAL_EXCEL_ROW=13
python examples/wan2.1/predict_i2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P" \
    --GPU_memory_mode="model_full_load_and_qfloat8" --ulysses_degree=1 --ring_degree=1 --fsdp_text_encoder --compile_dit \
    --enable_teacache --teacache_threshold=0.30 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=3 \
    --sample_size 480 832 --num_inference_steps=40 

export DIT_EXCEL_ROW=14 VAE_EXCEL_ROW=14 TOTAL_EXCEL_ROW=14
torchrun --nproc-per-node=2 examples/wan2.1/predict_i2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P" \
    --GPU_memory_mode="model_full_load_and_qfloat8" --ulysses_degree=2 --ring_degree=1 --fsdp_text_encoder --fsdp_dit \
    --enable_teacache --teacache_threshold=0.30 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=3 \
    --sample_size 480 832 --num_inference_steps=40 

export DIT_EXCEL_ROW=15 VAE_EXCEL_ROW=15 TOTAL_EXCEL_ROW=15
torchrun --nproc-per-node=4 examples/wan2.1/predict_i2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P" \
    --GPU_memory_mode="model_full_load_and_qfloat8" --ulysses_degree=4 --ring_degree=1 --fsdp_text_encoder --fsdp_dit \
    --enable_teacache --teacache_threshold=0.30 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=3 \
    --sample_size 480 832 --num_inference_steps=40 

export DIT_EXCEL_ROW=16 VAE_EXCEL_ROW=16 TOTAL_EXCEL_ROW=16
torchrun --nproc-per-node=8 examples/wan2.1/predict_i2v_speed.py --model_name="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P" \
    --GPU_memory_mode="model_full_load_and_qfloat8" --ulysses_degree=8 --ring_degree=1 --fsdp_text_encoder --fsdp_dit \
    --enable_teacache --teacache_threshold=0.30 --num_skip_start_steps=2 --cfg_skip_ratio=0.25 --shift=3 \
    --sample_size 480 832 --num_inference_steps=40 
