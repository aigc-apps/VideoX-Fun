export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# Download vidprom_filtered_extended.txt from:
# https://huggingface.co/gdhe17/Self-Forcing/blob/main/vidprom_filtered_extended.txt
accelerate launch --mixed_precision="bf16" scripts/wan2.1_self_forcing/generate_ode_pairs.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --video_sample_n_frames=81 \
  --height=480 \
  --width=832 \
  --guidance_scale=6.0 \
  --shift=8.0 \
  --num_inference_steps=48 \
  --caption_path="datasets/vidprom_filtered_extended.txt" \
  --output_folder="datasets/ode_pairs_output" \
  --sample_every_n_prompts=50