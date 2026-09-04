export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_META="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
export CACHE_ROOT="datasets/minimax_h3_pdd_prompt_cache"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/minimax_h3/generate_prompt_cache.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_meta=$DATASET_META \
  --output_folder=$CACHE_ROOT
