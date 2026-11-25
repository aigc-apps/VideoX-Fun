export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3-Image"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# The 33B backbone is sharded across the GPUs, so the run starts on 8x80GB and scales down by adding
# --gradient_checkpointing_save_on_cpu / --low_vram rather than by shrinking the model.
accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3_image/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --train_batch_size=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_image" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --train_mode="t2i" \
  --trainable_modules "."

# The default is the shared single projection.

# Image editing: the meta of every sample carries a `source_file_path`. Swap the dataset pair and
# `--train_mode` above for:
#   --train_mode="edit" \
#   --train_data_dir=datasets/X-Fun-Images-Edit-Demo/ \
#   --train_data_meta=datasets/X-Fun-Images-Edit-Demo/metadata_add_width_height.json \
# Validation is optional (README 3.4); under edit mode pair each validation prompt with a reference:
#   --validation_image_paths=datasets/X-Fun-Images-Edit-Demo/source/souce-20260421_140343_099985_000/10_00015405.png \
