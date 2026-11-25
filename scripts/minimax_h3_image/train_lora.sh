export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3-Image"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# `--target_name` is matched as a substring by the kohya network built here; with `--use_peft_lora` it is a peft
# module suffix instead, so the same list has to be spelled out in full (to_q,to_k,to_v,to_out.0,ff.net.2,...).
accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3_image/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --train_batch_size=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_image_lora" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --enable_bucket \
  --uniform_sampling \
  --rank=64 \
  --network_alpha=32 \
  --target_name="to_q,to_k,to_v,to_out,ff.net,adaln_proj,context_embedder,proj_in" \
  --low_vram \
  --train_mode="t2i"

# The default is the shared single projection.

# Image editing: swap the dataset pair and `--train_mode` above for the edit meta of
# datasets/X-Fun-Images-Edit-Demo/. Validation is optional (README 3.4); under edit mode pair each
# validation prompt with a reference through `--validation_image_paths`.
#
# Resumable checkpoints: add `--save_state` to write an accelerator state folder (adapters + optimizer + bucket
# position) instead of one loose `checkpoint-{step}.safetensors`; ZeRO-3 / FSDP-3 runs switch to it on their own.
# FSDP-3 state also includes the sharded transformer weights, so the checkpoint can be much larger than the LoRA.
