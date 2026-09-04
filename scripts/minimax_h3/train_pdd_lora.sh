# fl2va off a pre-encoded prompt cache (README_TRAIN_PDD_LORA.md §3.2, `--enable_preprocess_training`).
# For ref2va — direct load (Route A) or request cache (Route B) — see §3.2.1 and generate_ref2va_request_cache.sh.
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATA_DIR=""
export PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export VAL_PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_preprocess_training \
  --train_data_dir=$DATA_DIR \
  --train_data_meta=$PROMPT_CACHE_META \
  --val_data_meta=$VAL_PROMPT_CACHE_META \
  --video_sample_n_frames=124 \
  --fix_sample_size 768 1344 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --checkpointing_steps=200 \
  --learning_rate=1e-5 \
  --lora_learning_rate=1e-4 \
  --seed=43 \
  --output_dir="output_dir_minimax_h3_pdd_lora" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="no" \
  --adam_weight_decay=0.0 \
  --max_grad_norm=1.0 \
  --rank=64 \
  --network_alpha=64 \
  --low_vram \
  --target_name="to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear" \
  --train_mode="fl2va" \
  --pdd_num_steps=32 \
  --pdd_block_size=4 \
  --validation_steps=200
