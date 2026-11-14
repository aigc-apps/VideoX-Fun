# Reuse the module loaded during package initialization without importing the
# optional utilities eagerly. The normal import path is registered only when
# this parent package is initialized.
import sys

_training_module = getattr(sys.modules[__name__.rpartition(".")[0]], "_training_module", None)
if _training_module is not None:
    sys.modules.setdefault(f"{__name__}.training_utils", _training_module)

from .cfg_optimization import cfg_skip
from .discrete_sampler import DiscreteSampling
from .flex_chunking import (broadcast_chunk_sizes, build_pyramid_partitions,
                            chunk_boundaries, chunk_ends_tensor,
                            chunk_sizes_to_block_kwargs, normalize_chunk_spec,
                            refine_partition, sample_flex_level0,
                            sample_flexible_chunks, uniform_chunks,
                            validate_nested_partitions)
from .fm_solvers import FlowDPMSolverMultistepScheduler
from .fm_solvers_minimax_h3 import MiniMaxH3Scheduler
from .fm_solvers_unipc import FlowUniPCMultistepScheduler
from .fp4_optimization import (convert_float4_weight_dtype_wrapper,
                               convert_model_weight_to_float4,
                               dequantize_weight_from_float4,
                               quantize_weight_to_float4)
from .fp8_optimization import (autocast_model_forward,
                               convert_model_weight_to_float8,
                               convert_weight_dtype_wrapper,
                               replace_parameters_by_name)
from .fsdp_ema import FSDPEMA, LORAFSDPEMA
from .fsdp_utils import checkpoint_fsdp_forward
from .gpu_memory_mode import (GPU_MEMORY_MODES, QUANT_MODES,
                              apply_gpu_memory_mode, register_quant_mode,
                              split_gpu_memory_mode)
from .group_offload import (register_auto_device_hook,
                            safe_enable_group_offload,
                            safe_remove_group_offloading)
from .lora_utils import (LoadedTaomateH3LoRA, TaomateH3LoRACheckpointError,
                         canonical_taomate_h3_lora_targets,
                         convert_peft_lora_to_kohya_lora,
                         convert_taomate_h3_adapter, create_network,
                         load_taomate_h3_adapter, merge_lora,
                         official_to_kohya_lora_state_dict, unmerge_lora)
from .lora_utils_pdd import (PDD_EMA_WEIGHTS_NAME,
                             PDD_LEGACY_LIVE_WEIGHTS_NAME, PDD_WEIGHTS_NAME,
                             PDDLoRALinear, PDDParallelHead, add_pdd_lora,
                             pdd_sampling_plan, pdd_state_dict,
                             pdd_teacher_mode, pdd_time_grid,
                             pdd_training_plan)
from .perf_metrics import install as install_perf_metrics
from .perf_metrics import install_training as install_perf_training
from .perf_metrics import instrument_pipeline
from .sd3_sde_with_logprob import sde_step_with_logprob
from .tqdm_bar import PauseAwareTqdm
from .training_utils import (add_training_precision_args,
                             apply_trainable_module_learning_rates,
                             build_trainable_optimizer_groups,
                             configure_gradient_checkpointing_save_on_cpu,
                             configure_training_precision, create_optimizer,
                             initialize_training_seed,
                             parse_trainable_module_learning_rates)
from .trigflow_sampler import (RectifiedFlow_TrigFlowWrapper,
                               sample_trigflow_timesteps)
from .utils import (SegmentVideoSaver, StreamVideoSaver, calculate_dimensions,
                    filter_kwargs, get_autocast_dtype, get_image,
                    get_image_latent, get_image_to_video_latent,
                    get_video_to_video_latent, merge_video_audio,
                    save_videos_grid, save_videos_with_audio_grid)
from .utils_yolo import ObjectDetector, ObjectInstanceDetector
