import numpy as np
import torch


def get_teacache_coefficients(model_name):
    if "wan2.1-t2v-1.3b" in model_name.lower() or "wan2.1-fun-1.3b" in model_name.lower():
        return [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02]
    elif "wan2.1-t2v-14b" in model_name.lower():
        return [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01]
    elif "wan2.1-i2v-14b-480p" in model_name.lower():
        return [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01]
    elif "wan2.1-i2v-14b-720p" in model_name.lower() or "wan2.1-fun-14b" in model_name.lower():
        return [8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02]
    else:
        print(f"The model {model_name} is not supported by TeaCache.")
        return None


class TeaCache():
    """
    Timestep Embedding Aware Cache, a training-free caching approach that estimates and leverages
    the fluctuating differences among model outputs across timesteps, thereby accelerating the inference.
    Please refer to:
    1. https://github.com/ali-vilab/TeaCache.
    2. Liu, Feng, et al. "Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model." arXiv preprint arXiv:2411.19108 (2024).
    """
    def __init__(
        self,
        coefficients: list[float],
        num_steps: int,
        rel_l1_thresh: float = 0.0,
        num_skip_start_steps: int = 0,
        offload: bool = True,
    ):
        if num_steps < 1:
            raise ValueError(f"`num_steps` must be greater than 0 but is {num_steps}.")
        if rel_l1_thresh < 0:
            raise ValueError(f"`rel_l1_thresh` must be greater than or equal to 0 but is {rel_l1_thresh}.")
        if num_skip_start_steps < 0 or num_skip_start_steps > num_steps:
            raise ValueError(
                "`num_skip_start_steps` must be great than or equal to 0 and "
                f"less than or equal to `num_steps={num_steps}` but is {num_skip_start_steps}."
            )
        self.coefficients = coefficients
        self.num_steps = num_steps
        self.rel_l1_thresh = rel_l1_thresh
        self.num_skip_start_steps = num_skip_start_steps
        self.offload = offload
        self.rescale_func = np.poly1d(self.coefficients)

        self.cnt = 0
        self.should_calc = True
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        # Some pipelines concatenate the unconditional and text guide in forward.
        self.previous_residual = None
        # Some pipelines perform forward propagation separately on the unconditional and text guide.
        self.previous_residual_cond = None
        self.previous_residual_uncond = None

    @staticmethod
    def compute_rel_l1_distance(prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        rel_l1_distance = (torch.abs(cur - prev).mean()) / torch.abs(prev).mean()

        return rel_l1_distance.cpu().item()

    def reset(self):
        self.cnt = 0
        self.should_calc = True
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.previous_residual_cond = None
        self.previous_residual_uncond = None