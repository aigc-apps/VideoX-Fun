import torch

# Copied from https://github.com/NVlabs/rcm/blob/main/rcm/utils/denoiser_scaling.py
class RectifiedFlow_TrigFlowWrapper:
    def __init__(self, sigma_data: float = 1.0, t_scaling_factor: float = 1.0):
        assert abs(sigma_data - 1.0) < 1e-6, "sigma_data must be 1.0 for RectifiedFlowScaling"
        self.t_scaling_factor = t_scaling_factor

    def __call__(self, trigflow_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trigflow_t = trigflow_t.to(torch.float64)
        c_skip = 1 / (torch.cos(trigflow_t) + torch.sin(trigflow_t))
        c_out = -1 * torch.sin(trigflow_t) / (torch.cos(trigflow_t) + torch.sin(trigflow_t))
        c_in = 1 / (torch.cos(trigflow_t) + torch.sin(trigflow_t))
        c_noise = (torch.sin(trigflow_t) / (torch.cos(trigflow_t) + torch.sin(trigflow_t))) * self.t_scaling_factor
        return c_skip, c_out, c_in, c_noise

# Sample timesteps
def sample_trigflow_timesteps(batch_size, device, P_mean=0.0, P_std=1.6):
    """Sample timesteps for training"""
    sigma = torch.randn(batch_size, device=device)
    sigma = (sigma * P_std + P_mean).exp()
    timesteps = torch.arctan(sigma)
    return timesteps