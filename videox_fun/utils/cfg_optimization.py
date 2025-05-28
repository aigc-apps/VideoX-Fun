import numpy as np
import torch


def cfg_skip():
    def decorator(func):
        def wrapper(self, x, *args, **kwargs):
            bs = len(x)
            if bs >= 2 and self.cfg_skip_ratio is not None and self.current_steps >= self.num_inference_steps * (1 - self.cfg_skip_ratio):
                bs_half = int(bs // 2)

                new_x = x[bs_half:]
                
                new_args = []
                for arg in args:
                    if isinstance(arg, (torch.Tensor, list, tuple, np.ndarray)):
                        new_args.append(arg[bs_half:])
                    else:
                        new_args.append(arg)

                new_kwargs = {}
                for key, content in kwargs.items():
                    if isinstance(content, (torch.Tensor, list, tuple, np.ndarray)):
                        new_kwargs[key] = content[bs_half:]
                    else:
                        new_kwargs[key] = content
            else:
                new_x = x
                new_args = args
                new_kwargs = kwargs

            result = func(self, new_x, *new_args, **new_kwargs)

            if bs >= 2 and self.cfg_skip_ratio is not None and self.current_steps >= self.num_inference_steps * (1 - self.cfg_skip_ratio):
                result = torch.cat([result, result], dim=0)

            return result
        return wrapper
    return decorator