import numpy as np
import torch.nn as nn

__all__ = ['default_params_handlers']


def module(module, inputs, outputs):
    # only leaf modules should be profiled
    if list(module.children()):
        return None

    return np.sum(param.numel() for param in module.parameters())


default_params_handlers = [
    (nn.Module, module)
]
