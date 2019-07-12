import numpy as np
import torch.nn as nn

__all__ = ['params_handlers']


def module(module, inputs, outputs):
    return np.sum(param.numel() for param in module.parameters())


params_handlers = [
    (nn.Module, module)
]
