import functools

import numpy as np
import torch
import torch.nn as nn

from .handlers import flops_handlers, params_handlers

__all__ = ['profile', 'profile_params', 'profile_flops']


def profile(model, *inputs, handlers):
    stats = {}

    def hook(module, inputs, outputs, name):
        # rename module for data-parallel model
        if isinstance(model, nn.DataParallel):
            name = name + '/' + str(outputs.device)

        for types, handler in handlers:
            if isinstance(module, types):
                stats[name] = handler(module, inputs, outputs)
                return

        # todo: issue a warning (for leaf modules)
        return

    handles = []
    for name, module in model.named_modules():
        handle = module.register_forward_hook(functools.partial(hook, name=name))
        handles.append(handle)

    with torch.no_grad():
        model(*inputs)

    for handle in handles:
        handle.remove()

    return stats


def profile_params(model, *inputs, handlers=None):
    if handlers is None:
        handlers = params_handlers
    else:
        handlers += params_handlers

    stats = profile(model, *inputs, handlers=handlers)
    return np.sum(list(stats.values())), stats


def profile_flops(model, *inputs, handlers=None):
    if handlers is None:
        handlers = flops_handlers
    else:
        handlers += flops_handlers

    stats = profile(model, *inputs, handlers=handlers)
    return np.sum(list(stats.values())), stats
