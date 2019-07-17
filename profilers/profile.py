import functools
import warnings

import numpy as np
import torch
import torch.nn as nn

from .default_handlers import default_flops_handlers, default_params_handlers

__all__ = ['profile', 'profile_params', 'profile_flops']


def profile(model, *inputs, handlers):
    stats = {}

    def hook_stats(module, inputs, outputs, handler, name):
        if isinstance(model, nn.DataParallel):
            name += '/cuda:' + str(torch.cuda.current_device())
        stats[name] = handler(module, inputs, outputs)

    hooks = []
    for name, module in model.named_modules():
        for types, handler in handlers:
            if isinstance(module, types):
                if handler is not None:
                    hook = functools.partial(hook_stats, handler=handler, name=name)
                    hooks.append(module.register_forward_hook(hook))
                break
        else:
            if not list(module.children()):
                warnings.warn('no handler for {}'.format(type(module)))

    with torch.no_grad():
        model(*inputs)

    for hook in hooks:
        hook.remove()

    return stats


def profile_params(model, *inputs, handlers=None):
    if handlers is None:
        handlers = default_params_handlers
    else:
        handlers += default_params_handlers

    stats = profile(model, *inputs, handlers=handlers)
    return np.sum(list(stats.values())), stats


def profile_flops(model, *inputs, handlers=None):
    if handlers is None:
        handlers = default_flops_handlers
    else:
        handlers += default_flops_handlers

    stats = profile(model, *inputs, handlers=handlers)
    return np.sum(list(stats.values())), stats
