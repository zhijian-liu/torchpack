import functools
import warnings

import numpy as np
import torch
import torch.nn as nn

from .default_handlers import default_flops_handlers, default_params_handlers

__all__ = ['profile', 'profile_flops', 'profile_params']


def profile(model, *inputs, handlers):
    stats = {}

    def hook_stats(module, inputs, outputs, handler, name):
        stats[name] = handler(module, inputs, outputs)

    # quick fix for data-parallel model
    if isinstance(model, nn.DataParallel):
        model = model.module

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
                warnings.warn('missing handler for {}'.format(type(module)), UserWarning)

    with torch.no_grad():
        model(*inputs)

    for hook in hooks:
        hook.remove()

    # filter out nones from stats
    return {k: v for k, v in stats.items() if v is not None}


def profile_flops(model, *inputs, handlers=None):
    if handlers is None:
        handlers = default_flops_handlers
    else:
        handlers += default_flops_handlers

    stats = profile(model, *inputs, handlers=handlers)
    return np.sum(list(stats.values())), stats


def profile_params(model, *inputs, handlers=None):
    if handlers is None:
        handlers = default_params_handlers
    else:
        handlers += default_params_handlers

    stats = profile(model, *inputs, handlers=handlers)
    return np.sum(list(stats.values())), stats
