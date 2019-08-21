from collections import defaultdict, deque

import numpy as np
import torch

from torchpack.callbacks.callback import Callback

__all__ = ['Monitor', 'Monitors']


class Monitor(Callback):
    """
    Base class for all monitors.
    """

    master_only = True

    def add_scalar(self, name, scalar):
        if isinstance(scalar, np.integer):
            scalar = int(scalar)
        if isinstance(scalar, np.floating):
            scalar = float(scalar)
        assert isinstance(scalar, (int, float)), type(scalar)
        self._add_scalar(name, scalar)

    def _add_scalar(self, name, scalar):
        pass

    def add_image(self, name, tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
        assert isinstance(tensor, np.ndarray), type(tensor)
        if tensor.ndim == 2:
            tensor = tensor[np.newaxis, :, :, np.newaxis]
        elif tensor.ndim == 3:
            if tensor.shape[0] in [1, 3, 4]:
                tensor = np.transpose(tensor, (1, 2, 0))
            if tensor.shape[-1] in [1, 3, 4]:
                tensor = tensor[np.newaxis, ...]
        assert tensor.ndim == 4 and tensor.shape[-1] in [1, 3, 4], tensor.shape
        self._add_image(name, tensor)

    def _add_image(self, name, tensor):
        pass


class Monitors(object):
    def __init__(self):
        self.scalars = defaultdict(list)

    def set_trainer(self, trainer):
        self.trainer = trainer

    def add_scalar(self, name, scalar):
        for callback in self.trainer.callbacks:
            if isinstance(callback, Monitor):
                callback.add_scalar(name, scalar)
        self._add_scalar(name, scalar)

    def _add_scalar(self, name, scalar):
        self.scalars[name].append((self.trainer.global_step, scalar))

    def add_image(self, name, tensor):
        for callback in self.trainer.callbacks:
            if isinstance(callback, Monitor):
                callback.add_image(name, tensor)
        self._add_image(name, tensor)

    def _add_image(self, name, tensor):
        pass

    def get(self, name):
        return self.scalars.get(name)

    def keys(self):
        return self.scalars.keys()

    def __contains__(self, name):
        return name in self.scalars and self.scalars[name]

    def __getitem__(self, name):
        return self.scalars[name][-1]
