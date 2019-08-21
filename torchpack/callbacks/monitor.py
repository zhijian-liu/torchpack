from collections import defaultdict

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
    def __init__(self, monitors):
        for monitor in monitors:
            assert isinstance(monitor, Monitor), type(monitor)
        self.monitors = monitors

    def set_trainer(self, trainer):
        self.trainer = trainer
        self._set_trainer(trainer)

    def _set_trainer(self, trainer):
        # TODO: keep `maxlen` scalars and images
        self.scalars = defaultdict(list)
        self.images = defaultdict(list)

    def add_scalar(self, name, scalar):
        self._add_scalar(name, scalar)
        for callback in self.monitors:
            callback.add_scalar(name, scalar)

    def _add_scalar(self, name, scalar):
        self.scalars[name].append((self.trainer.global_step, scalar))

    def add_image(self, name, tensor):
        self._add_image(name, tensor)
        for callback in self.monitors:
            callback.add_image(name, tensor)

    def _add_image(self, name, tensor):
        self.images[name].append((self.trainer.global_step, tensor))

    def get(self, name):
        return self.scalars.get(name)

    def keys(self):
        return self.scalars.keys()

    def __contains__(self, name):
        return name in self.scalars and self.scalars[name]

    def __getitem__(self, name):
        return self.scalars[name][-1]
