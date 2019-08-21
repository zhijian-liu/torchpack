from collections import deque

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
        self.summaries = dict()

    def set_trainer(self, trainer):
        self.trainer = trainer

    def add_scalar(self, name, scalar):
        self._add_scalar(name, scalar)
        for monitor in self.monitors:
            monitor.add_scalar(name, scalar)

    def _add_scalar(self, name, scalar):
        if name not in self.summaries:
            self.summaries[name] = deque(maxlen=65536)
        self.summaries[name].append((scalar, self.trainer.global_step))

    def add_image(self, name, tensor):
        self._add_image(name, tensor)
        for monitor in self.monitors:
            monitor.add_image(name, tensor)

    def _add_image(self, name, tensor):
        if name not in self.summaries:
            self.summaries[name] = deque(maxlen=4)
        self.summaries[name].append((tensor, self.trainer.global_step))

    def get(self, name):
        return self.summaries.get(name)

    def __contains__(self, name):
        return name in self.summaries

    def __getitem__(self, name):
        return self.summaries[name][-1][0]
