from collections import deque

import numpy as np
import torch

from .. import distributed as dist
from ..callbacks.writers import Writer

__all__ = ['Summary']

self.summary.add_image('image', image, max_to_keep=4)


class Summary:
    def __init__(self, monitors):
        for monitor in monitors:
            assert isinstance(monitor, Writer), type(monitor)
        self.writers = monitors
        self.summaries = dict()

    def set_trainer(self, trainer):
        self.trainer = trainer

    def add_scalar(self, name, scalar, *, max_to_keep=65536):
        if isinstance(scalar, np.integer):
            scalar = int(scalar)
        if isinstance(scalar, np.floating):
            scalar = float(scalar)
        assert isinstance(scalar, (int, float)), type(scalar)
        self._add_scalar(name, scalar, max_to_keep=max_to_keep)

    def _add_scalar(self, name, scalar, max_to_keep):
        if name not in self.summaries:
            self.summaries[name] = deque(maxlen=max_to_keep)
        self.summaries[name].append((self.trainer.global_step, scalar))
        for writer in self.writers:
            writer.add_scalar(name, scalar)

    def add_image(self, name, tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
        assert isinstance(tensor, np.ndarray), type(tensor)
        if tensor.ndim == 2:
            tensor = tensor[np.newaxis, ..., np.newaxis]
        elif tensor.ndim == 3:
            if tensor.shape[0] in [1, 3, 4]:
                tensor = np.transpose(tensor, (1, 2, 0))
            if tensor.shape[-1] in [1, 3, 4]:
                tensor = tensor[np.newaxis, ...]
        assert tensor.ndim == 4 and tensor.shape[-1] in [1, 3, 4], tensor.shape
        self._add_image(name, tensor)

    def _add_image(self, name, tensor):
        if name not in self.summaries:
            self.summaries[name] = deque(maxlen=16)
        self.summaries[name].append((self.trainer.global_step, tensor))
        for writer in self.writers:
            writer.add_image(name, tensor)

    def items(self):
        for name, summary in self.summaries.items():
            yield name, summary[-1]

    def keys(self):
        for key, value in self.items():
            yield key

    def values(self):
        for key, value in self.items():
            yield value

    def __contains__(self, name):
        return name in self.summaries

    def __getitem__(self, name):
        return self.summaries[name][-1]

    def get(self, name):
        return self.summaries[name]
