import typing
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch

from torchpack.callbacks import SummaryWriter

if typing.TYPE_CHECKING:
    from torchpack.train import Trainer

__all__ = ['Summary']


class Summary:

    def __init__(self) -> None:
        self.history: Dict[str, Deque[Tuple[int, Any]]] = defaultdict(deque)

    def set_trainer(self, trainer: 'Trainer') -> None:
        self.trainer = trainer
        self._set_trainer(trainer)

    def _set_trainer(self, trainer: 'Trainer') -> None:
        self.writers = []
        for callback in trainer.callbacks:
            if isinstance(callback, SummaryWriter):
                self.writers.append(callback)

    def add_scalar(
        self,
        name: str,
        scalar: Union[int, float, np.integer, np.floating],
        *,
        max_to_keep: Optional[int] = None,
    ) -> None:
        if isinstance(scalar, np.integer):
            scalar = int(scalar)
        if isinstance(scalar, np.floating):
            scalar = float(scalar)
        assert isinstance(scalar, (int, float)), type(scalar)
        self._add_scalar(name, scalar, max_to_keep=max_to_keep)

    def _add_scalar(
        self,
        name: str,
        scalar: Union[int, float],
        *,
        max_to_keep: Optional[int],
    ) -> None:
        self.history[name].append((self.trainer.global_step, scalar))
        while max_to_keep is not None and \
                len(self.history[name]) > max_to_keep:
            self.history[name].popleft()
        for writer in self.writers:
            writer.add_scalar(name, scalar)

    def add_image(
        self,
        name: str,
        tensor: Union[np.ndarray, torch.Tensor],
        *,
        max_to_keep: Optional[int] = None,
    ) -> None:
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        assert isinstance(tensor, np.ndarray), type(tensor)
        if tensor.ndim == 2:
            tensor = tensor[np.newaxis, ...]
        elif tensor.ndim == 3 and tensor.shape[-1] in [1, 3, 4]:
            tensor = tensor.transpose(2, 0, 1)
        assert tensor.ndim == 3 and tensor.shape[0] in [1, 3, 4], tensor.shape
        self._add_image(name, tensor, max_to_keep=max_to_keep)

    def _add_image(
        self,
        name: str,
        tensor: np.ndarray,
        *,
        max_to_keep: Optional[int],
    ) -> None:
        self.history[name].append((self.trainer.global_step, tensor))
        while max_to_keep is not None and \
                len(self.history[name]) > max_to_keep:
            self.history[name].popleft()
        for writer in self.writers:
            writer.add_image(name, tensor)

    def keys(self) -> Iterable[str]:
        yield from self.history.keys()

    def values(self) -> Iterable[Deque[Tuple[int, Any]]]:
        yield from self.history.values()

    def items(self) -> Iterable[Tuple[str, Deque[Tuple[int, Any]]]]:
        yield from self.history.items()

    def __contains__(self, key: str) -> bool:
        return key in self.history

    def __getitem__(self, key: str) -> Deque[Tuple[int, Any]]:
        return self.history[key]
