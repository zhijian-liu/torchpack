import json
import os
import typing
from typing import Dict, List, Optional, Union

import numpy as np

from torchpack.environ import get_run_dir
from torchpack.utils import fs
from torchpack.utils.logging import logger
from torchpack.utils.matching import NameMatcher

from .callback import Callback

if typing.TYPE_CHECKING:
    from torchpack.train import Trainer

__all__ = ['SummaryWriter', 'ConsoleWriter', 'TFEventWriter', 'JSONLWriter']


class SummaryWriter(Callback):
    """Base class for all summary writers."""

    master_only: bool = True

    def add_scalar(self, name: str, scalar: Union[int, float]) -> None:
        if self.enabled:
            self._add_scalar(name, scalar)

    def _add_scalar(self, name: str, scalar: Union[int, float]) -> None:
        pass

    def add_image(self, name: str, tensor: np.ndarray) -> None:
        if self.enabled:
            self._add_image(name, tensor)

    def _add_image(self, name: str, tensor: np.ndarray) -> None:
        pass


class ConsoleWriter(SummaryWriter):
    """Write scalar summaries to console (and logger)."""

    def __init__(self, scalars: Union[str, List[str]] = '*') -> None:
        self.matcher = NameMatcher(patterns=scalars)

    def _set_trainer(self, trainer: 'Trainer') -> None:
        self.scalars: Dict[str, Union[int, float]] = {}

    def _add_scalar(self, name: str, scalar: Union[int, float]) -> None:
        self.scalars[name] = scalar

    def _trigger_epoch(self) -> None:
        self._trigger()

    def _trigger(self) -> None:
        texts = []
        for name, scalar in sorted(self.scalars.items()):
            if self.matcher.match(name):
                texts.append(f'[{name}] = {scalar:.5g}')
        if texts:
            logger.info('\n+ '.join([''] + texts))
        self.scalars.clear()


class TFEventWriter(SummaryWriter):
    """Write summaries to TensorFlow event file."""

    def __init__(self, *, save_dir: Optional[str] = None) -> None:
        if save_dir is None:
            save_dir = os.path.join(get_run_dir(), 'tensorboard')
        self.save_dir = fs.normpath(save_dir)

    def _set_trainer(self, trainer: 'Trainer') -> None:
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.save_dir)

    def _add_scalar(self, name: str, scalar: Union[int, float]) -> None:
        self.writer.add_scalar(name, scalar, self.trainer.global_step)

    def _add_image(self, name: str, tensor: np.ndarray) -> None:
        self.writer.add_image(name, tensor, self.trainer.global_step)

    def _after_train(self) -> None:
        self.writer.close()


class JSONLWriter(SummaryWriter):
    """Write scalar summaries to JSONL file."""

    def __init__(self, save_dir: Optional[str] = None) -> None:
        if save_dir is None:
            save_dir = os.path.join(get_run_dir(), 'summary')
        self.save_dir = fs.normpath(save_dir)

    def _set_trainer(self, trainer: 'Trainer') -> None:
        self.scalars: Dict[str, Union[int, float]] = {}
        fs.makedir(self.save_dir)
        self.file = open(os.path.join(self.save_dir, 'scalars.jsonl'), 'a')

    def _add_scalar(self, name: str, scalar: Union[int, float]) -> None:
        self.scalars[name] = scalar

    def _trigger_step(self) -> None:
        self.trigger()

    def _trigger_epoch(self) -> None:
        self.trigger()

    def _trigger(self) -> None:
        if self.scalars:
            summary: Dict[str, Union[int, float]] = {
                'epoch_num': self.trainer.epoch_num,
                'local_step': self.trainer.local_step,
                'global_step': self.trainer.global_step,
                **self.scalars
            }
            self.scalars.clear()
            self.file.write(json.dumps(summary) + '\n')
            self.file.flush()

    def _after_train(self) -> None:
        self.file.close()
