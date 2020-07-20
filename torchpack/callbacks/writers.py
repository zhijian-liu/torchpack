import os
from typing import List, Optional, Union

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from torchpack import distributed as dist
from torchpack.callbacks.callback import Callback
from torchpack.environ import get_run_dir
from torchpack.utils import fs, io
from torchpack.utils.logging import logger
from torchpack.utils.matching import NameMatcher
from torchpack.utils.typing import Trainer

__all__ = ['Writer', 'LoggingWriter', 'TFEventWriter', 'JSONWriter']


class Writer(Callback):
    """
    Base class for all summary writers.
    """
    def add_scalar(self, name: str,
                   scalar: Union[int, float, np.integer, np.floating]) -> None:
        if dist.is_master() or not self.master_only:
            self._add_scalar(name, scalar)

    def _add_scalar(self, name: str,
                    scalar: Union[int, float, np.integer, np.floating]
                    ) -> None:
        pass

    def add_image(self, name: str, tensor) -> None:
        if dist.is_master() or not self.master_only:
            self._add_image(name, tensor)

    def _add_image(self, name: str, tensor) -> None:
        pass


class LoggingWriter(Writer):
    """
    Write scalar summaries to logging.
    """
    master_only = True

    def __init__(self, scalars: Union[str, List[str]] = '*') -> None:
        self.matcher = NameMatcher(patterns=scalars)

    def _set_trainer(self, trainer: Trainer) -> None:
        self.scalars = dict()

    def _add_scalar(self, name: str,
                    scalar: Union[int, float, np.integer, np.floating]
                    ) -> None:
        self.scalars[name] = scalar

    def _trigger_epoch(self) -> None:
        self._trigger()

    def _trigger(self) -> None:
        texts = []
        for name, scalar in sorted(self.scalars.items()):
            if self.matcher.match(name):
                texts.append('[{}] = {:.5g}'.format(name, scalar))
        if texts:
            logger.info('\n+ '.join([''] + texts))
        self.scalars.clear()


class TFEventWriter(Writer):
    """
    Write summaries to TensorFlow event file.
    """
    master_only = True

    def __init__(self, *, save_dir: Optional[str] = None) -> None:
        if save_dir is None:
            save_dir = os.path.join(get_run_dir(), 'tensorboard')
        self.save_dir = fs.normpath(save_dir)

    def _set_trainer(self, trainer: Trainer) -> None:
        self.writer = SummaryWriter(self.save_dir)

    def _add_scalar(self, name: str,
                    scalar: Union[int, float, np.integer, np.floating]
                    ) -> None:
        self.writer.add_scalar(name, scalar, self.trainer.global_step)

    def _add_image(self, name: str, tensor) -> None:
        self.writer.add_image(name, tensor, self.trainer.global_step)

    def _after_train(self) -> None:
        self.writer.close()


class JSONWriter(Writer):
    """
    Write scalar summaries to JSON file.
    """
    def __init__(self, save_dir: Optional[str] = None) -> None:
        if save_dir is None:
            save_dir = os.path.join(get_run_dir(), 'summary')
        self.save_dir = fs.normpath(save_dir)
        self.save_fname = os.path.join(save_dir, 'scalars.json')

    def _set_trainer(self, trainer: Trainer) -> None:
        self.summaries = []

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        try:
            io.save(self.save_fname, self.summaries)
        except (OSError, IOError):
            logger.exception(
                'Error occurred when saving JSON file "{}".'.format(
                    self.save_fname))

    def _after_train(self):
        self._trigger()

    def _add_scalar(self, name: str,
                    scalar: Union[int, float, np.integer, np.floating]
                    ) -> None:
        self.summaries.append({
            'epoch_num': self.trainer.epoch_num,
            'global_step': self.trainer.global_step,
            'local_step': self.trainer.local_step,
            name: scalar
        })
