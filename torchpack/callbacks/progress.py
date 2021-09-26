import time
from collections import deque
from typing import Deque, List, Union

import numpy as np

from torchpack.utils import humanize, tqdm
from torchpack.utils.logging import logger
from torchpack.utils.matching import NameMatcher

from .callback import Callback

__all__ = ['ProgressBar', 'EstimatedTimeLeft']


class ProgressBar(Callback):
    """A progress bar based on `tqdm`."""

    master_only: bool = True

    def __init__(self, scalars: Union[str, List[str]] = '*') -> None:
        self.matcher = NameMatcher(patterns=scalars)

    def _before_epoch(self) -> None:
        self.pbar = tqdm.trange(self.trainer.steps_per_epoch)

    def _trigger_step(self) -> None:
        texts: List[str] = []
        for name in sorted(self.trainer.summary.keys()):
            step, scalar = self.trainer.summary[name][-1]
            if self.matcher.match(name) and step == self.trainer.global_step and \
                    isinstance(scalar, (int, float)):
                texts.append(f'[{name}] = {scalar:.3g}')
        if texts:
            self.pbar.set_description(', '.join(texts))
        self.pbar.update()

    def _after_epoch(self) -> None:
        self.pbar.close()


class EstimatedTimeLeft(Callback):
    """Estimate the time left until completion."""

    master_only: bool = True

    def __init__(self, *, last_k_epochs: int = 8) -> None:
        self.last_k_epochs = last_k_epochs

    def _before_train(self) -> None:
        self.times: Deque[float] = deque(maxlen=self.last_k_epochs)
        self.last_time = time.perf_counter()

    def _trigger_epoch(self) -> None:
        if self.trainer.epoch_num < self.trainer.num_epochs:
            self.times.append(time.perf_counter() - self.last_time)
            self.last_time = time.perf_counter()

            estimated_time = (self.trainer.num_epochs
                              - self.trainer.epoch_num) * np.mean(self.times)
            logger.info('Estimated time left: {}.'.format(
                humanize.naturaldelta(estimated_time)))
