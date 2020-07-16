import time
from collections import deque
from typing import List, Union

import numpy as np
import tqdm

import torchpack.utils.humanize as humanize
from torchpack.callbacks.callback import Callback
from torchpack.utils.logging import logger
from torchpack.utils.matching import NameMatcher

__all__ = ['ProgressBar', 'EstimatedTimeLeft']


class ProgressBar(Callback):
    """
    A progress bar based on `tqdm`.
    """
    master_only: bool = True

    def __init__(self, scalars: Union[str, List[str]] = '*') -> None:
        self.matcher = NameMatcher(patterns=scalars)

    def _before_epoch(self) -> None:
        self.pbar = tqdm.trange(self.trainer.steps_per_epoch, ncols=0)

    def _trigger_step(self) -> None:
        texts = []
        for name, (step, scalar) in sorted(self.trainer.summary.items()):
            if step == self.trainer.global_step and isinstance(
                    scalar, (int, float)) and self.matcher.match(name):
                texts.append('[{}] = {:.3g}'.format(name, scalar))
        if texts:
            self.pbar.set_description(', '.join(texts))
        self.pbar.update()

    def _after_epoch(self) -> None:
        self.pbar.close()


class EstimatedTimeLeft(Callback):
    """
    Estimate the time left until completion.
    """
    master_only: bool = True

    def __init__(self, *, last_k_epochs: int = 8) -> None:
        self.last_k_epochs = last_k_epochs

    def _before_train(self) -> None:
        self.times = deque(maxlen=self.last_k_epochs)
        self.last_time = time.time()

    def _trigger_epoch(self) -> None:
        if self.trainer.epoch_num < self.trainer.max_epoch:
            self.times.append(time.time() - self.last_time)
            self.last_time = time.time()

            estimated_time = (self.trainer.max_epoch -
                              self.trainer.epoch_num) * np.mean(self.times)
            logger.info('Estimated time left: {}.'.format(
                humanize.naturaldelta(estimated_time)))
