import time
from collections import deque

import numpy as np
import tqdm
from tensorpack.utils.utils import get_tqdm_kwargs
from tensorpack.utils.utils import humanize_time_delta

from torchpack.callbacks.callback import Callback
from torchpack.callbacks.monitor import Monitor
from torchpack.utils.logging import logger
from torchpack.utils.matching import IENameMatcher

__all__ = ['ProgressBar', 'EstimatedTimeLeft']


class ProgressBar(Monitor):
    """
    A progress bar based on `tqdm`.
    This callback is one of the :func:`DEFAULT_CALLBACKS()`.
    """

    master_only = True

    def __init__(self, include='*', exclude=None, tqdm_kwargs=None):
        self.matcher = IENameMatcher(include, exclude)
        self.tqdm_kwargs = tqdm_kwargs or get_tqdm_kwargs()

    def _before_epoch(self):
        self.pbar = tqdm.trange(self.trainer.steps_per_epoch, **self.tqdm_kwargs)
        self.scalars = dict()

    def _trigger_step(self):
        texts = []
        for name, scalar in sorted(self.scalars.items()):
            if self.matcher.match(name):
                texts.append('[{}] = {:.4f}'.format(name, scalar))
        if texts:
            self.pbar.set_description(', '.join(texts))
        self.pbar.update()

    def _after_epoch(self):
        self.pbar.close()

    def _add_scalar(self, name, scalar):
        if hasattr(self, 'scalars'):
            self.scalars[name] = scalar


class EstimatedTimeLeft(Callback):
    """
    Estimate the time left until completion.
    This callback is one of the :func:`DEFAULT_CALLBACKS()`.
    """

    master_only = True

    def __init__(self, last_k_epochs=5, median=True):
        self.last_k_epochs = last_k_epochs
        self.median = median

    def _before_train(self):
        self.times = deque(maxlen=self.last_k_epochs)
        self.last_time = time.time()

    def _trigger_epoch(self):
        if self.trainer.epoch_num == self.trainer.max_epoch:
            return

        self.times.append(time.time() - self.last_time)
        self.last_time = time.time()

        epoch_time = np.median(self.times) if self.median else np.mean(self.times)
        time_left = (self.trainer.max_epoch - self.trainer.epoch_num) * epoch_time
        logger.info('Estimated time left: {}.'.format(humanize_time_delta(time_left)))
