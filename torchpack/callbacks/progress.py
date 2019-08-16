import time
from collections import deque

import numpy as np
import tqdm
from tensorpack.utils.utils import get_tqdm_kwargs
from tensorpack.utils.utils import humanize_time_delta

from torchpack.callbacks.callback import Callback
from torchpack.utils.logging import logger

__all__ = ['ProgressBar', 'EstimatedTimeLeft']


class ProgressBar(Callback):
    """
    A progress bar based on tqdm.
    This callback is one of the :func:`DEFAULT_CALLBACKS()`.
    """

    chief_only = True

    def __init__(self, names=None, tqdm_args=None):
        """
        Args:
            names(list): list of string, the names of the tensors to monitor
                on the progress bar.
        """
        self.tqdm_args = tqdm_args
        self.pbar = None

    def _before_train(self):
        self.last_updated = self.trainer.local_step
        self.total = self.trainer.steps_per_epoch
        self.tqdm_args = self.tqdm_args or get_tqdm_kwargs(leave=True)
        # self._tqdm_args['bar_format'] = self._tqdm_args['bar_format'] + "{postfix} "

    def _before_epoch(self):
        self.pbar = tqdm.trange(self.total, **self.tqdm_args)

    def _before_step(self, *args, **kwargs):
        if self.last_updated != self.trainer.local_step:
            self.last_updated = self.trainer.local_step

    def _after_step(self, *args, **kwargs):
        # self._bar.set_postfix(dict(loss=1))
        pass

    def _trigger_step(self):
        self.pbar.update()

    def _after_epoch(self):
        self.pbar.close()

    def _after_train(self):
        if self.pbar:
            self.pbar.close()


class EstimatedTimeLeft(Callback):
    """
    Estimate the time left until completion of training.
    This callback is one of the :func:`DEFAULT_CALLBACKS()`.
    """

    chief_only = True

    def __init__(self, last_k_epochs=5, estimator=np.mean):
        self.estimator = estimator
        self.durations = deque(maxlen=last_k_epochs)

    def _before_train(self):
        self.last_time = time.time()

    def _trigger_epoch(self):
        if self.trainer.epoch_num == self.trainer.max_epoch:
            return

        self.durations.append(time.time() - self.last_time)
        self.last_time = time.time()

        estimated_time = (self.trainer.max_epoch - self.trainer.epoch_num) * self.estimator(self.durations)
        logger.info('Estimated time left: {}.'.format(humanize_time_delta(estimated_time)))
