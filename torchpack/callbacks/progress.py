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
    """ A progress bar based on tqdm.
    This callback is one of the :func:`DEFAULT_CALLBACKS()`.
    """

    chief_only = True

    def __init__(self, names=None):
        """
        Args:
            names(list): list of string, the names of the tensors to monitor
                on the progress bar.
        """
        super().__init__()
        # self._names = [get_op_tensor_name(n)[1] for n in names]
        # self._tags = [get_op_tensor_name(n)[0].split("/")[-1] for n in names]
        self.pbar = None

    def before_train(self):
        self.last_updated = self.trainer.local_step

        self.total = self.trainer.steps_per_epoch
        self.tqdm_args = get_tqdm_kwargs(leave=True)
        # self._tqdm_args['bar_format'] = self._tqdm_args['bar_format'] + "{postfix} "

    def before_epoch(self):
        self.pbar = tqdm.trange(self.total, **self.tqdm_args)

    def after_epoch(self):
        self.pbar.close()

    def before_step(self, *args, **kwargs):
        # update progress bar when local step changed (one step is finished)
        if self.last_updated != self.trainer.local_step:
            self.last_updated = self.trainer.local_step

    def after_step(self, *args, **kwargs):
        # self._bar.set_postfix(dict(loss=1))
        pass

    def trigger_step(self):
        self.trigger()

    def trigger(self):
        self.pbar.update()

    def after_train(self):
        # training may get killed before the first step
        if self.pbar:
            self.pbar.close()


class EstimatedTimeLeft(Callback):
    """ Estimate the time left until completion of training.
    """

    chief_only = True

    def __init__(self, last_k_epochs=5, estimator=np.mean):
        self.durations = deque(maxlen=last_k_epochs)
        self.estimator = estimator

    def before_train(self):
        self.last_time = time.time()

    def trigger_epoch(self):
        self.trigger()

    def trigger(self):
        if self.trainer.epoch_num == self.trainer.max_epoch:
            return

        self.durations.append(time.time() - self.last_time)
        self.last_time = time.time()

        duration = self.estimator(self.durations)
        time_left = (self.trainer.max_epoch - self.trainer.epoch_num) * duration
        logger.info('Estimated time left is {}.'.format(humanize_time_delta(time_left)))
