import time
from collections import deque

import numpy as np
from tensorpack.utils.utils import humanize_time_delta
from torchpack.utils.logging import logger

from .callback import Callback

__all__ = ['EstimatedTimeLeft']


class EstimatedTimeLeft(Callback):
    """
    Estimate the time left until completion of training.
    """

    def __init__(self, last_k_epochs=5, median=False):
        """
        Args:
            last_k_epochs (int): Use the time spent on last k epochs to estimate total time left.
            median (bool): Use mean by default. If True, use the median time spent on last k epochs.
        """
        self.durations = deque(maxlen=last_k_epochs)
        self.median = median

    def before_train(self):
        self.last_time = time.time()

    def trigger_epoch(self):
        duration = time.time() - self.last_time
        self.last_time = time.time()
        self.durations.append(duration)

        epoch_time = np.median(self.durations) if self.median else np.mean(self.durations)
        time_left = (self.trainer.max_epoch - self.trainer.epoch_num) * epoch_time
        if time_left > 0:
            logger.info('Estimated time left is {}.'.format(humanize_time_delta(time_left)))
