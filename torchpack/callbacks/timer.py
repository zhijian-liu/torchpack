import time
from collections import deque

import numpy as np
from tensorpack.utils.utils import humanize_time_delta

from torchpack.utils.logging import logger
from .base import Callback

__all__ = ['EstimatedTimeLeft']


class EstimatedTimeLeft(Callback):
    """
    Estimate the time left until completion of training.
    """

    def __init__(self, last_k_epochs=5, estimator=np.mean):
        self.durations = deque(maxlen=last_k_epochs)
        self.estimator = estimator

    def before_train(self):
        self.last_time = time.time()

    def trigger_epoch(self):
        self.trigger()

    def trigger(self):
        if self.trainer.epoch_num == self.trainer.epoch_num:
            return

        self.durations.append(time.time() - self.last_time)
        self.last_time = time.time()

        duration = self.estimator(self.durations)
        time_left = (self.trainer.max_epoch - self.trainer.epoch_num) * duration
        logger.info('Estimated time left is {}.'.format(humanize_time_delta(time_left)))
