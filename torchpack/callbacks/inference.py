import sys
import time

import torch
import tqdm
from tensorpack.utils.utils import humanize_time_delta

from ..utils.logging import logger
from .callback import Callback, Callbacks

__all__ = ['InferenceRunner']


class InferenceRunner(Callback):
    """
    A callback that runs inference with a list of :class:`Callback`.
    """
    def __init__(self, dataflow, *, callbacks):
        for callback in callbacks:
            assert isinstance(callback, Callback), type(callback)
        self.callbacks = Callbacks(callbacks)
        self.dataflow = dataflow

    def _set_trainer(self, trainer):
        self.callbacks.set_trainer(trainer)

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        start_time = time.time()
        self.callbacks.before_epoch()

        # TODO: training / evaluation context
        with torch.no_grad():
            for feed_dict in tqdm.tqdm(self.dataflow):
                self.callbacks.before_step(feed_dict)
                output_dict = self.trainer.run_step(feed_dict)
                self.callbacks.after_step(output_dict)

        self.callbacks.after_epoch()
        logger.info('Inference finished in {}.'.format(
            humanize_time_delta(time.time() - start_time)))
