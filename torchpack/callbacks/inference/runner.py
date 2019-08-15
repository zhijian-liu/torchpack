import time

import torch
import tqdm
from tensorpack.utils.utils import get_tqdm_kwargs
from tensorpack.utils.utils import humanize_time_delta

from torchpack.callbacks.callback import Callback
from torchpack.callbacks.inference.callback import InferenceCallback, InferenceCallbacks
from torchpack.cuda.copy import async_copy_to
from torchpack.utils.logging import logger

__all__ = ['InferenceRunner']


class InferenceRunner(Callback):
    """
    A callback that runs inference with a list of :class:`InferenceCallback`.
    """

    def __init__(self, dataflow, callbacks, device=None):
        for callback in callbacks:
            assert isinstance(callback, InferenceCallback), callback
        self.dataflow = dataflow
        self.callbacks = callbacks
        self.device = device

    def _set_trainer(self, trainer):
        self.device = self.device or trainer.device
        self.callbacks = InferenceCallbacks(self.callbacks)
        self.callbacks.set_trainer(trainer)

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        start_time = time.time()
        self.callbacks.before_inference()

        self.trainer.model.eval()
        with torch.no_grad():
            for feed_dict in tqdm.tqdm(self.dataflow, **get_tqdm_kwargs()):
                feed_dict = async_copy_to(feed_dict, device=self.device)

                self.callbacks.before_step(feed_dict)
                output_dict = self.trainer.model(feed_dict)
                self.callbacks.after_step(feed_dict, output_dict)

        self.callbacks.after_inference()
        logger.info('Inference finished in {}.'.format(humanize_time_delta(time.time() - start_time)))
