import time
from typing import List

import torch
import tqdm
from torch.utils.data import DataLoader

import torchpack.utils.humanize as humanize
from torchpack.callbacks.callback import Callback, Callbacks
from torchpack.train import Trainer
from torchpack.utils.logging import logger

__all__ = ['InferenceRunner']


class InferenceRunner(Callback):
    """
    A callback that runs inference with a list of :class:`Callback`.
    """
    def __init__(self, dataflow: DataLoader, *,
                 callbacks: List[Callback]) -> None:
        self.dataflow = dataflow
        self.callbacks = Callbacks(callbacks)

    def _set_trainer(self, trainer: Trainer) -> None:
        self.callbacks.set_trainer(trainer)

    def _trigger_epoch(self) -> None:
        self._trigger()

    def _trigger(self) -> None:
        start_time = time.time()
        self.callbacks.before_epoch()

        with torch.no_grad():
            for feed_dict in tqdm.tqdm(self.dataflow, ncols=0):
                self.callbacks.before_step(feed_dict)
                output_dict = self.trainer.run_step(feed_dict)
                self.callbacks.after_step(output_dict)

        self.callbacks.after_epoch()
        logger.info('Inference finished in {}.'.format(
            humanize.naturaldelta(time.time() - start_time)))
