import time
from typing import List

import torch
from torch.utils.data import DataLoader

from ..utils import humanize
from ..utils.logging import logger
from ..utils import tqdm
from ..utils.typing import Trainer
from .callback import Callback, Callbacks

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
        start_time = time.perf_counter()
        self.callbacks.before_epoch()

        with torch.no_grad():
            for feed_dict in tqdm.tqdm(self.dataflow):
                self.callbacks.before_step(feed_dict)
                output_dict = self.trainer.run_step(feed_dict)
                self.callbacks.after_step(output_dict)

        self.callbacks.after_epoch()
        logger.info('Inference finished in {}.'.format(
            humanize.naturaldelta(time.perf_counter() - start_time)))
