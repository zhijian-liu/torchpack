import heapq
import os
import re

import torch

from torchpack.callbacks.callback import Callback
from torchpack.utils.logging import logger, get_logger_dir

__all__ = ['ModelSaver', 'MinSaver', 'MaxSaver']


class ModelSaver(Callback):
    """
    Save the trainer's state dict once triggered.
    """

    def __init__(self, checkpoint_dir=None, max_to_keep=10):
        """
        Args:
            checkpoint_dir (str): Defaults to ``logger.get_logger_dir()``.
            max_to_keep (int): Maximum number of recent checkpoint files to keep.
        """
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(get_logger_dir(), 'checkpoints')
        checkpoint_dir = os.path.normpath(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.checkpoints = []

    def _add_checkpoint(self, filename):
        heapq.heappush(self.checkpoints, (os.path.getmtime(filename), filename))
        while len(self.checkpoints) > self.max_to_keep:
            filename = heapq.heappop(self.checkpoints)[1]
            try:
                os.remove(filename)
            except (OSError, IOError):
                logger.exception('Error occurred when removing checkpoint "{}".'.format(filename))

    def _before_train(self):
        regex = re.compile('^step-[0-9]+.pth$')
        for filename in os.listdir(self.checkpoint_dir):
            if regex.match(filename):
                filename = os.path.join(self.checkpoint_dir, filename)
                self._add_checkpoint(filename)

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        filename = os.path.join(self.checkpoint_dir, 'step-{}.pth'.format(self.trainer.global_step))
        try:
            torch.save(self.trainer.state_dict(), filename)
        except (OSError, IOError):
            logger.exception('Error occurred when saving checkpoint "{}".'.format(filename))
        else:
            logger.info('Checkpoint saved: "{}".'.format(filename))
            self._add_checkpoint(filename)


class BestSaver(Callback):
    """
    Save the model with best value of some statistics.
    """

    def __init__(self, key, checkpoint_dir=None, filename=None):
        """
        Args:
            key(str): the name of the statistics.
            checkpoint_dir (str): the directory containing checkpoints.
            filename (str): the name for the saved model. Defaults to ``{key}-min.pth``.
        """
        self.key = key
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(get_logger_dir(), 'checkpoints')
        checkpoint_dir = os.path.normpath(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        if filename is None:
            filename = '{}-{}.pth'.format(key.replace('/', '-'), self.extreme)
        self.filename = filename

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        try:
            step, value = self.trainer.monitors.get_history(self.key)[-1]
        except (KeyError, IndexError):
            return

        try:
            best = self.trainer.monitors.get_history(self.key + '/' + self.extreme)[-1]
        except (KeyError, IndexError):
            best = None

        if best is None or (self.extreme == 'min' and value < best[1]) or (self.extreme == 'max' and value > best[1]):
            filename = os.path.join(self.checkpoint_dir, self.filename)
            try:
                torch.save(self.trainer.state_dict(), filename)
            except (OSError, IOError):
                logger.exception('Error occurred when saving checkpoint "{}".'.format(filename))
            else:
                logger.info('Checkpoint saved: "{}" ({:.5g}).'.format(filename, value))

        self.trainer.monitors.add_scalar(self.key + '/' + self.extreme, self.best[1])


class MinSaver(BestSaver):
    """
    Save the model with minimum value of some statistics.
    """

    extreme = 'min'


class MaxSaver(BestSaver):
    """
    Save the model with maximum value of some statistics.
    """

    extreme = 'max'
