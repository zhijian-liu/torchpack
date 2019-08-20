import heapq
import os
import re
import shutil

from torchpack.callbacks.callback import Callback
from torchpack.utils.logging import logger, get_logger_dir

__all__ = ['Saver', 'MinSaver', 'MaxSaver']


class Saver(Callback):
    """
    Save the checkpoint once triggered.
    """

    def __init__(self, max_to_keep=10, checkpoint_dir=None):
        """
        Args:
            max_to_keep (int): Maximum number of recent checkpoint files to keep.
            checkpoint_dir (str): Defaults to ``logger.get_logger_dir()``.
        """
        self.max_to_keep = max_to_keep
        self.checkpoint_dir = os.path.normpath(checkpoint_dir or os.path.join(get_logger_dir(), 'checkpoints'))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoints = list()

    def _add_checkpoint(self, path):
        heapq.heappush(self.checkpoints, (os.path.getmtime(path), path))
        while self.max_to_keep is not None and len(self.checkpoints) > self.max_to_keep:
            path = heapq.heappop(self.checkpoints)[1]
            try:
                shutil.rmtree(path)
            except (OSError, IOError):
                logger.exception('Error occurred when removing checkpoint "{}".'.format(path))

    def _before_train(self):
        regex = re.compile('^step-[0-9]+$')
        for filename in os.listdir(self.checkpoint_dir):
            if regex.match(filename):
                filename = os.path.join(self.checkpoint_dir, filename)
                self._add_checkpoint(filename)

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        path = os.path.join(self.checkpoint_dir, 'step-{}'.format(self.trainer.global_step))
        try:
            os.makedirs(path, exist_ok=True)
            self.trainer.save_checkpoint(path)
        except (OSError, IOError):
            logger.exception('Error occurred when saving checkpoint "{}".'.format(path))
        else:
            logger.info('Checkpoint saved: "{}".'.format(path))
            self._add_checkpoint(path)


class BestSaver(Callback):
    """
    Save the checkpoint with best value of some statistics.
    """

    def __init__(self, key, name=None, checkpoint_dir=None):
        """
        Args:
            key (str): the name of the statistics.
            name (str): the name for the saved model. Defaults to ``min-{key}``.
            checkpoint_dir (str): the directory containing checkpoints.
        """
        self.key = key
        self.name = name
        self.checkpoint_dir = os.path.normpath(checkpoint_dir or os.path.join(get_logger_dir(), 'checkpoints'))
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        # TODO: `self.key in self.train.monitors`
        try:
            step, value = self.trainer.monitors.get_history(self.key)[-1]
        except (KeyError, IndexError):
            return

        # TODO: `self.key + '/' + self.extreme in self.train.monitors`
        try:
            best = self.trainer.monitors.get_history(self.key + '/' + self.extreme)[-1]
        except (KeyError, IndexError):
            best = None

        if best is None or (self.extreme == 'min' and value < best[1]) or (self.extreme == 'max' and value > best[1]):
            path = os.path.join(self.checkpoint_dir, self.name or (self.extreme + '-' + self.key.replace('/', '-')))
            try:
                os.makedirs(path, exist_ok=True)
                self.trainer.save_checkpoint(path)
            except (OSError, IOError):
                logger.exception('Error occurred when saving checkpoint "{}".'.format(path))
            else:
                logger.info('Checkpoint saved: "{}" ({:.5g}).'.format(path, value))
                best = (step, value)

        if best is not None:
            self.trainer.monitors.add_scalar(self.key + '/' + self.extreme, best[1])


class MinSaver(BestSaver):
    """
    Save the checkpoint with minimum value of some statistics.
    """

    extreme = 'min'


class MaxSaver(BestSaver):
    """
    Save the checkpoint with maximum value of some statistics.
    """

    extreme = 'max'
