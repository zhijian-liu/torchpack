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

    def __init__(self, max_to_keep=10, save_path=None):
        """
        Args:
            max_to_keep (int): Maximum number of recent checkpoint files to keep.
            save_path (str): Defaults to ``logger.get_logger_dir()``.
        """
        self.max_to_keep = max_to_keep
        self.save_path = os.path.normpath(save_path or os.path.join(get_logger_dir(), 'checkpoints'))
        os.makedirs(self.save_path, exist_ok=True)
        self.checkpoints = list()

    def _add_checkpoint(self, checkpoint):
        heapq.heappush(self.checkpoints, (os.path.getmtime(checkpoint), checkpoint))
        while self.max_to_keep is not None and len(self.checkpoints) > self.max_to_keep:
            checkpoint = heapq.heappop(self.checkpoints)[1]
            try:
                shutil.rmtree(checkpoint)
            except (OSError, IOError):
                logger.exception('Error occurred when removing checkpoint "{}".'.format(checkpoint))

    def _before_train(self):
        regex = re.compile('^step-[0-9]+$')
        for dirname in os.listdir(self.save_path):
            if regex.match(dirname):
                self._add_checkpoint(os.path.join(self.save_path, dirname))

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        save_path = os.path.join(self.save_path, 'step-{}'.format(self.trainer.global_step))
        try:
            os.makedirs(save_path, exist_ok=True)
            self.trainer.save_checkpoint(save_path)
        except (OSError, IOError):
            logger.exception('Error occurred when saving checkpoint "{}".'.format(save_path))
        else:
            logger.info('Checkpoint saved: "{}".'.format(save_path))
            self._add_checkpoint(save_path)


class BestSaver(Callback):
    """
    Save the checkpoint with best value of some statistics.
    """

    def __init__(self, key, name=None, save_path=None):
        """
        Args:
            key (str): the name of the statistics.
            name (str): the name for the saved model. Defaults to ``min-{key}``.
            save_path (str): the directory containing checkpoints.
        """
        self.key = key
        self.name = name
        self.save_path = os.path.normpath(save_path or os.path.join(get_logger_dir(), 'checkpoints'))
        os.makedirs(self.save_path, exist_ok=True)

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
            save_path = os.path.join(self.save_path, self.name or (self.extreme + '-' + self.key.replace('/', '-')))
            try:
                os.makedirs(save_path, exist_ok=True)
                self.trainer.save_checkpoint(save_path)
            except (OSError, IOError):
                logger.exception('Error occurred when saving checkpoint "{}".'.format(save_path))
            else:
                logger.info('Checkpoint saved: "{}" ({:.5g}).'.format(save_path, value))
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
