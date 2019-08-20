import heapq
import os
import re
import shutil

from torchpack.callbacks.callback import Callback
from torchpack.utils.logging import logger, get_logger_dir

__all__ = ['Saver', 'MinSaver', 'MaxSaver', 'Resumer']


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

    def _add_checkpoint(self, checkpoint_path):
        heapq.heappush(self.checkpoints, (os.path.getmtime(checkpoint_path), checkpoint_path))
        while self.max_to_keep is not None and len(self.checkpoints) > self.max_to_keep:
            checkpoint_path = heapq.heappop(self.checkpoints)[1]
            try:
                shutil.rmtree(checkpoint_path)
            except (OSError, IOError):
                logger.exception('Error occurred when removing checkpoint "{}".'.format(checkpoint_path))

    def _before_train(self):
        regex = re.compile('^step-[0-9]+$')
        for dirname in os.listdir(self.save_path):
            if regex.match(dirname):
                checkpoint_path = os.path.join(self.save_path, dirname)
                self._add_checkpoint(checkpoint_path)

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        checkpoint_path = os.path.join(self.save_path, 'step-{}'.format(self.trainer.global_step))
        try:
            os.makedirs(checkpoint_path, exist_ok=True)
            self.trainer.save_checkpoint(checkpoint_path)
        except (OSError, IOError):
            logger.exception('Error occurred when saving checkpoint "{}".'.format(checkpoint_path))
        else:
            logger.info('Checkpoint saved: "{}".'.format(checkpoint_path))
            self._add_checkpoint(checkpoint_path)


class BestSaver(Callback):
    """
    Save the checkpoint with best value of some statistics.
    """

    def __init__(self, key, save_name=None, save_path=None):
        """
        Args:
            key (str): the name of the statistics.
            save_name (str): the name for the saved model. Defaults to ``min-{key}``.
            save_path (str): the directory containing checkpoints.
        """
        self.key = key
        self.save_name = save_name or (self.extreme + '-' + key.replace('/', '-'))
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
            checkpoint_path = os.path.join(self.save_path, self.save_name)
            try:
                os.makedirs(checkpoint_path, exist_ok=True)
                self.trainer.save_checkpoint(checkpoint_path)
            except (OSError, IOError):
                logger.exception('Error occurred when saving checkpoint "{}".'.format(checkpoint_path))
            else:
                logger.info('Checkpoint saved: "{}" ({:.5g}).'.format(checkpoint_path, value))
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


class Resumer(Callback):
    def __init__(self, resume_path=None):
        self.resume_path = os.path.normpath(resume_path or os.path.join(get_logger_dir(), 'checkpoints'))

    def _before_train(self):
        self.trainer.load_checkpoint(self.resume_path)
        logger.info('Checkpoint resumed: "{}"'.format(self.resume_path))
