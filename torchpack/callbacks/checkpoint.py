import glob
import os.path as osp
from collections import deque

import torchpack.utils.fs as fs
import torchpack.utils.io as io
from torchpack.callbacks.callback import Callback
from torchpack.utils.logging import logger, get_logger_dir

__all__ = ['Saver', 'MinSaver', 'MaxSaver', 'AutoResumer']


class Saver(Callback):
    """
    Save the checkpoint once triggered.
    """

    def __init__(self, max_to_keep=10, save_path=None):
        """
        Args:
            max_to_keep (int): maximum number of recent checkpoint files to keep.
            save_path (str): Defaults to `logger.get_logger_dir()`.
        """
        self.max_to_keep = max_to_keep
        self.save_path = fs.makedir(save_path or osp.join(get_logger_dir(), 'checkpoints'))
        self.checkpoints = deque()

    def _add_checkpoint(self, checkpoint):
        self.checkpoints.append(checkpoint)
        while self.max_to_keep is not None and len(self.checkpoints) > self.max_to_keep:
            checkpoint = self.checkpoints.popleft()
            try:
                fs.remove(checkpoint)
            except (OSError, IOError):
                logger.exception('Error occurred when removing checkpoint "{}".'.format(checkpoint))

    def _before_train(self):
        checkpoints = glob.glob(osp.join(self.save_path, 'step-*'))
        for checkpoint in sorted(checkpoints, key=osp.getmtime):
            self._add_checkpoint(checkpoint)

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        checkpoint = fs.makedir(osp.join(self.save_path, 'step-{}'.format(self.trainer.global_step)))
        try:
            self.trainer.save(checkpoint)
        except (OSError, IOError):
            logger.exception('Error occurred when saving checkpoint "{}".'.format(checkpoint))
        else:
            logger.info('Checkpoint saved: "{}".'.format(checkpoint))
            self._add_checkpoint(checkpoint)


class BestSaver(Callback):
    """
    Save the checkpoint with best value of some statistics.
    """

    def __init__(self, key, save_path=None, save_name=None):
        """
        Args:
            key (str): the name of the statistics.
            save_path (str): the directory for saving checkpoints.
            save_name (str): the name for the saved checkpoint. Defaults to `min-{key}`.
        """
        self.key = key
        self.save_path = fs.makedir(save_path or osp.join(get_logger_dir(), 'checkpoints'))
        self.save_name = save_name or (self.extreme + '-' + key.replace('/', '-'))
        self.step, self.best = None, None

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        if self.key not in self.trainer.monitors:
            logger.warning('`{}` has not been added to `trainer.monitors` yet.'.format(self.key))
            return
        step, value = self.trainer.monitors[self.key]

        if self.step is not None and step <= self.step:
            logger.warning('`{}` has not been updated since the last trigger at step {}.'.format(self.key, self.step))
            return
        self.step = step

        if self.best is None or \
                (self.extreme == 'min' and value < self.best[1]) or \
                (self.extreme == 'max' and value > self.best[1]):
            checkpoint = fs.makedir(osp.join(self.save_path, self.save_name))
            try:
                self.trainer.save(checkpoint)
            except (OSError, IOError):
                logger.exception('Error occurred when saving checkpoint "{}".'.format(checkpoint))
            else:
                logger.info('Checkpoint saved: "{}" ({:.5g}).'.format(checkpoint, value))
                self.best = (step, value)

        if self.best is not None:
            self.trainer.monitors.add_scalar(self.key + '/' + self.extreme, self.best[1])


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


class AutoResumer(Callback):
    def __init__(self, resume_path=None):
        self.resume_path = osp.normpath(resume_path or osp.join(get_logger_dir(), 'checkpoints'))

    def _before_train(self):
        checkpoints = glob.glob(osp.join(self.resume_path, 'step-*'))
        if not checkpoints:
            logger.warning('No checkpoints found: "{}".'.format(self.resume_path))
            return

        checkpoint = max(checkpoints, key=osp.getmtime)
        try:
            self.trainer.load(checkpoint)
        except (OSError, IOError):
            logger.exception('Error occurred when loading checkpoint "{}".'.format(checkpoint))
        else:
            logger.info('Checkpoint resumed: "{}".'.format(checkpoint))
