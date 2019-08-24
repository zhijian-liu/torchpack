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
        self.save_path = fs.mkdir(save_path or osp.join(get_logger_dir(), 'checkpoints'))
        self.checkpoints = deque()

    def _add_checkpoint(self, checkpoint_path):
        self.checkpoints.append(checkpoint_path)
        while self.max_to_keep is not None and len(self.checkpoints) > self.max_to_keep:
            checkpoint_path = self.checkpoints.popleft()
            try:
                fs.remove(checkpoint_path)
            except (OSError, IOError):
                logger.exception('Error occurred when removing checkpoint "{}".'.format(checkpoint_path))

    def _before_train(self):
        files = glob.glob(osp.join(self.save_path, 'step-*'))
        for checkpoint_path in sorted(files, key=osp.getmtime):
            self._add_checkpoint(checkpoint_path)

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        checkpoint_path = osp.join(self.save_path, 'step-{}'.format(self.trainer.global_step))
        try:
            fs.mkdir(checkpoint_path)
            self.trainer.save(checkpoint_path)
        except (OSError, IOError):
            logger.exception('Error occurred when saving checkpoint "{}".'.format(checkpoint_path))
        else:
            logger.info('Checkpoint saved: "{}".'.format(checkpoint_path))
            self._add_checkpoint(checkpoint_path)


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
        self.save_path = fs.mkdir(save_path or osp.join(get_logger_dir(), 'checkpoints'))
        self.save_name = save_name or (self.extreme + '-' + key.replace('/', '-'))
        self.best = None
        self.last_step = None

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        if self.key in self.trainer.monitors:
            step, value = self.trainer.monitors[self.key]
        else:
            logger.warning('skipped.')
            return

        if self.last_step is not None and step <= self.last_step:
            logger.warning('skipped.')
            return
        self.last_step = step

        if self.best is None or (self.extreme == 'min' and value < self.best[1]) or \
                (self.extreme == 'max' and value > self.best[1]):
            self.best = (step, value)
            checkpoint_path = osp.join(self.save_path, self.save_name)
            try:
                fs.mkdir(checkpoint_path)
                self.trainer.save(checkpoint_path)
                # TODO: a quick hack, should move this into self.trainer.save_checkpoint
                self.save(checkpoint_path)
            except (OSError, IOError):
                logger.exception('Error occurred when saving checkpoint "{}".'.format(checkpoint_path))
            else:
                logger.info('Checkpoint saved: "{}" ({:.5g}).'.format(checkpoint_path, value))

        if self.best is not None:
            self.trainer.monitors.add_scalar(self.key + '/' + self.extreme, self.best[1])

    def save(self, save_path):
        io.dump(osp.join(save_path, 'max-saver.json'), self.best)

    def load(self, resume_path):
        self.best = io.load(osp.join(resume_path, 'max-saver.json'))


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
        if not osp.exists(self.resume_path):
            return

        checkpoints = glob.glob(osp.join(self.resume_path, 'step-*'))
        if not checkpoints:
            return

        checkpoint_path = max(checkpoints, key=osp.getmtime)
        try:
            self.trainer.load(checkpoint_path)
        except (OSError, IOError):
            logger.exception('Error occurred when loading checkpoint "{}".'.format(checkpoint_path))
        else:
            logger.info('Checkpoint resumed: "{}".'.format(checkpoint_path))
