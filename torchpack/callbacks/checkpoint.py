import glob
import os.path as osp
from collections import deque

import torchpack.utils.fs as fs
from torchpack.callbacks.callback import Callback
from torchpack.environ import get_run_dir
from torchpack.logging import logger

__all__ = ['Saver', 'MinSaver', 'MaxSaver', 'Resumer']


class Saver(Callback):
    """
    Save the checkpoint once triggered.
    """
    def __init__(self, max_to_keep=10, save_dir=None):
        self.max_to_keep = max_to_keep
        self.save_dir = save_dir or osp.join(get_run_dir(), 'checkpoints')
        self.save_dir = fs.makedir(self.save_dir)
        self.checkpoints = deque()

    def _add_checkpoint(self, checkpoint):
        self.checkpoints.append(checkpoint)
        while self.max_to_keep is not None and len(
                self.checkpoints) > self.max_to_keep:
            checkpoint = self.checkpoints.popleft()
            try:
                fs.remove(checkpoint)
            except (OSError, IOError):
                logger.exception(
                    'Error occurred when removing checkpoint "{}".' \
                        .format(checkpoint))

    def _before_train(self):
        checkpoints = glob.glob(osp.join(self.save_dir, 'step-*'))
        for checkpoint in sorted(checkpoints, key=osp.getmtime):
            self._add_checkpoint(checkpoint)

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        checkpoint = fs.makedir(
            osp.join(self.save_dir,
                     'step-{}'.format(self.trainer.global_step)))
        try:
            self.trainer.save(checkpoint)
        except (OSError, IOError):
            logger.exception(
                'Error occurred when saving checkpoint "{}".'.format(
                    checkpoint))
        else:
            logger.info('Checkpoint saved: "{}".'.format(checkpoint))
            self._add_checkpoint(checkpoint)


class BestSaver(Callback):
    """
    Save the checkpoint with best value of some scalar.
    """
    def __init__(self, scalar_name, save_dir=None, save_name=None):
        self.scalar_name = scalar_name
        self.save_dir = save_dir or osp.join(get_run_dir(), 'checkpoints')
        self.save_dir = fs.makedir(self.save_dir)
        self.save_name = save_name or (self.extreme + '-' +
                                       scalar_name.replace('/', '-'))
        self.best, self.step = None, None

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        if self.scalar_name not in self.trainer.monitors:
            logger.warning(
                'Scalar `{}` has not been added to `trainer.monitors` yet.'.
                format(self.scalar_name))
            return
        step, value = self.trainer.monitors[self.scalar_name]

        if self.step is not None and step <= self.step:
            logger.warning(
                'Scalar `{}` has not been updated since the last trigger.'.
                format(self.scalar_name))
            return
        self.step = step

        if self.best is None or \
                (self.extreme == 'min' and value < self.best[1]) or \
                (self.extreme == 'max' and value > self.best[1]):
            checkpoint = fs.makedir(osp.join(self.save_dir, self.save_name))
            try:
                self.trainer.save(checkpoint)
            except (OSError, IOError):
                logger.exception(
                    'Error occurred when saving checkpoint "{}".'.format(
                        checkpoint))
            else:
                logger.info('Checkpoint saved: "{}" ({:.5g}).'.format(
                    checkpoint, value))
                self.best = (step, value)

        if self.best is not None:
            self.trainer.monitors.add_scalar(
                self.scalar_name + '/' + self.extreme, self.best[1])


class MinSaver(BestSaver):
    """
    Save the checkpoint with minimum value of some scalar.
    """

    extreme = 'min'


class MaxSaver(BestSaver):
    """
    Save the checkpoint with maximum value of some scalar.
    """

    extreme = 'max'


class Resumer(Callback):
    def __init__(self, load_dir=None):
        self.load_dir = load_dir or osp.join(get_run_dir(), 'checkpoints')
        self.load_dir = osp.normpath(self.load_dir)

    def _before_train(self):
        checkpoints = glob.glob(osp.join(self.load_dir, 'step-*'))
        if not checkpoints:
            logger.warning('No checkpoints found: "{}".'.format(self.load_dir))
            return

        checkpoint = max(checkpoints, key=osp.getmtime)
        try:
            self.trainer.load(checkpoint)
        except (OSError, IOError):
            logger.exception(
                'Error occurred when loading checkpoint "{}".'.format(
                    checkpoint))
        else:
            logger.info('Checkpoint resumed: "{}".'.format(checkpoint))
