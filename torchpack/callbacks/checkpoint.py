import glob
import os.path as osp
from collections import deque

from ..environ import get_run_dir
from ..logging import logger
from ..utils import fs
from .callback import Callback

__all__ = ['Saver', 'MinSaver', 'MaxSaver', 'SaverRestore']


class Saver(Callback):
    """
    Save the checkpoint once triggered.
    """
    master_only = True

    def __init__(self, *, max_to_keep=5, save_dir=None):
        self.max_to_keep = max_to_keep
        if save_dir is None:
            save_dir = osp.join(get_run_dir(), 'checkpoints')
        self.save_dir = fs.normpath(save_dir)
        fs.makedir(self.save_dir)

    def _before_train(self):
        self.checkpoints = deque()
        checkpoints = glob.glob(osp.join(self.save_dir, 'step-*'))
        for dirpath in sorted(checkpoints, key=osp.getmtime):
            self._add_checkpoint(dirpath)

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        save_dir = osp.join(self.save_dir, f'step-{self.trainer.global_step}')
        try:
            self.trainer.save_checkpoint(save_dir)
        except OSError:
            logger.exception(
                f'Error occurred when saving checkpoint "{save_dir}".')
        else:
            logger.info(f'Checkpoint saved: "{save_dir}".')
            self._add_checkpoint(save_dir)

    def _add_checkpoint(self, dirpath):
        self.checkpoints.append(dirpath)
        if self.max_to_keep is None:
            return
        while len(self.checkpoints) > self.max_to_keep:
            dirpath = self.checkpoints.popleft()
            try:
                fs.remove(dirpath)
            except OSError:
                logger.exception(
                    f'Error occurred when removing checkpoint "{dirpath}".')


class BestSaver(Callback):
    """
    Save the checkpoint with best value of some scalar.
    """
    master_only = True

    def __init__(self, scalar, *, name=None, save_dir=None):
        self.scalar = scalar
        if name is None:
            name = self.extreme + '-' + scalar.replace('/', '-')
        self.name = name
        if save_dir is None:
            save_dir = osp.join(get_run_dir(), 'checkpoints')
        self.save_dir = fs.normpath(save_dir)
        fs.makedir(self.save_dir)

    def _before_train(self):
        self.step = None
        self.best = None

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        if self.scalar not in self.trainer.monitors:
            logger.warning(
                f'`{self.scalar}` has not been added to `trainer.monitors`.')
            return
        step, value = self.trainer.monitors[self.scalar]

        if self.step is not None and step <= self.step:
            logger.warning(
                f'`{self.scalar}` has not been updated since last trigger.')
            return
        self.step = step

        if self.best is None or (self.extreme == 'min' and value < self.best[1]) \
                             or (self.extreme == 'max' and value > self.best[1]):
            save_dir = osp.join(self.save_dir, self.name)
            try:
                fs.remove(save_dir)
            except OSError:
                logger.exception(
                    f'Error occurred when removing checkpoint "{save_dir}".')
            try:
                self.trainer.save_checkpoint(save_dir)
            except OSError:
                logger.exception(
                    f'Error occurred when saving checkpoint "{save_dir}".')
            else:
                logger.info(f'Checkpoint saved: "{save_dir}" ({value:.5g}).')
                self.best = (step, value)

        if self.best is not None:
            self.trainer.monitors.add_scalar(self.scalar + '/' + self.extreme,
                                             self.best[1])


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


class SaverRestore(Callback):
    def __init__(self, load_dir=None):
        if load_dir is None:
            load_dir = osp.join(get_run_dir(), 'checkpoints')
        self.load_dir = fs.normpath(load_dir)

    def _before_train(self):
        checkpoints = glob.glob(osp.join(self.load_dir, 'step-*'))
        if not checkpoints:
            logger.warning(f'No checkpoints found: "{self.load_dir}".')
            return

        load_dir = max(checkpoints, key=osp.getmtime)
        try:
            self.trainer.load_checkpoint(load_dir)
        except OSError:
            logger.exception(
                f'Error occurred when loading checkpoint "{load_dir}".')
        else:
            logger.info(f'Checkpoint loaded: "{load_dir}".')
