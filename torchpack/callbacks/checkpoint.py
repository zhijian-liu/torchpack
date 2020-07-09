import glob
import os.path as osp
from collections import deque

from ..environ import get_run_dir
from ..utils import fs, io
from ..utils.logging import logger
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

        self.checkpoints = deque()
        for fpath in sorted(glob.glob(osp.join(self.save_dir, 'step-*.pt')),
                            key=osp.getmtime):
            self._add_checkpoint(fpath)

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        save_path = osp.join(self.save_dir,
                             f'step-{self.trainer.global_step}.pt')
        try:
            io.save(save_path, self.trainer.state_dict())
        except OSError:
            logger.exception(
                f'Error occurred when saving checkpoint "{save_path}".')
        else:
            logger.info(f'Checkpoint saved: "{save_path}".')
            self._add_checkpoint(save_path)

    def _add_checkpoint(self, fpath):
        self.checkpoints.append(fpath)
        if self.max_to_keep is None:
            return
        while len(self.checkpoints) > self.max_to_keep:
            fpath = self.checkpoints.popleft()
            try:
                fs.remove(fpath)
            except OSError:
                logger.exception(
                    f'Error occurred when removing checkpoint "{fpath}".')


class BestSaver(Callback):
    """
    Save the checkpoint with best value of some scalar in `trainer.summary`.
    """
    master_only = True
    extreme = None

    def __init__(self, scalar, *, name=None, save_dir=None):
        self.scalar = scalar
        self.step, self.best = None, None

        if name is None:
            name = self.extreme + '-' + scalar.replace('/', '-')
        self.name = name

        if save_dir is None:
            save_dir = osp.join(get_run_dir(), 'checkpoints')
        self.save_dir = fs.normpath(save_dir)

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        if self.scalar not in self.trainer.summary:
            logger.warning(
                f'`{self.scalar}` has not been added to `trainer.summary`.')
            return
        step, value = self.trainer.summary[self.scalar]

        if self.step is not None and step <= self.step:
            logger.warning(
                f'`{self.scalar}` has not been updated since last trigger.')
            return
        self.step = step

        if self.best is None or (self.extreme == 'min' and value < self.best[1]) \
                             or (self.extreme == 'max' and value > self.best[1]):
            save_path = osp.join(self.save_dir, self.name + '.pt')
            try:
                io.save(save_path, self.trainer.state_dict())
            except OSError:
                logger.exception(
                    f'Error occurred when saving checkpoint "{save_path}".')
            else:
                logger.info(f'Checkpoint saved: "{save_path}" ({value:.5g}).')
                self.best = (step, value)

        if self.best is not None:
            self.trainer.summary.add_scalar(self.scalar + '/' + self.extreme,
                                            self.best[1])

    def _state_dict(self):
        return {'step': self.step, 'best': self.best}

    def _load_state_dict(self, state_dict):
        self.step, self.best = state_dict['step'], state_dict['best']


class MinSaver(BestSaver):
    """
    Save the checkpoint with minimum value of some scalar in `trainer.summary`.
    """
    extreme = 'min'


class MaxSaver(BestSaver):
    """
    Save the checkpoint with maximum value of some scalar in `trainer.summary`.
    """
    extreme = 'max'


class SaverRestore(Callback):
    def __init__(self, load_dir=None):
        if load_dir is None:
            load_dir = osp.join(get_run_dir(), 'checkpoints')
        self.load_dir = fs.normpath(load_dir)

    def _before_train(self):
        checkpoints = glob.glob(osp.join(self.load_dir, 'step-*.pt'))
        if not checkpoints:
            logger.warning(f'No checkpoints found: "{self.load_dir}".')
            return

        load_path = max(checkpoints, key=osp.getmtime)
        try:
            state_dict = io.load(load_path)
            self.trainer.load_state_dict(state_dict)
        except OSError:
            logger.exception(
                f'Error occurred when loading checkpoint "{load_path}".')
        else:
            logger.info(f'Checkpoint loaded: "{load_path}".')
