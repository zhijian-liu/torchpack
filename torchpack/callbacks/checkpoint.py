import glob
import os
from collections import deque
from typing import Any, ClassVar, Dict, Optional

from torchpack.callbacks.callback import Callback
from torchpack.environ import get_run_dir
from torchpack.utils import fs, io
from torchpack.utils.logging import logger
from torchpack.utils.typing import Trainer

__all__ = ['Saver', 'MinSaver', 'MaxSaver', 'SaverRestore']


class Saver(Callback):
    """
    Save the checkpoint once triggered.
    """
    master_only: bool = True

    def __init__(self, *, max_to_keep: int = 4,
                 save_dir: Optional[str] = None) -> None:
        self.max_to_keep = max_to_keep
        if save_dir is None:
            save_dir = os.path.join(get_run_dir(), 'checkpoints')
        self.save_dir = fs.normpath(save_dir)

    def _set_trainer(self, trainer: Trainer) -> None:
        self.checkpoints = deque()
        for fpath in sorted(glob.glob(os.path.join(self.save_dir,
                                                   'step-*.pt')),
                            key=os.path.getmtime):
            self._add_checkpoint(fpath)

    def _trigger_epoch(self) -> None:
        self._trigger()

    def _trigger(self) -> None:
        save_path = os.path.join(self.save_dir,
                                 f'step-{self.trainer.global_step}.pt')
        try:
            io.save(save_path, self.trainer.state_dict())
        except OSError:
            logger.exception(
                f'Error occurred when saving checkpoint "{save_path}".')
        else:
            logger.info(f'Checkpoint saved: "{save_path}".')
            self._add_checkpoint(save_path)

    def _add_checkpoint(self, fpath: str) -> None:
        self.checkpoints.append(fpath)
        if self.max_to_keep is not None:
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
    master_only: bool = True
    extreme: ClassVar[str]

    def __init__(self,
                 scalar: str,
                 *,
                 name: Optional[str] = None,
                 save_dir: Optional[str] = None) -> None:
        self.scalar = scalar
        if name is None:
            name = self.extreme + '-' + scalar.replace('/', '-')
        self.name = name
        if save_dir is None:
            save_dir = os.path.join(get_run_dir(), 'checkpoints')
        self.save_dir = fs.normpath(save_dir)

    def _set_trainer(self, trainer: Trainer) -> None:
        self.step, self.best = None, None

    def _trigger_epoch(self) -> None:
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
            self.best = (step, value)
            save_path = os.path.join(self.save_dir, self.name + '.pt')
            try:
                io.save(save_path, self.trainer.state_dict())
            except OSError:
                logger.exception(
                    f'Error occurred when saving checkpoint "{save_path}".')
            else:
                logger.info(f'Checkpoint saved: "{save_path}" ({value:.5g}).')

        if self.best is not None:
            self.trainer.summary.add_scalar(self.scalar + '/' + self.extreme,
                                            self.best[1])

    def _state_dict(self) -> Dict[str, Any]:
        return {'step': self.step, 'best': self.best}

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.step, self.best = state_dict['step'], state_dict['best']


class MinSaver(BestSaver):
    """
    Save the checkpoint with minimum value of some scalar in `trainer.summary`.
    """
    extreme: ClassVar[str] = 'min'


class MaxSaver(BestSaver):
    """
    Save the checkpoint with maximum value of some scalar in `trainer.summary`.
    """
    extreme: ClassVar[str] = 'max'


class SaverRestore(Callback):
    def __init__(self, load_dir: Optional[str] = None) -> None:
        if load_dir is None:
            load_dir = os.path.join(get_run_dir(), 'checkpoints')
        self.load_dir = fs.normpath(load_dir)

    def _before_train(self) -> None:
        checkpoints = glob.glob(os.path.join(self.load_dir, 'step-*.pt'))
        if not checkpoints:
            logger.warning(f'No checkpoints found: "{self.load_dir}".')
            return

        load_path = max(checkpoints, key=os.path.getmtime)
        try:
            state_dict = io.load(load_path, map_location='cpu')
            self.trainer.load_state_dict(state_dict)
        except OSError:
            logger.exception(
                f'Error occurred when loading checkpoint "{load_path}".')
        else:
            logger.info(f'Checkpoint loaded: "{load_path}".')
