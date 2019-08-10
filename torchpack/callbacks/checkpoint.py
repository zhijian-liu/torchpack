import os

import torch

from torchpack.utils.logging import logger
from .callback import Callback

__all__ = ['ModelSaver', 'MinSaver', 'MaxSaver']


class ModelSaver(Callback):
    """ Save the model once triggered.
    """

    def __init__(self, max_to_keep=10, checkpoint_dir=None):
        """
        Args:
            max_to_keep (int): Maximum number of recent checkpoint files to keep.
            checkpoint_dir (str): Defaults to ``logger.get_logger_dir()``.
        """
        self.max_to_keep = max_to_keep
        if checkpoint_dir is None:
            checkpoint_dir = logger.get_logger_dir()
        self.checkpoint_dir = os.path.normpath(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def trigger_epoch(self):
        self.trigger()

    def trigger(self):
        state_dict = self.trainer.state_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, 'step-{}.pth'.format(self.trainer.global_step))
        try:
            torch.save(state_dict, checkpoint_path)
            logger.info('Checkpoint saved to {}.'.format(checkpoint_path))
        except (OSError, IOError):
            logger.exception('Exception in ModelSaver!')


class MinSaver(Callback):
    """ Separately save the model with minimum value of some statistics.
    """

    def __init__(self, monitor_stat, reverse=False, filename=None, checkpoint_dir=None):
        """
        Args:
            monitor_stat(str): the name of the statistics.
            reverse (bool): if True, will save the maximum.
            filename (str): the name for the saved model.
                Defaults to ``min-{monitor_stat}.tfmodel``.
            checkpoint_dir (str): the directory containing checkpoints.
        Example:
            Save the model with minimum validation error to
            "min-val-error.tfmodel":
            .. code-block:: python
                MinSaver('val-error')
        Note:
            1. It assumes that :class:`ModelSaver` is used with the same ``checkpoint_dir``
               and appears earlier in the callback list.
               The default for both :class:`ModelSaver` and :class:`MinSaver`
               is ``checkpoint_dir=logger.get_logger_dir()``
            2. Callbacks are executed in the order they are defined. Therefore you'd want to
               use this callback after the callback (e.g. InferenceRunner) that produces the statistics.
        """
        self.monitor_stat = monitor_stat
        self.reverse = reverse
        self.filename = filename
        self.best = None
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is None:
            self.checkpoint_dir = logger.get_logger_dir()

    def before_train(self):
        # todo: fetch best values from current checkpoint (resume)
        pass

    def trigger_epoch(self):
        self.trigger()

    def trigger(self):
        try:
            curr_step, curr_val = self.trainer.monitors.get_history(self.monitor_stat)[-1]
        except (KeyError, IndexError):
            return

        if self.best is None or (curr_val > self.best[1] if self.reverse else curr_val < self.best[1]):
            self.best = (curr_step, curr_val)

            extreme_name = 'maximum' if self.reverse else 'minimum'

            def _escape_name(name):
                return name.replace('/', '-')

            state_dict = self.trainer.state_dict()
            filename = self.filename or ('max-' if self.reverse else 'min-') + _escape_name(self.monitor_stat) + '.pth'
            checkpoint_path = os.path.join(self.checkpoint_dir, filename)

            try:
                torch.save(state_dict, checkpoint_path)
                logger.info('Checkpoint with {} {}={:.5g} saved.'.format(extreme_name, self.monitor_stat, self.best[1]))
            except (OSError, IOError):
                logger.exception('Exception in ModelSaver!')

        # fixme: use min/max instead of best
        self.trainer.monitors.add_scalar(self.monitor_stat + '/best', self.best[1])


class MaxSaver(MinSaver):
    """
    Separately save the model with maximum value of some statistics.
    See docs of :class:`MinSaver` for details.
    """

    def __init__(self, monitor_stat, filename=None, checkpoint_dir=None):
        """
        Args:
            monitor_stat(str): the name of the statistics.
            filename (str): the name for the saved model.
                Defaults to ``max-{monitor_stat}.pth``.
        """
        super(MaxSaver, self).__init__(monitor_stat, True, filename=filename, checkpoint_dir=checkpoint_dir)
