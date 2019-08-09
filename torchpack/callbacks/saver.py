import os
from datetime import datetime

from tensorpack.compat import tfv1 as tf
from tensorpack.utils import logger
from .base import Callback

__all__ = [ 'MinSaver', 'MaxSaver']

class MinSaver(Callback):
    """
    Separately save the model with minimum value of some statistics.
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

    def _get_stat(self):
        try:
            v = self.trainer.monitors.get_history(self.monitor_stat)[-1]
        except (KeyError, IndexError):
            v = None, None
        return v

    def trigger(self):
        curr_step, curr_val = self._get_stat()
        if curr_step is None:
            return

        if self.best is None or (curr_val > self.best[1] if self.reverse else curr_val < self.best[1]):
            self.best = (curr_step, curr_val)
            print(self.best, 'best-checkpoint')
            # self._save()

    def _save(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt is None:
            raise RuntimeError(
                "[MinSaver] Cannot find a checkpoint state. Do you forget to use ModelSaver?")
        path = ckpt.model_checkpoint_path

        extreme_name = 'maximum' if self.reverse else 'minimum'
        if not path.endswith(str(self.best[0])):
            logger.warn("[MinSaver] New {} '{}' found at global_step={}, but the latest checkpoint is {}.".format(
                extreme_name, self.monitor_stat, self.best[0], path
            ))
            logger.warn("MinSaver will do nothing this time. "
                        "The callbacks may have inconsistent frequency or wrong order.")
            return

        newname = os.path.join(self.checkpoint_dir,
                               self.filename or
                               ('max-' + self.monitor_stat if self.reverse else 'min-' + self.monitor_stat))
        files_to_copy = tf.gfile.Glob(path + '*')
        for file_to_copy in files_to_copy:
            tf.gfile.Copy(file_to_copy, file_to_copy.replace(path, newname), overwrite=True)
        logger.info("Model at global_step={} with {} {}={:.5g} saved.".format(
            self.best[0], extreme_name, self.monitor_stat, self.best[1]))


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
                Defaults to ``max-{monitor_stat}.tfmodel``.
        """
        super(MaxSaver, self).__init__(monitor_stat, True, filename=filename, checkpoint_dir=checkpoint_dir)