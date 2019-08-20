import json
import os
import shutil

from tensorboardX import SummaryWriter

from torchpack.callbacks.monitor import Monitor
from torchpack.utils.logging import logger, get_logger_dir

__all__ = ['TFEventWriter', 'JSONWriter']


class TFEventWriter(Monitor):
    """
    Write summaries to TensorFlow event file.
    """

    def __init__(self, logdir=None):
        self.logdir = os.path.normpath(logdir or get_logger_dir())
        os.makedirs(self.logdir, exist_ok=True)

    def _before_train(self):
        self.writer = SummaryWriter(self.logdir)

    def _after_train(self):
        self.writer.close()

    def _add_scalar(self, name, scalar):
        self.writer.add_scalar(name, scalar, self.trainer.global_step)

    def _add_image(self, name, tensor):
        self.writer.add_image(name, tensor, self.trainer.global_step)


class JSONWriter(Monitor):
    """
    Write scalar summaries to JSON file.
    """

    FILENAME = 'stats.json'

    def __init__(self, logdir=None):
        self.logdir = os.path.normpath(logdir or get_logger_dir())
        os.makedirs(self.logdir, exist_ok=True)

    def load_existing_json(self):
        """
        Look for an existing json under :meth:`logger.get_logger_dir()` named "stats.json",
        and return the loaded list of statistics if found. Returns None otherwise.
        """
        filename = os.path.join(self.logdir, JSONWriter.FILENAME)
        if os.path.exists(filename):
            with open(filename) as fp:
                stats = json.load(fp)
            assert isinstance(stats, list), type(stats)
            return stats
        return None

    def load_existing_epoch_number(self):
        """
        Try to load the latest epoch number from an existing json stats file (if any).
        Returns None if not found.
        """
        stats = self.load_existing_json()
        try:
            return int(stats[-1]['epoch_num'])
        except:
            return None

    def _before_train(self):
        self.records = []

        stats = self.load_existing_json()
        if stats is not None:
            try:
                epoch = stats[-1]['epoch_num'] + 1
            except:
                epoch = None

            if epoch is not None and epoch != self.trainer.starting_epoch:
                logger.warning(
                    'History epoch={} from JSON is not the predecessor of the current starting_epoch={}'.format(
                        epoch - 1, self.trainer.starting_epoch))
                logger.warning('If you want to resume old training, either use `AutoResumeTrainConfig` '
                               'or correctly set the new starting_epoch yourself to avoid inconsistency.')

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        filename = os.path.join(self.logdir, self.FILENAME)
        try:
            with open(filename + '.tmp', 'w') as fp:
                json.dump(self.records, fp)
            shutil.move(filename + '.tmp', filename)
        except (OSError, IOError):
            logger.exception('Error occurred when saving JSON file "{}".'.format(filename))

    def _after_train(self):
        self._trigger()

    def _add_scalar(self, name, scalar):
        if not self.records or self.records[-1]['global_step'] != self.trainer.global_step:
            self.records += [{'epoch_num': self.trainer.epoch_num, 'global_step': self.trainer.global_step}]
        self.records[-1][name] = scalar
