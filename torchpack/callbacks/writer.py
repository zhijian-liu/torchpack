import os.path as osp

from tensorboardX import SummaryWriter

import torchpack.utils.fs as fs
import torchpack.utils.io as io
from torchpack.callbacks.monitor import Monitor
from torchpack.environ import get_default_dir
from torchpack.logging import logger
from torchpack.utils.matching import IENameMatcher

__all__ = ['ConsoleWriter', 'TFEventWriter', 'JSONWriter']


class ConsoleWriter(Monitor):
    """
    Write scalar summaries into terminal.
    This callback is one of the :func:`DEFAULT_CALLBACKS()`.
    """
    def __init__(self, include='*', exclude=None):
        self.matcher = IENameMatcher(include, exclude)

    def _before_train(self):
        self.scalars = dict()

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        texts = []
        for name, scalar in sorted(self.scalars.items()):
            if self.matcher.match(name):
                texts.append('[{}] = {:.5g}'.format(name, scalar))
        if texts:
            logger.info('\n+ '.join([''] + texts))
        self.scalars.clear()

    def _add_scalar(self, name, scalar):
        self.scalars[name] = scalar


class TFEventWriter(Monitor):
    """
    Write summaries to TensorFlow event file.
    This callback is one of the :func:`DEFAULT_CALLBACKS()`.
    """
    def __init__(self, save_dir=None):
        self.save_dir = fs.makedir(save_dir or \
                                   osp.join(get_default_dir(), 'tensorboard'))

    def _before_train(self):
        self.writer = SummaryWriter(self.save_dir)

    def _after_train(self):
        self.writer.close()

    def _add_scalar(self, name, scalar):
        self.writer.add_scalar(name, scalar, self.trainer.global_step)

    def _add_image(self, name, tensor):
        self.writer.add_image(name, tensor, self.trainer.global_step)


class JSONWriter(Monitor):
    """
    Write scalar summaries to JSON file.
    This callback is one of the :func:`DEFAULT_CALLBACKS()`.
    """
    def __init__(self, save_dir=None):
        self.save_dir = fs.makedir(save_dir or \
                                   osp.join(get_default_dir(), 'summaries'))

    def _before_train(self):
        self.summaries = []

        filename = osp.join(self.save_dir, 'scalars.json')
        if not osp.exists(filename):
            return

        self.summaries = io.load(filename)
        try:
            epoch = self.summaries[-1]['epoch_num'] + 1
        except:
            return
        if epoch != self.trainer.starting_epoch:
            logger.warning(
                'History epoch={} from JSON is not the predecessor of the current starting_epoch={}'
                .format(epoch - 1, self.trainer.starting_epoch))
            logger.warning(
                'If you want to resume old training, either use `AutoResumeTrainConfig` '
                'or correctly set the new starting_epoch yourself to avoid inconsistency.'
            )

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        filename = osp.join(self.save_dir, 'scalars.json')
        try:
            io.save(filename, self.summaries)
        except (OSError, IOError):
            logger.exception(
                'Error occurred when saving JSON file "{}".'.format(filename))

    def _after_train(self):
        self._trigger()

    def _add_scalar(self, name, scalar):
        self.summaries.append({
            'epoch-num': self.trainer.epoch_num,
            'global-step': self.trainer.global_step,
            'local-step': self.trainer.local_step,
            name: scalar
        })
