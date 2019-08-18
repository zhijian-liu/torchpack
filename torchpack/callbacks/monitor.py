import json
import operator
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime

import numpy as np
from tensorboardX import SummaryWriter

from torchpack.callbacks.callback import Callback
from torchpack.utils.logging import logger, get_logger_dir

__all__ = ['Monitor', 'Monitors', 'TFEventWriter', 'JSONWriter', 'ScalarPrinter']


class Monitor(Callback):
    """
    Base class for monitors which monitor a training progress,
    by processing different types of summary/statistics from trainer.
    """

    master_only = True

    def add_scalar(self, name, scalar):
        if isinstance(scalar, np.integer):
            scalar = int(scalar)
        if isinstance(scalar, np.floating):
            scalar = float(scalar)
        self._add_scalar(name, scalar)

    def _add_scalar(self, name, scalar):
        pass

    def add_image(self, name, tensor):
        assert isinstance(tensor, np.ndarray), type(tensor)
        if tensor.ndim == 2:
            tensor = tensor[np.newaxis, :, :, np.newaxis]
        elif tensor.ndim == 3:
            # TODO: check whether the transform is correct
            if tensor.shape[-1] in [1, 3, 4]:
                tensor = tensor[np.newaxis, ...]
            else:
                tensor = tensor[..., np.newaxis]
        assert tensor.ndim == 4, tensor.shape
        self._add_image(name, tensor)

    def _add_image(self, name, tensor):
        pass


class Monitors(Monitor):
    """
    A container to hold all monitors.
    """

    def __init__(self, monitors):
        for monitor in monitors:
            assert isinstance(monitor, Monitor), type(monitor)
        self.monitors = monitors
        self.scalars = defaultdict(list)

    def _set_trainer(self, trainer):
        for monitor in self.monitors:
            monitor.set_trainer(trainer)

    def _before_train(self):
        for monitor in self.monitors:
            monitor.before_train()

    def _before_epoch(self):
        for monitor in self.monitors:
            monitor.before_epoch()

    def _before_step(self, *args, **kwargs):
        for monitor in self.monitors:
            monitor.before_step(*args, **kwargs)

    def _after_step(self, *args, **kwargs):
        for monitor in self.monitors:
            monitor.after_step(*args, **kwargs)

    def _trigger_step(self):
        for monitor in self.monitors:
            monitor.trigger_step()

    def _after_epoch(self):
        for monitor in self.monitors:
            monitor.after_epoch()

    def _trigger_epoch(self):
        for monitor in self.monitors:
            monitor.trigger_epoch()

    def _trigger(self):
        for monitor in self.monitors:
            monitor.trigger()

    def _after_train(self):
        for monitor in self.monitors:
            monitor.after_train()

    def _add_scalar(self, name, scalar):
        # TODO: track scalar/image history in `Monitor.add()`
        self.scalars[name].append((self.trainer.global_step, scalar))
        for monitor in self.monitors:
            monitor.add_scalar(name, scalar)

    def _add_image(self, name, tensor):
        for monitor in self.monitors:
            monitor.add_image(name, tensor)

    def get_latest(self, name):
        return self.scalars[name][-1][1]

    def get_history(self, name):
        return self.scalars[name]


class TFEventWriter(Monitor):
    """
    Write summaries to TensorFlow event file.
    """

    def __init__(self, logdir=None):
        if logdir is None:
            logdir = get_logger_dir()
        self.logdir = logdir

    def _before_train(self):
        self.writer = SummaryWriter(self.logdir)

    def _add_scalar(self, name, scalar):
        self.writer.add_scalar(name, scalar, self.trainer.global_step)

    def _add_image(self, name, tensor):
        self.writer.add_image(name, tensor, self.trainer.global_step)

    def _after_train(self):
        self.writer.close()


class JSONWriter(Monitor):
    """
    Write all scalar data to a json file under ``logger.get_logger_dir()``, grouped by their global step.
    If found an earlier json history file, will append to it.
    """

    FILENAME = 'stats.json'

    @staticmethod
    def load_existing_json():
        """
        Look for an existing json under :meth:`logger.get_logger_dir()` named "stats.json",
        and return the loaded list of statistics if found. Returns None otherwise.
        """
        logdir = get_logger_dir()
        filename = os.path.join(logdir, JSONWriter.FILENAME)
        if os.path.exists(filename):
            with open(filename) as fp:
                stats = json.load(fp)
            assert isinstance(stats, list), type(stats)
            return stats
        return None

    @staticmethod
    def load_existing_epoch_number():
        """
        Try to load the latest epoch number from an existing json stats file (if any).
        Returns None if not found.
        """
        stats = JSONWriter.load_existing_json()
        try:
            return int(stats[-1]['epoch_num'])
        except Exception:
            return None

    def _before_train(self):
        self.records = []
        self.record = dict()

        stats = JSONWriter.load_existing_json()
        self.filename = os.path.join(get_logger_dir(), JSONWriter.FILENAME)
        if stats is not None:
            try:
                epoch = stats[-1]['epoch_num'] + 1
            except Exception:
                epoch = None

            starting_epoch = self.trainer.starting_epoch
            if epoch is None or epoch == starting_epoch:
                logger.info('Found existing JSON inside {}, will append to it.'.format(get_logger_dir()))
                self.records = stats
            else:
                logger.warning(
                    'History epoch={} from JSON is not the predecessor of the current starting_epoch={}'.format(
                        epoch - 1, starting_epoch))
                logger.warning('If you want to resume old training, either use `AutoResumeTrainConfig` '
                               'or correctly set the new starting_epoch yourself to avoid inconsistency.')

                backup_fname = JSONWriter.FILENAME + '.' + datetime.now().strftime('%m%d-%H%M%S')
                backup_fname = os.path.join(get_logger_dir(), backup_fname)

                logger.warning('Now, we will train with starting_epoch={} and backup old json to {}'.format(
                    self.trainer.starting_epoch, backup_fname))
                shutil.move(self.filename, backup_fname)

        self._trigger()

    def _trigger_step(self):
        # will do this in trigger_epoch
        if self.trainer.local_step != self.trainer.steps_per_epoch - 1:
            self._trigger()

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        if len(self.record):
            self.record['epoch_num'] = self.trainer.epoch_num
            self.record['global_step'] = self.trainer.global_step

            self.records.append(self.record)
            self.record = dict()

            tmp_filename = self.filename + '.tmp'
            try:
                with open(tmp_filename, 'w') as fp:
                    json.dump(self.records, fp)
                shutil.move(tmp_filename, self.filename)
            except IOError:
                logger.exception('Error occurred in JSONWriter._write_stat()!')

    def _add_scalar(self, name, scalar):
        self.record[name] = scalar


class ScalarPrinter(Monitor):
    """
    Print scalar data into terminal.
    """

    def __init__(self, trigger_epoch=True, trigger_step=False, whitelist=None, blacklist=None):
        """
        Args:
            enable_step, enable_epoch (bool): whether to print the
                monitor data (if any) between steps or between epochs.
            whitelist (list[str] or None): A list of regex. Only names
                matching some regex will be allowed for printing.
                Defaults to match all names.
            blacklist (list[str] or None): A list of regex. Names matching
                any regex will not be printed. Defaults to match no names.
        """

        def compile_regex(rs):
            if rs is None:
                return None
            rs = set([re.compile(r) for r in rs])
            return rs

        self.whitelist = compile_regex(whitelist)
        if blacklist is None:
            blacklist = []
        self.blacklist = compile_regex(blacklist)

        self.enable_epoch = trigger_epoch
        self.enable_step = trigger_step
        self.scalars = dict()

    def _before_train(self):
        self._trigger()

    def _trigger_step(self):
        if self.enable_step:
            if self.trainer.local_step != self.trainer.steps_per_epoch - 1:
                self._trigger()
            elif not self.enable_epoch:
                self._trigger()

    def _trigger_epoch(self):
        if self.enable_epoch:
            self._trigger()

    def _trigger(self):
        def match_regex_list(regexs, name):
            for r in regexs:
                if r.search(name) is not None:
                    return True
            return False

        texts = []
        for k, v in sorted(self.scalars.items(), key=operator.itemgetter(0)):
            if self.whitelist is None or match_regex_list(self.whitelist, k):
                if not match_regex_list(self.blacklist, k):
                    texts.append('[{}] = {:.5g}'.format(k, v))

        if texts:
            logger.info('\n+ '.join([''] + texts))

        self.scalars = dict()

    def _add_scalar(self, name, scalar):
        self.scalars[name] = scalar
