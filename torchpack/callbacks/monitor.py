import json
import operator
import os
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import six
import tensorflow as tf
from tensorpack.tfutils.summary import create_image_summary, create_scalar_summary

from torchpack.callbacks.callback import Callback
from torchpack.utils.logging import logger

__all__ = ['Monitor', 'Monitors', 'TFEventWriter', 'JSONWriter', 'ScalarPrinter']


class Monitor(Callback):
    """ Base class for monitors which monitor a training progress,
    by processing different types of summary/statistics from trainer.
    """

    chief_only = False

    def add(self, tag, val):
        pass

    def add_scalar(self, tag, val):
        pass

    def add_image(self, name, val):
        pass

    def add_summary(self, summary):
        pass

    def add_event(self, event):
        pass


class Monitors(Monitor):
    """ A container to hold all monitors.
    """

    def __init__(self, monitors):
        for monitor in monitors:
            assert isinstance(monitor, Monitor), type(monitor)
        self.monitors = monitors
        self.scalars = defaultdict(list)

    def set_trainer(self, trainer):
        self.trainer = trainer
        for monitor in self.monitors:
            monitor.set_trainer(trainer)

    def before_train(self):
        for monitor in self.monitors:
            monitor.before_train()

    def after_train(self):
        for monitor in self.monitors:
            monitor.after_train()

    def before_epoch(self):
        for monitor in self.monitors:
            monitor.before_epoch()

    def after_epoch(self):
        for monitor in self.monitors:
            monitor.after_epoch()

    def before_step(self, *args, **kwargs):
        for monitor in self.monitors:
            monitor.before_step(*args, **kwargs)

    def after_step(self, *args, **kwargs):
        for monitor in self.monitors:
            monitor.after_step(*args, **kwargs)

    def trigger_epoch(self):
        for monitor in self.monitors:
            monitor.trigger_epoch()

    def trigger_step(self):
        for monitor in self.monitors:
            monitor.trigger_step()

    def trigger(self):
        for monitor in self.monitors:
            monitor.trigger()

    def add(self, tag, val):
        for monitor in self.monitors:
            monitor.add(tag, val)

    def add_scalar(self, tag, val):
        if isinstance(val, np.integer):
            val = int(val)
        if isinstance(val, np.floating):
            val = float(val)

        self.scalars[tag].append((self.trainer.global_step, val))

        summary = create_scalar_summary(tag, val)
        for monitor in self.monitors:
            monitor.add_scalar(tag, val)
            monitor.add_summary(summary)

    def add_image(self, tag, val):
        assert isinstance(val, np.ndarray), type(val)

        # todo: double check whether transform is correct
        if val.ndim == 2:
            val = val[np.newaxis, :, :, np.newaxis]
        elif val.ndim == 3:
            if val.shape[-1] in [1, 3, 4]:
                val = val[np.newaxis, ...]
            else:
                val = val[..., np.newaxis]
        assert val.ndim == 4, val.shape

        summary = create_image_summary(tag, val)
        for monitor in self.monitors:
            monitor.add_image(tag, val)
            monitor.add_summary(summary)

    def add_summary(self, summary):
        if isinstance(summary, six.binary_type):
            summary = tf.Summary.FromString(summary)
        assert isinstance(summary, tf.Summary), type(summary)

        for val in summary.value:
            if val.WhichOneof('value') == 'simple_value':
                for monitor in self.monitors:
                    monitor.add_scalar(val.tag, val.simple_value)

        for monitor in self.monitors:
            monitor.add_summary(summary)

    def add_event(self, event):
        event.step = self.global_step
        event.wall_time = time.time()
        for monitor in self.monitors:
            monitor.add_event(event)

    def get_latest(self, name):
        return self.scalars[name][-1][1]

    def get_history(self, name):
        return self.scalars[name]


class TFEventWriter(Monitor):
    """ Write summaries to TensorFlow event file.
    """

    def __init__(self, logdir=None, max_queue=10, flush_secs=120, split_files=False):
        """
        Args:
            logdir: ``logger.get_logger_dir()`` by default.
            max_queue, flush_secs: Same as in :class:`tf.summary.FileWriter`.
            split_files: if True, split events to multiple files rather than
                append to a single file. Useful on certain filesystems where append is expensive.
        """
        if logdir is None:
            logdir = logger.get_logger_dir()
        assert tf.gfile.IsDirectory(logdir), logdir
        self._logdir = logdir
        self._max_queue = max_queue
        self._flush_secs = flush_secs
        self._split_files = split_files

    def __new__(cls, logdir=None, max_queue=10, flush_secs=120, **kwargs):
        if logdir is None:
            logdir = logger.get_logger_dir()

        if logdir is not None:
            return super(TFEventWriter, cls).__new__(cls)
        else:
            logger.warn('logger directory was not set. Ignore TFEventWriter.')
            return Monitor()

    def before_train(self):
        self.writer = tf.summary.FileWriter(
            self._logdir, graph=tf.get_default_graph(),
            max_queue=self._max_queue, flush_secs=self._flush_secs)

    def trigger_epoch(self):
        self.trigger()

    def trigger(self):
        self.writer.flush()
        if self._split_files:
            self.writer.close()
            self.writer.reopen()

    def after_train(self):
        self.writer.close()

    def add_summary(self, summary):
        self.writer.add_summary(summary, self.trainer.global_step)

    def add_event(self, event):
        self.writer.add_event(event)


class JSONWriter(Monitor):
    """
    Write all scalar data to a json file under ``logger.get_logger_dir()``, grouped by their global step.
    If found an earlier json history file, will append to it.
    """

    FILENAME = 'stats.json'
    """
    The name of the json file. Do not change it.
    """

    def __new__(cls):
        if logger.get_logger_dir():
            return super(JSONWriter, cls).__new__(cls)
        else:
            logger.warn("logger directory was not set. Ignore JSONWriter.")
            return Monitor()

    @staticmethod
    def load_existing_json():
        """
        Look for an existing json under :meth:`logger.get_logger_dir()` named "stats.json",
        and return the loaded list of statistics if found. Returns None otherwise.
        """
        dir = logger.get_logger_dir()
        fname = os.path.join(dir, JSONWriter.FILENAME)
        if tf.gfile.Exists(fname):
            with open(fname) as f:
                stats = json.load(f)
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

    # initialize the stats here, because before_train from other callbacks may use it
    def before_train(self):
        self._stats = []
        self._stat_now = {}
        self._last_gs = -1

        stats = JSONWriter.load_existing_json()
        self._fname = os.path.join(logger.get_logger_dir(), JSONWriter.FILENAME)
        if stats is not None:
            try:
                epoch = stats[-1]['epoch_num'] + 1
            except Exception:
                epoch = None

            # check against the current training settings
            # therefore this logic needs to be in before_train stage
            starting_epoch = self.trainer.loop.starting_epoch
            if epoch is None or epoch == starting_epoch:
                logger.info("Found existing JSON inside {}, will append to it.".format(logger.get_logger_dir()))
                self._stats = stats
            else:
                logger.warning(
                    "History epoch={} from JSON is not the predecessor of the current starting_epoch={}".format(
                        epoch - 1, starting_epoch))
                logger.warning("If you want to resume old training, either use `AutoResumeTrainConfig` "
                               "or correctly set the new starting_epoch yourself to avoid inconsistency. ")

                backup_fname = JSONWriter.FILENAME + '.' + datetime.now().strftime('%m%d-%H%M%S')
                backup_fname = os.path.join(logger.get_logger_dir(), backup_fname)

                logger.warn("Now, we will train with starting_epoch={} and backup old json to {}".format(
                    self.trainer.loop.starting_epoch, backup_fname))
                shutil.move(self._fname, backup_fname)

        # in case we have something to log here.
        self.trigger()

    def trigger_step(self):
        # will do this in trigger_epoch
        if self.local_step != self.trainer.steps_per_epoch - 1:
            self.trigger()

    def trigger_epoch(self):
        self.trigger()

    def trigger(self):
        """
        Add stats to json and dump to disk.
        Note that this method is idempotent.
        """
        if len(self._stat_now):
            self._stat_now['epoch_num'] = self.trainer.epoch_num
            self._stat_now['global_step'] = self.trainer.global_step

            self._stats.append(self._stat_now)
            self._stat_now = {}
            self._write_stat()

    def _write_stat(self):
        tmp_filename = self._fname + '.tmp'
        try:
            with open(tmp_filename, 'w') as f:
                json.dump(self._stats, f)
            shutil.move(tmp_filename, self._fname)
        except IOError:  # disk error sometimes..
            logger.exception("Exception in JSONWriter._write_stat()!")

    def add_scalar(self, tag, val):
        self._stat_now[tag] = val


class ScalarPrinter(Monitor):
    """ Print scalar data into terminal.
    """

    def __init__(self, trigger_epoch=True, trigger_step=False,
                 whitelist=None, blacklist=None):
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

        self._whitelist = compile_regex(whitelist)
        if blacklist is None:
            blacklist = []
        self._blacklist = compile_regex(blacklist)

        self._enable_step = trigger_step
        self._enable_epoch = trigger_epoch
        self._dic = {}

    def before_train(self):
        self.trigger()

    def trigger_step(self):
        if self._enable_step:
            if self.trainer.local_step != self.trainer.steps_per_epoch - 1:
                # not the last step
                self.trigger()
            else:
                if not self._enable_epoch:
                    self.trigger()
                # otherwise, will print them together

    def trigger_epoch(self):
        if self._enable_epoch:
            self.trigger()

    def trigger(self):
        # Print stats here
        def match_regex_list(regexs, name):
            for r in regexs:
                if r.search(name) is not None:
                    return True
            return False

        texts = []
        for k, v in sorted(self._dic.items(), key=operator.itemgetter(0)):
            if self._whitelist is None or match_regex_list(self._whitelist, k):
                if not match_regex_list(self._blacklist, k):
                    texts.append('[{}] = {:.5g}'.format(k, v))

        if texts:
            logger.info('\n+ '.join([''] + texts))

        self._dic = {}

    def add_scalar(self, tag, val):
        self._dic[tag] = float(val)
