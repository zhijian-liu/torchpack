import re
from collections import defaultdict, deque

import numpy as np
import torch

from torchpack.callbacks.callback import Callback
from torchpack.utils.logging import logger

__all__ = ['Monitor', 'Monitors', 'ScalarPrinter']


class Monitor(Callback):
    """
    Base class for all monitors.
    """

    master_only = True

    def add_scalar(self, name, scalar):
        if isinstance(scalar, np.integer):
            scalar = int(scalar)
        if isinstance(scalar, np.floating):
            scalar = float(scalar)
        assert isinstance(scalar, (int, float)), type(scalar)
        self._add_scalar(name, scalar)

    def _add_scalar(self, name, scalar):
        pass

    def add_image(self, name, tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
        assert isinstance(tensor, np.ndarray), type(tensor)
        if tensor.ndim == 2:
            tensor = tensor[np.newaxis, :, :, np.newaxis]
        elif tensor.ndim == 3:
            if tensor.shape[0] in [1, 3, 4]:
                tensor = np.transpose(tensor, (1, 2, 0))
            if tensor.shape[-1] in [1, 3, 4]:
                tensor = tensor[np.newaxis, ...]
        assert tensor.ndim == 4 and tensor.shape[-1] in [1, 3, 4], tensor.shape
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
        # TODO: track scalar & image history separately (with different `maxlen`)
        self.history = defaultdict(deque)

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
        # TODO: track in `Monitor` or by `HistoryTracker` wrapper
        self.history[name].append((self.trainer.global_step, scalar))
        for monitor in self.monitors:
            monitor.add_scalar(name, scalar)

    def _add_image(self, name, tensor):
        # TODO: track in `Monitor` or by `HistoryTracker` wrapper
        self.history[name].append((self.trainer.global_step, tensor))
        for monitor in self.monitors:
            monitor.add_image(name, tensor)

    def get_history(self, name):
        return self.history[name]

    def get_latest(self, name):
        return self.history[name][-1][1]

    def append(self, monitor):
        self.monitors.append(monitor)

    def extend(self, monitors):
        self.monitors.extend(monitors)

    def __getitem__(self, index):
        return self.monitors[index]

    def __len__(self):
        return len(self.monitors)


class ScalarPrinter(Monitor):
    """
    Print scalar data into terminal.
    """

    def __init__(self, whitelist=None, blacklist=None):
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
        self.blacklist = compile_regex(blacklist)
        self.scalars = dict()

    def _before_train(self):
        self._trigger()

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        def match_regex_list(regexs, name):
            for r in regexs:
                if r.search(name) is not None:
                    return True
            return False

        texts = []
        for k, v in sorted(self.scalars.items()):
            if self.whitelist is None or match_regex_list(self.whitelist, k):
                if self.blacklist is None or not match_regex_list(self.blacklist, k):
                    texts.append('[{}] = {:.5g}'.format(k, v))
        if texts:
            logger.info('\n+ '.join([''] + texts))
        self.scalars = dict()

    def _add_scalar(self, name, scalar):
        self.scalars[name] = scalar
