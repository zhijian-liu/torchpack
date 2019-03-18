import traceback
from contextlib import contextmanager
from time import perf_counter as timer

from .base import Callback
from ..utils import logger
from ..utils.utils import humanize_time_delta

__all__ = ['Callbacks']


class CallbackTimeLogger(object):
    def __init__(self):
        self.times = []
        self.tot = 0

    def add(self, name, time):
        self.tot += time
        self.times.append((name, time))

    @contextmanager
    def timed_callback(self, name):
        start = timer()
        yield
        self.add(name, timer() - start)

    def log(self):
        if self.tot < 3:
            return
        msgs = []
        for name, t in self.times:
            if t / self.tot > 0.3 and t > 1:
                msgs.append(name + ": " + humanize_time_delta(t))
        logger.info("Callbacks took {:.3f} sec in total. {}".format(self.tot, '; '.join(msgs)))


class Callbacks(Callback):
    """
    A container to hold all callbacks, and trigger them iteratively.
    Note that it does nothing to before_run/after_run.
    """

    def __init__(self, callbacks):
        """
        Args:
            callbacks(list): a list of :class:`Callback` instances.
        """
        super(Callbacks, self).__init__()
        for callback in callbacks:
            assert isinstance(callback, Callback), callback.__class__
        self.callbacks = callbacks

    def _before_train(self):
        for callback in self.callbacks:
            callback.before_train()

    def _after_train(self):
        for callback in self.callbacks:
            # make sure callbacks are properly finalized
            try:
                callback.after_train()
            except Exception:
                traceback.print_exc()

    def trigger_step(self):
        for callback in self.callbacks:
            callback.trigger_step()

    def _trigger_epoch(self):
        logger = CallbackTimeLogger()

        for callback in self.callbacks:
            name = str(callback)
            with logger.timed_callback(name):
                callback.trigger_epoch()

        logger.log()

    def _before_epoch(self):
        for callback in self.callbacks:
            callback.before_epoch()

    def _after_epoch(self):
        for callback in self.callbacks:
            callback.after_epoch()
