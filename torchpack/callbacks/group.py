import traceback
from contextlib import contextmanager
from time import perf_counter as timer

from tensorpack.utils.utils import humanize_time_delta

from torchpack.utils.logging import logger
from .callback import Callback

__all__ = ['CallbackGroup']


class CallbackTimeLogger(object):
    def __init__(self):
        self.times = []
        self.tot = 0

    def add(self, name, time):
        self.tot += time
        self.times.append((name, time))

    @contextmanager
    def timed_callback(self, name):
        s = timer()
        yield
        self.add(name, timer() - s)

    def log(self):
        if self.tot < 3:
            return
        msgs = []
        for name, t in self.times:
            if t / self.tot > 0.3 and t > 1:
                msgs.append(name + ": " + humanize_time_delta(t))
        logger.info(
            "Callbacks took {:.3f} sec in total. {}".format(
                self.tot, '; '.join(msgs)))


class CallbackGroup(Callback):
    """
    A container to hold all callbacks, and trigger them iteratively.
    """

    def __init__(self, callbacks):
        """
        Args:
            callbacks(list): a list of :class:`Callback` instances.
        """
        # check type
        for callback in callbacks:
            assert isinstance(callback, Callback), callback.__class__
        self.callbacks = callbacks

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def before_train(self):
        for callback in self.callbacks:
            callback.before_train()

    def after_train(self):
        # make sure callbacks are properly finalized
        for callback in self.callbacks:
            try:
                callback.after_train()
            except Exception:
                traceback.print_exc()

    def before_epoch(self):
        for callback in self.callbacks:
            callback.before_epoch()

    def after_epoch(self):
        for callback in self.callbacks:
            callback.after_epoch()

    def before_step(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.before_step(*args, **kwargs)

    def after_step(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.after_step(*args, **kwargs)

    def trigger_epoch(self):
        tm = CallbackTimeLogger()
        for callback in self.callbacks:
            name = str(callback)
            with tm.timed_callback(name):
                callback.trigger_epoch()
        tm.log()

    def trigger_step(self):
        for callback in self.callbacks:
            callback.trigger_step()

    def trigger(self):
        for callback in self.callbacks:
            callback.trigger()
