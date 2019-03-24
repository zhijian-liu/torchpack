import traceback
from time import perf_counter as timer

from .base import Callback
from ..utils import logger
from ..utils.utils import humanize_time

__all__ = ['Callbacks']


class Callbacks(Callback):
    """
    A container to hold all callbacks, and trigger them iteratively.
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
        total_time = 0
        times = []

        for callback in self.callbacks:
            start = timer()
            callback.trigger_epoch()
            time = timer() - start

            total_time += time
            times.append((str(callback), time))

        if total_time < 3:
            return

        messages = []
        for name, time in self.times:
            if time / self.total_time > 0.3 and time > 1:
                messages.append(name + ': ' + humanize_time(time))

        logger.info('Callbacks took {:.3f} sec in total. {}'.format(self.total_time, '; '.join(messages)))

    def _before_epoch(self):
        for callback in self.callbacks:
            callback.before_epoch()

    def _after_epoch(self):
        for callback in self.callbacks:
            callback.after_epoch()

    def _before_run(self):
        for callback in self.callbacks:
            callback.before_run()

    def _after_run(self):
        for callback in self.callbacks:
            callback.after_run()
