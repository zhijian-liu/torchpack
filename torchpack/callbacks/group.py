import traceback
from time import perf_counter as timer

from tensorpack.utils.utils import humanize_time_delta

from torchpack.utils.logging import logger
from .base import Callback

__all__ = ['CallbackGroup', 'TimedCallbackGroup']


class CallbackGroup(Callback):
    """ A container to hold all callbacks, and trigger them iteratively.
    """

    def __init__(self, callbacks):
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
        # make sure all callbacks are properly finalized
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
        for callback in self.callbacks:
            callback.trigger_step()

    def trigger_step(self):
        for callback in self.callbacks:
            callback.trigger_step()

    def trigger(self):
        for callback in self.callbacks:
            callback.trigger()


class TimedCallbackGroup(CallbackGroup):
    def trigger_epoch(self):
        self.trigger()

    def trigger(self):
        tot = 0
        timers = []

        for callback in self.callbacks:
            s = timer()
            callback.trigger_epoch()
            duration = timer() - s
            tot += duration
            timers.append((str(callback), duration))

        if tot >= 1:
            texts = ['[{}] took {} in total.'.format(str(self), humanize_time_delta(tot))]
            for name, t in timers:
                if t >= 1:
                    texts.append('[{}] took {}.'.format(name, humanize_time_delta(t)))
            logger.info('\n+ '.join(texts))
