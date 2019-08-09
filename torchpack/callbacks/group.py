import traceback
from time import perf_counter as timer

from tensorpack.utils.utils import humanize_time_delta

from torchpack.utils.logging import logger
from .base import CallbackGroup

__all__ = ['TimedCallbackGroup']


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

    def after_train(self):
        # make sure all callbacks are properly finalized
        for callback in self.callbacks:
            try:
                callback.after_train()
            except Exception:
                traceback.print_exc()
