import traceback
from time import perf_counter as timer

from tensorpack.utils.utils import humanize_time_delta

from torchpack.callbacks.callback import CallbackGroup
from torchpack.utils.logging import logger

__all__ = ['TimedCallbackGroup']


class TimedCallbackGroup(CallbackGroup):
    def trigger_epoch(self):
        self.trigger()

    def trigger(self):
        tot = 0
        timers = []

        for callback in self.callbacks:
            start_time = timer()
            callback.trigger_epoch()
            duration = timer() - start_time
            tot += duration
            timers.append((duration, str(callback)))

        timers = sorted(timers, reverse=True)[:5]
        if tot >= 1:
            texts = ['[{}] took {} in total.'.format(str(self), humanize_time_delta(tot))]
            for t, name in timers:
                texts.append('[{}] took {}.'.format(name, humanize_time_delta(t)))
            logger.info('\n+ '.join(texts))

    def after_train(self):
        # make sure all callbacks are properly finalized
        for callback in self.callbacks:
            try:
                callback.after_train()
            except Exception:
                traceback.print_exc()
