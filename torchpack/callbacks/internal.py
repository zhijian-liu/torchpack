import traceback
from time import perf_counter as timer

from tensorpack.utils.utils import humanize_time_delta

from torchpack.callbacks.group import CallbackGroup
from torchpack.utils.logging import logger

__all__ = ['TimedCallbackGroup']


class TimedCallbackGroup(CallbackGroup):
    def trigger_epoch(self):
        self.trigger()

    def trigger(self):
        times = []
        for callback in self.callbacks:
            start_time = timer()
            callback.trigger_epoch()
            times.append((timer() - start_time, str(callback)))

        total_time = sum(time for time, _ in times)

        times = sorted(times, reverse=True)[:5]

        if total_time >= 1:
            texts = ['[{}] took {} in total.'.format(str(self), humanize_time_delta(total_time))]
            for time, name in times:
                texts.append('[{}] took {}.'.format(name, humanize_time_delta(time)))
            logger.info('\n+ '.join(texts))

    def after_train(self):
        # make sure all callbacks are properly finalized
        for callback in self.callbacks:
            try:
                callback.after_train()
            except Exception:
                traceback.print_exc()
