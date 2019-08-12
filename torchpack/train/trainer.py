import traceback
import weakref
from time import perf_counter as timer

from tensorpack.utils.utils import humanize_time_delta

from torchpack.callbacks import Monitor, Monitors
from torchpack.callbacks.base import Callbacks, Callback
from torchpack.cuda.copy import async_copy_to
from torchpack.train.exception import StopTraining
from torchpack.utils.logging import logger

__all__ = ['Trainer']


class Trainer(object):
    """
    Base class for a trainer.
    """

    is_chief = True
    """
    Whether this process is the chief worker in distributed training.
    Certain callbacks will only be run by chief worker.
    """

    # fixme: too ugly
    def __init__(self, device='cuda'):
        self.device = device
        self.callbacks = None
        self.epoch_num = 0
        self.global_step = -1
        self.local_step = -1

    def setup_callbacks(self, callbacks, monitors):
        assert isinstance(callbacks, list), callbacks
        for callback in callbacks:
            assert isinstance(callback, Callback), type(callback)

        assert isinstance(monitors, list), monitors
        for monitor in monitors:
            assert isinstance(monitor, Monitor), type(monitor)

        self.callbacks = callbacks
        self.monitors = monitors

        self.monitors = Monitors(self.monitors)
        self.callbacks = Callbacks(self.callbacks + [self.monitors])
        self.callbacks.set_trainer(weakref.proxy(self))

    def run_step(self, feed_dict):
        """
        Defines what to do in one iteration.
        """
        # raise NotImplementedError
        return self.model(feed_dict)

    def main_loop(self, steps_per_epoch, starting_epoch, max_epoch):
        """
        Run the main training loop.

        Args:
            steps_per_epoch, starting_epoch, max_epoch (int):
        """

        self.steps_per_epoch = int(steps_per_epoch)
        self.starting_epoch = int(starting_epoch)
        self.max_epoch = int(max_epoch)

        # Allow empty epoch (no steps), if we want to run the callbacks only.
        assert self.steps_per_epoch >= 0 and self.max_epoch >= 0

        self.epoch_num = starting_epoch - 1
        self.global_step = self.epoch_num * self.steps_per_epoch

        try:
            self.callbacks.before_train()
            for self.epoch_num in range(self.starting_epoch, self.max_epoch + 1):
                logger.info('Training epoch {}/{} started.'.format(self.epoch_num, self.max_epoch))

                start_time = timer()
                self.callbacks.before_epoch()

                self.model.train()
                for self.local_step, feed_dict in enumerate(self.dataflow):
                    feed_dict = async_copy_to(feed_dict, device=self.device)
                    self.global_step += 1

                    self.callbacks.before_step()
                    self.run_step(feed_dict)
                    self.callbacks.after_step()

                    self.callbacks.trigger_step()

                self.callbacks.after_epoch()
                logger.info('Training epoch finished in {}.'.format(humanize_time_delta(timer() - start_time)))

                text = ['Training epoch finished in {}.'.format(humanize_time_delta(timer() - start_time))]
                for callback in self.callbacks:
                    start_time = timer()
                    callback.trigger_epoch()
                    duration = timer() - start_time
                    if duration >= 1e-2:
                        text.append('[{}] took {}.'.format(str(callback), humanize_time_delta(timer() - start_time)))
                logger.info('\n+ '.join(text))

            logger.info('Training has finished!')
        except StopTraining as e:
            logger.info('Training was stopped by exception {}.'.format(str(e)))
        except KeyboardInterrupt:
            logger.info('Detected Ctrl-C and exiting main loop.')
            raise
        finally:
            # make sure all callbacks are properly finalized
            for callback in self.callbacks:
                try:
                    callback.after_train()
                except Exception:
                    traceback.print_exc()

    def train(self, dataflow, model,
              callbacks=None, monitors=None,
              steps_per_epoch=None, starting_epoch=1, max_epoch=9999999):
        """
        Implemented by two lines:

        .. code-block:: python

            self.setup_callbacks(callbacks, monitors)
            self.main_loop(steps_per_epoch, starting_epoch, max_epoch)

        You can call those methods by yourself to have better control on details if needed.
        """
        self.dataflow = dataflow
        self.model = model
        steps_per_epoch = len(self.dataflow)
        self.setup_callbacks(callbacks, monitors)
        self.main_loop(steps_per_epoch, starting_epoch, max_epoch)

    def state_dict(self):
        return dict(model=self.model.state_dict())

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])

    def __new__(cls, *args, **kwargs):
        return super(Trainer, cls).__new__(cls)
