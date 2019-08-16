import time
import traceback
import weakref

from tensorpack.utils.utils import humanize_time_delta

from torchpack.callbacks import Monitor, Monitors
from torchpack.callbacks.callback import Callbacks, Callback
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

    def __init__(self, device='cuda'):
        self.device = device
        self.callbacks = None
        self.epoch_num = 0
        self.global_step = -1
        self.local_step = -1

    def set_callbacks(self, callbacks, monitors):
        for callback in callbacks:
            assert isinstance(callback, Callback), type(callback)
        for monitor in monitors:
            assert isinstance(monitor, Monitor), type(monitor)
        self.monitors = Monitors(monitors)
        self.callbacks = Callbacks(callbacks + [self.monitors])
        self.callbacks.set_trainer(weakref.proxy(self))

    def run_step(self, feed_dict):
        self.callbacks.before_step(feed_dict)
        self._run_step(feed_dict)
        self.callbacks.after_step()

    def _run_step(self, feed_dict):
        """
        Defines what to do in one iteration.
        """
        # raise NotImplementedError
        outputs = self.model(feed_dict)
        return outputs

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
            start_train = time.time()
            self.callbacks.before_train()

            for self.epoch_num in range(self.starting_epoch, self.max_epoch + 1):
                logger.info('Epoch {}/{} started.'.format(self.epoch_num, self.max_epoch))

                start_epoch = time.time()
                self.callbacks.before_epoch()

                self.model.train()
                for self.local_step, feed_dict in enumerate(self.dataflow):
                    feed_dict = async_copy_to(feed_dict, device=self.device)

                    self.global_step += 1
                    self.run_step(feed_dict)
                    self.callbacks.trigger_step()

                self.callbacks.after_epoch()
                logger.info('Training finished in {}.'.format(humanize_time_delta(time.time() - start_epoch)))

                self.callbacks.trigger_epoch()
                logger.info('Epoch finished in {}.'.format(humanize_time_delta(time.time() - start_epoch)))

            logger.info('{} epochs of training finished in {}.'.format(self.max_epoch - self.starting_epoch + 1,
                                                                       humanize_time_delta(time.time() - start_train)))
        except StopTraining as e:
            logger.info('Training was stopped by {}.'.format(str(e)))
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
        self.dataflow = dataflow
        self.model = model
        steps_per_epoch = steps_per_epoch or len(self.dataflow)
        self.set_callbacks(callbacks, monitors)
        self.main_loop(steps_per_epoch, starting_epoch, max_epoch)

    def state_dict(self):
        return dict(model=self.model.state_dict())

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])

    def __new__(cls, *args, **kwargs):
        return super(Trainer, cls).__new__(cls)
