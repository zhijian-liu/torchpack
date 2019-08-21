import time
import traceback
import weakref

from tensorpack.utils.utils import humanize_time_delta

from torchpack.callbacks.callback import Callback, Callbacks
from torchpack.train.exception import StopTraining
from torchpack.callbacks.monitor import Monitors
from torchpack.utils.logging import logger

__all__ = ['Trainer']


class Trainer(object):
    """
    Base class for a trainer.
    """

    is_master = True

    def __init__(self):
        self.monitors = Monitors()
        self.monitors.set_trainer(weakref.proxy(self))

    def set_callbacks(self, callbacks):
        for callback in callbacks:
            assert isinstance(callback, Callback), type(callback)
        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_trainer(weakref.proxy(self))

    def train(self, dataflow, callbacks=None, starting_epoch=1, max_epoch=9999999):
        self.dataflow = dataflow
        self.set_callbacks(callbacks)

        self.steps_per_epoch = len(self.dataflow)
        self.starting_epoch = starting_epoch
        self.max_epoch = max_epoch

        self.main_loop()

    def run_step(self, feed_dict):
        """
        Defines what to do in one iteration.
        """
        raise NotImplementedError

    def main_loop(self):
        """
        Run the main training loop.
        """

        self.epoch_num = self.starting_epoch - 1

        try:
            train_time = time.time()
            self.callbacks.before_train()

            for self.epoch_num in range(self.starting_epoch, self.max_epoch + 1):
                self.global_step = self.epoch_num * self.steps_per_epoch - 1
                self.local_step = -1

                logger.info('Epoch {}/{} started.'.format(self.epoch_num, self.max_epoch))

                epoch_time = time.time()
                self.callbacks.before_epoch()

                for self.local_step, feed_dict in enumerate(self.dataflow):
                    self.global_step += 1

                    self.callbacks.before_step(feed_dict)
                    output_dict = self.run_step(feed_dict)
                    self.callbacks.after_step(feed_dict, output_dict)

                    self.callbacks.trigger_step()

                self.callbacks.after_epoch()
                logger.info('Training finished in {}.'.format(humanize_time_delta(time.time() - epoch_time)))

                self.callbacks.trigger_epoch()
                logger.info('Epoch finished in {}.'.format(humanize_time_delta(time.time() - epoch_time)))

            logger.info('{} epochs of training finished in {}.'.format(self.max_epoch - self.starting_epoch + 1,
                                                                       humanize_time_delta(time.time() - train_time)))
        except StopTraining as e:
            logger.info('Training was stopped by {}.'.format(str(e)))
        except KeyboardInterrupt:
            logger.info('Detected Ctrl-C and exiting main training loop.')
            raise
        finally:
            # make sure all callbacks are properly finalized
            for callback in self.callbacks:
                try:
                    callback.after_train()
                except Exception:
                    traceback.print_exc()

    def save_checkpoint(self, checkpoint_dir):
        pass

    def load_checkpoint(self, checkpoint_dir):
        pass
