import time
import weakref

from ..callbacks import Callback, Callbacks, Monitors, Monitor
from ..utils import logger
from ..utils.utils import humanize_time
from .context import Context

__all__ = ['StopTraining', 'BaseTrainer']


class StopTraining(Exception):
    """
    An exception thrown to stop training.
    """
    pass


class BaseTrainer(object):
    """ Base class for a trainer.
    """

    def __init__(self):
        self.callbacks = []
        self.context = Context()

    def register_callback(self, callback):
        """
        Register callbacks to the trainer.
        It can only be called before :meth:`Trainer.train()`.
        Args:
            callback (Callback): a callback
        Returns:
            succeed or not
        """
        assert isinstance(callback, Callback), callback

        callback.trainer = weakref.proxy(self)
        self.callbacks.append(callback)

        return True

    def setup_callbacks(self, callbacks, monitors):
        """
        Setup callbacks and monitors. Must be called after the main graph is built.
        Args:
            callbacks ([Callback]):
            monitors ([MonitorBase]):
        """
        assert isinstance(callbacks, list), callbacks
        assert isinstance(monitors, list), monitors

        for callback in callbacks:
            assert not isinstance(callback, Monitor), 'Monitor cannot be registered as callbacks!'
            assert self.register_callback(callback), callback

        self.monitors = []
        for monitor in monitors:
            if self.register_callback(monitor):
                self.monitors.append(monitor)

        self.monitors = Monitors(self.monitors)
        self.register_callback(self.monitors)

        # some final operations that might modify the graph
        logger.info('Set up callbacks')
        self.callbacks = Callbacks(self.callbacks)

    def run_step(self):
        """
        Defines what to do in one iteration.
        """
        raise NotImplementedError('Please provide an implementation of Trainer.run_step()!')

    def main_loop(self, starting_epoch, max_epoch, steps_per_epoch):
        """
        Run the main training loop.

        Args:
            starting_epoch, steps_per_epoch, max_epoch (int):
        """
        self.starting_epoch = int(starting_epoch)
        self.max_epoch = int(max_epoch)
        self.steps_per_epoch = int(steps_per_epoch)

        self.epoch_num = starting_epoch - 1
        self.global_step = self.epoch_num * self.steps_per_epoch

        try:
            self.callbacks.before_train()

            for self.epoch_num in range(self.starting_epoch, self.max_epoch + 1):
                logger.info('Start Epoch {} ...'.format(self.epoch_num))

                self.callbacks.before_epoch()
                start_time = time.time()

                for self.local_step in range(self.steps_per_epoch):
                    self.callbacks.before_run()
                    self.run_step()
                    self.callbacks.after_run()

                    self.global_step += 1
                    self.callbacks.trigger_step()

                # self.callbacks.after_epoch()
                logger.info("Epoch {} finished, time: {}.".format(self.epoch_num, self.global_step,
                                                                  humanize_time(time.time() - start_time)))

                # trigger epoch outside the timing region.
                self.callbacks.trigger_epoch()

            logger.info('Training has finished!')
        except StopTraining as e:
            logger.info('Training was stopped by exception {}.'.format(str(e)))
        except KeyboardInterrupt:
            logger.info('Detected Ctrl-C and exiting main loop.')
            raise
        finally:
            self.callbacks.after_train()

    def train(self, callbacks, monitors, steps_per_epoch, starting_epoch=1, max_epoch=9999999):
        """
        Implemented by two lines:

        .. code-block:: python

            self.setup_callbacks(callbacks, monitors)
            self.main_loop(steps_per_epoch, starting_epoch, max_epoch)

        You can call those methods by yourself to have better control on details if needed.
        """
        self.setup_callbacks(callbacks, monitors)
        self.main_loop(starting_epoch, max_epoch, steps_per_epoch)
