import time
import weakref

from ..callbacks import Callback, Callbacks, Monitors, MonitorBase
from ..utils import logger
from ..utils.utils import humanize_time_delta

__all__ = ['StopTraining', 'Trainer']


class StopTraining(Exception):
    """
    An exception thrown to stop training.
    """
    pass


class TrainLoop(object):
    """
    Manage the double for loop.
    """

    def __init__(self):
        self._epoch_num = 0
        self._global_step = -1
        self._local_step = -1

    def config(self, steps_per_epoch, starting_epoch, max_epoch):
        """
        Configure the loop given the settings.
        """
        self.starting_epoch = int(starting_epoch)
        self.max_epoch = int(max_epoch)
        self.steps_per_epoch = int(steps_per_epoch)
        # Allow empty epoch (no steps), if we want to run the callbacks only.
        assert self.steps_per_epoch >= 0 and self.max_epoch >= 0

        self._epoch_num = starting_epoch - 1

    def update_global_step(self):
        """
        Update the Python-side global_step from TF.
        This must be called under initialized default session.
        """
        self._global_step += 1

    @property
    def epoch_num(self):
        """
        The number of the currently ongoing epoch.

        An epoch is defined to cover the moment before calling `before_epoch` until after calling `trigger_epoch`.
        i.e., in the `trigger_epoch` of epoch 3, `self.epoch_num` is 3.
        If you need use `self.epoch_num` in your callback, you'll need to know this.
        """
        return self._epoch_num

    @property
    def global_step(self):
        """
        The tensorflow global_step, i.e. how many times ``hooked_sess.run`` has been called.

        Note:
            1. global_step is incremented **after** each ``hooked_sess.run`` returns from TF runtime.
            2. If you make zero or more than one calls to ``hooked_sess.run`` in one
               :meth:`run_step`, local_step and global_step may increment at different speed.
        """
        return self._global_step

    @property
    def local_step(self):
        """
        The number of steps that have finished in the current epoch.
        """
        return self._local_step


class Trainer(object):
    """ Base class for a trainer.
    """

    def __init__(self):
        self._callbacks = []
        self.loop = TrainLoop()

    def register_callback(self, callback):
        """
        Register callbacks to the trainer.
        It can only be called before :meth:`Trainer.train()`.
        Args:
            callback (Callback or [Callback]): a callback or a list of callbacks
        Returns:
            succeed or not
        """
        if isinstance(callback, (list, tuple)):
            success = True
            for x in callback:
                success = success and self.register_callback(x)
            return success

        assert isinstance(callback, Callback), callback
        assert not isinstance(self._callbacks, Callbacks), 'Cannot register more callbacks after trainer was set up!'

        callback.trainer = weakref.proxy(self)
        self._callbacks.append(callback)
        return True

    def run_step(self):
        """
        Defines what to do in one iteration. The default is:
        ``self.hooked_sess.run(self.train_op)``.
        The behavior of each iteration can be changed by either setting ``trainer.train_op``,
        or overriding this method.
        """
        raise NotImplementedError('Please provide an implementation of Trainer.run_step()!')

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
            self.register_callback(callback)
        for callback in self._callbacks:
            assert not isinstance(callback, MonitorBase), 'Monitor cannot be pre-registered for now!'

        registered_monitors = []
        for monitor in monitors:
            if self.register_callback(monitor):
                registered_monitors.append(monitor)
        self.monitors = Monitors(registered_monitors)
        self.register_callback(self.monitors)  # monitors is also a callback

        # some final operations that might modify the graph
        logger.info('Set up callbacks')
        self._callbacks = Callbacks(self._callbacks)

    def main_loop(self, steps_per_epoch, starting_epoch, max_epoch):
        """
        Run the main training loop.

        Args:
            steps_per_epoch, starting_epoch, max_epoch (int):
        """
        self.loop.config(steps_per_epoch, starting_epoch, max_epoch)
        try:
            self._callbacks.before_train()
            self.loop.update_global_step()

            for self.loop._epoch_num in range(self.loop.starting_epoch, self.loop.max_epoch + 1):
                logger.info("Start Epoch {} ...".format(self.loop.epoch_num))
                self._callbacks.before_epoch()
                start_time = time.time()

                for self.loop._local_step in range(self.loop.steps_per_epoch):
                    self.run_step()  # implemented by subclass
                    self._callbacks.trigger_step()

                self._callbacks.after_epoch()
                logger.info("Epoch {} (global_step {}) finished, time:{}.".format(
                    self.loop.epoch_num, self.loop.global_step, humanize_time_delta(time.time() - start_time)))

                # trigger epoch outside the timing region.
                self._callbacks.trigger_epoch()

            logger.info('Training has finished!')
        except StopTraining as e:
            logger.info('Training was stopped by exception {}.'.format(str(e)))
        except KeyboardInterrupt:
            logger.info('Detected Ctrl-C and exiting main loop.')
            raise
        finally:
            self._callbacks.after_train()

    def train(self,
              callbacks, monitors,
              steps_per_epoch, starting_epoch=1, max_epoch=9999999):
        """
        Implemented by two lines:

        .. code-block:: python

            self.setup_callbacks(callbacks, monitors)
            self.main_loop(steps_per_epoch, starting_epoch, max_epoch)

        You can call those methods by yourself to have better control on details if needed.
        """
        self.setup_callbacks(callbacks, monitors)
        self.main_loop(steps_per_epoch, starting_epoch, max_epoch)
