import time
import weakref

from tensorpack.utils.argtools import call_only_once
from tensorpack.utils.utils import humanize_time_delta

from torchpack.callbacks import Callback, Monitor, MonitorGroup, TimedCallbackGroup
from torchpack.trainer.exception import StopTraining
from torchpack.utils.logging import logger

__all__ = ['Trainer']


class Trainer(object):
    """ Base class for a trainer.
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
        self.global_step = 0
        self.local_step = -1

    def run_step(self, *args, **kwargs):
        """
        Defines what to do in one iteration.
        """
        raise NotImplementedError

    @call_only_once
    def setup_callbacks(self, callbacks, monitors):
        assert isinstance(callbacks, list), callbacks
        assert isinstance(monitors, list), monitors

        self.callbacks = []
        for callback in callbacks:
            assert isinstance(callback, Callback), type(callback)
            if not self.is_chief and callback.chief_only:
                logger.warning('Callback {} is chief-only, skipped.'.format(callback))
                continue
            self.callbacks.append(callback)

        self.monitors = []
        for monitor in monitors:
            assert isinstance(monitor, Monitor), type(monitor)
            self.monitors.append(monitor)

        self.monitors = MonitorGroup(self.monitors)
        self.callbacks = TimedCallbackGroup(self.callbacks + [self.monitors])
        self.callbacks.set_trainer(weakref.proxy(self))

    @call_only_once
    def main_loop(self, steps_per_epoch, starting_epoch, max_epoch):
        """
        Run the main training loop.

        Args:
            steps_per_epoch, starting_epoch, max_epoch (int):
        """

        self.starting_epoch = int(starting_epoch)
        self.max_epoch = int(max_epoch)
        self.steps_per_epoch = int(steps_per_epoch)

        # Allow empty epoch (no steps), if we want to run the callbacks only.
        assert self.steps_per_epoch >= 0 and self.max_epoch >= 0

        self.epoch_num = starting_epoch - 1
        self.global_step = self.epoch_num * self.steps_per_epoch

        try:
            self.callbacks.before_train()
            for self.epoch_num in range(self.starting_epoch, self.max_epoch + 1):
                logger.info('Training epoch {}/{} started.'.format(self.epoch_num, self.max_epoch))
                self.callbacks.before_epoch()
                start_time = time.time()

                self.model.train()
                # for self.local_step in range(self.steps_per_epoch):
                for self.local_step, (inputs, targets) in enumerate(self.loader):
                    # fixme
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    fd = dict(inputs=inputs, targets=targets)

                    self.callbacks.before_step()
                    self.run_step(fd)
                    self.callbacks.after_step()

                    self.callbacks.trigger_step()
                    self.global_step += 1

                self.callbacks.after_epoch()
                logger.info('Training epoch finished in {}.'.format(humanize_time_delta(time.time() - start_time)))
                self.callbacks.trigger_epoch()
            logger.info('Training has finished!')
        except StopTraining as e:
            logger.info('Training was stopped by exception {}.'.format(str(e)))
        except KeyboardInterrupt:
            logger.info('Detected Ctrl-C and exiting main loop.')
            raise
        finally:
            self.callbacks.after_train()

    def train(self, loader, model, criterion,
              callbacks=None, monitors=None,
              steps_per_epoch=None, starting_epoch=1, max_epoch=9999999):
        """
        Implemented by two lines:

        .. code-block:: python

            self.setup_callbacks(callbacks, monitors)
            self.main_loop(steps_per_epoch, starting_epoch, max_epoch)

        You can call those methods by yourself to have better control on details if needed.
        """
        self.loader = loader
        self.model = model
        self.criterion = criterion
        steps_per_epoch = len(self.loader)
        self.setup_callbacks(callbacks, monitors)
        self.main_loop(steps_per_epoch, starting_epoch, max_epoch)

    def state_dict(self):
        return dict(model=self.model.state_dict())

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])

    def __new__(cls, *args, **kwargs):
        return super(Trainer, cls).__new__(cls)
