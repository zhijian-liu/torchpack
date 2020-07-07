import time
import traceback
import weakref

from tensorpack.utils.utils import humanize_time_delta
from torch.utils.data import DataLoader, DistributedSampler

from ..callbacks import (ConsoleWriter, EstimatedTimeLeft, MetaInfoSaver,
                         ProgressBar, TFEventWriter)
from ..callbacks.callback import Callback, Callbacks
from ..callbacks.monitor import Monitor, Monitors
from ..utils.logging import logger
from .exception import StopTraining

__all__ = ['Trainer']


class Trainer:
    """
    Base class for a trainer.
    """
    def train(self,
              dataflow,
              *,
              starting_epoch=1,
              max_epoch=9999999,
              callbacks=None):
        self.dataflow = dataflow
        self.steps_per_epoch = len(self.dataflow)
        self.starting_epoch = starting_epoch
        self.max_epoch = max_epoch

        if callbacks is None:
            callbacks = []

        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_trainer(weakref.proxy(self))

        monitors = []
        for callback in callbacks:
            assert isinstance(callback, Callback), type(callback)
            if isinstance(callback, Monitor):
                monitors.append(callback)

        self.monitors = Monitors(monitors)
        self.monitors.set_trainer(weakref.proxy(self))

        self.epoch_num = self.starting_epoch - 1
        self.global_step = self.epoch_num * self.steps_per_epoch

        try:
            train_time = time.time()
            self.callbacks.before_train()

            while self.epoch_num < self.max_epoch:
                self.epoch_num += 1
                self.local_step = 0

                logger.info('Epoch {}/{} started.'.format(
                    self.epoch_num, self.max_epoch))
                epoch_time = time.time()
                self.callbacks.before_epoch()

                if isinstance(self.dataflow, DataLoader) and isinstance(
                        self.dataflow.sampler, DistributedSampler):
                    self.dataflow.sampler.set_epoch(self.epoch_num)

                for feed_dict in self.dataflow:
                    self.local_step += 1
                    self.global_step += 1

                    self.callbacks.before_step(feed_dict)
                    output_dict = self.run_step(feed_dict)
                    self.callbacks.after_step(output_dict)

                    self.callbacks.trigger_step()

                self.callbacks.after_epoch()
                logger.info('Training finished in {}.'.format(
                    humanize_time_delta(time.time() - epoch_time)))

                self.callbacks.trigger_epoch()
                logger.info('Epoch finished in {}.'.format(
                    humanize_time_delta(time.time() - epoch_time)))

            logger.success('{} epochs of training finished in {}.'.format(
                self.max_epoch - self.starting_epoch + 1,
                humanize_time_delta(time.time() - train_time)))
        except StopTraining as e:
            logger.info('Training was stopped by {}.'.format(str(e)))
        except KeyboardInterrupt:
            logger.info('Detected Ctrl-C and exiting training loop.')
            raise
        finally:
            for callback in self.callbacks:
                try:
                    callback.after_train()
                except Exception:
                    traceback.print_exc()

    def train_with_defaults(self,
                            dataflow,
                            *,
                            starting_epoch=1,
                            max_epoch=9999999,
                            callbacks=None):
        if callbacks is None:
            callbacks = []

        self.train(dataflow=dataflow,
                   starting_epoch=starting_epoch,
                   max_epoch=max_epoch,
                   callbacks=callbacks + [
                       MetaInfoSaver(),
                       ConsoleWriter(),
                       TFEventWriter(),
                       ProgressBar(),
                       EstimatedTimeLeft()
                   ])

    def run_step(self, feed_dict):
        output_dict = self._run_step(feed_dict)
        return output_dict

    def _run_step(self, feed_dict):
        """
        Defines what to do in one iteration.
        """
        raise NotImplementedError

    def state_dict(self):
        state_dict = self._state_dict() or dict()
        state_dict.update({
            'epoch_num': self.epoch_num,
            'local_step': self.local_step,
            'global_step': self.global_step
        })
        return state_dict

    def _state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        self.epoch_num = state_dict['epoch_num']
        self.local_step = state_dict['local_step']
        self.global_step = state_dict['global_step']
        self._load_state_dict(state_dict)

    def _load_state_dict(self, state_dict):
        pass
