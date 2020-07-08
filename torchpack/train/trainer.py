import time
import traceback
import weakref

import torch
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
    def train_with_defaults(self,
                            dataflow,
                            *,
                            starting_epoch=1,
                            max_epoch=9999999,
                            callbacks=None):
        if callbacks is None:
            callbacks = []
        callbacks.extend([
            MetaInfoSaver(),
            ConsoleWriter(),
            TFEventWriter(),
            ProgressBar(),
            EstimatedTimeLeft()
        ])
        self.train(dataflow=dataflow,
                   starting_epoch=starting_epoch,
                   max_epoch=max_epoch,
                   callbacks=callbacks)

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
            if isinstance(callback, Monitor):
                monitors.append(callback)
        self.monitors = Monitors(monitors)
        self.monitors.set_trainer(weakref.proxy(self))

        try:
            self.epoch_num = self.starting_epoch - 1
            self.global_step = self.epoch_num * self.steps_per_epoch

            train_time = time.time()
            self.before_train()

            while self.epoch_num < self.max_epoch:
                self.epoch_num += 1
                self.local_step = 0

                logger.info('Epoch {}/{} started.'.format(
                    self.epoch_num, self.max_epoch))
                epoch_time = time.time()
                self.before_epoch()

                for feed_dict in self.dataflow:
                    self.local_step += 1
                    self.global_step += 1

                    self.before_step(feed_dict)
                    output_dict = self.run_step(feed_dict)
                    self.after_step(output_dict)

                    self.trigger_step()

                self.after_epoch()
                logger.info('Training finished in {}.'.format(
                    humanize_time_delta(time.time() - epoch_time)))

                self.trigger_epoch()
                logger.info('Epoch finished in {}.'.format(
                    humanize_time_delta(time.time() - epoch_time)))

            logger.success('{} epochs of training finished in {}.'.format(
                self.max_epoch - self.starting_epoch + 1,
                humanize_time_delta(time.time() - train_time)))
        except StopTraining as e:
            logger.info('Training was stopped by {}.'.format(str(e)))
        finally:
            self.after_train()

    def before_train(self):
        self._before_train()
        self.callbacks.before_train()

    def _before_train(self):
        pass

    def before_epoch(self):
        torch.set_grad_enabled(True)
        if isinstance(self.dataflow, DataLoader) and isinstance(
                self.dataflow.sampler, DistributedSampler):
            self.dataflow.sampler.set_epoch(self.epoch_num)
        self._before_epoch()
        self.callbacks.before_epoch()

    def _before_epoch(self):
        pass

    def before_step(self, feed_dict):
        self._before_step(feed_dict)
        self.callbacks.before_step(feed_dict)

    def _before_step(self, feed_dict):
        pass

    def run_step(self, feed_dict):
        output_dict = self._run_step(feed_dict)
        return output_dict

    def _run_step(self, feed_dict):
        """
        Defines what to do in one iteration.
        """
        raise NotImplementedError

    def after_step(self, output_dict):
        self.callbacks.after_step(output_dict)
        self._after_step(output_dict)

    def _after_step(self, output_dict):
        pass

    def trigger_step(self):
        self.callbacks.trigger_step()
        self._trigger_step()

    def _trigger_step(self):
        pass

    def after_epoch(self):
        self.callbacks.after_epoch()
        torch.set_grad_enabled(False)
        self._after_epoch()

    def _after_epoch(self):
        pass

    def trigger_epoch(self):
        self.callbacks.trigger_epoch()
        self._trigger_epoch()

    def _trigger_epoch(self):
        pass

    def after_train(self):
        for callback in self.callbacks:
            try:
                callback.after_train()
            except Exception:
                traceback.print_exc()
        self._after_train()

    def _after_train(self):
        pass

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
