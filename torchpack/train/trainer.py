import os.path as osp
import time
import traceback
import weakref

from tensorpack.utils.utils import humanize_time_delta

from ..callbacks import (ConsoleWriter, EstimatedTimeLeft, ProgressBar,
                         TFEventWriter)
from ..callbacks.callback import Callback, Callbacks
from ..callbacks.monitor import Monitor, Monitors
from ..logging import logger
from ..utils import fs, io
from .exception import StopTraining

__all__ = ['Trainer']


class Trainer:
    """
    Base class for a trainer.
    """
    is_master = True

    def set_callbacks(self, callbacks):
        monitors = []
        for callback in callbacks:
            assert isinstance(callback, Callback), type(callback)
            if isinstance(callback, Monitor):
                monitors.append(callback)
        # callbacks
        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_trainer(weakref.proxy(self))
        # monitors
        self.monitors = Monitors(monitors)
        self.monitors.set_trainer(weakref.proxy(self))

    def train(self,
              dataflow,
              *,
              callbacks=None,
              starting_epoch=1,
              max_epoch=9999999):
        self.dataflow = dataflow
        self.set_callbacks(callbacks)

        self.steps_per_epoch = len(self.dataflow)
        self.starting_epoch = starting_epoch
        self.max_epoch = max_epoch

        self.run()

    def train_with_defaults(self,
                            dataflow,
                            *,
                            callbacks=None,
                            starting_epoch=1,
                            max_epoch=9999999):
        callbacks += [
            ConsoleWriter(),
            TFEventWriter(),
            # JSONWriter(),
            ProgressBar(),
            EstimatedTimeLeft()
        ]
        self.train(dataflow=dataflow,
                   callbacks=callbacks,
                   starting_epoch=starting_epoch,
                   max_epoch=max_epoch)

    def run(self):
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

            logger.info('{} epochs of training finished in {}.'.format(
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

    def run_step(self, feed_dict):
        output_dict = self._run_step(feed_dict)
        return output_dict

    def _run_step(self, feed_dict):
        """
        Defines what to do in one iteration.
        """
        raise NotImplementedError

    def save_checkpoint(self, save_dir):
        save_dir = osp.normpath(save_dir)
        fs.makedir(save_dir)
        io.save(
            osp.join(save_dir, 'loop.json'), {
                'epoch_num': self.epoch_num,
                'global_step': self.global_step,
                'local_step': self.local_step
            })
        self._save_checkpoint(save_dir)

    def _save_checkpoint(self, save_dir):
        pass

    def load_checkpoint(self, load_dir):
        load_dir = osp.normpath(load_dir)

        loop = io.load(osp.join(load_dir, 'loop.json'))
        self.epoch_num = loop['epoch_num']
        self.global_step = self.epoch_num * self.steps_per_epoch

        self._load_checkpoint(load_dir)

    def _load_checkpoint(self, load_dir):
        pass
