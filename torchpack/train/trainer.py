import time
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader, DistributedSampler

from torchpack.callbacks import (Callback, Callbacks, ConsoleWriter,
                                 EstimatedTimeLeft, JSONLWriter, MetaInfoSaver,
                                 ProgressBar, TFEventWriter)
from torchpack.utils import humanize
from torchpack.utils.logging import logger

from .exception import StopTraining
from .summary import Summary

__all__ = ['Trainer']


class Trainer:
    """Base class for a trainer."""

    def train_with_defaults(
        self,
        dataflow: DataLoader,
        *,
        steps_per_epoch: Optional[int] = None,
        num_epochs: int = 9999999,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        if callbacks is None:
            callbacks = []
        callbacks += [
            MetaInfoSaver(),
            ConsoleWriter(),
            TFEventWriter(),
            JSONLWriter(),
            ProgressBar(),
            EstimatedTimeLeft()
        ]
        self.train(
            dataflow=dataflow,
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
        )

    def train(
        self,
        dataflow: DataLoader,
        *,
        steps_per_epoch: Optional[int] = None,
        num_epochs: int = 9999999,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self.dataflow = dataflow
        if steps_per_epoch is None:
            steps_per_epoch = len(self.dataflow)
        self.steps_per_epoch = steps_per_epoch
        self.num_epochs = num_epochs

        if callbacks is None:
            callbacks = []
        self.callbacks = Callbacks(callbacks)
        self.summary = Summary()

        iterator = iter(self.dataflow)

        try:
            self.callbacks.set_trainer(self)
            self.summary.set_trainer(self)

            self.epoch_num = 0
            self.global_step = 0

            train_time = time.perf_counter()
            self.before_train()

            while self.epoch_num < self.num_epochs:
                self.epoch_num += 1
                self.local_step = 0

                logger.info('Epoch {}/{} started.'.format(
                    self.epoch_num, self.num_epochs))
                epoch_time = time.perf_counter()
                self.before_epoch()

                while self.local_step < self.steps_per_epoch:
                    self.local_step += 1
                    self.global_step += 1

                    try:
                        feed_dict = next(iterator)
                    except StopIteration:
                        iterator = iter(self.dataflow)
                        feed_dict = next(iterator)

                    self.before_step(feed_dict)
                    output_dict = self.run_step(feed_dict)
                    self.after_step(output_dict)

                    self.trigger_step()

                self.after_epoch()
                logger.info('Training finished in {}.'.format(
                    humanize.naturaldelta(time.perf_counter() - epoch_time)))

                self.trigger_epoch()
                logger.info('Epoch finished in {}.'.format(
                    humanize.naturaldelta(time.perf_counter() - epoch_time)))

            logger.success('{} epochs of training finished in {}.'.format(
                self.num_epochs,
                humanize.naturaldelta(time.perf_counter() - train_time)))
        except StopTraining as e:
            logger.info(f'Training was stopped by {str(e)}.')
        finally:
            self.after_train()

    def before_train(self) -> None:
        self._before_train()
        self.callbacks.before_train()

    def _before_train(self) -> None:
        pass

    def before_epoch(self) -> None:
        if isinstance(self.dataflow, DataLoader) and \
                isinstance(self.dataflow.sampler, DistributedSampler):
            self.dataflow.sampler.set_epoch(self.epoch_num)
        self._before_epoch()
        self.callbacks.before_epoch()

    def _before_epoch(self) -> None:
        pass

    def before_step(self, feed_dict: Dict[str, Any]) -> None:
        self._before_step(feed_dict)
        self.callbacks.before_step(feed_dict)

    def _before_step(self, feed_dict: Dict[str, Any]) -> None:
        pass

    def run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        output_dict = self._run_step(feed_dict)
        return output_dict

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Define what to do in one iteration."""
        raise NotImplementedError

    def after_step(self, output_dict: Dict[str, Any]) -> None:
        self.callbacks.after_step(output_dict)
        self._after_step(output_dict)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        pass

    def trigger_step(self) -> None:
        self.callbacks.trigger_step()
        self._trigger_step()

    def _trigger_step(self) -> None:
        pass

    def after_epoch(self) -> None:
        self.callbacks.after_epoch()
        self._after_epoch()

    def _after_epoch(self) -> None:
        pass

    def trigger_epoch(self) -> None:
        self.callbacks.trigger_epoch()
        self._trigger_epoch()

    def _trigger_epoch(self) -> None:
        pass

    def after_train(self) -> None:
        self.callbacks.after_train()
        self._after_train()

    def _after_train(self) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        state_dict = self._state_dict()
        state_dict['callbacks'] = self.callbacks.state_dict()
        state_dict['epoch_num'] = self.epoch_num
        state_dict['local_step'] = self.local_step
        state_dict['global_step'] = self.global_step
        return state_dict

    def _state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.epoch_num = state_dict.pop('epoch_num')
        self.local_step = state_dict.pop('local_step')
        self.global_step = state_dict.pop('global_step')
        self.callbacks.load_state_dict(state_dict.pop('callbacks'))
        self._load_state_dict(state_dict)

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
