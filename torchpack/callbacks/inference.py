import time
from abc import ABCMeta

import six
import torch
import tqdm
from tensorpack.utils.utils import get_tqdm_kwargs
from tensorpack.utils.utils import humanize_time_delta

from torchpack.callbacks.callback import Callback
from torchpack.cuda.copy import async_copy_to
from torchpack.utils.logging import logger

__all__ = ['InferenceCallback', 'InferenceCallbacks', 'InferenceRunner']


@six.add_metaclass(ABCMeta)
class InferenceCallback(object):
    """
    Base class of inference callbacks.
    """

    def set_trainer(self, trainer):
        self.trainer = trainer
        self._set_trainer(trainer)

    def _set_trainer(self, trainer):
        pass

    def before_inference(self):
        self._before_inference()

    def _before_inference(self):
        """
        Called before inference starts.
        """
        pass

    def before_step(self, *args, **kwargs):
        self._before_step(*args, **kwargs)

    def _before_step(self, *args, **kwargs):
        """
        Called before every step.
        """
        pass

    def after_step(self, *args, **kwargs):
        self._after_step(*args, **kwargs)

    def _after_step(self, *args, **kwargs):
        """
        Called after every step.
        """
        pass

    def after_inference(self):
        self._after_inference()

    def _after_inference(self):
        """
        Called after inference ends.
        """
        pass


@six.add_metaclass(ABCMeta)
class InferenceCallbacks(InferenceCallback):
    """
    A container to hold inference callbacks.
    """

    def __init__(self, callbacks):
        for callback in callbacks:
            assert isinstance(callback, InferenceCallback), type(callback)
        self.callbacks = callbacks

    def _set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def _before_inference(self):
        for callback in self.callbacks:
            callback.before_inference()

    def _before_step(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.before_step(*args, **kwargs)

    def _after_step(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.after_step(*args, **kwargs)

    def _after_inference(self):
        for callback in self.callbacks:
            callback.after_inference()

    def __len__(self):
        return len(self.callbacks)

    def __getitem__(self, index):
        return self.callbacks[index]


class InferenceRunner(Callback):
    """
    A callback that runs inference with a list of :class:`InferenceCallback`.
    """

    def __init__(self, dataflow, callbacks, device=None):
        for callback in callbacks:
            assert isinstance(callback, InferenceCallback), callback
        self.dataflow = dataflow
        self.callbacks = InferenceCallbacks(callbacks)
        self.device = device

    def _set_trainer(self, trainer):
        self.device = self.device or trainer.device
        self.callbacks.set_trainer(trainer)

    def _trigger_epoch(self):
        self._trigger()

    def _trigger(self):
        start_time = time.time()
        self.callbacks.before_inference()

        self.trainer.model.eval()
        with torch.no_grad():
            for feed_dict in tqdm.tqdm(self.dataflow, **get_tqdm_kwargs()):
                # todo: maybe move `async_copy_to` to `dataflow`
                feed_dict = async_copy_to(feed_dict, device=self.device)

                self.callbacks.before_step(feed_dict)
                output_dict = self.trainer.model(feed_dict)
                self.callbacks.after_step(feed_dict, output_dict)

        self.callbacks.after_inference()
        logger.info('Inference finished in {}.'.format(humanize_time_delta(time.time() - start_time)))
