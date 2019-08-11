from abc import ABCMeta

import six
import torch
import tqdm
from tensorpack.utils import logger
from tensorpack.utils.utils import get_tqdm_kwargs

from torchpack.callbacks.callback import Callback
from torchpack.utils.logging import logger

__all__ = ['InferenceRunner', 'InferenceCallback']


class InferenceRunner(Callback):
    """
    A callback that runs a list of :class:`Inferencer` on some :class:`InputSource`.
    """

    def __init__(self, dataflow, callbacks, device=0):
        """
        Args:
            dataflow (InputSource or DataFlow): The :class:`InputSource` to run
                inference on.  If given a DataFlow, will use :class:`FeedInput`.
            callbacks (list): a list of :class:`Inferencer` instances.
            device (int): the device to use
        """
        self.dataflow = dataflow
        if not isinstance(callbacks, list):
            self.callbacks = [callbacks]
        else:
            self.callbacks = callbacks
        for v in self.callbacks:
            assert isinstance(v, InferenceCallback), v

        try:
            self._size = len(dataflow)
        except NotImplementedError:
            self._size = 0

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def trigger_epoch(self):
        self.trigger()

    def trigger(self):
        for callback in self.callbacks:
            callback.before_inference()

        logger.info('Starting the inference.')
        with tqdm.tqdm(total=self._size, **get_tqdm_kwargs()) as pbar:
            # num_itr = self._size if self._size > 0 else sys.maxsize
            # for _ in range(num_itr):

            self.trainer.model.eval()
            with torch.no_grad():
                for inputs, targets in self.dataflow:
                    inputs = inputs.to('cuda', non_blocking=True)
                    targets = targets.to('cuda', non_blocking=True)

                    fd = dict(inputs=inputs, targets=targets)
                    outputs = self.trainer.model(fd['inputs'])
                    od = dict(outputs=outputs)

                    for callback in self.callbacks:
                        callback.after_step(fd, od)

                    pbar.update()

        # fixme
        for callback in self.callbacks:
            callback.after_inference()
            callback.trigger()


@six.add_metaclass(ABCMeta)
class InferenceCallback(Callback):
    """ Base class of InferenceCallback.
    Inferencer is a special kind of callback that should be called by :class:`InferenceRunner`.
    It has the methods ``_get_fetches`` and ``_on_fetches`` which are like
    :class:`SessionRunHooks`, except that they will be used only by :class:`InferenceRunner`.
    .. document private functions
    .. automethod:: _before_inference
    .. automethod:: _after_inference
    .. automethod:: _get_fetches
    .. automethod:: _on_fetches
    """

    def before_epoch(self):
        self.before_inference()

    def before_inference(self):
        """
        Called before a new round of inference starts.
        """
        pass

    def trigger_epoch(self):
        self.trigger()

    def trigger(self):
        monitors = self.after_inference()
        if monitors is None:
            return
        for k, v in monitors.items():
            try:
                v = float(v)
            except ValueError:
                logger.warn('{} returns a non-scalar statistics!'.format(type(self).__name__))
                continue
            else:
                self.trainer.monitors.add_scalar(k, v)

    def after_inference(self):
        """
        Called after a round of inference ends.
        Returns a dict of scalar statistics which will be logged to monitors.
        """
        pass
