import torch
import tqdm
from tensorpack.utils import logger
from tensorpack.utils.utils import get_tqdm_kwargs

from torchpack.callbacks.callback import Callback
from torchpack.callbacks.inference.callback import InferenceCallback
from torchpack.utils.logging import logger

__all__ = ['InferenceRunner']


class InferenceRunner(Callback):
    """ A callback that runs a list of :class:`InferenceCallback`.
    """

    def __init__(self, dataflow, callbacks, device=0):
        """
        Args:
            dataflow (InputSource or DataFlow): The :class:`InputSource` to run
                inference on.  If given a DataFlow, will use :class:`FeedInput`.
            callbacks (list): a list of :class:`Inferencer` instances.
            device (int): the device to use
        """
        for callback in callbacks:
            assert isinstance(callback, InferenceCallback), callback

        self.dataflow = dataflow
        self.callbacks = callbacks
        self.size = len(dataflow)

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
        with tqdm.tqdm(total=self.size, **get_tqdm_kwargs()) as pbar:
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
