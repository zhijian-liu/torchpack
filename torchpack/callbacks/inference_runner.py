from contextlib import contextmanager

import torch
import tqdm
from tensorpack.utils import logger
from tensorpack.utils.utils import get_tqdm_kwargs

from .callback import Callback
from .inference import InferenceCallback

__all__ = ['InferenceRunnerBase', 'InferenceRunner']


# def _device_from_int(dev):
#     return '/gpu:{}'.format(dev) if dev >= 0 else '/cpu:0'


# class InferencerToHook(tfv1.train.SessionRunHook):
#     def __init__(self, inf, fetches):
#         self._inf = inf
#         self._fetches = fetches
#
#     def before_run(self, _):
#         return tf.train.SessionRunArgs(fetches=self._fetches)
#
#     def after_run(self, _, run_values):
#         self._inf.on_fetches(run_values.results)


@contextmanager
def _inference_context():
    msg = "You might need to check your input implementation."
    try:
        yield
    except (StopIteration):
        logger.error("[InferenceRunner] input stopped before reaching its __len__()! " + msg)
        raise


class InferenceRunnerBase(Callback):
    """ Base class for inference runner.
    Note:
        1. InferenceRunner will use `input.size()` to determine
           how much iterations to run, so you're responsible to ensure that
           `input.size()` is accurate.
        2. Only works with instances of `TowerTrainer`.
    """

    def __init__(self, dataflow, callbacks):
        """
        Args:
            dataflow (InputSource): the input to use. Must have an accurate ``size()``.
            callbacks (list[InferenceCallback]): list of :class:`Inferencer` to run.
        """
        self._input_source = dataflow
        if not isinstance(callbacks, list):
            self.callbacks = [callbacks]
        else:
            self.callbacks = callbacks
        for v in self.callbacks:
            assert isinstance(v, InferenceCallback), v

        try:
            # self._size = input.size()
            self._size = len(dataflow)
        except NotImplementedError:
            self._size = 0

        self._hooks = []

    # def register_hook(self, hook):
    #     """
    #     Args:
    #         hook (tf.train.SessionRunHook):
    #     """
    #     self._hooks.append(hook)

    def after_train(self):
        pass
        # self._input_callbacks.after_train()


class InferenceRunner(InferenceRunnerBase):
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
        super().__init__(dataflow, callbacks)

    def set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def trigger(self):
        for callback in self.callbacks:
            callback.before_epoch()

        with _inference_context(), tqdm.tqdm(total=self._size, **get_tqdm_kwargs()) as pbar:
            # num_itr = self._size if self._size > 0 else sys.maxsize
            # for _ in range(num_itr):

            self.trainer.model.eval()
            with torch.no_grad():
                for inputs, targets in self._input_source:
                    inputs = inputs.to('cuda', non_blocking=True)
                    targets = targets.to('cuda', non_blocking=True)

                    fd = dict(inputs=inputs, targets=targets)
                    outputs = self.trainer.model(fd['inputs'])
                    od = dict(outputs=outputs)

                    for callback in self.callbacks:
                        callback.on_fetches(fd, od)

                    pbar.update()

        for callback in self.callbacks:
            callback.trigger_epoch()
