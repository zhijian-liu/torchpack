from contextlib import contextmanager

import torch
import tqdm
from tensorpack.utils import logger
from tensorpack.utils.utils import get_tqdm_kwargs

from .callback import Callback
from .inference import Inferencer

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

    def __init__(self, input, callbacks):
        """
        Args:
            input (InputSource): the input to use. Must have an accurate ``size()``.
            callbacks (list[Inferencer]): list of :class:`Inferencer` to run.
        """
        self._input_source = input
        if not isinstance(callbacks, list):
            self.infs = [callbacks]
        else:
            self.infs = callbacks
        for v in self.infs:
            assert isinstance(v, Inferencer), v

        try:
            # self._size = input.size()
            self._size = len(input)
        except NotImplementedError:
            self._size = 0

        self._hooks = []

    # def register_hook(self, hook):
    #     """
    #     Args:
    #         hook (tf.train.SessionRunHook):
    #     """
    #     self._hooks.append(hook)

    def _after_train(self):
        pass
        # self._input_callbacks.after_train()


class InferenceRunner(InferenceRunnerBase):
    """
    A callback that runs a list of :class:`Inferencer` on some :class:`InputSource`.
    """

    def __init__(self, input, callbacks, device=0):
        """
        Args:
            input (InputSource or DataFlow): The :class:`InputSource` to run
                inference on.  If given a DataFlow, will use :class:`FeedInput`.
            callbacks (list): a list of :class:`Inferencer` instances.
            device (int): the device to use
        """
        # if isinstance(input, DataFlow):
        #     # use infinite=False so that a dataflow without size will stop normally
        #     # TODO a better way to handle inference size
        #     input = FeedInput(input, infinite=False)
        # assert isinstance(input, InputSource), input
        # assert not isinstance(input, StagingInput), input
        # self._device_id = device
        # self._device = _device_from_int(device)
        super(InferenceRunner, self).__init__(input, callbacks)

    # def _build_hook(self, inf):
    #     out_names = inf.get_fetches()
    #     fetches = self._tower_handle.get_tensors(out_names)
    #     return InferencerToHook(inf, fetches)

    def _setup_trainer(self):
        # if self._tower_func is None:
        #     assert self.trainer.tower_func is not None, "You must set tower_func of the trainer to use InferenceRunner!"
        #     self._tower_func = self.trainer.tower_func
        # input_callbacks = self._input_source.setup(self._tower_func.inputs_desc)

        # vs_name = self.trainer._vs_name_for_predictor(self._device_id)
        # logger.info("[InferenceRunner] Building tower '{}' on device {} {}...".format(
        #     self._tower_name, self._device,
        #     "with variable scope '{}'".format(vs_name) if vs_name else ''))
        # with tf.variable_scope(tf.get_variable_scope(), reuse=True), \
        #      tf.device(self._device), \
        #      PredictTowerContext(self._tower_name, vs_name=vs_name):
        #     self._tower_func(*self._input_source.get_input_tensors())
        #     self._tower_handle = self._tower_func.towers[-1]

        # for h in [self._build_hook(inf) for inf in self.infs]:
        #     self.register_hook(h)
        # trigger_{step,epoch}, {before,after}_epoch is ignored.
        # We assume that InputSource callbacks won't use these methods
        # self._input_callbacks = Callbacks(input_callbacks)
        # for h in self._input_callbacks.get_hooks():
        #     self.register_hook(h)

        for inf in self.infs:
            inf.setup_trainer(self.trainer)
        # self._input_callbacks.setup_graph(self.trainer)

    def _trigger(self):
        for inf in self.infs:
            inf.before_epoch()

        # self._input_source.reset_state()
        # iterate over the data, and run the hooked session
        with _inference_context(), tqdm.tqdm(total=self._size, **get_tqdm_kwargs()) as pbar:
            # num_itr = self._size if self._size > 0 else sys.maxsize
            # for _ in range(num_itr):

            self.trainer.model.eval()
            with torch.no_grad():
                for inputs, targets in self._input_source:
                    inputs = inputs.to('cuda', non_blocking=True)
                    targets = targets.to('cuda', non_blocking=True)

                    input_dict = dict(inputs=inputs, targets=targets)
                    outputs = self.trainer.model(input_dict['inputs'])
                    output_dict = dict(outputs=outputs)

                    for inf in self.infs:
                        inf.on_fetches(input_dict, output_dict)

                    pbar.update()

        for inf in self.infs:
            inf.trigger_epoch()
