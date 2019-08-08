import tqdm
from tensorpack.utils.utils import get_tqdm_kwargs

from .callback import Callback


class ProgressBar(Callback):
    """ A progress bar based on tqdm.
    This callback is one of the :func:`DEFAULT_CALLBACKS()`.
    """

    _chief_only = False

    def __init__(self, names=[]):
        """
        Args:
            names(list): list of string, the names of the tensors to monitor
                on the progress bar.
        """
        super().__init__()
        # self._names = [get_op_tensor_name(n)[1] for n in names]
        # self._tags = [get_op_tensor_name(n)[0].split("/")[-1] for n in names]
        self._bar = None

    def _before_train(self):
        self._last_updated = self.trainer.local_step

        self._total = self.trainer.steps_per_epoch
        self._tqdm_args = get_tqdm_kwargs(leave=True)

        # self._fetches = self.get_tensors_maybe_in_tower(self._names) or None
        # if self._fetches:
        #     for t in self._fetches:
        #         assert t.shape.ndims == 0, "ProgressBar can only print scalars, not {}".format(t)
        #     self._fetches = tf.train.SessionRunArgs(self._fetches)
        #     self._tqdm_args['bar_format'] = self._tqdm_args['bar_format'] + "{postfix} "

    def _before_epoch(self):
        self._bar = tqdm.trange(self._total, **self._tqdm_args)

    def _after_epoch(self):
        self._bar.close()

    def _before_step(self, _):
        # update progress bar when local step changed (one step is finished)
        if self.trainer.local_step != self._last_updated:
            self._last_updated = self.trainer.local_step
            return None
            # return self._fetches
        else:
            return None

    def _after_step(self, _, run_values):
        # res = run_values.results
        # if res:
        #     self._bar.set_postfix(zip(self._tags, res))
        pass

    def _trigger_step(self):
        self._bar.update()

    def _after_train(self):
        # training may get killed before the first step
        if self._bar:
            self._bar.close()
