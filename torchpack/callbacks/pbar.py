import tqdm
from tensorpack.utils.utils import get_tqdm_kwargs

from .base import Callback

__all__ = ['ProgressBar']


class ProgressBar(Callback):
    """ A progress bar based on tqdm.
    This callback is one of the :func:`DEFAULT_CALLBACKS()`.
    """

    chief_only = False

    def __init__(self, names=None):
        """
        Args:
            names(list): list of string, the names of the tensors to monitor
                on the progress bar.
        """
        super().__init__()
        # self._names = [get_op_tensor_name(n)[1] for n in names]
        # self._tags = [get_op_tensor_name(n)[0].split("/")[-1] for n in names]
        self.pbar = None

    def before_train(self):
        self.last_updated = self.trainer.local_step

        self.total = self.trainer.steps_per_epoch
        self.tqdm_args = get_tqdm_kwargs(leave=True)
        # self._tqdm_args['bar_format'] = self._tqdm_args['bar_format'] + "{postfix} "

    def before_epoch(self):
        self.pbar = tqdm.trange(self.total, **self.tqdm_args)

    def after_epoch(self):
        self.pbar.close()

    def before_step(self, *args, **kwargs):
        # update progress bar when local step changed (one step is finished)
        if self.last_updated != self.trainer.local_step:
            self.last_updated = self.trainer.local_step

    def after_step(self, *args, **kwargs):
        # self._bar.set_postfix(dict(loss=1))
        pass

    def trigger_step(self):
        self.trigger()

    def trigger(self):
        self.pbar.update()

    def after_train(self):
        # training may get killed before the first step
        if self.pbar:
            self.pbar.close()
