from collections import defaultdict, deque

from torchpack.callbacks.monitor import Monitor

__all__ = ['Summaries']


class Summaries(object):
    def __init__(self):
        self.scalars = defaultdict(deque)

    def set_trainer(self, trainer):
        self.trainer = trainer

    def add_scalar(self, name, scalar):
        for callback in self.trainer.callbacks:
            if isinstance(callback, Monitor):
                callback.add_scalar(name, scalar)
        self._add_scalar(name, scalar)

    def _add_scalar(self, name, scalar):
        self.scalars[name].append((self.trainer.global_step, scalar))

    def get_history(self, name):
        return self.scalars[name]

    def keys(self):
        return self.scalars.keys()

    def __contains__(self, name):
        return name in self.scalars and self.scalars[name]

    def __getitem__(self, name):
        return self.scalars[name][-1]
