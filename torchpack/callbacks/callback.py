from abc import ABCMeta

import six

__all__ = ['Callback', 'LambdaCallback', 'ProxyCallback', 'Callbacks']


@six.add_metaclass(ABCMeta)
class Callback:
    """
    Base class for all callbacks.
    """

    master_only = False

    def set_trainer(self, trainer):
        self.trainer = trainer
        self._set_trainer(trainer)

    def _set_trainer(self, trainer):
        pass

    def before_train(self):
        if self.trainer.is_master or not self.master_only:
            self._before_train()

    def _before_train(self):
        """
        Called before training.
        """
        pass

    def before_epoch(self):
        if self.trainer.is_master or not self.master_only:
            self._before_epoch()

    def _before_epoch(self):
        """
        Called before every epoch.
        """
        pass

    def before_step(self, *args, **kwargs):
        if self.trainer.is_master or not self.master_only:
            self._before_step(*args, **kwargs)

    def _before_step(self, *args, **kwargs):
        """
        Called before every step.
        """
        pass

    def after_step(self, *args, **kwargs):
        if self.trainer.is_master or not self.master_only:
            self._after_step(*args, **kwargs)

    def _after_step(self, *args, **kwargs):
        """
        Called after every step.
        """
        pass

    def trigger_step(self):
        if self.trainer.is_master or not self.master_only:
            self._trigger_step()

    def _trigger_step(self):
        """
        Called after after step.
        """
        pass

    def after_epoch(self):
        if self.trainer.is_master or not self.master_only:
            self._after_epoch()

    def _after_epoch(self):
        """
        Called after every epoch.
        """
        pass

    def trigger_epoch(self):
        if self.trainer.is_master or not self.master_only:
            self._trigger_epoch()

    def _trigger_epoch(self):
        """
        Called after after epoch.
        """
        pass

    def trigger(self):
        if self.trainer.is_master or not self.master_only:
            self._trigger()

    def _trigger(self):
        """
        Override this method to define a general trigger behavior, to be used with trigger schedulers.
        Note that the schedulers (e.g. :class:`PeriodicTrigger`) might call this method
        both inside an epoch and after an epoch.
        """
        pass

    def after_train(self):
        if self.trainer.is_master or not self.master_only:
            self._after_train()

    def _after_train(self):
        """
        Called after training.
        """
        pass

    def __str__(self):
        return type(self).__name__


class LambdaCallback(Callback):
    """
    A callback created with lambda functions.
    """

    def __init__(self, before_train=None, before_epoch=None, before_step=None, after_step=None, trigger_step=None,
                 after_epoch=None, trigger_epoch=None, trigger=None, after_train=None, master_only=False):
        self.before_train_func = before_train
        self.before_epoch_func = before_epoch
        self.before_step_func = before_step
        self.after_step_func = after_step
        self.trigger_step_func = trigger_step
        self.after_epoch_func = after_epoch
        self.trigger_epoch_func = trigger_epoch
        self.trigger_func = trigger
        self.after_train_func = after_train
        self.master_only = master_only

    def _before_train(self):
        if self.before_train_func:
            self.before_train_func(self)

    def _before_epoch(self):
        if self.before_epoch_func:
            self.before_epoch_func(self)

    def _before_step(self, *args, **kwargs):
        if self.before_step_func:
            self.before_step_func(self, *args, **kwargs)

    def _after_step(self, *args, **kwargs):
        if self.after_step_func:
            self.after_step_func(self, *args, **kwargs)

    def _trigger_step(self):
        if self.trigger_step_func:
            self.trigger_step_func(self)

    def _after_epoch(self):
        if self.after_epoch_func:
            self.after_epoch_func(self)

    def _trigger_epoch(self):
        if self.trigger_epoch_func:
            self.trigger_epoch_func(self)

    def _trigger(self):
        if self.trigger_func:
            self.trigger_func(self)

    def _after_train(self):
        if self.after_train_func:
            self.after_train_func(self)


class ProxyCallback(Callback):
    """
    A callback which proxy all methods to another callback.
    """

    def __init__(self, callback):
        assert isinstance(callback, Callback), type(callback)
        self.callback = callback

    def _set_trainer(self, trainer):
        self.callback.set_trainer(trainer)

    def _before_train(self):
        self.callback.before_train()

    def _before_epoch(self):
        self.callback.before_epoch()

    def _before_step(self, *args, **kwargs):
        self.callback.before_step(*args, **kwargs)

    def _after_step(self, *args, **kwargs):
        self.callback.after_step(*args, **kwargs)

    def _trigger_step(self):
        self.callback.trigger_step()

    def _after_epoch(self):
        self.callback.after_epoch()

    def _trigger_epoch(self):
        self.callback.trigger_epoch()

    def _trigger(self):
        self.callback.trigger()

    def _after_train(self):
        self.callback.after_train()

    def __str__(self):
        return 'Proxy-' + str(self.callback)


class Callbacks(Callback):
    """
    A container to hold callbacks.
    """

    def __init__(self, callbacks):
        for callback in callbacks:
            assert isinstance(callback, Callback), type(callback)
        self.callbacks = callbacks

    def _set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def _before_train(self):
        for callback in self.callbacks:
            callback.before_train()

    def _before_epoch(self):
        for callback in self.callbacks:
            callback.before_epoch()

    def _before_step(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.before_step(*args, **kwargs)

    def _after_step(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.after_step(*args, **kwargs)

    def _trigger_step(self):
        for callback in self.callbacks:
            callback.trigger_step()

    def _after_epoch(self):
        for callback in self.callbacks:
            callback.after_epoch()

    def _trigger_epoch(self):
        for callback in self.callbacks:
            callback.trigger_epoch()

    def _trigger(self):
        for callback in self.callbacks:
            callback.trigger()

    def _after_train(self):
        for callback in self.callbacks:
            callback.after_train()

    def append(self, callback):
        self.callbacks.append(callback)

    def extend(self, callbacks):
        self.callbacks.extend(callbacks)

    def __getitem__(self, index):
        return self.callbacks[index]

    def __len__(self):
        return len(self.callbacks)
