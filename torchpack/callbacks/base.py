from abc import ABCMeta

import six

__all__ = ['Callback', 'ProxyCallback', 'LambdaCallback', 'CallbackGroup']


@six.add_metaclass(ABCMeta)
class Callback(object):
    """ Base class for all callbacks.
    """

    chief_only = True

    def set_trainer(self, trainer):
        self.trainer = trainer

    def before_train(self):
        """
        Called before training.
        """
        pass

    def after_train(self):
        """
        Called after training.
        """
        pass

    def before_epoch(self):
        """
        Called before every epoch.
        Usually you should use the :meth:`trigger` callback to run something between epochs.
        Use this method only when something really needs to be run **immediately** before each epoch.
        """
        pass

    def after_epoch(self):
        """
        Called after every epoch.
        Usually you should use the :meth:`trigger` callback to run something between epochs.
        Use this method only when something really needs to be run **immediately** after each epoch.
        """
        pass

    def before_step(self, *args, **kwargs):
        """
        Called before every step.
        """
        pass

    def after_step(self, *args, **kwargs):
        """
        Called after every step.
        """
        pass

    def trigger_epoch(self):
        """
        Called after after epoch.
        """
        pass

    def trigger_step(self):
        """
        Called after after step.
        """
        pass

    def trigger(self):
        """
        Override this method to define a general trigger behavior, to be used with trigger schedulers.
        Note that the schedulers (e.g. :class:`PeriodicTrigger`) might call this
        method both inside an epoch and after an epoch.
        """
        pass

    def __str__(self):
        return type(self).__name__


class ProxyCallback(Callback):
    """ A callback which proxy all methods to another callback.
        It's useful as a base class of callbacks which decorate other callbacks.
    """

    def __init__(self, callback):
        assert isinstance(callback, Callback), type(callback)
        self.callback = callback
        self.chief_only = callback.chief_only

    def set_trainer(self, trainer):
        self.trainer = trainer
        self.callback.set_trainer(trainer)

    def before_train(self):
        self.callback.before_train()

    def after_train(self):
        self.callback.after_train()

    def before_epoch(self):
        self.callback.before_epoch()

    def after_epoch(self):
        self.callback.after_epoch()

    def before_step(self, *args, **kwargs):
        self.callback.before_step(*args, **kwargs)

    def after_step(self, *args, **kwargs):
        self.callback.after_step(*args, **kwargs)

    def trigger_epoch(self):
        self.callback.trigger_epoch()

    def trigger_step(self):
        self.callback.trigger_step()

    def trigger(self):
        self.callback.trigger()

    def __str__(self):
        return 'Proxy-' + str(self.callback)


class LambdaCallback(Callback):
    """ Create a callback with some lambdas.
    """

    def __init__(self,
                 before_train=None, after_train=None,
                 before_epoch=None, after_epoch=None,
                 before_step=None, after_step=None,
                 trigger_epoch=None, trigger_step=None, trigger=None,
                 chief_only=True):
        self.before_train_fn = before_train
        self.after_train_fn = after_train
        self.before_epoch_fn = before_epoch
        self.after_epoch_fn = after_epoch
        self.before_step_fn = before_step
        self.after_step_fn = after_step
        self.trigger_epoch_fn = trigger_epoch
        self.trigger_step_fn = trigger_step
        self.trigger_fn = trigger
        self.chief_only = chief_only

    def before_train(self):
        if self.before_train_fn:
            self.before_train_fn(self)

    def after_train(self):
        if self.after_train_fn:
            self.after_train_fn(self)

    def before_epoch(self):
        if self.before_epoch_fn:
            self.before_epoch_fn(self)

    def after_epoch(self):
        if self.after_epoch_fn:
            self.after_epoch_fn(self)

    def before_step(self, *args, **kwargs):
        if self.before_step_fn:
            self.before_step_fn(self, *args, **kwargs)

    def after_step(self, *args, **kwargs):
        if self.after_step_fn:
            self.after_step_fn(self, *args, **kwargs)

    def trigger_epoch(self):
        if self.trigger_epoch_fn:
            self.trigger_epoch_fn(self)

    def trigger_step(self):
        if self.trigger_step_fn:
            self.trigger_step_fn(self)

    def trigger(self):
        if self.trigger_fn:
            self.trigger_fn(self)


class CallbackGroup(Callback):
    """ A container to hold all callbacks, and trigger them iteratively.
    """

    def __init__(self, callbacks):
        for callback in callbacks:
            assert isinstance(callback, Callback), callback.__class__
        self.callbacks = callbacks

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def before_train(self):
        for callback in self.callbacks:
            callback.before_train()

    def after_train(self):
        for callback in self.callbacks:
            callback.after_train()

    def before_epoch(self):
        for callback in self.callbacks:
            callback.before_epoch()

    def after_epoch(self):
        for callback in self.callbacks:
            callback.after_epoch()

    def before_step(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.before_step(*args, **kwargs)

    def after_step(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.after_step(*args, **kwargs)

    def trigger_epoch(self):
        for callback in self.callbacks:
            callback.trigger_step()

    def trigger_step(self):
        for callback in self.callbacks:
            callback.trigger_step()

    def trigger(self):
        for callback in self.callbacks:
            callback.trigger()
