from .callback import Callback, ProxyCallback

__all__ = ['PeriodicTrigger', 'PeriodicCallback', 'EnableCallbackIf']


class PeriodicTrigger(ProxyCallback):
    """
    Trigger a callback every k global steps or every k epochs by its :meth:`trigger()` method.
    Most existing callbacks which do something every epoch are implemented
    with :meth:`trigger()` method. By default the :meth:`trigger()` method will be called every epoch.
    This wrapper can make the callback run at a different frequency.
    All other methods (``before/after_run``, ``trigger_step``, etc) of the given callback
    are unaffected. They will still be called as-is.
    """

    def __init__(self, callback, every_k_steps=None, every_k_epochs=None, before_train=False):
        """
        Args:
            callback (Callback): a Callback instance with a trigger method to be called.
            every_k_steps (int): trigger when ``global_step % k == 0``.
            every_k_epochs (int): trigger when ``epoch_num % k == 0``.
            before_train (bool): trigger in the :meth:`before_train` method.
        """
        assert isinstance(callback, Callback), type(callback)
        super().__init__(callback)
        if before_train is False:
            assert (every_k_epochs is not None) or (every_k_steps is not None), \
                "Arguments to PeriodicTrigger have disabled the triggerable!"
        self.every_k_steps = every_k_steps
        self.every_k_epochs = every_k_epochs
        self.trigger_before_train = before_train

    def before_train(self):
        self.callback.before_train()
        if self.trigger_before_train:
            self.callback.trigger()

    def trigger_epoch(self):
        if self.every_k_epochs is None:
            return
        if self.trainer.epoch_num % self.every_k_epochs == 0:
            self.callback.trigger()

    def trigger_step(self):
        if self.every_k_steps is None:
            return
        if self.trainer.global_step % self.every_k_steps == 0:
            self.callback.trigger()

    def __str__(self):
        return 'PeriodicTrigger-' + str(self.callback)


class EnableCallbackIf(ProxyCallback):
    """
    Disable the ``{before,after}_epoch``, ``{before,after}_step``, ``trigger_{epoch,step}``
    methods of a callback, unless some condition satisfies.
    """

    def __init__(self, callback, predicate):
        """
        Args:
            callback (Callback):
            predicate (self -> bool): a callable predicate. Has to be a pure function.
                The callback is disabled unless this predicate returns True.
        """
        super().__init__(callback)
        self.predicate = predicate

    def before_epoch(self):
        self.enabled_epoch = self.predicate(self)
        if self.enabled_epoch:
            super().before_epoch()

    def after_epoch(self):
        if self.enabled_epoch:
            super().after_epoch()

    def before_step(self, *args, **kwargs):
        self.enabled_step = self.predicate(self)
        if self.enabled_step:
            super().before_step(*args, **kwargs)

    def after_step(self, *args, **kwargs):
        if self.enabled_step:
            super().after_step(*args, **kwargs)

    def trigger_epoch(self):
        if self.enabled_epoch:
            super().trigger_epoch()

    def trigger_step(self):
        if self.enabled_step:
            super().trigger_step()

    def __str__(self):
        return 'EnableCallbackIf-' + str(self.callback)


class PeriodicCallback(EnableCallbackIf):
    """
    The ``{before,after}_epoch``, ``{before,after}_run``, ``trigger_{epoch,step}``
    methods of the given callback will be enabled only when ``global_step % every_k_steps == 0`
    or ``epoch_num % every_k_epochs == 0``. The other methods are unaffected.
    Note that this can only makes a callback **less** frequent than itself.
    If you have a callback that by default runs every epoch by its :meth:`trigger()` method,
    use :class:`PeriodicTrigger` to schedule it more frequent than itself.
    """

    def __init__(self, callback, every_k_steps=None, every_k_epochs=None):
        """
        Args:
            callback (Callback): a Callback instance.
            every_k_steps (int): enable the callback when ``global_step % k == 0``.
            every_k_epochs (int): enable the callback when ``epoch_num % k == 0``.
                Also enable when the last step finishes (``epoch_num == max_epoch``
                and ``local_step == steps_per_epoch - 1``).
        every_k_steps and every_k_epochs can be both set, but cannot be both None.
        """
        assert isinstance(callback, Callback), type(callback)
        assert (every_k_epochs is not None) or (every_k_steps is not None), \
            'every_k_steps and every_k_epochs cannot both be None!'
        self.every_k_steps = every_k_steps
        self.every_k_epochs = every_k_epochs
        super().__init__(callback, PeriodicCallback.predicate)

    def predicate(self):
        if self.every_k_steps is not None and self.trainer.global_step % self.every_k_steps == 0:
            return True
        if self.every_k_epochs is not None and self.trainer.epoch_num % self.every_k_epochs == 0:
            return True
        if self.every_k_epochs is not None:
            if self.trainer.local_step == self.trainer.steps_per_epoch - 1 and \
                    self.trainer.epoch_num == self.trainer.max_epoch:
                return True
        return False

    def __str__(self):
        return 'PeriodicCallback-' + str(self.callback)
