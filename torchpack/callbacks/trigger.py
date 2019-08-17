from torchpack.callbacks.callback import ProxyCallback

__all__ = ['EnableCallbackIf', 'PeriodicTrigger', 'PeriodicCallback']


class EnableCallbackIf(ProxyCallback):
    """
    Disable `{before,after}_epoch`, `{before,after}_step`, `trigger_{epoch,step}` methods of
    the given callback, unless some condition satisfies.
    """

    def __init__(self, callback, predicate):
        """
        Args:
            callback (Callback): a Callback instance.
            predicate (self -> bool): a callable predicate, which has to be a pure function.
                The callback is disabled unless this predicate returns True.
        """
        super().__init__(callback)
        self.predicate = predicate

    def before_epoch(self):
        if self.predicate(self):
            super().before_epoch()

    def before_step(self, *args, **kwargs):
        if self.predicate(self):
            super().before_step(*args, **kwargs)

    def after_step(self, *args, **kwargs):
        if self.predicate(self):
            super().after_step(*args, **kwargs)

    def trigger_step(self):
        if self.predicate(self):
            super().trigger_step()

    def after_epoch(self):
        if self.predicate(self):
            super().after_epoch()

    def trigger_epoch(self):
        if self.predicate(self):
            super().trigger_epoch()

    def __str__(self):
        return 'EnableCallbackIf-' + str(self.callback)


class PeriodicTrigger(ProxyCallback):
    """
    Trigger the given callback every k steps or every k epochs.
    """

    def __init__(self, callback, every_k_epochs=None, every_k_steps=None):
        """
        Args:
            callback (Callback): a Callback instance with a trigger method to be called.
            every_k_epochs (int): trigger when ``epoch_num % k == 0``.
            every_k_steps (int): trigger when ``global_step % k == 0``.
        """
        super().__init__(callback)
        assert (every_k_epochs is not None) or (every_k_steps is not None), \
            '`every_k_epochs` and `every_k_steps` cannot both be None!'
        self.every_k_epochs = every_k_epochs
        self.every_k_steps = every_k_steps

    def _trigger_epoch(self):
        if self.every_k_epochs is not None and self.trainer.epoch_num % self.every_k_epochs == 0:
            self.callback.trigger()

    def _trigger_step(self):
        if self.every_k_steps is not None and self.trainer.global_step % self.every_k_steps == 0:
            self.callback.trigger()

    def __str__(self):
        return 'PeriodicTrigger-' + str(self.callback)


class PeriodicCallback(EnableCallbackIf):
    """
    The `{before,after}_epoch`, `{before,after}_run`, `trigger_{epoch,step}` methods of
    the given callback will be enabled only when `epoch_num % every_k_epochs == 0` or
    `global_step % every_k_steps == 0`. The other methods are unaffected.
    Note that this can only makes a callback less frequent than itself.
    """

    def __init__(self, callback, every_k_epochs=None, every_k_steps=None):
        """
        Args:
            callback (Callback): a Callback instance.
            every_k_epochs (int): enable the callback when ``epoch_num % k == 0``.
            every_k_steps (int): enable the callback when ``global_step % k == 0``.
        """
        super().__init__(callback, PeriodicCallback.predicate)
        assert (every_k_epochs is not None) or (every_k_steps is not None), \
            '`every_k_epochs` and `every_k_steps` cannot both be None!'
        self.every_k_epochs = every_k_epochs
        self.every_k_steps = every_k_steps

    def predicate(self):
        if self.every_k_epochs is not None and self.trainer.epoch_num % self.every_k_epochs == 0:
            return True
        if self.every_k_steps is not None and self.trainer.global_step % self.every_k_steps == 0:
            return True
        return False

    def __str__(self):
        return 'PeriodicCallback-' + str(self.callback)
