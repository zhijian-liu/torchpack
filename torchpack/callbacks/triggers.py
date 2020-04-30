from .callback import ProxyCallback

__all__ = ['EnableCallbackIf', 'PeriodicTrigger', 'PeriodicCallback']


class EnableCallbackIf(ProxyCallback):
    """
    Enable the callback only if some condition holds.
    """
    def __init__(self, callback, predicate):
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
    Trigger the callback every k steps or every k epochs.
    """
    def __init__(self, callback, *, every_k_epochs=None, every_k_steps=None):
        super().__init__(callback)
        assert every_k_epochs is not None or every_k_steps is not None, \
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
    Enable the callback every k steps or every k epochs.
    Note that this can only make a callback less frequent.
    """
    def __init__(self, callback, *, every_k_epochs=None, every_k_steps=None):
        super().__init__(callback, PeriodicCallback.predicate)
        assert every_k_epochs is not None or every_k_steps is not None, \
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
