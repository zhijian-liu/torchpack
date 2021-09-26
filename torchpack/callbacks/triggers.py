from typing import Any, Callable, Dict, Optional

from .callback import Callback, ProxyCallback

__all__ = ['EnableCallbackIf', 'PeriodicTrigger', 'PeriodicCallback']


class EnableCallbackIf(ProxyCallback):
    """Enable the callback only if some condition holds."""

    def __init__(
        self,
        callback: Callback,
        predicate: Callable[[Callback], bool],
    ) -> None:
        super().__init__(callback)
        self.predicate = predicate

    def _before_epoch(self) -> None:
        if self.predicate(self):
            super()._before_epoch()

    def _before_step(self, feed_dict: Dict[str, Any]) -> None:
        if self.predicate(self):
            super()._before_step(feed_dict)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        if self.predicate(self):
            super()._after_step(output_dict)

    def _trigger_step(self) -> None:
        if self.predicate(self):
            super()._trigger_step()

    def _after_epoch(self) -> None:
        if self.predicate(self):
            super()._after_epoch()

    def _trigger_epoch(self) -> None:
        if self.predicate(self):
            super()._trigger_epoch()

    def __str__(self) -> str:
        return 'EnableCallbackIf-' + str(self.callback)


class PeriodicTrigger(ProxyCallback):
    """Trigger the callback every k steps or every k epochs."""

    def __init__(
        self,
        callback: Callback,
        *,
        every_k_epochs: Optional[int] = None,
        every_k_steps: Optional[int] = None,
    ) -> None:
        super().__init__(callback)
        assert every_k_epochs is not None or every_k_steps is not None, \
            '`every_k_epochs` and `every_k_steps` cannot both be None!'
        self.every_k_epochs = every_k_epochs
        self.every_k_steps = every_k_steps

    def _trigger_step(self) -> None:
        if (self.every_k_steps is not None
                and self.trainer.global_step % self.every_k_steps == 0):
            super()._trigger()

    def _trigger_epoch(self) -> None:
        if (self.every_k_epochs is not None
                and self.trainer.epoch_num % self.every_k_epochs == 0):
            super()._trigger()

    def __str__(self) -> str:
        return 'PeriodicTrigger-' + str(self.callback)


class PeriodicCallback(EnableCallbackIf):
    """Enable the callback every k steps or every k epochs.

    Note that this can only make a callback less frequent.
    """

    def __init__(
        self,
        callback: Callback,
        *,
        every_k_epochs: Optional[int] = None,
        every_k_steps: Optional[int] = None,
    ) -> None:
        assert every_k_epochs is not None or every_k_steps is not None, \
            '`every_k_epochs` and `every_k_steps` cannot both be None!'
        self.every_k_epochs = every_k_epochs
        self.every_k_steps = every_k_steps

        def predicate(self) -> bool:
            if (self.every_k_epochs is not None
                    and self.trainer.epoch_num % self.every_k_epochs == 0):
                return True
            if (self.every_k_steps is not None
                    and self.trainer.global_step % self.every_k_steps == 0):
                return True
            return False

        super().__init__(callback, predicate)

    def __str__(self) -> str:
        return 'PeriodicCallback-' + str(self.callback)
