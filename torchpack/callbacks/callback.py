import traceback
import typing
from typing import Any, Callable, Dict, Iterator, List, Optional

from torchpack import distributed as dist

if typing.TYPE_CHECKING:
    from torchpack.train import Trainer
else:
    Trainer = None

__all__ = ['Callback', 'LambdaCallback', 'ProxyCallback', 'Callbacks']


class Callback:
    """Base class for all callbacks."""

    master_only: bool = False

    @property
    def enabled(self) -> bool:
        return dist.is_master() or not self.master_only

    def set_trainer(self, trainer: Trainer) -> None:
        self.trainer = trainer
        if self.enabled:
            self._set_trainer(trainer)

    def _set_trainer(self, trainer: Trainer) -> None:
        pass

    def before_train(self) -> None:
        if self.enabled:
            self._before_train()

    def _before_train(self) -> None:
        """Define what to do before training."""
        pass

    def before_epoch(self) -> None:
        if self.enabled:
            self._before_epoch()

    def _before_epoch(self) -> None:
        """Define what to do before every epoch."""
        pass

    def before_step(self, feed_dict: Dict[str, Any]) -> None:
        if self.enabled:
            self._before_step(feed_dict)

    def _before_step(self, feed_dict: Dict[str, Any]) -> None:
        """Define what to do before every step."""
        pass

    def after_step(self, output_dict: Dict[str, Any]) -> None:
        if self.enabled:
            self._after_step(output_dict)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        """Define what to do after every step."""
        pass

    def trigger_step(self) -> None:
        if self.enabled:
            self._trigger_step()

    def _trigger_step(self) -> None:
        """Define what to do after after step."""
        pass

    def after_epoch(self) -> None:
        if self.enabled:
            self._after_epoch()

    def _after_epoch(self) -> None:
        """Define what to do after every epoch."""
        pass

    def trigger_epoch(self) -> None:
        if self.enabled:
            self._trigger_epoch()

    def _trigger_epoch(self) -> None:
        """Define what to do after after epoch."""
        pass

    def trigger(self) -> None:
        if self.enabled:
            self._trigger()

    def _trigger(self) -> None:
        """Define a general trigger behavior, to be used with trigger schedulers.

        Note that the schedulers (e.g. :class:`PeriodicTrigger`) might call
        this method both inside an epoch and after an epoch.
        """
        pass

    def after_train(self) -> None:
        if self.enabled:
            try:
                self._after_train()
            except Exception:
                traceback.print_exc()

    def _after_train(self) -> None:
        """Define what to do after training."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        return self._state_dict() if self.enabled else {}

    def _state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.enabled:
            self._load_state_dict(state_dict)

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

    def __str__(self) -> str:
        return type(self).__name__


class LambdaCallback(Callback):
    """A callback created with lambda functions."""

    def __init__(
        self,
        *,
        set_trainer: Optional[Callable] = None,
        before_train: Optional[Callable] = None,
        before_epoch: Optional[Callable] = None,
        before_step: Optional[Callable] = None,
        after_step: Optional[Callable] = None,
        trigger_step: Optional[Callable] = None,
        after_epoch: Optional[Callable] = None,
        trigger_epoch: Optional[Callable] = None,
        trigger: Optional[Callable] = None,
        after_train: Optional[Callable] = None,
        state_dict: Optional[Callable] = None,
        load_state_dict: Optional[Callable] = None,
        master_only: bool = False,
    ) -> None:
        self.set_trainer_fn = set_trainer
        self.before_train_fn = before_train
        self.before_epoch_fn = before_epoch
        self.before_step_fn = before_step
        self.after_step_fn = after_step
        self.trigger_step_fn = trigger_step
        self.after_epoch_fn = after_epoch
        self.trigger_epoch_fn = trigger_epoch
        self.trigger_fn = trigger
        self.after_train_fn = after_train
        self.state_dict_fn = state_dict
        self.load_state_dict_fn = load_state_dict
        self.master_only = master_only

    def _set_trainer(self, trainer: Trainer) -> None:
        if self.set_trainer_fn:
            self.set_trainer_fn(self, trainer)

    def _before_train(self) -> None:
        if self.before_train_fn:
            self.before_train_fn(self)

    def _before_epoch(self) -> None:
        if self.before_epoch_fn:
            self.before_epoch_fn(self)

    def _before_step(self, feed_dict: Dict[str, Any]) -> None:
        if self.before_step_fn:
            self.before_step_fn(self, feed_dict)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        if self.after_step_fn:
            self.after_step_fn(self, output_dict)

    def _trigger_step(self) -> None:
        if self.trigger_step_fn:
            self.trigger_step_fn(self)

    def _after_epoch(self) -> None:
        if self.after_epoch_fn:
            self.after_epoch_fn(self)

    def _trigger_epoch(self) -> None:
        if self.trigger_epoch_fn:
            self.trigger_epoch_fn(self)

    def _trigger(self) -> None:
        if self.trigger_fn:
            self.trigger_fn(self)

    def _after_train(self) -> None:
        if self.after_train_fn:
            self.after_train_fn(self)

    def _state_dict(self) -> Dict[str, Any]:
        return self.state_dict_fn(self) if self.state_dict_fn else {}

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.load_state_dict_fn:
            self.load_state_dict_fn(self, state_dict)


class ProxyCallback(Callback):
    """A callback which proxy all methods to another callback."""

    def __init__(self, callback: Callback) -> None:
        assert isinstance(callback, Callback), type(callback)
        self.callback = callback

    def _set_trainer(self, trainer: Trainer) -> None:
        self.callback.set_trainer(trainer)

    def _before_train(self) -> None:
        self.callback.before_train()

    def _before_epoch(self) -> None:
        self.callback.before_epoch()

    def _before_step(self, feed_dict: Dict[str, Any]) -> None:
        self.callback.before_step(feed_dict)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        self.callback.after_step(output_dict)

    def _trigger_step(self) -> None:
        self.callback.trigger_step()

    def _after_epoch(self) -> None:
        self.callback.after_epoch()

    def _trigger_epoch(self) -> None:
        self.callback.trigger_epoch()

    def _trigger(self) -> None:
        self.callback.trigger()

    def _after_train(self) -> None:
        self.callback.after_train()

    def _state_dict(self) -> Dict[str, Any]:
        return self.callback.state_dict()

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.callback.load_state_dict(state_dict)

    def __str__(self) -> str:
        return 'Proxy-' + str(self.callback)


class Callbacks(Callback):
    """A container to hold callbacks."""

    def __init__(self, callbacks: List[Callback]) -> None:
        for callback in callbacks:
            assert isinstance(callback, Callback), type(callback)
        self.callbacks = callbacks

    def _set_trainer(self, trainer: Trainer) -> None:
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def _before_train(self) -> None:
        for callback in self.callbacks:
            callback.before_train()

    def _before_epoch(self) -> None:
        for callback in self.callbacks:
            callback.before_epoch()

    def _before_step(self, feed_dict: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.before_step(feed_dict)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.after_step(output_dict)

    def _trigger_step(self) -> None:
        for callback in self.callbacks:
            callback.trigger_step()

    def _after_epoch(self) -> None:
        for callback in self.callbacks:
            callback.after_epoch()

    def _trigger_epoch(self) -> None:
        for callback in self.callbacks:
            callback.trigger_epoch()

    def _trigger(self) -> None:
        for callback in self.callbacks:
            callback.trigger()

    def _after_train(self) -> None:
        for callback in self.callbacks:
            callback.after_train()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = {}
        for k, callback in enumerate(self.callbacks):
            state = callback.state_dict()
            if state:
                name = f'{str(callback).lower()}.{k}'
                state_dict[name] = state
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for k, callback in enumerate(self.callbacks):
            name = f'{str(callback).lower()}.{k}'
            if name in state_dict:
                callback.load_state_dict(state_dict[name])

    def __getitem__(self, index: int) -> Callback:
        return self.callbacks[index]

    def __len__(self) -> int:
        return len(self.callbacks)

    def __iter__(self) -> Iterator[Callback]:
        return iter(self.callbacks)
