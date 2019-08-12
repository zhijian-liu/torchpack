from abc import ABCMeta

import six

__all__ = ['InferenceCallback', 'InferenceCallbacks']


@six.add_metaclass(ABCMeta)
class InferenceCallback(object):
    """
    Base class of inference callbacks.
    """

    def set_trainer(self, trainer):
        self.trainer = trainer
        self._set_trainer(trainer)

    def _set_trainer(self, trainer):
        pass

    def before_inference(self):
        self._before_inference()

    def _before_inference(self):
        """
        Called before inference starts.
        """
        pass

    def after_inference(self):
        self._after_inference()

    def _after_inference(self):
        """
        Called after inference ends.
        """
        pass

    def before_step(self, *args, **kwargs):
        self._before_step(*args, **kwargs)

    def _before_step(self, *args, **kwargs):
        """
        Called before every step.
        """
        pass

    def after_step(self, *args, **kwargs):
        self._after_step(*args, **kwargs)

    def _after_step(self, *args, **kwargs):
        """
        Called after every step.
        """
        pass


@six.add_metaclass(ABCMeta)
class InferenceCallbacks(InferenceCallback):
    """
    A container to hold inference callbacks.
    """

    def __init__(self, callbacks):
        for callback in callbacks:
            assert isinstance(callback, InferenceCallback), type(callback)
        self.callbacks = callbacks

    def _set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def _before_inference(self):
        for callback in self.callbacks:
            callback.before_inference()

    def _after_inference(self):
        for callback in self.callbacks:
            callback.after_inference()

    def _before_step(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.before_step(*args, **kwargs)

    def _after_step(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.after_step(*args, **kwargs)

    def __len__(self):
        return len(self.callbacks)

    def __getitem__(self, index):
        return self.callbacks[index]
