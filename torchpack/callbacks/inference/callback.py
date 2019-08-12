from abc import ABCMeta

import six

__all__ = ['InferenceCallback']


@six.add_metaclass(ABCMeta)
class InferenceCallback(object):
    """
    Base class of inference callbacks.
    """

    def set_trainer(self, trainer):
        self.trainer = trainer

    def before_inference(self):
        """
        Called before inference starts.
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

    def after_inference(self):
        """
        Called after inference ends.
        """
        pass
