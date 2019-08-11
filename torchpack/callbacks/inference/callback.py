from abc import ABCMeta

import six
from tensorpack.utils import logger

from torchpack.utils.logging import logger

__all__ = ['InferenceCallback']


@six.add_metaclass(ABCMeta)
class InferenceCallback(object):
    """ Base class of all inference callbacks.
    """

    def before_inference(self):
        """ Called before a round of inference starts.
        """
        pass

    def trigger_epoch(self):
        self.trigger()

    def trigger(self):
        monitors = self.after_inference()
        if monitors is None:
            return
        for k, v in monitors.items():
            try:
                v = float(v)
            except ValueError:
                logger.warn('{} returns a non-scalar statistics!'.format(type(self).__name__))
                continue
            else:
                self.trainer.monitors.add_scalar(k, v)

    def after_inference(self):
        """ Called after a round of inference ends.
        Returns a dict of scalar statistics which will be logged to monitors.
        """
        pass
