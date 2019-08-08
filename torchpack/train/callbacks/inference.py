from abc import ABCMeta

import six
from tensorpack.utils import logger

from .callback import Callback

__all__ = ['Inferencer', 'ClassificationError']


@six.add_metaclass(ABCMeta)
class Inferencer(Callback):
    """ Base class of Inferencer.
    Inferencer is a special kind of callback that should be called by :class:`InferenceRunner`.
    It has the methods ``_get_fetches`` and ``_on_fetches`` which are like
    :class:`SessionRunHooks`, except that they will be used only by :class:`InferenceRunner`.
    .. document private functions
    .. automethod:: _before_inference
    .. automethod:: _after_inference
    .. automethod:: _get_fetches
    .. automethod:: _on_fetches
    """

    def _before_epoch(self):
        self._before_inference()

    def _before_inference(self):
        """
        Called before a new round of inference starts.
        """
        pass

    def _trigger_epoch(self):
        ret = self._after_inference()
        if ret is None:
            return
        for k, v in six.iteritems(ret):
            try:
                v = float(v)
            except ValueError:
                logger.warn("{} returns a non-scalar statistics!".format(type(self).__name__))
                continue
            else:
                self.trainer.monitors.put_scalar(k, v)

    def _after_inference(self):
        """
        Called after a round of inference ends.
        Returns a dict of scalar statistics which will be logged to monitors.
        """
        pass

    def on_fetches(self, input_dict, output_dict):
        """
        Called after each new datapoint finished the forward inference.
        """
        self._on_fetches(input_dict, output_dict)

    def _on_fetches(self, input_dict, output_dict):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError()


class ClassificationError(Inferencer):
    """
    Compute **true** classification error in batch mode, from a ``wrong`` tensor.
    The ``wrong`` tensor is supposed to be an binary vector containing
    whether each sample in the batch is *incorrectly* classified.
    You can use ``tf.nn.in_top_k`` to produce this vector.
    This Inferencer produces the "true" error, which could be different from
    ``ScalarStats('error_rate')``.
    It takes account of the fact that batches might not have the same size in
    testing (because the size of test set might not be a multiple of batch size).
    Therefore the result can be different from averaging the error rate of each batch.
    You can also use the "correct prediction" tensor, then this inferencer will
    give you "classification accuracy" instead of error.
    """

    def __init__(self, k, logit_tensor_name='outputs', label_tensor_name='targets', summary_name='validation_error'):
        """
        Args:
            wrong_tensor_name(str): name of the ``wrong`` binary vector tensor.
            summary_name(str): the name to log the error with.
        """
        self.k = k
        self.logit_tensor_name = logit_tensor_name
        self.label_tensor_name = label_tensor_name
        self.summary_name = summary_name

    def _before_inference(self):
        self.num_examples = 0
        self.num_correct = 0

    def _on_fetches(self, input_dict, output_dict):
        outputs = output_dict[self.logit_tensor_name]
        targets = input_dict[self.label_tensor_name]

        _, indices = outputs.topk(self.k, 1, True, True)

        indices = indices.transpose(0, 1)
        masks = indices.eq(targets.view(1, -1).expand_as(indices))

        self.num_examples += targets.size(0)
        self.num_correct += masks[:self.k].view(-1).float().sum(0)

    def _after_inference(self):
        return {self.summary_name: self.num_correct / max(self.num_examples, 1) * 100.}
