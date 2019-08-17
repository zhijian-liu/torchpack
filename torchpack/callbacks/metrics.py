from torchpack.callbacks.callback import Callback

__all__ = ['ClassificationError']


class ClassificationError(Callback):
    def __init__(self, k, logits_name='outputs', labels_name='targets', metric_name='acc/valid-top1'):
        self.k = k
        self.logits_name = logits_name
        self.labels_name = labels_name
        self.metric_name = metric_name

    def _before_epoch(self):
        self.num_examples = 0
        self.num_correct = 0

    def _after_step(self, feed_dict, output_dict):
        logits = output_dict[self.logits_name]
        labels = feed_dict[self.labels_name]

        _, indices = logits.topk(self.k, 1, True, True)

        indices = indices.transpose(0, 1)
        masks = indices.eq(labels.view(1, -1).expand_as(indices))

        self.num_examples += labels.size(0)
        self.num_correct += masks[:self.k].view(-1).float().sum(0)

    def _after_epoch(self):
        self.trainer.monitors.add_scalar(self.metric_name, self.num_correct / max(self.num_examples, 1) * 100.)
