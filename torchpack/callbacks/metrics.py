from torchpack.callbacks.callback import Callback

__all__ = ['ClassificationError']


class ClassificationError(Callback):
    def __init__(self, topk=1, logits='outputs', labels='targets', name='accuracy'):
        self.topk = topk
        self.logits = logits
        self.labels = labels
        self.name = name

    def _before_epoch(self):
        self.num_examples = 0
        self.num_correct = 0

    def _after_step(self, feed_dict, output_dict):
        labels = feed_dict[self.labels]
        logits = output_dict[self.logits]

        _, indices = logits.topk(self.topk, dim=1)
        masks = indices.eq(labels.view(-1, 1).expand_as(indices))

        self.num_examples += labels.size(0)
        self.num_correct += masks.sum().item()

    def _after_epoch(self):
        self.trainer.monitors.add_scalar(self.name, self.num_correct / self.num_examples * 100)
