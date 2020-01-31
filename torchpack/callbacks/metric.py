from torchpack.callbacks.callback import Callback

__all__ = ['ClassificationError']


class ClassificationError(Callback):
    def __init__(self, topk=1, output_tensor='outputs', target_tensor='labels', name='error'):
        self.topk = topk
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.name = name

    def _before_epoch(self):
        self.num_examples = 0
        self.num_errors = 0

    def _after_step(self, feed_dict, output_dict):
        targets = feed_dict[self.target_tensor]
        outputs = output_dict[self.output_tensor]

        _, indices = outputs.topk(self.topk, dim=1)
        masks = indices.eq(targets.view(-1, 1).expand_as(indices))

        self.num_examples += targets.size(0)
        self.num_errors += targets.size(0) - masks.sum().item()

    def _after_epoch(self):
        self.trainer.monitors.add_scalar(self.name, self.num_errors / self.num_examples * 100)
