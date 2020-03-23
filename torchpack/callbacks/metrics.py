from torchpack.callbacks.callback import Callback

__all__ = ['TopKCategoricalAccuracy', 'CategoricalAccuracy']


class TopKCategoricalAccuracy(Callback):
    def __init__(self,
                 k=1,
                 output_tensor='outputs',
                 target_tensor='classes',
                 name='accuracy'):
        self.k = k
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.name = name

    def _before_epoch(self):
        self.num_examples = 0
        self.num_corrects = 0

    def _after_step(self, feed_dict):
        outputs = feed_dict[self.output_tensor]
        targets = feed_dict[self.target_tensor]

        _, indices = outputs.topk(self.k, dim=1)
        masks = indices.eq(targets.view(-1, 1).expand_as(indices))

        self.num_examples += targets.size(0)
        self.num_corrects += masks.sum().item()

    def _after_epoch(self):
        self.trainer.monitors.add_scalar(
            self.name, self.num_corrects / self.num_examples * 100)


class CategoricalAccuracy(TopKCategoricalAccuracy):
    def __init__(self,
                 output_tensor='outputs',
                 target_tensor='classes',
                 name='accuracy'):
        super().__init__(k=1,
                         output_tensor=output_tensor,
                         target_tensor=target_tensor,
                         name=name)
