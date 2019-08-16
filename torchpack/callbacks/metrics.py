from torchpack.callbacks.inference import InferenceCallback

__all__ = ['ClassificationAcc']


class ClassificationAcc(InferenceCallback):
    def __init__(self, k, logits_name='outputs', labels_name='targets', metric_name='validation_error'):
        self.k = k
        self.logits_name = logits_name
        self.labels_name = labels_name
        self.metric_name = metric_name

    def _before_inference(self):
        self.num_examples = 0
        self.num_correct = 0

    def _after_step(self, feed_dict, output_dict):
        outputs = output_dict[self.logits_name]
        targets = feed_dict[self.labels_name]

        _, indices = outputs.topk(self.k, 1, True, True)

        indices = indices.transpose(0, 1)
        masks = indices.eq(targets.view(1, -1).expand_as(indices))

        self.num_examples += targets.size(0)
        self.num_correct += masks[:self.k].view(-1).float().sum(0)

    def _after_inference(self):
        self.trainer.monitors.add_scalar(self.metric_name, self.num_correct / max(self.num_examples, 1) * 100.)
