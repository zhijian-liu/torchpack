from torchpack.callbacks.inference.callback import InferenceCallback

__all__ = ['ClassificationError']


class ClassificationError(InferenceCallback):
    def __init__(self, k, logit_tensor_name='outputs', label_tensor_name='targets', summary_name='validation_error'):
        self.k = k
        self.logit_tensor_name = logit_tensor_name
        self.label_tensor_name = label_tensor_name
        self.summary_name = summary_name

    def before_inference(self):
        self.num_examples = 0
        self.num_correct = 0

    def after_step(self, input_dict, output_dict):
        outputs = output_dict[self.logit_tensor_name]
        targets = input_dict[self.label_tensor_name]

        _, indices = outputs.topk(self.k, 1, True, True)

        indices = indices.transpose(0, 1)
        masks = indices.eq(targets.view(1, -1).expand_as(indices))

        self.num_examples += targets.size(0)
        self.num_correct += masks[:self.k].view(-1).float().sum(0)

    def after_inference(self):
        return {self.summary_name: self.num_correct / max(self.num_examples, 1) * 100.}
