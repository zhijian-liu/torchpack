from .meter import Meter

__all__ = ['TopKClassMeter']


class TopKClassMeter(Meter):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def reset(self):
        self.num_examples = 0
        self.num_correct = 0

    def update(self, outputs, targets):
        _, indices = outputs.topk(self.k, 1, True, True)

        indices = indices.transpose(0, 1)
        masks = indices.eq(targets.view(1, -1).expand_as(indices))

        self.num_examples += targets.size(0)
        self.num_correct += masks[:self.k].view(-1).float().sum(0)

    def compute(self):
        return self.num_correct / max(self.num_examples, 1) * 100.
