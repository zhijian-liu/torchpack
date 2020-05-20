import torch

from .. import distributed as dist
from .callback import Callback

__all__ = [
    'TopKCategoricalAccuracy', 'CategoricalAccuracy', 'MeanSquaredError',
    'MeanAbsoluteError'
]


class TopKCategoricalAccuracy(Callback):
    def __init__(self,
                 k=1,
                 *,
                 output_tensor='outputs',
                 target_tensor='targets',
                 name='accuracy'):
        self.k = k
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.name = name

    def _before_epoch(self):
        self.size = 0
        self.corrects = 0

    def _after_step(self, output_dict):
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]

        _, indices = outputs.topk(self.k, dim=1)
        masks = indices.eq(targets.view(-1, 1).expand_as(indices))

        self.size += targets.size(0)
        self.corrects += masks.sum().item()

    def _after_epoch(self):
        self.size = dist.allreduce(self.size)
        self.corrects = dist.allreduce(self.corrects)
        self.trainer.monitors.add_scalar(self.name,
                                         self.corrects / self.size * 100)


class CategoricalAccuracy(TopKCategoricalAccuracy):
    def __init__(self,
                 *,
                 output_tensor='outputs',
                 target_tensor='targets',
                 name='accuracy'):
        super().__init__(k=1,
                         output_tensor=output_tensor,
                         target_tensor=target_tensor,
                         name=name)


class MeanSquaredError(Callback):
    def __init__(self,
                 *,
                 output_tensor='outputs',
                 target_tensor='targets',
                 name='error'):
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.name = name

    def _before_epoch(self):
        self.size = 0
        self.errors = 0

    def _after_step(self, output_dict):
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]

        error = torch.mean((outputs - targets) ** 2)

        self.size += targets.size(0)
        self.errors += error.item() * targets.size(0)

    def _after_epoch(self):
        self.size = dist.allreduce(self.size)
        self.errors = dist.allreduce(self.errors)
        self.trainer.monitors.add_scalar(self.name, self.errors / self.size)


class MeanAbsoluteError(Callback):
    def __init__(self,
                 *,
                 output_tensor='outputs',
                 target_tensor='targets',
                 name='error'):
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.name = name

    def _before_epoch(self):
        self.size = 0
        self.errors = 0

    def _after_step(self, output_dict):
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]

        error = torch.mean(torch.abs(outputs - targets))

        self.size += targets.size(0)
        self.errors += error.item() * targets.size(0)

    def _after_epoch(self):
        self.size = dist.allreduce(self.size)
        self.errors = dist.allreduce(self.errors)
        self.trainer.monitors.add_scalar(self.name, self.errors / self.size)
