from typing import Any, Dict

import torch

import torchpack.distributed as dist
from torchpack.callbacks.callback import Callback

__all__ = [
    'TopKCategoricalAccuracy', 'CategoricalAccuracy', 'MeanSquaredError',
    'MeanAbsoluteError'
]


class TopKCategoricalAccuracy(Callback):
    def __init__(self,
                 k: int,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'accuracy') -> None:
        self.k = k
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.name = name

    def _before_epoch(self) -> None:
        self.size = 0
        self.corrects = 0

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]

        _, indices = outputs.topk(self.k, dim=1)
        masks = indices.eq(targets.view(-1, 1).expand_as(indices))

        self.size += targets.size(0)
        self.corrects += masks.sum().item()

    def _after_epoch(self) -> None:
        self.size = dist.allreduce(self.size, reduction='sum')
        self.corrects = dist.allreduce(self.corrects, reduction='sum')
        self.trainer.summary.add_scalar(self.name,
                                        self.corrects / self.size * 100)


class CategoricalAccuracy(TopKCategoricalAccuracy):
    def __init__(self,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'accuracy') -> None:
        super().__init__(k=1,
                         output_tensor=output_tensor,
                         target_tensor=target_tensor,
                         name=name)


class MeanSquaredError(Callback):
    def __init__(self,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'error') -> None:
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.name = name

    def _before_epoch(self):
        self.size = 0
        self.errors = 0

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]

        error = torch.mean((outputs - targets) ** 2)

        self.size += targets.size(0)
        self.errors += error.item() * targets.size(0)

    def _after_epoch(self) -> None:
        self.size = dist.allreduce(self.size, reduction='sum')
        self.errors = dist.allreduce(self.errors, reduction='sum')
        self.trainer.summary.add_scalar(self.name, self.errors / self.size)


class MeanAbsoluteError(Callback):
    def __init__(self,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'error') -> None:
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.name = name

    def _before_epoch(self) -> None:
        self.size = 0
        self.errors = 0

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]

        error = torch.mean(torch.abs(outputs - targets))

        self.size += targets.size(0)
        self.errors += error.item() * targets.size(0)

    def _after_epoch(self) -> None:
        self.size = dist.allreduce(self.size, reduction='sum')
        self.errors = dist.allreduce(self.errors, reduction='sum')
        self.trainer.summary.add_scalar(self.name, self.errors / self.size)
