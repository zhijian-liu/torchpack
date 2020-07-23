from typing import Callable

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.optim.optimizer import Optimizer

from torchpack.datasets.dataset import Dataset
from torchpack.datasets.vision import ImageNet
from torchpack.models.vision.mobilenetv2 import MobileNetV2
from torchpack.utils.config import configs

__all__ = ['make_dataset', 'make_model']


def make_dataset() -> Dataset:
    if configs.dataset.name == 'imagenet':
        dataset = ImageNet(root=configs.dataset.root)
    else:
        raise NotImplementedError(configs.dataset.name)
    return dataset


def make_model() -> nn.Module:
    if configs.model.name == 'mobilenetv2':
        model = MobileNetV2(num_classes=configs.dataset.num_classes)
    else:
        raise NotImplementedError(configs.model.name)
    return model


def make_criterion() -> Callable:
    if configs.criterion.name == 'xent':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion


def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=configs.optimizer.lr,
            momentum=configs.optimizer.momentum,
            weight_decay=configs.optimizer.weight_decay)
    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.max_epoch)
    else:
        raise NotImplementedError(configs.scheduler.name)
    return scheduler
