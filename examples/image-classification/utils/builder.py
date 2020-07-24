from typing import Callable

import torch
from torch import nn

from torchpack.datasets.vision import ImageNet
from torchpack.models.vision import MobileNetV1, MobileNetV2, ShuffleNetV2
from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler

__all__ = [
    'make_dataset', 'make_model', 'make_criterion', 'make_optimizer',
    'make_scheduler'
]


def make_dataset() -> Dataset:
    if configs.dataset.name == 'imagenet':
        dataset = ImageNet(root=configs.dataset.root,
                           num_classes=configs.dataset.num_classes)
    else:
        raise NotImplementedError(configs.dataset.name)
    return dataset


def make_model() -> nn.Module:
    if configs.model.name == 'mobilenetv1':
        model = MobileNetV1(num_classes=configs.dataset.num_classes,
                            width_multiplier=configs.model.width_multipler)
    elif configs.model.name == 'mobilenetv2':
        model = MobileNetV2(num_classes=configs.dataset.num_classes,
                            width_multiplier=configs.model.width_multipler)
    elif configs.model.name == 'shufflenetv2':
        model = ShuffleNetV2(num_classes=configs.dataset.num_classes,
                             width_multiplier=configs.model.width_multipler)
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
