import os
import os.path as osp
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchpack.distributed as dist
import torchpack.utils.io as io
from torchpack.callbacks import (InferenceRunner, LambdaCallback, MaxSaver,
                                 Saver, SaverRestore)
from torchpack.callbacks.metrics import TopKCategoricalAccuracy
from torchpack.datasets.vision import ImageNet
from torchpack.environ import get_run_dir, set_run_dir
from torchpack.logging import get_logger
from torchpack.models.vision import MobileNetV2
from torchpack.train import Trainer

logger = get_logger(__file__)


class ClassificationTrainer(Trainer):
    def __init__(self, model, criterion, optimizer, scheduler):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _run_step(self, feed_dict):
        inputs = feed_dict['images'].cuda(non_blocking=True)
        targets = feed_dict['classes'].cuda(non_blocking=True)

        outputs = self.model(inputs)

        if self.model.training:
            loss = self.criterion(outputs, targets)
            self.monitors.add_scalar('loss', loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return dict(outputs=outputs, targets=targets)

    def _state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

    def _load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])


def main():
    dist.init()

    set_run_dir(osp.join('runs', 'imagenet100.mobilenetv2.size=112'))
    logger.info(' '.join([sys.executable] + sys.argv))

    cudnn.benchmark = True

    logger.info('Loading the dataset.')
    dataset = ImageNet(root='/dataset/imagenet/',
                       num_classes=100,
                       image_size=112)

    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split], rank=dist.rank(), shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(dataset[split],
                                                      sampler=sampler,
                                                      batch_size=64,
                                                      num_workers=16,
                                                      pin_memory=True)

    model = MobileNetV2(num_classes=100)
    model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                find_unused_parameters=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.05,
                                momentum=0.9,
                                weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=150)
    trainer = ClassificationTrainer(model=model,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    scheduler=scheduler)
    trainer.train_with_defaults(
        dataflow['train'],
        max_epoch=150,
        callbacks=[
            LambdaCallback(before_epoch=lambda self: dataflow['train'].sampler.
                           set_epoch(self.trainer.epoch_num)),
            LambdaCallback(before_epoch=lambda self: model.train(),
                           after_epoch=lambda self: model.eval()),
            LambdaCallback(before_epoch=lambda self: scheduler.step()),
            InferenceRunner(dataflow['test'],
                            callbacks=[
                                TopKCategoricalAccuracy(k=1, name='acc/top1'),
                                TopKCategoricalAccuracy(k=5, name='acc/top5')
                            ]),
            Saver(),
            MaxSaver('acc/top1')
        ])


if __name__ == '__main__':
    main()
