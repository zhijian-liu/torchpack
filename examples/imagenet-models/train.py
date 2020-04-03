import json
import os
import os.path as osp
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import torchpack.utils.argparse as argparse
import torchpack.utils.io as io
from torchpack.callbacks import (AutoResumer, InferenceRunner, LambdaCallback,
                                 MaxSaver, Saver)
from torchpack.callbacks.metrics import TopKCategoricalAccuracy
from torchpack.cuda.copy import async_copy_to
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
        inputs = async_copy_to(feed_dict['images'], device='cuda')
        targets = async_copy_to(feed_dict['classes'], device='cuda')

        outputs = self.model(inputs)

        if self.model.training:
            loss = self.criterion(outputs, targets)
            self.monitors.add_scalar('loss', loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return dict(outputs=outputs, targets=targets)

    def _save_checkpoint(self, save_dir):
        io.save(osp.join(save_dir, 'model.pth'), self.model.state_dict())
        io.save(osp.join(save_dir, 'optimizer.pth'),
                self.optimizer.state_dict())
        io.save(osp.join(save_dir, 'scheduler.pth'),
                self.scheduler.state_dict())

    def _load_checkpoint(self, load_dir):
        self.model.load_state_dict(io.load(osp.join(load_dir, 'model.pth')))
        self.optimizer.load_state_dict(
            io.load(osp.join(load_dir, 'optimizer.pth')))
        self.scheduler.load_state_dict(
            io.load(osp.join(load_dir, 'scheduler.pth')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', action='set_device', default='*')
    parser.parse_args()

    set_run_dir(osp.join('runs', 'imagenet100.mobilenetv2.size=112'))

    logger.info(' '.join([sys.executable] + sys.argv))

    cudnn.benchmark = True

    logger.info('Loading the dataset.')
    dataset = ImageNet(root='/dataset/imagenet/',
                       num_classes=100,
                       image_size=112)

    dataflow = dict()
    for split in dataset:
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            shuffle=(split == 'train'),
            batch_size=256,
            num_workers=16,
            pin_memory=True)

    model = MobileNetV2(num_classes=100)
    model = nn.DataParallel(model.cuda())

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
        dataflow=dataflow['train'],
        max_epoch=150,
        callbacks=[
            AutoResumer(),
            LambdaCallback(before_epoch_fn=lambda self: model.train(),
                           after_epoch_fn=lambda self: model.eval()),
            LambdaCallback(before_epoch_fn=lambda self: scheduler.step()),
            InferenceRunner(dataflow['test'],
                            callbacks=[
                                TopKCategoricalAccuracy(k=1, name='acc/top1'),
                                TopKCategoricalAccuracy(k=5, name='acc/top5')
                            ]),
            Saver(),
            MaxSaver('acc/top1')
        ],
    )


if __name__ == '__main__':
    main()
