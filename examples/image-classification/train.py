import os.path as osp
import sys

import torch
import torch.nn as nn
import torchpack.distributed as dist
from torchpack.callbacks import (InferenceRunner, MaxSaver, Saver,
                                 SaverRestore, TopKCategoricalAccuracy)
from torchpack.datasets.vision import ImageNet
from torchpack.environ import set_run_dir
from torchpack.models.vision import MobileNetV2
from torchpack.utils.logging import logger

from utils.trainer import ClassificationTrainer


def main():
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    set_run_dir(osp.join('runs', 'imagenet100.mobilenetv2'))
    logger.info(' '.join([sys.executable] + sys.argv))

    logger.info('Loading the dataset.')
    dataset = ImageNet(root='/dataset/imagenet/', num_classes=100)
    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(dataset[split],
                                                      sampler=sampler,
                                                      batch_size=64,
                                                      num_workers=4,
                                                      pin_memory=True)

    logger.info('Building the trainer.')
    model = MobileNetV2(num_classes=100)
    model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                device_ids=[dist.local_rank()],
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
            SaverRestore(),
            Saver(),
            InferenceRunner(dataflow['test'],
                            callbacks=[
                                TopKCategoricalAccuracy(k=1, name='acc/top1'),
                                TopKCategoricalAccuracy(k=5, name='acc/top5')
                            ]),
            MaxSaver('acc/top1')
        ])


if __name__ == '__main__':
    main()
