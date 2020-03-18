import json
import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os.path as osp
import torchpack.utils.io as io
from torchpack.callbacks import *
from torchpack.cuda.copy import async_copy_to
from torchpack.datasets.vision.imagenet import ImageNet
from torchpack.models.vision.mobilenetv2 import MobileNetV2
from torchpack.train import Trainer
from torchpack.utils.argument import ArgumentParser
from torchpack.utils.logging import get_logger, set_logger_dir

logger = get_logger(__file__)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--devices',
        action='set_devices',
        default='*',
        help='list of device(s) to use.',
    )
    parser.parse_args()

    dump_dir = osp.join('runs', 'imagenet100.mobilenetv2.size=112')
    set_logger_dir(dump_dir)

    logger.info(' '.join([sys.executable] + sys.argv))

    cudnn.benchmark = True

    logger.info('Loading the dataset.')
    dataset = ImageNet(
        root='/dataset/imagenet/',
        num_classes=100,
        image_size=112,
    )

    dataflow = dict()
    for split in dataset:
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            shuffle=(split == 'train'),
            batch_size=256,
            num_workers=16,
            pin_memory=True,
        )

    model = MobileNetV2(num_classes=100)
    model = nn.DataParallel(model.cuda())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=4e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=150,
    )

    class ClassificationTrainer(Trainer):
        def run_step(self, feed_dict):
            feed_dict = async_copy_to(feed_dict, device='cuda')
            inputs, targets = feed_dict['images'], feed_dict['labels']

            outputs = model(inputs)
            output_dict = dict(outputs=outputs)

            if model.training:
                loss = criterion(outputs, targets)

                output_dict['loss'] = loss.item()
                self.monitors.add_scalar('loss', loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            return async_copy_to(output_dict, device='cpu')

        def save(self, checkpoint_dir):
            io.save(osp.join(checkpoint_dir, 'model.pth'), model.state_dict())
            io.save(osp.join(checkpoint_dir, 'optimizer.pth'),
                    optimizer.state_dict())
            io.save(osp.join(checkpoint_dir, 'loop.json'),
                    dict(epoch_num=self.epoch_num))

        def load(self, checkpoint_dir):
            model.load_state_dict(
                io.load(osp.join(checkpoint_dir, 'model.pth')))
            optimizer.load_state_dict(
                io.load(osp.join(checkpoint_dir, 'optimizer.pth')))
            self.epoch_num = io.load(osp.join(checkpoint_dir,
                                              'loop.json'))['epoch_num']
            self.global_step = self.epoch_num * self.steps_per_epoch

    trainer = ClassificationTrainer()
    trainer.train(
        dataflow=dataflow['train'],
        max_epoch=150,
        callbacks=[
            Resumer(),
            LambdaCallback(
                before_epoch=lambda self: model.train(),
                after_epoch=lambda self: model.eval(),
            ),
            LambdaCallback(before_epoch=lambda self: scheduler.step(
                self.trainer.epoch_num - 1)),
            InferenceRunner(
                dataflow['test'],
                callbacks=[
                    ClassificationError(topk=1, name='error/top1'),
                    ClassificationError(topk=5, name='error/top5')
                ],
            ),
            Saver(),
            MinSaver('error/top1'),
            ConsoleWriter(),
            TFEventWriter(),
            JSONWriter(),
            ProgressBar(),
            EstimatedTimeLeft(),
        ],
    )


if __name__ == '__main__':
    main()
