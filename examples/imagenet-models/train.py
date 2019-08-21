import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

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
    parser.add_argument('--devices', action='set_devices', default='*', help='list of device(s) to use.')
    parser.parse_args()

    dump_dir = os.path.join('runs', 'imagenet100')
    set_logger_dir(dump_dir)

    logger.info(' '.join([sys.executable] + sys.argv))

    cudnn.benchmark = True

    logger.info('Loading the dataset.')
    dataset = ImageNet(root='/dataset/imagenet/', num_classes=100, image_size=112)

    loaders = dict()
    for split in dataset:
        loaders[split] = torch.utils.data.DataLoader(
            dataset[split],
            shuffle=(split == 'train'),
            batch_size=256,
            num_workers=16,
            pin_memory=True
        )

    model = MobileNetV2(num_classes=100).cuda()
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    class ClassificationTrainer(Trainer):
        def run_step(self, feed_dict):
            feed_dict = async_copy_to(feed_dict, device='cuda')
            inputs, targets = feed_dict['inputs'], feed_dict['targets']

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

        def save_checkpoint(self, checkpoint_dir):
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model.pth'))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer.pth'))

    trainer = ClassificationTrainer()
    trainer.train(
        dataflow=loaders['train'],
        max_epoch=150,
        callbacks=[
            LambdaCallback(before_epoch=lambda _: model.train(), after_epoch=lambda _: model.eval()),
            LambdaCallback(before_epoch=lambda _: scheduler.step()),
            InferenceRunner(loaders['test'], callbacks=[
                ClassificationError(topk=1, name='error/top1'),
                ClassificationError(topk=5, name='error/top5')
            ]),
            Saver(),
            MinSaver('error/top1'),
            ProgressBar(),
            ScalarPrinter(regexes=['error/', 'loss']),
            EstimatedTimeLeft(),
            TFEventWriter(),
            JSONWriter()
        ]
    )


if __name__ == '__main__':
    main()
