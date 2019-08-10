import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torchpack.callbacks import *
from torchpack.datasets.vision.imagenet import ImageNet
from torchpack.models.vision.mobilenetv2 import MobileNetV2
from torchpack.trainer import Trainer
from torchpack.utils.argument import ArgumentParser
from torchpack.utils.logging import get_logger

logger = get_logger(__file__)


class ClassificationTrainer(Trainer):
    def run_step(self, fd):
        inputs, targets = fd['inputs'], fd['targets']

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()

        return dict(outputs=outputs, loss=loss)


def main():
    parser = ArgumentParser()
    parser.add_argument('--devices', action='set_devices', default='*', help='list of device(s) to use.')
    args = parser.parse_args()

    cudnn.benchmark = True

    logger.info('Loading the dataset.')
    dataset = ImageNet(root='/dataset/imagenet/', num_classes=100, image_size=224)

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

    trainer = ClassificationTrainer()
    trainer.train(
        loader=loaders['train'], model=model, criterion=criterion, max_epoch=150,
        callbacks=[
            LambdaCallback(before_step=lambda *_: optimizer.zero_grad(),
                           after_step=lambda *_: optimizer.step()),
            LambdaCallback(before_epoch=lambda *_: scheduler.step()),
            PeriodicTrigger(
                InferenceRunner(loaders['test'], callbacks=[
                    ClassificationError(k=1, summary_name='acc/test-top1'),
                    ClassificationError(k=5, summary_name='acc/test-top5')
                ]),
                every_k_epochs=2
            ),
            ModelSaver(checkpoint_dir='runs/'),
            MaxSaver(monitor_stat='acc/test-top1', checkpoint_dir='runs/'),
            ProgressBar(),
            EstimatedTimeLeft()
        ],
        monitors=[
            TFEventWriter(logdir='runs/'),
            ScalarPrinter()
        ]
    )


if __name__ == '__main__':
    main()
