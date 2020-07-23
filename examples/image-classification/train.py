import argparse
import os
import sys

import torch
import torch.backends.cudnn
import torch.utils.data

from torchpack import distributed as dist
from torchpack.callbacks import (InferenceRunner, MaxSaver, Saver,
                                 SaverRestore, TopKCategoricalAccuracy)
from torchpack.environ import set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from utils import builder
from utils.trainer import ClassificationTrainer


def main() -> None:
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    parser = argparse.ArgumentParser()
    parser.add_argument('cpath', metavar='FILE', help='path to config file.')
    args, opts = parser.parse_known_args()

    configs.load(args.cpath, recursive=True)
    configs.update(opts)

    run_name = '.'.join(os.path.splitext(args.cpath)[0].split(os.sep)[1:])
    set_run_dir(os.path.join('runs', run_name))

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info('\n' + str(configs))

    logger.info('Loading the dataset.')
    dataset = builder.make_dataset()

    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            sampler=sampler,
            batch_size=configs.batch_size // dist.size(),
            num_workers=configs.workers_per_gpu,
            pin_memory=True)

    logger.info('Building the trainer.')
    model = builder.make_model()
    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[dist.local_rank()],
        find_unused_parameters=True)
    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    trainer = ClassificationTrainer(model=model,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    scheduler=scheduler)
    trainer.train_with_defaults(
        dataflow['train'],
        max_epoch=configs.max_epoch,
        callbacks=[
            SaverRestore(),
            InferenceRunner(dataflow['test'],
                            callbacks=[
                                TopKCategoricalAccuracy(k=1, name='acc/top1'),
                                TopKCategoricalAccuracy(k=5, name='acc/top5')
                            ]),
            MaxSaver('acc/top1'),
            Saver()
        ])


if __name__ == '__main__':
    main()
