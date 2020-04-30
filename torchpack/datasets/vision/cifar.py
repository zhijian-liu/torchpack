import torchvision.datasets as datasets
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, Resize, ToTensor)

from ..dataset import Dataset

__all__ = ['CIFAR']


class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=True):
        super().__init__(root=root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

    def __getitem__(self, index):
        images, classes = super().__getitem__(index)
        return dict(images=images, classes=classes)


class CIFAR100Dataset(datasets.CIFAR100):
    def __init__(self,
                 root,
                 train,
                 transform=None,
                 target_transform=None,
                 download=True):
        super().__init__(root=root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

    def __getitem__(self, index):
        images, classes = super().__getitem__(index)
        return dict(images=images, classes=classes)


class CIFAR(Dataset):
    def __init__(self, root, num_classes=10, image_size=32, transforms=None):
        if num_classes == 10:
            CIFARDataset = CIFAR10Dataset
        elif num_classes == 100:
            CIFARDataset = CIFAR100Dataset
        else:
            raise NotImplementedError('only CIFAR-10/100 are supported')

        if transforms is None:
            transforms = dict()
        if 'train' not in transforms:
            transforms['train'] = Compose([
                RandomCrop(image_size, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.4914, 0.4822, 0.4465],
                          std=[0.2023, 0.1994, 0.2010])
            ])
        if 'test' not in transforms:
            transforms['test'] = Compose([
                Resize(image_size),
                ToTensor(),
                Normalize(mean=[0.4914, 0.4822, 0.4465],
                          std=[0.2023, 0.1994, 0.2010])
            ])

        super().__init__({
            split: CIFARDataset(root=root,
                                train=(split == 'train'),
                                transform=transforms[split])
            for split in ['train', 'test']
        })
