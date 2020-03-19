import torchvision.datasets as datasets
from torchvision.transforms import *

from torchpack.datasets.dataset import Dataset

__all__ = ['CIFAR']


class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, root, train=True,  transform=None, \
                 target_transform=None, download=False):
        super().__init__(root=root, train=train, transform=transform, \
                         target_transform=target_transform, download=download)

    def __getitem__(self, index):
        images, labels = super().__getitem__(index)
        return dict(images=images, labels=labels)


class CIFAR100Dataset(datasets.CIFAR100):
    def __init__(self, root, train=True,  transform=None, \
                 target_transform=None, download=False):
        super().__init__(root=root, train=train, transform=transform, \
                         target_transform=target_transform, download=download)

    def __getitem__(self, index):
        images, labels = super().__getitem__(index)
        return dict(images=images, labels=labels)


class CIFAR(Dataset):
    def __init__(self, root, num_classes=10, \
                 transforms=None, image_size=32):
        if num_classes == 10:
            CIFARDataset = CIFAR10Dataset
        elif num_classes == 100:
            CIFARDataset = CIFAR100Dataset
        else:
            raise NotImplementedError('only support CIFAR10/100 for now')

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
            'train': CIFARDataset(root=root, train=True, download=True, \
                                  transform=transforms['train']),
            'test': CIFARDataset(root=root, train=False, download=True, \
                                 transform=transforms['test'])
        })
