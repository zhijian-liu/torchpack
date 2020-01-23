import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchpack.datasets.dataset import Dataset

__all__ = ['CIFAR']


class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform,
                         download=download)

    def __getitem__(self, index):
        images, labels = super().__getitem__(index)
        return dict(images=images, labels=labels)


class CIFAR100Dataset(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform,
                         download=download)

    def __getitem__(self, index):
        images, labels = super().__getitem__(index)
        return dict(images=images, labels=labels)


class CIFAR(Dataset):
    def __init__(self, root, num_classes, image_size):
        if num_classes == 10:
            CIFARDataset = CIFAR10Dataset
        elif num_classes == 100:
            CIFARDataset = CIFAR100Dataset
        else:
            raise NotImplementedError('only support CIFAR10/100 for now')

        super().__init__({
            'train': CIFARDataset(
                root=root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            ),
            'test': CIFARDataset(
                root=root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            )
        })
