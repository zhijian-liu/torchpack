from torchvision import datasets, transforms

from ..dataset import Dataset

__all__ = ['CIFAR10']


class CIFAR10(Dataset):
    def __init__(self, root, image_size):
        super().__init__({
            'train': datasets.CIFAR10(
                root=root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            ),
            'test': datasets.CIFAR10(
                root=root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            )
        })
