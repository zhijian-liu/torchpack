from torchvision import datasets, transforms

from ..dataset import Dataset

__all__ = ['CIFAR']


class CIFAR(Dataset):
    def __init__(self, root, num_classes, image_size):
        assert num_classes in [10, 100]

        if num_classes == 10:
            dataset = datasets.CIFAR10
        else:
            dataset = datasets.CIFAR100

        super().__init__({
            'train': dataset(
                root=root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            ),
            'test': dataset(
                root=root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            )
        })
