import warnings

from torchvision import datasets, transforms

__all__ = ['ImageNet']

# filter warnings for corrupted data
warnings.filterwarnings('ignore')


class ImageNet(dict):
    def __init__(self, root, image_size):
        super().__init__({
            'train': datasets.ImageNet(
                root=root, split='train', download=True,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            ),
            'test': datasets.ImageNet(
                root=root, split='val', download=True,
                transform=transforms.Compose([
                    transforms.Resize(int(image_size / 0.875)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            )
        })
