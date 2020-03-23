import warnings

import torchvision.datasets as datasets
from torchvision.transforms import (CenterCrop, Compose, Normalize,
                                    RandomHorizontalFlip, RandomResizedCrop,
                                    Resize, ToTensor)

from torchpack.datasets.dataset import Dataset

__all__ = ['ImageNet']

# filter warnings for corrupted data
warnings.filterwarnings('ignore')


class ImageNetDataset(datasets.ImageNet):
    def __init__(self,
                 root,
                 split='train',
                 transform=None,
                 target_transform=None):
        super().__init__(root=root,
                         split=('train' if split == 'train' else 'val'),
                         transform=transform,
                         target_transform=target_transform)

    def __getitem__(self, index):
        images, classes = super().__getitem__(index)
        return dict(images=images, classes=classes)


class ImageNet(Dataset):
    def __init__(self, root, num_classes=1000, image_size=224,
                 transforms=None):
        if transforms is None:
            transforms = dict()
        if 'train' not in transforms:
            transforms['train'] = Compose([
                RandomResizedCrop(image_size),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])
        if 'test' not in transforms:
            transforms['test'] = Compose([
                Resize(int(image_size / 0.875)),
                CenterCrop(image_size),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])

        super().__init__({
            split: ImageNetDataset(root=root,
                                   split=split,
                                   transform=transforms[split])
            for split in ['train', 'test']
        })

        indices = dict()
        for k in range(num_classes):
            indices[k * (1000 // num_classes)] = k

        for split, dataset in self.items():
            samples = []
            for x, c in dataset.samples:
                if c in indices:
                    samples.append((x, indices[c]))
            dataset.samples = samples

            targets = []
            for c in dataset.targets:
                if c in indices:
                    targets.append(indices[c])
            dataset.targets = targets

            classes = []
            for c, x in enumerate(dataset.classes):
                if c in indices:
                    classes.append(x)
            dataset.classes = classes

            class_to_idx = {}
            for x, c in dataset.class_to_idx.items():
                if c in indices:
                    class_to_idx[x] = c
            dataset.class_to_idx = class_to_idx
