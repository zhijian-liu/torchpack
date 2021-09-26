import warnings
from typing import Any, Callable, Dict, Optional

from torchvision import datasets
from torchvision.transforms import (CenterCrop, Compose, Normalize,
                                    RandomHorizontalFlip, RandomResizedCrop,
                                    Resize, ToTensor)

from ..dataset import Dataset

__all__ = ['ImageNet']


class ImageNetDataset(datasets.ImageNet):

    def __init__(
        self,
        *,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            split=('train' if split == 'train' else 'val'),
            transform=transform,
            target_transform=target_transform,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            image, label = super().__getitem__(index)
        return {'image': image, 'class': label}


class ImageNet(Dataset):

    def __init__(
        self,
        *,
        root: str,
        num_classes: int = 1000,
        transforms: Optional[Dict[str, Callable]] = None,
    ) -> None:
        if transforms is None:
            transforms = {}
        if 'train' not in transforms:
            transforms['train'] = Compose([
                RandomResizedCrop(224),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
        if 'test' not in transforms:
            transforms['test'] = Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        super().__init__({
            split: ImageNetDataset(
                root=root,
                split=split,
                transform=transforms[split],
            ) for split in ['train', 'test']
        })

        indices = {}
        for k in range(num_classes):
            indices[k * (1000 // num_classes)] = k

        for dataset in self.values():
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
