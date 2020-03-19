import warnings

import torchvision.datasets as datasets
from torchvision.transforms import *

from torchpack.datasets.dataset import Dataset

__all__ = ['ImageNet']

# filter warnings for corrupted data
warnings.filterwarnings('ignore')


class ImageNetDataset(datasets.ImageNet):
    def __init__(self, root, split='train', **kwargs):
        super().__init__(root=root, split=split, **kwargs)

    def __getitem__(self, index):
        images, labels = super().__getitem__(index)
        return dict(images=images, labels=labels)


class ImageNet(Dataset):
    def __init__(self, root, num_classes=1000, \
                 transforms=None, image_size=224):
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
            'train': ImageNetDataset(root=root, split='train', \
                                     transform=transforms['train']),
            'test': ImageNetDataset(root=root, split='val', \
                                    transform=transforms['test'])
        })

        # sample classes by strided indexing
        classes = dict()
        for k in range(num_classes):
            classes[k * (1000 // num_classes)] = k

        # reduce dataset to sampled classes
        # FIXME: update wnids and wnid_to_idx accordingly
        for dataset in self.values():
            dataset.samples = [(x, classes[c]) for x, c in dataset.samples \
                               if c in classes]
            dataset.targets = [classes[c] for c in dataset.targets \
                               if c in classes]
            dataset.classes = [x for c, x in enumerate(dataset.classes) \
                               if c in classes]
            dataset.class_to_idx = {x: c for x, c in dataset.class_to_idx.items() \
                                    if c in classes}
