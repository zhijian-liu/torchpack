import warnings

import torchvision.datasets as datasets
import torchvision.transforms as transforms

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
    def __init__(self, root, num_classes, image_size):
        super().__init__({
            'train':
            ImageNetDataset(
                root=root,
                split='train',
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )
                ]),
            ),
            'test':
            ImageNetDataset(
                root=root,
                split='val',
                transform=transforms.Compose([
                    transforms.Resize(int(image_size / 0.875)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )
                ]),
            )
        })

        # sample classes by strided indexing
        classes = dict()
        for k in range(num_classes):
            classes[k * (1000 // num_classes)] = k

        # reduce dataset to sampled classes
        # FIXME: update wnids and wnid_to_idx accordingly
        for d in self.values():
            d.samples = [(x, classes[c]) for x, c in d.samples if c in classes]
            d.targets = [classes[c] for c in d.targets if c in classes]
            d.classes = [x for c, x in enumerate(d.classes) if c in classes]
            d.class_to_idx = {
                x: c
                for x, c in d.class_to_idx.items() if c in classes
            }
