import warnings

from torchvision import datasets, transforms

from ..dataset import Dataset

__all__ = ['ImageNet']

# filter warnings for corrupted data
warnings.filterwarnings('ignore')


class ImageNet(Dataset):
    def __init__(self, root, num_classes, image_size):
        # todo: support customized transforms
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

        # sample classes by strided indexing
        classes = dict()
        for k in range(num_classes):
            classes[k * (1000 // num_classes)] = k

        # reduce dataset to sampled classes
        # fixme: edit wnids and wnid_to_idx accordingly
        for split, dataset in self.items():
            self[split].samples = [(x, classes[c]) for x, c in dataset.samples if c in classes]
            self[split].targets = [classes[c] for c in dataset.targets if c in classes]
            self[split].classes = [x for c, x in enumerate(dataset.classes) if c in classes]
            self[split].class_to_idx = {x: c for x, c in dataset.class_to_idx.items() if c in classes}
