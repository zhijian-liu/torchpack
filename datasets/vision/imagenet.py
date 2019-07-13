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

        # sample subset by strided indexing
        classes = dict()
        for k in range(num_classes):
            classes[k * (1000 // num_classes)] = k

        # select samples and targets from subset
        for split, dataset in self.items():
            samples, targets = [], []
            for x, y in dataset.samples:
                if y in classes:
                    samples.append((x, classes[y]))
                    targets.append(classes[y])

            self[split].samples = samples
            self[split].targets = targets

        # for split in ['train', 'test']:
        #     print(self[split].targets)
        # print(self[split].wnids)
        # print(self[split].wnid_to_idx)

        # for split in ['train', 'test']:
        # wnid_to_classes = self._load_meta_file()[0]

        # super(ImageNet, self).__init__(self.split_folder, **kwargs)
        # self.root = root

        # idcs = [idx for _, idx in self.imgs]
        # self.wnids = self.classes
        # self.wnid_to_idx = {wnid: idx for idx, wnid in zip(idcs, self.wnids)}
        # self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        # self.class_to_idx = {cls: idx
        #                      for clss, idx in zip(self.classes, idcs)
        #                      for cls in clss}
