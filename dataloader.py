from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision.datasets
from torch.utils.data import random_split
import torch

torch.manual_seed(0)


def get_transform(split):
    if split == "train":
        return A.Compose(
            [
                A.ColorJitter(),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, p=0.9),
                A.RandomResizedCrop(224, 224, scale=[0.5, 1]),
                A.Blur(p=0.5),
                A.ToGray(p=0.1),
                A.CoarseDropout(max_height=16, max_width=16, p=0.1),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])


# https://stackoverflow.com/a/59615584
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            img = np.array(x)
            transformed = self.transform(image=img)
        return transformed["image"], y

    def __len__(self):
        return len(self.subset)


def create_dataloaders(augs=True):
    data = torchvision.datasets.ImageFolder("data/Animals-151")

    train_set_size = int(len(data) * 0.8)
    val_set_size = len(data) - train_set_size

    train_subset, val_subset = random_split(data, [train_set_size, val_set_size])

    train_set = DatasetFromSubset(
        train_subset, transform=get_transform("train" if augs else "not_train")
    )
    val_set = DatasetFromSubset(val_subset, transform=get_transform("val"))

    return train_set, val_set
