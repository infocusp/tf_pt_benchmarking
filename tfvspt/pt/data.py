"""Pytorch Dataset."""

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from tfvspt.config.config import Config


class ClassificationDataset(Dataset):

    def __init__(
        self,
        config: Config,
        images: np.ndarray,
        labels: np.ndarray,
        transforms=None,
    ) -> None:
        self.config = config
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> None:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transforms:
            image = self.transforms(image)
        return image, label


def get_transforms(data: str):
    transforms_ = []
    if data == "train":
        transforms_ += []
    transforms_ += [transforms.ToTensor()]
    return transforms.Compose(transforms_)


def get_dataloader(dataset: ClassificationDataset, data: str, config: Config):
    dataloader = None
    if data == "train":
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.bs,
            shuffle=True,
            num_workers=config.num_workers,
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.bs,
            shuffle=False,
            num_workers=config.num_workers,
        )
    return dataloader
