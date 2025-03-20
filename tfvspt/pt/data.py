"""Pytorch Dataset."""

import os

from PIL import Image
from torch.utils.data import Dataset

from tfvspt.config.config import Config


class ClassificationDataset(Dataset):

    def __init__(
        self,
        config: Config,
        paths: list[str],
        transforms=None,
    ) -> None:
        self.config = config
        self.paths = paths
        self.transforms = transforms

    def __len__(self) -> None:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert('RGB')
        label = int(path.split(os.path.sep)[-2])
        if self.transforms:
            image = self.transforms(image)
        return image, label
