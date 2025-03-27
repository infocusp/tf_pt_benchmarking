"""Pytorch Dataset."""

import os

from PIL import Image
from stocaching import get_shm_size
from stocaching import SharedCache
import torch
from torch.utils.data import Dataset

from tfvspt.config.config import Config


class ClassificationDataset(Dataset):

    def __init__(
        self,
        config: Config,
        paths: list[str],
        transforms=None,
        augmentations=None,
        cache: bool = False,
    ) -> None:
        self.config = config
        self.paths = paths
        self.transforms = transforms
        self.augmentations = augmentations
        self.cache = None
        if cache:
            self.cache = SharedCache(
                size_limit_gib=get_shm_size(),
                dataset_len=len(self.paths),
                data_dims=(self.config.imgsz[-1], *self.config.imgsz[:-1]),
                dtype=torch.float32,
            )

    def __len__(self) -> None:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = int(path.split(os.path.sep)[-2])
        image = None
        if self.cache:
            image = self.cache.get_slot(idx)
        if image is None:
            image = Image.open(path).convert('RGB')
            if self.transforms:
                image = self.transforms(image)
            if self.cache:
                self.cache.set_slot(idx, image)
        if self.augmentations:
            image = self.augmentations(image)
        return image, label
