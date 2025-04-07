"""Benchmarking Base."""

import matplotlib.pyplot as plt
import numpy as np

from tfvspt.config.config import get_config
from tfvspt.logger import get_logger
from tfvspt.utils import save_yaml


class BenchmarkingBase:

    def __init__(self, config_path: str):
        self.config = get_config(config_path)
        self.logger = get_logger(name=self.config.framework.value,
                                 path=str(self.config.output / "logs.log"))
        self.stats = self._init_stats()
        self.raw_data = None
        self.dataloaders = None

    def _init_stats(self) -> dict:
        keys = [
            "data_loading_time", "model_building_time", "training_time",
            "model_saving_time", "eval_time"
        ]
        return {key: None for key in keys}

    def set_seed(self, seed: int) -> None:
        raise NotImplementedError

    def _log_stats(self, key: str, value: float) -> None:
        self.stats[key] = value
        self.logger.info(f"{self.config.framework}, {key}, {value}")

    def _get_raw_data(self) -> dict[str, list[str]]:
        train = list(map(str, self.config.data.glob("train/*/*.jpg")))
        test = list(map(str, self.config.data.glob("test/*/*.jpg")))
        self.logger.info(
            f"{self.config.framework}, train data, {len(train)}, test data, {len(test)}"
        )
        return {"train": train, "test": test}

    def load_dataloaders(self) -> dict:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def eval(self) -> dict:
        raise NotImplementedError

    def plot_images(self, images: np.ndarray, name: str) -> None:
        grid_size = int(len(images)**0.5)
        _, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        for i, ax in enumerate(axes.flatten()):
            if i < len(images):
                img = (images[i] * 255).astype(np.uint8)
                ax.imshow(img)
                ax.axis('off')
        plt.savefig(str(self.config.output / f"{name}.jpg"))
        plt.show()

    def plot_history(self, history: dict) -> None:

        epochs = range(len(history['loss']))

        plt.figure(figsize=(12, 12))
        plt.subplot(2, 1, 1)
        plt.plot(epochs,
                 history['accuracy'],
                 '-o',
                 label='Train Accuracy',
                 color='#ff7f0e')
        plt.ylabel('Accuracy', size=14)
        plt.xlabel('Epoch', size=14)
        plt.title("Training Accuracy")

        plt.subplot(2, 1, 2)
        plt.plot(epochs,
                 history['loss'],
                 '-o',
                 label='Train Loss',
                 color='#1f77b4')
        plt.ylabel('Loss', size=14)
        plt.xlabel('Epoch', size=14)
        plt.title("Training Loss")

        plt.tight_layout()
        plt.savefig(str(self.config.output / "history.png"))
        plt.show()

    def start(self) -> None:
        # set seed
        self.set_seed(seed=self.config.seed)
        # save the config
        save_yaml(self.config.model_dump(),
                  str(self.config.output / "config.yaml"))
        # get raw data
        self.raw_data = self._get_raw_data()
        # load dataloaders
        self.dataloaders = self.load_dataloaders()
        # train
        self.train()
        # eval
        results = self.eval()
        # save results
        save_yaml(results, str(self.config.output / "results.yaml"))
        # save stats
        save_yaml(self.stats, str(self.config.output / "stats.yaml"))
