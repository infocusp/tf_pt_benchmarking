"""Train, Eval & Log Pytorch training."""

import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from tqdm.autonotebook import tqdm

from tfvspt.base import BenchmarkingBase
from tfvspt.pt.data import ClassificationDataset
from tfvspt.pt.model import Model


class PytorchBenchmarking(BenchmarkingBase):

    def __init__(self, config_path):
        super().__init__(config_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"{self.config.framework}, device, {self.device}")

    def set_seed(self, seed):
        # Set the random seed for numpy, random, and torch
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # Set the seed for CUDA (if using GPUs)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Ensure deterministic results for CuDNN (can be slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.logger.info(f"{self.config.framework}, seed, {self.config.seed}")

    def get_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.config.imgsz[:2]),
            transforms.ToTensor(),
        ])

    def get_augmentations(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    def _get_dataset(self, data: str) -> ClassificationDataset:
        return ClassificationDataset(
            config=self.config,
            paths=self.raw_data[data],
            transforms=self.get_transforms(),
            augmentations=self.get_augmentations() if data == "train" else None,
            cache=data == "train",
        )

    def load_dataloaders(self) -> dict:
        st = time.time()
        dataloaders = {
            "train":
                DataLoader(
                    dataset=self._get_dataset("train"),
                    batch_size=self.config.bs,
                    shuffle=True,
                    num_workers=self.config.num_workers,
                ),
            "test":
                DataLoader(
                    dataset=self._get_dataset("test"),
                    batch_size=self.config.bs,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                )
        }
        self._log_stats("data_loading_time", time.time() - st)
        return dataloaders

    def plot_images(self, dataloader, name: str) -> None:
        images, _ = next(iter(dataloader))
        images = images.numpy().transpose((0, 2, 3, 1))
        return super().plot_images(images, name)

    def _train(self, dataloader: DataLoader, model: torch.nn.Module,
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer) -> dict:

        model.to(self.device)

        history = {"loss": [], "accuracy": []}
        for epoch in range(self.config.epochs):

            model.train()
            running_loss = 0.0
            total = 0
            correct = 0

            self.logger.info(f"Epoch {epoch + 1} / {self.config.epochs}")
            for images, targets in tqdm(dataloader):

                images, targets = images.to(self.device), targets.to(
                    self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                predicted = torch.argmax(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            # Epoch Stats
            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = round(correct / total, 4)

            # Collect logs
            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_accuracy)

            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}"
            )

        return history

    def _eval(self, dataloader: DataLoader, model: torch.nn.Module,
              criterion: torch.nn.Module) -> dict:

        model.to(self.device)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in tqdm(dataloader):

                images, targets = images.to(self.device), targets.to(
                    self.device)

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

                predicted = torch.argmax(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss /= len(dataloader)
        val_accuracy = round(correct / total, 4)

        return {"loss": val_loss, "accuracy": val_accuracy}

    def train(self) -> None:
        # plot train images
        self.plot_images(dataloader=self.dataloaders["train"],
                         name="train_data")

        # load model
        st = time.time()
        model = Model(n_classes=self.config.n_classes)
        self._log_stats("model_building_time", time.time() - st)
        model_summary = summary(model,
                                input_size=(self.config.bs,
                                            self.config.imgsz[-1],
                                            *self.config.imgsz[:-1]))
        self.logger.info(f"{self.config.framework}, {model_summary}")

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        # criterion
        criterion = torch.nn.CrossEntropyLoss()

        # train model
        st = time.time()
        history = self._train(
            dataloader=self.dataloaders["train"],
            model=model,
            criterion=criterion,
            optimizer=optimizer,
        )
        self._log_stats("training_time", time.time() - st)

        # plot the history
        self.plot_history(history=history)

        # save model
        st = time.time()
        torch.jit.script(model).save(str(self.config.output / "cifar100.pt"))
        self._log_stats("model_saving_time", time.time() - st)

    def eval(self) -> dict:
        # plot test images
        self.plot_images(dataloader=self.dataloaders["test"], name="test_data")

        # load model
        st = time.time()
        model = torch.jit.load(str(self.config.output / "cifar100.pt"))
        self._log_stats("model_loading_time", time.time() - st)

        # evaluate
        st = time.time()
        results = self._eval(
            dataloader=self.dataloaders["test"],
            model=model,
            criterion=torch.nn.CrossEntropyLoss(),
        )
        self.logger.info(f"{self.config.framework}, test results, {results}")
        self._log_stats("eval_time", time.time() - st)

        return results
