"""Train, Eval & Log Tensorflow training."""

import math
import random
import time

import numpy as np
import tensorflow as tf

from tfvspt.base import BenchmarkingBase
from tfvspt.tf.data import ClassificationDataset
from tfvspt.tf.model import get_model


class TensorflowBenchmarking(BenchmarkingBase):

    def set_seed(self, seed: int) -> None:
        # Set random seed for Python
        random.seed(seed)
        # Set random seed for NumPy
        np.random.seed(seed)
        # Set random seed for TensorFlow
        tf.random.set_seed(seed)
        self.logger.info(f"{self.config.framework}, seed, {self.config.seed}")

    def load_dataloaders(self) -> dict:
        dataset = ClassificationDataset(config=self.config)
        st = time.time()
        dataloaders = {
            "train":
                dataset.get_dataloader(self.raw_data["train"],
                                       augment=True,
                                       shuffle=True,
                                       repeat=True),
            "test":
                dataset.get_dataloader(self.raw_data["test"],
                                       augment=False,
                                       shuffle=False,
                                       repeat=False),
        }
        self._log_stats("data_loading_time", time.time() - st)
        return dataloaders

    def plot_images(self, dataloader, name: str) -> None:
        images, _ = next(iter(dataloader))
        images = images.numpy()
        return super().plot_images(images, name)

    def get_callbacks(self) -> list:
        # train history logger
        csv_logger = tf.keras.callbacks.CSVLogger(str(self.config.output /
                                                      'cifar100_training.csv'),
                                                  separator=",",
                                                  append=False)
        return [csv_logger]

    def get_steps_per_epoch(self) -> None:
        return math.ceil(len(self.raw_data["train"]) / self.config.bs)

    def train(self) -> None:

        # plot train images
        self.plot_images(dataloader=self.dataloaders["train"],
                         name="train_data")

        # load model
        st = time.time()
        model = get_model(input_shape=self.config.imgsz,
                          n_classes=self.config.n_classes)
        self._log_stats("model_building_time", time.time() - st)
        self.logger.info(f"{self.config.framework}, {model.summary()}")

        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.lr)
        # loss
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        # compile model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"],
        )

        # train model
        st = time.time()
        history = model.fit(
            self.dataloaders["train"],
            epochs=self.config.epochs,
            shuffle=False,
            steps_per_epoch=self.get_steps_per_epoch(),
            callbacks=self.get_callbacks(),
        )
        self._log_stats("training_time", time.time() - st)

        # plot the history
        self.plot_history(history=history.history)

        # save model
        st = time.time()
        model.save(str(self.config.output / "cifar100.keras"))
        self._log_stats("model_saving_time", time.time() - st)

    def eval(self) -> dict:
        # plot test images
        self.plot_images(dataloader=self.dataloaders["test"], name="test_data")

        # load model
        st = time.time()
        model = tf.keras.models.load_model(
            str(self.config.output / "cifar100.keras"))
        self._log_stats("model_loading_time", time.time() - st)

        # evaluate
        st = time.time()
        results = model.evaluate(self.dataloaders["test"], return_dict=True)
        self.logger.info(f"{self.config.framework}, test results, {results}")
        self._log_stats("eval_time", time.time() - st)

        return results
