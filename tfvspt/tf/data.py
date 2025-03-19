"""Tensorflow Data Loader."""

import numpy as np
import tensorflow as tf
from tensorflow.data import AUTOTUNE

from tfvspt.config.config import Config


class Dataset:

    def __init__(self, config: Config) -> None:
        self.config = config

    @staticmethod
    def get_data():
        return tf.keras.datasets.cifar100.load_data()

    def preprocess_data(self, image, label):
        image = tf.cast(image, tf.float32)
        image = tf.cast(image / 255.0, tf.float32)
        label = tf.squeeze(tf.one_hot(label, self.config.n_classes), axis=0)
        return image, label

    def load_dataset(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        shuffle: bool = False,
        repeat: bool = False,
    ):
        # Create tf dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = (dataset.map(self.preprocess_data,
                               num_parallel_calls=AUTOTUNE).cache())
        # Shuffle
        if shuffle:
            dataset = dataset.shuffle(images.shape[0])
        # Batch
        dataset = (dataset.batch(self.config.bs).prefetch(AUTOTUNE))
        # Repeat
        if repeat:
            dataset = dataset.repeat()
        return dataset
