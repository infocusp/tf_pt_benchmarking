"""Tensorflow Data Loader."""

import os

import tensorflow as tf
from tensorflow.data import AUTOTUNE

from tfvspt.config.config import Config


class ClassificationDataset:

    def __init__(self, config: Config) -> None:
        self.config = config

    def preprocess_data(self, path: str):
        # read the image from disk, decode it, convert the data type to
        # floating point, and resize it
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.config.imgsz[:2])
        image = tf.cast(image / 255.0, tf.float32)
        # parse the class label from the file path
        label = tf.strings.split(path, os.path.sep)[-2]
        label = tf.strings.to_number(label, tf.int32)
        label = tf.one_hot(label, self.config.n_classes)
        return image, label

    def get_dataloader(
        self,
        paths: list,
        shuffle: bool = False,
        repeat: bool = False,
    ):
        # Create tf dataloader
        dataloader = tf.data.Dataset.from_tensor_slices(paths)
        dataloader = (dataloader.map(self.preprocess_data,
                                     num_parallel_calls=AUTOTUNE).cache())
        # Shuffle
        if shuffle:
            dataloader = dataloader.shuffle(len(paths))
        # Batch
        dataloader = (dataloader.batch(self.config.bs).prefetch(AUTOTUNE))
        # Repeat
        if repeat:
            dataloader = dataloader.repeat()
        return dataloader
