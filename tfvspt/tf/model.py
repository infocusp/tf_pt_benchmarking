"""Tensorflow Model."""

from tensorflow.keras import layers
from tensorflow.keras import models


def get_model(
    input_shape: tuple[int, int, int],
    n_classes: int,
):
    # backbone
    channels = [32, 64, 128, 128]
    layers_ = [layers.InputLayer(input_shape)]
    for i in range(len(channels)):
        layers_ += [
            layers.Conv2D(channels[i], (3, 3), activation=None, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
        ]
    # global average pooling
    layers_ += [layers.GlobalAveragePooling2D()]
    # fc
    layers_ += [
        layers.Dense(128, activation='relu'),
        layers.Dense(n_classes, activation=None),
    ]
    return models.Sequential(layers_)
