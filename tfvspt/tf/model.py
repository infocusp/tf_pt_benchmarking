"""Tensorflow Model."""

from tensorflow.keras import layers
from tensorflow.keras import models


def get_model(
    input_shape: tuple[int, int, int],
    n_classes: int,
):
    model = models.Sequential([
        layers.InputLayer(input_shape),
        layers.Conv2D(32, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation=None, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_classes, activation=None)
    ])
    return model
