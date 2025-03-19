"""Tensorflow Utils."""

import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_images(dataset) -> None:

    images, labels = next(iter(dataset))

    grid_size = int(len(images)**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            img = images[i].numpy()
            img = (img * 255).astype(np.uint8)
            ax.imshow(img)
            ax.axis('off')

    plt.show()


def plot_history(history) -> None:

    epochs = len(history.history['accuracy'])

    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(epochs),
             history.history['accuracy'],
             '-o',
             label='Train Accuracy',
             color='#ff7f0e')
    plt.ylabel('Accuracy', size=14)
    plt.xlabel('Epoch', size=14)
    plt.title("Training Accuracy")

    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(epochs),
             history.history['loss'],
             '-o',
             label='Train Loss',
             color='#1f77b4')
    plt.ylabel('Loss', size=14)
    plt.xlabel('Epoch', size=14)
    plt.title("Training Loss")

    plt.show()


def set_seed(seed: int) -> None:
    # Set random seed for Python
    random.seed(seed)
    # Set random seed for NumPy
    np.random.seed(seed)
    # Set random seed for TensorFlow
    tf.random.set_seed(seed)
