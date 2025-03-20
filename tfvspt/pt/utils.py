"""Pytorch Utils."""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int):
    """
    Sets the seed for random number generation in PyTorch, NumPy, and Python's random module.
    Ensures reproducibility across different devices (CPU and GPU).
    
    Parameters:
    seed (int): The seed value to be set.
    """
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


def plot_images(dataloader) -> None:

    images, labels = next(iter(dataloader))

    grid_size = int(len(images)**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            img = images[i].numpy()
            img = img.transpose((1, 2, 0))
            img = (img * 255).astype(np.uint8)
            ax.imshow(img)
            ax.axis('off')

    plt.show()


def plot_history(history: dict[str, list]) -> None:

    epochs = range(len(history['loss']))

    # Plot accuracy
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(epochs,
             history['accuracy'],
             '-o',
             label='Accuracy',
             color='#1f77b4')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')

    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history['loss'], '-o', label='Loss', color='#ff7f0e')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.tight_layout()
    plt.show()
