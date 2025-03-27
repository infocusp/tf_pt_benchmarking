"""Download CIFAR100 Data."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm.autonotebook import tqdm


# Function to save images as JPG
def save_images(x_data: np.ndarray, y_data: np.ndarray,
                directory: Path) -> None:
    for i, (img, label) in tqdm(enumerate(zip(x_data, y_data))):
        # Convert the image to a PIL Image object
        img_pil = Image.fromarray(img).resize((64, 64))

        # Create a label-specific folder if it doesn't exist
        label_dir = directory / str(label[0])
        label_dir.mkdir(parents=True, exist_ok=True)

        # Save the image as a .jpg file
        img_filename = str(label_dir / f'{i}.jpg')
        img_pil.save(img_filename)


def entrypoint() -> None:

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Download CIFAR100 Dataset")

    # Add the framework argument (string, e.g., 'pytorch', 'tensorflow')
    parser.add_argument('--output',
                        type=Path,
                        required=True,
                        help='Specify the output path')

    args = parser.parse_args()

    # Load CIFAR-100 data from TensorFlow
    data = tf.keras.datasets.cifar100.load_data()

    output_path = args.output
    # Create directories & Save Images
    for idx, split in enumerate(["train", "test"]):
        path = output_path / split
        path.mkdir(parents=True, exist_ok=True)
        save_images(*data[idx], path)

    print("CIFAR-100 images have been saved as JPG files!")
