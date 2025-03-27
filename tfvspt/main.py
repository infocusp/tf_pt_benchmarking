"""Main Script."""

import argparse

from tfvspt.config.config import Framework
from tfvspt.pt.main import PytorchBenchmarking
from tfvspt.tf.main import TensorflowBenchmarking

MAPPING = {
    Framework.tensorflow.value: TensorflowBenchmarking,
    Framework.pytorch.value: PytorchBenchmarking,
}


def entrypoint():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Select framework and provide configuration")

    # Add the framework argument (string, e.g., 'pytorch', 'tensorflow')
    parser.add_argument(
        '--framework',
        type=Framework,
        required=True,
        help=
        'Specify the machine learning framework to use (e.g., pytorch, tensorflow)'
    )

    # Add the config argument (string, e.g., path to config file)
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='Path to the configuration file')

    # Parse arguments
    args = parser.parse_args()

    # Print the arguments to verify
    print(f"Selected framework: {args.framework}")
    print(f"Configuration file: {args.config}")

    # Run Benchmarking
    Benchmarking = MAPPING[args.framework]
    benchmarking = Benchmarking(config_path=args.config)
    benchmarking.start()
