# Tensorflow VS Pytorch

Benchmarking of training and inference performance between Tensorflow and Pytorch.


## Config

```yaml
framework: pt        # [pt, tf]
seed: 25             # seed for reproducibility
n_classes: 100       # number of classes
bs: 32               # batch size
imgsz:               # image size
  - 32
  - 32
  - 3
lr: 0.001            # learning rate
epochs: 2            # number of epochs
num_workers: 0       # number of parallel workers
output: "output/pt"  # output directory path
data: "cifar100"     # dataset path
```

## Installation

- `pip install git+https://github.com/infocusp/tf_pt_benchmarking.git`

## How to run ?

### Download the data

- `downloadcifar100 --output path/to/data`

### Run benchmarking

- Update the `config.yaml` with required details
- `tfvspt --framework tf --config path/to/config.yaml`
- Benchmarking results will be saved in the `output` folder