from pydantic import BaseModel
import yaml


class Config(BaseModel):
    seed: int
    n_classes: int
    bs: int
    imgsz: tuple
    lr: float
    epochs: int
    num_workers: int
    output: str


def get_config(path: str) -> Config:
    config = yaml.safe_load(open(path))
    return Config(**config)
