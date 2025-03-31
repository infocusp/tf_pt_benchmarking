"""Config."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from tfvspt.utils import read_yaml


class Framework(str, Enum):
    tensorflow = "tf"
    pytorch = "pt"


class Config(BaseModel):
    framework: Framework
    seed: int
    n_classes: int
    bs: int
    imgsz: tuple
    lr: float
    epochs: int
    num_workers: int
    output: Path
    data: Path

    def model_post_init(self, __context):
        self.output.mkdir(parents=True, exist_ok=True)

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["framework"] = data["framework"].value
        # Convert PosixPath to string in the dictionary
        for path in ["output", "data"]:
            data[path] = str(data[path])
        return data


def get_config(path: str) -> Config:
    return Config(**read_yaml(path))
