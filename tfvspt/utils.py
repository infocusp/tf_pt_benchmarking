"""Common Utils."""

from typing import Any

import yaml


def read_yaml(path: str) -> dict:
    return yaml.safe_load(open(path))


def save_yaml(data: Any, path: str) -> None:
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
