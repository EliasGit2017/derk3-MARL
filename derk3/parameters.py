from typing import Dict, Union
from pathlib import Path
import json


class Parameters:
    """
    Hyperparameters taken form the `parameters.json` file
    """

    def __init__(self, params_or_json_path: Union[Dict, Path] = {}) -> None:
        if isinstance(params_or_json_path, dict):
            self.update(params_or_json_path)
        else:
            self.load(params_or_json_path)

    def save(self, json_path: Path) -> None:
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def load(self, json_path: Path) -> None:
        with open(json_path) as f:
            params = json.load(f)
        self.update(params)

    def update(self, params: Dict) -> None:
        self.__dict__.update(params)

    def __getitem__(self, name):
        return self.__dict__[name]

    @property
    def dict(self):
        return self.__dict__
