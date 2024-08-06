from __future__ import annotations

from typing import Dict

from PIL.Image import Image
from pathlib import Path

from sd_pipeline_typing.types import Module
from .config import MultiplyConfig


class Multiply(Module):
    config: MultiplyConfig

    def __init__(self, *, config: MultiplyConfig):
        self.config = config

    def run(self, input_data: Dict[str, str | Image], _) -> Dict[str, str | Image]:
        if self.config.factor == 1:
            return input_data
        else:
            res = ()
            for i in range(0, self.config.factor):
                new_dict = input_data.copy()
                new_dict["name"] = (
                    Path(new_dict["name"]).stem + f"_{i + 1}" + Path(new_dict["name"]).suffix
                )
                res += (new_dict,)

            return res
