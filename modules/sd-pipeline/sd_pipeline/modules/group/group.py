from __future__ import annotations

from typing import Dict, Tuple

from PIL.Image import Image
from pathlib import Path

from sd_pipeline_typing.types import Module
from .config import GroupConfig


class Group(Module):
    config: GroupConfig

    def __init__(self, *, config: GroupConfig):
        self.config = config

    def run(self, input_data: Tuple | Dict, _) -> Tuple:
        if isinstance(input_data, dict):
            return input_data
        else:
            res = ()
            for i in range(len(input_data[0])):
                temp_res = ()
                for j in range(len(input_data)):
                    temp_res += (input_data[j][i],)
                res += (temp_res,)

            return res
