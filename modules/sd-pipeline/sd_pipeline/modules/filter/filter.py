from __future__ import annotations

from typing import Dict, Tuple

from PIL.Image import Image
from pathlib import Path

from sd_pipeline_typing.types import Module
from .config import FilterConfig


class Filter(Module):
    config: FilterConfig

    def __init__(self, *, config: FilterConfig):
        self.config = config

    def run(
        self, input_data: Dict[str, str | Image] | Tuple[Dict[str, str | Image]], _
    ) -> None | Dict[str, str | Image] | Tuple[Dict[str, str | Image]]:
        if isinstance(input_data, dict):
            return input_data if self.config.filter_function(input_data) else None
        else:
            elements = list(input_data)
            res = ()
            for el in elements:
                if self.config.filter_function(el):
                    res += (el,)
            return res
