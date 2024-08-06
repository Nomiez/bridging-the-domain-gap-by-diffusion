from __future__ import annotations

from typing import Dict, Tuple

from PIL.Image import Image
from pathlib import Path

from sd_pipeline_typing.types import Module
from .config import SortConfig


class Sort(Module):
    config: SortConfig

    def __init__(self, *, config: SortConfig):
        self.config = config

    def run(
        self, input_data: Dict[str, str | Image] | Tuple[Dict[str, str | Image]], _
    ) -> Dict[str, str | Image] | Tuple[Dict[str, str | Image]]:
        if isinstance(input_data, dict):
            return input_data
        else:
            el = list(input_data)
            return tuple(sorted(el, key=self.config.key_function, reverse=self.config.reverse))
