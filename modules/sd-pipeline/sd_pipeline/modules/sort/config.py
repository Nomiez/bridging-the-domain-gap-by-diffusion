from __future__ import annotations

from sd_pipeline_typing.types import Config
from typing import Callable


class SortConfig(Config):
    def __init__(self, *, key_function: Callable, reverse: bool = False):
        self.key_function = key_function
        self.reverse = reverse
