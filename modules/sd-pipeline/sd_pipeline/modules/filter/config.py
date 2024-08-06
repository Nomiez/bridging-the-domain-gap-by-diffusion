from __future__ import annotations

from sd_pipeline_typing.types import Config
from typing import Callable


class FilterConfig(Config):
    def __init__(self, *, filter_function: Callable):
        self.filter_function = filter_function
