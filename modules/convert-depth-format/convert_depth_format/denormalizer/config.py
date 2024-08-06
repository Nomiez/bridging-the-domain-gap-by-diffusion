from __future__ import annotations

from sd_pipeline_typing.types import Config
from typing import Callable


class DenormalizerConfig(Config):
    def __init__(self, *, denormalize_function: Callable):
        self.denormalize_function = denormalize_function
