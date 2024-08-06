from __future__ import annotations

from sd_pipeline_typing.types import Config


class MultiplyConfig(Config):
    def __init__(self, *, factor: int = 1):
        self.factor = factor
