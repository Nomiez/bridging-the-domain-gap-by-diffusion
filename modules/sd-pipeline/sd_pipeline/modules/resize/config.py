from __future__ import annotations

from sd_pipeline_typing.types import Config


class ResizeConfig(Config):
    def __init__(self, *, W: int, H: int):
        self.W = W
        self.H = H
