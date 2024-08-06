from __future__ import annotations

from sd_pipeline_typing.types import Config


class LFFSConfig(Config):
    def __init__(self, *, input_dir: str, clear_dir: bool = False):
        self.input_dir = input_dir
        self.clear_dir = clear_dir
