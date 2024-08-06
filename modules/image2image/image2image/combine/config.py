from __future__ import annotations

from sd_pipeline_typing.types import Config


class I2ICombinedConfig(Config):
    def __init__(self, *, prompt: str, strength: float = 0.2):
        self.prompt = prompt
        self.strength = strength
