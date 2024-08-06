from __future__ import annotations

from sd_pipeline_typing.types import Config


class I2IConfig(Config):
    def __init__(self, *, prompt: str, negative_prompt: str, strength: float = 0.2):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.strength = strength
