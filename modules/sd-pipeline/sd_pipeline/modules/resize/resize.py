from typing import Dict

import PIL
from PIL.Image import Image

from sd_pipeline_typing.types import Module
from .config import ResizeConfig


class Resize(Module):
    config: ResizeConfig

    def __init__(self, *, config: ResizeConfig):
        self.config = config

    def run(self, input_data: Dict[str, str | Image], _) -> Dict[str, str | Image]:

        input_data["image"] = input_data["image"].resize((self.config.W, self.config.H), PIL.Image.NEAREST)

        return input_data

