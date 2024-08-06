from __future__ import annotations

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
        img = input_data["image"]

        if not isinstance(img, PIL.Image.Image):
            raise ValueError("Image is not a PIL Image object.")

        input_data["image"] = img.resize((self.config.W, self.config.H), PIL.Image.NEAREST)  # type: ignore

        return input_data
