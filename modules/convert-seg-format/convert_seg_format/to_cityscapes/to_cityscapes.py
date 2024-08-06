from __future__ import annotations


from PIL import Image

from sd_pipeline_typing.types import Module

from .config import ToCityscapesConfig
from .._helper import get_color_map, replace


class ToCityscapes(Module):
    def __init__(self, *, config: ToCityscapesConfig):
        self.config = config

    def run(
        self,
        input_data: dict[str, str | Image.Image],
        pipeline_config,
    ) -> dict[str, str | Image.Image]:
        color_coding_ade = get_color_map("ade20k")
        color_coding_cs = get_color_map("cityscapes")

        img = input_data["image"]
        input_data["image"] = replace(img, color_coding_ade, color_coding_cs)

        return input_data
