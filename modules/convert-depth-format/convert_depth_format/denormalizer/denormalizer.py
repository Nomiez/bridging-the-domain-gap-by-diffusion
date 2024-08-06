from __future__ import annotations

from PIL import Image
import numpy as np

from sd_pipeline_typing.types import Module

from .config import DenormalizerConfig


class Denormalizer(Module):
    def __init__(self, *, config: DenormalizerConfig):
        self.config = config

    def run(
        self,
        input_data: dict[str, str | Image.Image],
        pipeline_config,
    ) -> dict[str, str | Image.Image]:
        depth = np.array(input_data["image"])
        depth[depth > 0] = self.config.denormalize_function(depth)
        img_depth = Image.fromarray(depth)

        return {
            "image": img_depth,
            "name": input_data["name"],
        }
