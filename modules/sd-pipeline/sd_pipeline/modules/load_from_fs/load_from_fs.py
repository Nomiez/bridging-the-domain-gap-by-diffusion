import os
from typing import Dict, Tuple

from PIL.Image import Image, open

from sd_pipeline_typing.types import Module

from .config import LFFSConfig


class LFFS(Module):
    def __init__(self, *, config: LFFSConfig):
        self.config = config

    def run(self, input_data: str, _) -> Dict[str, str | Image]:
        # Open all images in the input directory
        output = {}
        for image_name in os.listdir(self.config.input_dir):
            image_path = os.path.join(self.config.input_dir, image_name)
            image = open(image_path)
            output["name"] = image_name
            output["image"] = image

        return output
