import os
from typing import Dict, Tuple

from PIL.Image import Image, open

from sd_pipeline_typing.types import Module

from .config import LFFSConfig


def _load_file(image_name: str, input_dir: str) -> dict:
    # Open all images in the input directory
    output = {}
    image_path = os.path.join(input_dir, image_name)
    image = open(image_path)
    output["name"] = image_name
    output["image"] = image
    return output


class LFFS(Module):
    def __init__(self, *, config: LFFSConfig):
        self.config = config

    def run(self, input_data: str, _) -> Dict[str, str | Image] | Tuple[Dict[str, str | Image]]:
        res = ()
        for image_name in os.listdir(self.config.input_dir):
            if image_name.endswith(".png") or image_name.endswith(".jpg"):
                res += (_load_file(image_name, self.config.input_dir),)
        if len(res) == 1:
            res = res[0]
        return res