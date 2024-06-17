from __future__ import annotations

import os
from typing import Dict, Tuple

from PIL.Image import Image, open

from sd_pipeline_typing.types import Module

from .config import LFFSConfig


def _load_file(image_name: str, input_dir: str) -> dict[str, str | Image]:
    # Open all images in the input directory
    output = {"name": "", "image": Image()}
    image_path = os.path.join(input_dir, image_name)
    image = open(image_path)
    output["name"] = image_name
    output["image"] = image
    return output  # type: ignore


class LFFS(Module):
    def __init__(self, *, config: LFFSConfig):
        self.config = config

    def run(self, input_data: str, _) -> Dict[str, str | Image] | Tuple[Dict[str, str | Image]]:
        res: tuple[()] | Tuple[Dict[str, str | Image]] | Dict[str, str | Image] = ()
        for image_name in os.listdir(self.config.input_dir):
            if image_name.endswith(".png") or image_name.endswith(".jpg"):
                res += (_load_file(image_name, self.config.input_dir),)  # type: ignore
        if len(res) == 1:
            res = res[0]  # type: ignore
        if len(res) == 0:
            raise ValueError("No images found in the input directory")

        if self.config.clear_dir:
            for image_name in os.listdir(self.config.input_dir):
                if image_name.endswith(".png") or image_name.endswith(".jpg"):
                    os.remove(os.path.join(self.config.input_dir, image_name))

        return res
