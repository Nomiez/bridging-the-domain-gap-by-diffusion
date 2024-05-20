import os
from typing import Dict, Tuple
from pathlib import Path

from PIL.Image import Image

from sd_pipeline_typing.types import Module
from .config import SIFSConfig


class SIFS(Module):
    def __init__(self, *, config: SIFSConfig):
        self.config = config

    def run(
        self, input_data: Dict[str, str | Image] | Tuple[Dict[str, str | Image]], _
    ) -> Dict[str, str | Image] | Tuple[Dict[str, str | Image]]:
        # Stores images in the file system

        def save_image(image: Image, name: str) -> str:
            new_name = name
            file = Path(os.path.join(self.config.output_dir, new_name))

            # Check if the imagename is already in the output directory
            while file.is_file():
                new_name = Path(new_name).stem + "_copy" + Path(new_name).suffix
                file = Path(os.path.join(self.config.output_dir, new_name))

            # Check if the output directory exists
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir)

            path = os.path.join(self.config.output_dir, new_name)
            image.save(path)
            return new_name

        if isinstance(input_data, dict):
            name = input_data["name"]
            image = input_data["image"]
            name = save_image(image, name)
            res = {"name": name, "image": image}
        else:
            res = ()
            for data in input_data:
                name = data["name"]
                image = data["image"]
                name = save_image(image, name)
                res += ({"name": name, "image": image},)
        return res
