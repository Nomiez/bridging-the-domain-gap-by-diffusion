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
            if not isinstance(name, str):
                raise ValueError("Name must be a string")
            image = input_data["image"]
            if not isinstance(image, Image):
                raise ValueError("Image must be a PIL.Image.Image")
            name = save_image(image, name)
            res = {"name": name, "image": image}
            return res  # type: ignore
        else:
            result: tuple[()] | Tuple[Dict[str, str | Image]] = ()

            for data in input_data:
                name = data["name"]
                if not isinstance(name, str):
                    raise ValueError("Name must be a string")
                image = data["image"]
                if not isinstance(image, Image):
                    raise ValueError("Image must be a PIL.Image.Image")
                name = save_image(image, name)
                result += ({"name": name, "image": image},)  # type: ignore

            if isinstance(result, tuple) and len(result) == 0:
                raise ValueError("No images were found in the input data")
            else:
                assert isinstance(result, tuple)
                return result
