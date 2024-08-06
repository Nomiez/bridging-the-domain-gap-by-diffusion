from __future__ import annotations


from PIL import Image, ImageOps

from sd_pipeline_typing.types import Module

from .config import InverterConfig


# Source: https://stackoverflow.com/questions/2498875/how-to-invert-colors-of-image-with-pil-python-imaging
class Inverter(Module):
    def __init__(self, *, config: InverterConfig):
        self.config = config

    def run(
        self,
        input_data: dict[str, str | Image.Image],
        pipeline_config,
    ) -> dict[str, str | Image.Image]:
        image = input_data["image"]

        if image.mode == "RGBA":
            r, g, b, a = image.split()
            rgb_image = Image.merge("RGB", (r, g, b))
            inverted_image = ImageOps.invert(rgb_image)
            r2, g2, b2 = inverted_image.split()
            final_transparent_image = Image.merge("RGBA", (r2, g2, b2, a))
            return {"image": final_transparent_image, "name": input_data["name"]}
        else:
            inverted_image = ImageOps.invert(image)
            return {"image": inverted_image, "name": input_data["name"]}
