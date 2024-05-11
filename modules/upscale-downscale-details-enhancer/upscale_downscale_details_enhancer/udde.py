from typing import Tuple

import torch

from PIL import Image
from pathlib import Path
from diffusers import StableDiffusionUpscalePipeline

from sd_pipeline_typing.types import Module

from .config import I2IConfig


class UDDE(Module):
    def __init__(self, *, config: I2IConfig):
        self.config = config

    def run(self, input_data: dict[str, str | Image.Image], pipeline_config) -> dict[str, str | Image.Image]:

        output = {}

        image = input_data["image"]
        img_name = Path(input_data["name"]).stem + f"_udde_details_enhanced" + Path(input_data["name"]).suffix
        width, height = image.size

        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipeline = pipeline.to("cuda")

        # prepare image
        prompt = self.config.prompt

        # pass prompt and image to pipeline
        image_in_pipeline = pipeline(
            prompt,
            image=image,
        ).images[0]

        output["name"] = img_name
        output["image"] = image_in_pipeline.resize((width, height))

        return output
