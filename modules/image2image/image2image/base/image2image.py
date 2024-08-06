from __future__ import annotations

import torch

from PIL import Image
from pathlib import Path
from diffusers import AutoPipelineForImage2Image

from sd_pipeline_typing.types import Module

from .config import I2IBaseConfig


class I2I(Module):
    def __init__(self, *, config: I2IBaseConfig):
        self.config = config

    def run(
        self, input_data: dict[str, str | Image.Image], pipeline_config
    ) -> dict[str, str | Image.Image]:
        output = {}

        image = input_data["image"]
        img_name = Path(input_data["name"]).stem + "_image2image" + Path(input_data["name"]).suffix  # type: ignore

        pipeline = AutoPipelineForImage2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")

        # prepare image
        prompt = self.config.prompt
        negative_prompt = (self.config.negative_prompt,)

        # pass prompt and image to pipeline
        image_in_pipeline = pipeline(
            prompt, negative_prompt=negative_prompt, image=image, strength=self.config.strength
        ).images[0]

        output["name"] = img_name
        output["image"] = image_in_pipeline

        return output
