from __future__ import annotations

from typing import Tuple, Callable, Union
import numpy as np

import torch

from PIL import Image
from pathlib import Path
from diffusers import AutoPipelineForImage2Image, ControlNetModel

from sd_pipeline_typing.types import Module

from .config import I2IDepthConfig


class I2IDepth(Module):
    def __init__(self, *, config: I2IDepthConfig):
        self.config = config

    def run(
        self, input_data: dict[str, str | Image.Image], pipeline_config
    ) -> dict[str, str | Image.Image]:
        output = {}

        image = input_data["image"]
        img_name = (
            Path(input_data["name"]).stem + "_image2image_depth" + Path(input_data["name"]).suffix
        )  # type: ignore
        control_img = input_data["depth"]

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch.float16,
        )

        pipeline = AutoPipelineForImage2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        pipeline.to("cuda")

        # prepare image
        prompt = self.config.prompt
        negative_prompt = self.config.negative_prompt

        # pass prompt and image to pipeline
        image_in_pipeline = pipeline(
            prompt,
            image=image,
            negative_prompt=negative_prompt,
            control_image=control_img,
            strength=self.config.strength,
        ).images[0]

        output["name"] = img_name
        output["image"] = image_in_pipeline

        return output

    @staticmethod
    def format_stream(
        *,
        name_of_depth_contains: str = "_depth",
    ):
        def prepare(input_data: Tuple, _):
            image, depth = input_data

            if type(depth) is tuple:
                if depth[0]["name"].find(name_of_depth_contains) == -1:
                    depth, image = image, depth

                if depth[0]["name"].find(name_of_depth_contains) == -1:
                    raise ValueError("Segmentation not found in the input data")

                res = ()
                for img, dep in zip(image, depth):
                    res += ({"image": img["image"], "name": img["name"], "depth": dep["image"]},)
                return res
            else:
                if depth["name"].find(name_of_depth_contains) == -1:
                    depth, image = image, depth

                if depth["name"].find(name_of_depth_contains) == -1:
                    raise ValueError("Segmentation not found in the input data")

                return {
                    "image": image["image"],
                    "name": image["name"],
                    "depth": depth["image"],
                }

        return prepare
