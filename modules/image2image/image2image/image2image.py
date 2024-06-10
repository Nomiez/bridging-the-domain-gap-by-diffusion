from __future__ import annotations

from typing import Tuple

import torch

from PIL import Image
from pathlib import Path
from diffusers import AutoPipelineForImage2Image, ControlNetModel

from sd_pipeline_typing.types import Module

from .config import I2IConfig


class I2I(Module):
    def __init__(self, *, config: I2IConfig):
        self.config = config

    def run(
        self, input_data: dict[str, str | Image.Image], pipeline_config
    ) -> dict[str, str | Image.Image]:
        output = {}

        image = input_data["image"]
        img_name = Path(input_data["name"]).stem + "_image2image" + Path(input_data["name"]).suffix  # type: ignore
        control_img = input_data["segmentation"]

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-seg",
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

        # pass prompt and image to pipeline
        image_in_pipeline = pipeline(
            prompt, image=image, control_image=control_img, strength=self.config.strength
        ).images[0]

        output["name"] = img_name
        output["image"] = image_in_pipeline

        return output

    @staticmethod
    def format_stream(*, name_of_seg_contains: str = "_segmentation"):
        def prepare(input_data: Tuple, _):
            image, segmentation = input_data

            if type(segmentation) is tuple:
                if segmentation[0]["name"].find(name_of_seg_contains) == -1:
                    segmentation, image = image, segmentation

                if segmentation[0]["name"].find(name_of_seg_contains) == -1:
                    raise ValueError("Segmentation not found in the input data")

                res = ()
                for img, seg in zip(image, segmentation):
                    res += (
                        {"image": img["image"], "name": img["name"], "segmentation": seg["image"]},
                    )
                return res
            else:
                if segmentation["name"].find(name_of_seg_contains) == -1:
                    segmentation, image = image, segmentation

                if segmentation["name"].find(name_of_seg_contains) == -1:
                    raise ValueError("Segmentation not found in the input data")

                return {
                    "image": image["image"],
                    "name": image["name"],
                    "segmentation": segmentation["image"],
                }

        return prepare
