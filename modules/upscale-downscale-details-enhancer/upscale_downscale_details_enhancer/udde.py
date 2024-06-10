from typing import Tuple

import torch
import gc
from PIL import Image
from pathlib import Path
from diffusers import StableDiffusionUpscalePipeline

from sd_pipeline_typing.types import Module

from .config import UDDEConfig


class UDDE(Module):
    def __init__(self, *, config: UDDEConfig):
        self.config = config

    def run(
        self, input_data: dict[str, str | Image.Image], pipeline_config
    ) -> dict[str, str | Image.Image]:
        output = {}

        image = input_data["image"]
        img_name = (
            Path(input_data["name"]).stem
            + "_udde_details_enhanced"
            + Path(input_data["name"]).suffix
        )
        width, height = image.size

        if self.config.pre_H is not None and self.config.pre_W is not None:
            image = image.resize((self.config.pre_W, self.config.pre_H))

        torch.cuda.empty_cache()
        gc.collect()

        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipeline = pipeline.to("cuda")

        pipeline.enable_sequential_cpu_offload()
        pipeline.enable_attention_slicing()

        # prepare image
        prompt = self.config.prompt

        # pass prompt and image to pipeline
        image_in_pipeline = pipeline(
            prompt,
            image=image,
        ).images[0]

        output["name"] = img_name

        if self.config.after_H is not None and self.config.after_W is not None:
            output["image"] = image_in_pipeline.resize((self.config.after_W, self.config.after_H))
        else:
            output["image"] = image_in_pipeline.resize((width, height))

        return output
