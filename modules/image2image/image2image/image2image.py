from typing import Tuple

import torch

from PIL import Image
from pathlib import Path
from diffusers import AutoPipelineForImage2Image, ControlNetModel
from diffusers.utils import make_image_grid

from sd_pipeline_typing.types import Module

from .config import I2IConfig


class I2I(Module):
    def __init__(self, *, config: I2IConfig):
        self.config = config

    def run(self, input_data: dict[str, str | Image.Image], pipeline_config) -> dict[str, str | Image.Image]:

        output = {}

        image = input_data["image"]
        img_name = Path(input_data["name"]).stem + f"_image2image" + Path(input_data["name"]).suffix
        control_img = input_data["segmentation"]

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth",
            torch_dtype=torch.float16,
            variant="fp16", use_safetensors=True
        )

        pipeline = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        pipeline.to("cuda")
        # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
        # pipeline.enable_xformers_memory_efficient_attention()

        refiner = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipeline.text_encoder_2,
            vae=pipeline.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner.to("cuda")

        # prepare image
        prompt = "ultra realism, sharp, detailed, 8k"

        # pass prompt and image to pipeline
        image_in_pipeline = pipeline(prompt, image=image, control_image=control_img, strength=0.1).images

        # Define how many steps and what % of steps to be run on each experts (80/20) here
        n_steps = 40
        high_noise_frac = 0.8

        image_in_pipeline = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image_in_pipeline,
        ).images[0]

        output["name"] = img_name
        output["image"] = image_in_pipeline

        return output

    @staticmethod
    def format_stream(*, name_of_seg_contains: str = "_segmentation"):
        def prepare(input_data: Tuple, _):
            image, segmentation = input_data

            if segmentation[0]["name"].find(name_of_seg_contains) == -1:
                segmentation, image = image, segmentation

            if segmentation[0]["name"].find(name_of_seg_contains) == -1:
                raise ValueError("Segmentation not found in the input data")

            if len(image) == 1:
                return {"image": image[0]["image"], "name": image[0]["name"], "segmentation": segmentation[0]["image"]}
            else:
                res = ()
                for (img, seg) in zip(image, segmentation):
                    res += ({"image": img["image"], "name": img["name"], "segmentation": seg["image"]},)
                return res

        return prepare
