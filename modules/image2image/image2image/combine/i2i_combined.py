from __future__ import annotations

from typing import Tuple, Callable, Union
import numpy as np

import torch

from PIL import Image
from pathlib import Path
from diffusers import AutoPipelineForImage2Image, ControlNetModel

from sd_pipeline_typing.types import Module

from .config import I2ICombinedConfig


class I2ICombined(Module):
    def __init__(self, *, config: I2ICombinedConfig):
        self.config = config

    def run(
        self, input_data: dict[str, str | Image.Image], pipeline_config
    ) -> dict[str, str | Image.Image]:
        output = {}

        image = input_data["image"]
        img_name = (
            Path(input_data["name"]).stem + "_image2image_depth" + Path(input_data["name"]).suffix
        )  # type: ignore

        control_img_seg = input_data["segmentation"]
        control_img_depth = input_data["depth"]

        controlnet_seg = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-seg",
            torch_dtype=torch.float16,
        ).to("cuda")

        controlnet_depth = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch.float16,
        ).to("cuda")

        pipeline = AutoPipelineForImage2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=[controlnet_seg, controlnet_depth],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")

        # prepare image
        prompt = self.config.prompt

        # pass prompt and image to pipeline
        image_in_pipeline = pipeline(
            prompt,
            image=image,
            control_image=[control_img_seg, control_img_depth],
            strength=self.config.strength,
        ).images[0]

        output["name"] = img_name
        output["image"] = image_in_pipeline

        return output

    @staticmethod
    def format_stream(
        *,
        name_of_seg_contains: str = "_seg",
        name_of_depth_contains: str = "_depth",
        denormalize_function_depth: Union[Callable, None] = None,
    ):
        def prepare(input_data: Tuple, _):
            image, segmentation, depth = input_data

            if type(depth) is tuple:
                if image[0]["name"].find(name_of_seg_contains) == -1:
                    image, segmentation, depth = segmentation, image, depth
                elif segmentation[0]["name"].find(name_of_seg_contains) == -1:
                    image, segmentation, depth = image, segmentation, depth
                elif depth[0]["name"].find(name_of_seg_contains) == -1:
                    image, segmentation, depth = image, depth, segmentation
                else:
                    raise ValueError("Segmentation not found in the input data")

                if image[0]["name"].find(name_of_depth_contains) == -1:
                    image, segmentation, depth = depth, segmentation, image
                elif segmentation[0]["name"].find(name_of_depth_contains) == -1:
                    image, segmentation, depth = segmentation, depth, image
                elif depth[0]["name"].find(name_of_depth_contains) == -1:
                    image, segmentation, depth = depth, image, segmentation
                else:
                    raise ValueError("Depth not found in the input data")

                res = ()
                if denormalize_function_depth is not None:
                    for img, seg, dep in zip(image, segmentation, depth):
                        array = denormalize_function_depth(np.array(dep["image"]))
                        image = Image.fromarray(array)
                        res += (
                            {
                                "image": img["image"],
                                "name": img["name"],
                                "segmentation": seg["image"],
                                "depth": image,
                            },
                        )
                else:
                    for img, seg, dep in zip(image, segmentation, depth):
                        res += (
                            {
                                "image": img["image"],
                                "name": img["name"],
                                "segmentation": seg["image"],
                                "depth": dep["image"],
                            },
                        )
                return res

            else:
                if image["name"].find(name_of_seg_contains) == -1:
                    image, segmentation, depth = segmentation, image, depth
                elif segmentation["name"].find(name_of_seg_contains) == -1:
                    image, segmentation, depth = image, segmentation, depth
                elif depth["name"].find(name_of_seg_contains) == -1:
                    image, segmentation, depth = image, depth, segmentation
                else:
                    raise ValueError("Segmentation not found in the input data")

                if image["name"].find(name_of_depth_contains) == -1:
                    image, segmentation, depth = depth, segmentation, image
                elif segmentation["name"].find(name_of_depth_contains) == -1:
                    image, segmentation, depth = segmentation, depth, image
                elif depth["name"].find(name_of_depth_contains) == -1:
                    image, segmentation, depth = depth, image, segmentation
                else:
                    raise ValueError("Depth not found in the input data")

                if denormalize_function_depth is not None:
                    array = denormalize_function_depth(np.array(depth["image"]))
                    image = Image.fromarray(array)
                    return {
                        "image": image["image"],
                        "name": image["name"],
                        "segmentation": segmentation["image"],
                        "depth": image,
                    }
                else:
                    return {
                        "image": image["image"],
                        "name": image["name"],
                        "segmentation": segmentation["image"],
                        "depth": depth["image"],
                    }

        return prepare
