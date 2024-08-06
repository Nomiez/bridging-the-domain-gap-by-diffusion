from __future__ import annotations

from typing import Tuple

from PIL import Image
from pathlib import Path
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json

from sd_pipeline_typing.types import Module

from .config import C2CConfig
from .classes import get_coco_classes


class C2C(Module):
    def __init__(self, *, config: C2CConfig):
        self.config = config

    def run(
        self,
        input_data: dict[str, str | Image.Image] | tuple[dict[str, str | Image.Image]],
        pipeline_config,
    ) -> dict[str, str | Image.Image]:
        """
        Not compatible with for_each -> directly supports multible image inputs
        """

        coco = Coco()
        classes = get_coco_classes()
        for cls in classes:
            coco.add_category(
                CocoCategory(id=cls["id"], name=cls["name"], supercategory=cls["supercategory"])
            )

        if isinstance(input_data, dict):
            input_data = (input_data,)

        for image_dict in input_data:
            img_name = Path(image_dict["name"])
            bboxes = image_dict["bbox"]

            coco_image = CocoImage(file_name=str(img_name), height=1080, width=1920)

            # Convert bbox format into COCO format
            bbox_objects = bboxes["bbox"]["bboxes"]

            # Default bboxes are build in the following way:
            #
            #  +------------------
            #  |    /----------| |
            #  |  /            | |
            #  | |__/O\____/O\_| |
            #  ------------------+
            #
            # The first coordinate marks the left top corner, the other marks the right bottom corner

            for bbox_array in bbox_objects:
                top_x, top_y = bbox_array[0][0], bbox_array[0][1]
                bottom_x, bottom_y = bbox_array[1][0], bbox_array[1][1]
                height = bottom_x - top_x
                width = bottom_y - top_y

                coco_image.add_annotation(
                    CocoAnnotation(
                        bbox=[top_x, top_y, width, height], category_id=3, category_name="vehicle"
                    )
                )

            coco.add_image(coco_image)

        save_json(data=coco.json, save_path=self.config.output_dir_json)
        return input_data

    @staticmethod
    def format_stream(*, name_of_bbox_contains: str = "_bbox"):
        def prepare(input_data: Tuple, _):
            image, bbox = input_data

            if type(bbox) is tuple:
                if bbox[0]["name"].find(name_of_bbox_contains) == -1:
                    bbox, image = image, bbox

                if bbox[0]["name"].find(name_of_bbox_contains) == -1:
                    raise ValueError("Bounding Boxes not found in the input data")

                image = sorted(image, key=lambda x: x["name"])
                bbox = sorted(bbox, key=lambda x: x["name"])

                res = ()
                for img, bbox in zip(image, bbox):
                    res += ({"image": img["image"], "name": img["name"], "bbox": bbox},)
                return res
            else:
                if bbox["name"].find(name_of_bbox_contains) == -1:
                    bbox, image = image, bbox

                if bbox["name"].find(name_of_bbox_contains) == -1:
                    raise ValueError("Bounding Boxes not found in the input data")

                return {"image": image["image"], "name": image["name"], "bbox": bbox}

        return prepare
