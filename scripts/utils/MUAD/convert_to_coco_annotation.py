from dataclasses import dataclass
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image
from pathlib import Path
import json


@dataclass
class Bbox:
    top_x: float
    top_y: float
    width: float
    height: float


def add_bbox_to_annotaion(img_bbox, coco_image):
    coco_image.add_annotation(
        CocoAnnotation(
            bbox=[img_bbox.top_x, img_bbox.top_y, img_bbox.width, img_bbox.height],
            category_id=3,
            category_name="car",
        )
    )


def get_bbox(annotation):
    bbox = annotation["2d-bounding-rectangle"]
    return Bbox(bbox["x"], bbox["y"], bbox["width"], bbox["height"])


def main():
    image_path = Path("./images")
    annotation_path = Path("./gtFine")
    all_images = [
        file for file in image_path.rglob("*") if file.is_file() and "leftLabel" in file.name
    ]

    splits = [
        all_images[i : i + int(0.8 * len(all_images))]
        for i in range(0, len(all_images), int(0.8 * len(all_images)))
    ]
    names = ["train.json", "val.json"]

    for i, split in enumerate(splits):
        coco = Coco()
        coco.add_category(CocoCategory(id=3, name="car"))
        for image in split:
            with (annotation_path / Path(image.name.split("_")[0] + "_leftAnnotation.json")).open(
                "r"
            ) as file:
                ann = json.loads(file.read())
            coco_image = CocoImage(file_name=image.name, height=1024, width=2048)

            # Map all vehicles to cars
            cars = ann["classes"]["Car"]
            for car in cars:
                bbox = get_bbox(car)
                add_bbox_to_annotaion(bbox, coco_image)

            try:
                vans = ann["classes"]["Van"]
                for car in vans:
                    bbox = get_bbox(car)
                    add_bbox_to_annotaion(bbox, coco_image)
            except Exception:
                pass

            try:
                buses = ann["classes"]["Bus"]
                for car in buses:
                    bbox = get_bbox(car)
                    add_bbox_to_annotaion(bbox, coco_image)
            except Exception:
                pass

            coco.add_image(coco_image)

        save_json(data=coco.json, save_path=Path(names[i]))


if __name__ == "__main__":
    main()
